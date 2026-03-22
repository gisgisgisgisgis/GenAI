const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const Groq = require("groq-sdk");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

if (!process.env.GROQ_API_KEY) {
  console.error("Missing GROQ_API_KEY. Set it in your environment before running the server.");
  process.exit(1);
}

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public"));

const memoryStore = new Map();
const MAX_MESSAGES = 20;
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
const CONTINUATION_MAX_ROUNDS = 8;
const THINK_MODE_MIN_TOKENS = 1200;
const THINK_MODE_PROMPT = `
You are in THINK mode.
Think deeply and reason internally before answering.
Never reveal chain-of-thought, hidden reasoning, or internal notes.
Only output the final answer in English.
The response must be complete, concrete, and self-contained.
`.trim();
const CONTINUE_PROMPT = `
Continue exactly from where you stopped.
Do not restart, do not repeat previous text, and do not add prefatory phrases.
Return only the remaining content so the final answer is complete.
`.trim();

function sanitizeSessionId(value) {
  if (typeof value !== "string") return "";
  return value.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
}

function upsertMemory(sessionId, role, content) {
  const history = memoryStore.get(sessionId) || [];
  history.push({ role, content });

  if (history.length > MAX_MESSAGES) {
    history.splice(0, history.length - MAX_MESSAGES);
  }

  memoryStore.set(sessionId, history);
}

function sanitizeHistoryMessages(messages, limit = MAX_MESSAGES) {
  if (!Array.isArray(messages)) return [];

  const cleaned = messages
    .map((message) => {
      if (!message || typeof message !== "object") return null;
      if (!["user", "assistant", "system"].includes(message.role)) return null;
      if (typeof message.content !== "string") return null;

      const content = message.content.trim();
      if (!content) return null;

      return {
        role: message.role,
        content: content.slice(0, 2400),
      };
    })
    .filter(Boolean);

  if (cleaned.length > limit) {
    return cleaned.slice(cleaned.length - limit);
  }
  return cleaned;
}

function sanitizeSourceConversations(sources) {
  if (!Array.isArray(sources)) return [];

  return sources
    .map((source) => {
      if (!source || typeof source !== "object") return null;
      const title =
        typeof source.title === "string" && source.title.trim() ? source.title.trim() : "Untitled";
      const messages = sanitizeHistoryMessages(source.messages, 18);

      if (messages.length === 0) return null;
      return { title, messages };
    })
    .filter(Boolean);
}

function extractLoadedMemorySummary(messages) {
  if (!Array.isArray(messages)) return "";
  const marker = "Loaded memory summary:\n";

  for (const message of messages) {
    if (!message || message.role !== "system" || typeof message.content !== "string") {
      continue;
    }
    if (message.content.startsWith(marker)) {
      return message.content.slice(marker.length).trim();
    }
  }

  return "";
}

function buildSourceDigest(sources) {
  const blocks = sources.map((source, sourceIndex) => {
    const lines = source.messages.map((message, messageIndex) => {
      return `${messageIndex + 1}. [${message.role}] ${message.content.slice(0, 320)}`;
    });

    return `Conversation ${sourceIndex + 1}: ${source.title}\n${lines.join("\n")}`;
  });

  return blocks.join("\n\n").slice(0, 24000);
}

async function summarizeSourceMemory(groqClient, model, digestText) {
  const completion = await groqClient.chat.completions.create({
    model: typeof model === "string" && model.trim() ? model.trim() : "llama-3.3-70b-versatile",
    stream: false,
    temperature: 0.2,
    max_tokens: 700,
    messages: [
      {
        role: "system",
        content:
          "You are a memory compressor. Summarize multiple conversations into reusable context memory. Output summary only.",
      },
      {
        role: "user",
        content:
          `Compress the following conversations into 8-15 concise bullets. Preserve preferences, facts, decisions, and unresolved tasks:\n\n${digestText}`,
      },
    ],
  });

  return completion.choices?.[0]?.message?.content?.trim() || "(empty summary)";
}

async function createCompleteReply(groqClient, baseParams, messages) {
  const workingMessages = [...messages];
  let fullReply = "";
  let finalFinishReason = "stop";

  for (let round = 0; round < CONTINUATION_MAX_ROUNDS; round += 1) {
    const completion = await groqClient.chat.completions.create({
      ...baseParams,
      stream: false,
      messages: workingMessages,
    });

    const choice = completion.choices?.[0];
    const piece = choice?.message?.content || "";
    const finishReason = choice?.finish_reason || "stop";

    fullReply += piece;
    finalFinishReason = finishReason;

    if (finishReason !== "length") {
      break;
    }

    if (piece.trim()) {
      workingMessages.push({ role: "assistant", content: piece });
    }
    workingMessages.push({ role: "user", content: CONTINUE_PROMPT });
  }

  return {
    reply: fullReply || "(empty response)",
    finishReason: finalFinishReason,
  };
}

app.post("/api/chat", async (req, res) => {
  try {
    const {
      sessionId,
      message,
      model,
      systemPrompt,
      stream,
      temperature,
      maxTokens,
      topP,
      presencePenalty,
      frequencyPenalty,
      reasoningMode,
      loadedMemorySummary,
    } = req.body || {};

    const cleanSessionId = sanitizeSessionId(sessionId);

    if (!cleanSessionId) {
      return res.status(400).json({ error: "Invalid or missing sessionId." });
    }

    if (typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "Message is required." });
    }

    const chosenModel =
      typeof model === "string" && model.trim() ? model.trim() : "llama-3.1-8b-instant";
    const chosenSystemPrompt =
      typeof systemPrompt === "string" && systemPrompt.trim()
        ? systemPrompt.trim()
        : DEFAULT_SYSTEM_PROMPT;
    const effectiveSystemPrompt = reasoningMode === "think"
      ? `${chosenSystemPrompt}\n\n${THINK_MODE_PROMPT}`
      : chosenSystemPrompt;
    const isThinkMode = reasoningMode === "think";
    const selectedMaxTokens = Number.isFinite(Number(maxTokens)) ? Number(maxTokens) : 400;
    const effectiveMaxTokens = isThinkMode
      ? Math.max(selectedMaxTokens, THINK_MODE_MIN_TOKENS)
      : selectedMaxTokens;
    const shouldStream = stream === true && !isThinkMode;

    let prior = memoryStore.get(cleanSessionId) || [];
    const incomingSummary =
      typeof loadedMemorySummary === "string" ? loadedMemorySummary.trim().slice(0, 8000) : "";
    const storedSummary = extractLoadedMemorySummary(prior);
    const mergedSummary = incomingSummary || storedSummary;

    if (!storedSummary && incomingSummary) {
      const seededMemory = { role: "system", content: `Loaded memory summary:\n${incomingSummary}` };
      prior = [seededMemory, ...prior];
      if (prior.length > MAX_MESSAGES) {
        prior = prior.slice(prior.length - MAX_MESSAGES);
      }
      memoryStore.set(cleanSessionId, prior);
    }

    const finalSystemPrompt = mergedSummary
      ? `${effectiveSystemPrompt}\n\nUse this loaded memory summary as high-priority context:\n${mergedSummary}`
      : effectiveSystemPrompt;
    const priorWithoutSummary = prior.filter(
      (message) =>
        !(
          message?.role === "system" &&
          typeof message.content === "string" &&
          message.content.startsWith("Loaded memory summary:\n")
        ),
    );

    const messages = [
      { role: "system", content: finalSystemPrompt },
      ...priorWithoutSummary,
      { role: "user", content: message.trim() },
    ];

    const commonParams = {
      model: chosenModel,
      messages,
      temperature: Number.isFinite(Number(temperature)) ? Number(temperature) : 0.7,
      max_tokens: effectiveMaxTokens,
      top_p: Number.isFinite(Number(topP)) ? Number(topP) : 1,
      presence_penalty: Number.isFinite(Number(presencePenalty)) ? Number(presencePenalty) : 0,
      frequency_penalty: Number.isFinite(Number(frequencyPenalty)) ? Number(frequencyPenalty) : 0,
    };

    upsertMemory(cleanSessionId, "user", message.trim());

    if (shouldStream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const streamResp = await groq.chat.completions.create({
        ...commonParams,
        stream: true,
      });

      let accumulated = "";

      for await (const chunk of streamResp) {
        const delta = chunk.choices?.[0]?.delta?.content || "";
        if (delta) {
          accumulated += delta;
          res.write(`data: ${JSON.stringify({ delta })}\n\n`);
        }
      }

      upsertMemory(cleanSessionId, "assistant", accumulated || "(empty response)");
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    const { reply } = await createCompleteReply(groq, commonParams, messages);
    upsertMemory(cleanSessionId, "assistant", reply);

    return res.json({ reply });
  } catch (error) {
    const status = error?.status || 500;
    const message = error?.message || "Unexpected error while calling Groq.";
    return res.status(status).json({ error: message });
  }
});

app.post("/api/memory/reset", (req, res) => {
  const cleanSessionId = sanitizeSessionId(req.body?.sessionId);

  if (!cleanSessionId) {
    return res.status(400).json({ error: "Invalid or missing sessionId." });
  }

  memoryStore.delete(cleanSessionId);
  return res.json({ ok: true });
});

app.post("/api/memory/replace", (req, res) => {
  const cleanSessionId = sanitizeSessionId(req.body?.sessionId);
  if (!cleanSessionId) {
    return res.status(400).json({ error: "Invalid or missing sessionId." });
  }

  const cleanedMessages = sanitizeHistoryMessages(req.body?.messages, MAX_MESSAGES);
  if (cleanedMessages.length === 0) {
    memoryStore.delete(cleanSessionId);
    return res.json({ ok: true, count: 0 });
  }

  memoryStore.set(cleanSessionId, cleanedMessages);
  return res.json({ ok: true, count: cleanedMessages.length });
});

app.post("/api/memory/load", async (req, res) => {
  try {
    const cleanTargetSessionId = sanitizeSessionId(req.body?.targetSessionId);
    if (!cleanTargetSessionId) {
      return res.status(400).json({ error: "Invalid or missing targetSessionId." });
    }

    const sources = sanitizeSourceConversations(req.body?.sources);
    if (sources.length === 0) {
      return res.status(400).json({ error: "At least one valid source conversation is required." });
    }

    const digestText = buildSourceDigest(sources);
    const summary = await summarizeSourceMemory(groq, req.body?.model, digestText);
    const targetMessages = sanitizeHistoryMessages(req.body?.targetMessages, MAX_MESSAGES - 1);

    const loadedMemory = [{ role: "system", content: `Loaded memory summary:\n${summary}` }, ...targetMessages];
    memoryStore.set(cleanTargetSessionId, loadedMemory);

    return res.json({ ok: true, summary });
  } catch (error) {
    const status = error?.status || 500;
    const message = error?.message || "Failed to load conversation memory.";
    return res.status(status).json({ error: message });
  }
});

app.get("/api/health", (_req, res) => {
  res.json({ ok: true });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
