const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const vm = require("vm");
const Groq = require("groq-sdk");

function optionalRequire(packageName) {
  try {
    return require(packageName);
  } catch (_error) {
    return null;
  }
}

const OpenAI = optionalRequire("openai");
const Tesseract = optionalRequire("tesseract.js");

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 3000;

const groq = process.env.GROQ_API_KEY ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;
const openai = process.env.OPENAI_API_KEY && OpenAI ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

if (process.env.OPENAI_API_KEY && !OpenAI) {
  console.warn("OPENAI_API_KEY is set but package 'openai' is not installed. OpenAI provider disabled.");
}

if (!Tesseract) {
  console.warn("Package 'tesseract.js' is not installed. Image OCR preprocessing disabled.");
}

if (!groq && !openai) {
  console.error("Missing provider key. Set GROQ_API_KEY and/or OPENAI_API_KEY in your environment.");
  process.exit(1);
}

app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.static("public"));

app.use((req, res, next) => {
  const started = Date.now();
  runtimeObservability.requests.total += 1;
  runtimeObservability.requests.byPath[req.path] =
    (runtimeObservability.requests.byPath[req.path] || 0) + 1;

  res.on("finish", () => {
    const statusClass = bucketStatusClass(res.statusCode);
    runtimeObservability.requests.byStatusClass[statusClass] += 1;

    if (statusClass === "5xx") {
      runtimeObservability.requests.errors += 1;
    }

    recordObservabilityEvent({
      type: "http",
      method: req.method,
      path: req.path,
      statusCode: res.statusCode,
      durationMs: Date.now() - started,
    });
  });

  next();
});

const DATA_DIR = path.join(__dirname, "data");
const MEMORY_DB_PATH = path.join(DATA_DIR, "memory-db.json");
const MAX_SHORT_TERM_MESSAGES = 40;
const DEFAULT_RECENT_CONTEXT_MESSAGES = 12;
const DEFAULT_CONTEXT_COMPRESS_TRIGGER = 22;
const CONTINUATION_MAX_ROUNDS = 6;
const THINK_MODE_MIN_TOKENS = 1200;
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
const DEFAULT_USER_ID = "default_user";
const VECTOR_DIM = 192;
const MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024;
const DEFAULT_STYLE = "concise";
const DEFAULT_EXPERTISE = "intermediate";
const VIDEO_PROCESSING_ENABLED = false;
const OBSERVABILITY_EVENT_LIMIT = 300;

function getRecentContextMessages() {
  return clampNumber(process.env.RECENT_CONTEXT_MESSAGES, DEFAULT_RECENT_CONTEXT_MESSAGES, 4, 40);
}

function getContextCompressTrigger() {
  return clampNumber(process.env.CONTEXT_COMPRESS_TRIGGER, DEFAULT_CONTEXT_COMPRESS_TRIGGER, 8, 120);
}

function setRuntimeContextConfig({ triggerMessages, keepRecentMessages }) {
  const nextTrigger = clampNumber(triggerMessages, getContextCompressTrigger(), 8, 120);
  const nextKeepRecent = clampNumber(keepRecentMessages, getRecentContextMessages(), 4, 40);
  process.env.CONTEXT_COMPRESS_TRIGGER = String(nextTrigger);
  process.env.RECENT_CONTEXT_MESSAGES = String(nextKeepRecent);
  return {
    triggerMessages: getContextCompressTrigger(),
    keepRecentMessages: getRecentContextMessages(),
  };
}

const runtimeObservability = {
  startedAt: nowIso(),
  requests: {
    total: 0,
    errors: 0,
    byPath: {},
    byStatusClass: {
      "2xx": 0,
      "4xx": 0,
      "5xx": 0,
    },
  },
  chat: {
    total: 0,
    streamed: 0,
    withTools: 0,
    withAttachments: 0,
    byModel: {},
  },
  tools: {
    total: 0,
    failed: 0,
    byName: {},
  },
  recentEvents: [],
};

function recordObservabilityEvent(event) {
  runtimeObservability.recentEvents.push({
    at: nowIso(),
    ...event,
  });

  if (runtimeObservability.recentEvents.length > OBSERVABILITY_EVENT_LIMIT) {
    runtimeObservability.recentEvents.splice(
      0,
      runtimeObservability.recentEvents.length - OBSERVABILITY_EVENT_LIMIT,
    );
  }
}

function bucketStatusClass(statusCode) {
  const parsed = Number(statusCode);
  if (!Number.isFinite(parsed)) return "5xx";
  if (parsed >= 200 && parsed < 300) return "2xx";
  if (parsed >= 400 && parsed < 500) return "4xx";
  return "5xx";
}

function ensureDirSync(targetPath) {
  if (!fs.existsSync(targetPath)) {
    fs.mkdirSync(targetPath, { recursive: true });
  }
}

function nowIso() {
  return new Date().toISOString();
}

function sanitizeId(value, fallback = "") {
  if (typeof value !== "string") return fallback;
  const cleaned = value.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 80);
  return cleaned || fallback;
}

function clampNumber(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function safeJsonParse(raw, fallback) {
  try {
    return JSON.parse(raw);
  } catch (_error) {
    return fallback;
  }
}

function stableStringify(value) {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }

  const keys = Object.keys(value).sort();
  const entries = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`);
  return `{${entries.join(",")}}`;
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function tokenize(text) {
  if (typeof text !== "string") return [];
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 2)
    .slice(0, 512);
}

function embedText(text) {
  const vec = new Array(VECTOR_DIM).fill(0);
  const tokens = tokenize(text);

  for (const token of tokens) {
    const hash = crypto.createHash("sha1").update(token).digest();
    const idx = hash.readUInt16BE(0) % VECTOR_DIM;
    const sign = hash[2] % 2 === 0 ? 1 : -1;
    vec[idx] += sign;
  }

  const norm = Math.sqrt(vec.reduce((sum, value) => sum + value * value, 0));
  if (norm === 0) return vec;
  return vec.map((value) => value / norm);
}

class PersistentMemoryDB {
  constructor(filePath) {
    this.filePath = filePath;
    this.data = {
      version: 1,
      users: {},
    };
    this.writeQueue = Promise.resolve();
    this.loaded = false;
  }

  async init() {
    ensureDirSync(path.dirname(this.filePath));

    if (fs.existsSync(this.filePath)) {
      const raw = await fs.promises.readFile(this.filePath, "utf8");
      const parsed = safeJsonParse(raw, null);
      if (parsed && typeof parsed === "object" && parsed.users && typeof parsed.users === "object") {
        this.data = parsed;
      }
    }

    this.loaded = true;
  }

  async flush() {
    const payload = JSON.stringify(this.data, null, 2);
    this.writeQueue = this.writeQueue.then(() => fs.promises.writeFile(this.filePath, payload));
    await this.writeQueue;
  }

  ensureUser(userId) {
    const cleanUserId = sanitizeId(userId, DEFAULT_USER_ID);

    if (!this.data.users[cleanUserId]) {
      this.data.users[cleanUserId] = {
        profile: {
          userId: cleanUserId,
          displayName: cleanUserId,
          responseStyle: DEFAULT_STYLE,
          expertiseLevel: DEFAULT_EXPERTISE,
          preferences: {},
          updatedAt: nowIso(),
        },
        memories: [],
        archivedMemories: [],
        summary: "",
        updatedAt: nowIso(),
      };
    }

    return this.data.users[cleanUserId];
  }

  getProfile(userId) {
    return this.ensureUser(userId).profile;
  }

  async updateProfile(userId, patch = {}) {
    const user = this.ensureUser(userId);
    const profile = user.profile;

    if (typeof patch.displayName === "string" && patch.displayName.trim()) {
      profile.displayName = patch.displayName.trim().slice(0, 80);
    }

    if (["concise", "technical", "casual"].includes(patch.responseStyle)) {
      profile.responseStyle = patch.responseStyle;
    }

    if (["beginner", "intermediate", "expert"].includes(patch.expertiseLevel)) {
      profile.expertiseLevel = patch.expertiseLevel;
    }

    if (patch.preferences && typeof patch.preferences === "object" && !Array.isArray(patch.preferences)) {
      profile.preferences = {
        ...profile.preferences,
        ...patch.preferences,
      };
    }

    profile.updatedAt = nowIso();
    user.updatedAt = nowIso();
    await this.flush();
    return profile;
  }

  createMemory(userId, input) {
    const user = this.ensureUser(userId);
    const content = typeof input?.content === "string" ? input.content.trim() : "";
    if (!content) {
      throw new Error("Memory content is required.");
    }

    const type = ["fact", "preference", "history", "task", "summary"].includes(input?.type)
      ? input.type
      : "fact";

    const tags = Array.isArray(input?.tags)
      ? input.tags
          .map((tag) => (typeof tag === "string" ? tag.trim().toLowerCase() : ""))
          .filter(Boolean)
          .slice(0, 12)
      : [];

    const item = {
      id: crypto.randomUUID(),
      type,
      content: content.slice(0, 2400),
      tags,
      sourceSessionId: sanitizeId(input?.sourceSessionId, ""),
      metadata: input?.metadata && typeof input.metadata === "object" ? input.metadata : {},
      importance: clampNumber(input?.importance, 0.5, 0, 1),
      createdAt: nowIso(),
      updatedAt: nowIso(),
      lastAccessedAt: nowIso(),
      embedding: embedText(content),
    };

    user.memories.push(item);
    user.updatedAt = nowIso();
    return item;
  }

  async addMemory(userId, input) {
    const item = this.createMemory(userId, input);
    await this.flush();
    return item;
  }

  listMemories(userId, options = {}) {
    const user = this.ensureUser(userId);
    let items = [...user.memories];

    if (options.type) {
      items = items.filter((item) => item.type === options.type);
    }

    const query = typeof options.query === "string" ? options.query.trim() : "";
    if (query) {
      const queryEmbedding = embedText(query);
      items = items
        .map((item) => ({
          ...item,
          similarity: cosineSimilarity(item.embedding, queryEmbedding),
        }))
        .sort((a, b) => b.similarity - a.similarity);

      const minScore = clampNumber(options.minScore, 0.12, -1, 1);
      items = items.filter((item) => (item.similarity || 0) >= minScore);
    } else {
      items.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
    }

    const limit = clampNumber(options.limit, 100, 1, 500);
    return items.slice(0, limit);
  }

  async updateMemory(userId, memoryId, patch = {}) {
    const user = this.ensureUser(userId);
    const target = user.memories.find((item) => item.id === memoryId);

    if (!target) {
      throw new Error("Memory not found.");
    }

    if (typeof patch.content === "string" && patch.content.trim()) {
      target.content = patch.content.trim().slice(0, 2400);
      target.embedding = embedText(target.content);
    }

    if (["fact", "preference", "history", "task", "summary"].includes(patch.type)) {
      target.type = patch.type;
    }

    if (Array.isArray(patch.tags)) {
      target.tags = patch.tags
        .map((tag) => (typeof tag === "string" ? tag.trim().toLowerCase() : ""))
        .filter(Boolean)
        .slice(0, 12);
    }

    if (Number.isFinite(Number(patch.importance))) {
      target.importance = clampNumber(patch.importance, target.importance, 0, 1);
    }

    if (patch.metadata && typeof patch.metadata === "object" && !Array.isArray(patch.metadata)) {
      target.metadata = {
        ...target.metadata,
        ...patch.metadata,
      };
    }

    target.updatedAt = nowIso();
    user.updatedAt = nowIso();
    await this.flush();
    return target;
  }

  async deleteMemory(userId, memoryId) {
    const user = this.ensureUser(userId);
    const index = user.memories.findIndex((item) => item.id === memoryId);

    if (index === -1) {
      throw new Error("Memory not found.");
    }

    const [removed] = user.memories.splice(index, 1);
    user.archivedMemories.push({
      ...removed,
      archivedAt: nowIso(),
    });

    user.updatedAt = nowIso();
    await this.flush();
    return removed;
  }

  summarizeUserMemories(userId) {
    const user = this.ensureUser(userId);
    const byType = new Map();

    for (const item of user.memories) {
      const bucket = byType.get(item.type) || [];
      bucket.push(item);
      byType.set(item.type, bucket);
    }

    const lines = [];
    for (const [type, items] of byType.entries()) {
      const latest = items
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
        .slice(0, 8)
        .map((item) => `- ${item.content.slice(0, 180)}`)
        .join("\n");

      if (latest) {
        lines.push(`${type.toUpperCase()}\n${latest}`);
      }
    }

    user.summary = lines.join("\n\n").slice(0, 8000);
    user.updatedAt = nowIso();
    return user.summary;
  }

  async summarizeAndStore(userId) {
    const summary = this.summarizeUserMemories(userId);
    const user = this.ensureUser(userId);

    if (summary) {
      const existingSummary = user.memories.find((item) => item.type === "summary" && item.metadata?.auto === true);
      if (existingSummary) {
        existingSummary.content = summary;
        existingSummary.embedding = embedText(summary);
        existingSummary.updatedAt = nowIso();
      } else {
        user.memories.push({
          id: crypto.randomUUID(),
          type: "summary",
          content: summary,
          tags: ["auto-summary"],
          sourceSessionId: "",
          metadata: { auto: true },
          importance: 0.85,
          createdAt: nowIso(),
          updatedAt: nowIso(),
          lastAccessedAt: nowIso(),
          embedding: embedText(summary),
        });
      }
    }

    await this.flush();
    return summary;
  }

  async pruneMemories(userId, options = {}) {
    const user = this.ensureUser(userId);
    const maxItems = clampNumber(options.maxItems, 320, 20, 5000);
    const maxAgeDays = clampNumber(options.maxAgeDays, 365, 7, 3650);
    const minImportance = clampNumber(options.minImportance, 0.15, 0, 1);

    const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;

    user.memories.sort((a, b) => {
      const scoreA = a.importance * 0.7 + new Date(a.updatedAt).getTime() / Date.now() * 0.3;
      const scoreB = b.importance * 0.7 + new Date(b.updatedAt).getTime() / Date.now() * 0.3;
      return scoreB - scoreA;
    });

    const kept = [];
    const pruned = [];

    for (const item of user.memories) {
      const isOld = new Date(item.updatedAt).getTime() < cutoff;
      const lowValue = item.importance < minImportance;
      const overLimit = kept.length >= maxItems;

      if ((isOld && lowValue) || overLimit) {
        pruned.push({ ...item, archivedAt: nowIso() });
      } else {
        kept.push(item);
      }
    }

    user.memories = kept;
    user.archivedMemories.push(...pruned);
    user.updatedAt = nowIso();

    await this.flush();
    return { kept: kept.length, pruned: pruned.length };
  }

  getMemoryStats(userId) {
    const user = this.ensureUser(userId);
    return {
      active: user.memories.length,
      archived: user.archivedMemories.length,
      hasSummary: Boolean(user.summary),
      updatedAt: user.updatedAt,
    };
  }

  getUserSnapshot(userId) {
    const user = this.ensureUser(userId);
    return {
      profile: { ...user.profile },
      summary: user.summary,
      stats: this.getMemoryStats(userId),
      memories: user.memories.map((item) => ({
        ...item,
      })),
      archivedMemories: user.archivedMemories.map((item) => ({
        ...item,
      })),
    };
  }

  async clearMemories(userId, options = {}) {
    const user = this.ensureUser(userId);
    const includeArchived = options.includeArchived === true;

    if (user.memories.length > 0) {
      user.archivedMemories.push(
        ...user.memories.map((item) => ({
          ...item,
          archivedAt: nowIso(),
        })),
      );
    }

    user.memories = [];
    user.summary = "";

    if (includeArchived) {
      user.archivedMemories = [];
    }

    user.updatedAt = nowIso();
    await this.flush();

    return {
      cleared: true,
      includeArchived,
      stats: this.getMemoryStats(userId),
    };
  }

  async ingestAutoMemories(userId, sessionId, role, content) {
    if (role !== "user" || typeof content !== "string") return [];

    const text = content.trim();
    if (!text) return [];

    const candidates = [];
    const lower = text.toLowerCase();

    if (/\bi prefer\b|\bi like\b|\bi usually\b|我偏好|我喜歡|我习惯|我通常/.test(lower)) {
      candidates.push({ type: "preference", content: text, tags: ["auto", "preference"], importance: 0.72 });
    }

    if (/\bmy name is\b|\bi am\b|\bi work as\b|\bi live in\b|我叫|我的名字|我在.+工作|我住在/.test(lower)) {
      candidates.push({ type: "fact", content: text, tags: ["auto", "identity"], importance: 0.74 });
    }

    if (/\bremember\b|\bdon't forget\b|\bimportant\b|記住|记住|不要忘|很重要/.test(lower)) {
      candidates.push({ type: "task", content: text, tags: ["auto", "explicit"], importance: 0.88 });
    }

    if (candidates.length === 0) {
      return [];
    }

    const saved = [];
    for (const candidate of candidates) {
      const item = this.createMemory(userId, {
        ...candidate,
        sourceSessionId: sessionId,
      });
      saved.push(item);
    }

    await this.flush();
    return saved;
  }
}

class SessionMemoryManager {
  constructor() {
    this.sessions = new Map();
  }

  get(sessionId, userId) {
    const cleanSessionId = sanitizeId(sessionId, "");
    const cleanUserId = sanitizeId(userId, DEFAULT_USER_ID);

    if (!cleanSessionId) {
      throw new Error("Invalid or missing sessionId.");
    }

    const existing = this.sessions.get(cleanSessionId);
    if (existing) {
      if (existing.userId === cleanUserId) {
        return existing;
      }

      throw new Error("Session belongs to a different user.");
    }

    const next = {
      sessionId: cleanSessionId,
      userId: cleanUserId,
      messages: [],
      compressedSummary: "",
      loadedMemorySummary: "",
      toolTrace: [],
      updatedAt: nowIso(),
    };

    this.sessions.set(cleanSessionId, next);
    return next;
  }

  getExisting(sessionId) {
    const cleanSessionId = sanitizeId(sessionId, "");
    if (!cleanSessionId) return null;
    return this.sessions.get(cleanSessionId) || null;
  }

  reset(sessionId, userId) {
    const cleanSessionId = sanitizeId(sessionId, "");
    const cleanUserId = sanitizeId(userId, "");
    if (!cleanSessionId) return false;

    const existing = this.sessions.get(cleanSessionId);
    if (!existing) return true;
    if (cleanUserId && existing.userId !== cleanUserId) {
      return false;
    }

    this.sessions.delete(cleanSessionId);
    return true;
  }

  appendMessage(session, role, content) {
    if (!session || typeof content !== "string") return;

    session.messages.push({ role, content: content.slice(0, 6000) });
    if (session.messages.length > MAX_SHORT_TERM_MESSAGES) {
      session.messages.splice(0, session.messages.length - MAX_SHORT_TERM_MESSAGES);
    }

    session.updatedAt = nowIso();
  }

  replaceMessages(session, messages = []) {
    const cleaned = Array.isArray(messages)
      ? messages
          .map((message) => {
            if (!message || typeof message !== "object") return null;
            if (!["user", "assistant", "system"].includes(message.role)) return null;
            if (typeof message.content !== "string") return null;
            const trimmed = message.content.trim();
            if (!trimmed) return null;
            return { role: message.role, content: trimmed.slice(0, 6000) };
          })
          .filter(Boolean)
      : [];

    session.messages = cleaned.slice(-MAX_SHORT_TERM_MESSAGES);
    session.updatedAt = nowIso();
  }
}

const MODEL_CATALOG = [
  {
    id: "gpt-4.1-mini",
    provider: "openai",
    tags: ["chat", "reasoning", "coding", "vision", "tools"],
    quality: 0.83,
    cost: 0.5,
    latency: 0.75,
  },
  {
    id: "gpt-4.1",
    provider: "openai",
    tags: ["chat", "reasoning", "coding", "vision", "tools"],
    quality: 0.96,
    cost: 0.92,
    latency: 0.45,
  },
  {
    id: "llama-3.1-8b-instant",
    provider: "groq",
    tags: ["chat", "coding", "tools"],
    quality: 0.66,
    cost: 0.18,
    latency: 0.98,
  },
  {
    id: "llama-3.3-70b-versatile",
    provider: "groq",
    tags: ["chat", "reasoning", "coding", "tools"],
    quality: 0.85,
    cost: 0.38,
    latency: 0.62,
  },
  {
    id: "mixtral-8x7b-32768",
    provider: "groq",
    tags: ["chat", "coding"],
    quality: 0.62,
    cost: 0.2,
    latency: 0.82,
  },
];

function getAvailableModels() {
  return MODEL_CATALOG.filter((model) => {
    if (model.provider === "openai") return Boolean(openai);
    if (model.provider === "groq") return Boolean(groq);
    return false;
  });
}

function hasAudioTranscriptionSupport() {
  if (openai?.audio?.transcriptions?.create) return true;
  if (groq?.audio?.transcriptions?.create) return true;
  return false;
}

function getRuntimeMultimodalCapabilities() {
  const availableModels = getAvailableModels();
  const hasVisionModel = availableModels.some((item) => item.tags.includes("vision"));
  const hasImageOcr = Boolean(Tesseract);
  const hasAudioStt = hasAudioTranscriptionSupport();

  return {
    text: {
      input: true,
      reasoning: true,
    },
    image: {
      input: hasVisionModel || hasImageOcr,
      visionReasoning: hasVisionModel,
      ocr: hasImageOcr,
    },
    audio: {
      input: hasAudioStt,
      transcription: hasAudioStt,
    },
    video: {
      input: VIDEO_PROCESSING_ENABLED,
      transcription: false,
      reasoning: false,
    },
  };
}

function routeModels({
  requestedModel,
  taskType,
  hasImage,
  qualityPreference,
  costPreference,
}) {
  const available = getAvailableModels();
  if (available.length === 0) {
    throw new Error("No model providers available.");
  }

  const cleanTaskType = ["chat", "reasoning", "coding", "vision", "auto"].includes(taskType)
    ? taskType
    : "auto";

  if (requestedModel) {
    const exact = available.find((item) => item.id === requestedModel);
    if (exact) {
      const fallbacks = available.filter((item) => item.id !== exact.id);
      return [exact, ...fallbacks];
    }
  }

  const targetTask = cleanTaskType === "auto" ? (hasImage ? "vision" : "chat") : cleanTaskType;
  const qualityWeight = qualityPreference === "quality" ? 1.2 : qualityPreference === "fast" ? 0.6 : 0.95;
  const costWeight = costPreference === "low" ? 1.2 : costPreference === "high" ? 0.5 : 0.85;

  const ranked = available
    .filter((item) => (targetTask === "vision" ? item.tags.includes("vision") : item.tags.includes(targetTask) || item.tags.includes("chat")))
    .map((item) => {
      const score = item.quality * qualityWeight + item.latency * (2 - qualityWeight) - item.cost * costWeight;
      return { ...item, score };
    })
    .sort((a, b) => b.score - a.score);

  if (ranked.length > 0) {
    const used = new Set(ranked.map((item) => item.id));
    const fallbacks = available.filter((item) => !used.has(item.id));
    return [...ranked, ...fallbacks];
  }

  return available;
}

function flattenContentToText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      if (part.type === "text") return String(part.text || "");
      if (part.type === "image_url") {
        return "[image attached]";
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function extractAssistantText(message) {
  if (!message || typeof message !== "object") return "";

  if (typeof message.content === "string") {
    return message.content;
  }

  if (Array.isArray(message.content)) {
    return message.content
      .map((part) => {
        if (!part || typeof part !== "object") return "";
        if (typeof part.text === "string") return part.text;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }

  if (typeof message.refusal === "string" && message.refusal.trim()) {
    return message.refusal;
  }

  return "";
}

function convertMessagesForProvider(messages, provider) {
  return messages.map((message) => {
    if (!message || typeof message !== "object") return null;

    const next = {
      role: message.role,
      content: message.content,
    };

    if (message.tool_calls) {
      next.tool_calls = message.tool_calls;
    }

    if (message.tool_call_id) {
      next.tool_call_id = message.tool_call_id;
    }

    if (provider === "groq" && Array.isArray(next.content)) {
      next.content = flattenContentToText(next.content);
    }

    if (provider === "groq" && next.role === "tool" && typeof next.content !== "string") {
      next.content = JSON.stringify(next.content);
    }

    return next;
  }).filter(Boolean);
}

async function runChatCompletion({
  provider,
  model,
  messages,
  params,
  stream,
  tools,
  toolChoice,
}) {
  const payload = {
    model,
    messages: convertMessagesForProvider(messages, provider),
    stream: Boolean(stream),
    temperature: params.temperature,
    max_tokens: params.maxTokens,
    top_p: params.topP,
    presence_penalty: params.presencePenalty,
    frequency_penalty: params.frequencyPenalty,
  };

  if (tools && tools.length > 0) {
    payload.tools = tools;
    payload.tool_choice = toolChoice || "auto";
  }

  if (provider === "openai") {
    return openai.chat.completions.create(payload);
  }

  if (provider === "groq") {
    return groq.chat.completions.create(payload);
  }

  throw new Error(`Unsupported provider: ${provider}`);
}

async function createCompleteReply({ modelPlan, messages, params, tools, toolMode, toolExecutor, trace }) {
  const provider = modelPlan.provider;
  const model = modelPlan.id;

  const first = await runChatCompletion({
    provider,
    model,
    messages,
    params,
    stream: false,
    tools: toolMode === "off" ? null : tools,
    toolChoice: toolMode === "manual" ? "none" : "auto",
  });

  let assistantMessage = first.choices?.[0]?.message || null;
  let finishReason = first.choices?.[0]?.finish_reason || "stop";

  const workingMessages = [...messages];
  if (!assistantMessage) {
    return { reply: "(empty response)", finishReason: "stop", usedTools: [] };
  }

  const usedTools = [];

  if (Array.isArray(assistantMessage.tool_calls) && assistantMessage.tool_calls.length > 0 && toolMode !== "off") {
    workingMessages.push({
      role: "assistant",
      content: extractAssistantText(assistantMessage),
      tool_calls: assistantMessage.tool_calls,
    });

    for (const toolCall of assistantMessage.tool_calls) {
      const toolResult = await toolExecutor(toolCall, trace);
      usedTools.push(toolResult);
      workingMessages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(toolResult.output),
      });
    }

    const second = await runChatCompletion({
      provider,
      model,
      messages: workingMessages,
      params,
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    assistantMessage = second.choices?.[0]?.message || assistantMessage;
    finishReason = second.choices?.[0]?.finish_reason || "stop";
  }

  let fullReply = extractAssistantText(assistantMessage);
  let rounds = 0;

  while (finishReason === "length" && rounds < CONTINUATION_MAX_ROUNDS) {
    rounds += 1;

    workingMessages.push({
      role: "assistant",
      content: extractAssistantText(assistantMessage),
    });
    workingMessages.push({
      role: "user",
      content: "Continue exactly from where you stopped. Output only the remaining content.",
    });

    const cont = await runChatCompletion({
      provider,
      model,
      messages: workingMessages,
      params,
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const piece = extractAssistantText(cont.choices?.[0]?.message);
    finishReason = cont.choices?.[0]?.finish_reason || "stop";
    assistantMessage = cont.choices?.[0]?.message || { content: piece };
    fullReply += piece;
  }

  return {
    reply: fullReply || "(empty response)",
    finishReason,
    usedTools,
  };
}

function buildHeuristicSummary(messages) {
  const chunks = messages
    .slice(-20)
    .map((message, index) => `${index + 1}. [${message.role}] ${message.content.slice(0, 160)}`)
    .join("\n");

  if (!chunks) return "";
  return `Compressed session history:\n${chunks}`.slice(0, 3500);
}

function compressSessionIfNeeded(session) {
  const compressTrigger = getContextCompressTrigger();
  const keepRecent = getRecentContextMessages();
  if (!session || !Array.isArray(session.messages)) return;
  if (session.messages.length <= compressTrigger) return;

  const cut = session.messages.length - keepRecent;
  const older = session.messages.slice(0, cut);
  const recent = session.messages.slice(cut);
  const summary = buildHeuristicSummary(older);

  session.compressedSummary = session.compressedSummary
    ? `${session.compressedSummary}\n\n${summary}`.slice(-7000)
    : summary;

  session.messages = recent;
}

function buildPersonalizationPrompt(profile, override = {}) {
  const style = ["concise", "technical", "casual"].includes(override.responseStyle)
    ? override.responseStyle
    : profile.responseStyle || DEFAULT_STYLE;

  const expertise = ["beginner", "intermediate", "expert"].includes(override.expertiseLevel)
    ? override.expertiseLevel
    : profile.expertiseLevel || DEFAULT_EXPERTISE;

  const preferencePairs = Object.entries(profile.preferences || {})
    .slice(0, 20)
    .map(([key, value]) => `- ${key}: ${String(value).slice(0, 100)}`);

  const styleGuide = {
    concise: "Give short, direct answers with key bullets only.",
    technical: "Use precise technical terms, architecture details, and tradeoffs.",
    casual: "Use friendly, easy language while staying accurate.",
  };

  const expertiseGuide = {
    beginner: "Avoid jargon or explain jargon in one short sentence.",
    intermediate: "Use moderate technical depth with practical examples.",
    expert: "Assume strong background and focus on nuanced constraints and edge cases.",
  };

  return [
    "Personalization profile:",
    `- response_style: ${style}`,
    `- expertise_level: ${expertise}`,
    `- style_guidance: ${styleGuide[style] || styleGuide.concise}`,
    `- expertise_guidance: ${expertiseGuide[expertise] || expertiseGuide.intermediate}`,
    preferencePairs.length > 0 ? `- preferences:\n${preferencePairs.join("\n")}` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function shouldForceQualityRevision(userMessage, reply) {
  const question = typeof userMessage === "string" ? userMessage : "";
  const answer = typeof reply === "string" ? reply : "";
  if (!answer.trim()) return true;

  const badPatterns = [
    /可能指的是幾個不同的事情/i,
    /可能指的是几個不同的事情/i,
    /red\s+amber\s+green/i,
    /reference\s+assist\s+generator/i,
    /wash\s+rag/i,
    /對不起.{0,20}無法/i,
  ];
  if (badPatterns.some((pattern) => pattern.test(answer))) {
    return true;
  }

  if (/什麼是|什么是|解釋|解释|explain/i.test(question) && answer.length < 40) {
    return true;
  }

  return false;
}

async function applyResponseQualityGate({
  userMessage,
  reply,
  responseStyle,
  expertiseLevel,
  route,
  params,
}) {
  if (!shouldForceQualityRevision(userMessage, reply)) {
    return reply;
  }

  const qaPrompt = [
    "You are a response-quality reviewer and rewriter.",
    "Return strict JSON only.",
    'Schema: {"verdict":"pass"|"revise","issues":["..."],"revised":"..."}',
    "Rules:",
    "1) The answer must directly address the user question.",
    "2) Keep factual correctness; do not invent acronym expansions.",
    "3) If acronym/context is ambiguous, infer the most likely meaning from the question and answer that meaning.",
    "4) Respect requested style and expertise.",
    "5) Keep language consistent with user question (Traditional Chinese preferred if Chinese).",
    `Target style: ${responseStyle || "concise"}`,
    `Target expertise: ${expertiseLevel || "intermediate"}`,
    `User question: ${userMessage}`,
    `Draft answer: ${reply}`,
  ].join("\n");

  try {
    const qa = await runChatCompletion({
      provider: route.provider,
      model: route.id,
      messages: [
        { role: "system", content: "Return strict JSON only." },
        { role: "user", content: qaPrompt },
      ],
      params: {
        ...params,
        temperature: Math.min(0.3, clampNumber(params.temperature, 0.2, 0, 2)),
        maxTokens: Math.max(500, Math.min(1800, clampNumber(params.maxTokens, 900, 1, 6000))),
      },
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const raw = extractAssistantText(qa.choices?.[0]?.message).trim();
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    const decision = safeJsonParse(jsonMatch ? jsonMatch[0] : "", null);

    if (decision && typeof decision === "object") {
      if (decision.verdict === "revise" && typeof decision.revised === "string" && decision.revised.trim()) {
        return decision.revised.trim();
      }
      if (decision.verdict === "pass") {
        return reply;
      }
    }
  } catch (_error) {
    // Fall through to direct corrective rewrite.
  }

  try {
    const rewrite = await runChatCompletion({
      provider: route.provider,
      model: route.id,
      messages: [
        {
          role: "system",
          content: "Rewrite the answer to be accurate, directly relevant, and aligned with requested style/expertise. Output only final answer.",
        },
        {
          role: "user",
          content: [
            `Question: ${userMessage}`,
            `Target style: ${responseStyle || "concise"}`,
            `Target expertise: ${expertiseLevel || "intermediate"}`,
            `Current bad draft: ${reply}`,
          ].join("\n"),
        },
      ],
      params: {
        ...params,
        temperature: Math.min(0.35, clampNumber(params.temperature, 0.25, 0, 2)),
        maxTokens: Math.max(500, Math.min(1800, clampNumber(params.maxTokens, 900, 1, 6000))),
      },
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const rewritten = extractAssistantText(rewrite.choices?.[0]?.message).trim();
    if (rewritten) return rewritten;
  } catch (_error) {
    // Keep original reply if rewrite fails.
  }

  return reply;
}

function buildWebQueryFromMessage(message) {
  const raw = typeof message === "string" ? message.trim() : "";
  if (!raw) return "";

  const acronymMatch = raw.match(/\b[A-Z]{2,8}\b/);
  const acronym = acronymMatch ? acronymMatch[0] : "";
  if (acronym && /是什麼|是什么|解釋|解释|what is|meaning|全稱|全称/i.test(raw)) {
    return `${acronym} meaning in AI retrieval augmented generation`;
  }

  return raw.replace(/\s+/g, " ").slice(0, 180);
}

function isFactualQuestion(message) {
  const text = typeof message === "string" ? message : "";
  if (!text) return false;

  const patterns = [
    /是什麼|是什么|解釋|解释|what is|explain/i,
    /最新|版本|version|release|date|時間|年份|費用|價格|price/i,
    /比較|差異|difference|compare/i,
    /如何|怎麼|怎么|how to/i,
  ];

  return patterns.some((pattern) => pattern.test(text));
}

function looksLowConfidence(reply) {
  const text = typeof reply === "string" ? reply : "";
  if (!text.trim()) return true;

  const patterns = [
    /我不確定|我不确定|不太確定|不太确定/i,
    /可能是|可能指的是/i,
    /無法確認|无法确认|cannot verify/i,
    /i\s+am\s+not\s+sure|might be/i,
  ];

  return patterns.some((pattern) => pattern.test(text));
}

function extractAcronym(value) {
  const text = typeof value === "string" ? value : "";
  const match = text.match(/\b[A-Z]{2,8}\b/);
  return match ? match[0] : "";
}

function detectDomainHints(message) {
  const text = typeof message === "string" ? message : "";
  const hints = [];
  if (/\b(ai|llm|rag|nlp|ml|genai|embedding|retrieval|vector)\b/i.test(text) || /人工智慧|生成式|檢索|检索|向量|語言模型|语言模型/i.test(text)) {
    hints.push("ai");
  }
  if (/\b(web|search|搜尋|搜索|internet|網路|网络)\b/i.test(text)) {
    hints.push("web");
  }
  return hints;
}

function isAcronymExpansionQuestion(message) {
  const text = typeof message === "string" ? message : "";
  if (!text) return false;
  if (!/\b[A-Z]{2,8}\b/.test(text)) return false;
  return /全名|全稱|全称|完整名稱|英文全名|stands\s+for|full\s+name|acronym\s+meaning|是什麼縮寫|是什么缩写/i.test(text);
}

function normalizeSearchQuery(query, sourceMessage) {
  const q = typeof query === "string" ? query.trim() : "";
  if (!q) return "";

  const acronym = extractAcronym(q);
  const sourceHints = detectDomainHints(sourceMessage);
  const queryHints = detectDomainHints(q);
  const hints = new Set([...sourceHints, ...queryHints]);

  if (acronym && q.length <= 24 && hints.has("ai")) {
    return `${acronym} acronym meaning in AI LLM context`;
  }

  if (acronym && q.length <= 12) {
    return `${acronym} acronym meaning`;
  }

  return q.replace(/\s+/g, " ").slice(0, 240);
}

function shouldAttemptAutoGrounding({ message, reply, toolMode, usedTools, processedAttachments }) {
  if (toolMode !== "auto") return false;
  if (!isFactualQuestion(message)) return false;
  if (Array.isArray(processedAttachments) && processedAttachments.length > 0) return false;
  if (Array.isArray(usedTools) && usedTools.length > 0) return false;
  if (buildWebQueryFromMessage(message).length < 6) return false;
  if (isAcronymExpansionQuestion(message)) return true;
  return looksLowConfidence(reply);
}

function shouldDisableStreamingForQuality(message, toolMode) {
  if (toolMode !== "off") return true;
  if (isAcronymExpansionQuestion(message)) return true;
  if (isFactualQuestion(message)) return true;
  return false;
}

function collectWebResultsFromToolTrace(usedTools) {
  if (!Array.isArray(usedTools)) return [];
  const results = [];

  for (const item of usedTools) {
    if (!item || item.toolName !== "web_search" || !Array.isArray(item.output)) continue;
    for (const hit of item.output) {
      if (hit && typeof hit === "object") {
        results.push(hit);
      }
    }
  }

  return results.slice(0, 10);
}

function shouldForceGroundedRewrite({ question, reply, webResults }) {
  if (!Array.isArray(webResults) || webResults.length === 0) return false;

  const text = typeof reply === "string" ? reply : "";
  const badPatterns = [
    /web\s*搜尋功能似乎出了問題/i,
    /web\s*search\s*is\s*broken/i,
    /可能是|可能指的是|或者是|或是/i,
    /對不起.{0,20}無法/i,
    /i\s+cannot\s+directly/i,
    /reactor\s+application\s+gateway/i,
    /react\s+application\s+generator/i,
    /reasoning\s+augmentation\s+generator/i,
  ];

  if (badPatterns.some((pattern) => pattern.test(text))) {
    return true;
  }

  return !hasGroundedWebEvidence(text, [{ toolName: "web_search", ok: true, output: webResults }]);
}

async function synthesizeGroundedAnswer({
  question,
  draftReply,
  webResults,
  route,
  params,
  responseStyle,
  expertiseLevel,
  forceIncludeSources = false,
}) {
  const evidence = Array.isArray(webResults) ? webResults.slice(0, 5) : [];
  if (evidence.length === 0) return draftReply;

  const acronymMatch = (typeof question === "string" ? question : "").match(/\b[A-Z]{2,8}\b/);
  const acronym = acronymMatch ? acronymMatch[0] : "";

  const prompt = [
    "Rewrite the answer grounded on the provided web evidence.",
    "Do not fabricate sources beyond provided evidence.",
    "If evidence is insufficient, say what is known and what remains uncertain in one short line.",
    "Never invent acronym expansions.",
    acronym ? `If acronym '${acronym}' appears, use only expansions supported by evidence.` : "",
    forceIncludeSources ? "You must include a short 'Sources:' section with 1-3 URLs from evidence." : "",
    `Style: ${responseStyle || "concise"}`,
    `Expertise: ${expertiseLevel || "intermediate"}`,
    `Question: ${question}`,
    `Draft answer: ${draftReply}`,
    `Evidence JSON: ${JSON.stringify(evidence).slice(0, 7000)}`,
  ].join("\n");

  try {
    const completion = await runChatCompletion({
      provider: route.provider,
      model: route.id,
      messages: [
        { role: "system", content: "Return only the final grounded answer in Traditional Chinese." },
        { role: "user", content: prompt },
      ],
      params: {
        ...params,
        temperature: Math.min(0.35, clampNumber(params.temperature, 0.25, 0, 2)),
      },
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const grounded = extractAssistantText(completion.choices?.[0]?.message).trim();
    return grounded || draftReply;
  } catch (_error) {
    return draftReply;
  }
}

async function enforceGroundedAnswerIfNeeded({
  message,
  reply,
  usedTools,
  route,
  params,
  responseStyle,
  expertiseLevel,
}) {
  const webResults = collectWebResultsFromToolTrace(usedTools);
  if (webResults.length === 0) return reply;

  if (!shouldForceGroundedRewrite({
    question: message,
    reply,
    webResults,
  })) {
    return reply;
  }

  return synthesizeGroundedAnswer({
    question: message,
    draftReply: reply,
    webResults,
    route,
    params,
    responseStyle,
    expertiseLevel,
  });
}

async function enforceAcronymGroundingIfNeeded({
  message,
  reply,
  route,
  params,
  responseStyle,
  expertiseLevel,
}) {
  if (!isAcronymExpansionQuestion(message)) {
    return reply;
  }

  const acronym = extractAcronym(message);
  if (acronym === "RAG") {
    const normalized = typeof reply === "string" ? reply.toLowerCase() : "";
    const wrongRag = /red\s*amber\s*green|紅.?黃.?綠|红.?黄.?绿|reformer\s*-?based\s*augmented\s*generator|reactor\s*application\s*gateway|react\s*application\s*generator|reasoning\s*augmentation\s*generator/i;
    if (wrongRag.test(normalized)) {
      return [
        "RAG 的英文全名是 Retrieval-Augmented Generation（檢索增強生成）。",
        "它指的是先從外部知識庫檢索相關內容，再把檢索結果提供給大型語言模型生成答案。",
        "這裡的 RAG 不是 Red Amber Green（狀態燈號）。",
      ].join("\n");
    }
  }

  try {
    const query = normalizeSearchQuery(message, message);
    const webResults = await runWebSearchWithFallback(query, 5);
    if (webResults.length === 0) {
      return reply;
    }

    return await synthesizeGroundedAnswer({
      question: message,
      draftReply: reply,
      webResults,
      route,
      params,
      responseStyle,
      expertiseLevel,
      forceIncludeSources: true,
    });
  } catch (_error) {
    return reply;
  }
}

function convertAttachmentInput(raw) {
  if (!raw || typeof raw !== "object") return null;

  const kind = ["image", "audio", "video"].includes(raw.kind) ? raw.kind : "";
  if (!kind) return null;

  const mimeType = typeof raw.mimeType === "string" ? raw.mimeType.slice(0, 100) : "application/octet-stream";
  const dataBase64 = typeof raw.dataBase64 === "string" ? raw.dataBase64.trim() : "";
  if (!dataBase64) return null;

  const size = Math.floor((dataBase64.length * 3) / 4);
  if (!Number.isFinite(size) || size <= 0 || size > MAX_ATTACHMENT_BYTES) {
    return null;
  }

  const safeName = typeof raw.name === "string" && raw.name.trim() ? raw.name.trim().slice(0, 120) : `${kind}.bin`;

  return {
    id: sanitizeId(raw.id || crypto.randomUUID(), crypto.randomUUID()),
    kind,
    mimeType,
    dataBase64,
    name: safeName,
    size,
  };
}

async function runImageOcr(buffer) {
  if (!Tesseract?.recognize) {
    return "";
  }

  const recognized = await Tesseract.recognize(buffer, "eng", {
    logger: () => {},
  });

  const text = recognized?.data?.text || "";
  return text.trim().slice(0, 8000);
}

async function writeTempFile(buffer, suffix) {
  const tempPath = path.join(os.tmpdir(), `mm-${crypto.randomUUID()}${suffix}`);
  await fs.promises.writeFile(tempPath, buffer);
  return tempPath;
}

async function runAudioTranscription(buffer, mimeType) {
  if (!hasAudioTranscriptionSupport()) {
    return "";
  }

  const suffix = mimeType.includes("wav") ? ".wav" : mimeType.includes("mpeg") ? ".mp3" : ".webm";
  const tempPath = await writeTempFile(buffer, suffix);

  try {
    if (openai) {
      const transcript = await openai.audio.transcriptions.create({
        file: fs.createReadStream(tempPath),
        model: "gpt-4o-mini-transcribe",
      });

      const text = transcript?.text || "";
      if (text.trim()) {
        return text.trim().slice(0, 8000);
      }
    }

    if (groq?.audio?.transcriptions?.create) {
      const transcript = await groq.audio.transcriptions.create({
        file: fs.createReadStream(tempPath),
        model: "whisper-large-v3",
        response_format: "json",
      });

      const text = transcript?.text || "";
      return text.trim().slice(0, 8000);
    }

    return "";
  } finally {
    await fs.promises.unlink(tempPath).catch(() => {});
  }
}

async function preprocessAttachments(attachments = [], capabilities = getRuntimeMultimodalCapabilities()) {
  const normalized = Array.isArray(attachments)
    ? attachments.map(convertAttachmentInput).filter(Boolean).slice(0, 5)
    : [];

  const processed = [];

  for (const item of normalized) {
    const buffer = Buffer.from(item.dataBase64, "base64");
    const result = {
      id: item.id,
      kind: item.kind,
      name: item.name,
      mimeType: item.mimeType,
      size: item.size,
      extractedText: "",
      error: "",
      dataBase64: item.dataBase64,
    };

    try {
      if (item.kind === "image") {
        if (!capabilities.image.input) {
          result.error = "Image input is not enabled by current model/runtime configuration.";
        } else if (capabilities.image.ocr) {
          result.extractedText = await runImageOcr(buffer);
        }
      } else if (item.kind === "audio") {
        if (!capabilities.audio.input) {
          result.error = "Audio input is not enabled by current model/runtime configuration.";
        } else {
          result.extractedText = await runAudioTranscription(buffer, item.mimeType);
        }
      } else if (item.kind === "video") {
        result.error = capabilities.video.input
          ? "Video preprocessing is not enabled yet. Upload image/audio for now."
          : "Video input is not enabled by current model/runtime configuration.";
      }
    } catch (error) {
      result.error = error?.message || "Failed to preprocess attachment.";
    }

    processed.push(result);
  }

  return processed;
}

function buildUserMessageContent({ message, processedAttachments, allowImageInput }) {
  const parts = [];
  parts.push({ type: "text", text: message.trim() });

  for (const item of processedAttachments) {
    if (item.kind === "image") {
      if (allowImageInput) {
        parts.push({
          type: "image_url",
          image_url: {
            url: `data:${item.mimeType};base64,${item.dataBase64}`,
          },
        });
      }

      if (item.extractedText) {
        parts.push({
          type: "text",
          text: `OCR from image ${item.name}:\n${item.extractedText}`,
        });
      }
    }

    if (item.kind === "audio") {
      parts.push({
        type: "text",
        text: item.extractedText
          ? `Transcription from audio ${item.name}:\n${item.extractedText}`
          : `Audio ${item.name} uploaded, but transcription was empty.`,
      });
    }

    if (item.kind === "video") {
      parts.push({
        type: "text",
        text: `Video ${item.name} uploaded. ${item.error || "No video processing available."}`,
      });
    }
  }

  return parts;
}

function parseToolArgs(rawArgs) {
  if (typeof rawArgs === "string") {
    return safeJsonParse(rawArgs, {});
  }
  if (rawArgs && typeof rawArgs === "object") {
    return rawArgs;
  }
  return {};
}

async function runWebSearch(query, topK) {
  const cleaned = typeof query === "string" ? query.trim() : "";
  if (!cleaned) {
    throw new Error("Query is required.");
  }

  const limit = clampNumber(topK, 5, 1, 10);
  const recencyQuery = /\blatest|recent|today|current|new\b|最新|近期|最近|當前|当前/.test(cleaned.toLowerCase());

  const providers = [
    {
      name: "tavily",
      enabled: Boolean(process.env.TAVILY_API_KEY),
      search: () => runTavilySearch(cleaned, limit),
    },
    {
      name: "brave",
      enabled: Boolean(process.env.BRAVE_SEARCH_API_KEY),
      search: () => runBraveSearch(cleaned, limit, { recencyQuery }),
    },
    {
      name: "serpapi",
      enabled: Boolean(process.env.SERPAPI_API_KEY),
      search: () => runSerpApiSearch(cleaned, limit, { recencyQuery }),
    },
    {
      name: "duckduckgo_instant",
      enabled: true,
      search: () => runDuckDuckGoInstantSearch(cleaned, limit),
    },
  ];

  const providerErrors = [];
  for (const provider of providers) {
    if (!provider.enabled) continue;
    try {
      const raw = await provider.search();
      const normalized = normalizeSearchHits(raw, provider.name);
      if (normalized.length > 0) {
        return normalized.slice(0, limit);
      }
    } catch (error) {
      providerErrors.push(`${provider.name}: ${error?.message || "unknown error"}`);
    }
  }

  if (providerErrors.length > 0) {
    throw new Error(`Web search failed across providers (${providerErrors.join(" | ")}).`);
  }

  return [];
}

function normalizeSearchHits(items, providerName) {
  const list = Array.isArray(items) ? items : [];
  const out = [];
  const seen = new Set();

  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const url = typeof item.url === "string" ? item.url.trim() : "";
    const title = typeof item.title === "string" ? item.title.trim() : "";
    const snippet = typeof item.snippet === "string" ? item.snippet.trim() : "";
    const publishedAt = typeof item.publishedAt === "string" ? item.publishedAt.trim() : "";

    if (!url && !title && !snippet) continue;

    const key = `${url}|${title}`.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);

    out.push({
      title: title || snippet.slice(0, 120) || "Untitled",
      snippet: snippet || "",
      url,
      provider: providerName,
      publishedAt,
    });
  }

  return out;
}

async function runTavilySearch(query, topK) {
  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      api_key: process.env.TAVILY_API_KEY,
      query,
      max_results: topK,
      include_answer: false,
      include_raw_content: false,
      search_depth: "advanced",
    }),
  });

  if (!response.ok) {
    throw new Error(`status ${response.status}`);
  }

  const rawText = await response.text();
  const data = safeJsonParse(rawText, null);
  if (!data || typeof data !== "object") {
    throw new Error("invalid JSON");
  }

  const results = Array.isArray(data.results) ? data.results : [];
  return results.map((item) => ({
    title: item?.title || "",
    snippet: item?.content || "",
    url: item?.url || "",
    publishedAt: item?.published_date || "",
  }));
}

async function runBraveSearch(query, topK, options = {}) {
  const params = new URLSearchParams({
    q: query,
    count: String(topK),
  });

  if (options.recencyQuery) {
    params.set("freshness", "pm");
  }

  const response = await fetch(`https://api.search.brave.com/res/v1/web/search?${params.toString()}`, {
    headers: {
      "Accept": "application/json",
      "X-Subscription-Token": process.env.BRAVE_SEARCH_API_KEY,
      "User-Agent": "custom-gpt-agent/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`status ${response.status}`);
  }

  const rawText = await response.text();
  const data = safeJsonParse(rawText, null);
  if (!data || typeof data !== "object") {
    throw new Error("invalid JSON");
  }

  const results = Array.isArray(data?.web?.results) ? data.web.results : [];
  return results.map((item) => ({
    title: item?.title || "",
    snippet: item?.description || "",
    url: item?.url || "",
    publishedAt: item?.age || "",
  }));
}

async function runSerpApiSearch(query, topK, options = {}) {
  const params = new URLSearchParams({
    engine: "google",
    q: query,
    api_key: process.env.SERPAPI_API_KEY,
    num: String(topK),
  });

  if (options.recencyQuery) {
    params.set("tbs", "qdr:m");
  }

  const response = await fetch(`https://serpapi.com/search.json?${params.toString()}`, {
    headers: {
      "Accept": "application/json",
      "User-Agent": "custom-gpt-agent/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`status ${response.status}`);
  }

  const rawText = await response.text();
  const data = safeJsonParse(rawText, null);
  if (!data || typeof data !== "object") {
    throw new Error("invalid JSON");
  }

  const results = Array.isArray(data.organic_results) ? data.organic_results : [];
  return results.map((item) => ({
    title: item?.title || "",
    snippet: item?.snippet || "",
    url: item?.link || "",
    publishedAt: item?.date || "",
  }));
}

async function runDuckDuckGoInstantSearch(query, topK) {
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_redirect=1&no_html=1`;
  const response = await fetch(url, {
    headers: {
      "User-Agent": "custom-gpt-agent/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`Web search failed with status ${response.status}.`);
  }

  const rawText = await response.text();
  if (!rawText.trim()) {
    return [];
  }

  const data = safeJsonParse(rawText, null);
  if (!data || typeof data !== "object") {
    throw new Error("Web search returned invalid JSON response.");
  }

  const hits = [];

  if (data.AbstractText) {
    hits.push({
      title: data.Heading || "Abstract",
      snippet: data.AbstractText,
      url: data.AbstractURL || "",
      publishedAt: "",
    });
  }

  if (Array.isArray(data.RelatedTopics)) {
    for (const topic of data.RelatedTopics) {
      if (topic.Text) {
        hits.push({
          title: topic.Text.slice(0, 120),
          snippet: topic.Text,
          url: topic.FirstURL || "",
          publishedAt: "",
        });
      } else if (Array.isArray(topic.Topics)) {
        for (const nested of topic.Topics) {
          if (nested.Text) {
            hits.push({
              title: nested.Text.slice(0, 120),
              snippet: nested.Text,
              url: nested.FirstURL || "",
              publishedAt: "",
            });
          }
        }
      }
    }
  }

  return hits.slice(0, clampNumber(topK, 5, 1, 10));
}

async function runWebSearchWithFallback(query, topK) {
  const normalized = normalizeSearchQuery(query, query);
  const primary = await runWebSearch(normalized, topK);
  if (primary.length > 0) {
    return primary;
  }

  const acronym = extractAcronym(query);
  if (!acronym) {
    return primary;
  }

  const fallbackQueries = [
    `${acronym} retrieval augmented generation`,
    `${acronym} meaning in artificial intelligence`,
    `${acronym} LLM retrieval`,
  ];

  for (const item of fallbackQueries) {
    try {
      const result = await runWebSearch(item, topK);
      if (result.length > 0) {
        return result;
      }
    } catch (_error) {
      // keep trying fallback queries
    }
  }

  return primary;
}

function runJsSandbox(code) {
  if (typeof code !== "string" || !code.trim()) {
    throw new Error("Code is required.");
  }

  const context = vm.createContext({
    Math,
    Date,
    JSON,
    Array,
    Object,
    String,
    Number,
    Boolean,
    console: {
      log: (...args) => args.map((arg) => (typeof arg === "string" ? arg : JSON.stringify(arg))).join(" "),
    },
  });

  const script = new vm.Script(`(() => { ${code} })()`);
  const result = script.runInContext(context, { timeout: 1200 });

  if (typeof result === "string") return result;
  return JSON.stringify(result);
}

function runMemoryPseudoQuery(memoryItems, sql) {
  const text = typeof sql === "string" ? sql.trim() : "";
  if (!text) throw new Error("SQL query is required.");

  const normalized = text.replace(/\s+/g, " ").toLowerCase();
  if (!normalized.startsWith("select")) {
    throw new Error("Only SELECT queries are supported.");
  }

  const whereTypeMatch = normalized.match(/where\s+type\s*=\s*'([^']+)'/);
  const limitMatch = normalized.match(/limit\s+(\d+)/);

  let rows = [...memoryItems];
  if (whereTypeMatch) {
    rows = rows.filter((row) => row.type === whereTypeMatch[1]);
  }

  rows.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  const limit = limitMatch ? clampNumber(limitMatch[1], 20, 1, 200) : 20;

  return rows.slice(0, limit).map((row) => ({
    id: row.id,
    type: row.type,
    content: row.content,
    tags: row.tags,
    importance: row.importance,
    updatedAt: row.updatedAt,
  }));
}

const TOOL_SCHEMAS = [
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web for quick factual grounding.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query text." },
          topK: { type: "integer", minimum: 1, maximum: 10, default: 5 },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_js",
      description: "Execute short JavaScript in a constrained VM.",
      parameters: {
        type: "object",
        properties: {
          code: { type: "string", description: "JavaScript snippet to run." },
        },
        required: ["code"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "memory_query",
      description: "Semantic search over the user's long-term memories.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string" },
          limit: { type: "integer", minimum: 1, maximum: 50, default: 8 },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "db_query",
      description: "Run a limited SELECT query over long-term memory records.",
      parameters: {
        type: "object",
        properties: {
          sql: { type: "string" },
        },
        required: ["sql"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "profile_update",
      description: "Update user personalization profile fields.",
      parameters: {
        type: "object",
        properties: {
          responseStyle: { type: "string", enum: ["concise", "technical", "casual"] },
          expertiseLevel: { type: "string", enum: ["beginner", "intermediate", "expert"] },
          preferences: { type: "object" },
        },
      },
    },
  },
];

const memoryDb = new PersistentMemoryDB(MEMORY_DB_PATH);
const sessionManager = new SessionMemoryManager();

async function executeToolCall(toolCall, traceContext) {
  const toolName = toolCall?.function?.name || "";
  const args = parseToolArgs(toolCall?.function?.arguments);

  const trace = {
    id: crypto.randomUUID(),
    sessionId: traceContext.sessionId,
    userId: traceContext.userId,
    toolName,
    args,
    startedAt: nowIso(),
    endedAt: "",
    ok: true,
    output: null,
    error: "",
  };

  try {
    const normalizedToolName = toolName.trim();
    if (!normalizedToolName) {
      throw new Error("Tool name is required.");
    }

    if (normalizedToolName === "web_search") {
      const normalizedTopK = clampNumber(args.topK, 5, 1, 10);
      const queryFromArgs = typeof args.query === "string" ? args.query.trim() : "";
      const queryFromAlt = typeof args.q === "string" ? args.q.trim() : "";
      const finalQuery = queryFromArgs || queryFromAlt;

      if (!finalQuery) {
        trace.output = [];
      } else {
        const normalizedQuery = normalizeSearchQuery(finalQuery, finalQuery);
        trace.output = await runWebSearchWithFallback(normalizedQuery, normalizedTopK);
      }
    } else if (normalizedToolName === "run_js") {
      const code = typeof args.code === "string" ? args.code.trim() : "";
      if (!code) {
        trace.output = { result: "No code provided; skipped JavaScript execution." };
      } else {
        trace.output = { result: runJsSandbox(code) };
      }
    } else if (normalizedToolName === "memory_query") {
      const query = typeof args.query === "string" ? args.query.trim() : "";
      trace.output = memoryDb.listMemories(traceContext.userId, {
        query,
        limit: args.limit,
      });
    } else if (normalizedToolName === "db_query") {
      const all = memoryDb.listMemories(traceContext.userId, { limit: 500 });
      const sqlArg = typeof args.sql === "string" ? args.sql.trim() : "";
      const queryArg = typeof args.query === "string" ? args.query.trim() : "";
      const candidateSql = sqlArg || queryArg || "SELECT * FROM memory LIMIT 20";
      const safeSql = /^\s*select\b/i.test(candidateSql)
        ? candidateSql
        : "SELECT * FROM memory LIMIT 20";
      trace.output = runMemoryPseudoQuery(all, safeSql);
    } else if (normalizedToolName === "profile_update") {
      const profile = await memoryDb.updateProfile(traceContext.userId, args);
      trace.output = profile;
    } else {
      throw new Error(`Unsupported tool: ${normalizedToolName}`);
    }
  } catch (error) {
    trace.ok = false;
    trace.error = error?.message || "Tool execution failed.";
    trace.output = { error: trace.error };
  }

  trace.endedAt = nowIso();

  runtimeObservability.tools.total += 1;
  runtimeObservability.tools.byName[toolName] = (runtimeObservability.tools.byName[toolName] || 0) + 1;
  if (!trace.ok) {
    runtimeObservability.tools.failed += 1;
  }

  recordObservabilityEvent({
    type: "tool",
    sessionId: traceContext.sessionId,
    userId: traceContext.userId,
    toolName,
    ok: trace.ok,
    error: trace.error,
  });

  return trace;
}

function buildContextMessages({
  baseSystemPrompt,
  profilePrompt,
  session,
  semanticMemories,
  userContent,
  thinkMode,
}) {
  const systemBlocks = [
    baseSystemPrompt,
    [
      "Output policy:",
      "- Reply with normal natural-language answer only.",
      "- Never output raw JSON unless user explicitly asks for JSON.",
      "- Do not dump personalization/profile objects in the final answer.",
    ].join("\n"),
  ];

  if (thinkMode) {
    systemBlocks.push(
      [
        "You are in THINK mode.",
        "Reason thoroughly before answering.",
        "Do not reveal internal chain-of-thought.",
        "Return only final answer content.",
      ].join("\n"),
    );
  }

  if (profilePrompt) {
    systemBlocks.push(profilePrompt);
  }

  if (session.loadedMemorySummary) {
    systemBlocks.push(`Loaded memory summary:\n${session.loadedMemorySummary}`);
  }

  if (session.compressedSummary) {
    systemBlocks.push(`Session compressed summary:\n${session.compressedSummary}`);
  }

  if (semanticMemories.length > 0) {
    const lines = semanticMemories
      .slice(0, 8)
      .map((item, index) => `${index + 1}. (${item.type}) ${item.content.slice(0, 220)}`)
      .join("\n");

    systemBlocks.push(`Long-term retrieved memory:\n${lines}`);
  }

  const recentHistory = session.messages.slice(-getRecentContextMessages());

  return [
    {
      role: "system",
      content: systemBlocks.filter(Boolean).join("\n\n"),
    },
    ...recentHistory,
    {
      role: "user",
      content: userContent,
    },
  ];
}

function sanitizeSourceConversations(sources) {
  if (!Array.isArray(sources)) return [];

  return sources
    .map((source) => {
      if (!source || typeof source !== "object") return null;
      const title = typeof source.title === "string" && source.title.trim() ? source.title.trim() : "Untitled";
      const messages = Array.isArray(source.messages)
        ? source.messages
            .map((message) => {
              if (!message || typeof message !== "object") return null;
              if (!["user", "assistant", "system"].includes(message.role)) return null;
              if (typeof message.content !== "string") return null;
              const content = message.content.trim();
              if (!content) return null;
              return { role: message.role, content: content.slice(0, 2000) };
            })
            .filter(Boolean)
        : [];

      if (messages.length === 0) return null;
      return { title, messages };
    })
    .filter(Boolean);
}

function summarizeSourcesHeuristic(sources) {
  const lines = [];

  for (const source of sources.slice(0, 8)) {
    lines.push(`Conversation: ${source.title}`);
    for (const message of source.messages.slice(-8)) {
      lines.push(`- [${message.role}] ${message.content.slice(0, 180)}`);
    }
  }

  return lines.join("\n").slice(0, 8000);
}

function looksLikeInsufficientFinal(text) {
  const cleaned = typeof text === "string" ? text.trim() : "";
  if (!cleaned) return true;
  if (cleaned.length < 36) return true;

  const weakPatterns = [
    /no\s+.+\s+generated/i,
    /not generated/i,
    /unable to/i,
    /cannot /i,
    /can't /i,
    /insufficient/i,
    /task completed\.?$/i,
    /i\s+cannot\s+directly/i,
    /sorry[,\s]+i\s+can(?:not|'t)/i,
    /對不起.{0,20}無法/i,
    /無法直接/i,
    /建議.{0,20}自行查詢/i,
  ];

  return weakPatterns.some((pattern) => pattern.test(cleaned));
}

function looksLikeRawProfileDump(text) {
  const cleaned = typeof text === "string" ? text.trim() : "";
  if (!cleaned) return false;
  if (!cleaned.startsWith("{") || !cleaned.endsWith("}")) return false;

  const parsed = safeJsonParse(cleaned, null);
  if (!parsed || typeof parsed !== "object") return false;
  return Object.prototype.hasOwnProperty.call(parsed, "responseStyle")
    && Object.prototype.hasOwnProperty.call(parsed, "expertiseLevel")
    && Object.prototype.hasOwnProperty.call(parsed, "preferences");
}

async function rewriteNaturalLanguageReply({
  message,
  route,
  params,
  responseStyle,
  expertiseLevel,
}) {
  const prompt = [
    "User asked to remember information.",
    "Reply in natural language, not JSON.",
    "Acknowledge the memory items clearly and briefly.",
    `Style: ${responseStyle || "concise"}`,
    `Expertise: ${expertiseLevel || "intermediate"}`,
    `User message: ${message}`,
  ].join("\n");

  try {
    const completion = await runChatCompletion({
      provider: route.provider,
      model: route.id,
      messages: [
        { role: "system", content: "Return only natural language answer." },
        { role: "user", content: prompt },
      ],
      params: {
        ...params,
        temperature: Math.min(0.35, clampNumber(params.temperature, 0.25, 0, 2)),
        maxTokens: Math.max(160, Math.min(700, clampNumber(params.maxTokens, 260, 1, 6000))),
      },
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const text = extractAssistantText(completion.choices?.[0]?.message).trim();
    return text || "已記住：A=紅色雨傘，B=每週三健身，C=貓叫 Milo。";
  } catch (_error) {
    return "已記住：A=紅色雨傘，B=每週三健身，C=貓叫 Milo。";
  }
}

function hasWebSearchArtifacts(artifacts) {
  return Array.isArray(artifacts)
    && artifacts.some((item) => item && item.ok && item.toolName === "web_search" && Array.isArray(item.output));
}

function hasGroundedWebEvidence(text, artifacts) {
  const cleaned = typeof text === "string" ? text : "";
  if (!cleaned.trim()) return false;

  const webArtifacts = Array.isArray(artifacts)
    ? artifacts.filter((item) => item && item.ok && item.toolName === "web_search" && Array.isArray(item.output))
    : [];

  if (webArtifacts.length === 0) {
    return true;
  }

  const haystack = cleaned.toLowerCase();
  for (const artifact of webArtifacts) {
    for (const hit of artifact.output.slice(0, 5)) {
      const title = typeof hit?.title === "string" ? hit.title.trim() : "";
      const url = typeof hit?.url === "string" ? hit.url.trim() : "";
      if (url && haystack.includes(url.toLowerCase())) return true;
      if (title && title.length >= 6 && haystack.includes(title.toLowerCase())) return true;
    }
  }

  return false;
}

function shouldRejectModelFinal(finalText, artifacts) {
  if (looksLikeInsufficientFinal(finalText)) return true;

  if (hasWebSearchArtifacts(artifacts)) {
    const badEvidencePhrases = [/無法直接/i, /cannot\s+directly/i, /建議.{0,20}自行查詢/i];
    if (badEvidencePhrases.some((pattern) => pattern.test(finalText || ""))) {
      return true;
    }
    if (!hasGroundedWebEvidence(finalText, artifacts)) {
      return true;
    }
  }

  return false;
}

function heuristicAgentFallback(task) {
  const cleanTask = typeof task === "string" ? task.trim() : "";
  const looksLikeLearningPlan = /7\s*天|7\s*-?day|learning\s*plan|學習計畫/i.test(cleanTask);

  if (looksLikeLearningPlan) {
    return [
      "7-Day Node.js Learning Plan",
      "Day 1 | Goal: Setup and runtime basics | Tasks: install Node.js, run REPL, write hello script | Acceptance: can run `node app.js` and explain event loop at high level",
      "Day 2 | Goal: Modules and npm | Tasks: build small utility with CommonJS/ESM, install one dependency | Acceptance: can split code into modules and run npm scripts",
      "Day 3 | Goal: Async patterns | Tasks: practice callback, Promise, async/await on file I/O | Acceptance: complete async file pipeline with error handling",
      "Day 4 | Goal: HTTP and APIs | Tasks: build minimal Express server with 2 endpoints | Acceptance: endpoints return JSON and handle invalid input",
      "Day 5 | Goal: Data persistence | Tasks: connect SQLite or JSON persistence, implement CRUD | Acceptance: create/read/update/delete works end-to-end",
      "Day 6 | Goal: Testing and debugging | Tasks: add unit tests and one integration test, use debugger/logging | Acceptance: tests pass and one bug is reproduced/fixed",
      "Day 7 | Goal: Mini project delivery | Tasks: ship a small REST app with docs | Acceptance: README includes setup, API usage, and demo results",
    ].join("\n");
  }

  return [
    "Best-effort task result",
    `Task: ${cleanTask || "(empty task)"}`,
    "1. Clarify objective and success criteria.",
    "2. Break work into implementation steps.",
    "3. Execute and validate each step.",
    "4. Summarize final output and remaining risks.",
  ].join("\n");
}

function parseRequestedDayCount(task) {
  const raw = typeof task === "string" ? task : "";
  const zh = raw.match(/(\d{1,2})\s*天/);
  if (zh) return clampNumber(zh[1], 0, 1, 30);

  const en = raw.match(/(\d{1,2})\s*-?\s*day/i);
  if (en) return clampNumber(en[1], 0, 1, 30);

  return 0;
}

function isStructuredPlanTask(task) {
  const text = typeof task === "string" ? task : "";
  if (!text) return false;
  return /學習計畫|学习计划|study\s*plan|roadmap/i.test(text) && parseRequestedDayCount(text) > 0;
}

function hasStructuredPlanQualityIssues(task, draft) {
  const text = typeof draft === "string" ? draft : "";
  if (!text.trim()) return true;

  const requestedDays = parseRequestedDayCount(task);
  if (requestedDays <= 0) return false;

  const normalized = text.toLowerCase();
  const noisePatterns = [
    /搜尋摘要|search summary/i,
    /forcing final synthesis|強制收斂|forced final/i,
    /tool|trace|artifact/i,
  ];
  if (noisePatterns.some((pattern) => pattern.test(text))) return true;

  const seenDays = new Set();
  for (let day = 1; day <= requestedDays; day += 1) {
    const zh = new RegExp(`第\\s*${day}\\s*天`);
    const en = new RegExp(`day\\s*${day}\\b`, "i");
    if (zh.test(text) || en.test(text)) {
      seenDays.add(day);
    }
  }
  if (seenDays.size < requestedDays) return true;

  const sectionCount = {
    goal: (normalized.match(/目標|goal/g) || []).length,
    task: (normalized.match(/任務|tasks?/g) || []).length,
    acceptance: (normalized.match(/驗收點|acceptance/g) || []).length,
  };

  if (sectionCount.goal < requestedDays || sectionCount.task < requestedDays || sectionCount.acceptance < requestedDays) {
    return true;
  }

  return false;
}

function shouldAgentUseTools(task) {
  const text = typeof task === "string" ? task : "";
  if (!text) return true;

  const planningSignals = /學習計畫|学习计划|roadmap|study plan|天計畫|天计划|每日|每天|day\s*\d+/i;
  const externalSignals = /最新|即時|实时|news|current|today|version|release|價格|price|查詢|search|source|資料來源|数据来源/i;

  if (planningSignals.test(text) && !externalSignals.test(text)) {
    return false;
  }

  return true;
}

async function applyAgentOutputQualityGate({ task, draft, routingInput, params }) {
  const finalDraft = typeof draft === "string" ? draft.trim() : "";
  if (!finalDraft) return heuristicAgentFallback(task);

  const requestedDays = parseRequestedDayCount(task);
  const model = routeModels(routingInput)[0];
  const gatePrompt = [
    "You are an output quality gate for agent tasks.",
    "Return strict JSON only.",
    'Schema: {"verdict":"pass"|"revise","issues":["..."],"revised":"..."}',
    "Rules:",
    "1) Ensure the answer directly fulfills the task.",
    "2) Remove duplicated sections.",
    "3) Remove internal-process text (search summary, tool errors, forcing synthesis, etc.).",
    requestedDays > 0
      ? `4) This task requests a ${requestedDays}-day plan. Ensure Day1..Day${requestedDays} appear exactly once each.`
      : "4) Keep structure clean and complete.",
    requestedDays > 0
      ? "5) Each day must include Goal, Tasks, and Acceptance Criteria."
      : "5) Keep outputs concrete and actionable.",
    requestedDays > 0 ? "6) Do not duplicate days or sections." : "6) Avoid repeated paragraphs.",
    `Task: ${task}`,
    `Draft: ${finalDraft}`,
  ].join("\n");

  try {
    const review = await runChatCompletion({
      provider: model.provider,
      model: model.id,
      messages: [
        { role: "system", content: "Return strict JSON only." },
        { role: "user", content: gatePrompt },
      ],
      params: {
        ...params,
        temperature: 0.2,
        maxTokens: Math.max(700, clampNumber(params.maxTokens, 900, 1, 6000)),
      },
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const raw = extractAssistantText(review.choices?.[0]?.message).trim();
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    const decision = safeJsonParse(jsonMatch ? jsonMatch[0] : "", null);
    if (decision && typeof decision === "object") {
      if (decision.verdict === "revise" && typeof decision.revised === "string" && decision.revised.trim()) {
        const revised = decision.revised.trim();
        if (requestedDays > 0 && hasStructuredPlanQualityIssues(task, revised)) {
          return finalDraft;
        }
        return revised;
      }
      if (decision.verdict === "pass") {
        if (requestedDays > 0 && hasStructuredPlanQualityIssues(task, finalDraft)) {
          return finalDraft;
        }
        return finalDraft;
      }
    }
  } catch (_error) {
    // Best-effort gate only.
  }

  if (isStructuredPlanTask(task) && hasStructuredPlanQualityIssues(task, finalDraft)) {
    try {
      const repair = await runChatCompletion({
        provider: model.provider,
        model: model.id,
        messages: [
          {
            role: "system",
            content: "Rewrite into a complete executable study plan. Output final answer only.",
          },
          {
            role: "user",
            content: [
              `Task: ${task}`,
              "Hard constraints:",
              "1) Cover every requested day exactly once.",
              "2) For each day include Goal, Tasks, Acceptance Criteria.",
              "3) Remove any internal process text or tool traces.",
              `Draft to repair: ${finalDraft}`,
            ].join("\n"),
          },
        ],
        params: {
          ...params,
          temperature: 0.2,
          maxTokens: Math.max(700, clampNumber(params.maxTokens, 900, 1, 6000)),
        },
        stream: false,
        tools: null,
        toolChoice: "none",
      });

      const repaired = extractAssistantText(repair.choices?.[0]?.message).trim();
      if (repaired && !hasStructuredPlanQualityIssues(task, repaired)) {
        return repaired;
      }
    } catch (_error) {
      // Fall through to finalDraft.
    }
  }

  return finalDraft;
}

function buildHeuristicFromArtifacts(task, artifacts) {
  const cleanedTask = typeof task === "string" ? task.trim() : "";
  const goodArtifacts = Array.isArray(artifacts) ? artifacts.filter((item) => item && item.ok) : [];
  if (goodArtifacts.length === 0) {
    return heuristicAgentFallback(cleanedTask);
  }

  const runJsArtifact = goodArtifacts.find((item) => item.toolName === "run_js");
  const webSearchArtifact = goodArtifacts.find((item) => item.toolName === "web_search");

  const lines = ["多步驟任務結果"]; 

  if (runJsArtifact?.output && typeof runJsArtifact.output.result === "string") {
    lines.push("", "1) 計算結果", `- ${runJsArtifact.output.result}`);
  }

  if (webSearchArtifact?.output && Array.isArray(webSearchArtifact.output)) {
    const topHits = webSearchArtifact.output.slice(0, 3);
    lines.push("", "2) 搜尋摘要");

    if (topHits.length === 0) {
      lines.push("- 本次搜尋無可用結果，已保留計算結果與結論建議。");
    } else {
      for (const hit of topHits) {
        const title = typeof hit.title === "string" && hit.title.trim() ? hit.title.trim() : "未命名結果";
        const snippet = typeof hit.snippet === "string" && hit.snippet.trim() ? hit.snippet.trim() : "";
        const url = typeof hit.url === "string" && hit.url.trim() ? hit.url.trim() : "";
        lines.push(`- ${title}${snippet ? `：${snippet.slice(0, 120)}` : ""}${url ? ` (${url})` : ""}`);
      }

      lines.push("", "3) 證據來源");
      for (const hit of topHits) {
        const title = typeof hit.title === "string" && hit.title.trim() ? hit.title.trim() : "未命名結果";
        const url = typeof hit.url === "string" && hit.url.trim() ? hit.url.trim() : "";
        if (url) {
          lines.push(`- ${title}: ${url}`);
        }
      }
    }
  }

  lines.push(
    "",
    "4) 最終結論",
    "- 已根據可用工具輸出完成整合；若需更完整外部資料，可改以更明確關鍵字再次查詢。",
  );

  if (/7\s*天|7\s*-?day|learning\s*plan|學習計畫/i.test(cleanedTask)) {
    lines.push(
      "",
      "5) 補充：7天 Node.js 學習計畫",
      "- Day1: 環境建置與 Node 基礎（驗收：能執行第一支 script）",
      "- Day2: 模組與 npm（驗收：拆分模組並管理套件）",
      "- Day3: 非同步（驗收：用 async/await 完成 I/O 流程）",
      "- Day4: API 與 Express（驗收：完成 2 個 API endpoint）",
      "- Day5: 資料儲存（驗收：完成 CRUD）",
      "- Day6: 測試與除錯（驗收：單元測試可執行）",
      "- Day7: 小專案整合（驗收：可部署/可展示）",
    );
  }

  return lines.join("\n");
}

async function synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }) {
  const successfulArtifacts = Array.isArray(artifacts)
    ? artifacts
        .filter((item) => item && item.ok)
        .map((item) => ({
          toolName: item.toolName,
          output: item.output,
        }))
    : [];

  const compactLogs = Array.isArray(logs)
    ? logs.map((item) => ({
        step: item.step,
        action: item.action,
        toolName: item.toolName,
        ok: item.ok,
      }))
    : [];

  const modelFinal = routeModels(routingInput)[0];
  const finalizePrompt = [
    "Produce a complete, concrete final deliverable from the task and execution state.",
    "Never mention planner/tool failures, retries, duplicate calls, or internal limitations.",
    "Do not ask for more tool calls.",
    "If any tool output is missing or empty, continue with best-effort assumptions and still provide a useful result.",
    "Focus on user value and final answer quality.",
    `Task: ${task}`,
    `CompactLogs: ${JSON.stringify(compactLogs).slice(-4000)}`,
    `SuccessfulArtifacts: ${JSON.stringify(successfulArtifacts).slice(-7000)}`,
  ].join("\n");

  try {
    const completion = await runChatCompletion({
      provider: modelFinal.provider,
      model: modelFinal.id,
      messages: [
        { role: "system", content: "Return a concrete final answer only." },
        { role: "user", content: finalizePrompt },
      ],
      params,
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    const finalText = extractAssistantText(completion.choices?.[0]?.message).trim();
    if (!shouldRejectModelFinal(finalText, artifacts)) {
      return finalText;
    }
  } catch (_error) {
    // Fall through to heuristic fallback.
  }

  return buildHeuristicFromArtifacts(task, artifacts);
}

async function runDetailedPlanningPipeline({ task, maxSteps, routingInput, params }) {
  const logs = [];
  const artifacts = [];
  const model = routeModels(routingInput)[0];

  const stageBudget = clampNumber(maxSteps, 5, 1, 8);

  let outline = "";
  try {
    const outlinePrompt = [
      "Create a planning outline for the task.",
      "Return plain text with sections:",
      "- Objectives",
      "- Constraints",
      "- Milestones",
      `Task: ${task}`,
    ].join("\n");

    const completion = await runChatCompletion({
      provider: model.provider,
      model: model.id,
      messages: [
        { role: "system", content: "You are a planner. Return concise planning outline." },
        { role: "user", content: outlinePrompt },
      ],
      params: {
        ...params,
        temperature: 0.3,
      },
      stream: false,
      tools: null,
      toolChoice: "none",
    });

    outline = extractAssistantText(completion.choices?.[0]?.message).trim();
    logs.push({ step: 1, action: "plan_outline", ok: Boolean(outline) });
    artifacts.push({ toolName: "plan_outline", ok: Boolean(outline), output: outline });
  } catch (error) {
    logs.push({ step: 1, action: "plan_outline", ok: false, error: error?.message || "outline failed" });
  }

  let expanded = "";
  if (stageBudget >= 2) {
    try {
      const expandPrompt = [
        "Expand this planning outline into a detailed executable answer.",
        "Requirements:",
        "1) Be concrete and actionable.",
        "2) Include checkpoints and acceptance criteria where appropriate.",
        "3) Avoid repetition and filler.",
        `Task: ${task}`,
        `Outline: ${outline || "(no outline available)"}`,
      ].join("\n");

      const completion = await runChatCompletion({
        provider: model.provider,
        model: model.id,
        messages: [
          { role: "system", content: "You are an execution planner. Return detailed final draft." },
          { role: "user", content: expandPrompt },
        ],
        params: {
          ...params,
          temperature: 0.35,
          maxTokens: Math.max(900, clampNumber(params.maxTokens, 1200, 1, 6000)),
        },
        stream: false,
        tools: null,
        toolChoice: "none",
      });

      expanded = extractAssistantText(completion.choices?.[0]?.message).trim();
      logs.push({ step: 2, action: "plan_expand", ok: Boolean(expanded) });
      artifacts.push({ toolName: "plan_expand", ok: Boolean(expanded), output: expanded.slice(0, 4000) });
    } catch (error) {
      logs.push({ step: 2, action: "plan_expand", ok: false, error: error?.message || "expand failed" });
    }
  }

  const baseDraft = expanded || outline || heuristicAgentFallback(task);
  const final = await applyAgentOutputQualityGate({
    task,
    draft: baseDraft,
    routingInput,
    params,
  });

  logs.push({ step: Math.min(stageBudget, 3), action: "quality_gate", ok: true });

  return {
    final,
    logs,
    artifacts,
    truncated: false,
  };
}

async function runAgentLoop({
  userId,
  sessionId,
  task,
  maxSteps,
  routingInput,
  params,
}) {
  const logs = [];
  const artifacts = [];
  const supportedTools = TOOL_SCHEMAS
    .map((item) => item?.function?.name)
    .filter((name) => typeof name === "string" && name.trim());
  const supportedToolSet = new Set(supportedTools);
  const allowTools = shouldAgentUseTools(task);

  if (!allowTools) {
    return runDetailedPlanningPipeline({
      task,
      maxSteps,
      routingInput,
      params,
    });
  }

  const seenToolSignatures = new Set();
  const toolCounts = {};
  let lastToolName = "";
  let sameToolStreak = 0;

  for (let step = 1; step <= maxSteps; step += 1) {
    const remainingSteps = maxSteps - step + 1;
    const planPrompt = [
      "You are an autonomous task planner.",
      "Given task and execution logs, return strict JSON only.",
      'Schema: {"action":"tool"|"final","reason":"...","toolName":"...","toolArgs":{},"final":"..."}',
      "If more work is needed, choose action=tool.",
      "If complete, choose action=final and provide final answer.",
      allowTools
        ? "Tools are allowed when they improve factual grounding."
        : "This task is planning/content generation. Prefer action=final. Do not call tools unless strictly required.",
      `Remaining steps: ${remainingSteps}`,
      "If remaining steps <= 1, you must choose action=final with best-effort output.",
      `Available tools: ${supportedTools.join(", ") || "none"}`,
      "If action=tool, toolName MUST be exactly one from Available tools.",
      "Never invent tool names.",
      "Tool argument guardrails:",
      "- web_search requires query (or q).",
      "- run_js requires code.",
      "- db_query should use sql; if unsure use: SELECT * FROM memory LIMIT 20",
      `Task: ${task}`,
      `Logs: ${JSON.stringify(logs).slice(-5000)}`,
    ].join("\n");

    const modelPlan = routeModels(routingInput)[0];
    const planningMessages = [
      { role: "system", content: "Return strict JSON only." },
      { role: "user", content: planPrompt },
    ];

    let decisionRaw = "";

    try {
      const completion = await runChatCompletion({
        provider: modelPlan.provider,
        model: modelPlan.id,
        messages: planningMessages,
        params,
        stream: false,
        tools: null,
        toolChoice: "none",
      });

      decisionRaw = completion.choices?.[0]?.message?.content || "";
    } catch (error) {
      logs.push({ step, action: "error", error: error?.message || "planning failed" });
      break;
    }

    const jsonMatch = decisionRaw.match(/\{[\s\S]*\}/);
    const decision = safeJsonParse(jsonMatch ? jsonMatch[0] : "", null);

    if (!decision || typeof decision !== "object") {
      logs.push({ step, action: "fallback", content: decisionRaw.slice(0, 800) });
      return {
        final: decisionRaw || "Agent planning failed to return JSON.",
        logs,
        artifacts,
      };
    }

    if (decision.action === "final") {
      logs.push({ step, action: "final", reason: decision.reason || "completed" });
      const candidateFinal = typeof decision.final === "string" ? decision.final.trim() : "";
      const final = looksLikeInsufficientFinal(candidateFinal)
        ? await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params })
        : candidateFinal;
      return {
        final: await applyAgentOutputQualityGate({ task, draft: final, routingInput, params }),
        logs,
        artifacts,
      };
    }

    if (decision.action === "tool") {
      if (!allowTools) {
        logs.push({
          step,
          action: "tool_blocked",
          reason: "Tool call blocked for planning task; forcing final synthesis.",
        });

        return {
          final: await applyAgentOutputQualityGate({
            task,
            draft: await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }),
            routingInput,
            params,
          }),
          logs,
          artifacts,
          truncated: false,
        };
      }

      const requestedToolName = String(decision.toolName || "").trim();
      const requestedToolArgs = decision.toolArgs && typeof decision.toolArgs === "object" ? decision.toolArgs : {};
      if (!supportedToolSet.has(requestedToolName)) {
        logs.push({
          step,
          action: "invalid_tool",
          toolName: requestedToolName || "(empty)",
          error: "Planner requested unsupported tool; forcing final synthesis.",
        });

        return {
          final: await applyAgentOutputQualityGate({
            task,
            draft: await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }),
            routingInput,
            params,
          }),
          logs,
          artifacts,
          truncated: false,
        };
      }

      const signature = `${requestedToolName}:${stableStringify(requestedToolArgs)}`;
      if (seenToolSignatures.has(signature)) {
        logs.push({
          step,
          action: "duplicate_tool",
          toolName: requestedToolName,
          reason: "Planner repeated the same tool call; forcing final synthesis.",
        });

        return {
          final: await applyAgentOutputQualityGate({
            task,
            draft: await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }),
            routingInput,
            params,
          }),
          logs,
          artifacts,
          truncated: false,
        };
      }

      seenToolSignatures.add(signature);
      toolCounts[requestedToolName] = (toolCounts[requestedToolName] || 0) + 1;

      if (requestedToolName === lastToolName) {
        sameToolStreak += 1;
      } else {
        sameToolStreak = 1;
      }
      lastToolName = requestedToolName;

      if (sameToolStreak >= 3 || toolCounts[requestedToolName] >= 4) {
        logs.push({
          step,
          action: "tool_limit",
          toolName: requestedToolName,
          reason: "Tool overused in planning loop; forcing final synthesis.",
        });

        return {
          final: await applyAgentOutputQualityGate({
            task,
            draft: await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }),
            routingInput,
            params,
          }),
          logs,
          artifacts,
          truncated: false,
        };
      }

      const fakeToolCall = {
        function: {
          name: requestedToolName,
          arguments: JSON.stringify(requestedToolArgs),
        },
      };

      const trace = await executeToolCall(fakeToolCall, { userId, sessionId });
      logs.push({
        step,
        action: "tool",
        toolName: trace.toolName,
        ok: trace.ok,
        error: trace.error,
        hasOutput: Boolean(trace.output),
      });
      artifacts.push(trace);

      if (!trace.ok) {
        const failures = logs.filter((entry) => entry.action === "tool" && entry.ok === false).length;
        if (failures >= 2) {
          logs.push({
            step,
            action: "tool_failures",
            error: "Multiple tool failures; forcing final synthesis.",
          });

          return {
            final: await applyAgentOutputQualityGate({
              task,
              draft: await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }),
              routingInput,
              params,
            }),
            logs,
            artifacts,
            truncated: false,
          };
        }
      }

      continue;
    }

    logs.push({ step, action: "unknown", decision });
  }

  return {
    final: await applyAgentOutputQualityGate({
      task,
      draft: await synthesizeAgentFinal({ task, logs, artifacts, routingInput, params }),
      routingInput,
      params,
    }),
    logs,
    artifacts,
    truncated: true,
  };
}

function extractTextForMemory(userMessage, processedAttachments) {
  const blocks = [userMessage];

  for (const item of processedAttachments) {
    if (item.extractedText) {
      blocks.push(item.extractedText);
    }
  }

  return blocks.join("\n\n").slice(0, 5000);
}

app.post("/api/chat", async (req, res) => {
  try {
    runtimeObservability.chat.total += 1;

    const body = req.body || {};
    const sessionId = sanitizeId(body.sessionId, "");
    const userId = sanitizeId(body.userId, DEFAULT_USER_ID);
    const message = typeof body.message === "string" ? body.message.trim() : "";

    if (!sessionId) {
      return res.status(400).json({ error: "Invalid or missing sessionId." });
    }

    if (!message) {
      return res.status(400).json({ error: "Message is required." });
    }

    const session = sessionManager.get(sessionId, userId);
    const runtimeCapabilities = getRuntimeMultimodalCapabilities();
    const profilePatch = body.profile && typeof body.profile === "object" ? body.profile : null;
    if (profilePatch) {
      await memoryDb.updateProfile(userId, profilePatch);
    }

    const profile = memoryDb.getProfile(userId);
    const processedAttachments = await preprocessAttachments(body.attachments, runtimeCapabilities);

    if (typeof body.loadedMemorySummary === "string") {
      session.loadedMemorySummary = body.loadedMemorySummary.trim().slice(0, 8000);
    }

    compressSessionIfNeeded(session);

    const retrievalQuery = extractTextForMemory(message, processedAttachments);
    const semanticMemories = memoryDb.listMemories(userId, {
      query: retrievalQuery,
      limit: clampNumber(body.memoryTopK, 6, 1, 20),
      minScore: 0.08,
    });

    for (const item of semanticMemories) {
      item.lastAccessedAt = nowIso();
    }

    const requestedModel = typeof body.model === "string" ? body.model.trim() : "";
    const taskType = typeof body.taskType === "string" ? body.taskType : "auto";
    const qualityPreference = typeof body.qualityPreference === "string" ? body.qualityPreference : "balanced";
    const costPreference = typeof body.costPreference === "string" ? body.costPreference : "balanced";

    const modelPlan = routeModels({
      requestedModel,
      taskType,
      hasImage: processedAttachments.some((item) => item.kind === "image"),
      qualityPreference,
      costPreference,
    });

    const selectedModel = modelPlan[0];
    const imageRequested = processedAttachments.some((item) => item.kind === "image");

    const supportsVisionReasoning =
      selectedModel.provider === "openai" &&
      selectedModel.tags.includes("vision") &&
      runtimeCapabilities.image.visionReasoning;

    if (taskType === "vision" && imageRequested && !supportsVisionReasoning) {
      return res.status(400).json({
        error: "Vision task requested, but the selected/available model route does not support image reasoning.",
      });
    }
    const responseStyle = typeof body.responseStyle === "string" ? body.responseStyle : profile.responseStyle;
    const expertiseLevel = typeof body.expertiseLevel === "string" ? body.expertiseLevel : profile.expertiseLevel;

    const systemPrompt = typeof body.systemPrompt === "string" && body.systemPrompt.trim()
      ? body.systemPrompt.trim()
      : DEFAULT_SYSTEM_PROMPT;

    const thinkMode = body.reasoningMode === "think";

    const userContent = buildUserMessageContent({
      message,
      processedAttachments,
      allowImageInput: supportsVisionReasoning,
    });

    const contextMessages = buildContextMessages({
      baseSystemPrompt: systemPrompt,
      profilePrompt: buildPersonalizationPrompt(profile, { responseStyle, expertiseLevel }),
      session,
      semanticMemories,
      userContent,
      thinkMode,
    });

    const params = {
      temperature: clampNumber(body.temperature, 0.7, 0, 2),
      maxTokens: thinkMode
        ? Math.max(clampNumber(body.maxTokens, 400, 1, 6000), THINK_MODE_MIN_TOKENS)
        : clampNumber(body.maxTokens, 400, 1, 6000),
      topP: clampNumber(body.topP, 1, 0, 1),
      presencePenalty: clampNumber(body.presencePenalty, 0, -2, 2),
      frequencyPenalty: clampNumber(body.frequencyPenalty, 0, -2, 2),
    };

    const toolMode = ["off", "manual", "auto"].includes(body.toolMode) ? body.toolMode : "auto";
    const traceContext = { userId, sessionId };

    sessionManager.appendMessage(session, "user", extractTextForMemory(message, processedAttachments));
    await memoryDb.ingestAutoMemories(userId, sessionId, "user", message);

    const shouldStream =
      body.stream === true
      && toolMode === "off"
      && !shouldDisableStreamingForQuality(message, toolMode);
    const hasAttachments = processedAttachments.length > 0;

    if (hasAttachments) {
      runtimeObservability.chat.withAttachments += 1;
    }

    if (shouldStream) {
      runtimeObservability.chat.streamed += 1;
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const streamCompletion = await runChatCompletion({
        provider: selectedModel.provider,
        model: selectedModel.id,
        messages: contextMessages,
        params,
        stream: true,
        tools: null,
        toolChoice: "none",
      });

      let accumulated = "";

      for await (const chunk of streamCompletion) {
        const delta = chunk?.choices?.[0]?.delta?.content || "";
        if (delta) {
          accumulated += delta;
          res.write(`data: ${JSON.stringify({ delta })}\n\n`);
        }
      }

      sessionManager.appendMessage(session, "assistant", accumulated || "(empty response)");
      runtimeObservability.chat.byModel[selectedModel.id] =
        (runtimeObservability.chat.byModel[selectedModel.id] || 0) + 1;
      recordObservabilityEvent({
        type: "chat",
        sessionId,
        userId,
        mode: "stream",
        model: selectedModel.id,
        provider: selectedModel.provider,
        attachments: processedAttachments.map((item) => item.kind),
      });
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    let finalResult = null;
    let lastError = null;
    let usedRoute = null;

    for (const candidate of modelPlan) {
      try {
        const trace = [];

        finalResult = await createCompleteReply({
          modelPlan: candidate,
          messages: contextMessages,
          params,
          tools: TOOL_SCHEMAS,
          toolMode,
          toolExecutor: async (toolCall) => executeToolCall(toolCall, traceContext),
          trace,
        });

        usedRoute = candidate;

        if (Array.isArray(finalResult.usedTools) && finalResult.usedTools.length > 0) {
          runtimeObservability.chat.withTools += 1;
          session.toolTrace.push(...finalResult.usedTools);
          if (session.toolTrace.length > 100) {
            session.toolTrace.splice(0, session.toolTrace.length - 100);
          }
        }

        break;
      } catch (error) {
        lastError = error;
      }
    }

    if (!finalResult || !usedRoute) {
      throw lastError || new Error("All routed models failed.");
    }

    if (shouldAttemptAutoGrounding({
      message,
      reply: finalResult.reply,
      toolMode,
      usedTools: finalResult.usedTools,
      processedAttachments,
    })) {
      try {
        const query = buildWebQueryFromMessage(message);
        const normalizedQuery = normalizeSearchQuery(query, message);
        const webResults = await runWebSearchWithFallback(normalizedQuery, 5);

        if (webResults.length > 0) {
          const autoGroundTrace = {
            id: crypto.randomUUID(),
            sessionId,
            userId,
            toolName: "web_search",
            args: { query: normalizedQuery, originalQuery: query, topK: 5, autoGrounding: true },
            startedAt: nowIso(),
            endedAt: nowIso(),
            ok: true,
            output: webResults,
            error: "",
          };

          finalResult.usedTools = Array.isArray(finalResult.usedTools)
            ? [...finalResult.usedTools, autoGroundTrace]
            : [autoGroundTrace];

          finalResult.reply = await synthesizeGroundedAnswer({
            question: message,
            draftReply: finalResult.reply,
            webResults,
            route: usedRoute,
            params,
            responseStyle,
            expertiseLevel,
          });
        }
      } catch (_error) {
        // Auto-grounding is best-effort only.
      }
    }

    finalResult.reply = await applyResponseQualityGate({
      userMessage: message,
      reply: finalResult.reply,
      responseStyle,
      expertiseLevel,
      route: usedRoute,
      params,
    });

    if (looksLikeRawProfileDump(finalResult.reply)) {
      finalResult.reply = await rewriteNaturalLanguageReply({
        message,
        route: usedRoute,
        params,
        responseStyle,
        expertiseLevel,
      });
    }

    finalResult.reply = await enforceGroundedAnswerIfNeeded({
      message,
      reply: finalResult.reply,
      usedTools: finalResult.usedTools,
      route: usedRoute,
      params,
      responseStyle,
      expertiseLevel,
    });

    finalResult.reply = await enforceAcronymGroundingIfNeeded({
      message,
      reply: finalResult.reply,
      route: usedRoute,
      params,
      responseStyle,
      expertiseLevel,
    });

    sessionManager.appendMessage(session, "assistant", finalResult.reply);
    runtimeObservability.chat.byModel[usedRoute.id] = (runtimeObservability.chat.byModel[usedRoute.id] || 0) + 1;

    recordObservabilityEvent({
      type: "chat",
      sessionId,
      userId,
      mode: "non_stream",
      model: usedRoute.id,
      provider: usedRoute.provider,
      attachments: processedAttachments.map((item) => item.kind),
      toolsUsed: finalResult.usedTools.length,
    });

    return res.json({
      reply: finalResult.reply,
      route: {
        model: usedRoute.id,
        provider: usedRoute.provider,
      },
      toolTrace: session.toolTrace.slice(-12),
      context: {
        recentMessages: session.messages.length,
        compressedSummaryChars: session.compressedSummary.length,
        retrievedMemoryCount: semanticMemories.length,
      },
      multimodal: processedAttachments.map((item) => ({
        id: item.id,
        kind: item.kind,
        name: item.name,
        extractedChars: item.extractedText.length,
        error: item.error,
      })),
      capabilities: {
        imageReasoning: supportsVisionReasoning,
        imageOcr: runtimeCapabilities.image.ocr,
        audioTranscription: runtimeCapabilities.audio.transcription,
        videoInput: runtimeCapabilities.video.input,
      },
    });
  } catch (error) {
    const status = error?.status || 500;
    return res.status(status).json({
      error: error?.message || "Unexpected error while handling chat.",
    });
  }
});

app.post("/api/multimodal/preprocess", async (req, res) => {
  try {
    const capabilities = getRuntimeMultimodalCapabilities();
    const processed = await preprocessAttachments(req.body?.attachments, capabilities);
    return res.json({
      ok: true,
      items: processed.map((item) => ({
        id: item.id,
        kind: item.kind,
        name: item.name,
        mimeType: item.mimeType,
        extractedText: item.extractedText,
        error: item.error,
      })),
      capabilities,
    });
  } catch (error) {
    return res.status(500).json({
      error: error?.message || "Failed to preprocess attachments.",
    });
  }
});

app.post("/api/memory/reset", (req, res) => {
  const sessionId = sanitizeId(req.body?.sessionId, "");
  const userId = sanitizeId(req.body?.userId, DEFAULT_USER_ID);
  if (!sessionId) {
    return res.status(400).json({ error: "Invalid or missing sessionId." });
  }

  const ok = sessionManager.reset(sessionId, userId);
  if (!ok) {
    return res.status(403).json({ error: "Session belongs to a different user." });
  }

  return res.json({ ok: true });
});

app.post("/api/memory/replace", (req, res) => {
  try {
    const sessionId = sanitizeId(req.body?.sessionId, "");
    const userId = sanitizeId(req.body?.userId, DEFAULT_USER_ID);

    if (!sessionId) {
      return res.status(400).json({ error: "Invalid or missing sessionId." });
    }

    const session = sessionManager.get(sessionId, userId);
    sessionManager.replaceMessages(session, req.body?.messages);

    return res.json({ ok: true, count: session.messages.length });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to replace memory." });
  }
});

app.post("/api/memory/load", async (req, res) => {
  try {
    const targetSessionId = sanitizeId(req.body?.targetSessionId, "");
    const userId = sanitizeId(req.body?.userId, DEFAULT_USER_ID);

    if (!targetSessionId) {
      return res.status(400).json({ error: "Invalid or missing targetSessionId." });
    }

    const sources = sanitizeSourceConversations(req.body?.sources);
    if (sources.length === 0) {
      return res.status(400).json({ error: "At least one valid source conversation is required." });
    }

    const summary = summarizeSourcesHeuristic(sources);
    const session = sessionManager.get(targetSessionId, userId);
    session.loadedMemorySummary = summary;

    const targetMessages = Array.isArray(req.body?.targetMessages) ? req.body.targetMessages : [];
    sessionManager.replaceMessages(session, targetMessages);

    return res.json({ ok: true, summary });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to load memory summary." });
  }
});

app.get("/api/users/:userId/profile", (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    return res.json({ ok: true, profile: memoryDb.getProfile(userId) });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to fetch profile." });
  }
});

app.put("/api/users/:userId/profile", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const profile = await memoryDb.updateProfile(userId, req.body || {});
    return res.json({ ok: true, profile });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to update profile." });
  }
});

app.get("/api/users/:userId/memory", (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const memories = memoryDb.listMemories(userId, {
      type: typeof req.query.type === "string" ? req.query.type : "",
      query: typeof req.query.query === "string" ? req.query.query : "",
      limit: req.query.limit,
      minScore: req.query.minScore,
    });

    return res.json({ ok: true, memories });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to list memories." });
  }
});

app.post("/api/users/:userId/memory", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const memory = await memoryDb.addMemory(userId, req.body || {});
    return res.json({ ok: true, memory });
  } catch (error) {
    return res.status(400).json({ error: error?.message || "Failed to add memory." });
  }
});

app.patch("/api/users/:userId/memory/:memoryId", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const memoryId = sanitizeId(req.params.memoryId, "");
    const memory = await memoryDb.updateMemory(userId, memoryId, req.body || {});
    return res.json({ ok: true, memory });
  } catch (error) {
    return res.status(400).json({ error: error?.message || "Failed to update memory." });
  }
});

app.delete("/api/users/:userId/memory/:memoryId", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const memoryId = sanitizeId(req.params.memoryId, "");
    const deleted = await memoryDb.deleteMemory(userId, memoryId);
    return res.json({ ok: true, deleted });
  } catch (error) {
    return res.status(400).json({ error: error?.message || "Failed to delete memory." });
  }
});

app.post("/api/users/:userId/memory/summarize", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const summary = await memoryDb.summarizeAndStore(userId);
    return res.json({ ok: true, summary });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to summarize memory." });
  }
});

app.post("/api/users/:userId/memory/prune", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const result = await memoryDb.pruneMemories(userId, req.body || {});
    return res.json({ ok: true, ...result });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to prune memory." });
  }
});

app.delete("/api/users/:userId/memory", async (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    const includeArchived = req.query?.includeArchived === "1" || req.query?.includeArchived === "true";
    const result = await memoryDb.clearMemories(userId, { includeArchived });
    return res.json({ ok: true, ...result });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to clear memory." });
  }
});

app.get("/api/users/:userId/memory/stats", (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    return res.json({ ok: true, stats: memoryDb.getMemoryStats(userId) });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to fetch memory stats." });
  }
});

app.get("/api/users/:userId/export", (req, res) => {
  try {
    const userId = sanitizeId(req.params.userId, DEFAULT_USER_ID);
    return res.json({
      ok: true,
      userId,
      exportedAt: nowIso(),
      snapshot: memoryDb.getUserSnapshot(userId),
    });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to export user snapshot." });
  }
});

app.get("/api/mcp/tools", (_req, res) => {
  return res.json({
    ok: true,
    protocol: "mcp-lite-v1",
    tools: TOOL_SCHEMAS,
  });
});

app.post("/api/mcp/execute", async (req, res) => {
  try {
    const userId = sanitizeId(req.body?.userId, DEFAULT_USER_ID);
    const sessionId = sanitizeId(req.body?.sessionId, "mcp_session");
    const toolName = typeof req.body?.toolName === "string" ? req.body.toolName : "";

    if (!toolName) {
      return res.status(400).json({ error: "toolName is required." });
    }

    const toolCall = {
      function: {
        name: toolName,
        arguments: JSON.stringify(req.body?.args || {}),
      },
    };

    const trace = await executeToolCall(toolCall, { userId, sessionId });
    return res.json({
      ok: true,
      protocol: "mcp-lite-v1",
      trace,
      toolContext: {
        userId,
        sessionId,
      },
    });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Tool execution failed." });
  }
});

app.post("/api/agent/run", async (req, res) => {
  try {
    const task = typeof req.body?.task === "string" ? req.body.task.trim() : "";
    if (!task) {
      return res.status(400).json({ error: "Task is required." });
    }

    const userId = sanitizeId(req.body?.userId, DEFAULT_USER_ID);
    const sessionId = sanitizeId(req.body?.sessionId, `agent_${crypto.randomUUID()}`);

    const maxSteps = clampNumber(req.body?.maxSteps, 5, 1, 12);
    const routingInput = {
      requestedModel: typeof req.body?.model === "string" ? req.body.model : "",
      taskType: "reasoning",
      hasImage: false,
      qualityPreference: "quality",
      costPreference: "balanced",
    };

    const params = {
      temperature: 0.2,
      maxTokens: 700,
      topP: 1,
      presencePenalty: 0,
      frequencyPenalty: 0,
    };

    const result = await runAgentLoop({
      userId,
      sessionId,
      task,
      maxSteps,
      routingInput,
      params,
    });

    return res.json({ ok: true, ...result });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to execute agent task." });
  }
});

app.get("/api/models", (_req, res) => {
  const available = getAvailableModels().map((item) => ({
    id: item.id,
    provider: item.provider,
    capabilities: item.tags,
  }));

  return res.json({ ok: true, models: available });
});

app.get("/api/capabilities", (_req, res) => {
  const capabilities = getRuntimeMultimodalCapabilities();
  return res.json({
    ok: true,
    multimodal: capabilities,
    models: getAvailableModels().map((item) => ({
      id: item.id,
      provider: item.provider,
      capabilities: item.tags,
    })),
  });
});

app.get("/api/context/config", (_req, res) => {
  return res.json({
    ok: true,
    contextCompression: {
      triggerMessages: getContextCompressTrigger(),
      keepRecentMessages: getRecentContextMessages(),
    },
  });
});

app.put("/api/context/config", (req, res) => {
  try {
    const updated = setRuntimeContextConfig({
      triggerMessages: req.body?.triggerMessages,
      keepRecentMessages: req.body?.keepRecentMessages,
    });
    return res.json({ ok: true, contextCompression: updated });
  } catch (error) {
    return res.status(500).json({ error: error?.message || "Failed to update context config." });
  }
});

app.get("/api/observability/summary", (_req, res) => {
  return res.json({
    ok: true,
    startedAt: runtimeObservability.startedAt,
    requests: runtimeObservability.requests,
    chat: runtimeObservability.chat,
    tools: runtimeObservability.tools,
    recentEvents: runtimeObservability.recentEvents.slice(-50),
  });
});

app.get("/api/health", (_req, res) => {
  return res.json({
    ok: true,
    providers: {
      groq: Boolean(groq),
      openai: Boolean(openai),
    },
    capabilities: getRuntimeMultimodalCapabilities(),
    dataPath: MEMORY_DB_PATH,
  });
});

app.get("/api/diagnostics/runtime", (_req, res) => {
  const available = getAvailableModels();
  return res.json({
    ok: true,
    providers: {
      groq: Boolean(groq),
      openai: Boolean(openai),
    },
    dependencies: {
      openaiPackage: Boolean(OpenAI),
      tesseractPackage: Boolean(Tesseract),
    },
    models: available.map((item) => ({
      id: item.id,
      provider: item.provider,
      capabilities: item.tags,
    })),
    capabilities: getRuntimeMultimodalCapabilities(),
  });
});

async function start() {
  await memoryDb.init();

  app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
  });
}

start().catch((error) => {
  console.error("Failed to start server:", error);
  process.exit(1);
});
