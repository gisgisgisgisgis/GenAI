const modelEl = document.getElementById("model");
const systemPromptEl = document.getElementById("systemPrompt");
const temperatureEl = document.getElementById("temperature");
const maxTokensEl = document.getElementById("maxTokens");
const topPEl = document.getElementById("topP");
const presencePenaltyEl = document.getElementById("presencePenalty");
const frequencyPenaltyEl = document.getElementById("frequencyPenalty");
const streamModeEl = document.getElementById("streamMode");
const responseModeEl = document.getElementById("responseMode");
const compressTriggerEl = document.getElementById("compressTrigger");
const keepRecentMessagesEl = document.getElementById("keepRecentMessages");
const applyContextConfigEl = document.getElementById("applyContextConfig");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chatForm");
const messageInputEl = document.getElementById("messageInput");
const statusEl = document.getElementById("status");
const sendBtnEl = document.getElementById("sendBtn");
const runAgentBtnEl = document.getElementById("runAgentBtn");
const resetMemoryEl = document.getElementById("resetMemory");
const conversationListEl = document.getElementById("conversationList");
const newChatBtnEl = document.getElementById("newChatBtn");
const activeConversationTitleEl = document.getElementById("activeConversationTitle");
const appLayoutEl = document.getElementById("appLayout");
const toggleSettingsEl = document.getElementById("toggleSettings");
const loadMemoryModalEl = document.getElementById("loadMemoryModal");
const loadMemoryListEl = document.getElementById("loadMemoryList");
const cancelLoadMemoryEl = document.getElementById("cancelLoadMemory");
const confirmLoadMemoryEl = document.getElementById("confirmLoadMemory");
const renameConversationModalEl = document.getElementById("renameConversationModal");
const renameConversationInputEl = document.getElementById("renameConversationInput");
const cancelRenameConversationEl = document.getElementById("cancelRenameConversation");
const confirmRenameConversationEl = document.getElementById("confirmRenameConversation");
const userIdEl = document.getElementById("userId");
const responseStyleEl = document.getElementById("responseStyle");
const expertiseLevelEl = document.getElementById("expertiseLevel");
const taskTypeEl = document.getElementById("taskType");
const toolModeEl = document.getElementById("toolMode");
const qualityPreferenceEl = document.getElementById("qualityPreference");
const costPreferenceEl = document.getElementById("costPreference");
const attachmentInputEl = document.getElementById("attachmentInput");
const attachmentListEl = document.getElementById("attachmentList");
const multimodalHintEl = document.getElementById("multimodalHint");
const capabilityBadgesEl = document.getElementById("capabilityBadges");
const openMemoryManagerEl = document.getElementById("openMemoryManager");
const memoryManagerModalEl = document.getElementById("memoryManagerModal");
const memoryManagerListEl = document.getElementById("memoryManagerList");
const addMemoryModalEl = document.getElementById("addMemoryModal");
const openAddMemoryModalEl = document.getElementById("openAddMemoryModal");
const cancelAddMemoryModalEl = document.getElementById("cancelAddMemoryModal");
const closeMemoryManagerEl = document.getElementById("closeMemoryManager");
const refreshMemoryListEl = document.getElementById("refreshMemoryList");
const summarizeMemoryBtnEl = document.getElementById("summarizeMemoryBtn");
const pruneMemoryBtnEl = document.getElementById("pruneMemoryBtn");
const exportMemoryBtnEl = document.getElementById("exportMemoryBtn");
const clearMemoryBtnEl = document.getElementById("clearMemoryBtn");
const memoryTypeInputEl = document.getElementById("memoryTypeInput");
const memoryContentInputEl = document.getElementById("memoryContentInput");
const addMemoryItemEl = document.getElementById("addMemoryItem");

const STORAGE_CONVERSATIONS_KEY = "custom_groq_conversations";
const STORAGE_ACTIVE_ID_KEY = "custom_groq_active_conversation_id";
const STORAGE_SETTINGS_OPEN_KEY = "custom_groq_settings_open";
const LEGACY_SESSION_KEY = "custom_gpt_session_id";
const DEFAULT_TITLE = "New Chat";
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant tailored to the user's needs.";
const FALLBACK_MODELS = [
  { id: "llama-3.1-8b-instant", provider: "groq", capabilities: ["chat", "coding", "tools"] },
  { id: "llama-3.3-70b-versatile", provider: "groq", capabilities: ["chat", "reasoning", "coding", "tools"] },
  { id: "mixtral-8x7b-32768", provider: "groq", capabilities: ["chat", "coding"] },
  { id: "gpt-4.1-mini", provider: "openai", capabilities: ["chat", "reasoning", "coding", "vision", "tools"] },
  { id: "gpt-4.1", provider: "openai", capabilities: ["chat", "reasoning", "coding", "vision", "tools"] },
];

const defaultSettings = {
  model: "",
  userId: "default_user",
  responseStyle: "concise",
  expertiseLevel: "intermediate",
  taskType: "auto",
  toolMode: "auto",
  qualityPreference: "balanced",
  costPreference: "balanced",
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  temperature: 0.7,
  maxTokens: 400,
  topP: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  stream: true,
  reasoningMode: "fast",
};

const settingsInputs = [
  modelEl,
  userIdEl,
  responseStyleEl,
  expertiseLevelEl,
  taskTypeEl,
  toolModeEl,
  qualityPreferenceEl,
  costPreferenceEl,
  systemPromptEl,
  temperatureEl,
  maxTokensEl,
  topPEl,
  presencePenaltyEl,
  frequencyPenaltyEl,
  streamModeEl,
  responseModeEl,
];

const state = {
  conversations: [],
  activeConversationId: "",
  settingsOpen: true,
  isBusy: false,
  openConversationMenuId: "",
  loadMemoryTargetId: "",
  renameTargetId: "",
  pendingAttachments: [],
  memoryManagerItems: [],
  availableModels: [],
  runtimeCapabilities: {
    text: { input: true, reasoning: true },
    image: { input: false, visionReasoning: false, ocr: false },
    audio: { input: false, transcription: false },
    video: { input: false, transcription: false, reasoning: false },
  },
};

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1"]);
const runtimeConfig =
  window.APP_CONFIG && typeof window.APP_CONFIG === "object" ? window.APP_CONFIG : {};
const configuredApiBase =
  typeof runtimeConfig.API_BASE_URL === "string"
    ? runtimeConfig.API_BASE_URL.trim().replace(/\/+$/, "")
    : "";
const API_BASE_URL =
  LOCAL_HOSTS.has(window.location.hostname) ? window.location.origin : configuredApiBase;

function apiPath(path) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function apiFetch(path, options) {
  return fetch(apiPath(path), options);
}

function getDefaultModelId() {
  if (state.availableModels.length > 0) {
    return state.availableModels[0].id;
  }
  return FALLBACK_MODELS[0].id;
}

function mergeCapabilities(capabilities) {
  return {
    text: {
      input: true,
      reasoning: true,
      ...(capabilities?.text || {}),
    },
    image: {
      input: false,
      visionReasoning: false,
      ocr: false,
      ...(capabilities?.image || {}),
    },
    audio: {
      input: false,
      transcription: false,
      ...(capabilities?.audio || {}),
    },
    video: {
      input: false,
      transcription: false,
      reasoning: false,
      ...(capabilities?.video || {}),
    },
  };
}

function setModelOptions(models) {
  const pool = Array.isArray(models) && models.length > 0 ? models : FALLBACK_MODELS;
  state.availableModels = pool.map((item) => ({
    id: item.id,
    provider: item.provider,
    capabilities: Array.isArray(item.capabilities) ? item.capabilities : [],
  }));

  const activeConversation = getActiveConversation();
  const previousValue = modelEl.value || activeConversation?.settings?.model || getDefaultModelId();

  modelEl.innerHTML = "";
  for (const model of state.availableModels) {
    const option = document.createElement("option");
    option.value = model.id;
    const suffix = model.capabilities.includes("vision") ? " · vision" : "";
    option.textContent = `${model.id} (${model.provider})${suffix}`;
    modelEl.appendChild(option);
  }

  const hasPrevious = state.availableModels.some((model) => model.id === previousValue);
  modelEl.value = hasPrevious ? previousValue : getDefaultModelId();
}

function renderCapabilityBadges() {
  const capabilities = state.runtimeCapabilities;
  const badges = [
    { label: "Text", enabled: capabilities.text.input },
    {
      label: capabilities.image.visionReasoning
        ? "Image (Vision + OCR)"
        : capabilities.image.ocr
          ? "Image (OCR)"
          : "Image",
      enabled: capabilities.image.input,
    },
    {
      label: capabilities.audio.transcription ? "Audio (STT)" : "Audio",
      enabled: capabilities.audio.input,
    },
    {
      label: "Video",
      enabled: capabilities.video.input,
    },
  ];

  capabilityBadgesEl.innerHTML = "";
  for (const badgeInfo of badges) {
    const badge = document.createElement("span");
    badge.className = `cap-badge ${badgeInfo.enabled ? "on" : "off"}`;
    badge.textContent = badgeInfo.enabled ? `${badgeInfo.label} on` : `${badgeInfo.label} off`;
    capabilityBadgesEl.appendChild(badge);
  }

  const supported = ["text"];
  if (capabilities.image.input) {
    supported.push(capabilities.image.visionReasoning ? "image (vision)" : "image (ocr-only)");
  }
  if (capabilities.audio.input) {
    supported.push("audio (speech-to-text)");
  }
  if (capabilities.video.input) {
    supported.push("video");
  }

  multimodalHintEl.textContent = `Supported now: ${supported.join(", ")}.`;

  const accept = [];
  if (capabilities.image.input) accept.push("image/*");
  if (capabilities.audio.input) accept.push("audio/*");
  if (capabilities.video.input) accept.push("video/*");
  attachmentInputEl.accept = accept.length > 0 ? accept.join(",") : "";
}

function applyRuntimeConfig(payload) {
  const capabilities = mergeCapabilities(payload?.multimodal);
  const models = Array.isArray(payload?.models) ? payload.models : FALLBACK_MODELS;

  state.runtimeCapabilities = capabilities;
  setModelOptions(models);

  for (const conversation of state.conversations) {
    if (!conversation?.settings) continue;
    const isValid = state.availableModels.some((model) => model.id === conversation.settings.model);
    if (!isValid) {
      conversation.settings.model = getDefaultModelId();
    }
  }

  saveState();
  renderCapabilityBadges();
}

async function fetchServerModels() {
  try {
    const response = await apiFetch("/api/models", { method: "GET" });
    const data = await response.json();
    if (!response.ok || !data.ok || !Array.isArray(data.models)) {
      throw new Error("Invalid model list response");
    }

    return data.models;
  } catch (_error) {
    return FALLBACK_MODELS;
  }
}

async function fetchRuntimeCapabilities() {
  try {
    const [capabilitiesResponse, serverModels] = await Promise.all([
      apiFetch("/api/capabilities", { method: "GET" }),
      fetchServerModels(),
    ]);

    const data = await capabilitiesResponse.json();
    if (!capabilitiesResponse.ok || !data.ok) {
      throw new Error(data.error || "Failed to load capabilities");
    }

    applyRuntimeConfig({
      multimodal: data.multimodal,
      models: Array.isArray(data.models) && data.models.length > 0 ? data.models : serverModels,
    });
  } catch (_error) {
    applyRuntimeConfig({
      multimodal: state.runtimeCapabilities,
      models: FALLBACK_MODELS,
    });
  }
}

function numberOr(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function sanitizeMessage(message) {
  if (!message || typeof message.content !== "string") return null;
  if (!["user", "assistant", "system"].includes(message.role)) return null;
  return { role: message.role, content: message.content };
}

function normalizeSettings(input) {
  const defaultModelId = getDefaultModelId();
  const reasoningMode = input?.reasoningMode === "think" ? "think" : "fast";

  const responseStyle = ["concise", "technical", "casual"].includes(input?.responseStyle)
    ? input.responseStyle
    : defaultSettings.responseStyle;

  const expertiseLevel = ["beginner", "intermediate", "expert"].includes(input?.expertiseLevel)
    ? input.expertiseLevel
    : defaultSettings.expertiseLevel;

  const taskType = ["auto", "chat", "reasoning", "coding", "vision"].includes(input?.taskType)
    ? input.taskType
    : defaultSettings.taskType;

  const toolMode = ["auto", "off", "manual"].includes(input?.toolMode)
    ? input.toolMode
    : defaultSettings.toolMode;

  const qualityPreference = ["balanced", "quality", "fast"].includes(input?.qualityPreference)
    ? input.qualityPreference
    : defaultSettings.qualityPreference;

  const costPreference = ["balanced", "low", "high"].includes(input?.costPreference)
    ? input.costPreference
    : defaultSettings.costPreference;

  return {
    model:
      typeof input?.model === "string" && input.model.trim()
        ? input.model.trim()
        : defaultModelId || defaultSettings.model,
    userId: typeof input?.userId === "string" && input.userId.trim()
      ? input.userId.trim().slice(0, 80)
      : defaultSettings.userId,
    responseStyle,
    expertiseLevel,
    taskType,
    toolMode,
    qualityPreference,
    costPreference,
    systemPrompt:
      typeof input?.systemPrompt === "string" && input.systemPrompt.trim()
        ? input.systemPrompt
        : defaultSettings.systemPrompt,
    temperature: numberOr(input?.temperature, defaultSettings.temperature),
    maxTokens: numberOr(input?.maxTokens, defaultSettings.maxTokens),
    topP: numberOr(input?.topP, defaultSettings.topP),
    presencePenalty: numberOr(input?.presencePenalty, defaultSettings.presencePenalty),
    frequencyPenalty: numberOr(input?.frequencyPenalty, defaultSettings.frequencyPenalty),
    stream: typeof input?.stream === "boolean" ? input.stream : defaultSettings.stream,
    reasoningMode,
  };
}

function createConversation(sessionId = crypto.randomUUID()) {
  return {
    id: crypto.randomUUID(),
    sessionId,
    title: DEFAULT_TITLE,
    messages: [],
    loadedMemorySummary: "",
    settings: {
      ...defaultSettings,
      model: getDefaultModelId() || defaultSettings.model,
    },
    updatedAt: Date.now(),
  };
}

function normalizeConversation(item) {
  if (!item || typeof item !== "object") return null;
  if (typeof item.id !== "string" || !item.id) return null;

  const cleanedMessages = Array.isArray(item.messages)
    ? item.messages.map(sanitizeMessage).filter(Boolean)
    : [];

  return {
    id: item.id,
    sessionId: typeof item.sessionId === "string" && item.sessionId ? item.sessionId : crypto.randomUUID(),
    title: typeof item.title === "string" && item.title.trim() ? item.title : DEFAULT_TITLE,
    messages: cleanedMessages,
    loadedMemorySummary:
      typeof item.loadedMemorySummary === "string" ? item.loadedMemorySummary.slice(0, 8000) : "",
    settings: normalizeSettings(item.settings),
    updatedAt: numberOr(item.updatedAt, Date.now()),
  };
}

function loadState() {
  let conversations = [];
  try {
    const raw = localStorage.getItem(STORAGE_CONVERSATIONS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) {
      conversations = parsed.map(normalizeConversation).filter(Boolean);
    }
  } catch (_error) {
    conversations = [];
  }

  if (conversations.length === 0) {
    const legacySessionId = localStorage.getItem(LEGACY_SESSION_KEY);
    const initial = createConversation(legacySessionId || crypto.randomUUID());
    conversations.push(initial);
  }

  const storedActiveId = localStorage.getItem(STORAGE_ACTIVE_ID_KEY);
  const matched = conversations.find((conversation) => conversation.id === storedActiveId);
  const settingsOpenStored = localStorage.getItem(STORAGE_SETTINGS_OPEN_KEY);

  state.conversations = conversations;
  state.activeConversationId = matched ? matched.id : conversations[0].id;
  state.settingsOpen = settingsOpenStored !== "0";
  state.openConversationMenuId = "";
  state.loadMemoryTargetId = "";
  state.renameTargetId = "";
  state.pendingAttachments = [];
  state.memoryManagerItems = [];
}

function saveState() {
  localStorage.setItem(STORAGE_CONVERSATIONS_KEY, JSON.stringify(state.conversations));
  localStorage.setItem(STORAGE_ACTIVE_ID_KEY, state.activeConversationId);
  localStorage.setItem(STORAGE_SETTINGS_OPEN_KEY, state.settingsOpen ? "1" : "0");

  const activeConversation = getActiveConversation();
  if (activeConversation) {
    localStorage.setItem(LEGACY_SESSION_KEY, activeConversation.sessionId);
  }
}

function getActiveConversation() {
  return state.conversations.find((conversation) => conversation.id === state.activeConversationId) || null;
}

function getConversationById(conversationId) {
  return state.conversations.find((conversation) => conversation.id === conversationId) || null;
}

function cloneMessages(messages) {
  return messages.map((message) => ({ role: message.role, content: message.content }));
}

function setStatus(text) {
  statusEl.textContent = text;
}

function findPreviousUserIndex(messages, fromIndex) {
  for (let index = fromIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      return index;
    }
  }
  return -1;
}

function moveConversationToTop(conversationId) {
  const index = state.conversations.findIndex((conversation) => conversation.id === conversationId);
  if (index <= 0) return;
  const [picked] = state.conversations.splice(index, 1);
  state.conversations.unshift(picked);
}

function deriveTitleFromMessage(message) {
  return message.slice(0, 36).trim() || DEFAULT_TITLE;
}

function formatUpdatedAt(timestamp) {
  return new Date(timestamp).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function closeLoadMemoryModal() {
  state.loadMemoryTargetId = "";
  loadMemoryModalEl.classList.remove("open");
  loadMemoryModalEl.setAttribute("aria-hidden", "true");
  loadMemoryListEl.innerHTML = "";
}

function closeRenameConversationModal() {
  state.renameTargetId = "";
  renameConversationInputEl.value = "";
  renameConversationModalEl.classList.remove("open");
  renameConversationModalEl.setAttribute("aria-hidden", "true");
}

function openRenameConversationModal(conversationId) {
  const targetConversation = getConversationById(conversationId);
  if (!targetConversation || state.isBusy) return;

  state.renameTargetId = conversationId;
  renameConversationInputEl.value = (targetConversation.title || DEFAULT_TITLE).trim() || DEFAULT_TITLE;
  renameConversationModalEl.classList.add("open");
  renameConversationModalEl.setAttribute("aria-hidden", "false");

  window.setTimeout(() => {
    renameConversationInputEl.focus();
    renameConversationInputEl.select();
  }, 0);
}

function openLoadMemoryModal(targetConversationId) {
  const targetConversation = getConversationById(targetConversationId);
  if (!targetConversation) return;

  const candidates = state.conversations.filter((conversation) => conversation.id !== targetConversationId);
  if (candidates.length === 0) {
    addMessage(targetConversation, "system", "No other conversations available to load.", true);
    return;
  }

  state.loadMemoryTargetId = targetConversationId;
  loadMemoryListEl.innerHTML = "";

  for (const conversation of candidates) {
    const row = document.createElement("label");
    row.className = "load-memory-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = conversation.id;

    const textWrap = document.createElement("div");

    const title = document.createElement("div");
    title.className = "load-memory-title";
    title.textContent = conversation.title;

    const meta = document.createElement("div");
    meta.className = "load-memory-meta";
    meta.textContent = `${formatUpdatedAt(conversation.updatedAt)} · ${conversation.messages.length} messages`;

    textWrap.appendChild(title);
    textWrap.appendChild(meta);
    row.appendChild(checkbox);
    row.appendChild(textWrap);
    loadMemoryListEl.appendChild(row);
  }

  loadMemoryModalEl.classList.add("open");
  loadMemoryModalEl.setAttribute("aria-hidden", "false");
}

function escapeHtml(raw) {
  return String(raw)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAssistantMarkdown(content) {
  const markdown = typeof content === "string" ? content : "";

  if (window.marked?.parse) {
    const parsed = window.marked.parse(markdown, {
      gfm: true,
      breaks: true,
      headerIds: false,
      mangle: false,
    });

    if (window.DOMPurify?.sanitize) {
      return window.DOMPurify.sanitize(parsed);
    }

    return parsed;
  }

  return escapeHtml(markdown).replaceAll("\n", "<br>");
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const ok = document.execCommand("copy");
  document.body.removeChild(textarea);
  if (!ok) {
    throw new Error("Clipboard unavailable");
  }
}

function attachCodeFeatures(bubble) {
  const blocks = bubble.querySelectorAll("pre > code");

  for (const codeBlock of blocks) {
    if (window.hljs?.highlightElement) {
      window.hljs.highlightElement(codeBlock);
    }

    const pre = codeBlock.parentElement;
    if (!pre) continue;

    pre.classList.add("code-block");

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "code-copy-btn";
    copyButton.textContent = "Copy";

    copyButton.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();

      const originalText = copyButton.textContent;
      const codeText = codeBlock.textContent || "";
      if (!codeText) return;

      try {
        await copyText(codeText);
        copyButton.textContent = "Copied";
      } catch (_error) {
        copyButton.textContent = "Failed";
      }

      window.setTimeout(() => {
        copyButton.textContent = originalText;
      }, 1200);
    });

    pre.appendChild(copyButton);
  }
}

function setBubbleContent(bubble, role, content) {
  if (role === "system") {
    const parsedTrace = parseTraceMessage(content);
    if (parsedTrace) {
      bubble.textContent = "";
      renderCollapsibleTrace(bubble, parsedTrace);
      return;
    }
  }

  if (role === "assistant" && !bubble.classList.contains("thinking")) {
    bubble.innerHTML = `<div class="md-content">${renderAssistantMarkdown(content)}</div>`;
    attachCodeFeatures(bubble);
    for (const link of bubble.querySelectorAll("a")) {
      link.target = "_blank";
      link.rel = "noopener noreferrer nofollow";
    }
    return;
  }

  bubble.textContent = content;
}

function appendMessageActions(bubble, options) {
  if (!options?.canRethink && !options?.canBranch) return;

  const actions = document.createElement("div");
  actions.className = "bubble-actions";

  if (options.canRethink) {
    const rethinkButton = document.createElement("button");
    rethinkButton.type = "button";
    rethinkButton.className = "bubble-action-btn";
    rethinkButton.textContent = "Rethink";
    rethinkButton.disabled = state.isBusy;
    rethinkButton.addEventListener("click", () => {
      rethinkFromAssistant(options.messageIndex);
    });
    actions.appendChild(rethinkButton);
  }

  if (options.canBranch) {
    const branchButton = document.createElement("button");
    branchButton.type = "button";
    branchButton.className = "bubble-action-btn";
    branchButton.textContent = "Branch Here";
    branchButton.disabled = state.isBusy;
    branchButton.addEventListener("click", () => {
      branchConversationFromMessage(options.messageIndex);
    });
    actions.appendChild(branchButton);
  }

  bubble.appendChild(actions);
}

function buildChatTraceLines(data) {
  const lines = [];
  if (data?.route?.model) {
    lines.push(`Model route: ${data.route.model} (${data.route.provider || "unknown"})`);
  }

  if (data?.context) {
    lines.push(
      `Context: recent=${data.context.recentMessages ?? 0}, compressedChars=${data.context.compressedSummaryChars ?? 0}, retrievedMemory=${data.context.retrievedMemoryCount ?? 0}`,
    );
  }

  if (Array.isArray(data?.toolTrace) && data.toolTrace.length > 0) {
    const lastTools = data.toolTrace.slice(-3).map((item) => {
      const toolName = item.toolName || "unknown";
      return `${toolName}:${item.ok ? "ok" : "fail"}`;
    });
    lines.push(`Tools: ${lastTools.join(", ")}`);
  } else {
    lines.push("Tools: none");
  }

  return lines;
}

function parseTraceMessage(content) {
  if (typeof content !== "string") return null;
  const lines = content.split("\n");
  const titleLine = lines[0] || "";
  const titleMatch = titleLine.match(/^\[(.+)\]$/);
  if (!titleMatch) return null;

  const title = titleMatch[1].trim();
  if (!/trace/i.test(title)) return null;

  const items = lines
    .slice(1)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- "))
    .map((line) => line.slice(2).trim())
    .filter(Boolean);

  if (items.length === 0) return null;
  return { title, items };
}

function classifyTraceLine(line) {
  const text = String(line || "").toLowerCase();
  if (/\berror=|\bfail\b/.test(text)) return "error";
  if (/duplicate_tool|tool_limit|tool_failures|forcing final synthesis|reached step limit: yes/.test(text)) {
    return "warn";
  }
  if (/\bok\b/.test(text)) return "ok";
  return "info";
}

function renderCollapsibleTrace(container, traceMessage) {
  const details = document.createElement("details");
  details.className = "trace-panel";

  const summary = document.createElement("summary");
  summary.className = "trace-title";
  summary.textContent = `${traceMessage.title} (${traceMessage.items.length})`;
  details.appendChild(summary);

  for (const item of traceMessage.items) {
    const row = document.createElement("div");
    row.className = `trace-line ${classifyTraceLine(item)}`;
    row.textContent = item;
    details.appendChild(row);
  }

  container.innerHTML = "";
  container.appendChild(details);
}

function pushTraceMessage(conversation, title, lines) {
  if (!conversation) return;
  if (!Array.isArray(lines) || lines.length === 0) return;

  const content = [`[${title}]`, ...lines.map((line) => `- ${line}`)].join("\n");
  addMessage(conversation, "system", content, true);
}

function appendBubble(role, content, options = null) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  setBubbleContent(bubble, role, content);
  if (options?.showActions) {
    appendMessageActions(bubble, options);
  }
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function appendThinkingBubble() {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant thinking";
  bubble.innerHTML =
    '<span class="thinking-shell"><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-text">Thinking</span></span>';
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function renderMessages() {
  const activeConversation = getActiveConversation();
  messagesEl.innerHTML = "";

  if (!activeConversation || activeConversation.messages.length === 0) {
    appendBubble("system", "Start a new chat.");
    return;
  }

  for (let index = 0; index < activeConversation.messages.length; index += 1) {
    const message = activeConversation.messages[index];
    const previousUserIndex = findPreviousUserIndex(activeConversation.messages, index);
    const canRethink = message.role === "assistant" && previousUserIndex >= 0;
    const canBranch = message.role === "assistant";

    appendBubble(message.role, message.content, {
      showActions: canRethink || canBranch,
      canRethink,
      canBranch,
      messageIndex: index,
    });
  }
}

function renderConversationList() {
  conversationListEl.innerHTML = "";

  for (const conversation of state.conversations) {
    const isActive = conversation.id === state.activeConversationId;
    const isMenuOpen = conversation.id === state.openConversationMenuId;

    const item = document.createElement("div");
    item.className = "conversation-item";
    if (isActive) {
      item.classList.add("active");
    }

    const main = document.createElement("button");
    main.type = "button";
    main.className = "conversation-main";
    main.disabled = state.isBusy;
    main.addEventListener("click", () => {
      if (conversation.id === state.activeConversationId) {
        if (state.openConversationMenuId) {
          state.openConversationMenuId = "";
          renderConversationList();
        }
        return;
      }
      state.activeConversationId = conversation.id;
      state.openConversationMenuId = "";
      saveState();
      renderAll();
    });

    const title = document.createElement("div");
    title.className = "conversation-title";
    title.textContent = conversation.title;

    const meta = document.createElement("div");
    meta.className = "conversation-meta";
    meta.textContent = formatUpdatedAt(conversation.updatedAt);

    main.appendChild(title);
    main.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "conversation-actions";

    const menuTrigger = document.createElement("button");
    menuTrigger.type = "button";
    menuTrigger.className = "menu-trigger";
    menuTrigger.textContent = "⋯";
    menuTrigger.disabled = state.isBusy;
    menuTrigger.setAttribute("aria-label", "Conversation options");
    menuTrigger.setAttribute("aria-haspopup", "menu");
    menuTrigger.setAttribute("aria-expanded", isMenuOpen ? "true" : "false");
    menuTrigger.addEventListener("click", (event) => {
      event.stopPropagation();
      state.openConversationMenuId = isMenuOpen ? "" : conversation.id;
      renderConversationList();
    });

    const menu = document.createElement("div");
    menu.className = `conversation-menu${isMenuOpen ? " open" : ""}`;

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "conversation-menu-item ghost";
    renameButton.textContent = "Rename Conversation";
    renameButton.disabled = state.isBusy;
    renameButton.addEventListener("click", (event) => {
      event.stopPropagation();
      state.openConversationMenuId = "";
      renderConversationList();
      openRenameConversationModal(conversation.id);
    });

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "conversation-menu-item ghost";
    copyButton.textContent = "Duplicate Conversation";
    copyButton.disabled = state.isBusy;
    copyButton.addEventListener("click", (event) => {
      event.stopPropagation();
      copyConversation(conversation.id);
    });

    const loadButton = document.createElement("button");
    loadButton.type = "button";
    loadButton.className = "conversation-menu-item ghost";
    loadButton.textContent = "Load Conversation Memory";
    loadButton.disabled = state.isBusy;
    loadButton.addEventListener("click", (event) => {
      event.stopPropagation();
      state.openConversationMenuId = "";
      renderConversationList();
      openLoadMemoryModal(conversation.id);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "conversation-menu-item danger";
    deleteButton.textContent = "Delete Conversation";
    deleteButton.disabled = state.isBusy;
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });

    menu.appendChild(renameButton);
    menu.appendChild(copyButton);
    menu.appendChild(loadButton);
    menu.appendChild(deleteButton);
    actions.appendChild(menuTrigger);
    actions.appendChild(menu);

    item.appendChild(main);
    item.appendChild(actions);
    conversationListEl.appendChild(item);
  }
}

function readSettingsFromInputs() {
  return {
    model: modelEl.value || getDefaultModelId(),
    userId: (userIdEl.value || defaultSettings.userId).trim().slice(0, 80) || defaultSettings.userId,
    responseStyle: responseStyleEl.value,
    expertiseLevel: expertiseLevelEl.value,
    taskType: taskTypeEl.value,
    toolMode: toolModeEl.value,
    qualityPreference: qualityPreferenceEl.value,
    costPreference: costPreferenceEl.value,
    systemPrompt: systemPromptEl.value.trim() || defaultSettings.systemPrompt,
    temperature: numberOr(temperatureEl.value, defaultSettings.temperature),
    maxTokens: numberOr(maxTokensEl.value, defaultSettings.maxTokens),
    topP: numberOr(topPEl.value, defaultSettings.topP),
    presencePenalty: numberOr(presencePenaltyEl.value, defaultSettings.presencePenalty),
    frequencyPenalty: numberOr(frequencyPenaltyEl.value, defaultSettings.frequencyPenalty),
    stream: streamModeEl.checked,
    reasoningMode: responseModeEl.value === "think" ? "think" : "fast",
  };
}

function applySettingsToInputs(settings) {
  const nextModel = settings.model || getDefaultModelId();
  const exists = Array.from(modelEl.options).some((option) => option.value === nextModel);
  modelEl.value = exists ? nextModel : getDefaultModelId();
  userIdEl.value = settings.userId || defaultSettings.userId;
  responseStyleEl.value = settings.responseStyle;
  expertiseLevelEl.value = settings.expertiseLevel;
  taskTypeEl.value = settings.taskType;
  toolModeEl.value = settings.toolMode;
  qualityPreferenceEl.value = settings.qualityPreference;
  costPreferenceEl.value = settings.costPreference;
  systemPromptEl.value = settings.systemPrompt;
  temperatureEl.value = String(settings.temperature);
  maxTokensEl.value = String(settings.maxTokens);
  topPEl.value = String(settings.topP);
  presencePenaltyEl.value = String(settings.presencePenalty);
  frequencyPenaltyEl.value = String(settings.frequencyPenalty);
  streamModeEl.checked = settings.stream;
  responseModeEl.value = settings.reasoningMode;
}

function syncActiveSettingsFromInputs() {
  const activeConversation = getActiveConversation();
  if (!activeConversation || state.isBusy) return;

  activeConversation.settings = readSettingsFromInputs();
  activeConversation.updatedAt = Date.now();
  saveState();
  renderConversationList();
}

function addMessage(conversation, role, content, rerender = true) {
  conversation.messages.push({ role, content });
  conversation.updatedAt = Date.now();

  if (role === "user" && (!conversation.title || conversation.title === DEFAULT_TITLE)) {
    conversation.title = deriveTitleFromMessage(content);
  }

  moveConversationToTop(conversation.id);
  state.activeConversationId = conversation.id;
  state.openConversationMenuId = "";
  saveState();

  if (rerender) {
    renderAll();
  }
}

async function replaceServerMemory(sessionId, userId, messages) {
  const response = await apiFetch("/api/memory/replace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, userId, messages }),
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to replace memory");
  }
}

async function copyConversation(conversationId) {
  if (state.isBusy) return;

  const source = getConversationById(conversationId);
  if (!source) return;

  const copiedConversation = {
    id: crypto.randomUUID(),
    sessionId: crypto.randomUUID(),
    title: `${source.title} (Copy)`,
    messages: cloneMessages(source.messages),
    loadedMemorySummary: source.loadedMemorySummary,
    settings: { ...source.settings },
    updatedAt: Date.now(),
  };

  state.conversations.unshift(copiedConversation);
  state.activeConversationId = copiedConversation.id;
  state.openConversationMenuId = "";
  saveState();
  renderAll();

  setBusy(true);
  setStatus("Copying...");

  try {
    await replaceServerMemory(copiedConversation.sessionId, copiedConversation.settings.userId, copiedConversation.messages);
    setStatus("Ready");
  } catch (error) {
    addMessage(copiedConversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
  }
}

function renameConversation(conversationId, nextTitleValue) {
  if (state.isBusy) return;

  const target = getConversationById(conversationId);
  if (!target) return;

  target.title = String(nextTitleValue || "").trim() || DEFAULT_TITLE;
  target.updatedAt = Date.now();
  state.openConversationMenuId = "";
  saveState();
  renderAll();
}

function confirmRenameConversation() {
  if (state.isBusy) return;

  const targetConversationId = state.renameTargetId;
  if (!targetConversationId) {
    closeRenameConversationModal();
    return;
  }

  renameConversation(targetConversationId, renameConversationInputEl.value);
  closeRenameConversationModal();
}

async function loadSelectedConversationMemory() {
  if (state.isBusy) return;

  const targetConversation = getConversationById(state.loadMemoryTargetId);
  if (!targetConversation) {
    closeLoadMemoryModal();
    return;
  }

  const selectedIds = Array.from(
    loadMemoryListEl.querySelectorAll('input[type="checkbox"]:checked'),
    (input) => input.value,
  );
  if (selectedIds.length === 0) {
    return;
  }

  const selectedConversations = state.conversations
    .filter((conversation) => selectedIds.includes(conversation.id))
    .map((conversation) => ({
      id: conversation.id,
      title: conversation.title,
      messages: cloneMessages(conversation.messages),
    }));

  closeLoadMemoryModal();
  setBusy(true);
  setStatus("Loading memory...");

  try {
    const response = await apiFetch("/api/memory/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        targetSessionId: targetConversation.sessionId,
        userId: targetConversation.settings.userId,
        targetMessages: cloneMessages(targetConversation.messages),
        sources: selectedConversations,
        model: targetConversation.settings.model,
      }),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Failed to load memory");
    }

    targetConversation.loadedMemorySummary = typeof data.summary === "string" ? data.summary : "";
    targetConversation.updatedAt = Date.now();
    saveState();

    addMessage(
      targetConversation,
      "system",
      `Loaded compressed memory from ${selectedConversations.length} conversation(s).`,
      true,
    );
    setStatus("Ready");
  } catch (error) {
    addMessage(targetConversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
  }
}

function buildPayloadForConversation(conversation, userMessage) {
  const attachmentPayload = state.pendingAttachments.map((item) => ({
    id: item.id,
    name: item.name,
    kind: item.kind,
    mimeType: item.mimeType,
    dataBase64: item.dataBase64,
  }));

  return {
    sessionId: conversation.sessionId,
    userId: conversation.settings.userId,
    message: userMessage,
    model: conversation.settings.model,
    taskType: conversation.settings.taskType,
    toolMode: conversation.settings.toolMode,
    qualityPreference: conversation.settings.qualityPreference,
    costPreference: conversation.settings.costPreference,
    responseStyle: conversation.settings.responseStyle,
    expertiseLevel: conversation.settings.expertiseLevel,
    profile: {
      responseStyle: conversation.settings.responseStyle,
      expertiseLevel: conversation.settings.expertiseLevel,
    },
    systemPrompt: conversation.settings.systemPrompt,
    stream:
      conversation.settings.stream &&
      conversation.settings.reasoningMode !== "think" &&
      conversation.settings.toolMode === "off" &&
      attachmentPayload.length === 0,
    temperature: conversation.settings.temperature,
    maxTokens: conversation.settings.maxTokens,
    topP: conversation.settings.topP,
    presencePenalty: conversation.settings.presencePenalty,
    frequencyPenalty: conversation.settings.frequencyPenalty,
    reasoningMode: conversation.settings.reasoningMode,
    loadedMemorySummary: conversation.loadedMemorySummary,
    attachments: attachmentPayload,
  };
}

async function sendConversationMessage(conversation, userMessage, statusOverride = "") {
  const payload = buildPayloadForConversation(conversation, userMessage);
  const pendingBubble = payload.reasoningMode === "think" ? appendThinkingBubble() : null;

  setStatus(statusOverride || (payload.reasoningMode === "think" ? "Thinking (Reasoning)..." : "Thinking..."));

  if (payload.stream) {
    await sendStreaming(payload, conversation, pendingBubble);
    return;
  }

  await sendNonStreaming(payload, conversation);
}

async function branchConversationFromMessage(messageIndex) {
  if (state.isBusy) return;

  const activeConversation = getActiveConversation();
  if (!activeConversation) return;
  if (messageIndex < 0 || messageIndex >= activeConversation.messages.length) return;

  const branchedConversation = {
    id: crypto.randomUUID(),
    sessionId: crypto.randomUUID(),
    title: `${activeConversation.title} (Branch)`,
    messages: cloneMessages(activeConversation.messages.slice(0, messageIndex + 1)),
    loadedMemorySummary: activeConversation.loadedMemorySummary,
    settings: { ...activeConversation.settings },
    updatedAt: Date.now(),
  };

  state.conversations.unshift(branchedConversation);
  state.activeConversationId = branchedConversation.id;
  state.openConversationMenuId = "";
  saveState();
  renderAll();

  setBusy(true);
  setStatus("Creating branch...");

  try {
    await replaceServerMemory(branchedConversation.sessionId, branchedConversation.settings.userId, branchedConversation.messages);
    setStatus("Ready");
  } catch (error) {
    addMessage(branchedConversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
  }
}

async function rethinkFromAssistant(messageIndex) {
  if (state.isBusy) return;

  const activeConversation = getActiveConversation();
  if (!activeConversation) return;
  if (activeConversation.messages[messageIndex]?.role !== "assistant") return;

  const userIndex = findPreviousUserIndex(activeConversation.messages, messageIndex);
  if (userIndex < 0) return;

  const userMessage = activeConversation.messages[userIndex].content;
  const preservedHistory = cloneMessages(activeConversation.messages.slice(0, userIndex));

  activeConversation.messages = preservedHistory;
  activeConversation.updatedAt = Date.now();
  saveState();
  renderAll();

  setBusy(true);
  setStatus("Rethinking...");

  try {
    await replaceServerMemory(activeConversation.sessionId, activeConversation.settings.userId, activeConversation.messages);
    addMessage(activeConversation, "user", userMessage, true);
    await sendConversationMessage(activeConversation, userMessage, "Rethinking...");
    setStatus("Ready");
  } catch (error) {
    addMessage(activeConversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
  }
}

function deleteConversation(conversationId) {
  if (state.isBusy) return;

  const index = state.conversations.findIndex((conversation) => conversation.id === conversationId);
  if (index === -1) return;

  const [removedConversation] = state.conversations.splice(index, 1);

  if (state.conversations.length === 0) {
    const replacement = createConversation();
    state.conversations.push(replacement);
    state.activeConversationId = replacement.id;
  } else if (state.activeConversationId === conversationId) {
    state.activeConversationId = state.conversations[0].id;
  }

  state.openConversationMenuId = "";
  saveState();
  renderAll();

  apiFetch("/api/memory/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId: removedConversation.sessionId, userId: removedConversation.settings.userId }),
  }).catch(() => {
    // Ignore memory cleanup errors because local deletion already succeeded.
  });
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  if (isBusy) {
    state.openConversationMenuId = "";
    if (loadMemoryModalEl.classList.contains("open")) {
      closeLoadMemoryModal();
    }
    if (renameConversationModalEl.classList.contains("open")) {
      closeRenameConversationModal();
    }
    if (memoryManagerModalEl.classList.contains("open")) {
      closeMemoryManagerModal();
    }
  }

  sendBtnEl.disabled = isBusy;
  runAgentBtnEl.disabled = isBusy;
  messageInputEl.disabled = isBusy;
  newChatBtnEl.disabled = isBusy;
  resetMemoryEl.disabled = isBusy;
  cancelLoadMemoryEl.disabled = isBusy;
  confirmLoadMemoryEl.disabled = isBusy;
  cancelRenameConversationEl.disabled = isBusy;
  confirmRenameConversationEl.disabled = isBusy;
  renameConversationInputEl.disabled = isBusy;
  attachmentInputEl.disabled = isBusy;
  openMemoryManagerEl.disabled = isBusy;
  openAddMemoryModalEl.disabled = isBusy;
  cancelAddMemoryModalEl.disabled = isBusy;
  applyContextConfigEl.disabled = isBusy;
  closeMemoryManagerEl.disabled = isBusy;
  refreshMemoryListEl.disabled = isBusy;
  summarizeMemoryBtnEl.disabled = isBusy;
  pruneMemoryBtnEl.disabled = isBusy;
  exportMemoryBtnEl.disabled = isBusy;
  clearMemoryBtnEl.disabled = isBusy;
  addMemoryItemEl.disabled = isBusy;
  memoryContentInputEl.disabled = isBusy;
  memoryTypeInputEl.disabled = isBusy;

  for (const input of settingsInputs) {
    input.disabled = isBusy;
  }

  renderConversationList();
  renderMessages();
}

async function fetchContextConfig() {
  try {
    const response = await apiFetch("/api/context/config", { method: "GET" });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Failed to load context config");
    }

    compressTriggerEl.value = data.contextCompression?.triggerMessages || 22;
    keepRecentMessagesEl.value = data.contextCompression?.keepRecentMessages || 12;
  } catch (_error) {
    compressTriggerEl.value = compressTriggerEl.value || 22;
    keepRecentMessagesEl.value = keepRecentMessagesEl.value || 12;
  }
}

async function applyContextConfig() {
  const triggerMessages = Number(compressTriggerEl.value);
  const keepRecentMessages = Number(keepRecentMessagesEl.value);

  const response = await apiFetch("/api/context/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ triggerMessages, keepRecentMessages }),
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to update context compression");
  }

  compressTriggerEl.value = data.contextCompression?.triggerMessages || triggerMessages;
  keepRecentMessagesEl.value = data.contextCompression?.keepRecentMessages || keepRecentMessages;
  addMessage(
    getActiveConversation(),
    "system",
    `Context compression updated: trigger=${compressTriggerEl.value}, keepRecent=${keepRecentMessagesEl.value}`,
    true,
  );
}

function renderAll() {
  const activeConversation = getActiveConversation();
  if (!activeConversation) return;

  activeConversationTitleEl.textContent = activeConversation.title;
  applySettingsToInputs(activeConversation.settings);
  renderConversationList();
  renderMessages();
}

function setSettingsOpen(isOpen) {
  state.settingsOpen = isOpen;
  appLayoutEl.classList.toggle("settings-collapsed", !isOpen);
  toggleSettingsEl.textContent = isOpen ? "❯" : "❮";
  toggleSettingsEl.setAttribute("aria-expanded", isOpen ? "true" : "false");
  toggleSettingsEl.setAttribute("aria-label", isOpen ? "Collapse settings" : "Expand settings");
  toggleSettingsEl.title = isOpen ? "Collapse settings" : "Expand settings";
  saveState();
}

async function sendNonStreaming(payload, conversation) {
  const response = await apiFetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }

  addMessage(conversation, "assistant", data.reply, true);

  const traceLines = buildChatTraceLines(data);
  if (traceLines.length > 0) {
    pushTraceMessage(conversation, "Execution Trace", traceLines);
  }
}

async function sendStreaming(payload, conversation, pendingBubble) {
  const response = await apiFetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed");
    }
    addMessage(conversation, "assistant", data.reply || "(empty response)", true);

    const traceLines = buildChatTraceLines(data);
    if (traceLines.length > 0) {
      pushTraceMessage(conversation, "Execution Trace", traceLines);
    }
    return;
  }

  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "Streaming request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const assistantMessage = { role: "assistant", content: "" };
  conversation.messages.push(assistantMessage);
  conversation.updatedAt = Date.now();
  moveConversationToTop(conversation.id);
  saveState();
  renderConversationList();

  const assistantBubble = pendingBubble || appendBubble("assistant", "");
  let hasOutput = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      if (!event.startsWith("data: ")) continue;
      const payloadText = event.slice(6).trim();
      if (payloadText === "[DONE]") {
        if (!hasOutput) {
          assistantMessage.content = "(empty response)";
          assistantBubble.classList.remove("thinking");
          setBubbleContent(assistantBubble, "assistant", "(empty response)");
        }
        conversation.updatedAt = Date.now();
        saveState();
        renderConversationList();
        return;
      }

      try {
        const parsed = JSON.parse(payloadText);
        if (parsed.delta) {
          if (!hasOutput) {
            assistantBubble.classList.remove("thinking");
          }
          hasOutput = true;
          assistantMessage.content += parsed.delta;
          setBubbleContent(assistantBubble, "assistant", assistantMessage.content);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      } catch (_error) {
        // Ignore malformed chunks and continue stream handling.
      }
    }
  }

  if (!hasOutput) {
    assistantMessage.content = "(empty response)";
    assistantBubble.classList.remove("thinking");
    setBubbleContent(assistantBubble, "assistant", "(empty response)");
  }
  conversation.updatedAt = Date.now();
  saveState();
  renderConversationList();
}

function inferAttachmentKind(mimeType) {
  if (typeof mimeType !== "string") return "";
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("audio/")) return "audio";
  if (mimeType.startsWith("video/")) return "video";
  return "";
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const value = typeof reader.result === "string" ? reader.result : "";
      const idx = value.indexOf(",");
      resolve(idx >= 0 ? value.slice(idx + 1) : "");
    };
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsDataURL(file);
  });
}

function renderAttachmentList() {
  attachmentListEl.innerHTML = "";
  if (state.pendingAttachments.length === 0) {
    return;
  }

  for (const item of state.pendingAttachments) {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";

    const label = document.createElement("span");
    label.textContent = `${item.name} (${item.kind})`;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "×";
    removeBtn.disabled = state.isBusy;
    removeBtn.setAttribute("aria-label", `Remove ${item.name}`);
    removeBtn.addEventListener("click", () => {
      if (state.isBusy) return;
      state.pendingAttachments = state.pendingAttachments.filter((entry) => entry.id !== item.id);
      renderAttachmentList();
    });

    chip.appendChild(label);
    chip.appendChild(removeBtn);
    attachmentListEl.appendChild(chip);
  }
}

function clearPendingAttachments() {
  state.pendingAttachments = [];
  attachmentInputEl.value = "";
  renderAttachmentList();
}

async function handleAttachmentInputChange() {
  if (!attachmentInputEl.files || attachmentInputEl.files.length === 0) {
    return;
  }

  const files = Array.from(attachmentInputEl.files).slice(0, 5);

  for (const file of files) {
    const kind = inferAttachmentKind(file.type);
    if (!kind) continue;

    const caps = state.runtimeCapabilities;
    const kindAllowed =
      (kind === "image" && caps.image.input) ||
      (kind === "audio" && caps.audio.input) ||
      (kind === "video" && caps.video.input);

    if (!kindAllowed) {
      addMessage(
        getActiveConversation(),
        "system",
        `Skipped ${file.name}: ${kind} is not enabled by current runtime/model capabilities.`,
        true,
      );
      continue;
    }

    if (file.size > 8 * 1024 * 1024) {
      addMessage(getActiveConversation(), "system", `Skipped ${file.name}: file too large (>8MB).`, true);
      continue;
    }

    const dataBase64 = await fileToBase64(file);
    if (!dataBase64) continue;

    state.pendingAttachments.push({
      id: crypto.randomUUID(),
      name: file.name,
      kind,
      mimeType: file.type || "application/octet-stream",
      dataBase64,
      size: file.size,
    });
  }

  attachmentInputEl.value = "";
  renderAttachmentList();
}

function getActiveUserId() {
  const activeConversation = getActiveConversation();
  if (!activeConversation) return defaultSettings.userId;
  return activeConversation.settings.userId || defaultSettings.userId;
}

function closeMemoryManagerModal() {
  memoryManagerModalEl.classList.remove("open");
  memoryManagerModalEl.setAttribute("aria-hidden", "true");
}

function openAddMemoryModal() {
  addMemoryModalEl.classList.add("open");
  addMemoryModalEl.setAttribute("aria-hidden", "false");
  setTimeout(() => {
    memoryContentInputEl.focus();
  }, 0);
}

function closeAddMemoryModal() {
  addMemoryModalEl.classList.remove("open");
  addMemoryModalEl.setAttribute("aria-hidden", "true");
}

function renderMemoryManagerList() {
  memoryManagerListEl.innerHTML = "";

  if (state.memoryManagerItems.length === 0) {
    const empty = document.createElement("div");
    empty.className = "memory-row empty";
    empty.textContent = "No long-term memory yet.";
    memoryManagerListEl.appendChild(empty);
    return;
  }

  for (const item of state.memoryManagerItems) {
    const row = document.createElement("div");
    row.className = "memory-row";

    const content = document.createElement("div");
    content.className = "memory-content";
    content.innerHTML = `<strong>${escapeHtml(item.type)}</strong><p>${escapeHtml(item.content)}</p>`;

    const actions = document.createElement("div");
    actions.className = "memory-actions";

    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "ghost";
    editBtn.textContent = "Edit";
    editBtn.disabled = state.isBusy;
    editBtn.addEventListener("click", async () => {
      const next = window.prompt("Edit memory content", item.content);
      if (typeof next !== "string") return;
      const trimmed = next.trim();
      if (!trimmed) return;

      const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory/${encodeURIComponent(item.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: trimmed }),
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Failed to edit memory");
      }
      await refreshMemoryManager();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "ghost";
    deleteBtn.textContent = "Delete";
    deleteBtn.disabled = state.isBusy;
    deleteBtn.addEventListener("click", async () => {
      if (!window.confirm("Delete this memory item?")) return;
      const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Failed to delete memory");
      }
      await refreshMemoryManager();
    });

    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(content);
    row.appendChild(actions);
    memoryManagerListEl.appendChild(row);
  }
}

async function refreshMemoryManager() {
  const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory?limit=200`, {
    method: "GET",
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to load long-term memory");
  }

  state.memoryManagerItems = Array.isArray(data.memories) ? data.memories : [];
  renderMemoryManagerList();
}

async function openMemoryManagerModal() {
  if (state.isBusy) return;
  memoryManagerModalEl.classList.add("open");
  memoryManagerModalEl.setAttribute("aria-hidden", "false");
  setStatus("Loading long-term memory...");
  try {
    await refreshMemoryManager();
    requestAnimationFrame(() => {
      memoryManagerListEl.scrollTop = 0;
    });
    setStatus("Ready");
  } catch (error) {
    setStatus("Error");
    addMessage(getActiveConversation(), "system", `Error: ${error.message}`, true);
  }
}

async function addMemoryItem() {
  const content = memoryContentInputEl.value.trim();
  if (!content) return;

  const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      type: memoryTypeInputEl.value,
      content,
    }),
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to add memory");
  }

  memoryContentInputEl.value = "";
  closeAddMemoryModal();
  await refreshMemoryManager();
}

async function summarizeLongMemory() {
  const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory/summarize`, {
    method: "POST",
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to summarize memory");
  }
  await refreshMemoryManager();
}

async function pruneLongMemory() {
  const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory/prune`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ maxItems: 300, maxAgeDays: 365, minImportance: 0.2 }),
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to prune memory");
  }

  await refreshMemoryManager();
}

async function exportLongMemory() {
  const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/export`, {
    method: "GET",
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to export memory");
  }

  const payload = JSON.stringify(data, null, 2);
  await copyText(payload);
  addMessage(getActiveConversation(), "system", "User memory exported and copied to clipboard.", true);
}

async function clearAllLongMemory() {
  const confirmClear = window.confirm("Clear all active long-term memories for this user?");
  if (!confirmClear) return;

  const response = await apiFetch(`/api/users/${encodeURIComponent(getActiveUserId())}/memory`, {
    method: "DELETE",
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to clear memory");
  }

  await refreshMemoryManager();
  addMessage(getActiveConversation(), "system", "Cleared active long-term memories for this user.", true);
}

function formatAgentLogs(logs) {
  if (!Array.isArray(logs) || logs.length === 0) {
    return ["No intermediate agent logs were returned."];
  }

  return logs.slice(0, 20).map((entry, index) => {
    const step = entry?.step ?? index + 1;
    const action = entry?.action || "unknown";
    const toolName = entry?.toolName ? ` tool=${entry.toolName}` : "";
    const actionOnly = new Set(["duplicate_tool", "tool_limit", "tool_failures", "tool_blocked", "final", "fallback"]);
    const status = actionOnly.has(action) ? "" : entry?.ok === false ? " fail" : " ok";
    const reason = entry?.reason ? ` reason=${entry.reason}` : "";
    const error = entry?.error ? ` error=${entry.error}` : "";
    return `STEP ${step}: action=${action}${toolName}${status}${reason}${error}`;
  });
}

async function runAgentTaskFromUi() {
  if (state.isBusy) return;
  const conversation = getActiveConversation();
  if (!conversation) return;

  const taskText = messageInputEl.value.trim();
  if (!taskText) {
    addMessage(conversation, "system", "Please enter a task in the input box before running agent mode.", true);
    return;
  }

  addMessage(conversation, "user", `[Agent Task]\n${taskText}`, true);
  messageInputEl.value = "";
  setBusy(true);
  setStatus("Agent running...");

  try {
    const response = await apiFetch("/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: taskText,
        userId: conversation.settings.userId,
        sessionId: conversation.sessionId,
        model: conversation.settings.model,
        maxSteps: 6,
      }),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Agent task failed");
    }

    const logLines = formatAgentLogs(data.logs);
    const artifactCount = Array.isArray(data.artifacts) ? data.artifacts.length : 0;
    addMessage(conversation, "assistant", data.final || "Agent completed.", true);

    pushTraceMessage(conversation, "Agent Trace", [
      ...logLines,
      `Artifacts: ${artifactCount}`,
      data.truncated ? "Reached step limit: yes" : "Reached step limit: no",
    ]);

    setStatus("Ready");
  } catch (error) {
    addMessage(conversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
    messageInputEl.focus();
  }
}

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.isBusy) return;

  const activeConversation = getActiveConversation();
  if (!activeConversation) return;

  const userMessage = messageInputEl.value.trim();
  if (!userMessage) return;

  activeConversation.settings = readSettingsFromInputs();
  addMessage(activeConversation, "user", userMessage, true);
  messageInputEl.value = "";

  setBusy(true);

  try {
    await sendConversationMessage(activeConversation, userMessage);
    clearPendingAttachments();
    setStatus("Ready");
  } catch (error) {
    addMessage(activeConversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
    messageInputEl.focus();
  }
});

runAgentBtnEl.addEventListener("click", () => {
  runAgentTaskFromUi();
});

applyContextConfigEl.addEventListener("click", () => {
  applyContextConfig().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

newChatBtnEl.addEventListener("click", () => {
  if (state.isBusy) return;

  const conversation = createConversation();
  state.conversations.unshift(conversation);
  state.activeConversationId = conversation.id;
  state.openConversationMenuId = "";
  saveState();
  renderAll();
  messageInputEl.focus();
});

resetMemoryEl.addEventListener("click", async () => {
  if (state.isBusy) return;

  const activeConversation = getActiveConversation();
  if (!activeConversation) return;

  try {
    const response = await apiFetch("/api/memory/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: activeConversation.sessionId, userId: activeConversation.settings.userId }),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Failed to reset memory");
    }

    activeConversation.messages = [];
    activeConversation.loadedMemorySummary = "";
    activeConversation.title = DEFAULT_TITLE;
    activeConversation.updatedAt = Date.now();
    saveState();
    renderAll();
    addMessage(activeConversation, "system", "Short-term memory reset for this chat.", true);
  } catch (error) {
    addMessage(activeConversation, "system", `Error: ${error.message}`, true);
  }
});

toggleSettingsEl.addEventListener("click", () => {
  setSettingsOpen(!state.settingsOpen);
});

cancelLoadMemoryEl.addEventListener("click", () => {
  closeLoadMemoryModal();
});

confirmLoadMemoryEl.addEventListener("click", () => {
  loadSelectedConversationMemory();
});

cancelRenameConversationEl.addEventListener("click", () => {
  closeRenameConversationModal();
});

confirmRenameConversationEl.addEventListener("click", () => {
  confirmRenameConversation();
});

renameConversationInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    confirmRenameConversation();
  }
});

loadMemoryModalEl.addEventListener("click", (event) => {
  if (event.target === loadMemoryModalEl) {
    closeLoadMemoryModal();
  }
});

renameConversationModalEl.addEventListener("click", (event) => {
  if (event.target === renameConversationModalEl) {
    closeRenameConversationModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;

  if (renameConversationModalEl.classList.contains("open")) {
    closeRenameConversationModal();
    return;
  }

  if (loadMemoryModalEl.classList.contains("open")) {
    closeLoadMemoryModal();
  }
});

document.addEventListener("click", (event) => {
  if (!state.openConversationMenuId) return;
  if (!(event.target instanceof Element)) return;
  if (event.target.closest(".conversation-actions")) return;

  state.openConversationMenuId = "";
  renderConversationList();
});

attachmentInputEl.addEventListener("change", () => {
  handleAttachmentInputChange().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

openMemoryManagerEl.addEventListener("click", () => {
  openMemoryManagerModal();
});

openAddMemoryModalEl.addEventListener("click", () => {
  openAddMemoryModal();
});

cancelAddMemoryModalEl.addEventListener("click", () => {
  closeAddMemoryModal();
});

closeMemoryManagerEl.addEventListener("click", () => {
  closeMemoryManagerModal();
});

refreshMemoryListEl.addEventListener("click", () => {
  refreshMemoryManager().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

addMemoryItemEl.addEventListener("click", () => {
  addMemoryItem().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

summarizeMemoryBtnEl.addEventListener("click", () => {
  summarizeLongMemory().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

pruneMemoryBtnEl.addEventListener("click", () => {
  pruneLongMemory().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

exportMemoryBtnEl.addEventListener("click", () => {
  exportLongMemory().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

clearMemoryBtnEl.addEventListener("click", () => {
  clearAllLongMemory().catch((error) => {
    addMessage(getActiveConversation(), "system", "Error: " + error.message, true);
  });
});

memoryManagerModalEl.addEventListener("click", (event) => {
  if (event.target === memoryManagerModalEl) {
    closeMemoryManagerModal();
  }
});

addMemoryModalEl.addEventListener("click", (event) => {
  if (event.target === addMemoryModalEl) {
    closeAddMemoryModal();
  }
});

for (const input of settingsInputs) {
  const eventType = input.tagName === "TEXTAREA" ? "input" : "change";
  input.addEventListener(eventType, syncActiveSettingsFromInputs);
}

loadState();
setModelOptions(FALLBACK_MODELS);
renderCapabilityBadges();
renderAll();
setSettingsOpen(state.settingsOpen);
setStatus("Ready");
fetchRuntimeCapabilities().then(() => {
  renderAll();
});
fetchContextConfig();
