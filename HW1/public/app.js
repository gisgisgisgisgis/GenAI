const modelEl = document.getElementById("model");
const systemPromptEl = document.getElementById("systemPrompt");
const temperatureEl = document.getElementById("temperature");
const maxTokensEl = document.getElementById("maxTokens");
const topPEl = document.getElementById("topP");
const presencePenaltyEl = document.getElementById("presencePenalty");
const frequencyPenaltyEl = document.getElementById("frequencyPenalty");
const streamModeEl = document.getElementById("streamMode");
const responseModeEl = document.getElementById("responseMode");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chatForm");
const messageInputEl = document.getElementById("messageInput");
const statusEl = document.getElementById("status");
const sendBtnEl = document.getElementById("sendBtn");
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

const STORAGE_CONVERSATIONS_KEY = "custom_groq_conversations";
const STORAGE_ACTIVE_ID_KEY = "custom_groq_active_conversation_id";
const STORAGE_SETTINGS_OPEN_KEY = "custom_groq_settings_open";
const LEGACY_SESSION_KEY = "custom_gpt_session_id";
const DEFAULT_TITLE = "New Chat";
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant tailored to the user's needs.";

const defaultSettings = {
  model: "llama-3.1-8b-instant",
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
};

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
  const reasoningMode = input?.reasoningMode === "think" ? "think" : "fast";

  return {
    model: typeof input?.model === "string" ? input.model : defaultSettings.model,
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
    settings: { ...defaultSettings },
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
    model: modelEl.value,
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
  modelEl.value = settings.model;
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

async function replaceServerMemory(sessionId, messages) {
  const response = await fetch("/api/memory/replace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, messages }),
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
    await replaceServerMemory(copiedConversation.sessionId, copiedConversation.messages);
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
    const response = await fetch("/api/memory/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        targetSessionId: targetConversation.sessionId,
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
  return {
    sessionId: conversation.sessionId,
    message: userMessage,
    model: conversation.settings.model,
    systemPrompt: conversation.settings.systemPrompt,
    stream: conversation.settings.stream && conversation.settings.reasoningMode !== "think",
    temperature: conversation.settings.temperature,
    maxTokens: conversation.settings.maxTokens,
    topP: conversation.settings.topP,
    presencePenalty: conversation.settings.presencePenalty,
    frequencyPenalty: conversation.settings.frequencyPenalty,
    reasoningMode: conversation.settings.reasoningMode,
    loadedMemorySummary: conversation.loadedMemorySummary,
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
    await replaceServerMemory(branchedConversation.sessionId, branchedConversation.messages);
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
    await replaceServerMemory(activeConversation.sessionId, activeConversation.messages);
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

  fetch("/api/memory/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId: removedConversation.sessionId }),
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
  }
  sendBtnEl.disabled = isBusy;
  messageInputEl.disabled = isBusy;
  newChatBtnEl.disabled = isBusy;
  resetMemoryEl.disabled = isBusy;
  cancelLoadMemoryEl.disabled = isBusy;
  confirmLoadMemoryEl.disabled = isBusy;
  cancelRenameConversationEl.disabled = isBusy;
  confirmRenameConversationEl.disabled = isBusy;
  renameConversationInputEl.disabled = isBusy;

  for (const input of settingsInputs) {
    input.disabled = isBusy;
  }

  renderConversationList();
  renderMessages();
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
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }

  addMessage(conversation, "assistant", data.reply, true);
}

async function sendStreaming(payload, conversation, pendingBubble) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

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
    setStatus("Ready");
  } catch (error) {
    addMessage(activeConversation, "system", `Error: ${error.message}`, true);
    setStatus("Error");
  } finally {
    setBusy(false);
    messageInputEl.focus();
  }
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
    const response = await fetch("/api/memory/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: activeConversation.sessionId }),
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

for (const input of settingsInputs) {
  const eventType = input.tagName === "TEXTAREA" ? "input" : "change";
  input.addEventListener(eventType, syncActiveSettingsFromInputs);
}

loadState();
renderAll();
setSettingsOpen(state.settingsOpen);
setStatus("Ready");
