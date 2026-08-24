const $ = (selector) => document.querySelector(selector);
const ui = {
  messages: $("#messages"), form: $("#form"), input: $("#message"), send: $("#send-btn"),
  list: $("#session-list"), title: $("#session-title"), meta: $("#session-meta"),
  tps: $("#tps-value"), ttft: $("#ttft-value"), tokens: $("#token-value"), context: $("#context-value"),
  generation: $("#generation-status"), settings: $("#settings-panel"), settingsScrim: $("#settings-scrim"),
  sidebar: $("#sidebar"), sidebarScrim: $("#sidebar-scrim"), toast: $("#toast"),
};
const sessions = new Map();
let activeId = null, serverStatus = {}, toastTimer;

if (window.marked) marked.setOptions({ breaks: true, gfm: true });
const escapeHtml = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function renderMath(tex, displayMode) {
  if (!window.katex) return `<span class="math-${displayMode ? "display" : "inline"}">${escapeHtml(tex)}</span>`;
  try { return katex.renderToString(tex, { displayMode, throwOnError: false, strict: "ignore" }); }
  catch { return escapeHtml(tex); }
}

function renderMarkdown(text) {
  const code = [], math = [];
  let source = String(text || "").replace(/```[\s\S]*?```|`[^`\n]+`/g, (value) => {
    code.push(value); return `INFENGCODE${code.length - 1}TOKEN`;
  });
  source = source.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
    math.push(renderMath(tex.trim(), true)); return `INFENGMATH${math.length - 1}TOKEN`;
  }).replace(/(^|[^\\$])\$([^\n$]+?)\$(?!\$)/g, (_, prefix, tex) => {
    math.push(renderMath(tex.trim(), false)); return `${prefix}INFENGMATH${math.length - 1}TOKEN`;
  });
  source = escapeHtml(source).replace(/INFENGCODE(\d+)TOKEN/g, (_, index) => code[index]);
  let html = window.marked ? marked.parse(source) : source.replace(/\n/g, "<br>");
  return html.replace(/INFENGMATH(\d+)TOKEN/g, (_, index) => math[index]);
}

function enhance(container) {
  container.querySelectorAll("pre code").forEach((block) => {
    if (window.hljs && !block.dataset.highlighted) hljs.highlightElement(block);
  });
}

async function api(path, options = {}) {
  const request = { ...options, headers: { ...(options.headers || {}) } };
  if (request.body && typeof request.body !== "string") {
    request.headers["content-type"] = "application/json"; request.body = JSON.stringify(request.body);
  }
  const response = await fetch(path, request);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function notify(message) {
  clearTimeout(toastTimer); ui.toast.textContent = message; ui.toast.classList.add("show");
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 2600);
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(2)} GB` : `${Math.ceil(bytes / 1048576)} MB`;
}

function formatMetric(value, digits = 0) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function mergeSession(value) {
  const current = sessions.get(value.id);
  if (current) Object.assign(current, value);
  else sessions.set(value.id, { ...value, messages: value.messages || null, draft: "" });
  return sessions.get(value.id);
}

function renderServer() {
  const dot = $("#model-dot"), label = $("#model-status"), detail = $("#server-detail"), load = $("#load-btn");
  dot.className = "status-dot";
  if (serverStatus.loaded) {
    dot.classList.add("loaded"); label.textContent = "Qwen 3.5 · online";
    detail.textContent = `${serverStatus.active_sessions || 0} active · ${serverStatus.mtp ? "MTP ready" : "target only"}`;
    load.textContent = "Ready"; load.disabled = true;
  } else if (serverStatus.loading) {
    dot.classList.add("loading"); label.textContent = "Loading model"; detail.textContent = "One shared instance";
    load.textContent = "Loading"; load.disabled = true;
  } else if (serverStatus.error) {
    dot.classList.add("error"); label.textContent = "Load failed"; detail.textContent = serverStatus.error;
    load.textContent = "Retry"; load.disabled = false;
  } else {
    label.textContent = "Model offline"; detail.textContent = "Metal runtime ready"; load.textContent = "Load"; load.disabled = false;
  }
  $("#memory-value").textContent = formatBytes(serverStatus.mapped_kv_bytes || 0);
  $("#session-count").textContent = `${sessions.size} / ${serverStatus.max_sessions || 8}`;
  $("#new-chat-btn").disabled = sessions.size >= (serverStatus.max_sessions || 8);
  const speculative = $("#speculative");
  speculative.disabled = serverStatus.loaded && !serverStatus.mtp;
  if (speculative.disabled) speculative.checked = false;
}

function renderSidebar() {
  ui.list.replaceChildren();
  const ordered = [...sessions.values()].sort((a, b) => b.updated_at - a.updated_at);
  for (const session of ordered) {
    const item = document.createElement("div"); item.className = `session-item${session.id === activeId ? " active" : ""}`;
    item.dataset.id = session.id;
    const select = document.createElement("button"); select.className = "session-select";
    const title = document.createElement("span"); title.className = "session-title"; title.textContent = session.title;
    const sub = document.createElement("span"); sub.className = "session-sub";
    if (session.generating) {
      sub.classList.add("live"); sub.innerHTML = `<i class="mini-dot"></i>${session.metrics?.tps ? `${session.metrics.tps} tok/s` : "generating"}`;
    } else sub.textContent = session.sequence_id === null ? "Not allocated" : `SEQ ${String(session.sequence_id).padStart(2, "0")} · ${session.message_count || session.messages?.length || 0} messages`;
    const remove = document.createElement("button"); remove.className = "delete-session"; remove.title = "Delete session";
    remove.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5M14 11v5"/></svg>';
    remove.onclick = (event) => { event.stopPropagation(); deleteSession(session); };
    select.onclick = () => selectSession(session.id);
    select.append(title, sub); item.append(select, remove); ui.list.append(item);
  }
  renderServer();
}

function renderTop() {
  const session = sessions.get(activeId), metrics = session?.metrics || {};
  ui.title.textContent = session?.title || "New session";
  ui.meta.textContent = session?.sequence_id === null || session?.sequence_id === undefined ? "No sequence allocated" : `Sequence ${session.sequence_id} · ${formatBytes(session.mapped_kv_bytes)} mapped`;
  ui.tps.textContent = formatMetric(metrics.tps, 1); ui.ttft.textContent = formatMetric(metrics.ttft_ms);
  ui.tokens.textContent = metrics.generated_tokens || 0; ui.context.textContent = metrics.context_tokens || 0;
  const generating = Boolean(session?.generating), stopping = Boolean(session?.stopping);
  const acceptance = metrics.acceptance_rate === null || metrics.acceptance_rate === undefined ? "" : ` · ${Math.round(metrics.acceptance_rate * 100)}% accepted`;
  ui.generation.textContent = stopping ? "Stopping after current block…" : generating ? `Generating${metrics.tps ? ` · ${metrics.tps} tok/s` : "…"}${acceptance}` : "Ready";
  ui.send.className = stopping ? "generating stopping" : generating ? "generating" : "";
  ui.send.disabled = !session || stopping || (!generating && !ui.input.value.trim());
  ui.send.setAttribute("aria-label", generating ? "Stop generation" : "Send message");
}

function messageStats(metrics, cancelled) {
  if (!metrics || !metrics.generated_tokens) return null;
  const stats = document.createElement("div"); stats.className = "message-stats";
  if (metrics.tps) stats.innerHTML += `<span>${metrics.tps} tok/s</span>`;
  stats.innerHTML += `<span>${metrics.generated_tokens} tokens</span><span>TTFT ${formatMetric(metrics.ttft_ms)} ms</span>`;
  if (metrics.speculative && metrics.drafted_tokens) stats.innerHTML += `<span>MTP ${Math.round(metrics.acceptance_rate * 100)}% accepted</span>`;
  if (cancelled) stats.innerHTML += '<span class="cancelled">stopped</span>';
  return stats;
}

function messageNode(message, index) {
  const root = document.createElement("article"); root.className = `message ${message.role}`; root.dataset.index = index;
  const body = document.createElement("div"); body.className = "message-body";
  if (message.role === "user") { body.textContent = message.content; root.append(body); return root; }
  const avatar = document.createElement("div"); avatar.className = "avatar"; avatar.textContent = "I";
  if (message.thinking) {
    const thought = document.createElement("details"); thought.className = "thought"; thought.open = Boolean(message.streaming);
    const summary = document.createElement("summary"); summary.textContent = message.streaming ? "Thinking" : "Reasoning";
    const content = document.createElement("div"); content.textContent = message.thinking; thought.append(summary, content); body.append(thought);
  }
  const response = document.createElement("div"); response.className = `response-content${message.error ? " error-text" : ""}`;
  response.dataset.status = message.status || (message.streaming ? "Generating…" : "No response generated.");
  if (message.error) response.textContent = `Error: ${message.error}`;
  else if (message.streaming) response.textContent = message.content || "";
  else response.innerHTML = renderMarkdown(message.content || "");
  body.append(response); if (!message.streaming) enhance(response);
  if (message.streaming) { const cursor = document.createElement("span"); cursor.className = "cursor"; body.append(cursor); }
  const stats = messageStats(message.metrics, message.cancelled); if (stats) body.append(stats);
  root.append(avatar, body); return root;
}

function renderMessages() {
  const session = sessions.get(activeId); ui.messages.replaceChildren();
  if (!session?.messages?.length) {
    const welcome = document.createElement("div"); welcome.className = "welcome";
    welcome.innerHTML = '<div class="welcome-mark">I</div><h1>Fast local inference, independent contexts.</h1><p>Run up to eight conversations on one Metal model. Active sessions are continuously batched without mixing their state.</p><div class="welcome-tags"><span>Private paged KV</span><span>Live tok/s</span><span>Cancelable streams</span></div>';
    ui.messages.append(welcome); return;
  }
  session.messages.forEach((message, index) => ui.messages.append(messageNode(message, index)));
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function refreshMessage(session, index) {
  if (session.renderFrame) return;
  session.renderFrame = requestAnimationFrame(() => {
    session.renderFrame = null;
    if (activeId !== session.id) return;
    const old = ui.messages.querySelector(`[data-index="${index}"]`);
    if (old) old.replaceWith(messageNode(session.messages[index], index)); else renderMessages();
    ui.messages.scrollTop = ui.messages.scrollHeight; renderTop();
  });
}

async function syncSessions() {
  const values = await api("/api/sessions");
  const ids = new Set(values.map((value) => value.id));
  for (const id of sessions.keys()) if (!ids.has(id)) sessions.delete(id);
  values.forEach(mergeSession); renderSidebar(); renderTop();
}

async function loadSession(id) {
  const session = sessions.get(id);
  if (!session || session.messages) return session;
  return mergeSession(await api(`/api/sessions/${id}`));
}

async function selectSession(id) {
  if (!sessions.has(id)) return;
  const current = sessions.get(activeId); if (current) current.draft = ui.input.value;
  activeId = id; await loadSession(id);
  if (activeId !== id) return;
  ui.input.value = sessions.get(id).draft || ""; resizeInput(); renderSidebar(); renderTop(); renderMessages(); closeSidebar();
}

async function createSession() {
  try { const session = mergeSession(await api("/api/sessions", { method: "POST" })); await selectSession(session.id); ui.input.focus(); }
  catch (error) { notify(error.message); }
}

async function deleteSession(session) {
  if (!confirm(`Delete “${session.title}”?`)) return;
  try {
    await api(`/api/sessions/${session.id}`, { method: "DELETE" }); sessions.delete(session.id);
    if (activeId === session.id) {
      activeId = [...sessions.values()].sort((a, b) => b.updated_at - a.updated_at)[0]?.id || null;
      if (!activeId) return createSession();
      await loadSession(activeId);
    }
    renderSidebar(); renderTop(); renderMessages();
  } catch (error) { notify(error.message); }
}

function getParameters() {
  return { thinking: $("#thinking").checked, temperature: Number($("#temperature").value),
           top_p: Number($("#top_p").value), top_k: Number($("#top_k").value),
           max_tokens: Number($("#max_tokens").value), speculative: $("#speculative").checked,
           draft_tokens: Number($("#draft_tokens").value) };
}

function applyStreamEvent(session, message, data) {
  if (data.error) throw new Error(data.error);
  if (data.status) message.status = data.status;
  if (data.sequence_id !== undefined) session.sequence_id = data.sequence_id;
  if (data.metrics) session.metrics = message.metrics = data.metrics;
  if (data.thinking_delta) message.thinking += data.thinking_delta;
  if (data.response_delta) message.content += data.response_delta;
  if (data.done) { message.streaming = false; message.cancelled = Boolean(data.cancelled); }
}

async function streamChat(session, text, message, index) {
  try {
    const response = await fetch(`/api/sessions/${session.id}/chat`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: text, ...getParameters() }),
    });
    if (!response.ok || !response.body) {
      const error = await response.json().catch(() => ({})); throw new Error(error.error || `Chat failed (${response.status})`);
    }
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "", finished = false;
    while (!finished) {
      const chunk = await reader.read(); if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true }).replace(/\r/g, "");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const payload = block.split("\n").filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart()).join("\n");
        if (!payload) continue;
        const data = JSON.parse(payload); applyStreamEvent(session, message, data); refreshMessage(session, index);
        if (data.done) { finished = true; break; }
      }
    }
    if (!finished) throw new Error("Generation stream ended unexpectedly");
    await reader.cancel();
  } catch (error) {
    message.error = error.message;
    message.streaming = false;
  } finally {
    session.generating = session.stopping = false;
    session.message_count = session.messages.length; session.updated_at = Date.now() / 1000;
    refreshMessage(session, index); renderSidebar(); renderTop(); pollStatus();
  }
}

async function stopSession(session) {
  if (!session?.generating || session.stopping) return;
  session.stopping = true; renderTop(); renderSidebar();
  try {
    await api(`/api/sessions/${session.id}/cancel`, { method: "POST" });
  } catch (error) { session.stopping = false; notify(error.message); renderTop(); }
}

async function pollStatus() {
  try { serverStatus = await api("/api/status"); renderServer(); }
  catch { serverStatus = { error: "Server unavailable", max_sessions: 8 }; renderServer(); }
}

function resizeInput() {
  ui.input.style.height = "auto"; ui.input.style.height = `${Math.min(ui.input.scrollHeight, 180)}px`; renderTop();
}
function openSettings() { ui.settings.classList.add("open"); ui.settingsScrim.classList.add("open"); }
function closeSettings() { ui.settings.classList.remove("open"); ui.settingsScrim.classList.remove("open"); }
function openSidebar() { ui.sidebar.classList.add("open"); ui.sidebarScrim.classList.add("open"); }
function closeSidebar() { ui.sidebar.classList.remove("open"); ui.sidebarScrim.classList.remove("open"); }

ui.form.addEventListener("submit", (event) => {
  event.preventDefault(); const session = sessions.get(activeId);
  if (session?.generating) return stopSession(session);
  const text = ui.input.value.trim(); if (!session || !text) return;
  ui.input.value = ""; resizeInput(); session.draft = "";
  session.messages ||= []; session.messages.push({ role: "user", content: text });
  const assistant = { role: "assistant", content: "", thinking: "", status: "Queued…", streaming: true, metrics: {} };
  session.messages.push(assistant); session.generating = true; session.updated_at = Date.now() / 1000;
  if (session.title === "New chat") session.title = text.slice(0, 48) + (text.length > 48 ? "…" : "");
  renderMessages(); renderSidebar(); renderTop(); streamChat(session, text, assistant, session.messages.length - 1);
});

ui.input.addEventListener("input", resizeInput);
ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ui.form.requestSubmit(); }
});
$("#new-chat-btn").onclick = createSession;
$("#load-btn").onclick = async () => { try { await api("/api/load", { method: "POST" }); pollStatus(); } catch (error) { notify(error.message); } };
$("#settings-btn").onclick = openSettings; $("#settings-close").onclick = closeSettings; ui.settingsScrim.onclick = closeSettings;
$("#sidebar-open").onclick = openSidebar; $("#sidebar-close").onclick = closeSidebar; ui.sidebarScrim.onclick = closeSidebar;
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); createSession(); }
  if (event.key === "Escape") { closeSettings(); closeSidebar(); }
});

const settingIds = ["thinking", "speculative", "draft_tokens", "temperature", "top_p", "top_k", "max_tokens"];
try {
  const saved = JSON.parse(localStorage.getItem("infeng-settings") || "{}");
  settingIds.forEach((id) => { if (saved[id] !== undefined) $(`#${id}`)[$(`#${id}`).type === "checkbox" ? "checked" : "value"] = saved[id]; });
} catch { /* Ignore corrupt local preferences. */ }
settingIds.forEach((id) => {
  const input = $(`#${id}`), output = $(`#${id}-val`);
  const update = () => {
    if (output) output.textContent = id === "temperature" || id === "top_p" ? Number(input.value).toFixed(2) : input.value;
    const saved = Object.fromEntries(settingIds.map((key) => [key, $(`#${key}`).type === "checkbox" ? $(`#${key}`).checked : $(`#${key}`).value]));
    localStorage.setItem("infeng-settings", JSON.stringify(saved));
  };
  input.addEventListener("input", update); update();
});

async function initialize() {
  try {
    await Promise.all([pollStatus(), syncSessions()]);
    if (!sessions.size) await createSession();
    else await selectSession([...sessions.values()].sort((a, b) => b.updated_at - a.updated_at)[0].id);
  } catch (error) { notify(error.message); renderServer(); renderTop(); renderMessages(); }
}

initialize();
setInterval(pollStatus, 3000);
