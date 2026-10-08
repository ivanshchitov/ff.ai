"use strict";

// Чат-фронтенд ff.ai: один файл, без сборщика и внешних библиотек.
//
// Лента — основная форма: вопрос пользователя, ответ ассистента, а под ответом — то, что относится
// именно к нему: фаза запроса, строки журнала, метрики, предупреждения, источники и цитаты.
//
// Правило экранирования: всё, что приходит с сервера (ответ модели, фрагменты документации, диффы
// патчей, отчёты, строки журнала), сначала экранируется целиком, и только потом в экранированном
// тексте распознаётся минимальная разметка. Разметку из чужих данных строить нельзя.

const PHASE_LABELS = {
  request: "Отправка вопроса…",
  compression: "Суммаризация контекста…",
  facts_update: "Обновление фактов…",
  mcp_connect: "Подключение к MCP-серверам…",
  mcp_tool: "Вызов инструмента…",
  docs_rerank: "Отбор фрагментов документации…",
  code_query: "Поисковый запрос по исходникам…",
  code_rerank: "Отбор фрагментов кода…",
  tool_choice: "Выбор инструментов…",
  task_plan: "Планирование задачи…",
  task_execute: "Выполнение подзадачи…",
  task_validate: "Проверка артефакта…",
};

const STAGES = [
  ["planning", "Планирование"],
  ["execution", "Выполнение"],
  ["validation", "Проверка"],
  ["done", "Завершено"],
];

const CONFIRMATION_LABELS = {
  patch: "Запись в репозиторий",
  ops: "Необратимый шаг конвейера",
};

const el = (id) => document.getElementById(id);

// --- тексты и минимальная разметка -----------------------------------------------------------

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

function escapeText(value) {
  return String(value === null || value === undefined ? "" : value).replace(
    /[&<>"']/g,
    (char) => ESCAPES[char],
  );
}

function textNode(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function button(label, className) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className || "";
  node.textContent = label;
  return node;
}

// Экранированный текст → фрагмент: блоки кода, заголовки, списки, внутристрочный код и жирный.
function markup(text) {
  const fragment = document.createDocumentFragment();
  const parts = String(text === null || text === undefined ? "" : text).split("```");
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      fragment.appendChild(codeBlock(part));
      return;
    }
    appendRichText(fragment, part);
  });
  return fragment;
}

function codeBlock(raw) {
  const lines = raw.replace(/^\n/, "").split("\n");
  const language = (lines[0] || "").trim();
  const hasLanguage = language && !/\s/.test(language) && language.length <= 20;
  const body = (hasLanguage ? lines.slice(1) : lines).join("\n").replace(/\n$/, "");
  const wrap = document.createElement("div");
  wrap.className = "code";
  const head = document.createElement("div");
  head.className = "code-head";
  head.appendChild(textNode("span", "", hasLanguage ? language : "текст"));
  const copy = button("Копировать", "ghost");
  copy.addEventListener("click", () => copyCode(copy, body));
  head.appendChild(copy);
  wrap.appendChild(head);
  const pre = document.createElement("pre");
  pre.appendChild(textNode("code", "", body));
  wrap.appendChild(pre);
  return wrap;
}

function inlineFragment(escaped) {
  const fragment = document.createDocumentFragment();
  const pattern = /`([^`]+)`|\*\*([^*]+)\*\*/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(escaped)) !== null) {
    if (match.index > last) fragment.appendChild(document.createTextNode(escaped.slice(last, match.index)));
    if (match[1] !== undefined) fragment.appendChild(textNode("code", "md-code", match[1]));
    else fragment.appendChild(textNode("strong", "", match[2]));
    last = pattern.lastIndex;
  }
  if (last < escaped.length) fragment.appendChild(document.createTextNode(escaped.slice(last)));
  return fragment;
}

function inlineText(escaped) {
  const holder = document.createDocumentFragment();
  holder.appendChild(inlineFragment(escaped));
  return holder;
}

function appendRichText(host, raw) {
  const escaped = escapeText(raw);
  let list = null;
  let tag = "";
  const flush = () => {
    if (list) {
      host.appendChild(list);
      list = null;
      tag = "";
    }
  };
  for (const line of escaped.split("\n")) {
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      const node = textNode("h4", "md-heading");
      node.appendChild(inlineText(heading[2]));
      host.appendChild(node);
      continue;
    }
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    const ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (bullet || ordered) {
      const wanted = ordered ? "ol" : "ul";
      if (!list || tag !== wanted) {
        flush();
        list = document.createElement(wanted);
        list.className = "md-list";
        tag = wanted;
      }
      const item = document.createElement("li");
      item.appendChild(inlineFragment((bullet || ordered)[1]));
      list.appendChild(item);
      continue;
    }
    flush();
    if (!line.trim()) continue;
    const paragraph = textNode("p", "md-p");
    paragraph.appendChild(inlineFragment(line));
    host.appendChild(paragraph);
  }
  flush();
}

async function copyCode(node, text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    node.textContent = "Скопировано";
  } catch (error) {
    node.textContent = "Не скопировано";
  }
  setTimeout(() => {
    node.textContent = "Копировать";
  }, 1500);
}

// --- обращения к серверу ---------------------------------------------------------------------

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      if (payload && payload.detail) detail = payload.detail;
    } catch (error) {
      /* тело не JSON — остаётся статус */
    }
    throw new Error(detail);
  }
  return response.json();
}

const postJson = (path, body) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

// --- лента: сообщения, прокрутка --------------------------------------------------------------

const state = { busy: false, stick: true, current: null, confirmations: new Map() };

function atBottom() {
  const feed = el("feed");
  return feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
}

function stickToBottom(force) {
  if (!state.stick && !force) return;
  const feed = el("feed");
  feed.scrollTop = feed.scrollHeight;
}

function message(role) {
  const node = document.createElement("article");
  node.className = "msg " + (role === "user" ? "user" : "assistant");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const body = document.createElement("div");
  body.className = "markup";
  bubble.appendChild(body);
  node.appendChild(bubble);
  el("feed").appendChild(node);
  stickToBottom();
  return { node, bubble, body, journal: new Set(), notes: null, phase: null, meta: null };
}

function userMessage(text) {
  const entry = message("user");
  entry.body.appendChild(markup(text));
  return entry;
}

function assistantStart() {
  const entry = message("assistant");
  entry.body.appendChild(textNode("span", "typing", "печатает"));
  state.current = entry;
  return entry;
}

function assistantFinish(entry) {
  const typing = entry.body.querySelector(".typing");
  if (typing) typing.remove();
  entry.body.normalize();
  if (state.current === entry) state.current = null;
}

function notesHost(entry) {
  if (!entry.notes) {
    entry.notes = textNode("div", "notes");
    entry.node.appendChild(entry.notes);
  }
  return entry.notes;
}

function addNote(entry, text) {
  notesHost(entry).appendChild(textNode("div", "", text));
  stickToBottom();
}

function setPhase(text) {
  const entry = state.current;
  if (!entry) return;
  if (!entry.phase) {
    entry.phase = textNode("div", "phase");
    entry.node.appendChild(entry.phase);
  }
  entry.phase.textContent = text;
  stickToBottom();
}

function addJournal(text) {
  const entry = state.current || lastAssistant();
  if (!entry) {
    const created = message("assistant");
    created.bubble.classList.add("command");
    addNote(created, text);
    return;
  }
  addJournalTo(entry, text);
}

function addJournalTo(entry, text) {
  if (entry.journal.has(text)) return;
  entry.journal.add(text);
  addNote(entry, text);
}

function lastAssistant() {
  const nodes = el("feed").querySelectorAll(".msg.assistant");
  if (!nodes.length) return null;
  const node = nodes[nodes.length - 1];
  return { node, bubble: node.querySelector(".bubble"), body: node.querySelector(".markup"), journal: new Set(), notes: node.querySelector(".notes") };
}

function metaLine(meta) {
  const seconds = typeof meta.elapsed_seconds === "number" ? meta.elapsed_seconds.toFixed(2) : "н/д";
  const cost = meta.cost_usd === null || meta.cost_usd === undefined
    ? "неизвестно"
    : "$" + Number(meta.cost_usd).toFixed(6);
  const speed = meta.elapsed_seconds > 0 && meta.completion_tokens > 0
    ? (meta.completion_tokens / meta.elapsed_seconds).toFixed(2) + " ток/сек"
    : "н/д";
  return "⏱ " + seconds + " с · токены " + meta.prompt_tokens + "+" + meta.completion_tokens +
    "=" + meta.total_tokens + " · стоимость " + cost + " · " + speed;
}

function renderMeta(entry, meta) {
  if (!meta) return;
  entry.meta = textNode("div", "meta", metaLine(meta));
  entry.node.appendChild(entry.meta);
  if (meta.finish_reason === "length") {
    const warn = meta.total_tokens > 0 && !meta.completion_tokens
      ? "⚠ модель исчерпала бюджет max_tokens: ответа нет"
      : "⚠ ответ мог быть обрезан техническим потолком запроса";
    entry.node.appendChild(textNode("div", "meta warn", warn));
  }
}

function renderCitations(entry, check) {
  if (!check || !check.line) return;
  entry.node.appendChild(textNode("div", "citations", check.line));
}

function renderSources(entry, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  details.appendChild(textNode("summary", "", "Источников: " + sources.length));
  const list = document.createElement("ul");
  for (const source of sources) {
    const item = document.createElement("li");
    item.appendChild(textNode("span", "ident", source.identifier || "—"));
    const where = [source.title, source.section].filter(Boolean).join(" — ");
    if (where) item.appendChild(document.createTextNode(" " + where));
    if (source.text) {
      const nested = document.createElement("details");
      nested.appendChild(textNode("summary", "", "фрагмент" + (source.truncated ? " (обрезан)" : "")));
      nested.appendChild(textNode("span", "quote", source.text));
      item.appendChild(nested);
    }
    list.appendChild(item);
  }
  details.appendChild(list);
  entry.node.appendChild(details);
}

// --- вопрос и команды ------------------------------------------------------------------------

async function ask(question) {
  if (state.busy) return;
  state.busy = true;
  el("send").disabled = true;
  userMessage(question);
  const entry = assistantStart();
  setPhase("Отправка вопроса…");
  try {
    const payload = await postJson("api/ask", { question });
    entry.body.replaceChildren(markup(payload.answer || ""));
    for (const line of payload.journal || []) addJournalTo(entry, line);
    renderMeta(entry, payload.meta);
    renderSources(entry, payload.sources);
    renderCitations(entry, payload.citations);
    if (!payload.answer) entry.bubble.classList.add("error");
  } catch (error) {
    entry.bubble.classList.add("error");
    entry.body.replaceChildren(textNode("p", "md-p", "Ошибка запроса: " + error.message));
  } finally {
    if (entry.phase) entry.phase.remove();
    assistantFinish(entry);
    state.busy = false;
    el("send").disabled = false;
    stickToBottom(true);
  }
}

async function runCommand(line) {
  userMessage(line);
  const entry = message("assistant");
  entry.bubble.classList.add("command");
  try {
    const payload = await postJson("api/command", { text: line });
    const lines = payload.lines || [];
    entry.body.appendChild(markup(lines.join("\n") || "Команда выполнена."));
    if (payload.unknown) addNote(entry, "команда неизвестна");
    if (payload.domain_changed) await loadStatus();
    if (payload.task_run_started) addNote(entry, "прогон задачи запущен");
    if (payload.profile_setup) {
      addNote(entry, "начата настройка профиля: ответьте на вопрос в терминале — в браузере диалог профиля пока не поддержан");
    }
  } catch (error) {
    entry.bubble.classList.add("error");
    entry.body.replaceChildren(textNode("p", "md-p", "Ошибка команды: " + error.message));
  }
  stickToBottom(true);
}

// --- задача ----------------------------------------------------------------------------------

function renderTask(snapshot, running) {
  const host = el("task");
  clear(host);
  const tasks = (snapshot && snapshot.tasks) || [];
  host.className = "task" + (tasks.length ? "" : " empty");
  if (!tasks.length) {
    host.textContent = "Задач нет. Добавить: команда «/task add <цель>».";
    return;
  }
  for (const task of tasks) {
    const card = document.createElement("div");
    const current = snapshot && snapshot.current === task.number;
    card.className = "task-card" + (current ? " current" : "");
    card.appendChild(textNode("div", "task-goal", "Задача " + task.number + ": " + (task.goal || "без цели")));
    const stage = (STAGES.find(([value]) => value === task.stage) || ["", task.stage])[1];
    const flags = [];
    if (task.status) flags.push(task.status);
    if (task.awaiting_edits) flags.push("ждёт правок");
    if (running && current) flags.push("выполняется");
    card.appendChild(textNode("div", "task-line", "Этап: " + stage + (flags.length ? " · " + flags.join(", ") : "")));
    if (task.current_step) card.appendChild(textNode("div", "task-line", task.current_step));
    if (task.expected_action) card.appendChild(textNode("div", "task-line", "Ожидается: " + task.expected_action));
    const plan = task.plan || [];
    if (plan.length) {
      const list = document.createElement("ul");
      list.className = "task-plan";
      for (const item of plan) {
        const li = document.createElement("li");
        li.className = "task-item" + (item.applied ? " applied" : "");
        li.textContent = (item.applied ? "[x] " : "[ ] ") + item.item;
        list.appendChild(li);
      }
      card.appendChild(list);
    }
    host.appendChild(card);
  }
}

function editsCard(event) {
  const entry = message("assistant");
  const card = document.createElement("div");
  card.className = "confirm";
  card.appendChild(textNode("div", "confirm-kind", "Правки плана"));
  card.appendChild(textNode("div", "confirm-summary", event.label || "Планирование"));
  const area = document.createElement("textarea");
  area.rows = 3;
  area.placeholder = "Пустая строка — выполнять план как есть";
  card.appendChild(area);
  const row = document.createElement("div");
  row.className = "row";
  const send = button("Отправить", "primary");
  const asIs = button("Как есть", "ghost");
  const submit = async (text) => {
    send.disabled = true;
    asIs.disabled = true;
    try {
      const payload = await postJson("api/task", { action: "edits", text });
      addNote(entry, payload.accepted ? "правки отправлены" : "правки не приняты: прогон не ждёт решения");
      card.remove();
    } catch (error) {
      addNote(entry, "правки не отправлены: " + error.message);
      send.disabled = false;
      asIs.disabled = false;
    }
  };
  send.addEventListener("click", () => submit(area.value));
  asIs.addEventListener("click", () => submit(""));
  row.appendChild(send);
  row.appendChild(asIs);
  card.appendChild(row);
  entry.node.appendChild(card);
  stickToBottom(true);
  area.focus();
}

// --- подтверждения ---------------------------------------------------------------------------

function confirmationCard(payload) {
  const entry = message("assistant");
  const card = document.createElement("div");
  card.className = "confirm";
  card.dataset.confirmation = payload.id;
  card.appendChild(textNode("div", "confirm-kind", CONFIRMATION_LABELS[payload.kind] || payload.kind));
  card.appendChild(textNode("div", "confirm-summary", payload.summary || ""));
  if (payload.detail) {
    const pre = document.createElement("pre");
    pre.className = "lines";
    pre.textContent = payload.detail;
    card.appendChild(pre);
  }
  const row = document.createElement("div");
  row.className = "row";
  const yes = button("Подтвердить", "primary");
  const no = button("Отклонить", "ghost");
  const answer = async (approved) => {
    yes.disabled = true;
    no.disabled = true;
    try {
      await postJson("api/confirm", { id: payload.id, approved });
    } catch (error) {
      addNote(entry, "ответ не отправлен: " + error.message);
      yes.disabled = false;
      no.disabled = false;
    }
  };
  yes.addEventListener("click", () => answer(true));
  no.addEventListener("click", () => answer(false));
  row.appendChild(yes);
  row.appendChild(no);
  card.appendChild(row);
  entry.node.appendChild(card);
  state.confirmations.set(payload.id, card);
  stickToBottom(true);
}

function markConfirmation(event) {
  const card = state.confirmations.get(event.id);
  if (!card) return;
  card.classList.add(event.approved ? "done" : "declined");
  for (const control of card.querySelectorAll(".row button")) control.remove();
  const verdict = event.timeout
    ? "ответа не было — отказ по таймауту"
    : (event.approved ? "подтверждено" : "отклонено");
  card.appendChild(textNode("div", "task-line", verdict));
}

async function loadConfirmations() {
  try {
    const payload = await request("api/confirmations");
    for (const pending of payload.pending || []) {
      if (!state.confirmations.has(pending.id)) confirmationCard(pending);
    }
  } catch (error) {
    addNote(message("assistant"), "запросы подтверждения не получены: " + error.message);
  }
}

// --- состояние, настройки и отчёты -----------------------------------------------------------

function fillSettings(status) {
  const settings = status.settings || {};
  el("set-format").value = settings.format || "free";
  el("set-strategy").value = settings.context_strategy || "summary";
  el("set-words").value = settings.max_words ?? "";
  el("set-list").value = settings.list_limit ?? "";
  el("set-temperature").value = settings.temperature ?? "";
  el("set-compress").value = settings.compress_after ?? "";
  el("set-tokens").value = settings.max_session_tokens ?? "";
  const select = el("set-model");
  clear(select);
  for (const name of status.models || [status.model]) {
    select.appendChild(textNode("option", "", name));
    select.lastChild.value = name;
  }
  select.value = status.model;
}

function renderStatus(status) {
  const settings = status.settings || {};
  el("status").textContent = [
    status.domain.title + " (" + status.domain.id + ")",
    "Модель: " + status.model,
    "Формат: " + (settings.format || "?"),
    "Стратегия: " + (settings.context_strategy || "?"),
    "Слов: " + settings.max_words,
  ].join(" · ");
  const flags = [];
  flags.push(status.docs_enabled ? "документация вкл" : "документация выкл");
  flags.push(status.code_enabled ? "код вкл" : "код выкл");
  if (status.mcp) flags.push(status.mcp);
  el("status-note").textContent = status.root + " · " + flags.join(" · ");
}

async function loadStatus() {
  try {
    const status = await request("api/status");
    renderStatus(status);
    fillSettings(status);
  } catch (error) {
    el("status").textContent = "Состояние не получено: " + error.message;
  }
}

async function loadReport(kind) {
  const host = el("report");
  host.className = "lines";
  host.textContent = "Загрузка отчёта «" + kind + "»…";
  try {
    const payload = await request("api/reports/" + kind);
    const lines = payload.lines || [];
    host.textContent = lines.length ? lines.join("\n") : "Отчёт пуст.";
    if (kind === "task") renderTask(payload.data);
  } catch (error) {
    host.textContent = "Отчёт не получен: " + error.message;
  }
}

// --- события ---------------------------------------------------------------------------------

function handleEvent(event) {
  switch (event.type) {
    case "phase":
      setPhase(PHASE_LABELS[event.phase] || event.phase);
      break;
    case "journal":
      addJournal(event.text);
      break;
    case "domain":
      loadStatus();
      break;
    case "confirmation_request":
      confirmationCard(event);
      break;
    case "confirmation_result":
      markConfirmation(event);
      break;
    case "task":
      renderTask(event.data, event.running);
      break;
    case "task_edits_request":
      editsCard(event);
      break;
    case "settings":
      loadStatus();
      break;
    case "answer":
    case "command":
      // Ответ и вывод команды рендерит тот, кто их запросил: поток лишь повторяет результат,
      // и второй разрисовки сообщения быть не должно.
      break;
    default:
      break;
  }
}

function connectEvents() {
  const stream = new EventSource("api/events");
  stream.addEventListener("message", (message) => {
    try {
      handleEvent(JSON.parse(message.data));
    } catch (error) {
      addJournal("нераспознанное событие потока");
    }
  });
  stream.addEventListener("error", () => {
    /* переподключение делает сам EventSource: строка в журнале не нужна */
  });
}

// --- композер и привязки ---------------------------------------------------------------------

function autoGrow(area) {
  area.style.height = "auto";
  area.style.height = Math.min(area.scrollHeight, 180) + "px";
}

function compose(text) {
  const value = text.trim();
  if (!value || state.busy) return;
  if (value.startsWith("/")) runCommand(value);
  else ask(value);
}

function bind() {
  el("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const area = el("question");
    const text = area.value;
    area.value = "";
    autoGrow(area);
    compose(text);
  });

  el("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el("composer").requestSubmit();
    }
  });

  el("question").addEventListener("input", () => autoGrow(el("question")));

  const feed = el("feed");
  feed.addEventListener("scroll", () => {
    state.stick = atBottom();
    el("jump").classList.toggle("hidden", state.stick);
  });

  el("jump-button").addEventListener("click", () => {
    state.stick = true;
    el("jump").classList.add("hidden");
    stickToBottom(true);
  });

  el("sidebar-open").addEventListener("click", () => document.body.classList.remove("sidebar-hidden"));
  el("sidebar-close").addEventListener("click", () => document.body.classList.add("sidebar-hidden"));

  el("new-session").addEventListener("click", async () => {
    await runCommand("/clear");
    clear(feed);
    addNote(message("assistant"), "новая сессия: диалог очищен, память и профиль сохранены");
  });

  el("settings-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const note = el("settings-note");
    note.className = "note";
    note.textContent = "Применяю…";
    const number = (id) => {
      const raw = el(id).value.trim();
      return raw === "" ? null : Number(raw);
    };
    const body = {
      format: el("set-format").value,
      context_strategy: el("set-strategy").value,
      model: el("set-model").value,
      max_words: number("set-words"),
      list_limit: number("set-list"),
      temperature: number("set-temperature"),
      compress_after: number("set-compress"),
      max_session_tokens: number("set-tokens"),
    };
    for (const key of Object.keys(body)) {
      if (body[key] === null) delete body[key];
    }
    try {
      const payload = await postJson("api/settings", body);
      note.className = "note ok";
      note.textContent = "Настройки применены";
      renderStatus({ ...(await request("api/status")), ...payload, settings: payload.settings });
    } catch (error) {
      note.className = "note error";
      note.textContent = "Не применено: " + error.message;
      await loadStatus();
    }
  });

  document.querySelectorAll("button[data-report]").forEach((node) => {
    node.addEventListener("click", () => loadReport(node.dataset.report));
  });

  document.querySelectorAll("button[data-task]").forEach((node) => {
    node.addEventListener("click", async () => {
      const action = node.dataset.task;
      if (action === "run") {
        await runCommand("/task run");
        return;
      }
      try {
        const payload = await postJson("api/task", { action });
        addNote(lastAssistant() || message("assistant"),
          "прогон: " + action + " — " + (payload.accepted ? "принято" : "не принято"));
      } catch (error) {
        addNote(lastAssistant() || message("assistant"), "прогон: " + action + " — " + error.message);
      }
    });
  });
}

bind();
connectEvents();
loadStatus();
loadConfirmations();
loadReport("task");
autoGrow(el("question"));
