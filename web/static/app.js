"use strict";

// Фронтенд ff.ai: один файл без сборщика и внешних библиотек.
//
// Слой ничего не решает сам: он показывает то, что отдали эндпоинты (`/api/ask`, `/api/command`,
// `/api/reports/...`) и поток событий `/api/events`. Всё, что приходит с сервера, попадает в DOM
// через textContent: это чужие тексты (ответы модели, фрагменты документации, диффы патчей), и
// разметку из них строить нельзя.

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

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function textNode(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  return node;
}

function setText(node, value) {
  node.textContent = value;
  node.classList.toggle("empty", !value);
}

// --- обращения к серверу ---------------------------------------------------------------------

async function request(path, options) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(detail || "Запрос не выполнен");
  }
  return payload || {};
}

const postJson = (path, body) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

// --- ответ, источники и цитаты ----------------------------------------------------------------

function usageLine(meta) {
  if (!meta) return "";
  const seconds = typeof meta.elapsed_seconds === "number" ? meta.elapsed_seconds.toFixed(2) : "н/д";
  const cost = meta.cost_usd === null || meta.cost_usd === undefined
    ? "неизвестно"
    : "$" + Number(meta.cost_usd).toFixed(6);
  const speed = meta.elapsed_seconds > 0 && meta.completion_tokens > 0
    ? (meta.completion_tokens / meta.elapsed_seconds).toFixed(2) + " ток/сек"
    : "н/д";
  let line = "⏱ " + seconds + "с | Токены: " + meta.prompt_tokens + "+" + meta.completion_tokens +
    "=" + meta.total_tokens + " | Стоимость: " + cost + " | Средняя скорость: " + speed;
  if (meta.finish_reason === "length") {
    line += "\n⚠ ответ мог быть обрезан техническим потолком запроса";
  }
  return line;
}

function citationsLine(check) {
  // Формулировка приходит с сервера: снимок проверки ссылок — источник истины, а не разметка.
  return check && check.line ? check.line : "";
}

function renderSources(sources) {
  const host = el("sources");
  clear(host);
  if (!sources || !sources.length) {
    host.className = "sources empty";
    host.textContent = "Источников не было.";
    return;
  }
  host.className = "sources";
  const list = document.createElement("ul");
  for (const source of sources) {
    const item = document.createElement("li");
    item.appendChild(textNode("span", "ident", source.identifier || "—"));
    item.appendChild(document.createTextNode(" "));
    const where = [source.title, source.section].filter(Boolean).join(" — ");
    if (where) item.appendChild(textNode("span", "where", where));
    if (source.text) {
      const details = document.createElement("details");
      details.appendChild(textNode("summary", "", "фрагмент" + (source.truncated ? " (обрезан)" : "")));
      details.appendChild(textNode("span", "quote", source.text));
      item.appendChild(details);
    }
    list.appendChild(item);
  }
  host.appendChild(list);
}

function renderAnswer(payload) {
  setText(el("answer"), payload.answer || "");
  setText(el("usage"), usageLine(payload.meta));
  renderSources(payload.sources);
  setText(el("citations"), citationsLine(payload.citations));
}

// --- задача и диффы патчей --------------------------------------------------------------------

function renderTask(snapshot) {
  const host = el("task");
  clear(host);
  const tasks = (snapshot && snapshot.tasks) || [];
  if (!tasks.length) {
    host.className = "task empty";
    host.textContent = "Задач нет. Добавить: команда «/task add <цель>».";
    return;
  }
  host.className = "task";
  tasks.forEach((task, position) => {
    const card = document.createElement("div");
    const head = (position === snapshot.active ? "▶ " : "") + "Задача " + task.number +
      " (" + task.title + "): " + task.goal;
    card.appendChild(textNode("div", "goal", head));

    const stages = document.createElement("div");
    stages.className = "stages";
    STAGES.forEach(([id, title], index) => {
      if (index) stages.appendChild(document.createTextNode(" → "));
      stages.appendChild(id === task.stage ? textNode("b", "", title) : textNode("span", "", title));
    });
    card.appendChild(stages);

    card.appendChild(textNode("div", "meta", "Шаг: " + task.current_step + " · Ожидается: " + task.expected_action));
    if (task.status) card.appendChild(textNode("div", "meta", "Состояние: " + task.status));

    if (task.plan && task.plan.length) {
      const list = document.createElement("ul");
      list.className = "plan";
      for (const entry of task.plan) {
        const item = document.createElement("li");
        const mark = textNode("span", entry.applied ? "mark-ok" : "mark-todo", entry.applied ? "[x]" : "[ ]");
        item.appendChild(mark);
        item.appendChild(document.createTextNode(" " + entry.index + ". " + entry.item));
        if (entry.patch) {
          const details = document.createElement("details");
          const state = entry.applied
            ? textNode("span", "applied", " — применён")
            : textNode("span", "skipped", " — " + (entry.reason || "не применён"));
          const summary = document.createElement("summary");
          summary.textContent = entry.summary || ("патч подзадачи " + entry.index);
          summary.appendChild(state);
          details.appendChild(summary);
          details.appendChild(textNode("pre", "", entry.patch));
          item.appendChild(details);
        } else if (entry.reason) {
          item.appendChild(textNode("span", "skipped", " — " + entry.reason));
        }
        list.appendChild(item);
      }
      card.appendChild(list);
    }

    if (task.issues && task.issues.length) {
      const issues = document.createElement("ul");
      issues.className = "issues";
      for (const issue of task.issues) issues.appendChild(textNode("li", "", issue));
      card.appendChild(issues);
    }
    if (task.result_path) card.appendChild(textNode("div", "meta", "Отчёт: " + task.result_path));
    if (task.fail_reason) card.appendChild(textNode("div", "issues", "Причина неудачи: " + task.fail_reason));
    host.appendChild(card);
  });

  el("edits").classList.toggle("hidden", !tasks.some((task) => task.awaiting_edits));
}

// --- подтверждения ----------------------------------------------------------------------------

const resolvedConfirmations = new Set();

function renderConfirmations(pending) {
  const host = el("confirmations");
  const open = pending.filter((item) => !resolvedConfirmations.has(item.id));
  clear(host);
  if (!open.length) {
    host.className = "confirmations empty";
    host.textContent = "Запросов нет.";
    return;
  }
  host.className = "confirmations";
  for (const item of open) {
    const card = document.createElement("div");
    card.className = "confirmation";
    card.dataset.confirm = item.id;
    card.appendChild(textNode("div", "kind", CONFIRMATION_LABELS[item.kind] || item.kind));
    card.appendChild(textNode("div", "", item.summary || ""));
    if (item.detail) card.appendChild(textNode("pre", "", item.detail));
    const actions = document.createElement("div");
    actions.className = "actions";
    const approve = textNode("button", "approve", "Применить");
    const reject = textNode("button", "reject", "Отклонить");
    approve.addEventListener("click", () => answerConfirmation(item.id, true));
    reject.addEventListener("click", () => answerConfirmation(item.id, false));
    actions.appendChild(approve);
    actions.appendChild(reject);
    card.appendChild(actions);
    host.appendChild(card);
  }
}

async function answerConfirmation(id, approved) {
  try {
    await postJson("api/confirm", { id, approved });
    resolvedConfirmations.add(id);
    const card = el("confirmations").querySelector('[data-confirm="' + id + '"]');
    if (card) {
      card.classList.add("resolved");
      const actions = card.querySelector(".actions");
      if (actions) actions.textContent = approved ? "Применено." : "Отклонено.";
    }
    logEvent("подтверждение: " + (approved ? "применено" : "отклонено") + " (" + id + ")");
  } catch (error) {
    logEvent("ошибка подтверждения: " + error.message);
  }
}

// --- журнал событий ---------------------------------------------------------------------------

const eventLines = [];

function logEvent(line) {
  eventLines.push(line);
  while (eventLines.length > 200) eventLines.shift();
  setText(el("events"), eventLines.join("\n"));
}

function handleEvent(event) {
  switch (event.type) {
    case "phase":
      setText(el("phase"), PHASE_LABELS[event.phase] || event.phase);
      logEvent("фаза: " + (PHASE_LABELS[event.phase] || event.phase));
      break;
    case "journal":
      logEvent(event.text);
      break;
    case "answer":
      setText(el("phase"), "");
      logEvent("ответ готов: " + (event.question || ""));
      break;
    case "domain":
      logEvent("домен: " + event.title + " (" + event.source + ")");
      break;
    case "command":
      logEvent("команда " + event.data.command + ": " + (event.data.lines || []).join(" "));
      break;
    case "task":
      renderTask(event.data);
      logEvent("задача: " + (event.running ? "прогон идёт" : "прогон остановлен"));
      break;
    case "task_edits_request":
      el("edits").classList.remove("hidden");
      renderTask(event.data);
      logEvent("нужны правки плана");
      break;
    case "confirmation_request":
      logEvent("запрос подтверждения: " + (event.summary || event.kind));
      renderConfirmations([event]);
      break;
    case "confirmation_result":
      if (event.timeout) logEvent("подтверждение не получено — отказ по таймауту");
      break;
    default:
      logEvent("событие: " + (event.name || event.type));
  }
}

function connectEvents() {
  const stream = new EventSource("api/events");
  stream.addEventListener("message", (message) => {
    try {
      handleEvent(JSON.parse(message.data));
    } catch (error) {
      logEvent("нераспознанное событие: " + message.data);
    }
  });
  stream.addEventListener("error", () => logEvent("поток событий переподключается…"));
}

// --- отчёты и команды -------------------------------------------------------------------------

async function loadReport(kind) {
  try {
    const payload = await request("api/reports/" + kind);
    setText(el("report"), "=== " + kind + " ===\n" + (payload.lines || []).join("\n"));
    if (kind === "task") renderTask(payload.data);
    if (kind === "docs") setText(el("citations"), citationsLine(payload.citations));
  } catch (error) {
    setText(el("report"), "Отчёт не получен: " + error.message);
  }
}

async function loadStatus() {
  try {
    const status = await request("api/status");
    const settings = status.settings || {};
    const parts = [
      "Домен: " + status.domain.title + " (" + status.domain.id + ")",
      "Модель: " + status.model,
      "Формат: " + (settings.format || "?"),
      "Стратегия: " + (settings.context_strategy || "?"),
      "Слов: " + settings.max_words,
    ];
    if (status.mcp) parts.push(status.mcp);
    setText(el("status"), parts.join(" · "));
  } catch (error) {
    setText(el("status"), "Состояние не получено: " + error.message);
  }
}

async function loadConfirmations() {
  try {
    const payload = await request("api/confirmations");
    renderConfirmations(payload.pending || []);
  } catch (error) {
    logEvent("запросы подтверждения не получены: " + error.message);
  }
}

async function runCommand(line) {
  try {
    const payload = await postJson("api/command", { text: line });
    setText(el("command-out"), (payload.lines || []).join("\n") || "Команда выполнена.");
    if (payload.task_run_started) logEvent("прогон задачи запущен");
  } catch (error) {
    setText(el("command-out"), "Ошибка: " + error.message);
  }
}

function bind() {
  el("ask-form").addEventListener("submit", async (formEvent) => {
    formEvent.preventDefault();
    const question = el("question").value.trim();
    if (!question) return;
    el("question").value = "";
    setText(el("phase"), "Отправка вопроса…");
    try {
      renderAnswer(await postJson("api/ask", { question }));
    } catch (error) {
      setText(el("answer"), "Ошибка запроса: " + error.message);
    } finally {
      setText(el("phase"), "");
    }
  });

  el("command-form").addEventListener("submit", (formEvent) => {
    formEvent.preventDefault();
    const line = el("command").value.trim();
    if (!line) return;
    el("command").value = "";
    runCommand(line);
  });

  el("edits-form").addEventListener("submit", async (formEvent) => {
    formEvent.preventDefault();
    const value = el("edits-input").value;
    el("edits-input").value = "";
    try {
      await postJson("api/task", { action: "edits", text: value });
      el("edits").classList.add("hidden");
    } catch (error) {
      logEvent("правки не приняты: " + error.message);
    }
  });

  document.querySelectorAll("button[data-report]").forEach((button) => {
    button.addEventListener("click", () => loadReport(button.dataset.report));
  });

  document.querySelectorAll("button[data-task]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.task;
      if (action === "run") {
        runCommand("/task run");
        return;
      }
      postJson("api/task", { action })
        .then((payload) => logEvent("прогон: " + action + " — " + (payload.accepted ? "принято" : "не принято")))
        .catch((error) => logEvent("прогон: " + action + " — " + error.message));
    });
  });
}

bind();
connectEvents();
loadStatus();
loadConfirmations();
loadReport("task");
