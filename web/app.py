"""Тонкий web-фронтенд над фасадом сессии (`core.session.AssistantSession`).

Слой знает о сессии ровно столько, сколько знает терминальный интерфейс: вызывает
`ask`/`run_command`, подписывается на события (`PhaseChanged`, `JournalLine`, `AnswerReady`,
`DomainSelected`) и печатает снимки (`docs_report`, `code_report`, `tool_flow_report`,
`task_state`, `ops_report`, `memory_report`, `profile_report`, `invariants_report`,
`context_report`). Внутренности агента он не читает — это та же граница изоляции, что и у TUI.

Подтверждения (запись патча задачи и необратимые шаги конвейера операций) приходят из браузера:
фасаду передаются обработчики, которые публикуют событие `confirmation_request` и ждут ответа
`POST /api/confirm`. Пока ответа нет, патч не применяется и шаг не выполняется — как в TUI, где
подтверждение читается с клавиатуры. Ожидание держит блокировку сессии (её берёт сама операция),
поэтому до ответа остальные операции и часть отчётов ждут её освобождения: браузер приходит к
решению по событию SSE, а не опросом. По истечении таймаута ожидания (`FFAI_WEB_CONFIRM_TIMEOUT`,
по умолчанию 600 с) решение считается отказом, чтобы прогон не завис навсегда.

Запуск: `python -m web --repo <каталог> [--port 8000]`; точка входа приложения подключает тот же
`serve()` при флаге `--web`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import config
from core.answer_settings import AnswerFormat, AnswerSettingsError, ContextStrategy
from core.api_client import APIError
from core.session import AnswerReady, AssistantSession, DomainSelected, JournalLine, PhaseChanged

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Таймаут ожидания подтверждения из браузера: без него прогон задачи висел бы на закрытой вкладке.
CONFIRM_TIMEOUT_ENV = "FFAI_WEB_CONFIRM_TIMEOUT"
DEFAULT_CONFIRM_TIMEOUT = 600.0
# Пульс SSE: молчащий поток рвут прокси, а комментарий-пульс держит соединение живым.
SSE_POLL_SECONDS = 15.0

REPORT_KINDS = (
    "docs",
    "code",
    "tool",
    "task",
    "ops",
    "memory",
    "profile",
    "invariants",
    "context",
    "usage",
)

TASK_ACTIONS = ("pause", "resume", "edits")


# --- сериализация ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Снимок фасада → JSON: dataclass, enum, Path, кортежи и словари раскрываются рекурсивно.

    Снимки — это данные ядра (в том числе чужие тексты), поэтому неизвестный объект превращается
    в строку, а не роняет ответ целиком.
    """
    if isinstance(value, float):
        # NaN и бесконечность не бывают валидным JSON: снимок с такой оценкой отдаётся как null.
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value.value if isinstance(value, Enum) else value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return str(value)


def _lines(value: Iterable[str]) -> List[str]:
    return [str(item) for item in value]


# --- снимки отчётов -------------------------------------------------------------------------


def task_snapshot(session: AssistantSession) -> Dict[str, Any]:
    """Снимок очереди задач: этап, план с отметками и текст каждого патча.

    Текст патча — то, что показывает панель задачи: интерфейс не читает ни `task.json`, ни
    внутренности конвейера, он печатает то, что положил в состояние сам конвейер.
    """
    state = session.task_state
    tasks: List[Dict[str, Any]] = []
    for task in state.tasks:
        plan: List[Dict[str, Any]] = []
        for index, item in enumerate(task.plan, start=1):
            patch = task.patch_for(index)
            plan.append(
                {
                    "index": index,
                    "item": item,
                    "applied": bool(patch is not None and patch.applied),
                    "reason": patch.reason if patch is not None else "",
                    "summary": patch.summary if patch is not None else "",
                    "patch": patch.text if patch is not None else "",
                }
            )
        tasks.append(
            {
                "number": task.number,
                "goal": task.goal,
                "stage": task.stage.value,
                "title": task.title,
                "status": task.status,
                "current_step": task.current_step,
                "expected_action": task.expected_action,
                "awaiting_edits": task.awaiting_edits,
                "plan": plan,
                "issues": [issue.line() for issue in task.issues],
                "transitions": [entry.line() for entry in task.transitions],
                "result_path": task.result_path,
                "fail_reason": task.fail_reason,
            }
        )
    current = state.current
    return {
        "tasks": tasks,
        "active": state.active,
        "paused": state.paused,
        "unfinished": len(state.unfinished()),
        "current": current.number if current is not None else None,
    }


def citations_payload(check: Any) -> Optional[Dict[str, Any]]:
    """Проверка ссылок: поля снимка плюс вычисленный вердикт и готовая строка для экрана.

    `confirmed` — производное свойство `CitationsCheck`, а не поле: без него интерфейс не отличил
    бы подтверждённый ответ от «ссылок не было», и строка собирается здесь, чтобы формулировка
    не разошлась между интерфейсами.
    """
    if check is None:
        return None
    data = to_jsonable(check)
    data["confirmed"] = bool(getattr(check, "confirmed", False))
    if getattr(check, "no_context", False):
        data["line"] = "📚 Контекста не было: ответ без источников."
    elif getattr(check, "replaced", False):
        data["line"] = "📚 Ссылки не подтверждены: ответ заменён — источников нет."
    elif getattr(check, "retried", False):
        data["line"] = "📚 Ответ не подтверждён — был повторный запрос."
    elif getattr(check, "opted_out", False):
        data["line"] = "📚 Ответ без ссылки: документация к вопросу не относится."
    elif data["confirmed"]:
        data["line"] = "📚 Ссылки подтверждены: ответ называет источник и цитирует фрагмент."
    else:
        data["line"] = ""
    return data


def _ops_payload(session: AssistantSession) -> Dict[str, Any]:
    """Снимок конвейера операций: домен без правил — это отсутствие конвейера, а не ошибка."""
    pipeline = session.ops_report()
    if pipeline is None:
        return {
            "available": False,
            "lines": ["Домен не объявляет конвейер операций."],
            "state": None,
        }
    return {
        "available": True,
        "lines": _lines(pipeline.lines()),
        "state": to_jsonable(getattr(pipeline, "state", None)),
    }


def _sources_payload(session: AssistantSession) -> List[Dict[str, Any]]:
    """Источники последнего ответа: фрагменты документации и кода с их идентификаторами.

    Идентификатор фрагмента кода уже содержит `путь:L10-L40`, идентификатор документации —
    chunk_id; цитата в ответе проверяется именно по ним, поэтому они и показываются.
    """
    sources: List[Dict[str, Any]] = []
    docs = session.docs_report()
    for fragment in getattr(docs, "fragments", ()) or ():
        title = str(getattr(fragment, "title", "") or "")
        section = str(getattr(fragment, "section", "") or "")
        sources.append(
            {
                "kind": "docs",
                "identifier": str(getattr(fragment, "identifier", "")),
                "path": str(getattr(fragment, "source", "")),
                "title": title,
                "section": section,
                "line": " — ".join(part for part in (title, section) if part),
                "text": str(getattr(fragment, "text", "")),
                "truncated": bool(getattr(fragment, "truncated", False)),
            }
        )
    code = session.code_report()
    for fragment in getattr(code, "fragments", ()) or ():
        identifier = str(getattr(fragment, "identifier", ""))
        path = str(getattr(fragment, "source", ""))
        sources.append(
            {
                "kind": "code",
                "identifier": identifier,
                "path": path,
                "title": "",
                "section": "",
                "line": f"{identifier} — {path}",
                "text": str(getattr(fragment, "text", "")),
                "truncated": bool(getattr(fragment, "truncated", False)),
            }
        )
    return sources


def report_payload(session: AssistantSession, kind: str) -> Dict[str, Any]:
    """Отчёт по имени: строки для печати плюс структурированный снимок. Модель не вызывается."""
    if kind not in REPORT_KINDS:
        raise HTTPException(
            status_code=404,
            detail=f"Неизвестный отчёт «{kind}»; доступны: {', '.join(REPORT_KINDS)}",
        )
    if kind == "docs":
        return {
            "kind": kind,
            "lines": _lines(session.docs_lines()),
            "data": to_jsonable(session.docs_report()),
            "citations": citations_payload(session.last_citations),
        }
    if kind == "code":
        return {
            "kind": kind,
            "lines": _lines(session.code_lines()),
            "data": to_jsonable(session.code_report()),
        }
    if kind == "tool":
        return {
            "kind": kind,
            "lines": _lines(session.tool_flow_lines()),
            "data": to_jsonable(session.tool_flow_report()),
        }
    if kind == "task":
        return {
            "kind": kind,
            "lines": _lines(session.task_lines()),
            "data": task_snapshot(session),
        }
    if kind == "ops":
        payload = _ops_payload(session)
        return {"kind": kind, "lines": payload["lines"], "data": payload}
    if kind == "memory":
        return {
            "kind": kind,
            "lines": _lines(session.memory_lines()),
            "data": to_jsonable(session.memory_report()),
        }
    if kind == "profile":
        return {
            "kind": kind,
            "lines": _lines(session.profile_lines()),
            "data": to_jsonable(session.profile_report()),
        }
    if kind == "invariants":
        return {
            "kind": kind,
            "lines": _lines(session.invariants_lines()),
            "data": to_jsonable(session.invariants_report()),
        }
    if kind == "context":
        return {
            "kind": kind,
            "lines": _lines(session.context_lines()),
            "data": to_jsonable(session.context_report()),
        }
    return {
        "kind": kind,
        "lines": _lines(session.usage_lines()),
        "data": {
            "last_result": to_jsonable(session.last_result),
            "session": to_jsonable(session.session_usage),
        },
    }


# --- события --------------------------------------------------------------------------------


def session_event_payload(event: object) -> Dict[str, Any]:
    """Событие фасада → событие SSE. Фазы отдаются кодом: подписи живут во фронтенде."""
    if isinstance(event, PhaseChanged):
        return {"type": "phase", "phase": event.phase.value}
    if isinstance(event, JournalLine):
        return {"type": "journal", "text": event.text}
    if isinstance(event, AnswerReady):
        return {
            "type": "answer",
            "question": event.question,
            "answer": event.answer,
            "meta": to_jsonable(event.meta),
        }
    if isinstance(event, DomainSelected):
        return {
            "type": "domain",
            "id": event.selection.domain.id,
            "title": event.selection.domain.title,
            "source": event.selection.source,
        }
    return {"type": "event", "name": type(event).__name__}


class EventBus:
    """Вещание событий подписчикам SSE: очередь на каждого клиента, публикация без блокировок.

    Медленный клиент теряет свои события (очередь переполнена), но никогда не задерживает
    операцию: подписчик — наблюдатель.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._lock = threading.Lock()
        self._clients: List["queue.Queue[Optional[Dict[str, Any]]]"] = []
        self._maxsize = maxsize

    def subscribe(self) -> "queue.Queue[Optional[Dict[str, Any]]]":
        client: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._clients.append(client)
        return client

    def unsubscribe(self, client: "queue.Queue[Optional[Dict[str, Any]]]") -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def publish(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("at", time.time())
        with self._lock:
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.put_nowait(payload)
            except queue.Full:
                continue

    def close(self) -> None:
        """Закрывает все потоки: SSE-генератор завершается, а не ждёт пульса."""
        with self._lock:
            clients = tuple(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.put_nowait(None)
            except queue.Full:
                pass


def sse_stream(
    bus: EventBus,
    client: "queue.Queue[Optional[Dict[str, Any]]]",
    poll: float = SSE_POLL_SECONDS,
) -> Iterator[str]:
    """SSE-кадры: событие — `data: {...}`, тишина — комментарий-пульс."""
    try:
        yield ": ff.ai\n\n"
        while True:
            try:
                event = client.get(timeout=poll)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    finally:
        bus.unsubscribe(client)


# --- подтверждения --------------------------------------------------------------------------


class _Pending:
    """Ожидаемый ответ браузера: идентификатор, суть запроса и событие его решения."""

    def __init__(self, request_id: str, kind: str, summary: str, detail: str) -> None:
        self.id = request_id
        self.kind = kind
        self.summary = summary
        self.detail = detail
        self.answered = threading.Event()
        self.approved = False

    def payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "detail": self.detail,
        }


class ConfirmationBroker:
    """Мост между блокирующим подтверждением фасада и ответом браузера.

    Фасад спрашивает синхронно (`bool`), браузер отвечает асинхронно (`POST /api/confirm`),
    поэтому запрос публикуется событием, а вызывающий поток ждёт решения. Не дождавшись,
    считаем отказом: необратимый шаг без ответа человека не выполняется.
    """

    def __init__(self, bus: EventBus, timeout: float) -> None:
        self._bus = bus
        self._timeout = timeout
        self._lock = threading.Lock()
        self._pending: Dict[str, _Pending] = {}

    @property
    def timeout(self) -> float:
        return self._timeout

    def request(self, kind: str, summary: str, detail: str) -> bool:
        pending = _Pending(uuid.uuid4().hex, kind, summary, detail)
        with self._lock:
            self._pending[pending.id] = pending
        self._bus.publish({"type": "confirmation_request", **pending.payload()})
        answered = pending.answered.wait(self._timeout if self._timeout > 0 else None)
        with self._lock:
            self._pending.pop(pending.id, None)
        approved = bool(answered and pending.approved)
        self._bus.publish(
            {
                "type": "confirmation_result",
                "id": pending.id,
                "kind": kind,
                "approved": approved,
                "timeout": not answered,
            }
        )
        return approved

    def resolve(self, request_id: str, approved: bool) -> bool:
        with self._lock:
            pending = self._pending.get(request_id)
        if pending is None:
            return False
        pending.approved = bool(approved)
        pending.answered.set()
        return True

    def pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [item.payload() for item in self._pending.values()]

    # Обработчики, которые регистрирует фасад: подписи — его контракт.
    def patch_confirmation(self, summary: str, patch_text: str) -> bool:
        return self.request("patch", summary, patch_text)

    def ops_confirmation(self, step: str, description: str) -> bool:
        return self.request("ops", step, description)


def run_process(argv: Sequence[str], cwd: Path, timeout: float) -> Tuple[int, str, float]:
    """Исполнитель процессов конвейера операций: списком аргументов, без оболочки.

    Белый список команд живёт в домене; интерфейс только запускает то, что ему назвали, — тот же
    раннер передаёт фасаду терминальный интерфейс.
    """
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"превышено время ожидания ({timeout} с)", time.monotonic() - started
    except OSError as error:
        return 127, str(error), time.monotonic() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output, time.monotonic() - started


# --- прогон задачи --------------------------------------------------------------------------


class TaskRunner:
    """Прогон задачи в фоновом потоке: одна операция за шаг, события — в шину.

    Терминальный интерфейс ведёт этот цикл сам (панель плюс опрос клавиши паузы); браузеру нужен
    фоновый исполнитель, потому что HTTP-ответ не может ждать весь прогон. Пауза, как и в TUI,
    берётся на границе операции: текущий запрос завершается, состояние уже на диске.
    """

    def __init__(self, session: AssistantSession, bus: EventBus) -> None:
        self._session = session
        self._bus = bus
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._paused = threading.Event()
        self._stopped = threading.Event()
        self._edits = threading.Event()
        self._edits_text = ""

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self._paused.clear()
            self._stopped.clear()
            self._edits.clear()
            self._edits_text = ""
            self._thread = threading.Thread(
                target=self._loop, name="ff-ai-web-task", daemon=True
            )
            self._thread.start()
        return True

    def pause(self) -> None:
        self._paused.set()
        # Разбудить ожидание правок: решения по плану не будет — прогон остановится на границе.
        self._edits.set()

    def stop(self) -> None:
        self._stopped.set()
        self._edits.set()

    def provide_edits(self, text: str) -> bool:
        if not self.running:
            return False
        self._edits_text = text
        self._edits.set()
        return True

    def _publish_state(self, running: bool) -> None:
        try:
            snapshot = task_snapshot(self._session)
        except Exception:  # noqa: BLE001 - снимок не должен ронять фоновый поток
            return
        self._bus.publish({"type": "task", "running": running, "data": snapshot})

    def _announce_pause(self) -> None:
        """Пауза — это решение интерфейса: фасаду она передаётся флагом, пользователю — строкой."""
        self._session.task_set_paused(True)
        self._bus.publish(
            {
                "type": "journal",
                "text": "Прогон остановлен на паузе: состояние сохранено, "
                "продолжить — действие resume",
            }
        )

    def _loop(self) -> None:
        self._publish_state(running=True)
        try:
            self._session.task_set_paused(False)
            while True:
                if self._stopped.is_set() or self._paused.is_set():
                    self._announce_pause()
                    break
                report = self._session.task_step()
                if report is None:
                    break
                for note in report.notes:
                    self._bus.publish({"type": "journal", "text": note})
                self._publish_state(running=True)
                if report.awaiting_edits:
                    self._bus.publish(
                        {
                            "type": "task_edits_request",
                            "label": report.label,
                            "data": task_snapshot(self._session),
                        }
                    )
                    text = self._wait_for_edits()
                    if text is None:
                        self._announce_pause()
                        break
                    self._session.task_answer_edits(text)
                    continue
                if report.finished:
                    break
        except Exception as error:  # noqa: BLE001 - сбой прогона не должен ронять сервер
            self._bus.publish({"type": "journal", "text": f"Прогон задачи остановлен: {error}"})
        finally:
            self._publish_state(running=False)

    def _wait_for_edits(self) -> Optional[str]:
        while True:
            if self._edits.wait(0.2):
                self._edits.clear()
                if self._paused.is_set() or self._stopped.is_set():
                    return None
                text = self._edits_text
                self._edits_text = ""
                return text
            if self._paused.is_set() or self._stopped.is_set():
                return None


# --- тела запросов --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = ""


class CommandRequest(BaseModel):
    text: str = ""


class ConfirmRequest(BaseModel):
    id: str
    approved: bool = False


class TaskActionRequest(BaseModel):
    action: str
    text: str = ""


class SettingsRequest(BaseModel):
    """Частичный набор полей настроек: приходит только то, что пользователь действительно изменил."""

    format: Optional[str] = None
    context_strategy: Optional[str] = None
    max_words: Optional[int] = None
    list_limit: Optional[int] = None
    temperature: Optional[float] = None
    compress_after: Optional[int] = None
    max_session_tokens: Optional[int] = None
    model: Optional[str] = None


# --- приложение -----------------------------------------------------------------------------


def _lifespan(bus: EventBus, runner: "TaskRunner"):
    """Завершение работы: прогон останавливается, подписчики SSE и подписка на сессию закрываются."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            runner.stop()
            bus.close()
            unsubscribe = getattr(app.state, "session_unsubscribe", None)
            if callable(unsubscribe):
                unsubscribe()

    return lifespan


def create_app(
    session: AssistantSession,
    base_url: str = "",
    confirmation_timeout: Optional[float] = None,
) -> FastAPI:
    """Фабрика приложения над готовой сессией.

    `base_url` — префикс, под которым приложение живёт (обратный прокси или подкаталог);
    по умолчанию пустой, и все адреса оказываются в корне. Своих решений слой не принимает:
    сессию ему передают собранной, подтверждения и исполнитель процессов регистрируются здесь.
    """
    prefix = base_url.rstrip("/")
    bus = EventBus()
    timeout = (
        confirmation_timeout
        if confirmation_timeout is not None
        else float(os.getenv(CONFIRM_TIMEOUT_ENV, str(DEFAULT_CONFIRM_TIMEOUT)))
    )
    confirmations = ConfirmationBroker(bus, timeout)
    runner = TaskRunner(session, bus)

    app = FastAPI(title="ff.ai web", version="0.1", lifespan=_lifespan(bus, runner))
    app.state.session = session
    app.state.bus = bus
    app.state.confirmations = confirmations
    app.state.runner = runner

    # События фасада уходят в SSE: подписка живёт столько же, сколько приложение.
    def _forward(event: object) -> None:
        bus.publish(session_event_payload(event))

    app.state.session_unsubscribe = session.subscribe(_forward)
    # Подтверждения и запуск процессов — то, что умеет только интерфейс.
    session.set_patch_confirmation(confirmations.patch_confirmation)
    session.set_ops_confirmation(confirmations.ops_confirmation)
    session.set_ops_runner(run_process)

    # --- страница и статика ---------------------------------------------------------------

    @app.get(f"{prefix}/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    app.mount(f"{prefix}/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --- состояние ------------------------------------------------------------------------

    @app.get(f"{prefix}/api/status")
    def api_status() -> Dict[str, Any]:
        return {
            "domain": {"id": session.domain.id, "title": session.domain.title},
            "root": str(session.root),
            "model": session.model,
            "models": list(config.AVAILABLE_MODELS),
            "settings": to_jsonable(session.settings),
            "docs_enabled": session.docs_enabled,
            "code_enabled": session.code_enabled,
            "mcp": session.mcp_summary(),
            "task_running": runner.running,
            "task_paused": session.task_paused,
            "exit_requested": session.exit_requested,
        }

    # --- операции -------------------------------------------------------------------------

    @app.post(f"{prefix}/api/ask")
    def api_ask(payload: AskRequest) -> Dict[str, Any]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Нужен непустой вопрос")
        journal: List[str] = []

        def collect(event: object) -> None:
            if isinstance(event, JournalLine):
                journal.append(event.text)

        unsubscribe = session.subscribe(collect)
        try:
            answer = session.ask(question)
        except APIError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            unsubscribe()
        return {
            "question": answer.question,
            "answer": answer.text,
            "meta": to_jsonable(answer.meta),
            "journal": journal,
            "sources": _sources_payload(session),
            "citations": citations_payload(session.last_citations),
        }

    @app.post(f"{prefix}/api/command")
    def api_command(payload: CommandRequest) -> Dict[str, Any]:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Нужна непустая команда")
        try:
            result = session.run_command(text)
        except APIError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        body = {
            "command": result.command,
            "lines": _lines(result.lines),
            "exit_requested": result.exit_requested,
            "unknown": result.unknown,
            "domain_changed": result.domain_changed,
            "task_run": result.task_run,
            "profile_setup": result.profile_setup,
        }
        # `/task run` в TUI ведёт интерфейс; здесь прогон ведёт фоновый исполнитель.
        body["task_run_started"] = runner.start() if result.task_run else False
        bus.publish({"type": "command", "data": body})
        return body

    @app.post(f"{prefix}/api/settings")
    def api_settings(payload: SettingsRequest) -> Dict[str, Any]:
        """Настройки и модель из браузера: частичный набор полей, отказ без изменения действующих.

        Порядок важен: сначала проверяется модель (она не часть `AnswerSettings`), и только потом
        собирается новый объект настроек. Недопустимое значение отдаёт 400, а сессия остаётся с
        прежними настройками — та же семантика, что у экрана настроек терминала.
        """
        model = payload.model.strip() if payload.model is not None else None
        if model is not None and model not in config.AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестная модель «{model}»; доступны: {', '.join(config.AVAILABLE_MODELS)}",
            )
        settings = session.settings
        try:
            if payload.format is not None:
                settings = settings.with_format(AnswerFormat(payload.format))
            if payload.context_strategy is not None:
                settings = settings.with_context_strategy(ContextStrategy(payload.context_strategy))
            if payload.max_words is not None:
                settings = settings.with_max_words(payload.max_words)
            if payload.list_limit is not None:
                settings = settings.with_list_limit(payload.list_limit)
            if payload.temperature is not None:
                settings = settings.with_temperature(payload.temperature)
            if payload.compress_after is not None:
                settings = settings.with_compress_after(payload.compress_after)
            if payload.max_session_tokens is not None:
                settings = settings.with_max_session_tokens(payload.max_session_tokens)
        except (AnswerSettingsError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        session.settings = settings
        if model is not None:
            session.model = model
        body = {"settings": to_jsonable(session.settings), "model": session.model}
        bus.publish({"type": "settings", "data": body})
        return body

    @app.post(f"{prefix}/api/task")
    def api_task(payload: TaskActionRequest) -> Dict[str, Any]:
        action = payload.action.strip().lower()
        if action not in TASK_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестное действие «{payload.action}»; доступны: {', '.join(TASK_ACTIONS)}",
            )
        if action == "pause":
            runner.pause()
            return {"action": action, "running": runner.running, "accepted": True}
        if action == "resume":
            return {"action": action, "running": runner.running, "accepted": runner.start()}
        accepted = runner.provide_edits(payload.text)
        return {"action": action, "running": runner.running, "accepted": accepted}

    # --- отчёты ---------------------------------------------------------------------------

    @app.get(f"{prefix}/api/reports/{{kind}}")
    def api_report(kind: str) -> Dict[str, Any]:
        return report_payload(session, kind)

    # --- события --------------------------------------------------------------------------

    @app.get(f"{prefix}/api/events")
    def api_events() -> StreamingResponse:
        client = bus.subscribe()
        return StreamingResponse(
            # Пульс читается на каждом запросе: тест может сжать его, не меняя поведение сервера.
            sse_stream(bus, client, poll=SSE_POLL_SECONDS),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- подтверждения --------------------------------------------------------------------

    @app.get(f"{prefix}/api/confirmations")
    def api_confirmations() -> Dict[str, Any]:
        return {"pending": confirmations.pending(), "timeout": confirmations.timeout}

    @app.post(f"{prefix}/api/confirm")
    def api_confirm(payload: ConfirmRequest) -> Dict[str, Any]:
        if not confirmations.resolve(payload.id, payload.approved):
            raise HTTPException(
                status_code=404,
                detail="Запрос на подтверждение неизвестен или уже отвечен",
            )
        return {"id": payload.id, "approved": bool(payload.approved)}

    return app


def serve(
    session: AssistantSession,
    host: str = "127.0.0.1",
    port: int = 8000,
    base_url: str = "",
    log_level: str = "",
) -> None:
    """Запуск приложения сервером: то, что вызывает точка входа при флаге `--web`."""
    import uvicorn

    app = create_app(session, base_url=base_url)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level or os.getenv("FFAI_WEB_LOG_LEVEL", "info"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m web", description="Web-фронтенд ff.ai над фасадом сессии."
    )
    parser.add_argument("--repo", default="", help="целевой репозиторий (по умолчанию текущий каталог)")
    parser.add_argument("--domain", default="", help="пакет домена; по умолчанию — по маркерам репозитория")
    parser.add_argument("--host", default="127.0.0.1", help="адрес прослушивания")
    parser.add_argument("--port", type=int, default=8000, help="порт")
    parser.add_argument("--base-url", default="", help="префикс адресов (подкаталог или прокси)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Сборка сессии и запуск сервера: тот же порядок, что у точки входа приложения."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    from core.domains import DomainError
    from core.repo import RepoRootError, resolve_root

    try:
        root = resolve_root(explicit=args.repo)
    except RepoRootError as error:
        print(f"Ошибка: {error}")
        return 2

    try:
        session = AssistantSession(root=root, domain_id=args.domain or None)
    except DomainError as error:
        print(f"Ошибка домена: {error}")
        return 2

    prefix = args.base_url.rstrip("/")
    print(f"ff.ai web: http://{args.host}:{args.port}{prefix}/", flush=True)
    serve(session, host=args.host, port=args.port, base_url=args.base_url)
    return 0
