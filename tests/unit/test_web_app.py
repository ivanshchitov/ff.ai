"""Web-слой: фабрика приложения, эндпоинты, поток событий и подтверждения — без сети.

Сессия поддельная: тест проверяет транспорт и представление web-слоя (что уходит в HTTP, что
попадает в SSE, кто и как отвечает на подтверждение), а не ядро. Снимки для отчётов берутся
настоящие — `TaskState`, `DocsReport`, `TaskStepReport` из ядра, — иначе тест не заметил бы
расхождения с реальным контрактом фасада.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
from fastapi.testclient import TestClient

from core.agent import ContextReport, RequestPhase
from core.answer_settings import AnswerSettings, ContextStrategy
from core.api_client import APIError, AnswerMeta
from core.aurora_ops import OpsState
from core.citations import CitationsCheck
from core.code_retrieval import CodeFragment, CodeReport
from core.docs_retrieval import DocsFragment, DocsReport
from core.session import (
    Answer,
    AnswerReady,
    CommandResult,
    DomainSelected,
    JournalLine,
    PhaseChanged,
)
from core.task_pipeline import TaskStepReport
from core.task_state import Stage, TaskItem, TaskPatch, TaskState
from web.app import (
    DEFAULT_CONFIRM_TIMEOUT,
    REPORT_KINDS,
    create_app,
    run_process,
    sse_stream,
    to_jsonable,
)

DIFF = """--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,3 @@
 первая строка
 вторая строка
+третья строка
"""
ANSWER_META = AnswerMeta(
    content="Ответ ассистента.",
    model="test-model",
    elapsed_seconds=0.5,
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=15,
    cost_usd=0.0001,
)


def _task_state(*, awaiting_edits: bool = False, applied: bool = False) -> TaskState:
    patch = (
        TaskPatch(index=1, text=DIFF, summary="правка notes.txt", applied=applied)
        if applied
        else None
    )
    task = TaskItem(
        number=1,
        goal="дописать заметки",
        stage=Stage.PLANNING if awaiting_edits else Stage.EXECUTION,
        plan=("правка заметок",),
        awaiting_edits=awaiting_edits,
        patches=(patch,) if patch is not None else (),
    )
    return TaskState(tasks=(task,), active=0)


class FakeOps:
    """Заглушка конвейера операций: снимок состояния и строки отчёта — как у настоящего."""

    def __init__(self) -> None:
        self.state = OpsState(architecture="x86_64", target="x86_64")

    def lines(self):
        return ("Цель: x86_64",)


class FakeSession:
    """Поддельная сессия: записывает вызовы, отдаёт заданные снимки и умеет вести прогон.

    Поверхность повторяет только то, что действительно использует web-слой; расхождение имён
    заметит сам слой — он вызывает эти методы напрямую.
    """

    def __init__(self) -> None:
        self._listeners: List[Any] = []
        self.asked: List[str] = []
        self.commands: List[str] = []
        self.edits: List[str] = []
        self.error: Optional[Exception] = None
        self.answer_text = ANSWER_META.content
        self.answer_meta = ANSWER_META
        self.patch_confirmation = None
        self.ops_confirmation = None
        self.ops_runner = None
        self.patch_applied = False
        self.steps_done = 0
        self.task_state = TaskState()
        self.docs: Optional[DocsReport] = None
        self.code: Optional[CodeReport] = None
        self.tool_flow = None
        self.ops = None
        self.citations = None
        self.domain = SimpleNamespace(id="aurora", title="ОС Аврора")
        self.root = Path("/tmp/ff-ai-web")
        self.model = "test-model"
        self.settings = AnswerSettings()
        self.docs_enabled = False
        self.code_enabled = False
        self.exit_requested = False
        self.task_paused = False
        self.mcp = "MCP: 0 серверов, 0 инструментов"

    # --- события --------------------------------------------------------------------------

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def emit(self, event: object) -> None:
        for listener in list(self._listeners):
            listener(event)

    # --- операции -------------------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        self.asked.append(question)
        if self.error is not None:
            raise self.error
        self.emit(PhaseChanged(RequestPhase.REQUEST))
        self.emit(JournalLine("Факты не обновлены"))
        self.emit(AnswerReady(question=question, answer=self.answer_text, meta=self.answer_meta))
        return Answer(question=question, text=self.answer_text, meta=self.answer_meta)

    def run_command(self, text: str) -> CommandResult:
        self.commands.append(text)
        if self.error is not None:
            raise self.error
        command = text.split(maxsplit=1)[0]
        if command == "/task" and text.split()[1:2] == ["run"]:
            return CommandResult(command="/task", lines=("Запускаю прогон задачи.",), task_run=True)
        return CommandResult(command=command, lines=("Строка отчёта.",))

    # --- подтверждения и процессы ---------------------------------------------------------

    def set_patch_confirmation(self, callback) -> None:
        self.patch_confirmation = callback

    def set_ops_confirmation(self, callback) -> None:
        self.ops_confirmation = callback

    def set_ops_runner(self, runner) -> None:
        self.ops_runner = runner

    # --- прогон задачи --------------------------------------------------------------------

    def task_step(self) -> TaskStepReport:
        self.steps_done += 1
        if self.steps_done == 1:
            self.task_state = _task_state(awaiting_edits=True)
            return TaskStepReport(
                stage=Stage.PLANNING, label="План", notes=("План построен.",), awaiting_edits=True
            )
        if self.steps_done == 2:
            approved = bool(
                self.patch_confirmation and self.patch_confirmation("правка notes.txt", DIFF)
            )
            self.patch_applied = approved
            self.task_state = _task_state(applied=approved)
            note = "Патч применён." if approved else "Патч не применён."
            return TaskStepReport(stage=Stage.EXECUTION, label="Подзадача 1", notes=(note,))
        return TaskStepReport(
            stage=Stage.DONE, label="Итог", notes=("Задача завершена.",), finished=True
        )

    def task_answer_edits(self, text: str) -> None:
        self.edits.append(text)

    def task_set_paused(self, paused: bool) -> None:
        self.task_paused = bool(paused)

    # --- снимки и отчёты ------------------------------------------------------------------

    def docs_report(self):
        return self.docs

    def code_report(self):
        return self.code

    def tool_flow_report(self):
        return self.tool_flow

    def ops_report(self):
        return self.ops

    @property
    def last_citations(self):
        return self.citations

    def memory_report(self):
        return {"exchanges": 2, "working": {"цель": "дописать заметки"}, "long_term": (), "rules": ()}

    def profile_report(self):
        return {"active": "", "sections": (), "names": (), "path": ""}

    def invariants_report(self):
        return {"rules": (), "check": None}

    def context_report(self):
        return ContextReport(
            strategy=ContextStrategy.SUMMARY,
            window=10,
            max_session_tokens=20000,
            log_exchanges=2,
            request_turns=2,
            summary_covers=0,
            has_summary=False,
            facts={},
            branch="main",
            branches=(("main", 2),),
            tokens_estimate=120,
        )

    def docs_lines(self):
        return ("Документация: поиска не было.",)

    def code_lines(self):
        return ("Код: поиска не было.",)

    def tool_flow_lines(self):
        return ("Автовызов инструментов: выключен",)

    def task_lines(self):
        return ("Задач нет.",)

    def memory_lines(self):
        return ("Обменов в диалоге: 2",)

    def profile_lines(self):
        return ("Активный профиль: нет",)

    def invariants_lines(self):
        return ("Правил домена: 0",)

    def context_lines(self):
        return ("Стратегия: summary",)

    def usage_lines(self):
        return ("Сессия: запросов — 0",)

    def mcp_summary(self):
        return self.mcp

    @property
    def last_result(self):
        return self.answer_meta

    @property
    def session_usage(self):
        return SimpleNamespace(
            requests=1, prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0001
        )


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def app(session: FakeSession):
    return create_app(session, confirmation_timeout=0.5)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    """Ждёт условие: фоновый прогон завершается через мгновение после своего последнего события."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _events(app) -> "queue.Queue":
    """Очередь подписчика SSE: тест читает те же события, что уходят браузеру."""
    return app.state.bus.subscribe()


def _next_event(events: "queue.Queue", timeout: float = 2.0) -> dict:
    return events.get(timeout=timeout)


def _wait_for(events: "queue.Queue", kind: str, timeout: float = 3.0) -> dict:
    """Ждёт событие нужного типа: в потоке между ним и другими событиями порядок не важен."""
    collected: List[dict] = []
    try:
        while True:
            event = _next_event(events, timeout=timeout)
            collected.append(event)
            if event.get("type") == kind:
                return event
    except queue.Empty:
        raise AssertionError(f"событие {kind} не пришло; получено: {collected}") from None


# --- состояние, статика и префикс -------------------------------------------------------------


def test_status_describes_domain_and_model(client: TestClient):
    payload = client.get("/api/status").json()
    assert payload["domain"] == {"id": "aurora", "title": "ОС Аврора"}
    assert payload["model"] == "test-model"
    assert payload["settings"]["format"] in ("free", "compact", "json")
    assert payload["task_running"] is False
    assert payload["mcp"].startswith("MCP:")


def test_page_and_script_are_served_locally(client: TestClient):
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "ff.ai" in page.text
    # Без внешних CDN: страница обязана собираться из своих файлов.
    assert "http://" not in page.text
    assert "https://" not in page.text
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "EventSource" in script.text


def test_base_url_prefix_moves_every_route(session: FakeSession):
    prefixed = TestClient(create_app(session, base_url="/ffai/", confirmation_timeout=0.5))
    assert prefixed.get("/ffai/api/status").status_code == 200
    assert prefixed.get("/api/status").status_code == 404
    assert prefixed.get("/ffai/").status_code == 200
    assert prefixed.get("/ffai/static/app.js").status_code == 200


# --- вопрос -----------------------------------------------------------------------------------


def test_ask_returns_answer_sources_and_citations(client: TestClient, session: FakeSession):
    session.docs = DocsReport(
        query="точка входа",
        sections=("guide",),
        version="5.1",
        status="ok",
        fragments=(
            DocsFragment(
                identifier="entry-point",
                source="https://example.invalid/entry",
                title="Точка входа",
                section="Сборка",
                version="5.1",
                text="Точка входа объявляется в файле приложения.",
            ),
        ),
    )
    session.code = CodeReport(
        question="где main",
        query="main",
        mode="baseline",
        threshold=0.0,
        before=1,
        after=1,
        database=Path("/tmp/index.sqlite3"),
        status="ok",
        fragments=(
            CodeFragment(
                identifier="src/main.cpp:L10-L20",
                source="src/main.cpp",
                text="int main() { return 0; }",
                start_line=10,
                end_line=20,
            ),
        ),
    )
    session.citations = CitationsCheck()
    payload = client.post("/api/ask", json={"question": "Где точка входа?"}).json()

    assert session.asked == ["Где точка входа?"]
    assert payload["answer"] == ANSWER_META.content
    assert payload["meta"]["total_tokens"] == 15
    identifiers = [item["identifier"] for item in payload["sources"]]
    assert identifiers == ["entry-point", "src/main.cpp:L10-L20"]
    assert payload["sources"][1]["path"] == "src/main.cpp"
    assert payload["citations"]["confirmed"] is True
    assert payload["journal"] == ["Факты не обновлены"]


def test_ask_rejects_empty_question(client: TestClient, session: FakeSession):
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400
    assert session.asked == []


def test_ask_reports_api_error(client: TestClient, session: FakeSession):
    session.error = APIError("Нет пригодного ключа OPENCODE_API_KEY")
    response = client.post("/api/ask", json={"question": "вопрос"})
    assert response.status_code == 502
    assert "OPENCODE_API_KEY" in response.json()["detail"]


# --- команды ----------------------------------------------------------------------------------


def test_command_returns_lines_and_announces_itself(client: TestClient, session: FakeSession, app):
    events = _events(app)
    payload = client.post("/api/command", json={"text": "/context"}).json()
    assert session.commands == ["/context"]
    assert payload["command"] == "/context"
    assert payload["lines"] == ["Строка отчёта."]
    assert payload["task_run"] is False
    announced = _wait_for(events, "command")
    assert announced["data"]["command"] == "/context"


def test_command_rejects_empty_text(client: TestClient):
    assert client.post("/api/command", json={"text": ""}).status_code == 400


# --- отчёты -----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_every_report_kind_answers_without_model_calls(kind: str, client: TestClient, session: FakeSession):
    payload = client.get(f"/api/reports/{kind}").json()
    assert payload["kind"] == kind
    assert payload["lines"]
    assert "data" in payload
    assert session.asked == []
    assert session.commands == []


def test_unknown_report_is_not_found(client: TestClient):
    assert client.get("/api/reports/whatever").status_code == 404


def test_task_report_carries_patch_diffs(client: TestClient, session: FakeSession):
    session.task_state = _task_state(applied=True)
    data = client.get("/api/reports/task").json()["data"]
    assert data["current"] == 1
    assert data["tasks"][0]["goal"] == "дописать заметки"
    entry = data["tasks"][0]["plan"][0]
    assert entry["item"] == "правка заметок"
    assert entry["applied"] is True
    assert entry["summary"] == "правка notes.txt"
    assert entry["patch"] == DIFF


def test_ops_report_without_pipeline_is_absent_not_broken(client: TestClient, session: FakeSession):
    payload = client.get("/api/reports/ops").json()
    assert payload["data"]["available"] is False
    assert payload["data"]["state"] is None

    session.ops = FakeOps()
    payload = client.get("/api/reports/ops").json()
    assert payload["data"]["available"] is True
    assert payload["data"]["state"] == {
        "architecture": "x86_64",
        "target": "x86_64",
        "sdk_version": "",
    }
    assert payload["lines"] == ["Цель: x86_64"]


def test_report_serializer_handles_enums_paths_and_non_finite_floats():
    payload = to_jsonable(
        {
            "stage": Stage.DONE,
            "path": Path("/tmp/x"),
            "bad": float("nan"),
            "items": (1, None, "текст"),
            "pair": SimpleNamespace(score=0.5),
        }
    )
    assert payload["stage"] == "done"
    assert payload["path"] == "/tmp/x"
    assert payload["bad"] is None
    assert payload["items"] == [1, None, "текст"]
    assert isinstance(payload["pair"], str)


# --- события ----------------------------------------------------------------------------------


def test_session_events_reach_the_event_bus(client: TestClient, app):
    events = _events(app)
    client.post("/api/ask", json={"question": "вопрос"})
    kinds = []
    for _ in range(3):
        kinds.append(_next_event(events)["type"])
    assert kinds == ["phase", "journal", "answer"]


def test_sse_stream_frames_events_and_pulses(client: TestClient, app):
    bus = app.state.bus
    client_queue = bus.subscribe()
    stream = sse_stream(bus, client_queue, poll=0.05)
    assert next(stream) == ": ff.ai\n\n"

    bus.publish({"type": "journal", "text": "строка журнала"})
    frame = next(stream)
    assert frame.startswith("data: ")
    assert json.loads(frame[len("data: ") :])["text"] == "строка журнала"

    assert next(stream) == ": ping\n\n"
    bus.close()
    with pytest.raises(StopIteration):
        next(stream)


def test_events_route_answers_with_an_event_stream(app):
    """Поток вживую проверяется сквозным тестом: `TestClient` буферизует ответ целиком."""
    route = next(route for route in app.routes if getattr(route, "path", "") == "/api/events")
    response = route.endpoint()
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    app.state.bus.close()


def test_domain_event_is_serialized(app, session: FakeSession):
    from web.app import session_event_payload

    events = _events(app)
    session.emit(PhaseChanged(RequestPhase.TASK_PLAN))
    session.emit(JournalLine("Патч применён."))
    assert _next_event(events)["phase"] == "task_plan"
    assert _next_event(events)["text"] == "Патч применён."

    assert session_event_payload(PhaseChanged(RequestPhase.TASK_PLAN)) == {
        "type": "phase",
        "phase": "task_plan",
    }
    assert session_event_payload(JournalLine("текст")) == {"type": "journal", "text": "текст"}
    selected = session_event_payload(
        DomainSelected(
            selection=SimpleNamespace(
                domain=SimpleNamespace(id="aurora", title="ОС Аврора"), source="по умолчанию"
            )
        )
    )
    assert selected == {"type": "domain", "id": "aurora", "title": "ОС Аврора", "source": "по умолчанию"}
    assert session_event_payload(object()) == {"type": "event", "name": "object"}


# --- подтверждения ----------------------------------------------------------------------------


def test_patch_confirmation_waits_for_the_browser(client: TestClient, session: FakeSession, app):
    events = _events(app)
    decision: List[bool] = []
    worker = threading.Thread(
        target=lambda: decision.append(session.patch_confirmation("правка notes.txt", DIFF))
    )
    worker.start()

    request = _wait_for(events, "confirmation_request")
    assert request["kind"] == "patch"
    assert request["summary"] == "правка notes.txt"
    assert request["detail"] == DIFF
    assert client.get("/api/confirmations").json()["pending"][0]["id"] == request["id"]

    response = client.post("/api/confirm", json={"id": request["id"], "approved": True})
    assert response.status_code == 200
    worker.join(3)
    assert decision == [True]
    assert client.get("/api/confirmations").json()["pending"] == []
    assert _wait_for(events, "confirmation_result")["approved"] is True


def test_patch_confirmation_refusal_denies_the_patch(client: TestClient, session: FakeSession, app):
    events = _events(app)
    decision: List[bool] = []
    worker = threading.Thread(
        target=lambda: decision.append(session.patch_confirmation("правка notes.txt", DIFF))
    )
    worker.start()
    request = _wait_for(events, "confirmation_request")
    client.post("/api/confirm", json={"id": request["id"], "approved": False})
    worker.join(3)
    assert decision == [False]


def test_unanswered_confirmation_times_out_as_refusal(session: FakeSession):
    app = create_app(session, confirmation_timeout=0.2)
    events = _events(app)
    assert session.patch_confirmation("правка", DIFF) is False
    result = _wait_for(events, "confirmation_result")
    assert result["approved"] is False
    assert result["timeout"] is True


def test_confirm_unknown_request_is_not_found(client: TestClient):
    assert client.post("/api/confirm", json={"id": "нет", "approved": True}).status_code == 404


def test_ops_confirmation_and_runner_are_registered(app, session: FakeSession):
    assert session.ops_runner is run_process
    events = _events(app)
    decision: List[bool] = []
    worker = threading.Thread(
        target=lambda: decision.append(session.ops_confirmation("sign", "подпись пакета"))
    )
    worker.start()
    request = _wait_for(events, "confirmation_request")
    assert request["kind"] == "ops"
    assert request["summary"] == "sign"
    assert request["detail"] == "подпись пакета"
    app.state.confirmations.resolve(request["id"], True)
    worker.join(3)
    assert decision == [True]


def test_run_process_returns_code_output_and_time():
    code, output, seconds = run_process(["python3", "-c", "print('ok')"], Path.cwd(), 30)
    assert code == 0
    assert "ok" in output
    assert seconds >= 0
    code, output, _ = run_process(["definitely-not-a-command-ffai"], Path.cwd(), 30)
    assert code == 127


# --- прогон задачи ----------------------------------------------------------------------------


def test_task_actions_are_validated(client: TestClient):
    assert client.post("/api/task", json={"action": "нечто"}).status_code == 400
    payload = client.post("/api/task", json={"action": "edits", "text": "правки"}).json()
    assert payload == {"action": "edits", "running": False, "accepted": False}


def test_task_run_applies_patch_only_after_confirmation(client: TestClient, session: FakeSession, app):
    events = _events(app)
    payload = client.post("/api/command", json={"text": "/task run"}).json()
    assert payload["task_run"] is True
    assert payload["task_run_started"] is True

    edits_request = _wait_for(events, "task_edits_request")
    assert edits_request["data"]["tasks"][0]["awaiting_edits"] is True
    assert client.post("/api/task", json={"action": "edits", "text": ""}).json()["accepted"] is True

    request = _wait_for(events, "confirmation_request")
    assert session.patch_applied is False, "до подтверждения патч применяться не должен"
    client.post("/api/confirm", json={"id": request["id"], "approved": True})

    finished = _wait_for(events, "task", timeout=3.0)
    while finished.get("running"):
        finished = _wait_for(events, "task", timeout=3.0)
    assert session.edits == [""]
    assert session.patch_applied is True
    assert session.steps_done == 3
    assert session.task_state.tasks[0].patches[0].applied is True
    assert _wait_until(lambda: not app.state.runner.running)


def test_task_run_refusal_keeps_the_repository_untouched(client: TestClient, session: FakeSession, app):
    events = _events(app)
    client.post("/api/command", json={"text": "/task run"})
    _wait_for(events, "task_edits_request")
    client.post("/api/task", json={"action": "edits", "text": "правки приняты"})
    request = _wait_for(events, "confirmation_request")
    client.post("/api/confirm", json={"id": request["id"], "approved": False})
    _wait_for(events, "journal")
    assert session.patch_applied is False
    assert session.edits == ["правки приняты"]


def test_task_pause_stops_the_run_between_operations(client: TestClient, session: FakeSession, app):
    events = _events(app)
    client.post("/api/command", json={"text": "/task run"})
    _wait_for(events, "task_edits_request")
    assert client.post("/api/task", json={"action": "pause"}).json()["accepted"] is True
    paused = _wait_for(events, "journal", timeout=3.0)
    assert "паузе" in paused["text"]
    assert session.task_paused is True
    assert _wait_until(lambda: not app.state.runner.running)
    assert session.steps_done == 1


def test_task_resume_starts_the_runner_again(client: TestClient, session: FakeSession, app):
    assert client.post("/api/task", json={"action": "resume"}).json()["accepted"] is True
    assert _wait_until(lambda: app.state.runner.running)
    app.state.runner.stop()
    assert _wait_until(lambda: not app.state.runner.running)


def test_default_confirmation_timeout_is_generous():
    assert DEFAULT_CONFIRM_TIMEOUT >= 60
