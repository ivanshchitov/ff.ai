"""Фасад сессии: операции, сериализация, события и снимки без обращения к модели."""

import threading
import time
from pathlib import Path

import pytest

from core import config
from core.agent import RequestPhase
from core.answer_settings import AnswerFormat
from core.api_client import APIError, AnswerMeta
from core.history_manager import HistoryManager
from core.session import AnswerReady, AssistantSession, JournalLine, PhaseChanged


class FakeClient:
    """Клиент-заглушка: записывает запросы и умеет блокироваться, чтобы проверить сериализацию."""

    def __init__(
        self,
        answer: str = "Это ответ.",
        started: threading.Event = None,
        release: threading.Event = None,
    ) -> None:
        self.answer = answer
        self.calls = []
        self.started = started
        self.release = release

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model": model,
            }
        )
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(5)
        return AnswerMeta(
            content=self.answer,
            model=model,
            elapsed_seconds=0.5,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    return tmp_path / "repo"


def _session(repo: Path, **kwargs) -> AssistantSession:
    kwargs.setdefault("client", FakeClient())
    return AssistantSession(root=repo, **kwargs)


def test_ask_returns_answer_and_records_history(repo: Path):
    client = FakeClient(answer="Инициализация в src/models.cpp.")
    session = _session(repo, client=client)
    answer = session.ask("где инициализируется модель?")
    assert answer.text == "Инициализация в src/models.cpp."
    assert answer.meta is not None
    assert session.history.count() == 1
    assert session.session_usage.requests == 1


def test_request_shape_is_system_then_user(repo: Path):
    client = FakeClient()
    session = _session(repo, client=client)
    session.ask("вопрос")
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert session.domain.prompt("system") in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "вопрос" in messages[-1]["content"]
    assert client.calls[0]["model"] == session.model
    assert client.calls[0]["max_tokens"] == config.max_tokens_for_words(session.settings.max_words)


def test_second_operation_waits_for_the_first(repo: Path):
    started, release = threading.Event(), threading.Event()
    session = _session(repo, client=FakeClient(started=started, release=release))
    done = []

    asker = threading.Thread(target=lambda: done.append(session.ask("вопрос")))
    asker.start()
    assert started.wait(2)

    reporter = threading.Thread(target=lambda: done.append(session.run_command("/context")))
    reporter.start()
    time.sleep(0.05)
    assert reporter.is_alive(), "вторая операция должна ждать завершения первой"

    release.set()
    asker.join(5)
    reporter.join(5)
    assert not reporter.is_alive()
    assert len(done) == 2


def test_subscribers_receive_phases_and_answer(repo: Path):
    events = []
    session = _session(repo)
    unsubscribe = session.subscribe(events.append)
    session.ask("вопрос")
    unsubscribe()
    kinds = [type(event) for event in events]
    assert PhaseChanged in kinds
    assert AnswerReady in kinds
    phases = [event.phase for event in events if isinstance(event, PhaseChanged)]
    assert RequestPhase.REQUEST in phases


def test_broken_subscriber_does_not_break_the_operation(repo: Path):
    seen = []
    session = _session(repo)
    session.subscribe(lambda event: (_ for _ in ()).throw(RuntimeError("сломанный подписчик")))
    session.subscribe(seen.append)
    answer = session.ask("вопрос")
    assert answer.text
    assert seen, "исправный подписчик обязан получить события"


def test_reports_do_not_touch_the_model(repo: Path):
    client = FakeClient()
    session = _session(repo, client=client)
    session.ask("вопрос")
    before = len(client.calls)
    session.run_command("/context")
    session.run_command("/usage")
    session.run_command("/domain")
    session.run_command("/clear")
    assert len(client.calls) == before
    assert session.session_usage.requests == 0  # /clear обнуляет расход сессии


def test_unknown_command_is_reported_not_raised(repo: Path):
    session = _session(repo)
    result = session.run_command("/nope")
    assert result.unknown
    assert "Неизвестная команда" in result.lines[0]


def test_clear_empties_the_dialogue_only(repo: Path):
    session = _session(repo)
    session.ask("первый вопрос")
    session.ask("второй вопрос")
    assert session.history.count() == 2
    session.run_command("/clear")
    assert session.history.count() == 0
    assert session.context_report().log_exchanges == 0


def test_domain_report_names_selection_and_versions(repo: Path):
    session = _session(repo)
    lines = session.run_command("/domain").lines
    text = "\n".join(lines)
    assert session.domain.id in text
    assert session.domain.platform in text
    assert str(session.root) in text


def test_unknown_domain_switch_is_a_message(repo: Path):
    session = _session(repo)
    result = session.run_command("/domain switch nope")
    assert not result.domain_changed
    assert "nope" in result.lines[0]


def test_context_snapshot_is_a_copy(repo: Path):
    session = _session(repo)
    report = session.context_report()
    report.facts["подложенный"] = "факт"
    assert "подложенный" not in session.context_report().facts


def test_exit_command_marks_the_session(repo: Path):
    session = _session(repo)
    result = session.run_command("/exit")
    assert result.exit_requested
    assert session.exit_requested


def test_ask_without_key_is_an_api_error(repo: Path, monkeypatch):
    """Нет ключа — нет запроса: интерфейс получит ожидаемую ошибку, а не падение."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    session = AssistantSession(root=repo)
    with pytest.raises(APIError) as error:
        session.ask("вопрос")
    assert "OPENCODE_API_KEY" in str(error.value)


def test_client_is_built_from_the_key(monkeypatch, repo: Path):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-probe-key")
    session = AssistantSession(root=repo, client=FakeClient())
    assert session.client is not None
    session.client = None
    session._require_client()
    assert session.client is not None


def test_history_file_appears_in_a_fresh_state_directory(repo: Path, isolated_state: Path):
    """Первый обмен создаёт файл истории: каталог состояния может ещё не существовать."""
    history = HistoryManager()
    assert not history.path.exists()
    session = AssistantSession(root=repo, client=FakeClient(), history=history)
    session.ask("вопрос")
    assert history.path.parent == isolated_state or isolated_state in history.path.parents
    assert history.path.is_file()
    assert history.last_error is None


def test_unwritable_history_does_not_break_the_answer(repo: Path, tmp_path: Path):
    """Сбой записи не отменяет ответ, но и не остаётся незамеченным."""
    blocker = tmp_path / "blocker"
    blocker.write_text("это файл, а не каталог", encoding="utf-8")
    session = AssistantSession(
        root=repo, client=FakeClient(), history=HistoryManager(path=blocker / "history.json")
    )
    answer = session.ask("вопрос")
    assert answer.text
    assert session.history.last_error is not None


def test_journal_lines_are_emitted_for_compression_or_facts_only(repo: Path):
    """Строки журнала — про то, что стратегия сделала перед ответом, и ничего лишнего."""
    events = []
    session = _session(repo)
    session.subscribe(events.append)
    session.ask("обычный вопрос")
    assert [event for event in events if isinstance(event, JournalLine)] == []


def test_settings_and_model_are_session_state(repo: Path):
    client = FakeClient()
    session = _session(repo, client=client)
    session.settings = session.settings.with_format(AnswerFormat.PATCH)
    session.model = config.AVAILABLE_MODELS[1]
    assert session.settings.format is AnswerFormat.PATCH
    assert session.model == config.AVAILABLE_MODELS[1]
    session.ask("вопрос")
    assert "дифф" in client.calls[0]["messages"][0]["content"].lower()
