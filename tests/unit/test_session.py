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
from core.mcp_client import MCPError
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


# --- MCP: обход реестра, отчёт, ручной вызов ------------------------------------------------


def test_sweep_fills_the_report(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    connections = session.connect_mcp_servers()
    assert [connection.available for connection in connections] == [True, True]
    assert len(session.mcp_report()) == 2
    assert len(mcp_factory.specs) == 2
    summary = session.mcp_summary()
    assert "MCP: 2/2 серверов" in summary and "инструментов" in summary


def test_sweep_reports_phases(repo: Path, mcp_factory):
    events = []
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.subscribe(events.append)
    session.connect_mcp_servers()
    phases = [event.phase for event in events if isinstance(event, PhaseChanged)]
    assert RequestPhase.MCP_CONNECT in phases


def test_unavailable_server_is_counted_and_does_not_stop_the_sweep(repo: Path):
    from tests.conftest import FakeMCPFactory

    factory = FakeMCPFactory(error="нет связи", error_for=("repo",))
    session = _session(repo, mcp_client_factory=factory)
    session.connect_mcp_servers()
    assert "1/2" in session.mcp_summary()
    assert "1 недоступно" in session.mcp_summary()
    lines = "\n".join(session.mcp_lines())
    assert "недоступен: нет связи" in lines
    assert "fake_echo" in lines  # живой сервер всё равно описан


def test_mcp_reports_do_not_touch_the_model(repo: Path, mcp_factory):
    client = FakeClient()
    session = _session(repo, client=client, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    session.run_command("/mcp")
    session.run_command("/tool")
    session.run_command("/tool call fake_echo message=привет")
    assert client.calls == []


def test_tool_listing_and_manual_call(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    listing = "\n".join(session.tool_lines())
    assert "fake_echo" in listing and "параметры:" in listing

    result = session.call_mcp_tool("fake_echo", {"message": "привет"})
    assert result.arguments == {"message": "привет"}
    assert "ответ инструмента fake_echo" in result.text
    assert mcp_factory.calls == [("fake_echo", {"message": "привет"})]


def test_unknown_tool_is_reported_without_starting_servers(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    before = len(mcp_factory.specs)
    with pytest.raises(MCPError) as error:
        session.call_mcp_tool("которого-нет")
    assert "которого-нет" in str(error.value)
    assert "fake_echo" in str(error.value)  # перечисляем, что доступно
    assert len(mcp_factory.specs) == before


def test_tool_command_shapes_and_error_paths(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()

    called = session.run_command("/tool call fake_echo message=привет")
    assert any("ответ инструмента fake_echo" in line for line in called.lines)

    unknown = session.run_command("/tool call которого-нет")
    assert not unknown.unknown
    assert "не вызван" in unknown.lines[0]

    malformed = session.run_command("/tool call fake_echo без-равенства")
    assert "ключ=значение" in malformed.lines[0]

    wrong_form = session.run_command("/tool дерни")
    assert "Форма команды" in wrong_form.lines[0]


def test_tool_error_result_is_data_not_exception(repo: Path):
    """Отказ инструмента — строка отчёта, а не исключение, ломающее сессию."""
    from core.mcp_client import MCPCallResult
    from tests.conftest import FakeMCPFactory

    factory = FakeMCPFactory(
        call_result=MCPCallResult(
            server="repo",
            tool="fake_echo",
            arguments={},
            text="файл не найден",
            is_error=True,
        )
    )
    session = _session(repo, mcp_client_factory=factory)
    session.connect_mcp_servers()
    result = session.run_command("/tool call fake_echo message=привет")
    assert "вернул ошибку" in result.lines[0]
    assert "файл не найден" in result.lines[1]


def test_refresh_walks_the_registry_again(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    first = len(mcp_factory.specs)
    session.run_command("/mcp refresh")
    assert len(mcp_factory.specs) == first * 2


def test_mcp_snapshot_is_a_tuple(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    assert isinstance(session.mcp_report(), tuple)
    assert session.mcp_report()


@pytest.fixture(autouse=True)
def without_environment_override(monkeypatch):
    """Снимает тестовый сторож из conftest: здесь проверяется настоящий реестр домена."""
    for name in ("FFAI_MCP_COMMAND", "FFAI_MCP_ARGS", "FFAI_MCP_URL"):
        monkeypatch.delenv(name, raising=False)


def test_tool_arguments_support_quoted_values(repo: Path, mcp_factory):
    """Значение с пробелами берётся в кавычки — иначе поисковый запрос не набрать."""
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    session.run_command('/tool call fake_echo message="как собрать пакет под aarch64"')
    assert mcp_factory.calls[-1] == ("fake_echo", {"message": "как собрать пакет под aarch64"})


def test_tool_arguments_split_quotes_fall_back_gracefully(repo: Path, mcp_factory):
    session = _session(repo, mcp_client_factory=mcp_factory)
    session.connect_mcp_servers()
    result = session.run_command('/tool call fake_echo message="незакрытая кавычка')
    assert not result.unknown
