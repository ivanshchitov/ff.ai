"""Фасад сессии: операции, сериализация, события и снимки без обращения к модели."""

import threading
import time
import os
from pathlib import Path

import pytest

from core import config
from core.agent import RequestPhase
from core.answer_settings import AnswerFormat
from core.api_client import APIError, AnswerMeta
from core.history_manager import HistoryManager
from core.mcp_client import MCPError
from core.session import AnswerReady, AssistantSession, JournalLine, PhaseChanged
from tests.conftest import rerank_answer


class FakeClient:
    """Клиент-заглушка: записывает запросы и умеет блокироваться, чтобы проверить сериализацию."""

    def __init__(
        self,
        answer: str = "Это ответ.",
        started: threading.Event = None,
        release: threading.Event = None,
        rerank_score: float = 0.9,
    ) -> None:
        self.answer = answer
        self.calls = []
        self.started = started
        self.release = release
        self.rerank_score = rerank_score

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model": model,
            }
        )
        rated = rerank_answer(messages, self.rerank_score)
        if rated is not None:
            return AnswerMeta(
                content=rated,
                model=model,
                elapsed_seconds=0.1,
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
                cost_usd=0.00001,
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
    # Поиск по документации по умолчанию выключен: тест, который его проверяет, передаёт свой
    # ретривер, а остальные не должны ни ходить в сеть, ни поднимать серверы.
    kwargs.setdefault("docs_retriever", None)
    # Корпус кода тоже выключен по умолчанию: у репозитория теста нет индекса, и вопрос не должен
    # получать сообщений о поиске, которых тест не ждёт.
    kwargs.setdefault("code_retriever", None)
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


# --- документация портала: режим, версия, отчёт ----------------------------------------------


def test_docs_lines_show_state_and_fragments(repo: Path, docs_retriever, docs_fragment):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(query="координаты", fragments=(docs_fragment,))
    session = _session(repo, docs_retriever=docs_retriever)
    session.ask("как получить координаты?")
    lines = "\n".join(session.docs_lines())
    assert "координаты" in lines
    assert "Состояние: ok" in lines
    assert docs_fragment.identifier in lines
    assert "Доставлено фрагментов: 1" in lines


def test_docs_report_is_available_without_a_question(repo: Path, docs_retriever):
    session = _session(repo, docs_retriever=docs_retriever)
    lines = session.docs_lines()
    assert any("Поиска ещё не было" in line for line in lines)
    assert any("Режим отбора" in line for line in lines), "настройки видны и без поиска"


def test_docs_mode_off_disables_search_and_check(repo: Path, docs_retriever, docs_fragment):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(query="вопрос", fragments=(docs_fragment,))
    session = _session(repo, docs_retriever=docs_retriever)
    result = session.run_command("/docs mode off")
    assert "выключен" in result.lines[0]
    assert not session.docs_enabled
    assert "выключен" in session.docs_lines()[0]

    session.ask("вопрос")
    assert docs_retriever.queries == [], "выключенный режим не ищет"

    session.run_command("/docs mode on")
    session.ask("вопрос")
    assert len(docs_retriever.queries) == 1


def test_docs_version_command_sets_the_retriever(repo: Path, docs_retriever):
    session = _session(repo, docs_retriever=docs_retriever)
    result = session.run_command("/docs version 5.1.5")
    assert "5.1.5" in result.lines[0]
    assert docs_retriever.version == "5.1.5"
    assert docs_retriever.refreshes == 1, "смена версии сбрасывает список версий"
    session.ask("вопрос")
    assert docs_retriever.queries[-1] == ("вопрос", "5.1.5"), "поиск идёт по заданной версии"

    empty = session.run_command("/docs version")
    assert "актуальная" in empty.lines[0]


def test_docs_command_reports_without_model_requests(repo: Path, docs_retriever):
    client = FakeClient()
    session = _session(repo, client=client, docs_retriever=docs_retriever)
    session.run_command("/docs")
    session.run_command("/docs trace")
    session.ask("вопрос")
    before = len(client.calls)
    session.run_command("/docs")
    session.run_command("/docs trace")
    assert len(client.calls) == before, "отчёты не обращаются к модели"


def test_docs_mode_form_is_checked(repo: Path, docs_retriever):
    session = _session(repo, docs_retriever=docs_retriever)
    result = session.run_command("/docs mode может")
    assert "Форма" in result.lines[0]
    assert session.docs_enabled


def test_citations_snapshot_is_exposed(repo: Path, docs_retriever, docs_fragment):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(query="вопрос", fragments=(docs_fragment,))
    session = _session(repo, docs_retriever=docs_retriever)
    session.ask("вопрос")
    check = session.last_citations
    assert check is not None and not check.confirmed
    assert check.replaced


# --- режим отбора и порог --------------------------------------------------------------------


def test_docs_retrieval_mode_command(repo: Path, docs_retriever):
    session = _session(repo, docs_retriever=docs_retriever)
    assert session.docs_retrieval == config.DOCS_RETRIEVAL_MODE

    result = session.run_command("/docs retrieval baseline")
    assert "baseline" in result.lines[0]
    assert session.docs_retrieval == "baseline"
    assert any("Режим отбора" in line for line in result.lines)

    session.run_command("/docs retrieval enhanced")
    assert session.docs_retrieval == "enhanced"

    wrong = session.run_command("/docs retrieval наугад")
    assert "Форма" in wrong.lines[0]
    assert session.docs_retrieval == "enhanced"


def test_docs_threshold_command(repo: Path, docs_retriever):
    session = _session(repo, docs_retriever=docs_retriever)
    result = session.run_command("/docs threshold 0,8")
    assert "0.80" in result.lines[0]
    assert session.docs_threshold == 0.8

    for bad in ("1.5", "-0.1", "много"):
        wrong = session.run_command(f"/docs threshold {bad}")
        assert session.docs_threshold == 0.8, f"значение {bad} не должно применяться"
        assert wrong.lines[0]
    assert "от 0 до 1" in session.run_command("/docs threshold 1.5").lines[0]


def test_report_shows_mode_threshold_and_scores(repo: Path, docs_retriever):
    from tests.conftest import FakeCandidate, FakeDocsReport

    docs_retriever.report = FakeDocsReport(
        query="координаты",
        candidates=(
            FakeCandidate(path="doc/one", title="Первый", score=0.9, reason="подходит"),
            FakeCandidate(path="doc/two", title="Второй", score=0.1, reason="другая тема"),
        ),
        rated=True,
        threshold=0.6,
    )
    session = _session(repo, docs_retriever=docs_retriever)
    session.ask("вопрос")
    lines = "\n".join(session.docs_lines())
    assert "Режим отбора: enhanced; порог: 0.60" in lines
    assert "оценены" in lines
    assert "0.90 — doc/one" in lines and "подходит" in lines
    assert "0.10 — doc/two" in lines and "другая тема" in lines


def test_no_matches_state_is_reported(repo: Path, docs_retriever):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(query="вопрос", status="no_matches", rated=True)
    session = _session(repo, docs_retriever=docs_retriever)
    session.ask("вопрос")
    lines = "\n".join(session.docs_lines())
    assert "Состояние: no_matches" in lines


# --- настройки корпуса кода -----------------------------------------------------------------


def test_code_retrieval_mode_command(repo: Path):
    session = _session(repo)
    result = session.run_command("/code retrieval baseline")
    assert "Режим отбора: baseline" in result.lines[0]
    assert session.code_retrieval == "baseline"

    wrong = session.run_command("/code retrieval какой-то")
    assert "Форма: /code retrieval" in wrong.lines[0]
    assert session.code_retrieval == "baseline", "негодное значение ничего не меняет"


def test_code_threshold_command(repo: Path):
    session = _session(repo)
    session.run_command("/code threshold 0,75")
    assert session.code_threshold == 0.75

    wrong = session.run_command("/code threshold 5")
    assert "от 0 до 1" in wrong.lines[0]
    assert session.code_threshold == 0.75


def test_code_tune_is_atomic(repo: Path):
    session = _session(repo)
    result = session.run_command("/code tune before=10 after=2")
    assert "до 10 кандидатов, до 2 фрагментов" in result.lines[0]

    wrong = session.run_command("/code tune before=2 after=10")
    assert "не изменены" in wrong.lines[0]
    config_ = session._agent.config
    assert (config_.code_before, config_.code_after) == (10, 2), "оба поля сохранились"

    form = session.run_command("/code tune before=10")
    assert "Форма: /code tune" in form.lines[0]


def test_code_mode_command(repo: Path):
    session = _session(repo)
    assert session.code_enabled is True
    session.run_command("/code mode off")
    assert session.code_enabled is False
    session.run_command("/code mode on")
    assert session.code_enabled is True


def test_code_trace_without_search(repo: Path):
    session = _session(repo)
    lines = session.run_command("/code trace").lines
    assert any("Режим отбора" in line for line in lines)
    assert any("Поиска ещё не было" in line for line in lines)


def test_code_commands_make_no_model_requests(repo: Path):
    client = FakeClient()
    session = _session(repo, client=client)
    for command in (
        "/code trace",
        "/code retrieval baseline",
        "/code threshold 0.5",
        "/code tune before=5 after=1",
        "/code mode on",
    ):
        session.run_command(command)
    assert client.calls == []


# --- автовызов инструментов -----------------------------------------------------------------


def test_tool_auto_command(repo: Path):
    session = _session(repo)
    assert session._agent.config.auto_tools is True
    session.run_command("/tool auto off")
    assert session._agent.config.auto_tools is False
    session.run_command("/tool auto on")
    assert session._agent.config.auto_tools is True


def test_tool_auto_command_form(repo: Path):
    session = _session(repo)
    result = session.run_command("/tool auto какой-то")
    assert "Форма: /tool auto on|off" in result.lines[0]


def test_tool_flow_report_without_flow(repo: Path):
    session = _session(repo)
    lines = session.run_command("/tool flow").lines
    assert "Автовызов инструментов: включён" in lines[0]
    assert any("Флоу не выполнялся" in line for line in lines)


def test_tool_command_help_lists_auto_and_flow(repo: Path):
    session = _session(repo)
    lines = session.run_command("/tool нет-такой-подкоманды").lines
    assert any("/tool auto on|off" in line for line in lines)
    assert any("/tool flow" in line for line in lines)


# --- расписание ------------------------------------------------------------------------------


def test_schedule_report_without_jobs(repo: Path):
    session = _session(repo)
    lines = session.run_command("/schedule").lines
    assert any("Заданий: 0" in line for line in lines)
    assert any("Заданий нет" in line for line in lines)


def test_schedule_startup_line(repo: Path):
    session = _session(repo)
    line = session.schedule_startup_line()
    assert "Планировщик" in line and "0 заданий" in line


def test_schedule_announcement_is_empty_without_runs(repo: Path):
    session = _session(repo)
    assert session.schedule_announcement() == ()
