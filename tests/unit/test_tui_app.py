"""Главный цикл интерфейса на инъекции зависимостей: команды, ответ, статус-бар.

Терминала здесь нет: консоль пишет в буфер, а сессия получает клиент-заглушку. Псевдотерминал
нужен только для клавиатуры и панелей — это отдельный слой (`test_keyboard.py`, e2e).
"""

from pathlib import Path
from typing import List

import pytest

from core import config
from core.api_client import AnswerMeta
from core.history_manager import HistoryManager
from core.session import AssistantSession
from ui.tui_app import DevAssistantTUI


class FakeClient:
    def __init__(self, answer: str = "Модель инициализируется в src/models.cpp.") -> None:
        self.answer = answer
        self.calls: List[dict] = []

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append({"messages": messages, "model": model})
        return AnswerMeta(
            content=self.answer,
            model=model,
            elapsed_seconds=1.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.0003,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    return tmp_path / "repo"


@pytest.fixture
def tui(repo: Path, recording_console, history_path: Path, monkeypatch, mcp_factory):
    # Ключ задан, иначе интерфейс спросит его у stdin — а stdin занят pytest.
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-test-key")
    monkeypatch.setattr(config, "_api_key_runtime", None, raising=False)
    client = FakeClient()
    session = AssistantSession(
        root=repo,
        client=client,
        history=HistoryManager(path=history_path),
        mcp_client_factory=mcp_factory,
        docs_retriever=None,  # поиск по документации проверяется отдельными тестами
    )
    return DevAssistantTUI(session=session, console=recording_console.console, typing_delay=0.0)


def test_answer_is_printed_with_metrics(tui, recording_console):
    tui._ask("где инициализируется модель?")
    assert recording_console.contains("src/models.cpp")
    assert recording_console.contains("Токены: 100+50=150")
    assert recording_console.contains("Стоимость: $0.000300")
    assert recording_console.contains("Средняя скорость: 50.00 ток/сек")


def test_reports_are_printed_without_model_calls(tui, recording_console):
    tui._handle_command("/context")
    assert recording_console.contains("Стратегия: summary")
    assert recording_console.contains("Оценка запроса")
    tui._handle_command("/usage")
    assert recording_console.contains("Вся история")
    tui._handle_command("/domain")
    assert recording_console.contains("Qt 5.6")
    assert tui.session.session_usage.requests == 0


def test_unknown_command_is_reported(tui, recording_console):
    tui._handle_command("/what")
    assert recording_console.contains("Неизвестная команда")


def test_clear_empties_the_dialogue(tui, recording_console):
    tui._ask("вопрос")
    assert tui.session.history.count() == 1
    tui._handle_command("/clear")
    assert recording_console.contains("Диалог очищен.")
    assert tui.session.history.count() == 0


def test_question_is_not_sent_without_a_key(tui, recording_console, monkeypatch, repo: Path):
    """Без ключа запрос не уходит: интерфейс сообщает об этом и возвращает управление."""
    import io
    import sys

    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setattr(config, "_api_key_runtime", None, raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    tui.session.client = None
    tui._ask("вопрос")
    assert recording_console.contains("OPENCODE_API_KEY")
    assert tui.session.session_usage.requests == 0


def test_status_bar_shows_session_state(tui, recording_console):
    tui._print_status_bar()
    assert recording_console.contains("Модель: deepseek-v4.1-flash")
    assert recording_console.contains("Домен: aurora-qt5")
    assert recording_console.contains("Формат: свободный")
    assert recording_console.contains("Стратегия: резюме")


def test_welcome_panel_names_the_domain_and_the_root(tui, recording_console):
    tui._print_welcome()
    assert recording_console.contains("ОС Аврора")
    assert recording_console.contains("Корень репозитория")
    assert recording_console.contains("Модель: deepseek-v4.1-flash")


def test_orchestration_lives_outside_the_interface():
    """Интерфейс не создаёт агента, клиента и хранилища: он берёт их из фасада.

    Проверка текстовая и потому дешёвая, но ловит именно ту ошибку, ради которой вводился
    фасад: оркестрация, расползающаяся обратно в терминальный слой.
    """
    for path in (config.BASE_DIR / "ui").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("APIClient(", "RepoAgent(", "HistoryManager(", "config.HISTORY_FILE"):
            assert forbidden not in text, f"{path.name}: интерфейс владеет оркестрацией ({forbidden})"


def test_keyboard_interrupt_exits_cleanly(tui, recording_console, monkeypatch):
    """Ctrl+C во время вопроса завершает приложение штатно, а не оставляет панель открытой."""
    inputs = iter(["вопрос"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        tui.session,
        "ask",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert tui.run() == 0
    assert recording_console.contains("До связи")


def test_mcp_and_tool_commands_are_printed(tui, recording_console):
    tui.session.connect_mcp_servers()
    tui._handle_command("/mcp")
    assert recording_console.contains("aurora-docs")
    assert recording_console.contains("fake_echo")
    assert recording_console.contains("протокол:")

    tui._handle_command("/tool")
    assert recording_console.contains("параметры: message")

    tui._handle_command("/tool call fake_echo message=привет")
    assert recording_console.contains("ответ инструмента fake_echo")
    assert tui.session.session_usage.requests == 0


def test_tool_call_error_is_reported_not_raised(tui, recording_console):
    tui.session.connect_mcp_servers()
    tui._handle_command("/tool call которого-нет")
    assert recording_console.contains("не вызван")


def test_startup_summary_line_names_servers_and_tools(tui, recording_console):
    tui._print_mcp_summary()
    assert recording_console.contains("MCP: 2/2 серверов")
    assert recording_console.contains("/mcp")


@pytest.fixture(autouse=True)
def without_environment_override(monkeypatch):
    """Снимает тестовый сторож из conftest: здесь проверяется настоящий реестр домена."""
    for name in ("FFAI_MCP_COMMAND", "FFAI_MCP_ARGS", "FFAI_MCP_URL"):
        monkeypatch.delenv(name, raising=False)


def test_sources_line_lists_delivered_fragments(tui, recording_console, docs_retriever, docs_fragment):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(
        query="вопрос", fragments=(docs_fragment,), version="5.2.1"
    )
    tui.session._agent.docs_retriever = docs_retriever
    tui._ask("как получить координаты?")
    assert recording_console.contains("📚 Источники (документация 5.2.1)")
    assert recording_console.contains(docs_fragment.identifier)
    assert recording_console.contains("раздел docs")


def test_docs_journal_reports_unavailable_and_replacement(tui, recording_console, docs_retriever):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(query="вопрос", status="unavailable", error="нет сети")
    tui.session._agent.docs_retriever = docs_retriever
    tui._ask("вопрос")
    assert recording_console.contains("Документация портала недоступна: нет сети")


def test_docs_journal_reports_empty_search(tui, recording_console, docs_retriever):
    from tests.conftest import FakeDocsReport

    docs_retriever.report = FakeDocsReport(query="вопрос", status="no_candidates")
    tui.session._agent.docs_retriever = docs_retriever
    tui._ask("вопрос")
    assert recording_console.contains("ничего не найдено")
