"""Общие фикстуры для всех уровней тестов."""

import io
import sys
import time
from pathlib import Path
from typing import List

import pytest
from rich.console import Console

from core import config, prompts
from core.answer_settings import AnswerSettings
from core.domains import Domain
from core.history_manager import HistoryManager


@pytest.fixture(autouse=True)
def clear_prompt_caches():
    """Сбрасывает кэши промптов до и после каждого теста.

    Без этого подмена каталога ассетов или пакета домена не влияет ни на что: тексты
    кэшируются навсегда при первом обращении и утекают между тестами.
    """
    prompts.get_format_instruction.cache_clear()
    Domain.prompt.cache_clear()
    yield
    prompts.get_format_instruction.cache_clear()
    Domain.prompt.cache_clear()


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path: Path) -> Path:
    """Уводит пути состояния в tmp_path: тест не должен касаться реальных файлов пользователя.

    Значения читаются модулем конфигурации при импорте, поэтому мало переопределить переменные —
    модуль перечитывается, и все, кто обращается к нему как `config.NAME`, видят новые пути.
    """
    import importlib

    state = tmp_path / "state"
    cache = tmp_path / "cache"
    monkeypatch.setenv("FFAI_STATE_DIR", str(state))
    monkeypatch.setenv("FFAI_CACHE_DIR", str(cache))
    monkeypatch.setenv("FFAI_HISTORY_FILE", str(state / "history.json"))
    monkeypatch.setenv("FFAI_MEMORY_FILE", str(state / "memory.json"))
    monkeypatch.setenv("FFAI_PROFILE_FILE", str(state / "profile.json"))
    monkeypatch.setenv("FFAI_TASK_FILE", str(state / "task.json"))
    monkeypatch.setenv("FFAI_SCHEDULE_FILE", str(state / "schedule.json"))
    monkeypatch.setenv("FFAI_TASKS_DIR", str(state / "tasks"))
    monkeypatch.setenv("FFAI_EXPORTS_DIR", str(state / "reports"))
    monkeypatch.delenv("FFAI_INDEX_FILE", raising=False)
    importlib.reload(config)
    importlib.reload(prompts)
    yield state
    importlib.reload(config)
    importlib.reload(prompts)


@pytest.fixture
def history_path(tmp_path: Path) -> Path:
    return tmp_path / "history.json"


@pytest.fixture
def history(history_path: Path) -> HistoryManager:
    return HistoryManager(path=history_path)


@pytest.fixture
def no_sleep(monkeypatch) -> List[float]:
    """Убирает реальные паузы бэкоффа и записывает запрошенные длительности (1, 2, 4 ...)."""
    slept: List[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    return slept


class RecordingConsole:
    """Console, пишущая в буфер, вместе с доступом к накопленному тексту."""

    def __init__(self, width: int = 100) -> None:
        self.buffer = io.StringIO()
        self.console = Console(
            file=self.buffer,
            width=width,
            force_terminal=False,
            no_color=True,
            highlight=False,
            legacy_windows=False,
        )

    @property
    def text(self) -> str:
        return self.buffer.getvalue()

    def contains(self, needle: str) -> bool:
        # rich переносит длинные строки по ширине консоли — ищем по тексту со схлопнутыми пробелами.
        return _collapse(needle) in _collapse(self.text)


def _collapse(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def recording_console() -> RecordingConsole:
    return RecordingConsole()


@pytest.fixture
def default_settings() -> AnswerSettings:
    return AnswerSettings()


@pytest.fixture
def real_assets_dir() -> Path:
    return config.ASSETS_DIR


@pytest.fixture
def skip_on_windows():
    if sys.platform.startswith("win"):
        pytest.skip("тест требует Unix-терминала")


def pytest_addoption(parser):
    parser.addoption(
        "--show-tui",
        action="store",
        default="off",
        choices=("off", "live", "screen"),
        help=(
            "показывать вывод приложения в терминале: live — транслировать поток по мере работы, "
            "screen — печатать итоговый экран. Требует -s, иначе pytest перехватит вывод."
        ),
    )
