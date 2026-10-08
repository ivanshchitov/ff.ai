"""Пути состояния, кэш индекса и переменные окружения."""

import importlib
import re
from pathlib import Path

import pytest

from core import config

_STATE_VARIABLES = (
    "FFAI_STATE_DIR",
    "FFAI_CACHE_DIR",
    "FFAI_HISTORY_FILE",
    "FFAI_MEMORY_FILE",
    "FFAI_PROFILE_FILE",
    "FFAI_TASK_FILE",
    "FFAI_SCHEDULE_FILE",
    "FFAI_TASKS_DIR",
    "FFAI_EXPORTS_DIR",
    "FFAI_INDEX_FILE",
)


@pytest.fixture
def load_config(monkeypatch):
    """Перечитывает конфигурацию с нужным набором переменных (значения читаются при импорте)."""

    def load(**env):
        for name in _STATE_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(config)

    return load


def test_state_and_cache_live_outside_the_repository(load_config):
    cfg = load_config()
    assert cfg.STATE_DIR != cfg.BASE_DIR
    assert cfg.CACHE_DIR != cfg.BASE_DIR
    assert cfg.BASE_DIR not in cfg.STATE_DIR.parents
    assert cfg.BASE_DIR not in cfg.CACHE_DIR.parents
    assert cfg.STATE_DIR.name == "ff-ai"
    assert cfg.CACHE_DIR.name == "ff-ai"


def test_store_paths_sit_under_state_dir(load_config):
    cfg = load_config()
    for path in (
        cfg.HISTORY_FILE,
        cfg.MEMORY_FILE,
        cfg.PROFILE_FILE,
        cfg.TASK_FILE,
        cfg.SCHEDULE_FILE,
        cfg.TASKS_DIR,
        cfg.EXPORTS_DIR,
    ):
        assert path.parent == cfg.STATE_DIR


def test_env_override_wins(load_config, tmp_path: Path):
    custom = tmp_path / "custom-history.json"
    cfg = load_config(FFAI_HISTORY_FILE=str(custom))
    assert cfg.HISTORY_FILE == custom


def test_index_is_kept_per_repository(load_config, tmp_path: Path):
    cfg = load_config()
    first = cfg.repo_index_file(tmp_path / "repo-a")
    second = cfg.repo_index_file(tmp_path / "repo-b")
    assert first != second
    assert first == cfg.repo_index_file(tmp_path / "repo-a")
    assert cfg.CACHE_DIR in first.parents
    assert first.name == "index.sqlite3"


def test_index_override_wins(load_config, tmp_path: Path):
    explicit = tmp_path / "explicit.sqlite3"
    cfg = load_config(FFAI_INDEX_FILE=str(explicit))
    assert cfg.repo_index_file(tmp_path / "whatever") == explicit


def test_max_tokens_never_below_the_reasoning_floor():
    # Reasoning-модели тратят токены до первого символа ответа: потолок для короткого
    # ответа не может опускаться ниже технического минимума.
    assert config.max_tokens_for_words(config.MIN_MAX_WORDS) == config.MIN_REQUEST_MAX_TOKENS
    assert config.max_tokens_for_words(config.MAX_MAX_WORDS) > config.MIN_REQUEST_MAX_TOKENS


def _env_names_from_sources() -> set:
    """Имена настроек, которые код действительно читает.

    Ищутся литералы с префиксами приложения и провайдера: так в набор попадают и прямые
    `os.getenv("FFAI_INDEX_FILE")`, и имена, переданные в помощник чтения путей, но в него
    не попадают системные переменные вроде `XDG_CACHE_HOME` — они не настройки приложения.
    """
    names = set()
    pattern = re.compile(r"[\"']((?:FFAI|OPENCODE)_[A-Z0-9_]+)[\"']")
    for directory in ("core", "ui"):
        for path in (config.BASE_DIR / directory).glob("*.py"):
            names.update(pattern.findall(path.read_text(encoding="utf-8")))
    return names


def _env_names_from_template() -> set:
    template = config.BASE_DIR / ".env.example"
    names = set()
    for line in template.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            # В шаблоне необязательные настройки закомментированы — они всё равно описаны.
            line = line.lstrip("#").strip()
        if "=" not in line:
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def test_template_and_code_agree_about_variables():
    """Шаблон `.env.example` — единственное описание настроек: он не имеет права устареть."""
    assert _env_names_from_template() == _env_names_from_sources()


def test_configured_paths_are_outside_the_target_repository(load_config, tmp_path: Path):
    """Раскладка состояния не зависит от целевого репозитория: чужой каталог не засоряется."""
    cfg = load_config()
    target = tmp_path / "aurora-project"
    target.mkdir()
    for path in (cfg.HISTORY_FILE, cfg.MEMORY_FILE, cfg.repo_index_file(target)):
        assert target not in path.parents


# --- состояние проекта ----------------------------------------------------------------------


def _without_state_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Снимает изоляцию тестов: она задаёт явные пути, а здесь проверяется путь по проекту."""
    for name in (
        "FFAI_HISTORY_FILE",
        "FFAI_MEMORY_FILE",
        "FFAI_TASK_FILE",
        "FFAI_SCHEDULE_FILE",
        "FFAI_TASKS_DIR",
        "FFAI_EXPORTS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_project_state_paths_are_keyed_by_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Состояние принадлежит проекту: у двух репозиториев разные файлы истории и памяти."""
    _without_state_overrides(monkeypatch)
    first = config.project_paths(tmp_path / "один")
    second = config.project_paths(tmp_path / "два")
    assert first["history"] != second["history"]
    assert first["memory"] != second["memory"]
    assert first["task"] != second["task"]
    assert first["schedule"] != second["schedule"]
    assert first["tasks_dir"] != second["tasks_dir"]
    assert "projects" in first["history"].parts


def test_project_state_paths_are_stable_for_one_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _without_state_overrides(monkeypatch)
    assert config.project_paths(tmp_path) == config.project_paths(tmp_path)


def test_explicit_environment_path_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Явный путь — то, чем пользуются тесты и демонстрации: он побеждает путь по проекту."""
    explicit = tmp_path / "явный-history.json"
    monkeypatch.setenv("FFAI_HISTORY_FILE", str(explicit))
    assert config.project_paths(tmp_path / "проект")["history"] == explicit


def test_session_stores_state_under_the_project(tmp_path: Path):
    """Фасад строит хранилища по путям проекта, а не по общему каталогу состояния."""
    from core.session import AssistantSession

    (tmp_path / "repo" / ".git").mkdir(parents=True)
    session = AssistantSession(root=tmp_path / "repo", domain_id="aurora-qt5", client=None)
    expected = config.project_paths(tmp_path / "repo")
    assert session._history.path == expected["history"]
    assert session._task_store.path == expected["task"]
    assert session._agent.long_term.path == expected["memory"]
