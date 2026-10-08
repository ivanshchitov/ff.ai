"""Корпус кода в живом приложении: сборка индекса, состояние, сравнение стратегий.

Приложение запускается в настоящем pty, индекс и состояние уведены в временные пути: целевой
репозиторий демонстрационного проекта не должен получить ни одного нового файла.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from .harness import AppSession

INDEX_TIMEOUT = 60
COMMAND_TIMEOUT = 30


def _project(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    # Приложение работает только внутри git-репозитория: корень ищется по каталогу цели.
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "rpm").mkdir(parents=True, exist_ok=True)
    lines = []
    for number in range(1, 60):
        lines.append(f"// строка {number}")
        if number == 10:
            lines.append("class ModelList : public QObject {")
        if number == 20:
            lines.append("void ModelList::load() {")
    (root / "src" / "models.cpp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "src" / "main.qml").write_text(
        "import QtQuick 2.0\n\nApplicationWindow {\n    id: window\n}\n", encoding="utf-8"
    )
    (root / "rpm" / "aurora-notes.spec").write_text(
        "Name: aurora-notes\nVersion: 0.1\n%build\nqmake\nmake\n", encoding="utf-8"
    )
    (root / "build").mkdir(exist_ok=True)
    (root / "build" / "generated.cpp").write_text("int generated;\n", encoding="utf-8")
    return root


def _launch(tmp_path: pathlib.Path, cache: pathlib.Path) -> AppSession:
    repo = _project(tmp_path / "repo")
    state = tmp_path / "state"
    return AppSession(
        history_file=state / "history.json",
        state_dir=state,
        cache_dir=cache,
        repo=repo,
        api_key=None,
        api_url=None,
        mcp_command="",
        mcp_args="",
    )


@pytest.fixture
def session(tmp_path: pathlib.Path):
    cache = tmp_path / "cache"
    app = _launch(tmp_path, cache)
    try:
        app.wait_for("MCP:", timeout=INDEX_TIMEOUT)
        yield app, cache
    finally:
        app.close()


def test_index_builds_outside_the_repository(session, tmp_path: pathlib.Path):
    app, cache = session
    repo_files_before = sorted(path.name for path in (tmp_path / "repo").rglob("*"))

    app.send_line("/rag-code index")
    app.wait_for("Индекс:", timeout=INDEX_TIMEOUT)
    screen = app.screen_text()

    assert "Стратегия: structural" in screen
    assert "index.sqlite3" in screen, "путь индекса виден в отчёте"
    assert "cache" in screen, "в отчёте назван каталог кэша"
    assert (cache / "index.sqlite3").is_file()
    assert sorted(path.name for path in (tmp_path / "repo").rglob("*")) == repo_files_before


def test_status_without_index_explains_what_to_do(session):
    app, _ = session
    app.send_line("/rag-code")
    screen = app.wait_for("Индекса нет", timeout=COMMAND_TIMEOUT)
    assert "/rag-code index" in screen


def test_compare_prints_both_strategies_without_model_requests(session):
    app, _ = session
    app.send_line("/rag-code compare")
    screen = app.wait_for("Сравнение стратегий", timeout=INDEX_TIMEOUT)

    assert "fixed: фрагментов" in screen
    assert "structural: фрагментов" in screen
    assert "L" in screen and "models.cpp" in screen, "примеры границ показывают путь и строки"

    app.send_line("/usage")
    usage = app.wait_for("Сессия: запросов", timeout=COMMAND_TIMEOUT)
    assert "запросов — 0" in usage, "команды корпуса не обращаются к модели"


def test_failed_build_keeps_the_previous_index(session, tmp_path: pathlib.Path):
    app, cache = session
    app.send_line("/rag-code index structural")
    app.wait_for("Стратегия: structural", timeout=INDEX_TIMEOUT)

    blocked = cache / "index.sqlite3.tmp"
    blocked.mkdir(parents=True, exist_ok=True)
    app.send_line("/rag-code index fixed")
    screen = app.wait_for("Индекс не собран", timeout=INDEX_TIMEOUT)
    assert "остался на месте" in screen

    blocked.rmdir()
    app.send_line("/rag-code status")
    assert "Стратегия: structural" in app.wait_for("Стратегия:", timeout=COMMAND_TIMEOUT)
