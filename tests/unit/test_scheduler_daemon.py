"""Фоновый исполнитель расписания как отдельный процесс.

Исполнитель — обычный MCP-клиент: он поднимает собственный сервер репозитория и зовёт выполнение
просроченных заданий по протоколу. Прогон подменяет файл планировщика, состояние и целевой
репозиторий, поэтому проверка не зависит ни от сети, ни от файлов пользователя: задание здесь
сканирует временный репозиторий по правилам домена и никуда не ходит.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="исполнитель и сервер говорят на MCP: нужен пакет mcp (Python 3.10+)")

from core import config  # noqa: E402
from core.mcp_client import MCPClient  # noqa: E402
from core.mcp_registry import repo_server_spec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DAEMON = REPO_ROOT / "ff-ai-scheduler.py"


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    """Целевой репозиторий с одним исходником, который правила домена считают нарушением."""
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src" / "main.cpp").write_text("QML_ELEMENT\n", encoding="utf-8")
    return root


@pytest.fixture
def environment(target_repo: Path, tmp_path: Path, monkeypatch) -> dict:
    """Окружение исполнителя и сервера: состояние, кэш и файл расписания — в tmp_path теста."""
    state = tmp_path / "state"
    cache = tmp_path / "cache"
    monkeypatch.setenv("FFAI_STATE_DIR", str(state))
    monkeypatch.setenv("FFAI_CACHE_DIR", str(cache))
    monkeypatch.setenv("FFAI_SCHEDULE_FILE", str(state / "schedule.json"))
    monkeypatch.setenv("FFAI_EXPORTS_DIR", str(state / "reports"))
    monkeypatch.setenv("FFAI_INDEX_FILE", str(cache / "index.sqlite3"))
    return dict(os.environ)


def schedule_file(environment: dict) -> Path:
    return Path(environment["FFAI_SCHEDULE_FILE"])


def add_job(environment: dict, root: Path, tool: str = "public_api_scan", arguments=None) -> None:
    """Ставит задание тем же путём, что и приложение, — вызовом инструмента сервера."""
    tools_path = config.DOMAINS_DIR / "aurora-qt5" / "tools.json"
    spec = repo_server_spec(root, tools_path)
    result = MCPClient(spec).call_tool(
        "schedule_add",
        {"tool": tool, "arguments": arguments or {}, "every_minutes": 1},
    )
    assert result.is_error is False, result.text
    assert "поставлено" in result.text


def run_daemon(environment: dict, root: Path, *args: str, timeout: int = 60):
    return subprocess.run(
        [sys.executable, str(DAEMON), "--repo", str(root), *args],
        env=environment,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_single_run_executes_the_due_job_and_exits(environment, target_repo):
    add_job(environment, target_repo)
    result = run_daemon(environment, target_repo, "--once")

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(schedule_file(environment).read_text(encoding="utf-8"))
    assert data["runs"], result.stdout + result.stderr
    assert data["runs"][0]["ok"] is True, data["runs"]
    assert "Проверка публичного API" in data["runs"][0]["summary"]
    # Накопленное переживает перезапуск: оно в том же файле, а не в памяти процесса.
    assert data["collected"]["нарушения"]
    assert data["runs"][0]["fresh"] == 1


def test_the_next_pass_has_nothing_to_do(environment, target_repo):
    """Срок задания сдвигается прогоном: второй проход не повторяет работу того же периода."""
    add_job(environment, target_repo)
    run_daemon(environment, target_repo, "--once")

    second = run_daemon(environment, target_repo, "--once")

    assert second.returncode == 0
    assert "просроченных заданий нет" in second.stdout
    data = json.loads(schedule_file(environment).read_text(encoding="utf-8"))
    assert len(data["runs"]) == 1


def test_unavailable_server_is_reported_without_a_crash(environment, target_repo, tmp_path):
    """Отказ сервера печатается и не роняет исполнитель: в цикле тик просто пропускается."""
    result = run_daemon(
        environment, target_repo, "--once", "--command", str(tmp_path / "нет-такой-команды")
    )

    assert result.returncode == 0
    assert "не удалось" in (result.stdout + result.stderr).lower()
