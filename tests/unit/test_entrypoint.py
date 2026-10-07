"""Точка входа: печать версии без запуска интерфейса и перезапуск в окружении проекта."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[2] / "ff-ai.py"


def _run(script: Path, *args: str, reexec: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["FFAI_NO_REEXEC"] = "0" if reexec else "1"
    # Копия точки входа лежит во временном каталоге — пакеты берём из настоящего репозитория.
    env["PYTHONPATH"] = str(ENTRY.parent)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(script.parent),
        timeout=60,
    )


def test_version_prints_without_starting_the_interface():
    result = _run(ENTRY, "--version")
    assert result.returncode == 0
    assert result.stdout.strip() == "ff.ai 0.1"
    assert "Домен" not in result.stdout


def test_help_lists_the_repository_and_domain_flags():
    result = _run(ENTRY, "--help")
    assert result.returncode == 0
    assert "--repo" in result.stdout
    assert "--domain" in result.stdout


def test_start_without_git_repository_reports_the_path(tmp_path: Path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    session_state = tmp_path / "state"
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--repo", str(plain)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FFAI_NO_REEXEC": "1",
            "FFAI_STATE_DIR": str(session_state),
            "FFAI_HISTORY_FILE": str(session_state / "history.json"),
        },
        cwd=str(plain),
        timeout=60,
    )
    assert result.returncode == 2
    assert str(plain) in result.stderr
    assert "git" in result.stderr


@pytest.mark.parametrize("reexec", [True, False])
def test_reexec_uses_the_venv_next_to_the_entry_point(tmp_path: Path, reexec: bool):
    script = tmp_path / "ff-ai.py"
    shutil.copy(ENTRY, script)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text('#!/bin/sh\necho "REEXEC $@"\n', encoding="utf-8")
    venv_python.chmod(venv_python.stat().st_mode | stat.S_IEXEC)

    # Без --version: он печатает версию раньше перезапуска, проверять надо обычный старт.
    result = _run(script, "--repo", str(tmp_path / "missing"), reexec=reexec)
    if reexec:
        assert "REEXEC" in result.stdout
    else:
        assert "REEXEC" not in result.stdout
        assert "не существует" in result.stderr
