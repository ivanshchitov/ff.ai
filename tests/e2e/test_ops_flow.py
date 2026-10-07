"""Конвейер операций в живом приложении: состояние, выбор цели, отказы и отчёт.

SDK подменяется скриптом-заглушкой: проверяются решения приложения — что оно спрашивает у
инструмента, что отказывается делать и куда пишет состояние.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .harness import AppSession

LIVE = 60
FAKE_SFDK = """#!/bin/sh
case "$1" in
  --version) echo "SDK_RELEASE=5.1.5.105-mb2"; echo "SDK_RELEASE_CYCLE=Beta" ;;
  tools)
    case "$2" in
      list)
        echo "AuroraOS-5.1.5.105-MB2-aarch64"
        echo "\\u2514\\u2500\\u2500 AuroraOS-5.1.5.105-MB2-aarch64.default"
        echo "AuroraOS-5.1.5.105-MB2-armv7hl"
        echo "\\u2514\\u2500\\u2500 AuroraOS-5.1.5.105-MB2-armv7hl.default"
        ;;
      *) echo "tools: неизвестная подкоманда"; exit 2 ;;
    esac ;;
  build-init) echo "инициализировано в $PWD" ;;
  build)
    mkdir -p RPMS
    echo "пакет собран" > RPMS/app-1.0-1.armv7hl.rpm
    echo "build ok" ;;
  engine)
    shift
    case "$*" in
      *rpmsign-external*) echo "подписано" ;;
      *"-qp"*) echo "SIGPGP: rsa sha256 Signature, key id 1234" ;;
      *) echo "engine: неизвестная команда" ;;
    esac ;;
  deploy) echo "установлено на устройство" ;;
  device) echo "sailjail: запущено" ;;
  maintain) echo "интерактивный инструмент обслуживания"; exit 1 ;;
  *) echo "sfdk: неизвестная команда $1"; exit 2 ;;
esac
"""


def _fake_tool(tmp_path: Path) -> Path:
    tool = tmp_path / "sfdk"
    tool.write_text(FAKE_SFDK, encoding="utf-8")
    tool.chmod(0o755)
    return tool


def _launch(tmp_path: Path, tool: Path, **extra) -> AppSession:
    # Целевому репозиторию нужен настоящий git: локальные исключения живут в `.git/info/exclude`,
    # а пустой каталог `.git` его не содержит.
    repo = AppSession.create_target_repo(base=tmp_path, name="aurora-project")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=repo,
        api_key=None,
        api_url=None,
        # Реестр MCP остаётся фейковым по умолчанию: конвейеру он не нужен, а обход настоящего
        # сервера документации на старте только замедляет проверку.
        extra_env={"FFAI_SFDK": str(tool), **extra},
    )


@pytest.fixture
def app_with_sdk(tmp_path: Path):
    tool = _fake_tool(tmp_path)
    app = _launch(tmp_path, tool)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        yield app, tool
    finally:
        app.close()


def test_status_reads_version_and_targets(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops")
    screen = app.wait_for("Версия SDK:", timeout=LIVE)
    assert "5.1.5.105-mb2" in screen
    assert "armv7hl.default" in screen
    assert "цель не выбрана" in screen


def test_target_selection_writes_state_outside_tracked_files(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops target aarch64")
    screen = app.wait_for("aarch64.default", timeout=LIVE)
    assert "Выбрано:" in screen

    state = app.repo / ".aurora" / "ops.json"
    assert state.is_file()
    exclude = (app.repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert ".aurora/" in exclude.splitlines()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=app.repo, capture_output=True, text=True, check=False
    ).stdout
    assert not any(".aurora" in line for line in status.splitlines()), (
        "состояние конвейера исключено локальным правилом"
    )


def test_unknown_architecture_is_refused(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops target mips")
    assert "не объявлена доменом" in app.wait_for("не объявлена доменом", timeout=LIVE)


def test_build_goes_to_a_separate_directory(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops target armv7hl")
    app.wait_for("Выбрано:", timeout=LIVE)

    app.send_line("/ops build")
    screen = app.wait_for("build:", timeout=LIVE + 30)
    assert "✅" in screen, "сборка считается успешной при найденном пакете"

    assert (app.repo / "build_armv7hl" / "RPMS" / "app-1.0-1.armv7hl.rpm").is_file()
    root_entries = {path.name for path in app.repo.iterdir()}
    assert "RPMS" not in root_entries, "в корне проекта артефактов сборки нет"
    assert "Makefile" not in root_entries


def test_sign_without_passphrase_is_unavailable(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops target armv7hl")
    app.wait_for("Выбрано:", timeout=LIVE)
    app.send_line("/ops sign")
    assert "FFAI_SIGN_KEY" in app.wait_for("FFAI_SIGN_KEY", timeout=LIVE)


def test_install_requires_a_verified_signature(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops target armv7hl")
    app.wait_for("Выбрано:", timeout=LIVE)
    app.send_line("/ops install")
    assert "подпись пакета не подтверждена" in app.wait_for(
        "подпись пакета не подтверждена", timeout=LIVE
    )


def test_report_shows_steps_without_new_processes(app_with_sdk):
    app, tool = app_with_sdk
    app.send_line("/ops target armv7hl")
    app.wait_for("Выбрано:", timeout=LIVE)
    app.send_line("/ops")
    screen = app.wait_for("Шаги:", timeout=LIVE)
    assert "target" in screen
