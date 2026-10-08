"""MCP в реальном приложении: сводка при старте, отчёт, обновление и ручной вызов инструмента.

Всё против локальных заглушек: обычный прогон не должен поднимать ни сервер документации портала,
ни собственный сервер репозитория (это делает `AppSession` по умолчанию).
"""

import json
import subprocess
import sys
from pathlib import Path

import time

import pytest

from .harness import AppSession

# Запас на обход реестра: он поднимает процесс сервера, поэтому бюджет задаётся по природе
# операции, а не по времени отрисовки экрана.
REFRESH_BUDGET = 60.0
WAIT_STEP = 15.0  # шаг ожидания: столько же, сколько даёт стенд по умолчанию

pytestmark = pytest.mark.e2e

FAKE_HTTP = Path(__file__).resolve().parent.parent / "fake_mcp_http.py"
TOOLS = ("fake_echo", "fake_sum", "fake_note", "fake_fail")


def _launch(tmp_path: Path, **kwargs) -> AppSession:
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=AppSession.create_target_repo(base=tmp_path),
        **kwargs,
    )


def test_startup_summary_names_servers_and_tools(app):
    screen = app.wait_for("MCP:")
    assert "1/1 серверов" in screen
    assert f"{len(TOOLS)} инструментов" in screen
    assert "/mcp" in screen  # подсказка, где подробности


def test_report_lists_server_protocol_and_tools(app, stub):
    app.send_line("/mcp")
    text = app.wait_for("протокол:")
    for name in TOOLS:
        assert name in text
    assert "параметры:" in text
    assert stub.call_count == 0


def test_refresh_walks_the_registry_again(app, stub):
    """Обход реестра — это запуск процесса сервера, поэтому ждём результат, а не таймер.

    Обычного запаса стенда здесь мало по природе операции: под нагрузкой запуск процесса и
    рукопожатие занимают больше секунд, чем отрисовка экрана. Поэтому команда повторяется, пока
    обход не завершится, и тест падает только если обход не удаётся вовсе.
    """
    deadline = time.monotonic() + REFRESH_BUDGET
    text = ""
    while time.monotonic() < deadline:
        app.send_line("/mcp refresh")
        try:
            text = app.wait_for("Реестр обойдён заново", timeout=WAIT_STEP)
            break
        except AssertionError:
            continue
    else:
        raise AssertionError(f"обход реестра не завершился за {REFRESH_BUDGET:g} с")
    assert "1/1 серверов" in text
    assert stub.call_count == 0


def test_manual_call_prints_the_result_of_the_tool(app, stub):
    app.send_line("/tool")
    listing = app.wait_for("fake_sum")
    assert "параметры:" in listing

    app.send_line("/tool call fake_echo message=привет")
    text = app.wait_for("привет")
    assert "fake_echo" in text
    assert stub.call_count == 0  # ручной вызов не обращается к модели


def test_unknown_tool_is_reported_and_the_session_continues(app, stub):
    stub.always_answer = None
    app.send_line("/tool call которого-нет")
    assert "не вызван" in app.wait_for("не вызван")
    assert app.ask("где инициализируется модель списка?")


def test_tool_text_stays_out_of_history(app, tmp_path: Path):
    app.send_line("/tool call fake_echo message=секретная-строка")
    app.wait_for("секретная-строка")
    history_file = tmp_path / "state" / "history.json"
    if history_file.exists():
        raw = history_file.read_text(encoding="utf-8")
    else:
        raw = ""
    assert "секретная-строка" not in raw


def test_unavailable_server_is_counted_and_the_session_works(tmp_path: Path, stub):
    session = _launch(tmp_path, api_url=stub.url, mcp_args="/нет/такого/сервера.py")
    try:
        screen = session.wait_for("MCP:")
        assert "0/1 серверов" in screen
        assert "1 недоступно" in screen
        session.send_line("/mcp")
        assert "недоступен" in session.wait_for("недоступен")
    finally:
        session.close()


def test_remote_server_over_address(tmp_path: Path, stub):
    process = subprocess.Popen(
        [sys.executable, str(FAKE_HTTP), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = process.stdout.readline().strip()
        assert line.startswith("URL="), line
        url = line[len("URL=") :]
        session = _launch(tmp_path, api_url=stub.url, mcp_url=url)
        try:
            screen = session.wait_for("MCP:")
            assert "1/1 серверов" in screen
            session.send_line("/tool call fake_echo message=через-адрес")
            assert "через-адрес" in session.wait_for("через-адрес")
        finally:
            session.close()
    finally:
        process.terminate()
        process.wait(timeout=10)
