"""Жизненный цикл локального llama-server: запуск, переиспользование, остановка, ошибки.

Проверки идут против настоящего локального HTTP-процесса, но без загрузки весов: скрипт запуска
подменяется мини-сервером с `/health`, а `llama-server` для проверки самого `server.sh` —
поддельным исполняемым файлом, который только печатает свои аргументы.
"""

from __future__ import annotations

import importlib.util
import io
import os
import socket
import sys
from pathlib import Path
from typing import List

import pytest
import requests
from rich.console import Console

from core import config
from core.llama_server import (
    AUTOSTART_ENV,
    START_TIMEOUT_ENV,
    LlamaServer,
    is_autostart_enabled,
    start_timeout,
)

# Бюджет готовности в тестах — только потолок против зависания: bash и python на общем раннере
# стартуют заметно медленнее, чем на рабочей машине.
TEST_TIMEOUT = 15.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def autostart_enabled(monkeypatch, isolated_state):
    """Включает автозапуск для проверок самого механизма.

    Общий слой тестов его выключает (`isolated_state`), потому что `llama-server` установлен на
    машине и юнит-прогон поднял бы настоящую модель; здесь сервером управляет сам тест — либо
    поддельным скриптом, либо явным `FFAI_LLAMA_AUTOSTART=0` в нужном тесте.
    """
    monkeypatch.delenv(AUTOSTART_ENV, raising=False)


@pytest.fixture
def server_script(tmp_path: Path):
    """Скрипт запуска и адрес поддельного сервера: процесс живёт, пока его не остановят."""
    port = free_port()
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import socketserver\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "# HTTPServer.server_bind зовёт socket.getfqdn(host) — обратный DNS-запрос — уже после\n"
        "# bind(), но до listen(). На Linux 127.0.0.1 отвечает /etc/hosts, на macOS PTR идёт к\n"
        "# внешнему резолверу, и на раннере он не отвечает за бюджет готовности: порт занят, но\n"
        "# никто его не слушает, и старт падает по таймауту. Имя сервера тесту не нужно.\n"
        "class Server(HTTPServer):\n"
        " def server_bind(self):\n"
        "  socketserver.TCPServer.server_bind(self)\n"
        "  self.server_name, self.server_port = self.server_address[:2]\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        "  self.send_response(200); self.end_headers(); self.wfile.write(b'{\"status\":\"ok\"}')\n"
        " def log_message(self, *args): pass\n"
        f"Server(('127.0.0.1', {port}), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    script = tmp_path / "server.sh"
    import shlex

    script.write_text(
        f"#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(worker))}\n",
        encoding="utf-8",
    )
    return script, f"http://127.0.0.1:{port}"


def manager(server_script, tmp_path: Path, **kwargs) -> LlamaServer:
    script, url = server_script
    return LlamaServer(
        script=script,
        base_url=url,
        log_path=tmp_path / "server.log",
        timeout=TEST_TIMEOUT,
        **kwargs,
    )


# --- адрес, бюджет и настройка автозапуска ----------------------------------------------------


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("0", False), ("false", False), ("нет", False), ("NO", False)]
)
def test_autostart_flag_reads_explicit_values(monkeypatch, value, expected):
    monkeypatch.setenv(AUTOSTART_ENV, value)
    assert is_autostart_enabled() is expected


def test_autostart_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv(AUTOSTART_ENV, raising=False)
    assert is_autostart_enabled() is True


def test_readiness_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(START_TIMEOUT_ENV, "5.5")
    assert start_timeout() == pytest.approx(5.5)

    monkeypatch.setenv(START_TIMEOUT_ENV, "soon")
    with pytest.raises(RuntimeError, match=START_TIMEOUT_ENV):
        start_timeout()

    monkeypatch.setenv(START_TIMEOUT_ENV, "0")
    with pytest.raises(RuntimeError, match=START_TIMEOUT_ENV):
        start_timeout()


def test_address_comes_from_the_local_api_setting(monkeypatch):
    """Порт не задаётся вторым числом: сервер обязан слушать там, куда приложение обращается."""
    monkeypatch.setattr(config, "LOCAL_API_URL", "http://127.0.0.1:12345/v1/chat/completions")
    assert LlamaServer().base_url == "http://127.0.0.1:12345"


def test_address_without_netloc_is_reported(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_API_URL", "/v1/chat/completions")
    with pytest.raises(RuntimeError, match="локального API"):
        LlamaServer()


# --- запуск, переиспользование, остановка -----------------------------------------------------


def test_start_waits_for_health_and_stop_releases_listener(server_script, tmp_path):
    server = manager(server_script, tmp_path)
    try:
        server.start()
        assert requests.get(server.base_url + "/health", timeout=1).status_code == 200
        process = server.process
    finally:
        server.stop()

    assert process is not None and process.poll() is not None
    with pytest.raises(requests.ConnectionError):
        requests.get(server.base_url + "/health", timeout=1)
    server.stop()  # повторное завершение безопасно


def test_early_process_exit_reports_log_and_cleans_up(server_script, tmp_path):
    script, _ = server_script
    script.write_text("echo 'bad model preset'\nexit 7\n", encoding="utf-8")
    server = manager(server_script, tmp_path)

    with pytest.raises(RuntimeError, match="bad model preset"):
        server.start()

    assert server.process is not None and server.process.poll() == 7


def test_start_timeout_terminates_spawned_process(server_script, tmp_path):
    script, _ = server_script
    script.write_text("exec sleep 30\n", encoding="utf-8")
    server = manager(server_script, tmp_path)
    server.timeout = 0.1

    with pytest.raises(RuntimeError, match="готовности"):
        server.start()

    assert server.process is not None and server.process.poll() is not None


def test_keyboard_interrupt_during_start_also_stops_process(server_script, tmp_path, monkeypatch):
    """Ctrl+C во время загрузки весов не оставляет модель в памяти."""
    server = manager(server_script, tmp_path)
    monkeypatch.setattr(server, "_ready", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        server.start()

    assert server.process is not None and server.process.poll() is not None


def test_existing_server_is_reused_and_stopped(server_script, tmp_path, monkeypatch):
    first = manager(server_script, tmp_path)
    second = manager(server_script, tmp_path)
    try:
        first.start()
        assert first.process is not None
        monkeypatch.setattr(second, "_listener_pid", lambda: first.process.pid)
        second.start()

        assert second.process is None  # второго экземпляра сервера не появилось
        second.stop()
        first.process.wait(timeout=3)
    finally:
        first.stop()

    assert first.process is not None and first.process.poll() is not None


def test_reused_server_is_stopped_even_when_health_stops_answering(server_script, tmp_path, monkeypatch):
    """Сервер мог перестать отвечать на health, но порт и память всё ещё его."""
    script, _ = server_script
    worker = script.parent / "worker.py"
    worker.write_text(
        "import signal, time, sys\n"
        "def stop(signum, frame):\n"
        " time.sleep(0.4)\n"
        " sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n" + worker.read_text(),
        encoding="utf-8",
    )
    first = manager(server_script, tmp_path)
    second = manager(server_script, tmp_path)
    try:
        first.start()
        assert first.process is not None
        monkeypatch.setattr(second, "_listener_pid", lambda: first.process.pid)
        second.start()
        monkeypatch.setattr(second, "_ready", lambda: False)

        second.stop()

        assert first.process.poll() is not None
    finally:
        first.stop()


def test_unrelated_listener_is_not_stopped(server_script, tmp_path):
    """Чужой процесс на порту — ошибка с причиной; сигнал ему не отправляется."""
    first = manager(server_script, tmp_path)
    second = manager(server_script, tmp_path)
    try:
        first.start()
        with pytest.raises(RuntimeError, match="не llama-server"):
            second.start()

        assert second.process is None
        second.stop()

        assert first.process is not None and first.process.poll() is None
    finally:
        first.stop()


def test_disabled_autostart_does_not_spawn(server_script, tmp_path, monkeypatch):
    server = manager(server_script, tmp_path)
    monkeypatch.setenv(AUTOSTART_ENV, "0")

    server.start()
    server.stop()

    assert server.process is None
    assert not (tmp_path / "server.log").exists()


# --- скрипт запуска ---------------------------------------------------------------------------


def test_launch_script_serves_the_preset_file_beside_it(tmp_path):
    """`server.sh` зовёт llama-server на том же порту и с тем же файлом пресетов, что и чат."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorded = tmp_path / "argv.txt"
    fake = fake_bin / "llama-server"
    fake.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {recorded}\n', encoding="utf-8")
    fake.chmod(0o755)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    import subprocess

    subprocess.run(
        ["bash", str(config.BASE_DIR / "llama_server" / "server.sh")],
        cwd=elsewhere,
        check=True,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"},
        timeout=30,
    )

    arguments: List[str] = recorded.read_text(encoding="utf-8").split()
    assert arguments[0] == "--host"
    assert arguments[1] == "127.0.0.1"
    assert "--port" in arguments
    assert arguments[arguments.index("--port") + 1] == "9999"
    assert arguments[arguments.index("--models-preset") + 1] == str(
        config.BASE_DIR / "llama_server" / "models.ini"
    )
    assert (config.BASE_DIR / "llama_server" / "models.ini").is_file()
    # Одна модель в памяти: роутер не поднимает весь список пресетов сразу.
    assert arguments[arguments.index("--models-max") + 1] == "1"


# --- точка входа ------------------------------------------------------------------------------

def entry_module():
    spec = importlib.util.spec_from_file_location("ff_ai_entry", config.BASE_DIR / "ff-ai.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTUI:
    """Интерфейс, который ничего не рисует: проверяется только порядок запуска и остановки."""

    def __init__(self, on_run=None):
        self.console = Console(file=io.StringIO(), width=100)
        self._on_run = on_run
        self.ran = False

    def run(self) -> int:
        self.ran = True
        if self._on_run is not None:
            return self._on_run(self)
        return 0

    def text(self) -> str:
        stream = self.console.file
        assert isinstance(stream, io.StringIO)
        return stream.getvalue()


def test_local_server_starts_before_the_interface_and_stops_after(
    server_script, tmp_path, monkeypatch
):
    server = manager(server_script, tmp_path)
    monkeypatch.setattr("core.llama_server.LlamaServer", lambda: server)
    tui = FakeTUI(
        on_run=lambda _tui: 200 if requests.get(server.base_url + "/health", timeout=1) else 0
    )

    code = entry_module()._run_with_local_server(tui)

    assert code == 200, "сервер должен быть готов до первого кадра интерфейса"
    assert server.process is not None and server.process.poll() is not None
    assert "готов" in tui.text()


@pytest.mark.parametrize("error", [None, KeyboardInterrupt, RuntimeError])
def test_interface_exit_always_stops_the_server(server_script, tmp_path, monkeypatch, error):
    server = manager(server_script, tmp_path)
    monkeypatch.setattr("core.llama_server.LlamaServer", lambda: server)

    def run(_tui):
        if error is not None:
            raise error()
        return 0

    tui = FakeTUI(on_run=run)
    if error is None:
        assert entry_module()._run_with_local_server(tui) == 0
    else:
        with pytest.raises(error):
            entry_module()._run_with_local_server(tui)

    assert server.process is not None and server.process.poll() is not None


def test_unavailable_server_is_reported_and_the_session_continues(tmp_path, monkeypatch):
    """Сбой запуска — сообщение, а не отказ: облачные модели работают и без сервера."""
    script = tmp_path / "server.sh"
    script.write_text("echo 'llama-server: command not found'\nexit 127\n", encoding="utf-8")
    server = LlamaServer(
        script=script,
        base_url=f"http://127.0.0.1:{free_port()}",
        log_path=tmp_path / "server.log",
        timeout=5.0,
    )
    monkeypatch.setattr("core.llama_server.LlamaServer", lambda: server)
    tui = FakeTUI()
    entry_module()._run_with_local_server(tui)

    # Rich переносит длинные строки, поэтому сверяется текст с схлопнутыми пробелами.
    printed = " ".join(tui.text().split())
    assert "Локальный сервер недоступен" in printed
    assert "command not found" in printed
    assert tui.ran is True
    assert server.process is not None and server.process.poll() is not None


def test_autostart_off_prints_nothing_and_touches_nothing(server_script, tmp_path, monkeypatch):
    server = manager(server_script, tmp_path)
    monkeypatch.setattr("core.llama_server.LlamaServer", lambda: server)
    monkeypatch.setenv(AUTOSTART_ENV, "0")
    tui = FakeTUI()

    entry_module()._run_with_local_server(tui)

    assert tui.ran is True
    assert server.process is None
    assert "llama-server" not in tui.text()


def test_entry_point_runs_the_interface_through_the_local_server(tmp_path, monkeypatch):
    """Точка входа обязана идти через сервер: без этого локальный режим остаётся именами."""
    import core.session
    import ui.tui_app

    repo = tmp_path / "aurora-project"
    (repo / ".git").mkdir(parents=True)
    calls = []

    class FakeSession:
        def __init__(self, **kwargs):
            calls.append(("session", kwargs))

    class FakeApp:
        def __init__(self, session):
            calls.append(("tui", session))

    monkeypatch.setattr(core.session, "AssistantSession", FakeSession)
    monkeypatch.setattr(ui.tui_app, "DevAssistantTUI", FakeApp)
    entry = entry_module()
    monkeypatch.setattr(entry, "_run_with_local_server", lambda tui: calls.append(("run", tui)) or 0)

    assert entry.main(["--repo", str(repo)]) == 0

    assert [step for step, _ in calls] == ["session", "tui", "run"]
