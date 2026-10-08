"""Локальная модель в живом приложении: автозапуск сервера, чужой порт и ответ без ключа.

Настоящий `llama-server` здесь не поднимается: модель на 3 ГБ и минуты загрузки не нужны ни для
проверки жизненного цикла (её берут юниты с поддельным скриптом), ни для проверки маршрута
вопроса. Приложение запускается в псевдотерминале, локальный адрес указывает на тот же
stub-сервер, что и облачный: разница между режимами — не в ответе, а в том, куда ушёл запрос.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Файл помечен slow: приложение поднимается в pty, и прогон занимает минуты.
pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from core import config

from .harness import AppSession

STARTUP_TIMEOUT = 30
LISTENER_MARKER = "занят процессом не llama-server"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _session(tmp_path: Path, stub_url: str, extra_env: dict, api_key: str = None) -> AppSession:
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=AppSession.create_target_repo(base=tmp_path),
        api_url=stub_url,
        api_key=api_key,
        extra_env=extra_env,
    )


@pytest.fixture
def foreign_listener(tmp_path: Path):
    """Посторонний процесс, занявший локальный порт: его не должен трогать ни один сигнал."""
    port = free_port()
    worker = tmp_path / "foreign_listener.py"
    worker.write_text(
        "import socketserver, sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "# Тот же обход обратного DNS, что и в юнитах: на macOS PTR 127.0.0.1 уходит к внешнему\n"
        "# резолверу и не отвечает за бюджет готовности — порт занят, но никто его не слушает.\n"
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
    process = subprocess.Popen(
        [sys.executable, str(worker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("Посторонний слушатель не поднялся")
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    else:
        process.kill()
        raise AssertionError("Посторонний слушатель не занял порт за 20 с")
    try:
        yield port, process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def test_autostart_off_leaves_the_session_cloud_only(tmp_path: Path, stub):
    """Выключенный автозапуск не запускает сервер и не печатает про него ничего."""
    app = _session(tmp_path, stub.url, {"FFAI_LLAMA_AUTOSTART": "0"})
    try:
        app.wait_for("MCP:", timeout=STARTUP_TIMEOUT)
        app.wait_for_prompt(timeout=STARTUP_TIMEOUT)
        app.read_for(0.5)

        assert "llama-server" not in app.scrollback()

        app.ask("Как собрать rpm-пакет?")
        assert "Ответ stub-сервера." in app.scrollback()
    finally:
        app.close()


def test_foreign_listener_on_the_local_port_is_reported_and_kept(
    tmp_path: Path, stub, foreign_listener
):
    """Порт занят чужим процессом: причина на экране, чужой процесс жив, сессия работает."""
    port, process = foreign_listener
    app = _session(
        tmp_path,
        stub.url,
        {
            "FFAI_LLAMA_AUTOSTART": "1",
            "FFAI_LOCAL_API_URL": f"http://127.0.0.1:{port}/v1/chat/completions",
        },
    )
    try:
        app.wait_for("Локальный сервер недоступен", timeout=STARTUP_TIMEOUT)
        assert LISTENER_MARKER in app.scrollback()
        app.wait_for_prompt(timeout=STARTUP_TIMEOUT)

        app.ask("Как собрать rpm-пакет?")
        assert "Ответ stub-сервера." in app.scrollback()

        app.send_line("/exit")
        app.wait_exit()
    finally:
        app.close()

    assert process.poll() is None, "посторонний процесс на порту не должен быть остановлен"


def test_local_preset_answers_without_a_cloud_key(tmp_path: Path, stub):
    """Локальный пресет отвечает без облачного ключа и без заголовка Authorization."""
    local_model = config.LOCAL_MODELS[-1]
    app = _session(
        tmp_path,
        stub.url,
        {"FFAI_LLAMA_AUTOSTART": "0", "FFAI_LOCAL_API_URL": stub.url},
        api_key="",  # ключа нет вовсе: локальному режиму он не нужен
    )
    try:
        app.wait_for("MCP:", timeout=STARTUP_TIMEOUT)
        app.wait_for_prompt(timeout=STARTUP_TIMEOUT)

        app.send_line("/models")
        app.wait_on_screen("Enter — применить", timeout=STARTUP_TIMEOUT)
        # Локальные пресеты дописаны в конец списка: одно нажатие вверх — последний из них.
        app.send_key(b"\x1b[A")
        time.sleep(0.15)
        app.send_key(b"\r")
        app.wait_for(f"Модель: {local_model}", timeout=STARTUP_TIMEOUT)
        app.wait_for_prompt(timeout=STARTUP_TIMEOUT)

        app.ask("Где в проекте читается список моделей?")

        assert stub.last_payload()["model"] == local_model
        assert "Authorization" not in stub.requests[-1]["headers"]
        app.send_line("/exit")
        app.wait_exit()
    finally:
        app.close()
