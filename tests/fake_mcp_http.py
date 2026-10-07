#!/usr/bin/env python3
"""Тот же набор инструментов, что в `tests/fake_mcp_server.py`, но поднятый по адресу.

Слушает `127.0.0.1` на порту из `--port` (по умолчанию 0 — свободный порт выбирает ядро),
печатает в stdout одну строку `URL=http://127.0.0.1:<порт>/mcp` и продолжает работу, пока
процесс не остановят: тест читает эту строку, узнаёт адрес и идёт к нему клиентом.

Порт берётся из уже привязанного сокета, а не из `MCPServer.run(...)`: с `port=0` занятый
порт знает только ядро, а `run()` не отдаёт наружу ни выбранный порт, ни сокет. Поэтому
приложение собирается тем же `MCPServer`, но обслуживается `uvicorn` напрямую — тем же
сервером, который `MCPServer.run(transport="streamable-http")` поднял бы сам.

Модуль зависит от пакета `mcp` (и его зависимости `uvicorn`) и стандартной библиотеки; ядро и
интерфейс приложения не трогает.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Optional, Sequence

# Каталог `tests` в пути импорта: файл запускают и скриптом (`python tests/fake_mcp_http.py`,
# тогда каталог уже первый в `sys.path`), и как модуль (`python -m tests.fake_mcp_http`, тогда
# его там нет), а набор инструментов у обоих режимов обязан быть один и тот же.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_mcp_server import build_server, reexec_in_venv  # noqa: E402

HOST = "127.0.0.1"
MCP_PATH = "/mcp"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fake_mcp_http.py",
        description="Заглушка MCP-сервера по адресу (streamable HTTP) для тестов клиента.",
    )
    parser.add_argument("--port", type=int, default=0, help="порт на 127.0.0.1; 0 — свободный")
    parser.add_argument("--empty", action="store_true", help="не объявлять ни одного инструмента")
    return parser.parse_args(argv)


def serve(port: int, empty: bool = False) -> int:
    """Привязывает сокет, печатает адрес и обслуживает его до остановки процесса."""
    # Импорт внутри функции: перезапуск в `.venv` (в `main`) должен случиться раньше, иначе
    # системный `python3` без `uvicorn` падает на импорте до всякой логики.
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, port))
    actual_port = sock.getsockname()[1]

    # Уровень в `MCPServer` — литерал из имён в верхнем регистре (в нижний его приводит сам
    # `MCPServer.run`, передавая значение в uvicorn).
    server = build_server(log_level="WARNING", empty=empty)
    app = server.streamable_http_app(streamable_http_path=MCP_PATH, host=HOST)

    # Строка адреса — единственный контракт с тестом, поэтому она выходит до начала обслуживания
    # и с принудительным сбросом буфера: stdout у процесса-родителя обычно канал, а не терминал.
    print(f"URL=http://{HOST}:{actual_port}{MCP_PATH}", flush=True)

    config = uvicorn.Config(app, host=HOST, port=actual_port, log_level="warning")
    uvicorn.Server(config).run(sockets=[sock])
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    reexec_in_venv(Path(__file__))
    return serve(args.port, empty=args.empty)


if __name__ == "__main__":
    sys.exit(main())
