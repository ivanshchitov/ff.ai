#!/usr/bin/env python3
"""Локальная заглушка MCP-сервера по stdio: тесты клиента берут её, а не сеть.

Инструменты нарочно разные по назначению, чтобы каждый проверял своё:

- `fake_echo` — возвращает переданный текст: инструмент проверки соединения;
- `fake_sum` — складывает два целых: проверка параметров схемы инструмента;
- `fake_note` — возвращает текст с символами rich-разметки (`[bold]`, `[/dim]`) без изменений:
  проверка экранирования чужого текста в интерфейсе;
- `fake_fail` — всегда ошибочный результат (`is_error`): проверка ветки ошибки инструмента.

Режимы через argv (по умолчанию — все четыре инструмента):

- `--empty` — ни одного инструмента: пустой список это успех, а не сбой;
- `--garbage` — печатает мусор вместо протокола и выходит с кодом 0 (то же делает
  `tests/fake_mcp_garbage.py`): проверка «процесс не говорит по протоколу».

Модуль зависит от пакета `mcp` и стандартной библиотеки; ядро и интерфейс приложения не трогает.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # только для аннотаций: импорт `mcp` отложен до перезапуска в .venv
    from mcp.server.mcpserver import MCPServer

SERVER_NAME = "fake-mcp"
SERVER_VERSION = "1.0"
SERVER_INSTRUCTIONS = (
    "Заглушка MCP-сервера для тестов клиента: возврат текста, сумма двух чисел, текст с "
    "rich-разметкой и нарочная ошибка."
)

TOOL_NAMES = ("fake_echo", "fake_sum", "fake_note", "fake_fail")

# Мусор вместо протокола: строки не являются JSON-RPC, но процесс завершается успешно —
# клиент должен увидеть не «сервер упал», а «процесс не говорит по протоколу».
GARBAGE_LINES: Sequence[str] = (
    "это не JSON-RPC, а обычный текст",
    "заглушка не отвечает на initialize",
    "конец вывода",
)

BASE_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"


def reexec_in_venv(script: Path) -> None:
    """Перезапускает процесс интерпретатором проекта, если он лежит рядом с репозиторием.

    Тестовый реестр объявляет команду запуска как `python3` (см. `tests/conftest.py`), а
    системный питон на macOS — 3.9 без пакета `mcp`: без перезапуска заглушка падала бы на
    импорте и выглядела бы сломанным сервером вместо ошибки окружения. Тот же приём, что
    в `ff-ai.py`. Скрипт передаётся явно: `__file__` этой функции — не тот файл, которым
    запущен процесс, когда её зовёт `fake_mcp_http`.
    """
    if os.environ.get("FFAI_NO_REEXEC") == "1" or not VENV_PYTHON.exists():
        return
    try:
        current = Path(sys.executable).resolve()
    except OSError:  # pragma: no cover - экзотическая платформа
        return
    if current == VENV_PYTHON.resolve():
        return
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(script).resolve()), *sys.argv[1:]],
    )


def build_server(log_level: str = "INFO", empty: bool = False) -> "MCPServer":
    """Сервер с инструментами заглушки; `empty=True` — совсем без инструментов."""
    # Импорт внутри функции, а не сверху модуля: перезапуск в `.venv` должен случиться раньше,
    # иначе системный `python3` (3.9, без `mcp`) падает на импорте до всякой логики.
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        log_level=log_level,
    )
    if empty:
        return server

    @server.tool()
    def fake_echo(message: str) -> str:
        """Возвращает переданный текст: инструмент проверки соединения."""
        return message

    @server.tool()
    def fake_sum(a: int, b: int) -> str:
        """Складывает два целых числа: проверка параметров схемы инструмента."""
        return str(a + b)

    @server.tool()
    def fake_note(text: str) -> str:
        """Возвращает текст как есть, вместе с символами rich-разметки: проверка экранирования."""
        return text

    @server.tool()
    def fake_fail(reason: str = "нарочная ошибка") -> str:
        """Всегда отдаёт ошибочный результат с переданной причиной: проверка признака ошибки."""
        raise ToolError(reason)

    return server


def print_garbage() -> None:
    """Пишет мусор в stdout и ничего больше не делает: процесс выйдет с кодом 0."""
    for line in GARBAGE_LINES:
        print(line, flush=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fake_mcp_server.py",
        description="Заглушка MCP-сервера по stdio для тестов клиента.",
    )
    parser.add_argument("--empty", action="store_true", help="не объявлять ни одного инструмента")
    parser.add_argument(
        "--garbage",
        action="store_true",
        help="напечатать мусор вместо протокола и выйти с кодом 0",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    reexec_in_venv(Path(__file__))

    if args.garbage:
        print_garbage()
        return 0

    server = build_server(empty=args.empty)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
