#!/usr/bin/env python3
"""Процесс, который печатает мусор и завершается с кодом 0: проверка «не говорит по протоколу».

Нужен тестам клиента: подняв этот файл командой вместо MCP-сервера, тест проверяет, что
причиной отказа назван протокол (процесс не отвечает на рукопожатие), а не падение команды —
код возврата нулевой, и падать в смысле «команды нет» тут нечему.

Тот же режим доступен флагом `--garbage` у `tests/fake_mcp_server.py`.
"""

from __future__ import annotations

import sys
from typing import Sequence

GARBAGE_LINES: Sequence[str] = (
    "это не JSON-RPC, а обычный текст",
    "заглушка не отвечает на initialize",
    "конец вывода",
)


def main() -> int:
    for line in GARBAGE_LINES:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
