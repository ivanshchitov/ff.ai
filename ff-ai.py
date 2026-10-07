#!/usr/bin/env python3
"""Точка входа ff.ai — ассистента по разработке приложений под ОС Аврора.

Файл намеренно запускается только напрямую: дефис в имени не даёт импортировать его как
модуль. Если рядом лежит окружение `.venv`, интерпретатор из него перезапускает этот же файл:
у системного питона нет зависимостей проекта (а `mcp` требует 3.10+), у `.venv` они есть.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
__version__ = "0.1"


def _reexec_in_venv() -> None:
    """Перезапускает процесс интерпретатором проекта, если он лежит рядом с точкой входа."""
    if os.environ.get("FFAI_NO_REEXEC") == "1" or not VENV_PYTHON.exists():
        return
    try:
        current = Path(sys.executable).resolve()
    except OSError:  # pragma: no cover - экзотическая платформа
        return
    if current == VENV_PYTHON.resolve():
        return
    os.execv(
        str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    )


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ff-ai",
        description="Ассистент по разработке: вопросы по репозиторию и по платформе, задачи по коду.",
    )
    parser.add_argument("--repo", default="", help="целевой репозиторий (по умолчанию текущий каталог)")
    parser.add_argument("--domain", default="", help="пакет домена; по умолчанию — по маркерам репозитория")
    parser.add_argument("--version", action="store_true", help="показать версию и выйти")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        print(f"ff.ai {__version__}")
        return 0

    _reexec_in_venv()

    from core.domains import DomainError
    from core.repo import RepoRootError, resolve_root
    from core.session import AssistantSession
    from ui.tui_app import DevAssistantTUI

    try:
        root = resolve_root(explicit=args.repo)
    except RepoRootError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    try:
        session = AssistantSession(root=root, domain_id=args.domain or None)
    except DomainError as error:
        print(f"Ошибка домена: {error}", file=sys.stderr)
        return 2

    return DevAssistantTUI(session=session).run()


if __name__ == "__main__":
    sys.exit(main())
