#!/usr/bin/env python3
"""Точка входа ff.ai — ассистента по разработке приложений под ОС Аврора.

Файл намеренно запускается только напрямую: дефис в имени не даёт импортировать его как
модуль. Если рядом лежит окружение `.venv`, интерпретатор из него перезапускает этот же файл:
у системного питона нет зависимостей проекта (а `mcp` требует 3.10+), у `.venv` они есть.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"

# Запуск «как есть» (./ff-ai.py) идёт через `env python3`, то есть интерпретатором системы: у него
# нет зависимостей проекта, и первый же импорт из `core` падает. Поэтому перезапускаем себя
# интерпретатором окружения — до любых импортов проекта, иначе перезапуск не успевает случиться.
# Признак «мы уже внутри окружения» — `sys.prefix`, а не путь исполняемого файла: `.venv/bin/python`
# сам по себе симлинк на базовый интерпретатор, и сравнение разрешённых путей считало бы запуск
# системным интерпретатором запуском изнутри окружения, поэтому перезапуск не срабатывал.
if (
    VENV_PYTHON.exists()
    and Path(sys.prefix) != VENV_DIR
    and os.environ.get("FFAI_NO_REEXEC") != "1"
):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
__version__ = "0.1"
# Сигналы, по которым сервер нужно убрать за собой: SIGKILL не перехватывается, это граница.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ff-ai",
        description="Ассистент по разработке: вопросы по репозиторию и по платформе, задачи по коду.",
    )
    parser.add_argument("--repo", default="", help="целевой репозиторий (по умолчанию текущий каталог)")
    parser.add_argument("--domain", default="", help="пакет домена; по умолчанию — по маркерам репозитория")
    parser.add_argument(
        "--web",
        action="store_true",
        help="поднять тонкий web-фронтенд вместо консольного интерфейса",
    )
    parser.add_argument("--web-host", default="127.0.0.1", help="адрес web-сервера")
    parser.add_argument("--web-port", type=int, default=8000, help="порт web-сервера")
    parser.add_argument("--web-base-url", default="", help="внешний префикс адреса, если нужен")
    parser.add_argument("--version", action="store_true", help="показать версию и выйти")
    return parser.parse_args(argv)


def _run_with_local_server(tui) -> int:
    """Поднимает локальный llama-server до интерфейса и убирает его при любом выходе.

    Локальная модель — это режим без облачного ключа и без сети, поэтому сервер поднимает само
    приложение; интерфейсу остаётся только работать. Остановка в `finally` нужна и при обычном
    выходе, и при SIGTERM/SIGHUP: иначе загруженная модель осталась бы в памяти после закрытия
    терминала. Сбой запуска — не отказ приложения: облачные модели работают и без сервера,
    поэтому причина печатается, а сессия продолжается.
    """
    # Импорт внутри функции: точка входа сначала перезапускает себя интерпретатором `.venv`,
    # и только у него есть зависимости проекта.
    from rich.markup import escape

    from core.llama_server import LlamaServer, is_autostart_enabled

    server = LlamaServer()
    enabled = is_autostart_enabled()
    previous_handlers = {}

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for sig in STOP_SIGNALS:
            previous_handlers[sig] = signal.signal(sig, terminate)
        if enabled:
            # Загрузка весов идёт минутами, поэтому о начале сообщаем до ожидания готовности.
            tui.console.print(f"[cyan]Запуск локального llama-server на {server.base_url}...[/cyan]")
        try:
            server.start()
        except (RuntimeError, OSError) as error:
            # Текст ошибки приходит из чужого процесса и из лога сервера: это данные.
            tui.console.print(
                f"[yellow]Локальный сервер недоступен: {escape(str(error))}[/yellow]"
            )
        else:
            if enabled:
                tui.console.print("[green]Локальный llama-server готов.[/green]")
        return tui.run()
    finally:
        try:
            server.stop()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        print(f"ff.ai {__version__}")
        return 0

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

    if args.web:
        # Web-режим ведёт свою сессию, а жизненный цикл локального сервера остаётся у TUI-пути:
        # в браузере локальная модель работает через тот же фасад, что и в терминале.
        from web.app import serve

        serve(session, host=args.web_host, port=args.web_port, base_url=args.web_base_url)
        return 0
    return _run_with_local_server(DevAssistantTUI(session=session))


if __name__ == "__main__":
    sys.exit(main())
