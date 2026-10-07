#!/usr/bin/env python3
"""Фоновый исполнитель расписания ff.ai.

Отдельный процесс, потому что иначе «круглосуточно» не получается: MCP-клиент проекта одноразовый
(запуск процесса, рукопожатие, вызов, закрытие), поэтому цикл расписания не может жить внутри
сервера, а поток внутри приложения работал бы только при открытом приложении.

Исполнитель — обычный MCP-клиент: он зовёт у собственного сервера репозитория выполнение
просроченных заданий, то есть проверяет тот же путь, которым идут все прочие вызовы. Результаты
попадают в файл планировщика, откуда их читает приложение.

Запуск: `./ff-ai-scheduler.py --repo /path/to/aurora-project` (тик раз в минуту) или
`./ff-ai-scheduler.py --repo /path/to/aurora-project --once` — выполнить просроченное один раз
и выйти (этот режим гоняют тесты и демонстрация).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Та же причина, что и в ff-ai.py: `env python3` — системный интерпретатор, где нет ни зависимостей
# проекта, ни пакета mcp (он требует 3.10+).
_VENV_DIR = Path(__file__).resolve().parent / ".venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"


def _reexec_in_venv() -> None:
    """Перезапускает процесс интерпретатором проекта, если он лежит рядом с исполнителем."""
    if os.environ.get("FFAI_NO_REEXEC") == "1" or not _VENV_PYTHON.exists():
        return
    try:
        current = Path(sys.executable).resolve()
    except OSError:  # pragma: no cover - экзотическая платформа
        return
    if current == _VENV_PYTHON.resolve():
        return
    os.execv(
        str(_VENV_PYTHON), [str(_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    )

from core import config, domains  # noqa: E402  (импорт после возможного перезапуска)
from core.mcp_client import MCPClient, MCPError  # noqa: E402
from core.mcp_registry import repo_server_spec  # noqa: E402
from core.repo import RepoRootError, resolve_root  # noqa: E402

RUN_DUE_TOOL = "schedule_run_due"
# Пауза между проходами: минута — это и есть «ночное расписание» без спешки. Своего значения
# в `core/config.py` нет, потому что расписание целиком принадлежит исполнителю.
DEFAULT_TICK_SECONDS = 60


def build_client(root: Path, tools_path: Path | None, command: str = "") -> MCPClient:
    """Клиент собственного сервера репозитория; `--command` подменяет команду запуска.

    Запись реестра берётся у `core.mcp_registry`, а не у приложения: переопределение реестра
    переменной `FFAI_MCP_COMMAND` (им пользуются тесты) иначе увело бы исполнителя на чужой
    сервер, у которого нет ни заданий, ни расписания.
    """
    spec = repo_server_spec(root, tools_path)
    if command:
        spec = replace(spec, command=command)
    return MCPClient(spec)


def tick(client: MCPClient) -> bool:
    """Один проход расписания. False — проход не удался; цикл от этого не останавливается."""
    try:
        result = client.call_tool(RUN_DUE_TOOL, {})
    except MCPError as error:
        print(f"Планировщик: не удалось выполнить задания — {error}", flush=True)
        return False
    if result.is_error:
        print(f"Планировщик: не удалось выполнить задания — {result.text}", flush=True)
        return False
    print(result.text, flush=True)
    return True


def tools_path_for(root: Path) -> Path | None:
    """Файл белых списков пакета домена для целевого репозитория.

    Так исполнитель поднимает тот же сервер, что и приложение: сервер получает правила области
    применимости и умеет выполнять задания, а не только расписание.
    """
    try:
        return domains.select_domain(root=root).domain.tools_path
    except domains.DomainError:
        # Без пакета домена расписание работает: задания, которым нужны правила, назовут причину.
        return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ff-ai-scheduler", description="Фоновый исполнитель расписания ff.ai"
    )
    parser.add_argument("--repo", default="", help="целевой репозиторий (по умолчанию текущий каталог)")
    parser.add_argument(
        "--once", action="store_true", help="выполнить просроченные задания один раз и выйти"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_TICK_SECONDS,
        help=f"пауза между проходами в секундах (по умолчанию {DEFAULT_TICK_SECONDS})",
    )
    parser.add_argument("--command", default="", help="команда запуска сервера вместо записи реестра")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: list[str] | None = None) -> int:
    _reexec_in_venv()
    options = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        root = resolve_root(explicit=options.repo)
    except RepoRootError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    client = build_client(root, tools_path_for(root), options.command)
    if options.once:
        tick(client)
        return 0

    print(
        f"Планировщик запущен: проход каждые {options.interval} с, файл {config.SCHEDULE_FILE}. "
        "Остановка — Ctrl+C.",
        flush=True,
    )
    try:
        while True:
            tick(client)
            time.sleep(options.interval)
    except KeyboardInterrupt:
        print("Планировщик остановлен.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
