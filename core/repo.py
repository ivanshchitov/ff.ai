"""Целевой репозиторий: какой каталог обслуживает ассистент и что считается его границей.

Корень задаётся явно (`--repo`), переменной окружения или текущим каталогом; от указанного
каталога вверх ищется ближайший git-репозиторий — так же, как это делает сам git, поэтому
запуск из подкаталога проекта работает ожидаемо. Каталог, у которого нет git-родителя, —
ошибка с названным путём, а не тихая работа «в никуда».

Второе назначение модуля — граница файловых операций: `ensure_inside` пускает только пути
внутри корня и разрешает символические ссылки до проверки, поэтому ссылка наружу отклоняется
так же, как прямой выход за корень.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import config


class RepoRootError(Exception):
    """Целевой каталог не может быть корнем репозитория."""


class PathOutsideRepoError(Exception):
    """Путь намеренно или случайно ведёт за пределы целевого репозитория."""


def find_git_root(start: Path) -> Optional[Path]:
    """Ближайший вверх каталог, в котором есть `.git` (каталог репозитория или файл worktree)."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_root(
    explicit: Optional[str] = None,
    env: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    """Корень по приоритету: аргумент → переменная окружения → текущий каталог."""
    raw = explicit or env or config.REPO_ROOT_ENV or None
    start = Path(raw).expanduser() if raw else Path(cwd or Path.cwd())
    if not start.exists():
        raise RepoRootError(f"путь {start} не существует")
    if not start.is_dir():
        raise RepoRootError(f"путь {start} не является каталогом")
    root = find_git_root(start)
    if root is None:
        raise RepoRootError(
            f"каталог {start} не является git-репозиторием, "
            "и ни один родительский каталог им тоже не является"
        )
    return root


def ensure_inside(path: str | Path, root: Path) -> Path:
    """Абсолютный путь внутри корня; любое отклонение — исключение, а не «как получится»."""
    root = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise PathOutsideRepoError(
            f"путь {path} ведёт за пределы целевого репозитория {root}"
        )
    return resolved


def relative_to_root(path: str | Path, root: Path) -> str:
    """Путь относительно корня — в таком виде он попадает в цитаты и отчёты."""
    return os.path.relpath(ensure_inside(path, root), Path(root).resolve())
