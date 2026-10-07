"""Патчи как артефакт задачи: разбор, проверка границ и применение.

Патч — единственная форма изменения чужого проекта, которую можно проверить до записи: он называет
файлы и строки, применяется инструментом git и откатывается им же. Своего применения diff-ов здесь
нет намеренно: `git apply --check` знает про контекст и переводы строк больше, чем собственный
построчный разбор, а ошибка в нём оставила бы файл в половинчатом состоянии.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .repo import PathOutsideRepoError, ensure_inside

GIT_TIMEOUT = 60
_HUNK_RE = re.compile(r"^@@ ")
_OLD_PATH_RE = re.compile(r"^--- (?P<path>.+)$")
_NEW_PATH_RE = re.compile(r"^\+\+\+ (?P<path>.+)$")


class PatchError(Exception):
    """Патч не разобран: причина называется текстом."""


@dataclass(frozen=True)
class ParsedFile:
    """Файл внутри патча: путь и добавленные строки (по ним идут проверки содержимого)."""

    path: str
    added: Tuple[str, ...] = ()
    removed: Tuple[str, ...] = ()
    is_new: bool = False

    @property
    def added_lines(self) -> int:
        return len(self.added)

    @property
    def removed_lines(self) -> int:
        return len(self.removed)


@dataclass(frozen=True)
class ParsedPatch:
    """Разобранный патч: файлы, объёмы и признак того, что он вообще что-то меняет."""

    files: Tuple[ParsedFile, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.files

    @property
    def paths(self) -> Tuple[str, ...]:
        return tuple(item.path for item in self.files)

    def added_lines(self) -> Tuple[str, ...]:
        return tuple(line for item in self.files for line in item.added)

    def summary(self) -> str:
        """Короткая подпись патча для журнала и отчёта: файлы и объёмы."""
        if self.empty:
            return "патч пуст"
        parts = [
            f"{item.path} (+{item.added_lines}/-{item.removed_lines})" for item in self.files
        ]
        return "; ".join(parts)


def _clean_path(raw: str) -> str:
    """Путь из заголовка патча без префиксов git и без служебного «/dev/null»."""
    path = raw.strip().split("\t", 1)[0].strip()
    if path in ("/dev/null", ""):
        return ""
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def parse(text: str) -> ParsedPatch:
    """Разобрать унифицированный патч: файлы, добавленные и удалённые строки.

    Разбор намеренно простой и предсказуемый: он нужен не для применения (этим занимается git), а
    для проверок содержимого — какие именно строки патч *вносит* в проект.
    """
    if not (text or "").strip():
        raise PatchError("патч пуст")
    files: List[ParsedFile] = []
    current_path: Optional[str] = None
    is_new = False
    added: List[str] = []
    removed: List[str] = []
    in_hunk = False

    def flush() -> None:
        nonlocal current_path, added, removed, is_new
        if current_path:
            files.append(
                ParsedFile(
                    path=current_path,
                    added=tuple(added),
                    removed=tuple(removed),
                    is_new=is_new,
                )
            )
        current_path, added, removed, is_new = None, [], [], False

    for line in text.splitlines():
        old = _OLD_PATH_RE.match(line)
        new = _NEW_PATH_RE.match(line)
        if old:
            flush()
            old_path = _clean_path(old.group("path"))
            current_path = old_path
            is_new = old_path == ""
            in_hunk = False
            continue
        if new and current_path is not None or (new and is_new):
            new_path = _clean_path(new.group("path"))
            if new_path:
                current_path = new_path
            in_hunk = False
            continue
        if _HUNK_RE.match(line):
            in_hunk = True
            continue
        if not in_hunk or current_path is None:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    flush()
    if not files:
        raise PatchError("в патче нет ни одного файла")
    return ParsedPatch(files=tuple(files))


def outside_root(patch: ParsedPatch, root: Path) -> Tuple[str, ...]:
    """Пути патча, ведущие за пределы целевого репозитория: до применения их надо поймать.

    Проверка идёт общим правилом границ (`core.repo.ensure_inside`): абсолютный путь, выход через
    «..» и символьная ссылка одинаково недопустимы.
    """
    root = Path(root)
    escaped: List[str] = []
    for item in patch.files:
        try:
            ensure_inside(root / item.path, root)
        except PathOutsideRepoError:
            escaped.append(item.path)
    return tuple(escaped)


def _git(root: Path, args: Sequence[str], patch_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=patch_text.encode("utf-8"),
        capture_output=True,
        timeout=GIT_TIMEOUT,
        check=False,
    )


def _reason(completed: subprocess.CompletedProcess) -> str:
    text = (completed.stderr or completed.stdout or b"").decode("utf-8", errors="replace").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else f"git apply завершился с кодом {completed.returncode}"


def check(root: Path, patch_text: str) -> str:
    """Проверка чистоты применения: пустая строка означает «патч ложится на файлы»."""
    try:
        completed = _git(root, ["apply", "--check", "--whitespace=nowarn", "-"], patch_text)
    except (OSError, subprocess.SubprocessError) as error:
        return f"проверка применения не выполнена: {error}"
    if completed.returncode != 0:
        return _reason(completed)
    return ""


def apply(root: Path, patch_text: str) -> str:
    """Применение патча: пустая строка означает успех, иначе — причина отказа."""
    try:
        completed = _git(root, ["apply", "--whitespace=nowarn", "-"], patch_text)
    except (OSError, subprocess.SubprocessError) as error:
        return f"применение не выполнено: {error}"
    if completed.returncode != 0:
        return _reason(completed)
    return ""


def changed_paths(root: Path) -> Tuple[str, ...]:
    """Изменённые файлы рабочего дерева: нужны, чтобы показать, что именно изменил прогон."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    names: List[str] = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        name = line[3:].strip()
        if name:
            names.append(name)
    return tuple(names)
