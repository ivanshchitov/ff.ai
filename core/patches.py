"""Патчи как артефакт задачи: разбор, проверка границ и применение.

Патч — единственная форма изменения чужого проекта, которую можно проверить до записи: он называет
файлы и строки, применяется инструментом git и откатывается им же. Своего применения diff-ов здесь
нет намеренно: `git apply --check` знает про контекст и переводы строк больше, чем собственный
построчный разбор, а ошибка в нём оставила бы файл в половинчатом состоянии.
"""

from __future__ import annotations

import re
import ast
import shlex
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
    old_path: str = ""
    new_path: str = ""

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
    path = raw.split("\t", 1)[0]
    if not path.startswith('"') and path != path.strip():
        raise PatchError("неоднозначные пробелы в пути: используйте quoted-путь")
    if path.startswith('"'):
        try:
            path = ast.literal_eval(path)
            if re.search(r"\\[0-7]{3}", raw):
                path = path.encode("latin1").decode("utf-8")
        except (ValueError, SyntaxError, UnicodeError) as error:
            raise PatchError("неверный quoted-путь") from error
    if not isinstance(path, str) or any(ord(c) < 32 for c in path):
        raise PatchError("недопустимый путь")
    if path in ("/dev/null", ""):
        return ""
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _patch_lines(text):
    """Distinguish actual headers from hunk content using Git's line counts."""
    old_left = new_left = 0
    for line in text.splitlines():
        if old_left or new_left:
            if line.startswith('\\ No newline at end of file'):
                yield line, True
                continue
            if not line or line[0] not in ('+', '-', ' '):
                raise PatchError('неверный контекст патча')
            old_left -= int(line[0] in ('-', ' '))
            new_left -= int(line[0] in ('+', ' '))
            if old_left < 0 or new_left < 0:
                raise PatchError('неверное число строк патча')
            yield line, True
            continue
        match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if match:
            old_left = int(match[2]) if match[2] is not None else 1
            new_left = int(match[4]) if match[4] is not None else 1
        yield line, False
    if old_left or new_left:
        raise PatchError('патч оборван')


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
    old_path = new_path = ""

    def flush() -> None:
        nonlocal current_path, added, removed, is_new, old_path, new_path
        if current_path:
            files.append(
                ParsedFile(
                    path=current_path,
                    added=tuple(added),
                    removed=tuple(removed),
                    is_new=is_new,
                    old_path=old_path,
                    new_path=new_path,
                )
            )
        current_path, added, removed, is_new = None, [], [], False
        old_path = new_path = ""

    for line, content in _patch_lines(text):
        if content:
            if current_path is None:
                raise PatchError('патч не называет файл')
            if line.startswith('+'):
                added.append(line[1:])
            elif line.startswith('-'):
                removed.append(line[1:])
            continue
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


def validate_text_patch(text: str, root: Path) -> ParsedPatch:
    """Validate every header before git interprets it; permit text edits only."""
    if re.search(r"^(GIT binary patch|Binary files |rename |copy |similarity index)", text, re.M):
        raise PatchError("binary, rename, copy и symlink-патчи не поддерживаются")
    for mode in re.findall(r"^(?:new file|deleted file|old|new) mode (\S+)$", text, re.M):
        if not re.fullmatch(r"[0-7]+", mode) or int(mode, 8) & 0o170000 != 0o100000:
            raise PatchError("поддерживаются только обычные файлы")
    parsed = parse(text)
    headers = []
    for line, content in _patch_lines(text):
        if content:
            continue
        if line.startswith("diff --git "):
            raw = line[len("diff --git "):]
            if raw.startswith('"'):
                # Retain C-style escaping for the common path decoder.
                names = re.findall(r'"(?:\\.|[^"\\])*"|\S+', raw)
            else:
                match = re.fullmatch(r"(a/.*) (b/.*)", raw)
                names = list(match.groups()) if match else shlex.split(raw)
            if len(names) != 2:
                raise PatchError("неверный diff --git")
            headers.extend(_clean_path(name) for name in names)
        elif line.startswith(("--- ", "+++ ")):
            headers.append(_clean_path(line[4:]))
    for item in parsed.files:
        if item.old_path and item.new_path and item.old_path != item.new_path:
            raise PatchError("переименование не поддерживается")
        if not item.added and not item.removed:
            raise PatchError("нужен текстовый патч")
    for name in headers:
        if name:
            safe_path(root, name)
            if name not in parsed.paths:
                raise PatchError("заголовки патча называют разные файлы")
    return parsed


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal, Git internals and all symlinks, including internal links."""
    relative = Path(name)
    if not name or relative.is_absolute() or any(p in ("..", ".git") for p in relative.parts):
        raise PatchError("путь вне проекта или служебный путь: " + name)
    current = Path(root).resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PatchError("символьная ссылка: " + name)
    try:
        ensure_inside(current, Path(root).resolve())
    except PathOutsideRepoError as error:
        raise PatchError(str(error)) from error
    if current.exists() and not current.is_file():
        raise PatchError("нужен обычный файл: " + name)
    return current


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
        completed = _git(root, ["apply", "--check", "--unidiff-zero", "--whitespace=nowarn", "-"], patch_text)
    except (OSError, subprocess.SubprocessError) as error:
        return f"проверка применения не выполнена: {error}"
    if completed.returncode != 0:
        return _reason(completed)
    return ""


def apply(root: Path, patch_text: str) -> str:
    """Применение патча: пустая строка означает успех, иначе — причина отказа."""
    try:
        completed = _git(root, ["apply", "--unidiff-zero", "--whitespace=nowarn", "-"], patch_text)
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
