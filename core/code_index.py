"""Корпус кода целевого репозитория: отбор файлов, разбиение на фрагменты и индекс в кэше.

Модуль не знает предметной области: расширения исходников, исключения, пределы и признаки начала
блока приходят из правил домена (`domains/<id>/corpus.json`). Фрагмент — дословный кусок файла с
путём и диапазоном строк: именно на этот диапазон ссылается цитата `path:L10-L40`.

Индекс — производное репозитория: он лежит в кэше состояния, собирается только по команде и
никогда не пишется в целевой репозиторий. Сборка идёт во временный файл рядом и заменяет прежний
индекс одним `os.replace` после успешной записи.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .domains import DomainCorpus

STRATEGY_FIXED = "fixed"
STRATEGY_STRUCTURAL = "structural"
STRATEGIES = (STRATEGY_FIXED, STRATEGY_STRUCTURAL)
SCHEMA_VERSION = "1"

# Причины пропуска файла — для отчёта: пользователь должен видеть, почему корпус меньше каталога.
SKIP_EXTENSION = "расширение вне корпуса"
SKIP_GLOBS = "исключён правилами"
SKIP_BINARY = "двоичный файл"
SKIP_TOO_LARGE = "крупнее предела"
SKIP_EMPTY = "пустой файл"

_BINARY_SNIFF_BYTES = 8192


class CodeIndexError(Exception):
    """Корпус кода собрать не удалось: причина называется текстом."""


@dataclass(frozen=True)
class CodeChunk:
    """Фрагмент корпуса: дословный текст файла, его путь и диапазон строк."""

    strategy: str
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def label(self) -> str:
        """Ссылка на фрагмент в том виде, в каком она нужна цитате."""
        return f"{self.path}:L{self.start_line}-L{self.end_line}"


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True)
class IndexReport:
    """Снимок сборки: что вошло в индекс, что пропущено и где индекс лежит."""

    strategy: str
    database: Path
    root: Path
    files: int
    chunks: int
    lines: int
    size_bytes: int
    source: str
    built_at: float
    skipped: Tuple[SkippedFile, ...] = ()

    @property
    def built_at_text(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.built_at))

    def skip_reasons(self) -> Dict[str, int]:
        """Сколько файлов пропущено по каждой причине — сводкой, а не списком."""
        counts: Dict[str, int] = {}
        for item in self.skipped:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts


@dataclass(frozen=True)
class ScanResult:
    """Что нашлось в репозитории: файлы корпуса, пропуски и способ получения списка."""

    files: Tuple[Path, ...]
    skipped: Tuple[SkippedFile, ...]
    source: str


# --- отбор файлов ---------------------------------------------------------------------------


def _git_files(root: Path) -> Optional[Tuple[Path, ...]]:
    """Список файлов репозитория от git: он сам учитывает `.gitignore` и не спускается в сборку.

    Возвращает `None`, если git недоступен или каталог не репозиторий: тогда работает обычный
    обход, а отчёт называет способ, которым получен список.
    """
    if not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return tuple(root / name for name in names)


def _walk_files(root: Path) -> Tuple[Path, ...]:
    """Обычный обход: все файлы, кроме каталога git."""
    found = []
    for path in sorted(Path(root).rglob("*")):
        if ".git" in path.parts:
            continue
        if path.is_file():
            found.append(path)
    return tuple(found)


def _glob_matches(relative: str, pattern: str) -> bool:
    """Совпадение пути с правилом исключения.

    Правило с косой чертой — шаблон всего пути, без неё — имя в любом месте пути: так «build»
    исключает каталог сборки на любом уровне, а «build/*» — его содержимое.
    """
    parts = relative.split("/")
    if "/" in pattern:
        # Правило с косой чертой — шаблон пути: проверяется и весь путь, и каждый его хвост,
        # начинающийся с границы каталога («vendor/*» исключает содержимое любого vendor).
        return any(
            fnmatch.fnmatch("/".join(parts[index:]), pattern) for index in range(len(parts))
        )
    return any(fnmatch.fnmatch(part, pattern) for part in parts)


def _excluded(relative: str, corpus: DomainCorpus) -> bool:
    return any(_glob_matches(relative, pattern) for pattern in corpus.exclude_globs)


def _is_binary(path: Path) -> bool:
    """Двоичный файл узнаётся по нулевому байту в начале: расширения для этого недостаточно."""
    try:
        with path.open("rb") as handle:
            head = handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\0" in head


def scan(root: Path, corpus: DomainCorpus) -> ScanResult:
    """Отбирает файлы корпуса и объясняет пропуски.

    Порядок проверок — от дешёвой к дорогой: расширение, правила исключения, размер, содержимое.
    Способ получения списка (git или обход) попадает в отчёт: расхождение «файл есть, а в корпусе
    его нет» должно объясняться, а не выглядеть ошибкой индекса.
    """
    root = Path(root)
    found = _git_files(root)
    source = "git ls-files"
    if found is None:
        found = _walk_files(root)
        source = "обход каталога"

    files: List[Path] = []
    skipped: List[SkippedFile] = []
    for path in found:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if path.suffix.lower() not in corpus.include_extensions:
            skipped.append(SkippedFile(relative, SKIP_EXTENSION))
            continue
        if _excluded(relative, corpus):
            skipped.append(SkippedFile(relative, SKIP_GLOBS))
            continue
        try:
            size = path.stat().st_size
        except OSError:
            skipped.append(SkippedFile(relative, SKIP_TOO_LARGE))
            continue
        if size == 0:
            skipped.append(SkippedFile(relative, SKIP_EMPTY))
            continue
        if size > corpus.max_file_bytes:
            skipped.append(SkippedFile(relative, SKIP_TOO_LARGE))
            continue
        if _is_binary(path):
            skipped.append(SkippedFile(relative, SKIP_BINARY))
            continue
        files.append(path)
    return ScanResult(files=tuple(files), skipped=tuple(skipped), source=source)


def read_lines(path: Path) -> List[str]:
    """Строки файла: перевод строки не хранится, чтобы диапазон и текст не расходились."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise CodeIndexError(f"{path}: не читается ({error})") from error
    return text.splitlines()


# --- разбиение ------------------------------------------------------------------------------


def _chunk(strategy: str, path: str, start: int, end: int, lines: Sequence[str]) -> CodeChunk:
    text = "\n".join(lines[start - 1 : end])
    return CodeChunk(
        strategy=strategy,
        chunk_id=f"{path}:{start}-{end}",
        path=path,
        start_line=start,
        end_line=end,
        text=text,
    )


def chunk_fixed(
    lines: Sequence[str], path: str, chunk_lines: int, overlap_lines: int
) -> List[CodeChunk]:
    """Окна строк с перекрытием: простое разбиение, не зависящее от языка файла."""
    if chunk_lines < 1:
        raise CodeIndexError("размер окна должен быть положительным")
    if not 0 <= overlap_lines < chunk_lines:
        raise CodeIndexError("перекрытие должно быть меньше окна и неотрицательным")
    total = len(lines)
    step = chunk_lines - overlap_lines
    chunks: List[CodeChunk] = []
    start = 1
    while start <= total:
        end = min(start + chunk_lines - 1, total)
        chunks.append(_chunk(STRATEGY_FIXED, path, start, end, lines))
        if end == total:
            break
        start += step
    return chunks


def _block_starts(lines: Sequence[str], patterns: Sequence[str]) -> List[int]:
    """Номера строк, с которых начинаются блоки: первая строка файла и совпадения с признаками."""
    compiled = [re.compile(pattern) for pattern in patterns]
    starts = [1]
    for number, line in enumerate(lines, start=1):
        if number == 1:
            continue
        if any(pattern.search(line) for pattern in compiled):
            starts.append(number)
    return starts


def _merge_and_split(
    lines: Sequence[str],
    path: str,
    starts: Sequence[int],
    max_lines: int,
    merge_below: int,
) -> List[CodeChunk]:
    """Склейка мелких блоков и нарезка крупных: один блок не должен быть ни строкой, ни файлом."""
    total = len(lines)
    bounds: List[Tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else total
        if bounds and end - start + 1 < merge_below:
            bounds[-1] = (bounds[-1][0], end)
            continue
        bounds.append((start, end))
    chunks: List[CodeChunk] = []
    for start, end in bounds:
        while end - start + 1 > max_lines:
            chunks.append(_chunk(STRATEGY_STRUCTURAL, path, start, start + max_lines - 1, lines))
            start += max_lines
        chunks.append(_chunk(STRATEGY_STRUCTURAL, path, start, end, lines))
    return chunks


def chunk_structural(
    lines: Sequence[str], path: str, corpus: DomainCorpus
) -> List[CodeChunk]:
    """Разбиение по структуре файла: границы блоков берутся из правил домена.

    Для файлов, у которых признаков структуры нет, разбиение вырождается в окна: это честнее, чем
    делать вид, что структура найдена.
    """
    patterns = corpus.patterns_for(Path(path).suffix.lower())
    if not patterns:
        return chunk_fixed(lines, path, corpus.fixed_chunk_lines, corpus.fixed_overlap_lines)
    return _merge_and_split(
        lines,
        path,
        _block_starts(lines, patterns),
        corpus.structural_max_lines,
        corpus.structural_merge_below,
    )


def chunk_file(path: Path, relative: str, corpus: DomainCorpus, strategy: str) -> List[CodeChunk]:
    """Фрагменты одного файла выбранной стратегией."""
    lines = read_lines(path)
    if not lines:
        return []
    if strategy == STRATEGY_STRUCTURAL:
        return chunk_structural(lines, relative, corpus)
    if strategy == STRATEGY_FIXED:
        return chunk_fixed(lines, relative, corpus.fixed_chunk_lines, corpus.fixed_overlap_lines)
    raise CodeIndexError(f"неизвестная стратегия разбиения: {strategy}")


# --- хранилище ------------------------------------------------------------------------------


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE chunks ("
        "strategy TEXT NOT NULL, chunk_id TEXT NOT NULL, path TEXT NOT NULL, "
        "start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, content TEXT NOT NULL, "
        "sha1 TEXT NOT NULL, PRIMARY KEY(strategy, chunk_id))"
    )
    connection.execute("CREATE INDEX chunks_by_path ON chunks(strategy, path)")
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_index(
    root: Path,
    corpus: DomainCorpus,
    strategy: str,
    database: Path,
) -> IndexReport:
    """Собирает индекс и заменяет прежний только при успехе.

    Прежний индекс остаётся нетронутым до последнего шага: сборка во временном файле и один
    `os.replace`. Ошибка обхода, разбора или записи оставляет пользователя с рабочим индексом,
    а не с полупустым — прежний ещё пригоден.
    """
    root = Path(root)
    database = Path(database)
    if strategy not in STRATEGIES:
        raise CodeIndexError(f"неизвестная стратегия разбиения: {strategy}")
    scan_result = scan(root, corpus)
    chunks: List[CodeChunk] = []
    skipped = list(scan_result.skipped)
    for path in scan_result.files:
        relative = path.relative_to(root).as_posix()
        try:
            chunks.extend(chunk_file(path, relative, corpus, strategy))
        except CodeIndexError as error:
            skipped.append(SkippedFile(relative, str(error)))

    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_name(database.name + ".tmp")
    built_at = time.time()
    lines_total = sum(chunk.end_line - chunk.start_line + 1 for chunk in chunks)
    size_total = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
    try:
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
            connection.executemany(
                "INSERT INTO chunks (strategy, chunk_id, path, start_line, end_line, content, sha1) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        strategy,
                        chunk.chunk_id,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        _digest(chunk.text),
                    )
                    for chunk in chunks
                ],
            )
            metadata = {
                "schema": SCHEMA_VERSION,
                "strategy": strategy,
                "root": str(root),
                "built_at": repr(built_at),
                "files": str(len(scan_result.files)),
                "chunks": str(len(chunks)),
                "lines": str(lines_total),
                "size_bytes": str(size_total),
                "source": scan_result.source,
                "skipped": json.dumps(
                    [{"path": item.path, "reason": item.reason} for item in skipped],
                    ensure_ascii=False,
                ),
            }
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)", sorted(metadata.items())
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, database)
    except (OSError, sqlite3.Error) as error:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise CodeIndexError(f"индекс не собран: {error}") from error

    return IndexReport(
        strategy=strategy,
        database=database,
        root=root,
        files=len(scan_result.files),
        chunks=len(chunks),
        lines=lines_total,
        size_bytes=size_total,
        source=scan_result.source,
        built_at=built_at,
        skipped=tuple(skipped),
    )


def index_exists(database: Path) -> bool:
    """Есть ли индекс в базе: пустая или чужая база индексом не считается."""
    database = Path(database)
    if not database.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if row is None:
            return False
        return connection.execute("SELECT 1 FROM metadata LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def read_index(database: Path) -> Optional[IndexReport]:
    """Снимок прежней сборки — из метаданных индекса, без обращения к репозиторию."""
    database = Path(database)
    if not index_exists(database):
        return None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        rows = dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if rows.get("schema") != SCHEMA_VERSION:
        return None
    try:
        skipped = tuple(
            SkippedFile(str(item.get("path", "")), str(item.get("reason", "")))
            for item in json.loads(rows.get("skipped", "[]"))
        )
        built_at = float(rows.get("built_at", "0"))
    except (TypeError, ValueError):
        skipped, built_at = (), 0.0
    return IndexReport(
        strategy=rows.get("strategy", ""),
        database=database,
        root=Path(rows.get("root", "")),
        files=int(rows.get("files", "0")),
        chunks=int(rows.get("chunks", "0")),
        lines=int(rows.get("lines", "0")),
        size_bytes=int(rows.get("size_bytes", "0")),
        source=rows.get("source", ""),
        built_at=built_at,
        skipped=skipped,
    )


def read_chunks(database: Path) -> Tuple[CodeChunk, ...]:
    """Фрагменты из индекса — в порядке путей и строк: так поиск получает стабильный вход.

    Чтение только на чтение и без обращения к репозиторию: индекс самодостаточен, а тексты в нём
    уже дословные. Битый или чужой файл — пустой результат, а не исключение: вызывающий решает,
    что показать пользователю.
    """
    database = Path(database)
    if not index_exists(database):
        return ()
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error:
        return ()
    try:
        rows = connection.execute(
            "SELECT strategy, chunk_id, path, start_line, end_line, content FROM chunks "
            "ORDER BY path, start_line"
        ).fetchall()
    except sqlite3.Error:
        return ()
    finally:
        connection.close()
    return tuple(
        CodeChunk(
            strategy=strategy,
            chunk_id=chunk_id,
            path=path,
            start_line=int(start_line),
            end_line=int(end_line),
            text=content,
        )
        for strategy, chunk_id, path, start_line, end_line, content in rows
    )


def sample_labels(chunks: Sequence[CodeChunk], limit: int = 3) -> Tuple[str, ...]:
    """Примеры границ фрагментов — по одному на файл, чтобы пример не повторял один и тот же файл."""
    labels: List[str] = []
    seen = set()
    for chunk in chunks:
        if chunk.path in seen:
            continue
        seen.add(chunk.path)
        labels.append(chunk.label)
        if len(labels) >= limit:
            break
    return tuple(labels)


def collect(root: Path, corpus: DomainCorpus, strategy: str) -> Tuple[CodeChunk, ...]:
    """Все фрагменты репозитория без записи индекса: нужны сравнению стратегий."""
    root = Path(root)
    scan_result = scan(root, corpus)
    chunks: List[CodeChunk] = []
    for path in scan_result.files:
        relative = path.relative_to(root).as_posix()
        try:
            chunks.extend(chunk_file(path, relative, corpus, strategy))
        except CodeIndexError:
            continue
    return tuple(chunks)
