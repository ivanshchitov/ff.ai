"""Чистая логика инструментов собственного MCP-сервера над целевым репозиторием.

Здесь нет ни `mcp`, ни терминала: функции принимают `RepoContext` и возвращают текст для модели,
а отказ — исключением `ToolsError`. Поэтому инструменты проверяются без процессов
(`tests/unit/test_repo_tools.py`), а `repo_server.py` остаётся тонкой обёрткой над ними.

Границы держатся кодом, а не дисциплиной вызова. Путь разрешается и проверяется
`core.repo.ensure_inside` (символические ссылки — тоже), команда запускается только из белого
списка и без оболочки, git вызывается лишь подкомандами чтения, выгрузка пишется только
в каталог выгрузок и никогда не перезаписывает существующий файл. Записи в целевой репозиторий
у инструментов этой фазы нет вовсе: изменение кода появится вместе с конвейером задачи (P6).

Белые списки — данные: `Tools.load` читает `tools.json` пакета домена. Дублировать их в коде
нельзя: домен для того и вынесен в данные, чтобы правка области применимости не требовала правки
кода.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.repo import PathOutsideRepoError, ensure_inside, relative_to_root

# Пределы объёма и числа результатов: у каждого инструмента есть свой, иначе один вызов съел бы
# контекст модели целиком. Значения видны в описаниях инструментов, а не только в коде.
MAX_TREE_RESULTS = 200
MAX_SEARCH_RESULTS = 50
MAX_READ_LINES = 400
MAX_LINE_CHARS = 240
MAX_OUTPUT_CHARS = 8000
MAX_FILE_BYTES = 4 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 120

# Аргументы git, которыми можно обойти границу «только чтение внутри корня»: `--output` пишет
# файл (в том числе в целевой репозиторий), `--no-index` читает произвольные пути на диске.
BLOCKED_GIT_ARGUMENTS = ("--output", "--no-index")

# Имена переменных окружения, которые не наследует запущенная команда: сборка не должна видеть
# ключи и токены приложения. Правило по имени, а не по списку имён: список пришлось бы вести.
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")

_SAFE_FILE_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")


class ToolsError(Exception):
    """Отказ инструмента: его текст уходит модели как результат с пометкой об ошибке."""


@dataclass(frozen=True)
class Tools:
    """Белые списки пакета домена: что разрешено запускать и какие git-подкоманды читать."""

    run_allowed: Tuple[str, ...] = ()
    git_allowed: Tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "Tools":
        """Читает `tools.json`; битый или отсутствующий файл — отказ с его именем."""
        path = Path(path)
        if not path.is_file():
            raise ToolsError(f"файл белых списков не найден: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolsError(f"файл белых списков {path} не читается: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ToolsError(f"файл белых списков {path} не разбирается как JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ToolsError(
                f"файл белых списков {path} должен быть объектом с разделами «run» и «git»"
            )
        return cls(
            run_allowed=_allowed_list(data, "run", path),
            git_allowed=_allowed_list(data, "git", path),
        )


@dataclass(frozen=True)
class RepoContext:
    """Целевой репозиторий, каталог выгрузок и белые списки — всё, что нужно инструменту."""

    root: Path
    exports_dir: Path
    tools: Tools


def repo_tree(ctx: RepoContext, pattern: str = "**/*", max_results: int = MAX_TREE_RESULTS) -> str:
    """Перечень файлов репозитория с учётом правил игнорирования git.

    Пути относительны корня; `pattern` — шаблон `fnmatch` по этому относительному пути.
    """
    limit = _limit(max_results, MAX_TREE_RESULTS, "repo_tree")
    files = [name for name in _list_files(ctx) if _matches(name, pattern)]
    shown = files[:limit]
    header = f"Файлы: {len(shown)} из {len(files)} (pattern «{pattern}»)"
    body = "\n".join(f"  {name}" for name in shown) if shown else "  (нет файлов под шаблон)"
    note = ""
    if len(files) > len(shown):
        note = (
            f"\n…показаны первые {len(shown)} из {len(files)}; "
            "уточните pattern или увеличьте max_results"
        )
    return f"{header}\n{body}{note}"


def repo_read(
    ctx: RepoContext,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Содержимое файла с номерами строк; путь указывается относительно корня."""
    target = _inside(ctx, path)
    name = _relative(ctx, target)
    if target.is_dir():
        raise ToolsError(f"{name} — каталог, а не файл; перечень файлов даёт repo_tree")
    if not target.is_file():
        raise ToolsError(f"файл {name} не найден в целевом репозитории")
    lines = _read_text(target, name).splitlines()
    total = len(lines)
    if total == 0:
        return f"{name}: файл пуст (0 строк)"

    first = 1 if start_line is None else _line_number(start_line, "start_line")
    last = total if end_line is None else _line_number(end_line, "end_line")
    if last < first:
        raise ToolsError(f"end_line ({last}) меньше start_line ({first})")
    if first > total:
        raise ToolsError(f"start_line ({first}) больше числа строк в файле ({total})")
    last = min(last, total)
    shown_last = min(last, first + MAX_READ_LINES - 1)

    numbered = "".join(
        f"{number:>6}\t{lines[number - 1]}\n" for number in range(first, shown_last + 1)
    )
    if shown_last < last:
        numbered += (
            f"…показаны строки {first}–{shown_last} (предел {MAX_READ_LINES} строк); "
            "уточните start_line/end_line\n"
        )
    return f"{name}: строки {first}–{shown_last} из {total}\n{numbered}"


def repo_search(
    ctx: RepoContext,
    query: str,
    glob: Optional[str] = None,
    max_results: int = MAX_SEARCH_RESULTS,
) -> str:
    """Поиск подстроки без учёта регистра по текстовым файлам репозитория.

    Возвращает совпадения с файлом и номером строки; `glob` — шаблон `fnmatch` по пути файла.
    """
    limit = _limit(max_results, MAX_SEARCH_RESULTS, "repo_search")
    needle = query.lower()
    if not needle.strip():
        raise ToolsError("query пуст: назовите искомый текст")
    pattern = glob or "**/*"

    matches: List[str] = []
    truncated = False
    for name in _list_files(ctx):
        if not _matches(name, pattern):
            continue
        text = _read_text_quiet(ctx.root / name)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                if len(matches) == limit:
                    truncated = True
                    break
                matches.append(f"{name}:{number}: {_clip(line.strip(), MAX_LINE_CHARS)}")
        if truncated:
            break

    if not matches:
        return f"Поиск «{query}»: ничего не найдено (glob «{pattern}»)"
    header = f"Поиск «{query}»: найдено {len(matches)} (glob «{pattern}»)"
    note = ""
    if truncated:
        note = f"\n…показано не всё: предел max_results = {limit}; уточните query или glob"
    return f"{header}\n" + "\n".join(matches) + note


def repo_git(ctx: RepoContext, args: str) -> str:
    """Git только на чтение: подкоманда обязана быть в белом списке домена."""
    parts = _split(args, "аргументы git")
    subcommand = parts[0]
    if subcommand not in ctx.tools.git_allowed:
        allowed = ", ".join(ctx.tools.git_allowed) or (
            "нет — сервер запущен без файла белых списков (--tools)"
        )
        raise ToolsError(
            f"подкоманда git «{subcommand}» не разрешена: git вызывается только на чтение; "
            f"допустимые: {allowed}"
        )
    for part in parts[1:]:
        if part.startswith(BLOCKED_GIT_ARGUMENTS):
            raise ToolsError(
                f"аргумент «{part}» запрещён: им git выходит за границу «только чтение» "
                "внутри корня"
            )
    result = _run(ctx, ["git", "-C", str(ctx.root), *parts])
    return _command_report(f"git {args.strip()}", result)


def repo_run(ctx: RepoContext, command: str) -> str:
    """Разрешённая команда сборки: без оболочки, с таймаутом и обрезкой вывода."""
    text = command.strip()
    parts = _split(text, "команда")
    if _allowed_prefix(text, ctx.tools.run_allowed) is None:
        allowed = ", ".join(ctx.tools.run_allowed) or (
            "нет — сервер запущен без файла белых списков (--tools)"
        )
        raise ToolsError(
            f"команда «{text}» не входит в белый список команд; допустимые: {allowed}"
        )
    result = _run(ctx, parts)
    return _command_report(f"$ {text}", result)


def repo_save(ctx: RepoContext, name: str, text: str) -> str:
    """Выгрузка в каталог выгрузок: безопасное имя, существующий файл не перезаписывается."""
    if not text.strip():
        raise ToolsError("пустой текст не сохраняется: выгрузка без содержимого бесполезна")
    clean = name.strip()
    if not _SAFE_FILE_NAME.match(clean) or clean in (".", ".."):
        raise ToolsError(
            f"имя «{name}» недопустимо: разрешены буквы, цифры, «.», «_» и «-», без разделителей "
            "пути"
        )
    exports = Path(ctx.exports_dir)
    _ensure_exports_outside(ctx, exports)

    target = exports / clean
    counter = 2
    while target.exists():
        target = exports / f"{Path(clean).stem}-{counter}{Path(clean).suffix}"
        counter += 1
    try:
        exports.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ToolsError(f"выгрузку не удалось сохранить в {target}: {exc}") from exc
    return f"Выгрузка сохранена: {target}"


# --- Белые списки -----------------------------------------------------------


def _allowed_list(data: Mapping, section: str, path: Path) -> Tuple[str, ...]:
    block = data.get(section, {})
    if block is None:
        return ()
    if not isinstance(block, Mapping):
        raise ToolsError(f"{path}: раздел «{section}» должен быть объектом с полем «allowed»")
    values = block.get("allowed", [])
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ToolsError(f"{path}: поле «{section}.allowed» должно быть списком строк")
    allowed = []
    for item in values:
        if not isinstance(item, str):
            raise ToolsError(f"{path}: поле «{section}.allowed» должно быть списком строк")
        if item.strip():
            allowed.append(item.strip())
    return tuple(allowed)


def _allowed_prefix(command: str, allowed: Sequence[str]) -> Optional[str]:
    """Разрешённый префикс команды или None.

    Сравнение — по началу строки, но с границей слова: запись «make» не должна открывать
    «makeevil». Если сам префикс кончается не словесным символом (как «sfdk config target=»),
    он уже стоит на границе и продолжение допускается.
    """
    for prefix in allowed:
        if not command.startswith(prefix):
            continue
        rest = command[len(prefix):]
        if not rest:
            return prefix
        if prefix[-1].isalnum() or prefix[-1] == "_":
            if rest[0].isspace():
                return prefix
        else:
            return prefix
    return None


# --- Файлы ------------------------------------------------------------------


def _inside(ctx: RepoContext, path: str) -> Path:
    """Путь внутри корня; выход за корень и ссылка наружу — отказ с причиной."""
    try:
        return ensure_inside(path, ctx.root)
    except PathOutsideRepoError as exc:
        raise ToolsError(str(exc)) from exc


def _relative(ctx: RepoContext, path: Path) -> str:
    return relative_to_root(path, ctx.root)


def _list_files(ctx: RepoContext) -> List[str]:
    """Относительные пути файлов: правила игнорирования даёт git, иначе — обход дерева."""
    files = _git_files(ctx)
    if files is None:
        files = _walk_files(ctx.root)
    return sorted(files)


def _git_files(ctx: RepoContext) -> Optional[List[str]]:
    """`git ls-files -co --exclude-standard -z` или None, если git недоступен."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ctx.root), "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=_command_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [
        name.decode("utf-8", "surrogateescape")
        for name in result.stdout.split(b"\0")
        if name
    ]


def _walk_files(root: Path) -> List[str]:
    """Запасной обход дерева: правила .gitignore знает только git, здесь исключается .git."""
    found = []
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name != ".git")
        found.extend(
            (Path(current) / name).relative_to(root).as_posix() for name in sorted(names)
        )
    return found


def _matches(name: str, pattern: str) -> bool:
    """Шаблон по относительному пути: `**` — любая глубина, сегмент сверяется `fnmatch`.

    `fnmatch` сам по себе не знает `**`, а его `*` перешагивает через «/», поэтому шаблон
    разбирается по сегментам. Как в правилах gitignore, шаблон без «/» сверяется ещё и с именем
    файла на любой глубине: «*.cpp» находит исходники во всех каталогах.
    """
    pattern = pattern.strip()
    if not pattern or pattern == "**":
        return True
    if _match_segments(name.split("/"), pattern.split("/")):
        return True
    if "/" not in pattern:
        return fnmatch.fnmatchcase(name.rsplit("/", 1)[-1], pattern)
    return False


def _match_segments(parts: List[str], pattern: List[str]) -> bool:
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_match_segments(parts[index:], rest) for index in range(len(parts) + 1))
    if not parts:
        return False
    return fnmatch.fnmatchcase(parts[0], head) and _match_segments(parts[1:], rest)


def _read_text(path: Path, name: str) -> str:
    data = _read_bytes(path, name)
    return data.decode("utf-8", "replace")


def _read_text_quiet(path: Path) -> Optional[str]:
    """Текст файла или None: нечитаемые, двоичные и слишком большие файлы поиск пропускает."""
    try:
        return _read_text(path, str(path.name))
    except ToolsError:
        return None


def _read_bytes(path: Path, name: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ToolsError(f"файл {name} не читается: {exc}") from exc
    if size > MAX_FILE_BYTES:
        raise ToolsError(
            f"файл {name} больше предела {MAX_FILE_BYTES} байт и не отдаётся целиком"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ToolsError(f"файл {name} не читается: {exc}") from exc
    if b"\0" in data[:4096]:
        raise ToolsError(f"{name} — двоичный файл, текст из него не читается")
    return data


# --- Процессы ---------------------------------------------------------------


def _command_env() -> Dict[str, str]:
    """Окружение команды: без переменных с секретами и без постраничного вывода git."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in SECRET_ENV_MARKERS)
    }
    env["GIT_PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _split(text: str, what: str) -> List[str]:
    """Разбор строки команды без оболочки: список аргументов, а не подстановки shell."""
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        raise ToolsError(f"{what} не разбираются: {exc}") from exc
    if not parts:
        raise ToolsError(f"{what} пусты")
    return parts


def _run(ctx: RepoContext, argv: Sequence[str]) -> "subprocess.CompletedProcess[bytes]":
    try:
        return subprocess.run(
            list(argv),
            shell=False,
            cwd=str(ctx.root),
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=_command_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolsError(
            f"«{argv[0]}» не завершилась за {COMMAND_TIMEOUT_SECONDS:g} с и была прервана"
        ) from exc
    except OSError as exc:
        raise ToolsError(f"«{argv[0]}» не удалось запустить: {exc}") from exc


def _command_report(title: str, result: "subprocess.CompletedProcess[bytes]") -> str:
    """Код возврата и вывод: ненулевой код — результат, а не отказ инструмента."""
    out = result.stdout.decode("utf-8", "replace").rstrip("\n")
    err = result.stderr.decode("utf-8", "replace").rstrip("\n")
    body = out
    if err.strip():
        body = f"{body}\nstderr:\n{err}" if body.strip() else f"stderr:\n{err}"
    # Эхо самой команды тоже ограничено: длинный аргумент иначе удвоил бы ответ.
    lines = [_clip(title, MAX_LINE_CHARS), f"код возврата: {result.returncode}"]
    if not body.strip():
        lines.append("вывод пуст")
        return "\n".join(lines)
    note = ""
    if len(body) > MAX_OUTPUT_CHARS:
        note = f"\n…вывод обрезан: показаны первые {MAX_OUTPUT_CHARS} символов из {len(body)}"
        body = body[:MAX_OUTPUT_CHARS]
    lines.append(body)
    return "\n".join(lines) + note


# --- Общее ------------------------------------------------------------------


def _limit(requested: int, ceiling: int, tool: str) -> int:
    if not isinstance(requested, int) or isinstance(requested, bool) or not 1 <= requested <= ceiling:
        raise ToolsError(f"{tool}: max_results должен быть целым от 1 до {ceiling}")
    return requested


def _line_number(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ToolsError(f"{name} — номер строки, начиная с 1")
    return value


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _ensure_exports_outside(ctx: RepoContext, exports: Path) -> None:
    resolved = exports.resolve()
    root = Path(ctx.root).resolve()
    if resolved == root or root in resolved.parents:
        raise ToolsError(
            f"каталог выгрузок {resolved} находится внутри целевого репозитория {root}: "
            "выгрузки пишутся только за его пределами"
        )
