"""Планировщик собственного MCP-сервера репозитория: расписание и задания здоровья репозитория.

Модуль отделён от `repo_server.py` по той же причине, по которой логика инструментов отделена от
протокола: здесь нет ни `mcp`, ни асинхронности, исполнитель вызова инжектируется — значит вся
логика расписания (что просрочено, как сдвигается срок, как строится агрегат) проверяется обычным
юнит-тестом без запуска процесса и без сети.

Задание вызывает инструмент **этого же** сервера, внутри процесса: кросс-серверный вызов потребовал
бы второго MCP-клиента внутри сервера. Инструменты самого планировщика в задание не ставятся —
задание, зовущее `schedule_run_due`, зациклило бы исполнитель.

Отказ — это текст результата, а не исключение: его читает модель, которая и выбирала аргументы.
Исключение — задания: их неудача должна быть видна в журнале прогонов отказом, поэтому они говорят
`ToolsError` (его сервер переводит в пометку об ошибке), а планировщик записывает прогон как неудачу
и продолжает остальные задания.

Правила проверок приходят из пакета домена (`validation.json` и `invariants.json` рядом с файлом
белых списков): ни одного имени модуля, команды или версии платформы в этом модуле нет.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core import code_index, config, domains
from core.schedule_store import (
    MAX_EVERY_MINUTES,
    MAX_START_DELAY_MINUTES,
    MIN_EVERY_MINUTES,
    MIN_START_DELAY_MINUTES,
    ScheduleStore,
)
from mcp_server.repo_tools import RepoContext, ToolsError, repo_run

# Итог вызова, записываемый в журнал, обрезается: журнал уезжает в файл и в отчёт, а полный
# текст результата инструмента там не нужен — он в любом случае получен заново следующим вызовом.
SUMMARY_MAX_CHARS = 200

# Имена инструментов планировщика: по ним сервер разбирает вызов, а планировщик не даёт поставить
# их же в задание.
SCHEDULE_ADD = "schedule_add"
SCHEDULE_LIST = "schedule_list"
SCHEDULE_RUN_DUE = "schedule_run_due"
SCHEDULE_SUMMARY = "schedule_summary"
SCHEDULE_TOOL_NAMES = (SCHEDULE_ADD, SCHEDULE_LIST, SCHEDULE_RUN_DUE, SCHEDULE_SUMMARY)

# Задания здоровья репозитория: собственные инструменты сервера, которые и ставятся в расписание.
TARGET_BUILD = "target_build"
INDEX_REFRESH = "index_refresh"
PUBLIC_API_SCAN = "public_api_scan"
JOB_TOOL_NAMES = (TARGET_BUILD, INDEX_REFRESH, PUBLIC_API_SCAN)

ARGUMENT_ERROR_PREFIX = "Неверные аргументы"

# Пределы сканирования: один прогон не должен превращаться ни в мегабайтный отчёт, ни в минуты
# ожидания. Найденное накапливается, поэтому «показано не всё» не значит «потеряно».
MAX_SCAN_RESULTS = 50
MAX_SCAN_LINE_CHARS = 160
# Ключ накопленного: найденные нарушения — одно множество на все прогоны, поэтому нарушение,
# встреченное вчера, не считается новым сегодня.
SCAN_KEY = "нарушения"

# Формы объявления модуля в исходниках: включение заголовка и значение qmake. Ни имён модулей,
# ни их списка здесь нет — их приносит домен и аргумент задания.
_INCLUDE_MODULE = re.compile(r"^\s*#\s*include\s*[<\"]([A-Za-z0-9_]+)/")
_QMAKE_MODULE = re.compile(r"^\s*QT\s*[+*]?=\s*(.+)$")


class Scheduler:
    """Инструменты планировщика поверх хранилища и исполнителя вызова."""

    def __init__(
        self,
        store: ScheduleStore,
        call_tool: Callable[[str, Dict[str, Any]], Tuple[bool, str]],
        tool_names: Sequence[str],
        now: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self._call_tool = call_tool
        # Ставить можно только инструменты, не принадлежащие самому планировщику.
        self._tool_names = tuple(name for name in tool_names if name not in SCHEDULE_TOOL_NAMES)
        self._now = now

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        return self.call_result(name, arguments)[1]

    def call_result(self, name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        """То же, но с признаком успеха: по нему сервер ставит протокольный признак ошибки."""
        # Файл общий для приложения и фонового исполнителя: состояние перечитывается перед каждым
        # вызовом, иначе запись одного процесса затёрла бы работу другого.
        self.store.reload()
        handlers = {
            SCHEDULE_ADD: self._add,
            SCHEDULE_LIST: self._list,
            SCHEDULE_RUN_DUE: self._run_due,
            SCHEDULE_SUMMARY: self._report,
        }
        handler = handlers.get(name)
        if handler is None:
            return False, (
                f"Инструмент «{name}» не объявлен. Доступны: {', '.join(SCHEDULE_TOOL_NAMES)}."
            )
        text = handler(arguments or {})
        # Все отказы планировщика начинаются одной фразой: разбирать каждую ветку отдельно
        # значило бы переписать инструменты ради флага, которого прежде не требовалось.
        return not text.startswith(ARGUMENT_ERROR_PREFIX), text

    # --- инструменты ---

    def _add(self, arguments: Dict[str, Any]) -> str:
        tool = str(arguments.get("tool") or "").strip()
        if not tool:
            return f"{ARGUMENT_ERROR_PREFIX}: параметр tool обязателен и не может быть пустым."
        if tool in SCHEDULE_TOOL_NAMES:
            return (
                f"{ARGUMENT_ERROR_PREFIX}: инструмент планировщика «{tool}» нельзя поставить "
                f"в задание. Доступны: {', '.join(self._tool_names)}."
            )
        if tool not in self._tool_names:
            return (
                f"{ARGUMENT_ERROR_PREFIX}: инструмент «{tool}» этот сервер не объявляет. "
                f"Доступны: {', '.join(self._tool_names)}."
            )

        every = self._whole(
            arguments, "every_minutes", MIN_EVERY_MINUTES, MAX_EVERY_MINUTES
        )
        if isinstance(every, str):
            return every
        delay = self._whole(
            arguments,
            "start_in_minutes",
            MIN_START_DELAY_MINUTES,
            MAX_START_DELAY_MINUTES,
            default=MIN_START_DELAY_MINUTES,
        )
        if isinstance(delay, str):
            return delay

        job_arguments = self._job_arguments(arguments.get("arguments"))
        if isinstance(job_arguments, str):
            return job_arguments
        job = self.store.add_job(
            tool=tool,
            arguments=job_arguments,
            every_minutes=every,
            next_run=self._now() + delay * 60,
        )
        when = "сразу" if delay == 0 else f"через {delay} мин"
        return (
            f"Задание {job.number} поставлено: инструмент {job.tool}, каждые {job.every_minutes} мин, "
            f"первый запуск {when}."
        )

    def _list(self, arguments: Dict[str, Any]) -> str:
        jobs = self.store.jobs()
        if not jobs:
            return "Планировщик: заданий нет."
        lines = ["Задания планировщика:"]
        for job in jobs:
            lines.append(
                f"- {job.number}: {job.tool} {self._arguments_text(job.arguments)}, "
                f"каждые {job.every_minutes} мин, следующий запуск {self._when(job.next_run)}, "
                f"прогонов {job.runs}"
            )
        return "\n".join(lines)

    def _run_due(self, arguments: Dict[str, Any]) -> str:
        now = self._now()
        due = self.store.due_jobs(now)
        if not due:
            return "Планировщик: просроченных заданий нет, выполнять нечего."
        lines = []
        for job in due:
            before = self.store.collected_total()
            ok, text = self._call_tool(job.tool, job.arguments)
            summary = self._summary(text)
            after = self.store.collected_total()
            self.store.record_run(
                job,
                at=now,
                ok=ok,
                summary=summary,
                collected=after,
                fresh=max(0, after - before),
            )
            mark = "выполнено" if ok else "отказ"
            lines.append(f"- {job.number} ({job.tool}): {mark} — {summary}")
        return "\n".join([f"Выполнено заданий: {len(due)}."] + lines)

    def _report(self, arguments: Dict[str, Any]) -> str:
        jobs = self.store.jobs()
        if not jobs:
            return "Планировщик: заданий нет, прогонов нет."
        runs = self.store.runs()
        lines = [
            f"Планировщик: заданий {len(jobs)}, прогонов {sum(job.runs for job in jobs)}, "
            f"накоплено записей {self.store.collected_total()}."
        ]
        for job in jobs:
            last = next((run for run in reversed(runs) if run.number == job.number), None)
            lines.append(
                f"- {job.number}: {job.tool} {self._arguments_text(job.arguments)}, "
                f"каждые {job.every_minutes} мин, следующий запуск {self._when(job.next_run)}, "
                f"прогонов {job.runs}"
            )
            if last is not None:
                mark = "выполнено" if last.ok else "отказ"
                lines.append(f"  последний прогон: {mark} — {last.summary}")
        return "\n".join(lines)

    # --- разбор аргументов и форматирование ---

    def _whole(
        self,
        arguments: Dict[str, Any],
        key: str,
        minimum: int,
        maximum: int,
        default: Any = None,
    ):
        """Целое в диапазоне — либо текст отказа, который уйдёт результатом вызова."""
        raw = arguments.get(key, default)
        if raw is None:
            return f"{ARGUMENT_ERROR_PREFIX}: параметр {key} обязателен."
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return f"{ARGUMENT_ERROR_PREFIX}: {key} должен быть целым числом, получено «{raw}»."
        if not minimum <= value <= maximum:
            return (
                f"{ARGUMENT_ERROR_PREFIX}: {key} должен быть от {minimum} до {maximum}, "
                f"получено {value}."
            )
        return value

    def _job_arguments(self, raw: Any):
        """Аргументы задания: объект — как есть, строка — JSON (иначе текст отказа).

        Строкой они приходят двумя путями: из ручного `/tool call`, который разбирает ввод
        парами «ключ=значение», и от моделей, охотно присылающих вложенный объект строкой.
        """
        if isinstance(raw, dict):
            return dict(raw)
        if raw in (None, ""):
            return {}
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                return (
                    f"{ARGUMENT_ERROR_PREFIX}: arguments должен быть объектом или строкой JSON, "
                    f"получено «{raw}»."
                )
            if isinstance(parsed, dict):
                return parsed
        return f"{ARGUMENT_ERROR_PREFIX}: arguments должен быть объектом, получено «{raw}»."

    def _summary(self, text: str) -> str:
        first = (text or "").strip().splitlines()
        summary = first[0] if first else ""
        return summary[:SUMMARY_MAX_CHARS]

    def _arguments_text(self, arguments: Dict[str, Any]) -> str:
        if not arguments:
            return "без аргументов"
        return ", ".join(f"{key}={value}" for key, value in arguments.items())

    def _when(self, moment: float) -> str:
        return time.strftime("%H:%M:%S", time.localtime(moment))


# --- Задания здоровья репозитория ------------------------------------------


def run_job(
    name: str,
    arguments: Dict[str, Any],
    ctx: RepoContext,
    domain: Optional[domains.Domain],
    store: ScheduleStore,
) -> str:
    """Выполняет задание по имени: один вход и для ручного вызова, и для прогона расписания.

    Один вход — не удобство: ручной вызов и задание обязаны делать одно и то же, иначе отчёт
    по заданию говорил бы не о том, что произойдёт с тем же заданием в расписании.
    """
    handlers = {
        TARGET_BUILD: target_build,
        INDEX_REFRESH: index_refresh,
        PUBLIC_API_SCAN: public_api_scan,
    }
    handler = handlers.get(name)
    if handler is None:
        raise ToolsError(
            f"задание «{name}» не объявлено. Доступны: {', '.join(JOB_TOOL_NAMES)}."
        )
    return handler(ctx, domain, store, dict(arguments or {}))


def target_build(
    ctx: RepoContext, domain: Optional[domains.Domain], store: ScheduleStore, arguments: Dict[str, Any]
) -> str:
    """Сборка цели командами домена — теми же, что проверяет этап проверки задачи.

    Своего раннера команд задание не заводит: команды идут через `repo_run`, где белый список
    домена один на всё приложение. Шаги берутся из правил домена, а аргумент задания их заменяет —
    так в расписание попадает конкретная цель (`sfdk config target=…`), не переписывая правила.
    """
    _only(arguments, ("steps",))
    steps = _steps(arguments.get("steps"), domain)
    lines: List[str] = []
    for step in steps:
        # Отказ белого списка или незапускаемая команда — исключение: сборка не состоялась.
        report = repo_run(ctx, command=step)
        code = _return_code(report)
        if code != 0:
            shown = "код возврата неизвестен" if code is None else f"код возврата {code}"
            raise ToolsError(
                f"Сборка цели не завершена: команда «{step}» — {shown}.\n"
                + "\n".join(lines)
                + f"\n{report}"
            )
        lines.append(f"- {step}: ок")
    return "\n".join([f"Сборка цели: шагов {len(steps)}."] + lines)


def index_refresh(
    ctx: RepoContext, domain: Optional[domains.Domain], store: ScheduleStore, arguments: Dict[str, Any]
) -> str:
    """Пересборка индекса кода: без модели и без сети, правила корпуса — данные домена.

    Индекс заменяется атомарно (`code_index.build_index`), поэтому неудачный прогон оставляет
    прежний рабочий индекс, а не полупустой.
    """
    _only(arguments, ("strategy",))
    corpus = _corpus(domain)
    strategy = _text(arguments.get("strategy")).strip() or corpus.strategy
    if strategy not in code_index.STRATEGIES:
        raise ToolsError(
            f"{ARGUMENT_ERROR_PREFIX}: стратегия «{strategy}» неизвестна, "
            f"допустимы: {', '.join(code_index.STRATEGIES)}."
        )
    database = config.repo_index_file(ctx.root)
    try:
        report = code_index.build_index(ctx.root, corpus, strategy, database)
    except code_index.CodeIndexError as error:
        raise ToolsError(f"Индекс не обновлён: {error}") from error
    return (
        f"Индекс обновлён: фрагментов {report.chunks}, файлов {report.files}, "
        f"строк {report.lines}, стратегия {report.strategy}, пропущено файлов "
        f"{len(report.skipped)}; база {report.database}"
    )


def public_api_scan(
    ctx: RepoContext, domain: Optional[domains.Domain], store: ScheduleStore, arguments: Dict[str, Any]
) -> str:
    """Поиск запрещённых конструкций домена и модулей вне публичного API — без модели.

    Два разных нарушения в одном проходе, потому что источники у них разные: конструкции — правила
    домена (`validation.json` → `forbidden`, там и идиомы Qt 6; плюс стемы инвариантов с их
    номерами), модули — аргумент задания со списком публичных модулей. Список модулей именно
    аргумент, а не данные домена: таблица публичных API живёт в документации портала, и пока её
    нет файлом, приложение передаёт список в задании. Без списка проверка модулей не выполняется —
    и это сказано в отчёте: молча пропущенная проверка выглядит как пройденная.

    Найденное накапливается в хранилище, поэтому отчёт называет и новые нарушения: вчерашнее
    нарушение — не новость сегодня.
    """
    _only(arguments, ("modules", "max_results"))
    corpus = _corpus(domain)
    rules = _construct_rules(domain)
    if not rules:
        raise ToolsError(
            "правила домена не объявили запрещённых конструкций: проверять нечего "
            "(нужен validation.json пакета домена)"
        )
    modules = _allowed_modules(arguments.get("modules"))
    limit = _limit(arguments.get("max_results"))

    scan_result = code_index.scan(ctx.root, corpus)
    violations: List[str] = []
    unreadable: List[str] = []
    truncated = False
    for path in scan_result.files:
        relative = path.relative_to(ctx.root).as_posix()
        text = _read_text(path)
        if text is None:
            unreadable.append(relative)
            continue
        for number, line in enumerate(text.splitlines(), 1):
            # Одно сработавшее правило на строку: иначе строка с двумя запретами заняла бы две
            # записи в отчёте и в накопленном, а значит и дважды считалась бы новой.
            for reason in _rule_reasons(line, rules):
                violations.append(f"{relative}:{number}: {reason} — {_clip(line.strip())}")
                break
            if modules:
                for module in _modules_of(line):
                    if not _is_public(module, modules):
                        violations.append(
                            f"{relative}:{number}: модуль «{module}» вне публичного API — "
                            f"{_clip(line.strip())}"
                        )
            if len(violations) >= limit:
                truncated = True
                break
        if truncated:
            break

    fresh = store.remember_collected(SCAN_KEY, violations)
    header = (
        f"Проверка публичного API: файлов корпуса {len(scan_result.files)}, "
        f"нарушений {len(violations)} (новых {len(fresh)})."
    )
    lines = [header]
    lines.extend(f"- {item}" for item in violations)
    if not violations:
        lines.append("Нарушений не найдено." if modules else "Запрещённых конструкций не найдено.")
    if truncated:
        lines.append(f"…показаны первые {limit} нарушений: предел max_results")
    if not modules:
        lines.append(
            "Проверка модулей не выполнялась: список публичных модулей не задан "
            "(параметр modules)."
        )
    if unreadable:
        lines.append(f"Не прочитаны файлы: {', '.join(unreadable[:10])}")
    return "\n".join(lines)


# --- Данные домена ---------------------------------------------------------


def load_package_domain(tools_path: Optional[Path]) -> Optional[domains.Domain]:
    """Пакет домена рядом с файлом белых списков; None — если его там нет.

    Так задания получают правила области применимости без нового аргумента запуска: белые списки
    уже лежат в пакете домена (`--tools domains/<id>/tools.json`), и `--root` про домен не говорит
    ничего. Битый пакет — тоже None: инструменты репозитория работают и без правил, а задания
    называют причину отказа сами.
    """
    if tools_path is None:
        return None
    package = Path(tools_path).parent
    if not (package / "domain.json").is_file():
        return None
    try:
        return domains.load_domain(package.name, domains_dir=package.parent)
    except (domains.DomainError, OSError, ValueError, KeyError):
        return None


def _corpus(domain: Optional[domains.Domain]) -> domains.DomainCorpus:
    corpus = getattr(domain, "corpus", None)
    if corpus is None:
        raise ToolsError(
            "домен не объявил корпус кода: проверять нечего (нужен corpus.json пакета домена)"
        )
    return corpus


def _steps(raw: Any, domain: Optional[domains.Domain]) -> Tuple[str, ...]:
    """Шаги сборки: аргумент задания или правила домена; пустой список — названный отказ."""
    text = _text(raw)
    if text.strip():
        steps = tuple(part.strip() for part in re.split(r"[;\n]+", text) if part.strip())
    else:
        checks = getattr(domain, "checks", None)
        steps = tuple(getattr(checks, "build_steps", ()) or ())
    if not steps:
        raise ToolsError(
            f"{ARGUMENT_ERROR_PREFIX}: шаги сборки не заданы, и домен их не объявил "
            "(нужен validation.json пакета домена или параметр steps)"
        )
    return steps


def _construct_rules(domain: Optional[domains.Domain]) -> Tuple[Tuple[re.Pattern, str], ...]:
    """Запрещённые конструкции: правила домена плюс стемы инвариантов с их номерами.

    Оба источника — данные пакета домена. Стемы инвариантов ищутся подстрокой без учёта регистра,
    ровно как их проверяет приложение в готовом ответе, поэтому идиома Qt 6, названная инвариантом,
    попадает в скан вместе с конструкцией из правил.
    """
    rules: List[Tuple[re.Pattern, str]] = []
    checks = getattr(domain, "checks", None)
    for rule in getattr(checks, "forbidden", ()) or ():
        # Образцы проверены при загрузке пакета домена (`_load_checks` компилирует каждый),
        # поэтому здесь нет второй проверки: непрошедший образец — это ошибка, а не пропуск.
        rules.append((re.compile(rule.pattern), rule.reason))
    for invariant in getattr(domain, "invariants", ()) or ():
        for stem in invariant.forbidden:
            if not stem.strip():
                continue
            rules.append(
                (
                    re.compile(re.escape(stem), re.IGNORECASE),
                    f"инвариант {invariant.number} («{invariant.rule}»)",
                )
            )
    return tuple(rules)


def _rule_reasons(line: str, rules: Sequence[Tuple[re.Pattern, str]]) -> List[str]:
    return [reason for pattern, reason in rules if pattern.search(line)]


def _allowed_modules(raw: Any) -> Tuple[str, ...]:
    text = _text(raw)
    return tuple(part for part in re.split(r"[,\s;]+", text.strip()) if part)


def _module_key(name: str) -> str:
    """Имя модуля без учёта регистра и префикса `Qt`: `QT += core` и `QtCore` — одно и то же."""
    lowered = name.strip().lower()
    if lowered.startswith("qt") and len(lowered) > 2:
        lowered = lowered[2:]
    return lowered


def _is_public(module: str, allowed: Sequence[str]) -> bool:
    return _module_key(module) in {_module_key(item) for item in allowed}


def _modules_of(line: str) -> Tuple[str, ...]:
    """Модули, названные в строке: включение заголовка или значение qmake."""
    found: List[str] = []
    match = _INCLUDE_MODULE.match(line)
    if match:
        found.append(match.group(1))
    match = _QMAKE_MODULE.match(line)
    if match:
        value = match.group(1).split("#", 1)[0]
        found.extend(part for part in re.split(r"[\s,]+", value.strip()) if part)
    return tuple(found)


# --- Общее -----------------------------------------------------------------


def _only(arguments: Dict[str, Any], allowed: Sequence[str]) -> None:
    """Лишний параметр — отказ: молча проигнорированный аргумент выглядит как учтённый."""
    unknown = [key for key in arguments if key not in allowed]
    if unknown:
        raise ToolsError(
            f"{ARGUMENT_ERROR_PREFIX}: неизвестные параметры: {', '.join(sorted(unknown))}; "
            f"допустимы: {', '.join(allowed) or 'нет'}"
        )


def _text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return " ".join(str(item) for item in raw)
    return str(raw)


def _limit(raw: Any) -> int:
    if raw in (None, ""):
        return MAX_SCAN_RESULTS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ToolsError(
            f"{ARGUMENT_ERROR_PREFIX}: max_results должен быть целым от 1 до {MAX_SCAN_RESULTS}, "
            f"получено «{raw}»"
        ) from None
    if not 1 <= value <= MAX_SCAN_RESULTS:
        raise ToolsError(
            f"{ARGUMENT_ERROR_PREFIX}: max_results должен быть целым от 1 до {MAX_SCAN_RESULTS}, "
            f"получено {value}"
        )
    return value


def _return_code(report: str) -> Optional[int]:
    """Код возврата команды из отчёта `repo_run`: им и решается, состоялась ли сборка.

    Своего запуска процесса задание не заводит — белый список и таймаут одни на всё приложение,
    — поэтому и код возврата берётся из того же отчёта. Отчёт строит сам сервер, формат его
    известен: строка «код возврата: N».
    """
    match = re.search(r"код возврата:\s*(-?\d+)", report or "")
    return int(match.group(1)) if match else None


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _clip(text: str) -> str:
    return text if len(text) <= MAX_SCAN_LINE_CHARS else text[:MAX_SCAN_LINE_CHARS] + "…"
