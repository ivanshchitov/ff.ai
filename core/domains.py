"""Домен — область применимости как данные.

Пакет `domains/<id>/` описывает роль и границы тематики (промпты), таблицу инвариантов и
маркеры автоопределения. Ядро не знает ни одного идентификатора домена и не содержит
домена по умолчанию: какой пакет активен, решает этот модуль — по аргументу, переменной
окружения, маркерам целевого репозитория или по пометке `default` в самом пакете.

Ошибка в пакете — отказ старта с названным файлом и полем. Работать с наполовину загруженным
доменом хуже, чем не работать: пользователь не поймёт, почему часть правил не действует.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import config, memory_layers
from .mcp_registry import MCPServerSpec

SCHEMA_VERSION = 1
_MARKER_TEXT_LIMIT = 200_000


class DomainError(Exception):
    """Домен не может быть выбран или загружен."""


class DomainSchemaError(DomainError):
    """Пакет домена не соответствует схеме."""


class UnknownDomainError(DomainError):
    """Запрошенного пакета домена нет."""


@dataclass(frozen=True)
class Marker:
    """Признак репозитория: файл по маске и, необязательно, подстрока внутри него."""

    glob: str
    contains: Optional[str] = None

    def matches(self, root: Path) -> Optional[str]:
        for candidate in sorted(Path(root).glob(self.glob)):
            if not candidate.is_file():
                continue
            if self.contains is None:
                return str(candidate.relative_to(root))
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")[
                    :_MARKER_TEXT_LIMIT
                ]
            except OSError:
                continue
            if self.contains.lower() in text.lower():
                return str(candidate.relative_to(root))
        return None


@dataclass(frozen=True)
class Invariant:
    """Правило домена. Пустой `forbidden` означает правило, которое проверяет только промпт."""

    number: int
    rule: str
    forbidden: Tuple[str, ...]
    source: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "number": self.number,
            "rule": self.rule,
            "forbidden": list(self.forbidden),
            "source": self.source,
        }


@dataclass(frozen=True)
class MemoryRule:
    """Правило маршрутизации памяти: образец реплики, слой, категория и ключ записи.

    Образец — регулярное выражение, ищется по реплике без учёта регистра (поэтому кириллица
    распознаётся в любом регистре). Значением записи становится предложение реплики, в котором
    образец сработал. Порядок правил значим: правило ниже заменяет значение того же ключа.
    """

    layer: str
    category: str
    key: str
    pattern: str
    description: str


@dataclass(frozen=True)
class DomainMemory:
    """Правила слоёв памяти домена: что и по какой реплике запоминается."""

    rules: Tuple[MemoryRule, ...]
    # Ключ цели рабочей памяти: «цель» — слово домена, в ядре его быть не должно.
    goal_key: str
    goal_category: str


@dataclass(frozen=True)
class ProfileSection:
    """Раздел профиля домена: машинное имя, подпись, вопрос опросника и значение по умолчанию.

    Значение по умолчанию подставляется, когда вопрос пропущен, а раздела у профиля ещё нет:
    так домен задаёт, каким он хочет видеть ответ, если пользователь ничего о себе не сказал.
    """

    id: str
    label: str
    question: str
    default: str = ""


@dataclass(frozen=True)
class DomainProfile:
    """Разделы профиля домена и вопрос об имени профиля.

    Имя — не раздел профиля: оно различает профили в файле, поэтому текст вопроса и заготовка
    свободного имени объявлены отдельно от разделов.
    """

    name_label: str
    name_question: str
    name_prefix: str
    sections: Tuple[ProfileSection, ...]


@dataclass(frozen=True)
class DocsTools:
    """Имена инструментов сервера документации: они тоже данные, а не код ядра."""

    versions: str
    search: str
    document: str


@dataclass(frozen=True)
class ForbiddenRule:
    """Запрещённая конструкция домена: образец для поиска и причина, почему её нельзя."""

    pattern: str
    reason: str

    def matches(self, line: str) -> bool:
        return re.search(self.pattern, line) is not None


@dataclass(frozen=True)
class DomainOps:
    """Правила конвейера операций: инструмент, цели, команды шагов и запреты.

    Команды, пути и признаки подписи принадлежат предметной области: у другой платформы другой
    SDK. Ядро не знает ни одного имени команды — оно подставляет значения в шаблоны пакета.
    """

    tool_env: str
    tool_default: str
    target_pattern: str
    architectures: Tuple[Tuple[str, str], ...]
    default_architecture: str
    state_dir: str
    steps: Tuple[Tuple[str, Tuple[str, ...]], ...]
    package_globs: Tuple[str, ...]
    # Образцы имени приложения: оформление пакета принадлежит домену, в ядре таких образцов нет.
    app_id_globs: Tuple[str, ...]
    signature_key_env: str
    signature_signed: Tuple[str, ...]
    signature_unsigned: Tuple[str, ...]
    forbidden_commands: Tuple[str, ...]
    package_check: Tuple[str, ...]

    def step(self, name: str) -> Tuple[str, ...]:
        """Шаблон шага по имени; неизвестный шаг — пустой шаблон, а не догадка."""
        for step_name, template in self.steps:
            if step_name == name:
                return template
        return ()

    def architecture_note(self, arch: str) -> str:
        for name, note in self.architectures:
            if name == arch:
                return note
        return ""


@dataclass(frozen=True)
class DomainChecks:
    """Правила детерминированной проверки артефакта задачи.

    Запрещённые конструкции, обязательные элементы описания пакета и команды сборки принадлежат
    предметной области: ядро не знает ни одного запрещённого слова.
    """

    forbidden: Tuple[ForbiddenRule, ...]
    spec_required: Tuple[str, ...]
    spec_globs: Tuple[str, ...]
    build_tool: str = ""
    build_steps: Tuple[str, ...] = ()
    build_description: str = ""


@dataclass(frozen=True)
class DomainCorpus:
    """Корпус кода домена: что считать исходниками и как резать их на фрагменты.

    Расширения, исключения, пределы и признаки начала блока принадлежат предметной области:
    у другой платформы другой набор файлов и другая структура. Ядро читает правила отсюда и
    не знает ни одного расширения.
    """

    strategy: str
    include_extensions: Tuple[str, ...]
    exclude_globs: Tuple[str, ...]
    max_file_bytes: int
    fixed_chunk_lines: int
    fixed_overlap_lines: int
    structural_max_lines: int
    structural_merge_below: int
    block_patterns: Tuple[Tuple[str, Tuple[str, ...]], ...]

    def patterns_for(self, suffix: str) -> Tuple[str, ...]:
        """Признаки начала блока для расширения файла: расширения без семьи получают общий набор."""
        key = _FAMILY_BY_SUFFIX.get(suffix, "")
        for family, patterns in self.block_patterns:
            if family == key:
                return patterns
        return ()


# Семьи файлов: расширение → набор признаков структуры. Таблица нужна потому, что правил
# больше, чем семейств (.h/.hpp/.cc — один и тот же C++), а сами признаки остаются данными.
_FAMILY_BY_SUFFIX = {
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".c": "cpp",
    ".qml": "qml",
    ".js": "qml",
    ".spec": "spec",
    ".pro": "make",
    ".pri": "make",
    ".prf": "make",
    ".cmake": "make",
    ".sh": "shell",
}


@dataclass(frozen=True)
class DomainDocs:
    """Корпус документации домена: какой сервер читать и как по нему искать.

    Разделы, слова-признаки версий и имена инструментов объявляет пакет: ядро не знает ни
    названия разделов портала, ни имён инструментов его сервера.
    """

    server: str
    source: str
    default_section: str
    version_sections: Tuple[str, ...]
    version_keywords: Tuple[str, ...]
    tools: DocsTools


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    platform: str
    docs_version: str
    local_sdk_version: str
    responses_language: str
    is_default: bool
    markers: Tuple[Marker, ...]
    invariants: Tuple[Invariant, ...]
    path: Path
    # Серверы, объявленные пакетом: записи реестра MCP, относящиеся к области применимости.
    servers: Tuple[MCPServerSpec, ...] = ()
    # Файл белых списков команд и git-подкоманд — данные для собственного сервера репозитория.
    tools_path: Optional[Path] = None
    # Корпус документации домена: сервер, разделы, слова-признаки версий и имена инструментов.
    docs: Optional[DomainDocs] = None
    # Корпус кода домена: расширения исходников, исключения и правила разбиения на фрагменты.
    corpus: Optional[DomainCorpus] = None
    # Правила детерминированной проверки артефакта задачи (P6).
    checks: Optional[DomainChecks] = None
    # Правила конвейера операций над проектом (P7).
    ops: Optional[DomainOps] = None
    # Правила маршрутизации слоёв памяти (P8).
    memory: Optional[DomainMemory] = None
    # Разделы профиля пользователя и вопрос об имени (P8).
    profile: Optional[DomainProfile] = None

    @lru_cache(maxsize=None)
    def prompt(self, name: str) -> str:
        """Текст промпта из пакета: `system`, `refusal` и другие по мере надобности."""
        path = self.path / "prompts" / f"{name}.md"
        if not path.is_file():
            raise DomainSchemaError(f"{path}: нет файла промпта «{name}»")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise DomainSchemaError(f"{path}: файл промпта «{name}» пуст")
        return text


@dataclass(frozen=True)
class DomainSelection:
    """Что выбрано, почему и на каком основании — для отчёта и для строки журнала."""

    domain: Domain
    source: str
    evidence: Tuple[str, ...] = ()


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise DomainSchemaError(f"{path}: файл не найден")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DomainSchemaError(f"{path}: некорректный JSON — {error}") from error
    if not isinstance(data, dict):
        raise DomainSchemaError(f"{path}: ожидался объект JSON")
    return data


def _require(data: Dict[str, object], key: str, path: Path, kind: type) -> object:
    if key not in data:
        raise DomainSchemaError(f"{path}: нет обязательного поля «{key}»")
    value = data[key]
    if not isinstance(value, kind):
        raise DomainSchemaError(
            f"{path}: поле «{key}» должно быть типа {kind.__name__}, а не {type(value).__name__}"
        )
    return value


def _load_markers(data: Dict[str, object], path: Path) -> Tuple[Marker, ...]:
    raw = _require(data, "markers", path, list)
    markers: List[Marker] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("glob"), str) or not item["glob"]:
            raise DomainSchemaError(
                f"{path}: поле «markers[{index}]» должно содержать непустую строку «glob»"
            )
        contains = item.get("contains")
        if contains is not None and not isinstance(contains, str):
            raise DomainSchemaError(f"{path}: поле «markers[{index}].contains» должно быть строкой")
        markers.append(Marker(glob=item["glob"], contains=contains))
    if not markers:
        raise DomainSchemaError(f"{path}: список «markers» пуст")
    return tuple(markers)


def _load_invariants(path: Path) -> Tuple[Invariant, ...]:
    data = _read_json(path)
    raw = _require(data, "invariants", path, list)
    invariants: List[Invariant] = []
    for index, item in enumerate(raw):
        where = f"{path}: инвариант #{index + 1}"
        if not isinstance(item, dict):
            raise DomainSchemaError(f"{where} должен быть объектом")
        if not isinstance(item.get("number"), int):
            raise DomainSchemaError(f"{where}: поле «number» должно быть целым числом")
        rule = item.get("rule")
        if not isinstance(rule, str) or not rule.strip():
            raise DomainSchemaError(f"{where}: поле «rule» должно быть непустой строкой")
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            raise DomainSchemaError(f"{where}: поле «source» должно быть непустой строкой")
        forbidden = item.get("forbidden", [])
        if not isinstance(forbidden, list) or any(
            not isinstance(word, str) or not word.strip() for word in forbidden
        ):
            raise DomainSchemaError(f"{where}: поле «forbidden» должно быть списком непустых строк")
        invariants.append(
            Invariant(
                number=item["number"],
                rule=rule.strip(),
                forbidden=tuple(word.strip() for word in forbidden),
                source=source.strip(),
            )
        )
    if not invariants:
        raise DomainSchemaError(f"{path}: список «invariants» пуст")
    return tuple(invariants)


def _load_memory(path: Path) -> Optional[DomainMemory]:
    """Необязательный раздел пакета: правила маршрутизации слоёв памяти.

    Отсутствие файла означает «домен не объявляет правил памяти» — тогда записи делают только
    явные команды пользователя. Сломанный файл обязан остановить загрузку: правило с опечаткой
    молча не срабатывало бы, и запись пропадала бы без следа.
    """
    if not path.is_file():
        return None
    data = _read_json(path)
    raw = _require(data, "rules", path, list)
    if not raw:
        raise DomainSchemaError(f"{path}: список «rules» пуст")
    rules: List[MemoryRule] = []
    for index, item in enumerate(raw):
        where = f"{path}: правило #{index + 1}"
        if not isinstance(item, dict):
            raise DomainSchemaError(f"{where} должно быть объектом")
        values: Dict[str, str] = {}
        for key in ("layer", "category", "key", "pattern", "description"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise DomainSchemaError(f"{where}: поле «{key}» должно быть непустой строкой")
            values[key] = value.strip()
        if values["layer"] not in memory_layers.STORABLE_LAYERS:
            allowed = ", ".join(memory_layers.STORABLE_LAYERS)
            raise DomainSchemaError(
                f"{where}: слой {values['layer']!r} не поддерживается (доступны: {allowed})"
            )
        try:
            re.compile(values["pattern"])
        except re.error as error:
            raise DomainSchemaError(
                f"{where}: образец {values['pattern']!r} не компилируется ({error})"
            ) from error
        rules.append(MemoryRule(**values))
    goal_key = str(data.get("goal_key", "")).strip()
    goal_category = str(data.get("goal_category", "")).strip()
    if not goal_key or not goal_category:
        raise DomainSchemaError(
            f"{path}: нужны «goal_key» и «goal_category» — ключ цели рабочей памяти"
        )
    return DomainMemory(rules=tuple(rules), goal_key=goal_key, goal_category=goal_category)


_SECTION_ID_RE = re.compile(r"[a-z][a-z0-9_]*")


def _load_profile(path: Path) -> Optional[DomainProfile]:
    """Необязательный раздел пакета: разделы профиля пользователя и вопрос об имени.

    Отсутствие файла означает «домен не объявляет профиля» — тогда персонализации нет. Сломанный
    файл обязан остановить загрузку: раздел без подписи или с нечитаемым именем дал бы профиль,
    который нельзя ни показать, ни сохранить.
    """
    if not path.is_file():
        return None
    data = _read_json(path)

    def _text(key: str) -> str:
        value = str(_require(data, key, path, str)).strip()
        if not value:
            raise DomainSchemaError(f"{path}: поле «{key}» не должно быть пустым")
        return value

    raw = _require(data, "sections", path, list)
    if not raw:
        raise DomainSchemaError(f"{path}: список «sections» пуст")
    sections: List[ProfileSection] = []
    seen: List[str] = []
    for index, item in enumerate(raw):
        where = f"{path}: раздел #{index + 1}"
        if not isinstance(item, dict):
            raise DomainSchemaError(f"{where} должен быть объектом")
        section_id = item.get("id")
        if not isinstance(section_id, str) or not _SECTION_ID_RE.fullmatch(section_id):
            raise DomainSchemaError(
                f"{where}: поле «id» должно быть латинским именем (строчные буквы, цифры, «_»)"
            )
        if section_id in seen:
            raise DomainSchemaError(f"{where}: раздел «{section_id}» объявлен дважды")
        seen.append(section_id)
        label = item.get("label")
        question = item.get("question")
        if not isinstance(label, str) or not label.strip():
            raise DomainSchemaError(f"{where}: поле «label» должно быть непустой строкой")
        if not isinstance(question, str) or not question.strip():
            raise DomainSchemaError(f"{where}: поле «question» должно быть непустой строкой")
        default = item.get("default", "")
        if not isinstance(default, str):
            raise DomainSchemaError(f"{where}: поле «default» должно быть строкой")
        sections.append(
            ProfileSection(
                id=section_id,
                label=label.strip(),
                question=question.strip(),
                default=default.strip(),
            )
        )
    return DomainProfile(
        name_label=_text("name_label"),
        name_question=_text("name_question"),
        name_prefix=_text("name_prefix"),
        sections=tuple(sections),
    )


def _load_servers(path: Path) -> Tuple[MCPServerSpec, ...]:
    """Необязательный раздел пакета: серверы области применимости.

    Отсутствие файла — это «домен без внешних серверов», а не ошибка: пакет может быть чисто
    локальным. Сломанный файл, наоборот, обязан остановить загрузку — иначе ассистент молча
    работал бы без источника знаний, который домен считает обязательным.
    """
    if not path.is_file():
        return ()
    data = _read_json(path)
    raw = _require(data, "servers", path, list)
    servers = []
    for index, item in enumerate(raw):
        try:
            servers.append(MCPServerSpec.from_data(item, source="домен"))
        except Exception as error:  # MCPSpecError — но ловим и неожиданное: причина нужна пользователю
            raise DomainSchemaError(f"{path}: сервер #{index + 1} — {error}") from error
    return tuple(servers)


def _load_docs(path: Path) -> Optional[DomainDocs]:
    """Необязательный раздел пакета: корпус документации домена.

    Отсутствие файла означает «у домена нет внешнего корпуса документации» — это законный случай,
    а сломанный файл обязан остановить загрузку: иначе поиск молча пойдёт не туда.
    """
    if not path.is_file():
        return None
    data = _read_json(path)
    tools_raw = _require(data, "tools", path, dict)
    tools = DocsTools(
        versions=str(_require(tools_raw, "versions", path, str)),
        search=str(_require(tools_raw, "search", path, str)),
        document=str(_require(tools_raw, "document", path, str)),
    )
    raw_sections = _require(data, "version_sections", path, list)
    if any(not isinstance(item, str) or not item.strip() for item in raw_sections):
        raise DomainSchemaError(f"{path}: поле «version_sections» должно быть списком непустых строк")
    raw_keywords = data.get("version_keywords", [])
    if not isinstance(raw_keywords, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_keywords
    ):
        raise DomainSchemaError(f"{path}: поле «version_keywords» должно быть списком непустых строк")
    return DomainDocs(
        server=str(_require(data, "server", path, str)),
        source=str(_require(data, "source", path, str)),
        default_section=str(_require(data, "default_section", path, str)),
        version_sections=tuple(item.strip() for item in raw_sections),
        version_keywords=tuple(item.strip() for item in raw_keywords),
        tools=tools,
    )


def _load_corpus(path: Path) -> Optional[DomainCorpus]:
    """Необязательный раздел пакета: корпус кода домена.

    Отсутствие файла означает «у домена нет правил отбора исходников» — тогда корпус кода просто
    не строится. Сломанный файл обязан остановить загрузку: иначе индекс молча соберётся не по тем
    файлам, и цитаты будут указывать не туда.
    """
    if not path.is_file():
        return None
    data = _read_json(path)

    def _strings(key: str) -> Tuple[str, ...]:
        raw = _require(data, key, path, list)
        if any(not isinstance(item, str) or not item.strip() for item in raw):
            raise DomainSchemaError(f"{path}: поле «{key}» должно быть списком непустых строк")
        return tuple(item.strip() for item in raw)

    fixed = _require(data, "fixed", path, dict)
    structural = _require(data, "structural", path, dict)
    raw_patterns = data.get("block_patterns", {})
    if not isinstance(raw_patterns, dict):
        raise DomainSchemaError(f"{path}: поле «block_patterns» должно быть объектом")
    patterns = []
    for family, raw in raw_patterns.items():
        if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw):
            raise DomainSchemaError(
                f"{path}: признаки блока семьи «{family}» должны быть списком непустых строк"
            )
        patterns.append((str(family), tuple(raw)))

    corpus = DomainCorpus(
        strategy=str(data.get("strategy", "structural")),
        include_extensions=_strings("include_extensions"),
        exclude_globs=_strings("exclude_globs"),
        max_file_bytes=int(_require(data, "max_file_bytes", path, int)),
        fixed_chunk_lines=int(_require(fixed, "chunk_lines", path, int)),
        fixed_overlap_lines=int(fixed.get("overlap_lines", 0)),
        structural_max_lines=int(_require(structural, "max_lines", path, int)),
        structural_merge_below=int(structural.get("merge_below_lines", 0)),
        block_patterns=tuple(patterns),
    )
    if corpus.strategy not in ("fixed", "structural"):
        raise DomainSchemaError(f"{path}: неизвестная стратегия разбиения {corpus.strategy!r}")
    if corpus.fixed_overlap_lines >= corpus.fixed_chunk_lines:
        raise DomainSchemaError(f"{path}: перекрытие окон не меньше самого окна")
    return corpus


def _load_checks(path: Path) -> Optional[DomainChecks]:
    """Необязательный раздел пакета: правила проверки артефакта задачи.

    Отсутствие файла означает, что домен не объявляет проверок — тогда задача проверится только
    модельным ревью. Сломанный файл обязан остановить загрузку: молча пропущенная проверка
    выглядит как пройденная.
    """
    if not path.is_file():
        return None
    data = _read_json(path)
    raw_forbidden = data.get("forbidden", [])
    if not isinstance(raw_forbidden, list):
        raise DomainSchemaError(f"{path}: поле «forbidden» должно быть списком")
    forbidden = []
    for item in raw_forbidden:
        if not isinstance(item, dict):
            raise DomainSchemaError(f"{path}: правило «forbidden» должно быть объектом")
        pattern = str(item.get("pattern", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not pattern or not reason:
            raise DomainSchemaError(f"{path}: у правила нужны и «pattern», и «reason»")
        try:
            re.compile(pattern)
        except re.error as error:
            raise DomainSchemaError(f"{path}: образец {pattern!r} не компилируется ({error})") from error
        forbidden.append(ForbiddenRule(pattern=pattern, reason=reason))

    def _strings(key: str) -> Tuple[str, ...]:
        raw = data.get(key, [])
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            raise DomainSchemaError(f"{path}: поле «{key}» должно быть списком непустых строк")
        return tuple(item.strip() for item in raw)

    raw_build = data.get("build", {})
    if not isinstance(raw_build, dict):
        raise DomainSchemaError(f"{path}: поле «build» должно быть объектом")
    return DomainChecks(
        forbidden=tuple(forbidden),
        spec_required=_strings("spec_required"),
        spec_globs=_strings("spec_globs"),
        build_tool=str(raw_build.get("tool", "")).strip(),
        build_steps=tuple(
            str(step).strip() for step in raw_build.get("steps", []) if str(step).strip()
        ),
        build_description=str(raw_build.get("description", "")).strip(),
    )


def _load_ops(path: Path) -> Optional[DomainOps]:
    """Необязательный раздел пакета: правила конвейера операций.

    Отсутствие файла означает, что домен не объявляет конвейера — тогда операции над проектом
    недоступны. Сломанный файл обязан остановить загрузку: молча собранная не та команда — это
    сборка не того пакета и подпись не тем ключом.
    """
    if not path.is_file():
        return None
    data = _read_json(path)

    def _strings(key: str) -> Tuple[str, ...]:
        raw = data.get(key, [])
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            raise DomainSchemaError(f"{path}: поле «{key}» должно быть списком непустых строк")
        return tuple(item.strip() for item in raw)

    def _template(key: str) -> Tuple[str, ...]:
        raw = data.get(key)
        if not isinstance(raw, list) or not raw or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            raise DomainSchemaError(f"{path}: поле «{key}» должно быть непустым списком строк")
        return tuple(item.strip() for item in raw)

    tool = _require(data, "tool", path, dict)
    raw_arch = _require(data, "architectures", path, dict)
    architectures = []
    for name, note in raw_arch.items():
        if not str(name).strip() or not str(note).strip():
            raise DomainSchemaError(f"{path}: у архитектуры нужны имя и пояснение")
        architectures.append((str(name).strip(), str(note).strip()))
    if not architectures:
        raise DomainSchemaError(f"{path}: список архитектур пуст")
    raw_steps = _require(data, "steps", path, dict)
    steps = []
    for name, value in raw_steps.items():
        if not isinstance(value, list) or not value:
            raise DomainSchemaError(f"{path}: шаг «{name}» должен быть непустым списком строк")
        if any(not isinstance(part, str) or not part.strip() for part in value):
            raise DomainSchemaError(f"{path}: шаг «{name}» содержит пустую часть команды")
        steps.append((str(name), tuple(str(part).strip() for part in value)))
    if not steps:
        raise DomainSchemaError(f"{path}: конвейер не объявляет ни одного шага")
    signature = data.get("signature", {})
    if not isinstance(signature, dict):
        raise DomainSchemaError(f"{path}: поле «signature» должно быть объектом")
    default_arch = str(data.get("default_architecture", "")).strip()
    if default_arch and default_arch not in {name for name, _ in architectures}:
        raise DomainSchemaError(f"{path}: архитектура по умолчанию {default_arch!r} не объявлена")
    return DomainOps(
        tool_env=str(_require(tool, "env", path, str)).strip(),
        tool_default=str(_require(tool, "default", path, str)).strip(),
        target_pattern=str(_require(data, "target_pattern", path, str)).strip(),
        architectures=tuple(architectures),
        default_architecture=default_arch or architectures[0][0],
        state_dir=str(data.get("state_dir", ".aurora")).strip() or ".aurora",
        steps=tuple(steps),
        package_globs=_strings("package_globs"),
        app_id_globs=_strings("app_id_globs"),
        signature_key_env=str(signature.get("key_env", "")).strip(),
        signature_signed=tuple(str(item) for item in signature.get("signed_markers", []) if str(item)),
        signature_unsigned=tuple(str(item) for item in signature.get("unsigned_markers", []) if str(item)),
        forbidden_commands=_strings("forbidden_commands"),
        package_check=tuple(str(part) for part in data.get("package_check_command", []) if str(part)),
    )


def available_domains(domains_dir: Optional[Path] = None) -> Tuple[str, ...]:
    """Идентификаторы установленных пакетов — по каталогам с `domain.json`."""
    root = Path(domains_dir or config.DOMAINS_DIR)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(child.name for child in root.iterdir() if (child / "domain.json").is_file())
    )


def load_domain(domain_id: str, domains_dir: Optional[Path] = None) -> Domain:
    root = Path(domains_dir or config.DOMAINS_DIR)
    path = root / domain_id
    data = _read_json(path / "domain.json")

    schema = _require(data, "schema", path / "domain.json", int)
    if schema != SCHEMA_VERSION:
        raise DomainSchemaError(
            f"{path / 'domain.json'}: версия схемы {schema} не поддерживается "
            f"(ожидается {SCHEMA_VERSION})"
        )
    if data.get("id") != domain_id:
        raise DomainSchemaError(
            f"{path / 'domain.json'}: поле «id» = {data.get('id')!r} не совпадает "
            f"с именем каталога {domain_id!r}"
        )

    domain = Domain(
        id=domain_id,
        title=str(_require(data, "title", path / "domain.json", str)),
        platform=str(_require(data, "platform", path / "domain.json", str)),
        docs_version=str(data.get("docs_version", "")),
        local_sdk_version=str(data.get("local_sdk_version", "")),
        responses_language=str(data.get("responses_language", "ru")),
        is_default=bool(data.get("default", False)),
        markers=_load_markers(data, path / "domain.json"),
        invariants=_load_invariants(path / "invariants.json"),
        path=path,
        servers=_load_servers(path / "servers.json"),
        tools_path=(path / "tools.json") if (path / "tools.json").is_file() else None,
        docs=_load_docs(path / "docs.json"),
        corpus=_load_corpus(path / "corpus.json"),
        checks=_load_checks(path / "validation.json"),
        ops=_load_ops(path / "ops.json"),
        memory=_load_memory(path / "memory.json"),
        profile=_load_profile(path / "profile.json"),
    )
    # Промпты обязательны: без них домен не сможет ни отвечать, ни отказать.
    domain.prompt("system")
    domain.prompt("refusal")
    return domain


def default_domain_id(domains_dir: Optional[Path] = None) -> str:
    """Пакет, помеченный `default`, — или единственный установленный, если пометки нет."""
    identifiers = available_domains(domains_dir)
    if not identifiers:
        raise DomainError(
            f"в каталоге {domains_dir or config.DOMAINS_DIR} нет ни одного пакета домена"
        )
    marked = [name for name in identifiers if load_domain(name, domains_dir).is_default]
    if marked:
        return marked[0]
    if len(identifiers) == 1:
        return identifiers[0]
    raise DomainError(
        "несколько пакетов домена и ни один не помечен как домен по умолчанию: "
        + ", ".join(identifiers)
    )


def detect_domain(
    root: Path, domains_dir: Optional[Path] = None
) -> Optional[Tuple[Domain, Tuple[str, ...]]]:
    """Наиболее подходящий домен по маркерам целевого репозитория."""
    if root is None or not Path(root).is_dir():
        return None
    scored: List[Tuple[int, Domain, Tuple[str, ...]]] = []
    for domain_id in available_domains(domains_dir):
        domain = load_domain(domain_id, domains_dir)
        evidence = tuple(
            found
            for found in (marker.matches(Path(root)) for marker in domain.markers)
            if found is not None
        )
        if evidence:
            scored.append((len(evidence), domain, evidence))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise DomainError(
            "маркеры репозитория подходят нескольким доменам одинаково: "
            + ", ".join(domain.id for _, domain, _ in scored if _ == scored[0][0])
            + " — укажите домен явно"
        )
    _, domain, evidence = scored[0]
    return domain, evidence


def select_domain(
    root: Optional[Path] = None,
    explicit: Optional[str] = None,
    env: Optional[str] = None,
    domains_dir: Optional[Path] = None,
) -> DomainSelection:
    """Выбор домена по приоритету: аргумент → переменная окружения → маркеры → по умолчанию."""
    requested = (explicit or "").strip() or (env if env is not None else config.DOMAIN_ENV).strip()
    if requested:
        if requested not in available_domains(domains_dir):
            known = ", ".join(available_domains(domains_dir)) or "нет ни одного"
            raise UnknownDomainError(
                f"домен «{requested}» не найден; установленные домены: {known}"
            )
        source = "аргумент" if explicit else "переменная окружения"
        return DomainSelection(domain=load_domain(requested, domains_dir), source=source)

    detected = detect_domain(root, domains_dir) if root is not None else None
    if detected is not None:
        domain, evidence = detected
        return DomainSelection(
            domain=domain, source="маркеры репозитория", evidence=evidence
        )

    domain_id = default_domain_id(domains_dir)
    return DomainSelection(domain=load_domain(domain_id, domains_dir), source="по умолчанию")


def identifiers_of(domain: Domain) -> Sequence[str]:
    """Слова домена, которых не должно быть в ядре и интерфейсе (используется тестом).

    Берутся только достаточно длинные слова: короткие вроде «qt6» или «.key» дают ложные
    срабатывания на обычных именах атрибутов, и тест перестаёт что-либо значить.
    """
    words = {domain.id, domain.platform.lower()}
    for marker in domain.markers:
        words.add(marker.glob.lower())
    for invariant in domain.invariants:
        words.update(word.lower() for word in invariant.forbidden)
    return tuple(sorted(word for word in words if len(word) >= 6))
