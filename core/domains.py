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

from . import config
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
