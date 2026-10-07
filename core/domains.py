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
