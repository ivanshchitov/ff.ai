"""Документация портала как корпус ответа: разделы, версия, кандидаты и фрагменты.

Поиск делает приложение, а не модель: инструменты сервера документации вызываются напрямую,
а модели достаются уже готовые фрагменты с путём документа, разделом и версией — их она и обязана
назвать в ответе. Локальных копий документации здесь нет: корпус живёт на портале, поэтому его
читают в момент вопроса, и он не расходится с версиями.

Домена модуль не знает: раздел по умолчанию, слова-признаки вопросов о версии, имена инструментов
и имя источника приходят из данных пакета (`core.domains.DomainDocs`), а способ подключения —
из записи реестра (`core.mcp_registry.MCPServerSpec`). Модель здесь не вызывается ни разу.

Сбой сервера — это состояние снимка (`unavailable`) с причиной, а не исключение: недоступный
портал обязан оставить сессию рабочей. Исключения наружу не уходят вообще — любая ошибка
превращается в отчёт.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import config, docs_reranking
from .domains import DomainDocs
from .mcp_registry import MCPServerSpec

# Состояния поиска: их печатает отчёт и по ним принимает решения агент. `disabled` ставит
# вызывающий, когда режим выключен: при выключенном режиме сюда не обращаются вовсе.
STATUS_OK = "ok"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_NO_MATCHES = "no_matches"
STATUS_RERANK_FAILED = "rerank_failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DISABLED = "disabled"

# Запас текста перед найденным местом: фрагмент не должен обрываться ровно на совпадении,
# иначе цитата из него читается как обрывок. Не больше четверти фрагмента, чтобы само
# совпадение гарантированно осталось внутри окна.
WINDOW_LEAD_CHARS = 200

# Слово короче этого в тексте документа не ищем: «как», «для», «что» встречаются где угодно,
# и окно вокруг первого такого совпадения не имело бы отношения к вопросу.
MIN_MATCH_WORD_CHARS = 4

# Склеенные сниппеты одного документа — по строке на сниппет: так видно, что это выборки.
SNIPPET_JOIN = "\n"

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class DocsCandidate:
    """Попадание поиска: путь документа, его раздел и склеенные сниппеты."""

    path: str
    title: str
    section: str
    url: str = ""
    snippet: str = ""
    # Оценка второй ступени: 0.0 означает «не оценивался» (режим baseline), а не «оценён в ноль».
    score: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class DocsFragment:
    """Фрагмент, доставленный модели: по его идентификатору и тексту проверяется ссылка в ответе."""

    identifier: str
    source: str
    title: str
    section: str
    version: str
    text: str
    truncated: bool = False


@dataclass(frozen=True)
class DocsReport:
    """Снимок поиска: что спросили, где искали, в какой версии и что доставили.

    Снимок — граница изоляции: интерфейс и агент рендерят его и не читают внутренности
    поисковика. Недоступный сервер — это `status` с причиной в `error`, а не исключение.
    """

    query: str
    sections: Tuple[str, ...]
    version: str
    status: str
    candidates: Tuple[DocsCandidate, ...] = ()
    fragments: Tuple[DocsFragment, ...] = ()
    error: str = ""
    sdk_version: str = ""
    # Прошла ли вторая ступень: нужна отчёту, чтобы отличить «не оценивали» от «оценили и отсеяли».
    rated: bool = False
    threshold: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class _Unavailable(Exception):
    """Сбой обращения к серверу документации: причина уходит в снимок, наружу не поднимается."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DocsRetriever:
    """Поиск по документации домена: разделы, версия, кандидаты и фрагменты.

    Клиент сервера создаётся фабрикой при первом обращении и живёт вместе с объектом:
    соединение поднимается и закрывается внутри каждого вызова инструмента, поэтому держать
    больше одного клиента незачем. Список версий кэшируется на объект — он нужен каждому
    поиску, а меняется редко.
    """

    def __init__(
        self,
        client_factory: Callable[[MCPServerSpec], Any],
        spec: MCPServerSpec,
        docs: DomainDocs,
        sdk_version: str = "",
        *,
        version: str = "",
        max_fragments: Optional[int] = None,
        fragment_chars: Optional[int] = None,
        snippet_chars: Optional[int] = None,
        search_limit: Optional[int] = None,
        fetch_documents: bool = True,
    ) -> None:
        self.client_factory = client_factory
        self.spec = spec
        self.docs = docs
        # Версия установленного SDK: уходит в снимок как есть, чтобы ответ мог назвать рассинхрон.
        self.sdk_version = str(sdk_version or "")
        # Заданная пользователем версия документации; пустая строка означает «взять актуальную».
        self.version = str(version or "")
        self._max_fragments = _bound(max_fragments, config.DOCS_MAX_FRAGMENTS)
        self._fragment_chars = _bound(fragment_chars, config.DOCS_FRAGMENT_CHARS)
        self._snippet_chars = _bound(snippet_chars, config.DOCS_SNIPPET_CHARS)
        self._search_limit = _bound(search_limit, config.DOCS_SEARCH_LIMIT)
        self._fetch_documents = bool(fetch_documents)
        self._client_instance: Any = None
        self._cached_versions: Optional[Tuple[str, ...]] = None
        self._cached_latest = ""

    # --- публичные операции ---------------------------------------------------------------

    @property
    def versions(self) -> Tuple[str, ...]:
        """Список версий документации; пустой кортеж, если сервер не ответил.

        Отчёт и команды не должны падать из-за недоступного портала, поэтому сбой здесь
        показывается пустым списком, а сбойный поиск сообщает причину сам.
        """
        if self._cached_versions is None:
            try:
                return self._load_versions()
            except Exception:  # noqa: BLE001 - свойство-снимок не поднимает исключений
                return ()
        return self._cached_versions

    def refresh_versions(self) -> Tuple[str, ...]:
        """Забывает кэш версий и запрашивает список заново."""
        self._cached_versions = None
        self._cached_latest = ""
        return self.versions

    def search(
        self,
        question: str,
        *,
        rate: Optional[Callable[[str, Sequence[DocsCandidate]], Any]] = None,
        threshold: Optional[float] = None,
    ) -> DocsReport:
        """Ищет по документации запросом из вопроса и приносит фрагменты верхних документов.

        Ни один сбой не поднимается наружу: непонятный ответ, отказ инструмента, недоступный
        сервер и негодная оценка кандидатов — это состояния снимка. Фрагменты строятся из полного
        текста документа, а если он не получен — из сниппетов поиска.

        `rate` — вторая ступень отбора: поиск делает приложение, а оценку кандидатов — модель
        сессии, поэтому она приходит функцией. Без неё (или в режиме baseline) доставляются все
        найденные кандидаты, как до появления ступени.
        """
        sections = self._sections(question)
        version = self.version
        try:
            # Один поиск — одно соединение: список версий, запросы по разделам и полные тексты
            # идут через него. Иначе каждый инструмент поднимал бы серверный процесс заново.
            with self._client().session():
                if not version:
                    version = self._documentation_version()
                candidates = self._collect(question, sections, version)
                if not candidates:
                    return self._report(question, sections, version, STATUS_NO_CANDIDATES)
                if rate is None:
                    fragments = self._fragments(question, candidates, version)
                else:
                    limit = float(config.DOCS_RELEVANCE_THRESHOLD if threshold is None else threshold)
                    marked, selected = self._rate(question, candidates, rate, limit)
                    if selected is None:
                        return self._report(
                            question,
                            sections,
                            version,
                            STATUS_RERANK_FAILED,
                            candidates=marked,
                            error=self.rerank_error(),
                            rated=True,
                            threshold=limit,
                        )
                    if selected.empty:
                        return self._report(
                            question,
                            sections,
                            version,
                            STATUS_NO_MATCHES,
                            candidates=marked,
                            rated=True,
                            threshold=limit,
                        )
                    candidates = selected.kept
                    fragments = self._fragments(question, candidates, version)
                    return self._report(
                        question,
                        sections,
                        version,
                        STATUS_OK,
                        candidates=marked,
                        fragments=fragments,
                        rated=True,
                        threshold=limit,
                    )
        except _Unavailable as failure:
            return self._report(question, sections, version, STATUS_UNAVAILABLE, error=failure.reason)
        except Exception as error:  # noqa: BLE001 - снимок обязан пережить любую ошибку разбора
            return self._report(question, sections, version, STATUS_UNAVAILABLE, error=_reason(error))
        return self._report(
            question, sections, version, STATUS_OK, candidates=candidates, fragments=fragments
        )

    # --- разделы, версия ------------------------------------------------------------------

    def _rate(
        self,
        question: str,
        candidates: Tuple[DocsCandidate, ...],
        rate: Callable[[str, Sequence[DocsCandidate]], Any],
        threshold: float,
    ) -> Tuple[Tuple[DocsCandidate, ...], Any]:
        """Оценивает кандидатов и отбирает прошедших порог.

        Негодная оценка — не исключение наружу, а признак того, что отбор не состоялся: вторая
        ступень существует ради качества, и «доставим что было» её обесценивает.
        """
        try:
            ratings = rate(question, candidates)
        except Exception as error:  # noqa: BLE001 - причину показываем пользователю текстом
            self._last_rerank_error = _reason(error)
            return candidates, None
        try:
            parsed = docs_reranking.parse_response(str(ratings or ""), candidates)
            marked = docs_reranking.mark(candidates, parsed)
            return marked, docs_reranking.select(marked, parsed, threshold)
        except Exception as error:  # noqa: BLE001 - негодный ответ оценщика тоже состояние
            self._last_rerank_error = _reason(error)
            return candidates, None

    def rerank_error(self) -> str:
        """Причина последней неудавшейся оценки, если она была."""
        return getattr(self, "_last_rerank_error", "")

    def _sections(self, question: str) -> Tuple[str, ...]:
        """Разделы поиска: основной и, для вопроса о версии, примечания к выпуску.

        Состав разделов и слова-признаки — данные пакета домена: ядро не знает ни названий
        разделов портала, ни того, что вопрос о версии ищется где-то ещё.
        """
        lowered = question.lower()
        sections: List[str] = []
        names = [self.docs.default_section]
        if any(
            keyword.lower() in lowered for keyword in self.docs.version_keywords if keyword.strip()
        ):
            names.extend(self.docs.version_sections)
        for name in names:
            section = name.strip()
            if section and section not in sections:
                sections.append(section)
        return tuple(sections)

    def _documentation_version(self) -> str:
        """Версия для поиска: помеченная сервером как актуальная, а не самая старшая.

        Пустой список версий — не ошибка: поиск идёт без версии. Сбой самого запроса версий —
        ошибка: без него нельзя утверждать, что поиск идёт по актуальной документации.
        """
        if self._cached_versions is None:
            self._load_versions()
        versions = self._cached_versions or ()
        if not versions:
            return ""
        return self._cached_latest or versions[0]

    def _load_versions(self) -> Tuple[str, ...]:
        data = self._tool_json(self.docs.tools.versions, {})
        versions, latest = _parse_versions(data)
        self._cached_versions = versions
        self._cached_latest = latest
        return versions

    # --- поиск и фрагменты ----------------------------------------------------------------

    def _collect(
        self, question: str, sections: Tuple[str, ...], version: str
    ) -> Tuple[DocsCandidate, ...]:
        """Кандидаты по всем разделам, сведённые по пути: порядок разделов сохраняется.

        Один документ может попасться в двух разделах: тогда остаётся первое попадание (и его
        раздел), а сниппеты складываются — так у документа не теряется найденное во втором месте.
        """
        order: List[str] = []
        meta: Dict[str, Dict[str, str]] = {}
        snippets: Dict[str, List[str]] = {}
        for section in sections:
            arguments: Dict[str, Any] = {
                "query": question,
                "index": section,
                "limit": self._search_limit,
            }
            if version:
                arguments["version"] = version
            data = self._tool_json(self.docs.tools.search, arguments)
            for item in _results(data):
                path = str(item.get("path", "") or "").strip()
                if not path:
                    continue
                if path not in meta:
                    order.append(path)
                    meta[path] = {
                        "title": str(item.get("title", "") or "").strip(),
                        "section": str(item.get("index", "") or "").strip() or section,
                        "url": str(item.get("url", "") or "").strip(),
                    }
                    snippets[path] = []
                snippets[path].extend(_snippet_list(item))
        return tuple(
            DocsCandidate(
                path=path,
                title=meta[path]["title"],
                section=meta[path]["section"],
                url=meta[path]["url"],
                snippet=_clip(SNIPPET_JOIN.join(snippets[path]), self._snippet_chars),
            )
            for path in order
        )

    def _fragments(
        self, question: str, candidates: Tuple[DocsCandidate, ...], version: str
    ) -> Tuple[DocsFragment, ...]:
        """Фрагменты верхних кандидатов: окно полного текста, а при сбое — сниппеты.

        Кандидат остаётся в снимке в любом случае: он найден и назван, даже если полный текст
        документа не отдался. Фрагмент без текста не доставляется — цитировать в нём нечего.
        """
        selected = candidates[: self._max_fragments] if self._max_fragments > 0 else ()
        fragments: List[DocsFragment] = []
        for candidate in selected:
            title = candidate.title
            full = ""
            if self._fetch_documents:
                full, title = self._document_text(candidate)
            if full:
                text, truncated = _window(full, question, self._fragment_chars)
            else:
                text, truncated = candidate.snippet, False
            if not text:
                continue
            fragments.append(
                DocsFragment(
                    identifier=candidate.path,
                    source=self.docs.source,
                    title=title or candidate.path,
                    section=candidate.section,
                    version=version,
                    text=text,
                    truncated=truncated,
                )
            )
        return tuple(fragments)

    def _document_text(self, candidate: DocsCandidate) -> Tuple[str, str]:
        """Полный текст документа; сбой или пустой ответ — пустая строка.

        Недоступность одного документа не отменяет поиск: у кандидата остаются сниппеты,
        и он всё равно доходит до модели как источник.
        """
        try:
            data = self._tool_json(
                self.docs.tools.document, {"path": candidate.path, "index": candidate.section}
            )
        except _Unavailable:
            return "", candidate.title
        text = str(data.get("content", "") or "").strip()
        title = str(data.get("title", "") or "").strip() or candidate.title
        return text, title

    # --- обращения к серверу --------------------------------------------------------------

    def _client(self) -> Any:
        """Клиент сервера: создаётся фабрикой один раз, соединение — внутри каждого вызова."""
        if self._client_instance is None:
            try:
                self._client_instance = self.client_factory(self.spec)
            except Exception as error:  # noqa: BLE001 - причина подключения важна пользователю
                raise _Unavailable(
                    f"подключение к серверу «{self.spec.name}»: {_reason(error)}"
                ) from error
        return self._client_instance

    def _call(self, tool: str, arguments: Mapping[str, Any]) -> str:
        """Вызов инструмента: отказ, сбой и пустой ответ — причина, а не текст для модели."""
        try:
            result = self._client().call_tool(tool, dict(arguments))
        except Exception as error:  # noqa: BLE001 - чужая библиотека, причину показываем текстом
            raise _Unavailable(f"вызов «{tool}»: {_reason(error)}") from error
        if getattr(result, "is_error", False):
            reason = str(getattr(result, "text", "") or "").strip()
            raise _Unavailable(f"инструмент «{tool}» ответил ошибкой: {reason or 'без причины'}")
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            raise _Unavailable(f"инструмент «{tool}» вернул пустой ответ")
        return text

    def _tool_json(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Ответ инструмента как объект JSON: сервер отдаёт JSON текстом, разбор защитный."""
        data = _loads(self._call(tool, arguments))
        if data is None:
            raise _Unavailable(f"ответ «{tool}» не разобран как JSON")
        return data

    # --- внутреннее -----------------------------------------------------------------------

    def _report(
        self,
        question: str,
        sections: Tuple[str, ...],
        version: str,
        status: str,
        *,
        candidates: Tuple[DocsCandidate, ...] = (),
        fragments: Tuple[DocsFragment, ...] = (),
        error: str = "",
        rated: bool = False,
        threshold: float = 0.0,
    ) -> DocsReport:
        return DocsReport(
            query=question,
            sections=tuple(sections),
            version=version,
            status=status,
            candidates=tuple(candidates),
            fragments=tuple(fragments),
            error=error,
            sdk_version=self.sdk_version,
            rated=rated,
            threshold=threshold,
        )


def _bound(value: Optional[int], default: int) -> int:
    """Граница из аргумента или из конфигурации: значения по умолчанию читаются при вызове."""
    return int(default if value is None else value)


def _window(text: str, question: str, limit: int) -> Tuple[str, bool]:
    """Окно текста вокруг первого совпадения слова вопроса; без совпадения — начало текста.

    Запас перед совпадением ограничен четвертью фрагмента, поэтому найденное место всегда
    остаётся внутри окна, даже если фрагмент короткий. Пометка об обрезке нужна, когда
    в окно не влез весь документ.
    """
    if limit <= 0:
        return "", True
    if len(text) <= limit:
        return text, False
    position = _match_position(text, _words(question))
    if position < 0:
        return text[:limit], True
    lead = min(WINDOW_LEAD_CHARS, limit // 4)
    start = max(0, position - lead)
    return text[start : start + limit], True


def _words(question: str) -> Tuple[str, ...]:
    """Слова вопроса, по которым ищем место в документе; короткие не годятся."""
    return tuple(
        word for word in _WORD.findall(question.lower()) if len(word) >= MIN_MATCH_WORD_CHARS
    )


def _match_position(text: str, words: Tuple[str, ...]) -> int:
    """Позиция самого раннего вхождения любого из слов; без учёта регистра."""
    lowered = text.lower()
    best = -1
    for word in words:
        position = lowered.find(word)
        if position >= 0 and (best < 0 or position < best):
            best = position
    return best


def _clip(text: str, limit: int) -> str:
    """Обрезка по объёму без пометок: пометку об обрезке несёт флаг фрагмента."""
    text = text.strip()
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[:limit]


def _results(data: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    """Попадания поиска; отсутствие списка — это пустой поиск, а не ошибка."""
    raw = data.get("results")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _snippet_list(item: Mapping[str, Any]) -> List[str]:
    """Сниппеты попадания: сервер отдаёт список, но одиночная строка тоже принимается."""
    raw = item.get("snippets")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if not isinstance(raw, (list, tuple)):
        return []
    return [value.strip() for value in raw if isinstance(value, str) and value.strip()]


def _parse_versions(data: Mapping[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """Список версий и помеченная сервером как актуальная; разбор защитный.

    Актуальная — та, у которой стоит признак сервера, а не самая старшая по номеру: на портале
    разработчиков это разные версии.
    """
    raw = data.get("versions")
    if not isinstance(raw, (list, tuple)):
        return (), ""
    versions: List[str] = []
    latest = ""
    for item in raw:
        name, is_latest = _version_entry(item)
        if not name:
            continue
        versions.append(name)
        if is_latest and not latest:
            latest = name
    return tuple(versions), latest


def _version_entry(item: Any) -> Tuple[str, bool]:
    if isinstance(item, str):
        return item.strip(), False
    if isinstance(item, Mapping):
        return str(item.get("version", "") or "").strip(), _flag(item.get("latest"))
    return "", False


def _flag(value: Any) -> bool:
    """Признак «актуальная» приходит и числом (0/1), и строкой."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return False


def _loads(text: str) -> Optional[Mapping[str, Any]]:
    """Разбор ответа инструмента: JSON, а при обёртке — первый объект внутри текста."""
    data = _json_object(text)
    if data is not None:
        return data
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return _json_object(text[start : end + 1])
    return None


def _json_object(text: str) -> Optional[Mapping[str, Any]]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, Mapping) else None


def _reason(error: BaseException, depth: int = 0) -> str:
    """Читаемая причина: SDK заворачивает сбои задач в группу исключений."""
    nested = getattr(error, "exceptions", None) if depth < 4 else None
    if nested:
        return _reason(nested[0], depth + 1)
    return f"{type(error).__name__}: {error}".strip()
