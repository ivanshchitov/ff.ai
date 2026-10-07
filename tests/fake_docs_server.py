#!/usr/bin/env python3
"""Заглушка MCP-сервера документации портала: маленький корпус и формы ответов как у живого.

Сервер повторяет четыре инструмента сервера `aurora-docs` — и ровно в том виде, в каком их
отдаёт портал: **JSON-текстом**, а не готовой структурой. Именно этот текст разбирает
`core/docs_retrieval.py`, поэтому форма ответа здесь и есть предмет проверки:

- `get_doc_versions()` → `{"versions": [{"version": "5.2.1", "latest": 1}, ...]}`: актуальная
  помечена `latest: 1` и нарочно **не самая старшая** (5.2.2 старше) — так проверяется, что
  выбор версии идёт по признаку сервера, а не по сортировке строк;
- `search(query, index, version, limit)` → `{"query", "total", "results": [...]}`, у попадания
  `path`, `url`, `title`, `index`, `version` и 1–2 сниппета по ~300 символов;
- `get_document(path, index)` → `{"path", "title", "content"}` с markdown-текстом;
- `echo(message)` → переданный текст: инструмент проверки соединения.

Корпус (четыре документа, у каждого своя характерная фраза — `Document.marker`, её и цитируют
тесты): руководство по геопозиции и руководство по сборке в разделе `docs`, примечания к выпуску
5.2.0 в разделе `release_notes`, статья про автотесты Qt в разделе `articles`. Фразы-маркеры
нарочно написаны без markdown-разметки внутри и занимают строку целиком: их приводят дословно,
а разметка или перенос строки попали бы в цитату вместе с текстом (проверка цитат переносы
нормализует, а `assert marker in content` — нет). Документы нарочно короткие: каждый целиком
влезает в одно окно фрагмента (`config.DOCS_FRAGMENT_CHARS`, 1500 символов), поэтому фраза-маркер
всегда попадает в доставленный модели фрагмент, где бы ни нашлось совпадение. Удлинить документ
сверх этого потолка нельзя, не пересмотрев сквозные тесты цитат.

Поиск — совместимый с живым порталом, но устроенный проще: подстрока без учёта регистра по
заголовку и тексту (первая ступень), а если её нет целиком — по словам запроса длиной от четырёх
символов (вторая: нашлись все слова, третья: хотя бы одно), с учётом усечённых основ на конце,
чтобы «геопозицию» находил документ со словом «геопозиция». Служебная лексика вопроса («как»,
«можно», «пожалуйста») из поиска исключена: без этого вопрос вне корпуса давал бы случайные
попадания. Порядок результатов — ступень совпадения, число найденных слов, совпадение в заголовке,
число вхождений в текст, затем путь. У попадания `index` — сам раздел (в режиме `all` — раздел
документа), `version` — запрошенная версия, а `latest` разворачивается в актуальную (5.2.1).

Режимы через argv (по умолчанию — все четыре инструмента и полный корпус):

- `--empty` — поиск всегда пустой (`total: 0`), документы и версии остаются на месте;
- `--no-versions` — пустой список версий: проверка ветки «версию взять негде»;
- `--broken-document` — `get_document` всегда ошибочный результат: проверка ветки «фрагмент
  доставить не удалось».

Логи — на уровне `ERROR`: в stdio-транспорте stdout занят протоколом, и всё лишнее, что туда
попадёт, ломает рукопожатие. Запускать можно и системным `python3` (так делает `tests/conftest.py`
через `FFAI_MCP_COMMAND`): процесс сам перезапустится интерпретатором `.venv`, потому что `mcp`
есть только там (см. `reexec_in_venv`). Модуль зависит от пакета `mcp` (2.3,
`mcp.server.mcpserver.MCPServer`) и стандартной библиотеки; ядро и интерфейс приложения не трогает.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:  # только для аннотаций: импорт `mcp` отложен до перезапуска в .venv
    from mcp.server.mcpserver import MCPServer

BASE_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"


def reexec_in_venv() -> None:
    """Перезапускает процесс интерпретатором проекта, если пакета `mcp` в этом нет.

    `tests/conftest.py` объявляет команду запуска как `python3`, а системный питон пакета `mcp`
    не имеет — без перезапуска заглушка падала бы на импорте и выглядела бы сломанным сервером
    вместо ошибки окружения.

    Признак здесь — наличие пакета, а не путь к исполняемому файлу (как в `fake_mcp_server.py`):
    в этом окружении `.venv/bin/python` — символическая ссылка на тот же бинарник, что и
    системный `python3`, поэтому сравнение `sys.executable` с `.venv/bin/python` по `resolve()`
    всегда говорит «уже в .venv» и не перезапускает процесс. Проверка `find_spec("mcp")` от
    устройства `.venv` не зависит.
    """
    if os.environ.get("FFAI_NO_REEXEC") == "1" or not VENV_PYTHON.exists():
        return
    if importlib.util.find_spec("mcp") is not None:
        return
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

SERVER_NAME = "aurora-docs"
SERVER_VERSION = "1.0"
SERVER_INSTRUCTIONS = (
    "Заглушка сервера документации портала разработчиков ОС Аврора: список версий, поиск по "
    "разделам, полный текст документа и проверка соединения."
)

TOOL_NAMES = ("get_doc_versions", "search", "get_document", "echo")

SITE = "https://developer.auroraos.ru"
SNIPPET_CHARS = 300
MAX_SNIPPETS = 2
MAX_SEARCH_LIMIT = 10
MIN_TOKEN_CHARS = 4
DOCUMENT_ERROR = "документ недоступен: сервер вернул ошибку"

# Слова вопроса и служебная лексика: сами по себе они ничего не значат для поиска, а в корпусе
# встречаются — без их исключения вопрос вне корпуса давал бы случайные попадания. Названия
# продукта («Аврора») и слова о версии («версия») нарочно оставлены: они встречаются в документах
# и как раз нужны вопросу о версии.
STOP_WORDS = frozenset(
    """
    как какой какая какие какое когда где чего чем чему чтобы зачем почему можно нужно надо
    есть если или либо только также тоже очень более менее это этот эта эти тот та те всё все
    весь вся быть был была было были будет же ли не нет да но по на в во из с со у о об от до
    за для при про над под к ко без через между том то там тут так свой своя свои пожалуйста
    скажи расскажи подскажи вопрос ответ сделать делает делаю делал может могут нужен нужна
    нужны использовать используется приложение приложении приложения проект проекте
    """.split()
)

# Ответ живого портала: актуальная версия помечена `latest: 1` и она не самая старшая в списке.
DOC_VERSIONS: Tuple[Tuple[str, int], ...] = (
    ("5.1.5", 0),
    ("5.2.1", 1),
    ("5.2.2", 0),
)
LATEST_VERSION = "5.2.1"


@dataclass(frozen=True)
class Document:
    """Документ корпуса: путь портала, раздел, заголовок, markdown-текст и фраза-маркер."""

    path: str
    index: str
    title: str
    content: str
    marker: str

    @property
    def url(self) -> str:
        return f"{SITE}/{self.path}"


POSITIONING = Document(
    path="doc/software_development/guides/cpp_api/positioning",
    index="docs",
    title="Геопозиция в C++ API",
    content="""# Геопозиция в C++ API

## Источник координат

Позицию устройства в ОС Аврора отдаёт класс `QGeoPositionInfoSource` из модуля Qt Positioning. Чтобы начать получать обновления, подпишитесь на сигнал `positionUpdated` и вызовите метод `startUpdates()`.
Координаты устройства приходят через сигнал positionUpdated в виде объекта QGeoPositionInfo.

Перед подпиской проверьте доступность источника методом `availableSources()`. Пустой список означает, что приложение собрано без модуля Qt Positioning.

## Разрешения и точность

Доступ к геопозиции требует разрешения `Location` в desktop-файле приложения. Без него источник завершится ошибкой `AccessError`, а координаты останутся недействительными. Точность зависит от выбранного метода определения: спутниковый приёмник точнее сетевого, но дольше выходит на первый фикс.

Значение координат проверяйте через `isValid()`. Метод `lastKnownPosition()` возвращает последнюю известную позицию, не дожидаясь нового обновления.

## Обновление из QML

Из QML тот же источник доступен как `PositionSource`. Свойства `latitude` и `longitude` обновляются после сигнала `positionChanged`.
""",
    marker="Координаты устройства приходят через сигнал positionUpdated в виде объекта QGeoPositionInfo",
)

BUILD = Document(
    path="doc/sdk/app_development/build/build_engine/sdk",
    index="docs",
    title="Сборка пакета средствами build-engine",
    content="""# Сборка пакета средствами build-engine

## Шаги сборки

Пакет для ОС Аврора собирается утилитой build-engine из состава SDK. Цель сборки задаётся флагом `-t`, например `AuroraOS-5.2.1-base-aarch64`. Готовые RPM-пакеты складываются в каталог build_aarch64/RPMS/.

Перед сборкой обновите цели SDK и проверьте ключи подписи. Подпись выполняется ключами Regular после сборки, до установки на устройство.

## Проверка результата

Наличие пакета в системе проверяют командой `rpm -q`. Помните о ловушке APM: менеджер пакетов видит только приложения, поставленные через штатный установщик. Если `rpm -q` не находит установленное приложение, смотрите список в APM, а не в базе RPM.
""",
    marker="Готовые RPM-пакеты складываются в каталог build_aarch64/RPMS/",
)

RELEASE_NOTES = Document(
    path="doc/release_notes/5.2.0",
    index="release_notes",
    title="Примечания к выпуску 5.2.0",
    content="""# Примечания к выпуску 5.2.0

## Геопозиция

В версии 5.2.0 появился фоновый режим источника геопозиции. Сигнал positionUpdated теперь приходит и при выключенном экране устройства. Для фонового режима добавьте разрешение `LocationBackground` в desktop-файл приложения.

## Сборка

Утилита build-engine получила цель `AuroraOS-5.2.1-base-aarch64` по умолчанию. Каталог сборки build_aarch64/RPMS/ больше не нужно создавать вручную.

## Совместимость

Обновление рассчитано на SDK версии не ниже 5.2.1. На более старых SDK часть изменений недоступна, включая фоновые обновления геопозиции.
""",
    marker="Сигнал positionUpdated теперь приходит и при выключенном экране устройства",
)

QT_TESTING = Document(
    path="articles/aurora_qt_testing",
    index="articles",
    title="Автотесты Qt-приложений на Авроре",
    content="""# Автотесты Qt-приложений на Авроре

## Сборка и запуск

Автотесты собираются вместе с основным проектом. Цель make check запускает все тесты, зарегистрированные через QtTest. Отчёт в формате JUnit понимает любая система непрерывной интеграции.

Для QML-кода тесты пишутся на самом QML через `TestCase` из модуля QtTest. Такой запуск требует графической сессии, поэтому агент CI использует offscreen-платформу Qt.

## Версии SDK

Портал описывает запуск тестов начиная с SDK 5.2.1. На более старых SDK цель тестов создаётся вручную, отдельным шагом сборки.
""",
    marker="Цель make check запускает все тесты, зарегистрированные через QtTest",
)

CORPUS: Tuple[Document, ...] = (POSITIONING, BUILD, RELEASE_NOTES, QT_TESTING)

# Фраза-маркер по пути документа: её цитируют тесты, поэтому таблица вынесена наружу.
MARKERS: Mapping[str, str] = {document.path: document.marker for document in CORPUS}


# --- формы ответов --------------------------------------------------------------------------


def versions_payload(no_versions: bool = False) -> Dict[str, Any]:
    """Ответ `get_doc_versions`: список версий с признаком актуальной."""
    if no_versions:
        return {"versions": []}
    return {
        "versions": [
            {"version": version, "latest": latest} for version, latest in DOC_VERSIONS
        ]
    }


def resolve_version(version: str) -> str:
    """Версия попадания: `latest` и пустая строка означают актуальную, остальное — как просили."""
    requested = (version or "").strip()
    if not requested or requested.lower() == "latest":
        return LATEST_VERSION
    return requested


def _tokens(query: str) -> List[str]:
    """Слова запроса длиной от четырёх символов без слов-заглушек, в порядке появления."""
    prepared = "".join(char if char.isalnum() else " " for char in query.lower())
    tokens: List[str] = []
    for word in prepared.split():
        if len(word) < MIN_TOKEN_CHARS or word in STOP_WORDS or word in tokens:
            continue
        tokens.append(word)
    return tokens


def locate(compact: str, needle: str) -> int:
    """Позиция слова в тексте (уже приведённом к нижнему регистру) или -1.

    Слово ищется целиком, а если целиком его нет — усечённой основой: «геопозицию» находит
    «геопозиция». Тот же приём, что у сниппета: совпадение слова и место вырезки считаются
    одинаково, иначе вырезка уходила бы к началу документа.
    """
    candidate = needle.lower()
    for cut in range(3):
        stem = candidate if cut == 0 else candidate[:-cut]
        if len(stem) < MIN_TOKEN_CHARS:
            break
        position = compact.find(stem)
        if position >= 0:
            return position
    return -1


def _token_present(haystack: str, token: str) -> bool:
    """Слово нашлось в тексте (приведённом к нижнему регистру) целиком или основой."""
    return locate(haystack, token) >= 0


def match_document(document: Document, query: str) -> Optional[Tuple[int, int, List[str]]]:
    """Совпадение документа с запросом: ступень, число найденных слов и сами слова.

    Ступень 0 — запрос целиком встречается в заголовке или тексте (правило живого портала);
    1 — нашлись все слова запроса; 2 — хотя бы одно. `None` — документ не подходит.
    """
    needle = query.strip().lower()
    if not needle:
        return None
    haystack = f"{document.title}\n{document.content}".lower()
    if needle in haystack:
        return (0, len(_tokens(needle)) or 1, [needle])
    tokens = _tokens(needle)
    if not tokens:
        return None
    hits = [token for token in tokens if _token_present(haystack, token)]
    if not hits:
        return None
    return (1 if len(hits) == len(tokens) else 2, len(hits), hits)


def snippet_at(text: str, position: int) -> str:
    """Вырезка ~`SNIPPET_CHARS` символов вокруг позиции; `position < 0` — начало текста.

    Пробелы схлопываются: живой портал отдаёт сниппет одной строкой, а переносы в markdown
    сделали бы длину вырезки случайной.
    """
    compact = " ".join(text.split())
    if not compact:
        return ""
    start = 0
    if position >= 0:
        start = max(0, min(position - SNIPPET_CHARS // 2, len(compact) - SNIPPET_CHARS))
    end = min(len(compact), start + SNIPPET_CHARS)
    fragment = compact[start:end]
    if start > 0:
        fragment = "… " + fragment
    if end < len(compact):
        fragment = fragment + " …"
    return fragment


def _positions(compact: str, needle: str, limit: int = 4) -> List[int]:
    """Позиции первых совпадений слова в тексте, приведённом к нижнему регистру."""
    positions: List[int] = []
    offset = 0
    while len(positions) < limit:
        found = locate(compact[offset:], needle)
        if found < 0:
            break
        positions.append(offset + found)
        offset += found + 1
    return positions


def document_snippets(document: Document, needles: Sequence[str]) -> List[str]:
    """Сниппеты попадания: по одному на найденное слово, не больше `MAX_SNIPPETS` штук.

    Окна не накладываются друг на друга: иначе первая вырезка («Геопозиция» в заголовке) съедала
    бы вторую, и попадание выглядело бы подтверждённым одним словом вместо двух.
    """
    compact = " ".join(document.content.split())
    if not compact:
        return []
    chosen: List[int] = []
    for needle in needles:
        for position in _positions(compact.lower(), needle):
            if all(abs(position - taken) > SNIPPET_CHARS for taken in chosen):
                chosen.append(position)
                break
    if not chosen:
        return [snippet_at(compact, -1)]
    return [snippet_at(compact, position) for position in sorted(chosen)[:MAX_SNIPPETS]]


def find_documents(query: str, index: str = "all") -> List[Tuple[Document, int, List[str]]]:
    """Найденные документы раздела: документ, число найденных слов и сами слова.

    Порядок — ступень совпадения (см. `match_document`), затем число найденных слов, затем
    совпадение в заголовке, затем число вхождений в текст, затем путь. Число слов стоит выше
    заголовка, иначе длинный вопрос («как получить координаты устройства на Авроре?») поднимал бы
    наверх статью, где из слов вопроса нашлось только «Авроре»; заголовок стоит выше числа
    вхождений, иначе слово «геопозиция» вело бы в примечания к выпуску, где оно встречается чаще,
    чем в руководстве «Геопозиция в C++ API».
    """
    section = (index or "all").strip() or "all"
    matched: List[Tuple[int, int, int, int, str, Document, List[str]]] = []
    for document in CORPUS:
        if section != "all" and document.index != section:
            continue
        found = match_document(document, query)
        if found is None:
            continue
        tier, hits, needles = found
        title = document.title.lower()
        body = document.content.lower()
        in_title = int(any(locate(title, needle) >= 0 for needle in needles))
        occurrences = sum(len(_positions(body, needle)) for needle in needles)
        matched.append(
            (tier, -hits, -in_title, -occurrences, document.path, document, needles)
        )
    matched.sort(key=lambda item: item[:5])
    return [(document, hits, needles) for _, _, hits, _, _, document, needles in matched]


def search_payload(
    query: str,
    index: str = "all",
    version: str = "latest",
    limit: int = MAX_SEARCH_LIMIT,
    empty: bool = False,
) -> Dict[str, Any]:
    """Ответ `search`: `total` считает все попадания, `results` обрезаны запрошенным лимитом."""
    resolved = resolve_version(version)
    found: List[Tuple[Document, int, List[str]]] = [] if empty else find_documents(query, index)
    size = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    results = [
        {
            "path": document.path,
            "url": document.url,
            "title": document.title,
            "index": document.index,
            "version": resolved,
            "snippets": document_snippets(document, needles),
        }
        for document, _, needles in found[:size]
    ]
    return {"query": query, "total": len(found), "results": results}


def document_payload(path: str) -> Dict[str, Any]:
    """Ответ `get_document`; неизвестный путь — `LookupError` для ошибочного результата."""
    for document in CORPUS:
        if document.path == path:
            return {"path": document.path, "title": document.title, "content": document.content}
    known = ", ".join(document.path for document in CORPUS)
    raise LookupError(f"документ не найден: {path or '<пустой путь>'}; известные пути: {known}")


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


# --- сервер ---------------------------------------------------------------------------------


def build_server(
    log_level: str = "ERROR",
    empty: bool = False,
    no_versions: bool = False,
    broken_document: bool = False,
) -> "MCPServer":
    """Сервер с инструментами портала; аргументы — режимы заглушки."""
    # Импорт внутри функции, а не сверху модуля: перезапуск в `.venv` должен случиться раньше,
    # иначе системный `python3` (без пакета `mcp`) падает на импорте до всякой логики.
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        log_level=log_level,
    )

    @server.tool()
    def get_doc_versions() -> str:
        """Список версий документации портала: актуальная помечена `latest: 1`."""
        return _json(versions_payload(no_versions=no_versions))

    @server.tool()
    def search(
        query: str,
        index: str = "all",
        version: str = "latest",
        limit: int = MAX_SEARCH_LIMIT,
    ) -> str:
        """Поиск по разделу портала: путь, заголовок, версия и сниппеты найденных документов."""
        return _json(
            search_payload(query, index=index, version=version, limit=limit, empty=empty)
        )

    @server.tool()
    def get_document(path: str, index: str = "docs") -> str:
        """Полный markdown-текст документа по пути из результата поиска."""
        if broken_document:
            raise ToolError(DOCUMENT_ERROR)
        try:
            return _json(document_payload(path))
        except LookupError as error:
            raise ToolError(str(error)) from error

    @server.tool()
    def echo(message: str) -> str:
        """Возвращает переданный текст: инструмент проверки соединения."""
        return message

    return server


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fake_docs_server.py",
        description="Заглушка сервера документации портала ОС Аврора по stdio.",
    )
    parser.add_argument(
        "--empty", action="store_true", help="поиск всегда пустой: ни одного попадания"
    )
    parser.add_argument(
        "--no-versions", action="store_true", help="пустой список версий документации"
    )
    parser.add_argument(
        "--broken-document",
        action="store_true",
        help="get_document всегда отдаёт ошибочный результат",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    reexec_in_venv()
    server = build_server(
        empty=args.empty,
        no_versions=args.no_versions,
        broken_document=args.broken_document,
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
