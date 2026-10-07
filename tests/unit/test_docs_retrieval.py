"""Поиск по документации портала: разделы, версия, фрагменты и состояния снимка.

Тесты идут на заглушке-клиенте (процессы не поднимаются) и на временном пакете домена: разделы,
слова-признаки версий и имена инструментов приходят из данных, поэтому в тестах они нарочно не
такие, как у установленного пакета, — так проверяется, что ядро ничего своего об этом не знает.
"""

import json
from pathlib import Path

import pytest

from core import config
from core.docs_retrieval import (
    STATUS_NO_CANDIDATES,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    DocsRetriever,
)
from core.domains import load_domain
from core.mcp_registry import MCPServerSpec

DOCS = {
    "server": "probe-docs",
    "source": "docs.example.invalid",
    "default_section": "guide",
    "version_sections": ["changes"],
    "version_keywords": ["версия", "появил"],
    "tools": {"versions": "tool_versions", "search": "tool_search", "document": "tool_text"},
}


# --- заглушка клиента и пакет домена -------------------------------------------------------


class StubResult:
    """Результат вызова инструмента: те же поля, что у настоящего `MCPCallResult`."""

    def __init__(self, tool: str, text: str = "", is_error: bool = False, arguments=None) -> None:
        self.server = "probe-docs"
        self.tool = tool
        self.arguments = dict(arguments or {})
        self.text = text
        self.is_error = is_error


class StubClient:
    """Клиент сервера документации без процессов: отвечает по имени инструмента."""

    def __init__(self, answers=None, *, failing=(), refusing=(), silent=()) -> None:
        self.answers = dict(answers or {})
        self.failing = set(failing)
        self.refusing = set(refusing)
        self.silent = set(silent)
        self.calls = []

    def session(self):
        """Соединение на время операции: у заглушки его нет, но контракт клиента соблюдён.

        Настоящий клиент держит на это время серверный процесс; здесь достаточно пустого
        контекста, потому что все вызовы и так идут в память.
        """
        import contextlib

        return contextlib.nullcontext(self)

    def call_tool(self, tool, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((tool, arguments))
        if tool in self.failing:
            raise RuntimeError("сервер недоступен")
        if tool in self.refusing:
            return StubResult(tool, text="отказ инструмента", is_error=True, arguments=arguments)
        if tool in self.silent:
            return StubResult(tool, text="", arguments=arguments)
        answer = self.answers.get(tool, "")
        if callable(answer):
            answer = answer(arguments)
        return StubResult(tool, text=answer, arguments=arguments)

    def calls_of(self, tool):
        """Аргументы всех вызовов одного инструмента — в порядке обращений."""
        return [arguments for name, arguments in self.calls if name == tool]

    @property
    def tools(self):
        return [name for name, _ in self.calls]


class StubFactory:
    """Фабрика клиента: запоминает записи реестра, для которых её позвали."""

    def __init__(self, client) -> None:
        self.client = client
        self.specs = []

    def __call__(self, spec):
        self.specs.append(spec)
        return self.client


def _pack(root: Path, docs=None) -> Path:
    """Пакет домена во временном каталоге: минимум, который требует схема загрузчика."""
    path = root / "probe"
    (path / "prompts").mkdir(parents=True, exist_ok=True)
    (path / "domain.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "id": "probe",
                "title": "Пробный домен",
                "platform": "Платформа 1.0",
                "docs_version": "9.9",
                "local_sdk_version": "9.1",
                "markers": [{"glob": "*.probe"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "invariants.json").write_text(
        json.dumps(
            {
                "invariants": [
                    {
                        "number": 1,
                        "rule": "Правило домена",
                        "source": "документ домена",
                        "forbidden": ["запретное слово"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "prompts" / "system.md").write_text("Роль домена.", encoding="utf-8")
    (path / "prompts" / "refusal.md").write_text("Текст отказа.", encoding="utf-8")
    if docs is not None:
        (path / "docs.json").write_text(
            json.dumps(docs, ensure_ascii=False), encoding="utf-8"
        )
    return path


@pytest.fixture
def domain(tmp_path: Path):
    _pack(tmp_path, DOCS)
    return load_domain("probe", tmp_path)


@pytest.fixture
def spec() -> MCPServerSpec:
    return MCPServerSpec(
        name="probe-docs", transport="http", url="https://docs.example.invalid/api/mcp"
    )


def retriever(domain, spec, client, **kwargs) -> DocsRetriever:
    return DocsRetriever(StubFactory(client), spec, domain.docs, **kwargs)


# --- ответы инструментов -------------------------------------------------------------------


def versions_text(*entries) -> str:
    return json.dumps({"versions": list(entries)})


def entry(version: str, latest=0) -> dict:
    return {"version": version, "latest": latest}


def hit(path: str, *, title="Документ", section="guide", snippets=("сниппет",), url="") -> dict:
    return {
        "path": path,
        "url": url or f"https://docs.example.invalid/{path}",
        "title": title,
        "index": section,
        "version": "9.9",
        "snippets": list(snippets),
    }


def search_text(*items) -> str:
    return json.dumps({"query": "вопрос", "total": len(items), "results": list(items)})


def by_section(mapping: dict):
    """Ответ поиска, зависящий от раздела: инструмент у заглушки один, раздел — аргумент."""

    def answer(arguments):
        return mapping.get(arguments.get("index"), search_text())

    return answer


def document_text(path: str, content: str, title: str = "Документ") -> str:
    return json.dumps({"path": path, "title": title, "content": content})


# --- удачный поиск -------------------------------------------------------------------------


def test_search_returns_candidates_and_fragments(domain, spec):
    question = "Как получить координаты устройства?"
    client = StubClient(
        {
            "tool_versions": versions_text(entry("1.0"), entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one", snippets=("первый сниппет",))),
            "tool_text": document_text("guide/one", "Текст документа про координаты устройства."),
        }
    )

    report = retriever(domain, spec, client).search(question)

    assert report.status == STATUS_OK and report.ok is True
    assert report.query == question
    assert report.sections == ("guide",)
    assert report.version == "2.0"
    assert report.error == ""
    assert [candidate.path for candidate in report.candidates] == ["guide/one"]
    candidate = report.candidates[0]
    assert candidate.section == "guide"
    assert candidate.title == "Документ"
    assert candidate.url == "https://docs.example.invalid/guide/one"
    assert candidate.snippet == "первый сниппет"
    (fragment,) = report.fragments
    assert fragment.identifier == "guide/one"
    assert fragment.source == "docs.example.invalid"
    assert fragment.title == "Документ"
    assert fragment.section == "guide"
    assert fragment.version == "2.0"
    assert fragment.text == "Текст документа про координаты устройства."
    assert fragment.truncated is False
    # Инструменты вызываются по именам из данных домена и с границами из конфигурации.
    assert client.tools == ["tool_versions", "tool_search", "tool_text"]
    assert client.calls_of("tool_versions") == [{}]
    assert client.calls_of("tool_search") == [
        {
            "query": question,
            "index": "guide",
            "limit": config.DOCS_SEARCH_LIMIT,
            "version": "2.0",
        }
    ]
    assert client.calls_of("tool_text") == [{"path": "guide/one", "index": "guide"}]


def test_document_title_from_the_payload_wins(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one", title="Заголовок из поиска")),
            "tool_text": document_text("guide/one", "Текст.", title="Заголовок документа"),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.fragments[0].title == "Заголовок документа"


# --- разделы -------------------------------------------------------------------------------


def test_version_question_searches_the_release_notes_too(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": by_section(
                {
                    "guide": search_text(hit("guide/one")),
                    "changes": search_text(hit("changes/two", section="changes")),
                }
            ),
            "tool_text": lambda arguments: document_text(
                arguments["path"], f"Текст документа {arguments['path']} про координаты."
            ),
        }
    )

    report = retriever(domain, spec, client).search("В какой версии появилась поддержка?")

    assert report.sections == ("guide", "changes")
    assert [candidate.path for candidate in report.candidates] == ["guide/one", "changes/two"]
    assert [candidate.section for candidate in report.candidates] == ["guide", "changes"]
    assert [arguments["index"] for arguments in client.calls_of("tool_search")] == [
        "guide",
        "changes",
    ]


def test_plain_question_searches_only_the_default_section(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты устройства?")

    assert report.sections == ("guide",)
    assert [arguments["index"] for arguments in client.calls_of("tool_search")] == ["guide"]


def test_sections_follow_the_domain_data(tmp_path, spec):
    """Разделы и слова-признаки не зашиты в код: другой пакет даёт другие разделы, и повтор
    раздела версий не удваивается."""
    other = {
        "server": "probe-docs",
        "source": "docs.example.invalid",
        "default_section": "manual",
        "version_sections": ["notes", "manual"],
        "version_keywords": ["выпуск"],
        "tools": {"versions": "tool_versions", "search": "tool_search", "document": "tool_text"},
    }
    _pack(tmp_path, other)
    local = load_domain("probe", tmp_path)
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(),
            "tool_text": document_text("manual/one", "Текст."),
        }
    )

    report = retriever(local, spec, client).search("Что нового в выпуске?")

    assert report.sections == ("manual", "notes")


# --- версия --------------------------------------------------------------------------------


def test_latest_flag_wins_over_the_highest_number(domain, spec):
    """Актуальная — помеченная сервером, а не самая старшая: на портале это разные версии."""
    client = StubClient(
        {
            "tool_versions": versions_text(entry("5.1.5"), entry("5.2.1", 1), entry("5.2.2")),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты устройства?")

    assert report.version == "5.2.1"
    assert client.calls_of("tool_search")[0]["version"] == "5.2.1"
    assert report.fragments[0].version == "5.2.1"


def test_explicit_version_is_used_as_is(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client, version="4.2").search("Как получить координаты?")

    assert report.version == "4.2"
    assert client.calls_of("tool_search")[0]["version"] == "4.2"
    # Заданная версия избавляет от лишнего обращения: список версий не спрашивают.
    assert client.calls_of("tool_versions") == []


def test_empty_version_list_searches_without_a_version(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_OK
    assert report.version == ""
    assert "version" not in client.calls_of("tool_search")[0]


def test_version_list_without_a_flag_falls_back_to_the_first(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("7.7"), entry("7.9")),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.version == "7.7"


def test_version_entries_are_parsed_defensively(domain, spec):
    """Сервер отдаёт версии объектами с числовым признаком; строки тоже принимаются."""
    client = StubClient(
        {
            "tool_versions": versions_text("1.0", {"version": "2.0", "latest": "1"}, {"latest": 1}),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.version == "2.0"


def test_versions_are_cached_and_refresh_resets_them(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("1.0"), entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )
    finder = retriever(domain, spec, client)

    finder.search("Первый вопрос про координаты?")
    finder.search("Второй вопрос про координаты?")

    assert len(client.calls_of("tool_versions")) == 1
    assert finder.versions == ("1.0", "2.0")

    assert finder.refresh_versions() == ("1.0", "2.0")
    assert len(client.calls_of("tool_versions")) == 2
    assert finder.versions == ("1.0", "2.0")
    assert len(client.calls_of("tool_versions")) == 2


# --- фрагменты -----------------------------------------------------------------------------


def test_fragment_is_cut_around_the_match_and_marked(domain, spec):
    content = "П" * 3000 + "координаты" + "К" * 3000
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", content),
        }
    )

    report = retriever(domain, spec, client, fragment_chars=200).search(
        "Как получить координаты устройства?"
    )

    (fragment,) = report.fragments
    assert fragment.truncated is True
    assert len(fragment.text) == 200
    assert fragment.text in content
    assert "координаты" in fragment.text
    # Это окно вокруг совпадения, а не начало документа.
    assert fragment.text != content[:200]


def test_whole_document_fits_and_is_not_marked_truncated(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", "Короткий документ про координаты."),
        }
    )

    report = retriever(domain, spec, client, fragment_chars=200).search(
        "Как получить координаты?"
    )

    assert report.fragments[0].truncated is False
    assert report.fragments[0].text == "Короткий документ про координаты."


def test_fragment_without_a_matching_word_starts_at_the_head(domain, spec):
    content = "Н" * 3000
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one")),
            "tool_text": document_text("guide/one", content),
        }
    )

    report = retriever(domain, spec, client, fragment_chars=120).search("как и что")

    (fragment,) = report.fragments
    assert fragment.text == content[:120]
    assert fragment.truncated is True


def test_fragments_are_bounded_by_the_limit(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(
                hit("guide/one"), hit("guide/two"), hit("guide/three")
            ),
            "tool_text": lambda arguments: document_text(
                arguments["path"], f"Текст документа {arguments['path']} про координаты."
            ),
        }
    )

    report = retriever(domain, spec, client, max_fragments=1).search(
        "Как получить координаты?"
    )

    assert len(report.candidates) == 3
    assert [fragment.identifier for fragment in report.fragments] == ["guide/one"]
    assert len(client.calls_of("tool_text")) == 1


def test_snippets_are_joined_and_clipped(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(
                hit("guide/one", snippets=("первый сниппет", "второй сниппет"))
            ),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )
    joined = "первый сниппет\nвторой сниппет"

    report = retriever(domain, spec, client, snippet_chars=20).search("Как получить координаты?")

    assert report.candidates[0].snippet == joined[:20]
    assert len(report.candidates[0].snippet) == 20


def test_short_snippets_are_kept_as_is(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(
                hit("guide/one", snippets=("первый сниппет", "второй сниппет"))
            ),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client, snippet_chars=200).search("Как получить координаты?")

    assert report.candidates[0].snippet == "первый сниппет\nвторой сниппет"


def test_candidates_are_merged_by_path(domain, spec):
    """Один документ в двух разделах — один кандидат с разделом первого попадания."""
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": by_section(
                {
                    "guide": search_text(hit("guide/one", snippets=("из документации",))),
                    "changes": search_text(
                        hit("guide/one", section="changes", snippets=("из примечаний",))
                    ),
                }
            ),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("В какой версии появилось?")

    assert [candidate.path for candidate in report.candidates] == ["guide/one"]
    assert report.candidates[0].section == "guide"
    assert report.candidates[0].snippet == "из документации\nиз примечаний"


def test_fragments_from_snippets_when_documents_are_not_fetched(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one", snippets=("сниппет с фактом",))),
            "tool_text": document_text("guide/one", "Полный текст документа."),
        }
    )

    report = retriever(domain, spec, client, fetch_documents=False).search(
        "Как получить координаты?"
    )

    assert client.calls_of("tool_text") == []
    assert report.fragments[0].text == "сниппет с фактом"
    assert report.fragments[0].truncated is False


@pytest.mark.parametrize("mode", ["failing", "refusing", "silent"])
def test_broken_document_falls_back_to_snippets(domain, spec, mode):
    """Полный текст не отдался — фрагментом становятся сниппеты, кандидат остаётся."""
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(hit("guide/one", snippets=("сниппет с фактом",))),
            "tool_text": document_text("guide/one", "Полный текст документа."),
        },
        **{mode: {"tool_text"}},
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_OK
    assert [candidate.path for candidate in report.candidates] == ["guide/one"]
    (fragment,) = report.fragments
    assert fragment.identifier == "guide/one"
    assert fragment.text == "сниппет с фактом"
    assert fragment.truncated is False


def test_fragment_is_dropped_when_there_is_nothing_to_quote(domain, spec):
    """Пустой фрагмент не доставляется: цитировать в нём нечего, кандидат при этом остаётся."""
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text({"path": "guide/one", "index": "guide", "snippets": []}),
            "tool_text": document_text("guide/one", ""),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_OK
    assert [candidate.path for candidate in report.candidates] == ["guide/one"]
    assert report.fragments == ()


# --- пустой поиск и сбои -------------------------------------------------------------------


def test_empty_search_gives_no_candidates(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_NO_CANDIDATES
    assert report.ok is False
    assert report.candidates == ()
    assert report.fragments == ()
    assert report.error == ""
    assert report.version == "2.0"
    assert client.calls_of("tool_text") == []


def test_hits_without_a_path_are_skipped(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text({"title": "Без пути", "index": "guide"}),
            "tool_text": document_text("guide/one", "Текст документа."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_NO_CANDIDATES


def test_server_failure_is_a_state_with_a_reason(domain, spec):
    client = StubClient({"tool_versions": versions_text(entry("2.0", 1))}, failing={"tool_search"})

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_UNAVAILABLE
    assert report.ok is False
    assert "RuntimeError: сервер недоступен" in report.error
    assert "tool_search" in report.error
    assert report.candidates == ()
    assert report.fragments == ()


def test_versions_failure_is_unavailable(domain, spec):
    client = StubClient({}, failing={"tool_versions"})

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_UNAVAILABLE
    assert "сервер недоступен" in report.error
    assert report.fragments == ()


def test_tool_refusal_is_unavailable(domain, spec):
    client = StubClient({"tool_versions": versions_text(entry("2.0", 1))}, refusing={"tool_search"})

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_UNAVAILABLE
    assert "отказ инструмента" in report.error


def test_unparsable_answer_is_unavailable(domain, spec):
    client = StubClient(
        {"tool_versions": versions_text(entry("2.0", 1)), "tool_search": "не json вовсе"}
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_UNAVAILABLE
    assert "JSON" in report.error


def test_json_inside_prose_is_parsed(domain, spec):
    client = StubClient(
        {
            "tool_versions": "Вот список версий:\n" + versions_text(entry("2.0", 1)) + "\nконец",
            "tool_search": "Результаты поиска:\n"
            + search_text(hit("guide/one"))
            + "\nконец списка",
            "tool_text": document_text("guide/one", "Текст документа про координаты."),
        }
    )

    report = retriever(domain, spec, client).search("Как получить координаты?")

    assert report.status == STATUS_OK
    assert report.version == "2.0"
    assert [candidate.path for candidate in report.candidates] == ["guide/one"]


def test_connect_failure_is_unavailable(domain, spec):
    class BrokenFactory:
        def __call__(self, spec):
            raise ConnectionError("порт закрыт")

    report = DocsRetriever(BrokenFactory(), spec, domain.docs).search("Как получить координаты?")

    assert report.status == STATUS_UNAVAILABLE
    assert "порт закрыт" in report.error
    assert report.fragments == ()


def test_report_carries_the_sdk_version(domain, spec):
    client = StubClient(
        {
            "tool_versions": versions_text(entry("2.0", 1)),
            "tool_search": search_text(),
        }
    )

    report = retriever(domain, spec, client, sdk_version="9.1.2-rc1").search("Вопрос про координаты?")

    assert report.sdk_version == "9.1.2-rc1"


def test_versions_property_is_quiet_on_failure(domain, spec):
    client = StubClient({}, failing={"tool_versions"})

    finder = retriever(domain, spec, client)

    assert finder.versions == ()
    # Сбой не кэшируется: следующий поиск снова спросит список версий.
    assert finder.versions == ()
    assert len(client.calls_of("tool_versions")) == 2


# --- вторая ступень отбора: оценка кандидатов ----------------------------------------------

from core import reranking  # noqa: E402

RERANK_TOOLS = {"versions": "tool_versions", "search": "tool_search", "document": "tool_document"}


def _rated_client(*, scores, documents=True) -> StubClient:
    """Заглушка с поиском, полным текстом и списком версий — минимальный набор для второй ступени."""
    answers = {
        "tool_versions": versions_text(entry("5.2.1", latest=1)),
        "tool_search": search_text(
            hit("guide/one", title="Первый"), hit("guide/two", title="Второй")
        ),
    }
    if documents:
        answers["tool_document"] = lambda arguments: document_text(
            arguments["path"], f"Текст документа {arguments['path']} про координаты и сборку."
        )
    client = StubClient(answers)
    client.scores = scores
    return client


def rating_json(*pairs) -> str:
    return json.dumps(
        {"results": [{"id": i, "score": s, "reason": f"оценка {i}"} for i, s in pairs]},
        ensure_ascii=False,
    )


def _rate_with(scores):
    def rate(question, candidates):
        return rating_json(*scores)

    return rate


def test_rate_keeps_only_candidates_above_threshold(domain, spec):
    client = _rated_client(scores=(0.9, 0.1))
    report = retriever(domain, spec, client).search(
        "как получить координаты?", rate=_rate_with([(1, 0.9), (2, 0.1)]), threshold=0.6
    )
    assert report.status == "ok"
    assert [fragment.identifier for fragment in report.fragments] == ["guide/one"]
    assert [candidate.score for candidate in report.candidates] == [0.9, 0.1]
    assert report.rated is True and report.threshold == 0.6
    assert client.calls_of("tool_text") == [{"path": "guide/one", "index": "guide"}], (
        "полный текст запрашивается только для прошедших порог"
    )


def test_rate_can_drop_everything(domain, spec):
    client = _rated_client(scores=(0.1, 0.1))
    report = retriever(domain, spec, client).search(
        "как получить координаты?", rate=_rate_with([(1, 0.1), (2, 0.1)]), threshold=0.6
    )
    assert report.status == "no_matches"
    assert report.fragments == ()
    assert len(report.candidates) == 2, "снимок сохраняет кандидатов с их оценками"
    assert client.calls_of("tool_text") == [], "без прошедших порог ничего не выгружается"


def test_rate_failure_is_a_visible_state(domain, spec):
    client = _rated_client(scores=(0.9, 0.9))

    def broken(question, candidates):
        return "не JSON вовсе"

    report = retriever(domain, spec, client).search(
        "как получить координаты?", rate=broken, threshold=0.6
    )
    assert report.status == "rerank_failed"
    assert report.fragments == ()
    assert "не разобран" in report.error
    assert client.calls_of("tool_text") == []


def test_rate_exception_is_a_visible_state(domain, spec):
    def exploding(question, candidates):
        raise RuntimeError("модель недоступна")

    client = _rated_client(scores=(0.9, 0.9))
    report = retriever(domain, spec, client).search(
        "как получить координаты?", rate=exploding, threshold=0.6
    )
    assert report.status == "rerank_failed"
    assert "модель недоступна" in report.error


def test_without_rate_everything_is_delivered_as_before(domain, spec):
    client = _rated_client(scores=(0.9, 0.9))
    report = retriever(domain, spec, client).search("как получить координаты?")
    assert report.status == "ok"
    assert len(report.fragments) == 2
    assert report.rated is False and report.threshold == 0.0
    assert [candidate.score for candidate in report.candidates] == [0.0, 0.0]
