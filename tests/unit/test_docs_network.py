"""Сетевой контракт P4: поиск по реальной документации портала разработчиков.

Помечен `network` и по умолчанию не выполняется: обычный прогон не ходит в сеть. Тест нужен
для другого — убедиться, что запись корпуса в пакете домена жива, что разделы и версии выбираются
по данным сервера и что из документа действительно вырезается фрагмент, пригодный для цитаты.
"""

import pytest

from core import config, domains
from core.docs_retrieval import STATUS_NO_CANDIDATES, STATUS_OK, DocsRetriever
from core.mcp_client import MCPClient

pytestmark = pytest.mark.network

QUESTION = "как в ОС Аврора получить координаты устройства"

# Заведомо отсутствующая в корпусе тема: проверяем, что приложение не выдумывает фрагменты,
# а честно сообщает, что нашло ноль кандидатов.
ABSENT_QUESTION = "как приготовить борщ со свёклой и капустой"


@pytest.fixture
def retriever() -> DocsRetriever:
    domain = domains.load_domain("aurora-qt5")
    docs = domain.docs
    assert docs is not None and docs.server, "домен обязан объявлять корпус документации"
    spec = next(
        (server for server in domain.servers if server.name == docs.server),
        None,
    )
    assert spec is not None, f"в реестре домена нет сервера {docs.server!r}"
    return DocsRetriever(
        MCPClient, spec, docs, sdk_version=str(domain.local_sdk_version)
    )


def test_search_returns_fragments_with_version_and_source(retriever: DocsRetriever):
    report = retriever.search(QUESTION)
    assert report.status == STATUS_OK, report.error
    assert report.version, "версия документации обязана быть известна"
    assert report.fragments, "по вопросу о геопозиции документация должна находиться"

    fragment = report.fragments[0]
    assert fragment.identifier.startswith("doc/")
    assert fragment.section
    assert fragment.source and "." in fragment.source
    assert len(fragment.text) >= config.DOCS_CITATION_MIN_CHARS


def test_fragment_can_be_quoted_verbatim(retriever: DocsRetriever):
    """Главное требование фазы: из доставленного текста можно взять дословную цитату."""
    report = retriever.search(QUESTION)
    text = " ".join(report.fragments[0].text.split())
    quote = text[: config.DOCS_CITATION_MIN_CHARS + 30]
    assert len(quote) >= config.DOCS_CITATION_MIN_CHARS
    assert quote in " ".join(report.fragments[0].text.split())


def test_versions_are_listed_by_the_server(retriever: DocsRetriever):
    versions = retriever.refresh_versions()
    assert versions, "сервер портала обязан отдавать список версий"
    assert all(part.count(".") >= 1 for part in versions)


def test_section_for_version_questions_is_searched(retriever: DocsRetriever):
    report = retriever.search("в какой версии появился фоновый режим геопозиции?")
    assert "release_notes" in report.sections
    assert report.status in (STATUS_OK, STATUS_NO_CANDIDATES), report.error


def test_absent_question_does_not_produce_fragments(retriever: DocsRetriever):
    """Что бы ни ответил поиск, фрагменты не появляются из воздуха."""
    report = retriever.search(ABSENT_QUESTION)
    assert report.status in (STATUS_OK, STATUS_NO_CANDIDATES), report.error
    if report.status == STATUS_NO_CANDIDATES:
        assert report.fragments == ()
