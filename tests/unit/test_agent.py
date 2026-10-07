"""Агент: шаг поиска по документации, контекст фрагментов и проверка ссылок.

Проверки идут против заглушек: клиент записывает запросы, ретривер отдаёт заранее собранный
снимок поиска. Ни сети, ни процессов серверов здесь нет — это дело сквозного слоя.
"""

from pathlib import Path

import pytest
from tests.conftest import FakeCandidate, FakeDocsReport, FakeDocsRetriever, FakeFragment

from core import citations, config, prompts
from core.agent import RepoAgent, RequestPhase
from core.answer_settings import AnswerFormat
from core.api_client import AnswerMeta
from core.domains import load_domain
from core.history_manager import HistoryManager
from tests.conftest import rerank_answer


class FakeClient:
    """Клиент модели: отдаёт ответы по очереди и помнит каждый запрос."""

    def __init__(self, *answers: str, rerank_score: float = 0.9) -> None:
        self.answers = list(answers) or ["Ответ без ссылок."]
        self.calls = []
        self.rerank_score = rerank_score

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(list(messages))
        rated = rerank_answer(messages, self.rerank_score)
        if rated is not None:
            return AnswerMeta(
                content=rated,
                model=model,
                elapsed_seconds=0.1,
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
                cost_usd=0.00001,
            )
        content = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return AnswerMeta(
            content=content,
            model=model,
            elapsed_seconds=1.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    return tmp_path / "repo"


def _agent(repo: Path, client, retriever=None, history_path: Path = None) -> RepoAgent:
    return RepoAgent(
        domain=load_domain("aurora-qt5"),
        client=client,
        root=repo,
        history=HistoryManager(path=history_path or (repo / "history.json")),
        docs_retriever=retriever,
    )


def answer_calls(client) -> list:
    """Запросы ответа, без служебного запроса оценки: тесты про путь ответа считают именно их."""
    return [call for call in client.calls if "оцениваешь" not in call[0]["content"]]


def _report(docs_fragment=None, status: str = "ok", error: str = "") -> FakeDocsReport:
    """Снимок поиска: фрагмент и его кандидат — как в настоящем отчёте, где фрагменты родом из поиска."""
    fragments = (docs_fragment,) if docs_fragment else ()
    candidates = (
        (FakeCandidate(path=docs_fragment.identifier, title="Документ"),) if docs_fragment else ()
    )
    return FakeDocsReport(
        query="вопрос", status=status, fragments=fragments, candidates=candidates, error=error
    )


CONFIRMED_ANSWER = (
    "Координаты даёт модуль Qt Positioning.\n"
    "Цитата: «Информацию о местоположении предоставляет служба Geoclue»\n"
    "Источник: doc/software_development/guides/cpp_api/positioning"
)


def test_fragments_go_into_the_request_above_the_dialogue(repo: Path, docs_fragment):
    client = FakeClient(CONFIRMED_ANSWER)
    retriever = FakeDocsRetriever(_report(docs_fragment))
    agent = _agent(repo, client, retriever)
    agent.ask("как получить координаты устройства?")

    messages = answer_calls(client)[0]
    contents = [message["content"] for message in messages]
    docs_index = next(i for i, text in enumerate(contents) if docs_fragment.text in text)
    instruction_index = next(
        i for i, text in enumerate(contents) if text == citations.citations_message()
    )
    assert docs_index < instruction_index
    assert messages[-1]["role"] == "user"
    assert docs_index < len(messages) - 1
    assert retriever.queries == [("как получить координаты устройства?", "")]


def test_confirmed_answer_is_accepted_without_a_retry(repo: Path, docs_fragment):
    client = FakeClient(CONFIRMED_ANSWER)
    agent = _agent(repo, client, FakeDocsRetriever(_report(docs_fragment)))
    meta = agent.ask("как получить координаты устройства?")
    assert len(answer_calls(client)) == 1
    assert meta.content == CONFIRMED_ANSWER
    assert agent.last_citations is not None and agent.last_citations.confirmed
    assert agent.history.dialogues[-1]["answer"] == CONFIRMED_ANSWER


def test_unconfirmed_answer_is_retried_then_replaced(repo: Path, docs_fragment):
    client = FakeClient("Координаты берутся из Qt Positioning, модуль должен быть подключён.")
    agent = _agent(repo, client, FakeDocsRetriever(_report(docs_fragment)))
    meta = agent.ask("как получить координаты устройства?")
    assert len(answer_calls(client)) == 2, "нарушение стоит одного повтора"
    assert meta.content == citations.disclaimer_text()
    check = agent.last_citations
    assert check is not None and check.replaced and check.retried
    assert check.violations
    assert agent.history.dialogues[-1]["answer"] == citations.disclaimer_text()
    assert agent.exchanges == 1, "в лог идёт только показанный текст"


def test_retry_can_confirm_the_answer(repo: Path, docs_fragment):
    client = FakeClient("Без ссылки.", CONFIRMED_ANSWER)
    agent = _agent(repo, client, FakeDocsRetriever(_report(docs_fragment)))
    meta = agent.ask("как получить координаты устройства?")
    assert len(answer_calls(client)) == 2
    assert meta.content == CONFIRMED_ANSWER
    assert agent.last_citations is not None and agent.last_citations.retried
    assert not agent.last_citations.replaced


@pytest.mark.parametrize("fmt", [AnswerFormat.JSON, AnswerFormat.COMPACT, AnswerFormat.PATCH])
def test_other_formats_keep_their_own_contract(repo: Path, docs_fragment, fmt):
    client = FakeClient("Ответ в своём формате.")
    agent = _agent(repo, client, FakeDocsRetriever(_report(docs_fragment)))
    agent.config.settings = agent.config.settings.with_format(fmt)
    meta = agent.ask("как получить координаты устройства?")
    assert len(answer_calls(client)) == 1, "у формата со своим контрактом повтора быть не должно"
    assert meta.content == "Ответ в своём формате."
    assert agent.last_citations is None
    contents = [message["content"] for message in answer_calls(client)[0]]
    assert citations.citations_message() not in contents
    assert any(docs_fragment.text in text for text in contents), "фрагменты всё равно доставляются"


def test_disabled_mode_does_not_search_or_inject(repo: Path, docs_fragment):
    client = FakeClient("Ответ.")
    retriever = FakeDocsRetriever(_report(docs_fragment))
    agent = _agent(repo, client, retriever)
    agent.config.docs_enabled = False
    agent.ask("вопрос")
    assert retriever.queries == []
    assert len(client.calls[0]) == 2, "без поиска в запросе только система и вопрос"
    assert agent.docs_report() is None


def test_unavailable_server_tells_the_model_and_keeps_going(repo: Path):
    client = FakeClient("Документация недоступна, повторите запрос.")
    agent = _agent(repo, client, FakeDocsRetriever(_report(status="unavailable", error="нет сети")))
    meta = agent.ask("как получить координаты?")
    assert meta.content == "Документация недоступна, повторите запрос."
    contents = [message["content"] for message in answer_calls(client)[0]]
    assert any("недоступна" in text and "нет сети" in text for text in contents)
    assert agent.last_citations is None, "без фрагментов проверка не выполняется"


def test_empty_search_tells_the_model_not_to_claim_platform_facts(repo: Path):
    client = FakeClient("В документации портала ответа нет.")
    agent = _agent(repo, client, FakeDocsRetriever(_report(status="no_candidates")))
    agent.ask("что такое несуществующая функция?")
    contents = " ".join(message["content"] for message in answer_calls(client)[0])
    assert "ничего не нашлось" in contents
    assert citations.citations_message() not in contents


def test_docs_report_is_replaced_by_the_next_question_and_survives_clear(repo: Path, docs_fragment):
    client = FakeClient(CONFIRMED_ANSWER)
    retriever = FakeDocsRetriever(_report(docs_fragment))
    agent = _agent(repo, client, retriever)
    agent.ask("первый вопрос")
    first = agent.docs_report()
    assert first is not None and first.query == "вопрос"

    retriever.report = _report(status="unavailable", error="позже упал")
    agent.ask("второй вопрос")
    assert agent.docs_report().status == "unavailable"

    agent.reset()
    assert agent.docs_report() is not None, "снимок поиска — результат уже сделанной работы"
    assert agent.last_citations is None, "снимок проверки относится к последнему ответу"


def test_search_itself_costs_no_model_request(repo: Path, docs_fragment):
    """Поиск — это инструменты MCP, а не модель: в baseline ровно один запрос на вопрос.

    В enhanced добавляется один вспомогательный запрос — оценка кандидатов; это цена второй
    ступени, и она проверяется отдельно.
    """
    client = FakeClient(CONFIRMED_ANSWER)
    agent = _agent(repo, client, FakeDocsRetriever(_report(docs_fragment)))
    agent.config.docs_retrieval = "baseline"
    agent.ask("вопрос")
    assert len(client.calls) == 1
    assert agent.session_usage.requests == 1


def test_enhanced_adds_exactly_one_rerank_request(repo: Path, docs_fragment):
    client = FakeClient(CONFIRMED_ANSWER)
    retriever = FakeDocsRetriever(_report(docs_fragment))
    agent = _agent(repo, client, retriever)
    assert agent.config.docs_retrieval == "enhanced"
    agent.ask("вопрос")
    assert len(retriever.rated) == 1, "оценка запрашивается один раз"
    assert len(client.calls) == 2, "оценка и ответ"
    assert agent.session_usage.requests == 2, "расход считает оба запроса"
    rerank_messages = client.calls[0]
    assert config.DOCS_RERANK_MAX_WORDS > 0
    assert rerank_messages[0]["role"] == "system"
    assert "оцениваешь" in rerank_messages[0]["content"]


def test_phase_events_cover_the_retry(repo: Path, docs_fragment):
    client = FakeClient("Без ссылки.")
    agent = _agent(repo, client, FakeDocsRetriever(_report(docs_fragment)))
    phases = []
    agent.ask("вопрос", on_phase=phases.append)
    assert phases.count(RequestPhase.REQUEST) == 2


def test_note_mentions_version_mismatch(repo: Path, docs_fragment):
    client = FakeClient(CONFIRMED_ANSWER)
    report = FakeDocsReport(
        query="вопрос", fragments=(docs_fragment,), version="5.2.1", sdk_version="5.1.5"
    )
    agent = _agent(repo, client, FakeDocsRetriever(report))
    agent.ask("вопрос")
    contents = " ".join(message["content"] for message in answer_calls(client)[0])
    assert "версия документации: 5.2.1" in contents
    assert "5.1.5" in contents
