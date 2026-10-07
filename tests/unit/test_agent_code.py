"""Агент и корпус кода: фрагменты в запросе, проверка цитат, состояния поиска.

Заглушки: клиент модели отдаёт ответы по очереди, поиск работает по настоящему индексу,
собранному во временном каталоге. Ни сети, ни серверных процессов здесь нет.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import FakeDocsReport

from core import code_index, code_retrieval, config, domains
from core.agent import RepoAgent
from core.answer_settings import AnswerFormat
from core.api_client import AnswerMeta
from core.domains import load_domain
from core.history_manager import HistoryManager

CORPUS = domains.load_domain("aurora-qt5").corpus
CODE = "src/models.cpp"
# Дословная цитата кода длиннее минимума для корпуса кода (40 символов после нормализации).
QUOTE = 'const QString path = settings_.value("models.ini");' 
CODE_TEXT = "\n".join(
    [
        "class ModelList : public QObject {",
        "public:",
        "    void load();",
        "};",
        "",
        "void ModelList::load() {",
        '    const QString path = settings_.value("models.ini");',
        "    model_ = new QAbstractListModel(this);",
        "}",
    ]
)


class FakeClient:
    """Клиент модели: помнит запросы и отдаёт ответы по очереди."""

    def __init__(self, *answers: str, score: float = 0.9) -> None:
        self.answers = list(answers) or ["Ответ без ссылок."]
        self.calls = []
        self.score = score

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(list(messages))
        system = messages[0]["content"]
        content = None
        if "оцениваешь" in system:
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            content = json.dumps(
                {
                    "results": [
                        {"id": item["id"], "score": self.score, "reason": "тест"}
                        for item in payload["candidates"]
                    ]
                },
                ensure_ascii=False,
            )
        elif "превращаешь вопрос" in system:
            content = "ModelList load models.ini"
        if content is None:
            content = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return AnswerMeta(
            content=content,
            model=model,
            elapsed_seconds=0.5,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
        )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "src" / "models.cpp").write_text(CODE_TEXT, encoding="utf-8")
    return tmp_path / "repo"


@pytest.fixture
def index_file(tmp_path: Path, repository: Path) -> Path:
    database = tmp_path / "cache" / "index.sqlite3"
    code_index.build_index(repository, CORPUS, code_index.STRATEGY_STRUCTURAL, database)
    return database


@pytest.fixture
def retriever(index_file: Path) -> code_retrieval.CodeRetriever:
    return code_retrieval.CodeRetriever(index_file, CORPUS)


def _agent(repository: Path, client, retriever, tmp_path: Path) -> RepoAgent:
    return RepoAgent(
        domain=load_domain("aurora-qt5"),
        client=client,
        root=repository,
        history=HistoryManager(path=tmp_path / "history.json"),
        code_retriever=retriever,
    )


def _fragment_text(agent: RepoAgent) -> str:
    return "\n".join(message["content"] for message in agent._build_messages("вопрос"))


def test_code_fragments_go_into_the_request(repository: Path, retriever, tmp_path: Path):
    quote = 'const QString path = settings_.value("models.ini");'
    client = FakeClient(f"Загрузка описана в {CODE}:L1-L9.\n\nЦитата: {quote}")
    agent = _agent(repository, client, retriever, tmp_path)
    meta = agent.ask("где инициализируется модель списка?")

    request = client.calls[-1]
    system_messages = [message["content"] for message in request if message["role"] == "system"]
    fragments = next(text for text in system_messages if CODE in text and "L" in text)
    assert "фрагменты, найденные по запросу" in fragments
    assert CODE_TEXT.splitlines()[0] in fragments, "текст фрагмента дословный"
    assert meta.content.startswith("Загрузка описана")


def test_citations_instruction_is_added_once(repository: Path, retriever, tmp_path: Path):
    quote = 'const QString path = settings_.value("models.ini");'
    client = FakeClient(f"{CODE}:L1-L9 — инициализация.\n\nЦитата: {quote}")
    agent = _agent(repository, client, retriever, tmp_path)
    agent.ask("где инициализируется модель списка?")
    instructions = [
        message["content"]
        for message in client.calls[-1]
        if message["role"] == "system" and "Подтверди это прямо в ответе" in message["content"]
    ]
    assert len(instructions) == 1


def test_unconfirmed_code_answer_is_retried_then_replaced(repository: Path, retriever, tmp_path: Path):
    client = FakeClient("Ответ без ссылки и без цитаты.", "И снова без ссылки.")
    agent = _agent(repository, client, retriever, tmp_path)
    meta = agent.ask("где инициализируется модель списка?")

    assert meta.content == citations_disclaimer()
    assert agent.last_citations.replaced is True
    assert agent.last_citations.retried is True


def citations_disclaimer() -> str:
    from core import citations

    return citations.disclaimer_text()


def test_confirmed_code_answer_is_accepted_without_a_retry(repository: Path, retriever, tmp_path: Path):
    answer = f"Смотри {CODE}:L1-L9.\n\nЦитата: {QUOTE}"
    client = FakeClient(answer)
    agent = _agent(repository, client, retriever, tmp_path)
    meta = agent.ask("где инициализируется модель списка?")

    assert meta.content == answer
    answer_requests = [
        call for call in client.calls if "оцениваешь" not in call[0]["content"]
        and "превращаешь вопрос" not in call[0]["content"]
    ]
    assert len(answer_requests) == 1


def test_no_matches_sends_a_state_message(repository: Path, tmp_path: Path, index_file: Path):
    retriever = code_retrieval.CodeRetriever(index_file, CORPUS)
    client = FakeClient("Отвечаю без источника.")
    client.score = 0.1
    agent = _agent(repository, client, retriever, tmp_path)
    agent.ask("где инициализируется модель списка?")

    request = "\n".join(message["content"] for message in client.calls[-1])
    assert "ни один не признан относящимся" in request
    assert "Ниже — фрагменты, найденные по запросу" not in request


def test_missing_index_adds_nothing(repository: Path, tmp_path: Path):
    empty = code_retrieval.CodeRetriever(tmp_path / "cache" / "нет.sqlite3", CORPUS)
    client = FakeClient("Обычный ответ.")
    agent = _agent(repository, client, empty, tmp_path)
    agent.ask("вопрос")

    assert agent.code_report() is None
    assert "Корпус исходников" not in _fragment_text(agent)
    assert "фрагменты, найденные по запросу" not in _fragment_text(agent)


def test_unreadable_index_is_reported(repository: Path, tmp_path: Path):
    database = tmp_path / "cache" / "index.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_text("не база", encoding="utf-8")
    retriever = code_retrieval.CodeRetriever(database, CORPUS)
    client = FakeClient("Отвечаю без источника.")
    agent = _agent(repository, client, retriever, tmp_path)
    agent.ask("вопрос")

    assert agent.code_report().status == code_retrieval.STATUS_UNAVAILABLE
    assert "недоступен" in _fragment_text(agent)


def test_baseline_makes_no_auxiliary_requests(repository: Path, retriever, tmp_path: Path):
    client = FakeClient("Ответ без ссылок.")
    client.score = 0.1
    agent = _agent(repository, client, retriever, tmp_path)
    agent.config.code_retrieval = "baseline"
    agent.ask("где инициализируется модель списка?")

    assert len(client.calls) == 1, "в baseline только запрос ответа"


def test_auxiliary_spend_lands_in_the_session_ledger(repository: Path, retriever, tmp_path: Path):
    client = FakeClient("Ответ без ссылок.")
    agent = _agent(repository, client, retriever, tmp_path)
    agent.ask("где инициализируется модель списка?")

    usage = agent.session_usage
    assert usage.requests >= 3, "переформулировка, оценка и ответ"
    assert usage.total_tokens > 15


def test_fragments_do_not_reach_history(repository: Path, retriever, tmp_path: Path):
    answer = f"{CODE}:L1-L9 — инициализация.\n\nЦитата: {QUOTE}"
    client = FakeClient(answer)
    agent = _agent(repository, client, retriever, tmp_path)
    agent.ask("где инициализируется модель списка?")

    saved = (tmp_path / "history.json").read_text(encoding="utf-8")
    assert "фрагменты, найденные по запросу" not in saved
    assert json.loads(saved)["dialogues"][-1]["answer"] == answer


def test_mixed_corpora_and_code_only_citation(repository: Path, retriever, tmp_path: Path, index_file: Path):
    """Смешанный вопрос: документация и код в одном запросе — ссылка на код подтверждает ответ."""
    from tests.conftest import FakeCandidate, FakeDocsRetriever, FakeFragment

    fragment = FakeFragment(
        identifier="doc/5.2.1/guide — Геопозиция",
        source="developer.auroraos.ru",
        text="x" * 60,
    )
    docs = FakeDocsRetriever(
        report=FakeDocsReport(
            query="вопрос",
            fragments=(fragment,),
            candidates=(FakeCandidate(path=fragment.identifier, title="Документ"),),
        )
    )
    answer = f"{CODE}:L1-L9 — инициализация.\n\nЦитата: {QUOTE}"
    client = FakeClient(answer)
    agent = RepoAgent(
        domain=load_domain("aurora-qt5"),
        client=client,
        root=repository,
        history=HistoryManager(path=tmp_path / "history.json"),
        docs_retriever=docs,
        code_retriever=retriever,
    )
    meta = agent.ask("где инициализируется модель списка?")

    assert meta.content == answer, "ответ по коду подтверждён, повтора нет"
    instructions = [
        message["content"]
        for message in client.calls[-1]
        if message["role"] == "system" and "Подтверди это прямо в ответе" in message["content"]
    ]
    assert len(instructions) == 1, "инструкция цитат одна на все корпуса"


def test_patch_format_gets_no_citation_instruction(repository: Path, retriever, tmp_path: Path):
    client = FakeClient("--- a/src/models.cpp\n+++ b/src/models.cpp\n")
    agent = _agent(repository, client, retriever, tmp_path)
    agent.config.settings = agent.config.settings.with_format(AnswerFormat.PATCH)
    agent.ask("где инициализируется модель списка?")

    request = "\n".join(message["content"] for message in client.calls[-1])
    assert "Подтверди это прямо в ответе" not in request
    assert agent.last_citations is None


def test_code_report_snapshot(repository: Path, retriever, tmp_path: Path):
    client = FakeClient("Ответ без ссылок.")
    agent = _agent(repository, client, retriever, tmp_path)
    agent.ask("где инициализируется модель списка?")

    report = agent.code_report()
    assert report.status == code_retrieval.STATUS_OK
    assert report.rated is True
    assert report.fragments
    assert report.fragments[0].identifier.startswith(CODE)
