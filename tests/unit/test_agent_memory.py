"""Память, профиль и правила домена в агенте: маршрутизация, сообщения, проверка ответа."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config, invariants, memory_layers, user_profile
from core.agent import RepoAgent
from core.api_client import AnswerMeta
from core.domains import load_domain
from core.history_manager import HistoryManager
from core.long_term_memory import LongTermMemory


class FakeClient:
    """Клиент модели: отдаёт ответы по очереди и помнит запросы."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers) or ["Ответ."]
        self.calls: list = []

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(list(messages))
        content = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return AnswerMeta(
            content=content,
            model=model,
            elapsed_seconds=0.1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    return tmp_path / "repo"


@pytest.fixture
def stores(tmp_path: Path):
    return (
        LongTermMemory(path=tmp_path / "memory.json"),
        user_profile.ProfileStore(path=tmp_path / "profile.json"),
    )


def _agent(repo: Path, client, stores, tmp_path: Path) -> RepoAgent:
    long_term, profiles = stores
    return RepoAgent(
        domain=load_domain("aurora-qt5"),
        client=client,
        root=repo,
        history=HistoryManager(path=tmp_path / "history.json"),
        long_term=long_term,
        profiles=profiles,
    )


def _messages(agent: RepoAgent, question: str = "вопрос") -> str:
    return "\n".join(message["content"] for message in agent._build_messages(question))


def test_routing_puts_a_replica_into_long_term(repo: Path, stores, tmp_path: Path):
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, stores, tmp_path)
    agent.ask("в этом проекте только qmake")

    records = agent.long_term.records()
    assert records, "реплика попала в долговременную память по правилу домена"
    assert "qmake" in records[0].value
    assert agent.last_routing, "журналу есть что показать"


def test_memory_message_reaches_the_request(repo: Path, stores, tmp_path: Path):
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, stores, tmp_path)
    agent.ask("в этом проекте только qmake")
    client.answers = ["Второй ответ."]
    agent.ask("а что с моделью?")

    request = "\n".join(
        message["content"] for message in client.calls[-1] if message["role"] == "system"
    )
    assert "Память ассистента" in request
    assert "qmake" in request


def test_invariants_message_goes_with_every_question(repo: Path, stores, tmp_path: Path):
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, stores, tmp_path)
    agent.ask("вопрос")

    request = "\n".join(message["content"] for message in client.calls[-1])
    assert invariants.invariants_message(agent.domain.invariants) in request


def test_violating_answer_is_retried_then_replaced(repo: Path, stores, tmp_path: Path):
    forbidden = agent_rule = agent_forbidden_word()
    client = FakeClient(f"Советую взять {agent_forbidden_word()}.", f"И снова {agent_forbidden_word()}.")
    agent = _agent(repo, client, stores, tmp_path)
    meta = agent.ask("что посоветуешь?")

    assert meta.content.startswith(invariants.REFUSAL_PREFIX)
    assert agent.last_invariants, "нарушение зафиксировано в снимке"


def agent_forbidden_word() -> str:
    """Слово из правил домена: берём первое правило со стемами."""
    for rule in load_domain("aurora-qt5").invariants:
        if rule.forbidden:
            return rule.forbidden[0]
    return ""


def test_negated_mention_is_not_a_violation(repo: Path, stores, tmp_path: Path):
    word = agent_forbidden_word()
    client = FakeClient(f"Всё делаем без {word} — так правильно.")
    agent = _agent(repo, client, stores, tmp_path)
    meta = agent.ask("вопрос")

    assert meta.content.startswith("Всё делаем без")
    assert agent.last_invariants is None


def test_profile_message_comes_from_the_store(repo: Path, stores, tmp_path: Path):
    long_term, profiles = stores
    profiles.save(user_profile.UserProfile(name="мой").with_value("style", "коротко"))
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, (long_term, profiles), tmp_path)
    agent.ask("вопрос")

    request = "\n".join(
        message["content"] for message in client.calls[-1] if message["role"] == "system"
    )
    assert "коротко" in request


def test_clear_keeps_memory_and_profile(repo: Path, stores, tmp_path: Path):
    long_term, profiles = stores
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, stores, tmp_path)
    agent.set_goal("сделать экран")
    agent.ask("в этом проекте только qmake")
    agent.reset()

    assert agent.history.working, "рабочая память задачи переживает очистку диалога"
    assert agent.long_term.records(), "долговременная память переживает очистку диалога"
    assert agent.exchanges == 0


def test_goal_survives_restart(repo: Path, stores, tmp_path: Path):
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, stores, tmp_path)
    agent.set_goal("сделать экран")

    restarted = _agent(repo, FakeClient("Ответ."), stores, tmp_path)
    assert restarted.memory_report()["goal"] == "сделать экран"


def test_forget_all_clears_long_term(repo: Path, stores, tmp_path: Path):
    client = FakeClient("Ответ.")
    agent = _agent(repo, client, stores, tmp_path)
    agent.ask("в этом проекте только qmake")
    removed = agent.forget("all")

    assert removed >= 1
    assert agent.long_term.records() == ()


def test_tool_arguments_are_checked_before_the_call(repo: Path, stores, tmp_path: Path):
    """Правила домена применяются к аргументам до запуска инструмента."""
    rules = load_domain("aurora-qt5").invariants
    secret_rule_term = next(
        (rule.forbidden[0] for rule in rules if rule.forbidden), ""
    )
    found = invariants.check_arguments({"text": f"секрет: {secret_rule_term}"}, rules)
    assert found, "проверка аргументов возвращает нарушение, вызов не выполняется"
