"""Сквозной прогон: приложение в псевдотерминале против локального stub-сервера.

Здесь проверяется то, чего не видно в юнит-слоях: что именно уходит в API, что видит
пользователь на отрисованном экране и что приложение делает с файлами вокруг себя.
"""

import json
from pathlib import Path

import pytest

from core import config

from .harness import AppSession
from .stub_api import StubAPI, answer

pytestmark = pytest.mark.e2e

QUESTION = "где инициализируется модель списка?"
REPLY = "Модель инициализируется в src/models.cpp, в конструкторе ListModel."


def _launch(stub: StubAPI, tmp_path: Path, repo: Path = None, **kwargs) -> AppSession:
    """Приложение с изолированным состоянием; целевой репозиторий создаётся, если не задан."""
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=repo if repo is not None else AppSession.create_target_repo(base=tmp_path),
        api_url=stub.url,
        **kwargs,
    )


def test_request_carries_the_domain_role_and_the_configured_limits(app, stub):
    stub.always(answer(REPLY))
    app.ask(QUESTION)
    system = stub.system_messages()[0]
    assert "ОС Аврора" in system
    assert "Qt 5.6" in system
    assert "вне области" in system  # правила отказа из пакета домена
    user = stub.user_messages()[0]
    # Объём по умолчанию — максимум диапазона, поэтому проверяется он, а не прежние 200.
    assert f"не более {config.DEFAULT_MAX_WORDS} слов" in user
    assert "не более 3 вариантов" in user


def test_refusal_reaches_the_user_unchanged(app, stub):
    """Вопрос не фильтруется клиентом: текст отказа модели показывается как есть."""
    refusal = "Не могу помочь: этот вопрос вне области применимости домена."
    stub.always(answer(refusal))
    screen = app.ask("какая погода в Москве?")
    assert "вне области" in screen
    assert app.metrics_count() == 1  # отказ — обычный ответ, запрос к модели был


def test_reports_come_from_snapshots_and_do_not_call_the_model(app, stub):
    """Отчёты проверяются одной сессией: каждый старт приложения стоит секунды, а проверки
    независимы и читают снимки, поэтому делить их по тестам незачем."""
    stub.always(answer(REPLY))
    app.ask(QUESTION)
    before = stub.call_count
    for line, marker in (
        ("/usage", "Вся история"),
        ("/context", "Стратегия: summary"),
        ("/invariants", "Правил домена: 10"),
        ("/schedule", "Заданий: 0"),
        ("/domain", "Инвариантов домена: 10"),
        ("/rag-docs", "Режим отбора"),
        ("/rag-code", "Индекса нет"),
    ):
        app.send_line(line)
        app.wait_for(marker)
    assert stub.call_count == before, "отчёты не обращаются к модели"


def test_clear_empties_the_dialogue(app, stub, tmp_path: Path):
    stub.always(answer(REPLY))
    app.ask(QUESTION)
    history_file = tmp_path / "state" / "history.json"
    assert json.loads(history_file.read_text(encoding="utf-8"))["dialogues"]
    app.send_line("/clear")
    app.wait_for("Диалог очищен.")
    assert json.loads(history_file.read_text(encoding="utf-8"))["dialogues"] == []


def test_second_question_carries_the_previous_exchange(app, stub):
    stub.always(answer(REPLY))
    app.ask(QUESTION)
    app.ask("а где он используется?")
    roles = [message["role"] for message in stub.last_payload()["messages"]]
    assert roles.count("assistant") == 1
    assert stub.call_count == 2


def test_history_is_restored_on_restart(stub, tmp_path: Path):
    stub.always(answer(REPLY))
    first = _launch(stub, tmp_path)
    try:
        first.ask(QUESTION)
    finally:
        first.close()

    second = _launch(stub, tmp_path)
    try:
        screen = second.wait_for("Восстановлено из истории")
        assert "1 обменов" in screen
        assert second.ask("продолжаем?")
        assert stub.call_count == 2
    finally:
        second.close()


def test_directory_without_git_is_refused(stub, tmp_path: Path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    session = AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=plain,
        api_url=stub.url,
    )
    try:
        session.wait_exit(timeout=30)
        assert session.contains("git")
        assert session.contains("not-a-repo")
        assert stub.call_count == 0
    finally:
        session.close()


def test_unknown_domain_is_refused_by_name(stub, tmp_path: Path):
    session = AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=AppSession.create_target_repo(base=tmp_path),
        api_url=stub.url,
        domain="no-such-domain",
    )
    try:
        session.wait_exit(timeout=30)
        assert session.contains("no-such-domain")
    finally:
        session.close()
