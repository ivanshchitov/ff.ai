"""Автовызов инструментов в живом приложении: цепочка, раунды, маршрутизация и сбои.

Приложение запускается в pty, реестр MCP заменяется серверами-заглушками, модель отвечает
локальным stub-сервером. Проверяется то, что уходит в запросы выбора и ответа, и то, что видно
на экране.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from .harness import AppSession
from .stub_api import answer

FAKE = Path(__file__).resolve().parent.parent / "fake_mcp_server.py"
ANSWER = "Ответ по данным инструментов."
LIVE = 60


def _model_double(stub, choices: list, *, final: str = ANSWER):
    """Двойник модели: ответы выбора по программе теста, на вопрос — фиксированный ответ."""
    state = {"left": list(choices)}

    def builder(payload):
        system = payload["messages"][0]["content"]
        if "Выбери инструменты" in system or "следующий раунд" in system:
            return answer(state["left"].pop(0) if state["left"] else '{"tool": null}')
        return answer(final)

    return stub.dynamic(builder)


def _launch(tmp_path: Path, stub, *, auto_tools: bool = True, servers: str = None) -> AppSession:
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=AppSession.create_target_repo(base=tmp_path, name="aurora-project"),
        api_url=stub.url,
        mcp_command=sys.executable,
        mcp_args=servers or str(FAKE),
        auto_tools=auto_tools,
    )


def _choice_requests(stub) -> list:
    return [
        request
        for request in stub.requests
        if "Выбери инструменты" in request["payload"]["messages"][0]["content"]
        or "следующий раунд" in request["payload"]["messages"][0]["content"]
    ]


def _answer_request(stub) -> str:
    for request in reversed(stub.requests):
        head = request["payload"]["messages"][0]["content"]
        if "Выбери инструменты" in head or "следующий раунд" in head:
            continue
        return "\n".join(message["content"] for message in request["payload"]["messages"])
    return ""


def _chain(*steps) -> str:
    return json.dumps({"steps": [dict(step) for step in steps]}, ensure_ascii=False)


@pytest.fixture
def session(stub, tmp_path: Path):
    _model_double(
        stub,
        [
            _chain(
                {"tool": "fake_echo", "arguments": {"message": "СЕКРЕТ-ИЗ-ИНСТРУМЕНТА"}},
                {"tool": "fake_echo", "arguments": {"message": "$1"}},
                {"tool": "fake_note", "arguments": {"text": "$2"}},
            )
        ],
    )
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        yield app, stub, tmp_path
    finally:
        app.close()


def test_chain_of_three_steps_on_one_choice_request(session):
    app, stub, _ = session
    app.ask("найди и передай данные инструментами", timeout=LIVE)

    assert len(_choice_requests(stub)) == 1, "цепочка стоит одного запроса выбора"
    request = _answer_request(stub)
    assert "шагов: 3" in request
    assert "← шаг 1" in request and "← шаг 2" in request
    assert "СЕКРЕТ-ИЗ-ИНСТРУМЕНТА" in request, "данные дошли через подстановку"

    screen = app.screen_text()
    assert "🔧" in screen
    assert "1 шаг · из окружения.fake_echo" in screen


def test_flow_report_shows_steps_and_stop_reason(session):
    app, stub, _ = session
    app.ask("найди и передай данные инструментами", timeout=LIVE)
    before = len(stub.requests)

    app.send_line("/tool flow")
    screen = app.wait_for("Раундов:", timeout=LIVE)
    assert "запросов выбора: 1" in screen
    assert "шагов: 3" in screen
    assert "Остановка: модель завершила флоу" in screen
    assert "text ← шаг 2:" in screen, "объёмы передачи видны в отчёте"
    assert len(stub.requests) == before, "отчёт не обращается к модели"


def test_two_round_flow_across_two_servers_with_rerouting(stub, tmp_path: Path):
    _model_double(
        stub,
        [
            _chain(
                {
                    "server": "из окружения 1",
                    "tool": "fake_facts",
                    "arguments": {"topic": "модель списка"},
                }
            )[:-1]
            + ', "more": true}',
            _chain({"tool": "fake_echo", "arguments": {"message": "$1"}}),
        ],
    )
    app = _launch(tmp_path, stub, servers=f"{FAKE} | {FAKE} --second")
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.ask("что известно про модель списка?", timeout=LIVE)

        choices = _choice_requests(stub)
        assert len(choices) == 2, "раунд продолжения"
        second_round = "\n".join(
            message["content"] for message in choices[1]["payload"]["messages"]
        )
        assert "тема: модель списка" in second_round, "результат первого раунда в запросе второго"
        assert "получит номер 2" in second_round

        request = _answer_request(stub)
        assert "шагов: 2" in request
        assert "тема: модель списка" in request
        assert "маршрут:" in app.screen_text(), "перенаправление видно в журнале"
    finally:
        app.close()


def test_tool_failure_stops_the_chain(stub, tmp_path: Path):
    _model_double(
        stub,
        [
            _chain(
                {"tool": "fake_echo", "arguments": {"message": "начало"}},
                {"tool": "fake_fail", "arguments": {"reason": "нет доступа"}},
                {"tool": "fake_note", "arguments": {"text": "$2"}},
            )
        ],
    )
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.ask("сделай три шага", timeout=LIVE)

        request = _answer_request(stub)
        assert "шагов: 3" not in request, "цепочка оборвана на втором шаге"
        assert "принято:" not in request, "данные упавшего шага в запрос не идут"
        screen = app.screen_text()
        assert "Флоу остановлен: сбой шага 2" in screen
    finally:
        app.close()


def test_auto_off_removes_the_choice_request(stub, tmp_path: Path):
    _model_double(stub, ['{"tool": null}'])
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.send_line("/tool auto off")
        app.wait_for("Автовызов инструментов выключен", timeout=LIVE)
        app.ask("обычный вопрос", timeout=LIVE)

        assert _choice_requests(stub) == [], "выключенный автовызов не выбирает инструменты"
        assert len(stub.requests) == 1, "только запрос ответа"
        app.send_line("/tool flow")
        assert "Автовызов инструментов: выключен" in app.wait_for(
            "Автовызов инструментов: выключен", timeout=LIVE
        )
    finally:
        app.close()


def test_tool_results_stay_out_of_history(session):
    app, stub, tmp_path = session
    app.ask("найди и передай данные инструментами", timeout=LIVE)

    history = json.loads((tmp_path / "state" / "history.json").read_text(encoding="utf-8"))
    assert history["dialogues"][-1]["answer"] == ANSWER
    assert "СЕКРЕТ-ИЗ-ИНСТРУМЕНТА" not in json.dumps(history, ensure_ascii=False)
