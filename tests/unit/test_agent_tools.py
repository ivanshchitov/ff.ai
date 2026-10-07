"""Автовызов инструментов в агенте: выбор, цепочка, раунды, лимиты и сбои.

Клиент модели отдаёт заранее заданные ответы выбора, хаб инструментов записывает вызовы. Ни сети,
ни серверных процессов: проверяется то, что агент отправляет и что складывает в снимок.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config, mcp_tools
from core.agent import RepoAgent
from core.api_client import APIError, AnswerMeta
from core.domains import load_domain
from core.history_manager import HistoryManager
from core.mcp_client import MCPCallResult, MCPConnection, MCPServerSpec, MCPTool


def _tool(name: str, description: str = "Описание") -> MCPTool:
    return MCPTool(
        name=name,
        description=description,
        input_schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
    )


class FakeHub:
    """Хаб инструментов: снимки задаются тестом, вызовы записываются, отказы — по имени."""

    def __init__(self, tools: dict, failing: tuple = (), error: str = "") -> None:
        self.specs = (
            MCPServerSpec(name="repo", transport="stdio", command="python3"),
            MCPServerSpec(name="docs", transport="http", url="https://docs.example.invalid/mcp"),
        )
        connections = []
        for spec in self.specs:
            declared = tuple(_tool(name) for name in tools.get(spec.name, ()))
            connections.append(
                MCPConnection(
                    server_name=spec.name,
                    tools=declared,
                    error=error if error and spec.name == "docs" else None,
                )
            )
        self.connections = tuple(connections)
        self.calls = []
        self.closed = 0
        self.failing = failing

    def views(self):
        return mcp_tools.views(self.specs, self.connections)

    def catalog(self):
        return tuple(view for view in self.views() if not view.error and view.tools)

    def call(self, server: str, tool: str, arguments: dict) -> MCPCallResult:
        self.calls.append((server, tool, dict(arguments)))
        if tool in self.failing:
            return MCPCallResult(
                server=server, tool=tool, arguments=dict(arguments), text="отказ", is_error=True
            )
        if tool == "repo_save":
            return MCPCallResult(
                server=server,
                tool=tool,
                arguments=dict(arguments),
                text="сохранено: отчёт.md",
            )
        return MCPCallResult(server=server, tool=tool, arguments=dict(arguments), text=f"итог {tool}")

    def close(self) -> None:
        self.closed += 1


class FakeClient:
    """Клиент модели: ответы выбора и ответы на вопрос по программе теста."""

    def __init__(self, *answers: str, choices: list = None) -> None:
        self.answers = list(answers) or ["Ответ без ссылок."]
        self.choices = list(choices or [])
        self.calls = []
        self.choice_calls = []

    def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(list(messages))
        if messages[0]["content"] == mcp_tools.choice_instruction() or "раунд" in messages[0]["content"]:
            self.choice_calls.append(list(messages))
            content = self.choices.pop(0) if self.choices else '{"tool": null}'
        else:
            content = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return AnswerMeta(
            content=content,
            model=model,
            elapsed_seconds=0.2,
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
            cost_usd=0.0002,
        )


def _agent(tmp_path: Path, client, hub) -> RepoAgent:
    (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
    return RepoAgent(
        domain=load_domain("aurora-qt5"),
        client=client,
        root=tmp_path / "repo",
        history=HistoryManager(path=tmp_path / "history.json"),
        tool_hub=hub,
    )


def _step(number: int, tool: str, arguments: dict, server: str = "") -> dict:
    step = {"tool": tool, "arguments": arguments}
    if server:
        step["server"] = server
    return step


def _chain(*steps) -> str:
    return json.dumps({"steps": [dict(step) for step in steps]}, ensure_ascii=False)


TOOLS = {"repo": ("repo_search", "repo_read", "repo_save"), "docs": ("docs_search",)}


# --- один раунд -----------------------------------------------------------------------------


def test_single_tool_result_goes_into_the_request(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.", choices=[_chain(_step(1, "repo_search", {"query": "модель"}))])
    agent = _agent(tmp_path, client, hub)
    agent.ask("где инициализируется модель списка?")

    request = client.calls[-1]
    system = [message["content"] for message in request if message["role"] == "system"]
    tool_message = next(text for text in system if "внешнего источника" in text)
    assert "repo_search" in tool_message and "итог repo_search" in tool_message
    assert hub.calls == [("repo", "repo_search", {"query": "модель"})]
    assert hub.closed == 1, "соединения флоу закрываются"


def test_chain_passes_data_by_reference(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ.",
        choices=[
            _chain(
                _step(1, "repo_search", {"query": "список"}),
                _step(2, "repo_read", {"query": "$1"}),
                _step(3, "repo_save", {"query": "$2"}),
            )
        ],
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди и сохрани")

    assert [call[1] for call in hub.calls] == ["repo_search", "repo_read", "repo_save"]
    assert hub.calls[1][2]["query"] == "итог repo_search", "подставлен полный текст шага"
    assert hub.calls[2][2]["query"] == "итог repo_read"

    request = "\n".join(
        message["content"] for message in client.calls[-1] if message["role"] == "system"
    )
    assert "шагов: 3" in request
    assert "← шаг 1" in request and "← шаг 2" in request
    report = agent.tool_flow_report()
    assert (report.rounds, report.choice_requests, report.executed) == (1, 1, 3)
    assert report.stop_reason == mcp_tools.FLOW_DONE


def test_manual_style_single_choice_without_steps(tmp_path: Path):
    """Ответ с одним инструментом (без `steps`) — цепочка из одного шага."""
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.", choices=[json.dumps({"tool": "repo_search", "arguments": {"query": "a"}})])
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    assert [call[1] for call in hub.calls] == ["repo_search"]
    assert agent.tool_flow_report().stop_reason == mcp_tools.FLOW_DONE


# --- раунды----------------------------------------------------------------------------------


def test_rounds_continue_with_results_and_global_numbering(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ.",
        choices=[
            json.dumps(
                {"steps": [_step(1, "repo_search", {"query": "файл"})], "more": True},
                ensure_ascii=False,
            ),
            json.dumps(
                {"steps": [_step(2, "docs_search", {"query": "$1"})], "more": False},
                ensure_ascii=False,
            ),
        ],
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди файл и спроси документацию про него")

    assert len(client.choice_calls) == 2
    second = "\n".join(message["content"] for message in client.choice_calls[1])
    assert "итог repo_search" in second, "результат первого раунда попал в запрос второго"
    assert "получит номер 2" in second
    report = agent.tool_flow_report()
    assert (report.rounds, report.choice_requests, report.executed) == (2, 2, 2)
    assert [step.step for step in report.steps] == [1, 2]
    assert [step.round for step in report.steps] == [1, 2]


def test_rerouted_step_is_marked(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ.", choices=[_chain(_step(1, "repo_search", {"query": "a"}, server="docs"))]
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    step = agent.tool_flow_report().steps[0]
    assert step.server == "repo"
    assert step.rerouted_from == "docs"
    assert "маршрут: docs → repo" in step.journal_line()


# --- лимиты и сбои --------------------------------------------------------------------------


def test_chain_limit_drops_extra_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "TOOL_CHAIN_MAX_STEPS", 2)
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ.",
        choices=[
            _chain(
                _step(1, "repo_search", {"query": "a"}),
                _step(2, "repo_search", {"query": "b"}),
                _step(3, "repo_search", {"query": "c"}),
            )
        ],
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    report = agent.tool_flow_report()
    assert report.executed == 2
    assert report.dropped == 1


def test_step_limit_stops_the_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "TOOL_FLOW_MAX_STEPS", 1)
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ.",
        choices=[
            _chain(
                _step(1, "repo_search", {"query": "a"}),
                _step(2, "repo_search", {"query": "b"}),
            )
        ],
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    report = agent.tool_flow_report()
    assert report.executed == 1
    assert "предел шагов" in report.stop_reason


def test_rounds_limit_stops_the_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "TOOL_FLOW_MAX_ROUNDS", 2)
    hub = FakeHub(TOOLS)
    never_done = json.dumps(
        {"steps": [_step(1, "repo_search", {"query": "a"})], "more": True}, ensure_ascii=False
    )
    client = FakeClient("Ответ.", choices=[never_done, never_done, never_done])
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    report = agent.tool_flow_report()
    assert report.rounds == 2
    assert "предел раундов" in report.stop_reason


def test_tool_error_stops_the_chain(tmp_path: Path):
    hub = FakeHub(TOOLS, failing=("repo_read",))
    client = FakeClient(
        "Ответ.",
        choices=[
            _chain(
                _step(1, "repo_search", {"query": "a"}),
                _step(2, "repo_read", {"query": "$1"}),
                _step(3, "repo_save", {"query": "$2"}),
            )
        ],
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди и сохрани")

    assert [call[1] for call in hub.calls] == ["repo_search", "repo_read"]
    report = agent.tool_flow_report()
    assert report.executed == 1
    assert "сбой шага" in report.stop_reason
    request = "\n".join(
        message["content"] for message in client.calls[-1] if message["role"] == "system"
    )
    assert "итог repo_read" not in request, "данные упавшего шага в запрос не идут"


def test_unknown_tool_stops_the_chain(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.", choices=[_chain(_step(1, "нет-такого", {"query": "a"}))])
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    assert hub.calls == []
    assert "сбой шага" in agent.tool_flow_report().stop_reason


def test_reference_to_unperformed_step_stops_the_chain(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.", choices=[_chain(_step(1, "repo_read", {"query": "$3"}))])
    agent = _agent(tmp_path, client, hub)
    agent.ask("найди")
    assert hub.calls == []
    assert "сбой шага" in agent.tool_flow_report().stop_reason


def test_unparsed_choice_does_not_block_the_answer(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Обычный ответ.", choices=["вообще не JSON"])
    agent = _agent(tmp_path, client, hub)
    meta = agent.ask("вопрос")

    assert meta.content == "Обычный ответ."
    report = agent.tool_flow_report()
    assert "сбой выбора" in report.stop_reason
    assert report.executed == 0


def test_failed_choice_request_does_not_block_the_answer(tmp_path: Path):
    class FailingClient(FakeClient):
        def ask_with_usage_messages(self, messages, max_tokens=None, temperature=None, model=None):
            if messages[0]["content"] == mcp_tools.choice_instruction():
                raise APIError("модель недоступна")
            return super().ask_with_usage_messages(messages, max_tokens, temperature, model)

    hub = FakeHub(TOOLS)
    agent = _agent(tmp_path, FailingClient("Ответ."), hub)
    meta = agent.ask("вопрос")
    assert meta.content == "Ответ."
    assert "сбой выбора" in agent.tool_flow_report().stop_reason


def test_no_tool_needed_leaves_the_request_shape_alone(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.", choices=['{"tool": null}'])
    agent = _agent(tmp_path, client, hub)
    agent.ask("объясни, что такое moc")

    request = "\n".join(
        message["content"] for message in client.calls[-1] if message["role"] == "system"
    )
    assert "внешнего источника" not in request
    report = agent.tool_flow_report()
    assert report.executed == 0
    assert report.stop_reason == mcp_tools.FLOW_NO_TOOL


# --- включение и каталог --------------------------------------------------------------------


def test_auto_tools_off_makes_no_requests(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.")
    agent = _agent(tmp_path, client, hub)
    agent.config.auto_tools = False
    agent.ask("вопрос")

    assert len(client.calls) == 1, "только запрос ответа"
    assert hub.calls == []
    assert agent.tool_flow_report() is None


def test_empty_catalog_makes_no_requests(tmp_path: Path):
    hub = FakeHub({})
    client = FakeClient("Ответ.")
    agent = _agent(tmp_path, client, hub)
    agent.ask("вопрос")
    assert len(client.calls) == 1
    assert agent.tool_flow_report() is None


def test_unavailable_servers_are_not_offered(tmp_path: Path):
    hub = FakeHub({"repo": ("repo_search",), "docs": ("docs_search",)}, error="нет связи")
    client = FakeClient("Ответ.", choices=['{"tool": null}'])
    agent = _agent(tmp_path, client, hub)
    agent.ask("вопрос")
    catalog_message = client.choice_calls[0][1]["content"]
    assert "repo_search" in catalog_message
    assert "docs_search" not in catalog_message


# --- расход и история -----------------------------------------------------------------------


def test_choice_requests_land_in_the_ledger(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient("Ответ.", choices=['{"tool": null}'])
    agent = _agent(tmp_path, client, hub)
    agent.ask("вопрос")
    usage = agent.session_usage
    assert usage.requests == 2, "выбор и ответ"
    assert usage.total_tokens == 56


def test_tool_results_do_not_reach_history(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ с цитатой.", choices=[_chain(_step(1, "repo_search", {"query": "модель"}))]
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("где инициализируется модель?")

    saved = (tmp_path / "history.json").read_text(encoding="utf-8")
    assert "итог repo_search" not in saved
    assert "Ответ с цитатой." in saved


def test_tool_message_sits_above_the_dialogue_turns(tmp_path: Path):
    hub = FakeHub(TOOLS)
    client = FakeClient(
        "Ответ.",
        choices=[
            _chain(_step(1, "repo_search", {"query": "модель"})),
            _chain(_step(1, "repo_search", {"query": "второй"})),
        ],
    )
    agent = _agent(tmp_path, client, hub)
    agent.ask("первый вопрос")
    client.answers = ["Второй ответ."]
    agent.ask("второй вопрос")

    roles = [message["role"] for message in client.calls[-1]]
    system_count = roles.count("system")
    assert system_count >= 2
    contents = [message["content"] for message in client.calls[-1]]
    tool_index = next(
        index for index, text in enumerate(contents) if "внешнего источника" in text
    )
    first_user = roles.index("user")
    assert tool_index < first_user, "результаты инструментов идут выше реплик диалога"
