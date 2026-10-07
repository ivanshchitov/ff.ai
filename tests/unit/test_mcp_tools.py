"""Автовызов инструментов: каталог, разбор выбора, ссылки, маршрутизация и бюджеты.

Здесь проверяется всё, что можно проверить без модели и без процессов серверов: снимки подключений
подставляются заглушками, а вызовы идут через поддельную фабрику клиентов.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config, mcp_tools
from core.mcp_client import MCPCallResult, MCPConnection, MCPServerSpec, MCPTool

SPECS = (
    MCPServerSpec(name="repo", transport="stdio", command="python3"),
    MCPServerSpec(name="docs", transport="http", url="https://docs.example.invalid/api/mcp"),
)


def _tool(name: str, description: str = "Описание", **schema) -> MCPTool:
    properties = schema or {"query": {"type": "string", "description": "Запрос"}}
    return MCPTool(
        name=name,
        description=description,
        input_schema={"properties": properties, "required": ["query"]},
    )


def _views(*servers) -> tuple:
    """(spec_name, tools, error) → снимки подключений в виде, который нужен выбору."""
    specs = tuple(MCPServerSpec(name=name, transport="stdio", command="python3") for name, _, _ in servers)
    connections = tuple(
        MCPConnection(server_name=name, tools=tuple(tools), error=error)
        for name, tools, error in servers
    )
    return specs, connections


def _views_of(*servers) -> tuple:
    specs, connections = _views(*servers)
    return mcp_tools.views(specs, connections)


# --- каталог --------------------------------------------------------------------------------


def test_catalog_lists_tools_and_parameters():
    tools = (
        MCPTool(
            name="repo_search",
            description="Поиск по исходникам",
            input_schema={
                "properties": {
                    "query": {"type": "string", "description": "Запрос"},
                    "limit": {"type": "integer", "description": "Сколько", "default": 20},
                },
                "required": ["query"],
            },
        ),
    )
    catalog = mcp_tools.render_catalog(_views_of(("repo", tools, None)))
    assert "Сервер «repo»" in catalog
    assert "repo_search: Поиск по исходникам" in catalog
    assert "query (string) — обязательный Запрос" in catalog
    assert "limit (integer) — необязательный Сколько по умолчанию 20" in catalog


def test_unavailable_server_is_not_in_the_catalog():
    views = _views_of(
        ("repo", (_tool("repo_search"),), None),
        ("docs", (_tool("docs_search"),), "нет связи"),
    )
    catalog = mcp_tools.render_catalog(views)
    assert "repo_search" in catalog
    assert "docs_search" not in catalog
    assert "Сервер «docs»" not in catalog


def test_catalog_shows_allowed_values():
    tool = MCPTool(
        name="docs_search",
        description="Поиск",
        input_schema={
            "properties": {"index": {"type": "string", "enum": ["docs", "release_notes"]}},
            "required": [],
        },
    )
    catalog = mcp_tools.render_catalog(_views_of(("docs", (tool,), None)))
    assert "возможные значения: docs, release_notes" in catalog


def test_choice_messages_carry_catalog_and_question():
    messages = mcp_tools.build_choice_messages(_views_of(("repo", (_tool("repo_search"),), None)), "где?")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "repo_search" in messages[1]["content"]
    assert "где?" in messages[1]["content"]
    assert messages[0]["content"] == mcp_tools.choice_instruction()


# --- разбор ---------------------------------------------------------------------------------


def test_parse_choice_single_tool():
    parsed = mcp_tools.parse_choice(
        '{"server": "repo", "tool": "repo_search", "arguments": {"query": "модель"}}'
    )
    assert parsed == mcp_tools.ToolChoice(
        tool="repo_search", arguments={"query": "модель"}, server="repo"
    )


def test_parse_choice_null_means_no_tool():
    assert mcp_tools.parse_choice('{"tool": null}') is None


def test_parse_choice_without_arguments_and_server():
    parsed = mcp_tools.parse_choice('{"tool": "repo_tree"}')
    assert parsed.arguments == {}
    assert parsed.server == ""


def test_parse_choice_garbage_is_unparsed():
    assert mcp_tools.parse_choice("инструментов не нужно, отвечу сам") is mcp_tools.UNPARSED
    assert mcp_tools.parse_choice("") is mcp_tools.UNPARSED


def test_parse_choice_ignores_non_dict_arguments():
    parsed = mcp_tools.parse_choice('{"tool": "repo_read", "arguments": "path=main.cpp"}')
    assert parsed.arguments == {}


def test_parse_choice_from_fenced_json():
    text = 'Вот выбор:\n```json\n{"tool": "repo_search", "arguments": {}}\n```\n'
    assert mcp_tools.parse_choice(text).tool == "repo_search"


def test_parse_chain_keeps_order_and_counts_dropped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "TOOL_CHAIN_MAX_STEPS", 2)
    text = json.dumps(
        {
            "steps": [
                {"tool": "repo_search", "arguments": {"query": "a"}},
                {"tool": "repo_read", "arguments": {"path": "b"}},
                {"tool": "repo_save", "arguments": {"text": "$2"}},
            ]
        }
    )
    steps, dropped = mcp_tools.parse_chain(text)
    assert [step.tool for step in steps] == ["repo_search", "repo_read"]
    assert dropped == 1


def test_parse_chain_single_tool_is_a_chain_of_one():
    steps, dropped = mcp_tools.parse_chain('{"tool": "repo_tree", "arguments": {}}')
    assert [step.tool for step in steps] == ["repo_tree"]
    assert dropped == 0


def test_parse_chain_empty_steps_is_none():
    assert mcp_tools.parse_chain('{"steps": []}') is None
    assert mcp_tools.parse_chain('{"tool": null}') is None


def test_parse_chain_skips_non_dict_steps():
    text = json.dumps({"steps": ["нет", {"tool": "repo_tree", "arguments": {}}]})
    steps, _ = mcp_tools.parse_chain(text)
    assert [step.tool for step in steps] == ["repo_tree"]


def test_parse_round_done_and_more():
    assert mcp_tools.parse_round('{"done": true}') is None
    parsed = mcp_tools.parse_round('{"steps": [{"tool": "repo_search"}], "more": true}')
    assert parsed.more is True and len(parsed.steps) == 1
    without = mcp_tools.parse_round('{"steps": [{"tool": "repo_search"}]}')
    assert without.more is False
    assert mcp_tools.parse_round("ерунда") is mcp_tools.UNPARSED


# --- маршрутизация --------------------------------------------------------------------------


def test_route_named_server_wins():
    views = _views_of(("repo", (_tool("repo_search"),), None), ("docs", (_tool("search"),), None))
    assert mcp_tools.route(views, "docs", "search") == mcp_tools.Route(server="docs")


def test_route_reroutes_to_the_only_declaring_server():
    views = _views_of(("repo", (_tool("repo_search"),), None), ("docs", (_tool("search"),), None))
    result = mcp_tools.route(views, "docs", "repo_search")
    assert result.server == "repo"
    assert result.rerouted_from == "docs"


def test_route_unknown_tool_is_an_error():
    views = _views_of(("repo", (_tool("repo_search"),), None))
    result = mcp_tools.route(views, "", "нет-такого")
    assert result.error and "не объявлен" in result.error


def test_route_ambiguous_without_server_is_an_error():
    views = _views_of(("repo", (_tool("search"),), None), ("docs", (_tool("search"),), None))
    result = mcp_tools.route(views, "", "search")
    assert "неоднозначно" in result.error


def test_route_ignores_unavailable_servers():
    views = _views_of(("repo", (_tool("repo_search"),), "нет связи"), ("docs", (_tool("search"),), None))
    assert mcp_tools.route(views, "", "repo_search").error


# --- ссылки и аргументы ---------------------------------------------------------------------


def test_reference_replaces_whole_value():
    resolved, sources = mcp_tools.resolve_references({"text": "$1"}, ["полный текст"])
    assert resolved == {"text": "полный текст"}
    assert sources == {"text": (1,)}


def test_reference_replaces_a_line_of_a_multiline_value():
    resolved, sources = mcp_tools.resolve_references(
        {"text": "Итог\n== Код ==\n$1\n== Документация ==\n$2"},
        ["код", "документация"],
    )
    assert resolved["text"] == "Итог\n== Код ==\nкод\n== Документация ==\nдокументация"
    assert sources == {"text": (1, 2)}


def test_reference_inside_a_line_is_left_alone():
    resolved, sources = mcp_tools.resolve_references({"text": "цена $5 и $1"}, ["текст"])
    assert resolved["text"] == "цена $5 и $1"
    assert sources == {}


def test_reference_to_unperformed_step_fails():
    with pytest.raises(mcp_tools.ReferenceFailure):
        mcp_tools.resolve_references({"text": "$3"}, ["один"])


def test_non_string_argument_is_kept():
    resolved, sources = mcp_tools.resolve_references({"limit": 5}, ["текст"])
    assert resolved == {"limit": 5} and sources == {}


def test_render_arguments_shows_reference_not_text():
    rendered = mcp_tools.render_arguments({"text": "очень длинный текст", "name": "итог"}, {"text": (1,)})
    assert rendered == "name=итог, text=← шаг 1"


def test_source_label_plural():
    assert mcp_tools.source_label((1,)) == "← шаг 1"
    assert mcp_tools.source_label((1, 2)) == "← шаги 1, 2"


def test_render_arguments_without_arguments():
    assert mcp_tools.render_arguments({}) == mcp_tools.NO_ARGUMENTS


# --- сообщения результата -------------------------------------------------------------------


def test_result_message_includes_instruction_and_text():
    message = mcp_tools.tool_result_message("repo", "repo_search", {"query": "модель"}, "найдено: 3")
    assert "Инструмент «repo_search» сервера «repo»" in message
    assert "query=модель" in message
    assert "найдено: 3" in message
    assert mcp_tools.result_instruction() in message


def test_result_message_clips_long_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "TOOL_ANSWER_RESULT_CHARS", 10)
    message = mcp_tools.tool_result_message("repo", "repo_read", {}, "x" * 50)
    assert "сокращено: показано 10 из 50" in message


def test_chain_message_numbers_steps_and_marks_references():
    message = mcp_tools.tool_chain_message(
        [
            ("repo", "repo_search", {"query": "a"}, {}, "первый"),
            ("repo", "repo_save", {"text": "первый"}, {"text": (1,)}, "сохранено: отчёт.md"),
        ]
    )
    assert "шагов: 2" in message
    assert "Шаг 1: инструмент «repo_search»" in message
    assert "text=← шаг 1" in message
    assert "сохранено: отчёт.md" in message


def test_round_messages_include_done_steps_and_next_number(monkeypatch: pytest.MonkeyPatch):
    steps = (
        mcp_tools.ToolStep(
            step=1,
            round=1,
            server="repo",
            tool="repo_search",
            arguments={"query": "a"},
            sources={},
            text="найдено",
        ),
    )
    messages = mcp_tools.build_round_messages(
        _views_of(("repo", (_tool("repo_search"),), None)), "где?", steps
    )
    assert "Выполненные шаги (1)" in messages[1]["content"]
    assert "получит номер 2" in messages[1]["content"]
    assert "найдено" in messages[1]["content"]
    assert mcp_tools.flow_instruction() in messages[0]["content"]


def test_round_budget_clips_old_rounds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "TOOL_FLOW_OLD_RESULT_CHARS", 5)
    monkeypatch.setattr(config, "TOOL_FLOW_CONTEXT_CHARS", 1000)
    old = mcp_tools.ToolStep(1, 1, "repo", "repo_search", {}, {}, "x" * 50)
    fresh = mcp_tools.ToolStep(2, 2, "repo", "repo_read", {}, {}, "y" * 50)
    messages = mcp_tools.build_round_messages(
        _views_of(("repo", (_tool("repo_search"),), None)), "где?", [old, fresh]
    )
    content = messages[1]["content"]
    assert content.count("сокращено: показано 5 из 50") == 1, "режется только старый раунд"
    assert "y" * 50 in content


# --- доставка и журнал ----------------------------------------------------------------------


def test_step_journal_line_shows_volumes_and_reroute():
    step = mcp_tools.ToolStep(
        step=2,
        round=1,
        server="repo",
        tool="repo_save",
        arguments={"text": "готовый итог"},
        sources={"text": (1,)},
        text="сохранено",
        rerouted_from="docs",
    )
    line = step.journal_line()
    assert "2 шаг · repo.repo_save" in line
    assert "маршрут: docs → repo" in line
    assert "text ← шаг 1: 12 симв." in line


# --- доступ к инструментам ------------------------------------------------------------------


class FakeSession:
    def __init__(self, name: str, calls: list) -> None:
        self.name = name
        self.calls = calls

    def call_tool(self, tool, arguments):
        self.calls.append((self.name, tool, arguments))
        return MCPCallResult(server=self.name, tool=tool, arguments=arguments, text=f"ответ {tool}")


class FakeClient:
    def __init__(self, spec, opened: list, calls: list) -> None:
        self.spec = spec
        self.opened = opened
        self.calls = calls
        self.closed = 0

    def session(self):
        client = self

        class _Context:
            def __enter__(self):
                client.opened.append(client.spec.name)
                return FakeSession(client.spec.name, client.calls)

            def __exit__(self, *exc_info):
                client.closed += 1
                return False

        return _Context()


def test_hub_opens_one_session_per_server_and_closes_it():
    specs, connections = _views(("repo", (_tool("repo_search"),), None), ("docs", (_tool("search"),), None))
    opened: list = []
    calls: list = []
    hub = mcp_tools.ToolHub(specs, connections, lambda spec: FakeClient(spec, opened, calls))
    with hub:
        hub.call("repo", "repo_search", {"query": "a"})
        hub.call("repo", "repo_read", {"path": "b"})
        hub.call("docs", "search", {"query": "c"})
    assert opened == ["repo", "docs"], "соединение открывается один раз на сервер"
    assert [item[1] for item in calls] == ["repo_search", "repo_read", "search"]
    assert len(opened) == 2


def test_hub_catalog_skips_unavailable_and_empty():
    specs, connections = _views(
        ("repo", (_tool("repo_search"),), None),
        ("empty", (), None),
        ("docs", (_tool("search"),), "нет связи"),
    )
    hub = mcp_tools.ToolHub(specs, connections, lambda spec: None)
    catalog = hub.catalog()
    assert [view.spec_name for view in catalog] == ["repo"]
    assert len(hub.views()) == 3


def test_hub_unknown_server_raises():
    specs, connections = _views(("repo", (_tool("repo_search"),), None))
    hub = mcp_tools.ToolHub(specs, connections, lambda spec: None)
    with pytest.raises(KeyError):
        hub.call("нет-такого", "repo_search", {})


def test_instructions_come_from_assets():
    assert "инструменты" in mcp_tools.choice_instruction()
    assert "раунд" in mcp_tools.flow_instruction()
    assert "данн" in mcp_tools.result_instruction()
    for name in (
        mcp_tools.TOOL_CHOICE_ASSET,
        mcp_tools.TOOL_FLOW_ASSET,
        mcp_tools.TOOL_RESULT_ASSET,
    ):
        assert (config.ASSETS_DIR / name).is_file()
