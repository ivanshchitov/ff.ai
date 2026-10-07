"""Клиент MCP: то, что проверяется без поднятия сервера — схемы, причины сбоев, разбор ответа."""

from pathlib import Path

import pytest

from core import config
from core.mcp_client import (
    MCPCallResult,
    MCPClient,
    MCPError,
    MCPParameter,
    MCPTool,
    _content_text,
    _describe,
)
from core.mcp_registry import MCPServerSpec, TRANSPORT_HTTP, TRANSPORT_STDIO


def test_parameters_are_read_from_the_schema():
    tool = MCPTool(
        name="fake_sum",
        description="сумма",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "первое число"},
                "b": {"type": "integer", "default": 0},
                "mode": {"type": "string", "enum": ["сумма", "разность"]},
            },
            "required": ["a"],
        },
    )
    parameters = {item.name: item for item in tool.parameters()}
    assert parameters["a"].required and parameters["a"].type == "integer"
    assert parameters["a"].render() == "a: integer (обязательный)"
    assert parameters["b"].default == "0"
    assert parameters["mode"].allowed == ("сумма", "разность")
    assert "[сумма, разность]" in parameters["mode"].render()
    assert "описание" not in parameters["a"].render()  # описание печатается отдельной строкой


@pytest.mark.parametrize("schema", [{}, {"properties": "нет"}, "строка", None])
def test_broken_schema_does_not_break_the_report(schema):
    tool = MCPTool(name="x", input_schema=schema)
    assert tool.parameters() == []


def test_schema_without_required_list_marks_everything_optional():
    tool = MCPTool(name="x", input_schema={"properties": {"a": {"type": "string"}}})
    assert [item.required for item in tool.parameters()] == [False]


def test_missing_command_becomes_a_report_error_with_the_command_name(tmp_path: Path):
    spec = MCPServerSpec(
        name="нету", transport=TRANSPORT_STDIO, command="нет-такой-команды-9f3a"
    )
    connection = MCPClient(spec, timeout=5).connect()
    assert not connection.available
    assert "нет-такой-команды-9f3a" in connection.error
    assert connection.transport == TRANSPORT_STDIO


def test_unreachable_http_server_becomes_a_report_error():
    spec = MCPServerSpec(
        name="пусто", transport=TRANSPORT_HTTP, url="http://127.0.0.1:9/mcp"
    )
    connection = MCPClient(spec, timeout=5).connect()
    assert not connection.available
    assert connection.error
    assert connection.target == "http://127.0.0.1:9/mcp"


def test_call_tool_on_unreachable_server_raises_mcp_error():
    spec = MCPServerSpec(
        name="пусто", transport=TRANSPORT_HTTP, url="http://127.0.0.1:9/mcp"
    )
    with pytest.raises(MCPError):
        MCPClient(spec, timeout=5).call_tool("echo", {"message": "привет"})


def test_content_text_joins_text_blocks_and_ignores_others():
    class Block:
        def __init__(self, text=None):
            if text is not None:
                self.text = text

    class Result:
        content = [Block("первая"), Block(), Block("вторая")]

    assert _content_text(Result()) == "первая\nвторая"


def test_content_text_of_empty_result_is_empty():
    class Result:
        content = []

    assert _content_text(Result()) == ""


def test_describe_unwraps_exception_groups():
    inner = ValueError("внутренняя причина")
    group = ExceptionGroup("группа задач", [inner])
    text = _describe(group)
    assert "внутренняя причина" in text


def test_result_snapshot_carries_everything_the_report_needs():
    result = MCPCallResult(
        server="repo", tool="repo_read", arguments={"path": "src/main.cpp"}, text="…"
    )
    assert result.server == "repo"
    assert result.arguments["path"] == "src/main.cpp"
    assert result.is_error is False


def test_client_timeout_defaults_to_configuration(tmp_path: Path):
    spec = MCPServerSpec(name="x", command="echo")
    assert MCPClient(spec).timeout == config.MCP_TIMEOUT
    assert MCPClient(spec, timeout=1).timeout == 1


# --- живые заглушки: stdio, по адресу, пустая и не говорящая по протоколу -------------------

import subprocess  # noqa: E402
import sys  # noqa: E402

FAKE_STDIO = config.BASE_DIR / "tests" / "fake_mcp_server.py"
FAKE_HTTP = config.BASE_DIR / "tests" / "fake_mcp_http.py"
FAKE_GARBAGE = config.BASE_DIR / "tests" / "fake_mcp_garbage.py"
SERVER_TIMEOUT = 60.0


def _stdio_spec(*args: str, name: str = "локальный", health_tool: str = "") -> MCPServerSpec:
    return MCPServerSpec(
        name=name,
        transport=TRANSPORT_STDIO,
        command=sys.executable,
        args=(str(FAKE_STDIO), *args),
        health_tool=health_tool,
    )


@pytest.fixture
def fake_http_url():
    """Поднимает заглушку по адресу и отдаёт её URL, напечатанный самим сервером."""
    process = subprocess.Popen(
        [sys.executable, str(FAKE_HTTP), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = process.stdout.readline().strip()
        assert line.startswith("URL="), f"заглушка не напечатала адрес: {line!r}"
        yield line[len("URL=") :]
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_stdio_server_lists_tools_with_parameters():
    connection = MCPClient(_stdio_spec(), timeout=SERVER_TIMEOUT).connect()
    assert connection.available, connection.error
    names = {tool.name for tool in connection.tools}
    assert {"fake_echo", "fake_sum", "fake_note", "fake_fail"} <= names
    echo = next(tool for tool in connection.tools if tool.name == "fake_echo")
    assert echo.description
    assert [item.name for item in echo.parameters()] == ["message"]
    assert echo.parameters()[0].required


def test_stdio_call_returns_text_and_arguments():
    result = MCPClient(_stdio_spec(), timeout=SERVER_TIMEOUT).call_tool(
        "fake_echo", {"message": "проверка"}
    )
    assert not result.is_error
    assert "проверка" in result.text
    assert result.server == "локальный"
    assert result.arguments == {"message": "проверка"}


def test_stdio_server_keeps_markup_intact():
    """Текст инструмента — чужие данные: клиент их не портит, экранирует уже интерфейс."""
    result = MCPClient(_stdio_spec(), timeout=SERVER_TIMEOUT).call_tool(
        "fake_note", {"text": "[/dim] и [bold]"}
    )
    assert "[/dim]" in result.text and "[bold]" in result.text


def test_failing_tool_is_data_not_exception():
    result = MCPClient(_stdio_spec(), timeout=SERVER_TIMEOUT).call_tool("fake_fail", {})
    assert result.is_error
    assert result.text


def test_empty_tool_list_is_a_successful_connection():
    connection = MCPClient(_stdio_spec("--empty"), timeout=SERVER_TIMEOUT).connect()
    assert connection.available, connection.error
    assert connection.tools == ()


def test_garbage_process_is_reported_as_a_failure():
    spec = MCPServerSpec(
        name="мусор", transport=TRANSPORT_STDIO, command=sys.executable, args=(str(FAKE_GARBAGE),)
    )
    connection = MCPClient(spec, timeout=15).connect()
    assert not connection.available
    assert connection.error


def test_health_check_confirms_the_connection():
    connection = MCPClient(_stdio_spec(health_tool="fake_echo"), timeout=SERVER_TIMEOUT).connect()
    assert connection.available, connection.error
    assert connection.tools


def test_health_check_reports_a_missing_tool():
    connection = MCPClient(_stdio_spec(health_tool="которого-нет"), timeout=SERVER_TIMEOUT).connect()
    assert not connection.available
    assert "которого-нет" in connection.error


def test_remote_server_over_address(fake_http_url: str):
    spec = MCPServerSpec(
        name="по адресу",
        transport=TRANSPORT_HTTP,
        url=fake_http_url,
        health_tool="fake_echo",
    )
    connection = MCPClient(spec, timeout=SERVER_TIMEOUT).connect()
    assert connection.available, connection.error
    assert connection.server_name
    names = {tool.name for tool in connection.tools}
    assert "fake_echo" in names

    result = MCPClient(spec, timeout=SERVER_TIMEOUT).call_tool(
        "fake_echo", {"message": "через адрес"}
    )
    assert not result.is_error
    assert "через адрес" in result.text
