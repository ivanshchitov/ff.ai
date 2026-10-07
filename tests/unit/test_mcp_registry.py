"""Реестр MCP: разбор записей, сборка из домена и переопределение окружением."""

import sys
from pathlib import Path

import pytest

from core import domains, mcp_registry
from core.mcp_registry import (
    MCPServerSpec,
    MCPSpecError,
    TRANSPORT_HTTP,
    TRANSPORT_STDIO,
    describe,
    registry,
    repo_server_spec,
)


@pytest.fixture(autouse=True)
def without_environment_override(monkeypatch):
    """Снимает тестовый сторож из conftest: здесь проверяется сборка настоящего реестра."""
    for name in ("FFAI_MCP_COMMAND", "FFAI_MCP_ARGS", "FFAI_MCP_URL"):
        monkeypatch.delenv(name, raising=False)


def test_stdio_record_needs_a_command():
    spec = MCPServerSpec.from_data({"name": "локальный", "command": "python3", "args": ["-m", "srv"]}, "тест")
    assert spec.transport == TRANSPORT_STDIO
    assert spec.command == "python3"
    assert spec.args == ("-m", "srv")
    assert spec.target == "python3 -m srv"


def test_http_record_needs_an_address():
    spec = MCPServerSpec.from_data(
        {"name": "портал", "transport": "http", "url": "https://example.org/mcp"}, "домен"
    )
    assert spec.transport == TRANSPORT_HTTP
    assert spec.target == "https://example.org/mcp"
    assert spec.source == "домен"


@pytest.mark.parametrize(
    "data, expected",
    [
        ({}, "name"),
        ({"name": ""}, "name"),
        ({"name": "x", "transport": "carrier-pigeon"}, "транспорт"),
        ({"name": "x"}, "команда"),
        ({"name": "x", "transport": "http"}, "адрес"),
        ({"name": "x", "transport": "http", "url": "ftp://example.org"}, "http"),
        ({"name": "x", "command": "c", "args": "не список"}, "args"),
        ({"name": "x", "command": "c", "env_keys": [1]}, "env_keys"),
    ],
)
def test_incomplete_records_are_refused_with_the_field(data, expected):
    with pytest.raises(MCPSpecError) as error:
        MCPServerSpec.from_data(data, "тест")
    assert expected in str(error.value)


def test_optional_fields_default_to_empty():
    spec = MCPServerSpec.from_data({"name": "x", "command": "c"}, "тест")
    assert spec.args == () and spec.env_keys == () and spec.health_tool == ""
    assert spec.description == ""


def test_repo_server_spec_points_at_our_script_and_root(tmp_path: Path):
    tools = tmp_path / "tools.json"
    spec = repo_server_spec(tmp_path, tools)
    assert spec.command == sys.executable
    assert "--root" in spec.args and str(tmp_path) in spec.args
    assert "--tools" in spec.args and str(tools) in spec.args
    assert spec.source == "инструмент"
    script = Path(spec.args[0])
    assert script.name == "repo_server.py"
    assert script.is_file(), "запись реестра указывает на существующий скрипт сервера"


def test_repo_server_spec_without_tools_file(tmp_path: Path):
    spec = repo_server_spec(tmp_path)
    assert "--tools" not in spec.args


def test_registry_is_domain_servers_then_our_server(tmp_path: Path):
    domain = domains.load_domain("aurora-qt5")
    specs = registry(domain, tmp_path)
    names = [spec.name for spec in specs]
    assert names == ["aurora-docs", "repo"]
    assert specs[0].transport == TRANSPORT_HTTP
    assert specs[0].source == "домен"
    assert specs[1].source == "инструмент"
    assert all("just a moment" not in spec.target for spec in specs)


def test_registry_without_root_has_no_repository_server():
    domain = domains.load_domain("aurora-qt5")
    names = [spec.name for spec in registry(domain, None)]
    assert names == ["aurora-docs"]


def test_environment_replaces_the_whole_registry(monkeypatch, tmp_path: Path):
    domain = domains.load_domain("aurora-qt5")
    monkeypatch.setenv("FFAI_MCP_COMMAND", "python3")
    monkeypatch.setenv("FFAI_MCP_ARGS", "-m tests.fake_mcp_server --empty")
    monkeypatch.delenv("FFAI_MCP_URL", raising=False)
    specs = registry(domain, tmp_path)
    assert len(specs) == 1
    assert specs[0].command == "python3"
    assert specs[0].args == ("-m", "tests.fake_mcp_server", "--empty")
    assert specs[0].source == "окружение"


def test_environment_can_point_at_a_remote_server(monkeypatch, tmp_path: Path):
    domain = domains.load_domain("aurora-qt5")
    monkeypatch.setenv("FFAI_MCP_URL", "http://127.0.0.1:9/mcp")
    specs = registry(domain, tmp_path)
    assert len(specs) == 1
    assert specs[0].transport == TRANSPORT_HTTP
    assert specs[0].url == "http://127.0.0.1:9/mcp"


def test_describe_lists_names_and_transports(tmp_path: Path):
    domain = domains.load_domain("aurora-qt5")
    text = describe(registry(domain, tmp_path))
    assert "aurora-docs (http)" in text and "repo (stdio)" in text


def test_identifiers_of_domain_do_not_include_server_names():
    """Имена серверов — тоже данные: сторож ядра не должен на них натыкаться."""
    domain = domains.load_domain("aurora-qt5")
    identifiers = domains.identifiers_of(domain)
    for spec in domain.servers:
        assert spec.name.lower() not in identifiers
