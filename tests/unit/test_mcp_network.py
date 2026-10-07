"""Сетевой контракт: реальный сервер документации портала разработчиков ОС Аврора.

Тест помечен `network` и по умолчанию не выполняется — обычный прогон не должен ходить в сеть.
Он нужен для другого: убедиться, что запись реестра в пакете домена жива, что сервер отвечает
на рукопожатие и что объявленный инструмент проверки действительно работает.
"""

import json

import pytest

from core import domains
from core.mcp_client import MCPClient
from core.mcp_registry import TRANSPORT_HTTP, MCPServerSpec

pytestmark = pytest.mark.network

PORTAL_URL = "https://developer.auroraos.ru/api/mcp"
TIMEOUT = 90.0


@pytest.fixture
def portal_spec() -> MCPServerSpec:
    """Запись берётся из пакета домена: если домен объявил не тот адрес, тест это покажет."""
    domain = domains.load_domain("aurora-qt5")
    servers = [spec for spec in domain.servers if spec.transport == TRANSPORT_HTTP]
    assert servers, "домен должен объявлять сервер документации портала"
    return servers[0]


def test_portal_server_answers_the_health_check(portal_spec: MCPServerSpec):
    connection = MCPClient(portal_spec, timeout=TIMEOUT).connect()
    assert connection.available, connection.error
    assert connection.server_name, "сервер обязан представиться"
    assert connection.protocol_version
    assert connection.tools, "сервер обязан объявить инструменты"


def test_portal_server_declares_the_expected_tools(portal_spec: MCPServerSpec):
    connection = MCPClient(portal_spec, timeout=TIMEOUT).connect()
    names = {tool.name for tool in connection.tools}
    assert {"search", "get_document", "get_doc_versions"} <= names
    described = [tool for tool in connection.tools if tool.description]
    assert len(described) == len(connection.tools), "у каждого инструмента должно быть описание"
    search = next(tool for tool in connection.tools if tool.name == "search")
    assert any(item.name == "query" for item in search.parameters())


def test_portal_server_answers_a_real_call(portal_spec: MCPServerSpec):
    result = MCPClient(portal_spec, timeout=TIMEOUT).call_tool("get_doc_versions", {})
    assert not result.is_error, result.text
    payload = json.loads(result.text)
    versions = [item["version"] for item in payload["versions"]]
    assert any(version.startswith("5.") for version in versions)
    assert versions == sorted(versions, key=lambda value: [int(part) for part in value.split(".")])


def test_search_tool_returns_document_paths(portal_spec: MCPServerSpec):
    """Инструмент поиска — то, на чём будет стоять P4: он обязан возвращать пути документов."""
    result = MCPClient(portal_spec, timeout=TIMEOUT).call_tool(
        "search", {"query": "геопозиция", "index": "docs", "limit": 3}
    )
    assert not result.is_error, result.text
    payload = json.loads(result.text)
    assert payload["results"], "поиск по разделу документации не должен быть пустым"
    for item in payload["results"]:
        assert item["path"] and item["title"] and item["index"] == "docs"
        assert item["snippets"], "сниппеты нужны для предварительного отбора"
