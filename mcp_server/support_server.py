#!/usr/bin/env python3
"""Read-only support JSON through stdio MCP."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from mcp_server.support_data import get_ticket, SupportDataError

server = MCPServer('support', log_level='ERROR')
_root = Path('.')
_data = 'support.json'

@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def support_health() -> str:
    """Проверка доступности сервера поддержки без чтения персональных данных."""
    return 'ok'

@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def support_get_ticket(ticket_id: str) -> str:
    """Получить выбранный тикет и связанного пользователя, только разрешённые поля.

    Args:
        ticket_id: идентификатор обращения, например TICKET-101; не путь файла.
    """
    try:
        return json.dumps(get_ticket(_root, _data, ticket_id), ensure_ascii=False)
    except SupportDataError as error:
        raise ToolError(str(error)) from None

def main():
    global _root, _data
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    _root, _data = Path(args.root), args.data
    server.run(transport='stdio')

if __name__ == '__main__':
    main()
