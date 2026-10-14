"""TUI + actual ticket MCP + local API stub in one isolated session."""
import json
import shutil
import subprocess
from pathlib import Path
import pytest
from .harness import AppSession
from .stub_api import answer

pytestmark=pytest.mark.e2e

def test_support_two_tickets_and_unknown_id(tmp_path, stub):
    source=Path(__file__).resolve().parents[2]/'examples/support-webdavclient'
    root=tmp_path/'support-project'
    shutil.copytree(source,root)
    subprocess.run(['git','init'],cwd=root,capture_output=True,check=True)
    reply='Завершите повторный вход. Цитата: «приложение открывает страницу OAuth для повторной авторизации.» Источник: docs/faq.md'
    stub.always(answer(reply))
    with AppSession(history_file=tmp_path/'state/history.json',repo=root,domain='product-support',api_url=stub.url,mcp_command='',mcp_args='',rows=55) as app:
        app.send_line('/rag-code index')
        app.wait_for('Файлов:',timeout=25)
        app.ask('/support TICKET-101 Почему не получается войти?',timeout=25)
        app.ask('/support TICKET-102 Почему не получается войти?',timeout=25)
        context=stub.last_payload()['messages'][1]['content']
        assert 'TICKET-102' in context and 'files.example.test' in context
        assert 'TICKET-101' not in context and 'expired_token' not in context
        assert not any(m['role']=='assistant' for m in stub.last_payload()['messages'])
        before=stub.call_count
        app.send_line('/support TICKET-999 вопрос')
        app.wait_for('Тикет не найден',timeout=25)
        assert stub.call_count==before
        assert app.metrics_count()==2
    assert not (tmp_path/'state/history.json').exists()
