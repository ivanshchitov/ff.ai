import json
import subprocess
import pytest
from core import code_index, config, domains
from core.api_client import APIError
from core.mcp_client import MCPCallResult
from core.session import AssistantSession
from mcp_server.support_data import get_ticket, SupportDataError
from tests.unit.test_session import FakeClient

TEXT = 'Для обычного сервера необходима полная ссылка к WebDAV-серверу, имя пользователя и пароль.'
ANSWER = f'Используйте полную ссылку. Цитата: «{TEXT}» Источник: FAQ.md:L1-L1'

@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    subprocess.run(['git', 'init'], cwd=root, capture_output=True, check=True)
    (root / 'FAQ.md').write_text(TEXT + '\n')
    (root / 'support.json').write_text(json.dumps({'schema':1,'users':[{'id':'U-1','name':'Первый'},{'id':'U-2','name':'Второй'}],
        'tickets':[{'id':'T-1','user_id':'U-1','provider':'Alpha','description':'Первое обращение'}, {'id':'T-2','user_id':'U-2','provider':'Beta','description':'Второе обращение'}]}))
    class TicketClient:
        def __init__(self, spec): pass
        def call_tool(self, tool, args):
            try: return MCPCallResult('support', tool, args, json.dumps(get_ticket(root,'support.json',args['ticket_id']),ensure_ascii=False))
            except SupportDataError as e: return MCPCallResult('support', tool, args, str(e), is_error=True)
    domain = domains.load_domain('product-support')
    code_index.build_index(root, domain.corpus, 'fixed', config.repo_index_file(root))
    model = FakeClient(ANSWER)
    session = AssistantSession(root, domain_id=domain.id, client=model, docs_retriever=None, mcp_client_factory=TicketClient)
    return root, session, model

def test_support_is_grounded_isolated_and_read_only(setup):
    root, session, model = setup
    session.history.add('Чужая история', 'Чужой ответ')
    session.run_command('/rag-code mode off')
    before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    for ticket in ['T-1','T-2']:
        result = session.run_command(f'/support {ticket} полная ссылка WebDAV сервер')
        assert result.answer.text == ANSWER
        assert 'MCP' in '\n'.join(result.lines)
    messages = '\n'.join(m['content'] for m in model.calls[-1]['messages'])
    assert 'Второе обращение' in messages and 'FAQ.md:L1' in messages
    assert 'Первое обращение' not in messages and 'Чужая история' not in messages
    assert session.history.count() == 1 and not session.task_state.tasks
    assert not session.code_enabled and session.session_usage.total_tokens > 0
    assert before == {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert len(model.calls) == 2

@pytest.mark.parametrize('command', ['/support','/support T-1','/support missing вопрос'])
def test_invalid_input_stops_before_model(setup, command):
    _, session, model = setup
    result = session.run_command(command)
    assert not result.unknown and result.answer is None and result.lines
    assert model.calls == []

def test_source_refresh_and_stale_corpus(setup):
    root, session, model = setup
    path = root / 'support.json'
    data = json.loads(path.read_text());data['tickets'][0]['description']='Свежая запись';path.write_text(json.dumps(data))
    (root / 'FAQ.md').write_text('Изменённый документ\n')
    result = session.run_command('/support T-1 полная ссылка')
    assert any('stale' in line for line in result.lines)
    assert 'Свежая запись' in str(model.calls[-1])
    assert TEXT not in str(model.calls[-1])

def test_api_failure_preserves_settings_and_history(setup):
    _, session, _ = setup
    class Failed(FakeClient):
        def ask_with_usage_messages(self, *a, **k): raise APIError('offline')
    session._agent.client=Failed()
    session.run_command('/rag-code mode off')
    with pytest.raises(APIError, match='offline'):
        session.run_command('/support T-1 полная ссылка')
    assert not session.code_enabled and session.history.count()==0

def test_support_web_sources_and_metrics(setup):
    from fastapi.testclient import TestClient
    from web.app import create_app
    _, session, _ = setup
    with TestClient(create_app(session)) as client:
        response = client.post('/api/command', json={'text':'/support T-1 полная ссылка WebDAV сервер'})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == ANSWER and body['meta']['total_tokens'] == 15
    assert {s['identifier'] for s in body['sources']} == {'FAQ.md:L1-L1'}
    assert not body['task_run_started']

def test_support_tui_sources_and_metrics(setup, recording_console, monkeypatch):
    from ui.tui_app import DevAssistantTUI
    monkeypatch.setenv('OPENCODE_API_KEY','sk-test-key')
    _, session, _ = setup
    DevAssistantTUI(session=session, console=recording_console.console, typing_delay=0)._handle_command('/support T-1 полная ссылка WebDAV сервер')
    assert recording_console.contains('Используйте полную ссылку')
    assert recording_console.contains('FAQ.md:L1')
    assert recording_console.contains('Токены: 10+5=15')

@pytest.mark.parametrize('state', ['missing','empty'])
def test_unavailable_corpus_is_explicit(setup, state):
    root, session, model = setup
    if state=='missing': config.repo_index_file(root).unlink()
    question = 'несуществующийтермин' if state=='empty' else 'полная ссылка'
    # Remove diagnostic terms so this query genuinely has no lexical matches.
    data = json.loads((root/'support.json').read_text());data['tickets'][0]['provider']='Alpha'
    (root/'support.json').write_text(json.dumps(data))
    result=session.run_command('/support T-1 '+question)
    assert any('Локальный корпус:' in line for line in result.lines)
    assert 'Назови ограничение' in str(model.calls[-1])

def test_no_support_in_development_domain(setup):
    root, _, model=setup
    session=AssistantSession(root,domain_id='aurora-qt5',client=model,docs_retriever=None)
    result=session.run_command('/support T-1 вопрос')
    assert result.answer is None and not model.calls

@pytest.mark.parametrize('command', ['/support','/support T-1','/support missing вопрос'])
def test_tui_invalid_support_does_not_request_api_credentials(setup, recording_console, monkeypatch, command):
    from ui.tui_app import DevAssistantTUI
    _,session,_=setup
    session._agent.client=None
    monkeypatch.delenv('OPENCODE_API_KEY',raising=False)
    tui=DevAssistantTUI(session=session,console=recording_console.console,typing_delay=0)
    def forbidden_prompt(): raise AssertionError('Invalid support command requested credentials')
    monkeypatch.setattr(tui,'_ensure_api_key',forbidden_prompt)
    tui._handle_command(command)
    assert recording_console.contains('Синтаксис') or recording_console.contains('Тикет не найден')

def test_previous_tool_and_memory_reports_not_attached_to_support(setup):
    from types import SimpleNamespace
    _,session,_=setup
    session._agent._last_tool_flow=SimpleNamespace(secret_previous_result='previous ticket')
    session._agent._last_routing=(SimpleNamespace(key='old user'),)
    session.run_command('/support T-1 полная ссылка WebDAV сервер')
    assert session.tool_flow_report() is None
    assert session._agent._last_routing==()

def test_support_trace_describes_completed_search_after_settings_restored(setup):
    _, session, _ = setup
    session.run_command('/rag-code retrieval enhanced')
    session.run_command('/rag-code mode off')
    session.run_command('/support T-1 полная ссылка WebDAV сервер')
    assert session._agent.config.code_retrieval == 'enhanced'
    assert not session.code_enabled
    trace = '\n'.join(session.code_trace_lines())
    assert 'Режим отбора: baseline' in trace
    assert 'FAQ.md:L1' in trace
