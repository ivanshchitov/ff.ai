"""Real facade, journal, stdio MCP and both interfaces; model double, no TCP listener."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient
from core import config, domains, code_index, code_retrieval
from core.api_client import AnswerMeta
from core.session import AssistantSession
from ui import keyboard
from ui.tui_app import DevAssistantTUI
from web.app import create_app
from tests.conftest import rerank_answer

PATCH = '--- a/source.cpp\n+++ b/source.cpp\n@@ -1 +1 @@\n-bool READY_MARKER() { return false; }\n+bool READY_MARKER() { return true; }\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-READY_MARKER is disabled.\n+READY_MARKER is enabled.\n'


class DevelopmentModel:
    def __init__(self):
        self.calls = []

    def ask_with_usage_messages(self, messages, **kwargs):
        self.calls.append(messages)
        first, last = messages[0]['content'], messages[-1]['content']
        rated = rerank_answer(messages, .9)
        if first.startswith('Классифицируй'):
            request = json.loads(last)['request']
            kind = 'investigate' if 'не меняй' in request else 'develop'
            content = json.dumps({'kind': kind, 'goal': request if kind == 'develop' else '', 'clarification': ''})
        elif first.startswith('Выбери инструменты'):
            content = json.dumps({'steps': [
                {'tool':'repo_search', 'arguments':{'query':'READY_MARKER'}},
                {'tool':'repo_read', 'arguments':{'path':'source.cpp'}},
                {'tool':'repo_read', 'arguments':{'path':'README.md'}},
            ]})
        elif last.startswith('Ты планируешь'):
            content = '{"items":["Enable the readiness flag and document it"]}'
        elif last.startswith('Ты выполняешь'):
            evidence = '\n'.join(m['content'] for m in messages[:-1])
            assert 'bool READY_MARKER() { return false; }' in evidence
            assert 'READY_MARKER is disabled.' in evidence
            content = PATCH
        elif last.startswith('Ты проверяешь'):
            evidence = '\n'.join(m['content'] for m in messages[:-1])
            assert 'bool READY_MARKER() { return true; }' in evidence
            assert 'READY_MARKER is enabled.' in evidence
            content = '{"issues":[]}'
        else:
            content = 'Исследование выполнено без изменений.'
        return AnswerMeta(content=content, model='development-double', elapsed_seconds=.01,
                          prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0)


@pytest.fixture
def development_project(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    subprocess.run(['git','init','-q',str(root)], check=True)
    (root / 'source.cpp').write_text('bool READY_MARKER() { return false; }\n')
    (root / 'README.md').write_text('READY_MARKER is disabled.\n')
    # A staged file belonging to the user must survive the complete agent cycle.
    (root / 'manual.txt').write_text('user work\n')
    subprocess.run(['git','-C',str(root),'add','manual.txt'], check=True)
    monkeypatch.delenv('FFAI_MCP_URL', raising=False)
    monkeypatch.setenv('FFAI_MCP_COMMAND', sys.executable)
    monkeypatch.setenv('FFAI_MCP_ARGS', str(config.BASE_DIR / 'mcp_server/repo_server.py') + ' --root ' + str(root) + ' --tools ' + str(config.BASE_DIR / 'domains/aurora-qt5/tools.json'))
    domain = domains.load_domain('aurora-qt5')
    code_index.build_index(root, domain.corpus, 'structural', config.repo_index_file(root))
    model = DevelopmentModel()
    def new_session():
        session = AssistantSession(root=root, client=model, docs_retriever=None,
            code_retriever=code_retrieval.CodeRetriever(config.repo_index_file(root), domain.corpus))
        session._agent.config.auto_tools = True
        return session
    return root, model, new_session


def wait_for(predicate, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError('development lifecycle timed out')


def staged(root):
    return subprocess.run(['git','-C',str(root),'diff','--cached'], check=True, capture_output=True).stdout


def assert_original(root):
    assert (root / 'source.cpp').read_text() == 'bool READY_MARKER() { return false; }\n'
    assert (root / 'README.md').read_text() == 'READY_MARKER is disabled.\n'


def test_tui_chat_read_plan_confirm_validate_restart_and_undo(development_project, monkeypatch, recording_console):
    root, model, new_session = development_project
    before = staged(root)
    session = new_session()
    tui = DevAssistantTUI(session, console=recording_console.console, typing_delay=0)
    monkeypatch.setattr(keyboard, 'raw_mode', lambda: nullcontext())
    monkeypatch.setattr(keyboard, 'read_char', lambda: 'y')
    monkeypatch.setattr(tui, '_drain_pause_key', lambda: '')
    monkeypatch.setattr(tui, '_ask_plan_edits', lambda _: '')
    tui._ask('Включи readiness и обнови описание')
    task = session.task_state.current
    assert task.status == 'done'
    change_id = task.patches[0].change_id
    assert change_id and recording_console.contains(change_id)
    assert recording_console.contains('Патч к применению')
    assert any('Источник MCP' in m['content'] for call in model.calls for m in call)
    assert session._code_retriever.stale_reason
    restarted = new_session()
    undo_tui = DevAssistantTUI(restarted, console=recording_console.console, typing_delay=0)
    undo_tui._handle_command('/changes undo ' + change_id)
    assert_original(root)
    assert staged(root) == before
    assert restarted.task_paused and restarted.task_state.current.patches[0].rolled_back
    assert restarted.task_step() is None


def test_web_chat_confirmation_replay_restart_and_undo(development_project):
    root, model, new_session = development_project
    before = staged(root)
    session = new_session()
    app = create_app(session, confirmation_timeout=10)
    events = app.state.bus.subscribe()
    with TestClient(app) as client:
        request = {'question':'Включи readiness и обнови описание', 'request_id':'web-development'}
        first = client.post('/api/ask', json=request).json()
        assert first['kind'] == 'task' and first['task_run']
        wait_for(lambda: session.task_state.current.awaiting_edits)
        assert client.post('/api/task', json={'action':'edits', 'text':''}).json()['accepted']
        pending = wait_for(lambda: app.state.confirmations.pending())
        assert_original(root)
        assert pending[0]['task_number'] == first['task_number']
        replay = client.post('/api/ask', json=request).json()
        assert replay['task_number'] == first['task_number'] and not replay['task_run']
        assert client.post('/api/ask', json={'question':'Другая цель','request_id':'another'}).json()['kind'] == 'busy'
        assert client.post('/api/confirm', json={'id':pending[0]['id'], 'approved':True}).status_code == 200
        wait_for(lambda: not app.state.runner.running)
        task = session.task_state.current
        assert task.status == 'done'
        change_id = task.patches[0].change_id
        snapshot = client.get('/api/reports/task').json()['data']['tasks'][0]
        assert snapshot['plan'][0]['change_id'] == change_id
        assert snapshot['changes'][0]['validation']
        assert snapshot['index_stale']
        replay = client.post('/api/ask', json=request).json()
        assert replay['task_number'] == first['task_number'] and not replay['task_run']
    restarted = new_session()
    restarted_app = create_app(restarted, confirmation_timeout=10)
    undo_events = restarted_app.state.bus.subscribe()
    with TestClient(restarted_app) as client:
        replay = client.post('/api/ask', json=request).json()
        assert replay['task_number'] == first['task_number'] and not replay['task_run']
        response = []
        thread = threading.Thread(target=lambda: response.append(client.post('/api/command', json={'text':'/changes undo ' + change_id})))
        thread.start()
        pending = wait_for(lambda: restarted_app.state.confirmations.pending())
        assert client.post('/api/confirm', json={'id':pending[0]['id'], 'approved':True}).status_code == 200
        thread.join(15)
        assert not thread.is_alive() and response[0].status_code == 200
        assert_original(root)
        assert staged(root) == before
        assert restarted.task_paused
        assert restarted.task_state.current.patches[0].rolled_back
        snapshots = []
        while not undo_events.empty():
            event = undo_events.get()
            if event and event['type'] == 'task':
                snapshots.append(event['data'])
        assert snapshots[-1]['paused']
        assert snapshots[-1]['tasks'][0]['plan'][0]['rolled_back']
    event_types = []
    while not events.empty():
        event = events.get()
        if event is not None:
            event_types.append(event['type'])
    assert 'confirmation_request' in event_types and 'task' in event_types
    app.state.bus.unsubscribe(events)


@pytest.mark.parametrize('decision', ['refuse', 'timeout'])
def test_web_refusal_or_timeout_preserves_the_project(development_project, decision):
    root, _, new_session = development_project
    session = new_session()
    app = create_app(session, confirmation_timeout=.5 if decision == 'timeout' else 10)
    with TestClient(app) as client:
        client.post('/api/ask', json={'question':'Включи readiness', 'request_id':decision})
        wait_for(lambda: session.task_state.current.awaiting_edits)
        client.post('/api/task', json={'action':'edits', 'text':''})
        pending = wait_for(lambda: app.state.confirmations.pending())
        if decision == 'refuse':
            client.post('/api/confirm', json={'id':pending[0]['id'], 'approved':False})
        wait_for(lambda: not app.state.runner.running)
        assert_original(root)
        assert all(not p.applied for p in session.task_state.current.patches)


def test_investigation_never_creates_a_task_or_changes_files(development_project):
    root, _, new_session = development_project
    session = new_session()
    app = create_app(session)
    with TestClient(app) as client:
        result = client.post('/api/ask', json={'question':'Исследуй readiness, ничего не меняй'}).json()
        assert result['kind'] == 'answer'
        assert not session.task_state.tasks
        assert not session._changes.list()
        assert_original(root)
