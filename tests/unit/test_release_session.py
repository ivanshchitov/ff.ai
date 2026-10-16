"""Release commands use the common serialized facade and isolated model accounting."""
import json
import threading
from types import SimpleNamespace

import pytest
import responses

from core import config
from core.session import AssistantSession, JournalLine
from tests.unit.test_session import FakeClient


@pytest.fixture
def release_session(tmp_path, monkeypatch):
    import core.session as module
    root = tmp_path / 'project'
    (root / '.git').mkdir(parents=True)
    client = FakeClient('{"findings":[]}')
    session = AssistantSession(root, client=client, docs_retriever=None, code_retriever=None)
    calls = []
    def run_preparation(**kwargs):
        calls.append(kwargs)
        kwargs['on_stage']('release_review')
        kwargs['ask']('release_review', 'Данные текущего выпуска')
        kwargs['ask']('release_notes', 'Описание текущего выпуска')
        return {'status': 'ready_for_manual_release'}
    class ReleaseError(Exception):
        pass
    stub = SimpleNamespace(run_preparation=run_preparation, load_status=lambda output_root: None,
                           status_lines=lambda report: ('Релиз: ' + report['status'],), ReleaseError=ReleaseError)
    monkeypatch.setattr(module, 'release_preparation', stub, raising=False)
    return session, client, calls, stub


@pytest.mark.parametrize('text', ['/release', '/release status'])
def test_release_status_without_runs_does_not_request_model(release_session, text):
    session, client, _, _ = release_session
    result = session.run_command(text)
    assert not result.unknown and 'нет' in ' '.join(result.lines).lower()
    assert not client.calls and session.session_usage.requests == 0


@pytest.mark.parametrize('text', ['/release prepare', '/release foo', '/release prepare --base a --unknown b',
                                '/release prepare --base a --base b', '/release prepare --base "'])
def test_invalid_release_syntax_is_rejected_before_pipeline(release_session, text):
    session, client, calls, _ = release_session
    result = session.run_command(text)
    assert not result.unknown and '/release prepare' in ' '.join(result.lines)
    assert not calls and not client.calls


def test_prepare_accounts_isolated_requests_preserves_settings_and_reports_stages(release_session):
    session, client, calls, _ = release_session
    session.history.add('Чужая история', 'Старый ответ')
    previous = session.settings
    events = []
    session.subscribe(events.append)
    result = session.run_command('/release prepare --base "release base" --head HEAD --arch probe')
    assert result.lines == ('Релиз: ready_for_manual_release',)
    assert calls[0]['base'] == 'release base' and calls[0]['head'] == 'HEAD'
    assert calls[0]['architecture'] == 'probe'
    assert calls[0]['output_root'] == config.project_paths(session.root)['exports_dir']
    assert len(client.calls) == 2 and session.session_usage.total_tokens == 30
    assert session.history.count() == 1 and session.settings == previous
    assert 'Чужая история' not in json.dumps(client.calls, ensure_ascii=False)
    assert any(isinstance(event, JournalLine) and 'release_review' in event.text for event in events)


def test_release_status_reads_saved_result_without_model(release_session):
    session, client, _, stub = release_session
    stub.load_status = lambda output_root: {'status': 'incomplete'}
    assert session.run_command('/release status').lines == ('Релиз: incomplete',)
    assert not client.calls


def test_prepare_input_error_is_user_facing(release_session):
    session, client, _, stub = release_session
    def fail(**kwargs):
        raise stub.ReleaseError('Рабочее дерево содержит изменения')
    stub.run_preparation = fail
    result = session.run_command('/release prepare --base main')
    assert 'Рабочее дерево' in ' '.join(result.lines) and not result.unknown
    assert not client.calls


def test_release_serializes_with_other_operations(release_session):
    session, _, _, stub = release_session
    started, finish, second_done = threading.Event(), threading.Event(), threading.Event()
    def prepare(**kwargs):
        started.set()
        assert finish.wait(3)
        return {'status': 'incomplete'}
    stub.run_preparation = prepare
    first = threading.Thread(target=lambda: session.run_command('/release prepare --base main'))
    second = threading.Thread(target=lambda: (session.run_command('/context'), second_done.set()))
    first.start()
    assert started.wait(2)
    second.start()
    assert not second_done.wait(.05)
    finish.set()
    first.join(3)
    second.join(3)
    assert second_done.is_set()


def test_web_uses_same_release_command(release_session):
    from fastapi.testclient import TestClient
    from web.app import create_app
    session, client, _, _ = release_session
    with TestClient(create_app(session)) as web:
        result = web.post('/api/command', json={'text': '/release prepare --base main'})
    assert result.status_code == 200
    assert result.json()['lines'] == ['Релиз: ready_for_manual_release']
    assert len(client.calls) == 2


def test_tui_prints_release_result(release_session, recording_console):
    from ui.tui_app import DevAssistantTUI
    session, _, _, _ = release_session
    DevAssistantTUI(session=session, console=recording_console.console, typing_delay=0)._handle_command(
        '/release prepare --base main')
    assert recording_console.contains('Релиз: ready_for_manual_release')


@pytest.mark.parametrize('role', ['release_review', 'release_notes'])
@responses.activate
def test_verified_release_model_requests_strict_schema(release_session, monkeypatch, role):
    from core.api_client import APIClient
    session, _, _, _ = release_session
    monkeypatch.setattr(config, 'API_URL', config.DEFAULT_API_URL)
    session.model = 'glm-5.3-flash'
    session._agent.client = APIClient('test-key')
    responses.add(responses.POST, config.DEFAULT_API_URL,
                  json={'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]})
    session._release_ask(role, 'release data')
    payload = json.loads(responses.calls[0].request.body)
    schema = payload['response_format']['json_schema']
    assert schema['strict'] and schema['name'] == role
    assert not schema['schema']['additionalProperties']
