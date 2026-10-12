"""Project help grounds read-only answers in local documents and fresh MCP data."""

import subprocess

import pytest

from core import code_index, code_retrieval, config, domains
from core.api_client import APIError
from core.mcp_client import MCPCallResult, MCPError
from core.session import AssistantSession, JournalLine
from tests.unit.test_session import FakeClient

GROUNDED_ANSWER = ('Клиент находится в app.\n'
                   'Цитата: «Project structure: app contains the client.»\n'
                   'Источник: README.md:L1-L1')


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    subprocess.run(['git', 'init', '-b', 'demo'], cwd=root, check=True, capture_output=True)
    (root / 'README.md').write_text('Project structure: app contains the client.\n')
    (root / 'docs').mkdir()
    (root / 'docs' / 'design.md').write_text('Project structure: library contains WebDAV logic.\n')
    return root


def session_for(root, *, indexed=True, factory=None, client=None, strategy='fixed'):
    corpus = domains.load_domain('aurora-qt5').corpus
    database = config.repo_index_file(root)
    if indexed:
        code_index.build_index(root, corpus, strategy, database)
    session = AssistantSession(root, domain_id='aurora-qt5', client=client or FakeClient(GROUNDED_ANSWER),
                               docs_retriever=None,
                               code_retriever=code_retrieval.CodeRetriever(database, corpus),
                               mcp_client_factory=factory)
    session.run_command('/rag-code retrieval baseline')
    return session


class GitClient:
    def __init__(self, spec):
        self.spec = spec

    def call_tool(self, tool, arguments):
        assert tool == 'repo_git'
        assert arguments == {'args': 'status --short --branch'}
        assert '--root' in self.spec.args
        return MCPCallResult('repo', tool, arguments, 'git status --short --branch\n## demo\n')


@pytest.mark.parametrize('command', ['/help', '/help Project structure: app library'])
def test_help_delivers_docs_and_git_without_development(project, command):
    client = FakeClient(GROUNDED_ANSWER)
    session = session_for(project, factory=GitClient, client=client)
    session.run_command('/rag-code mode off')
    events = []
    session.subscribe(events.append)
    before = {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()}
    result = session.run_command(command)
    assert not result.unknown
    assert result.answer.text == GROUNDED_ANSWER
    messages = '\n'.join(m['content'] for m in client.calls[-1]['messages'])
    assert 'README.md:L1' in messages
    assert 'docs/design.md:L1' in messages
    assert '## demo' in messages
    assert session.history.count() == 1
    assert session.session_usage.total_tokens > 0
    assert not session.task_state.tasks
    assert not result.task_run
    assert not session.code_enabled
    assert any('repo_git' in e.text and 'demo' in e.text for e in events if isinstance(e, JournalLine))
    assert before == {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()}


@pytest.mark.parametrize('state', ['missing', 'stale', 'corrupt'])
def test_help_names_unavailable_index(project, state):
    client = FakeClient()
    session = session_for(project, indexed=state != 'missing', factory=GitClient, client=client)
    if state == 'stale':
        session._code_retriever.invalidate()
    if state == 'corrupt':
        config.repo_index_file(project).write_bytes(b'broken database')
    result = session.run_command('/help Project structure')
    assert not result.unknown
    assert any('/rag-code index' in line for line in result.lines)
    messages = '\n'.join(m['content'] for m in client.calls[-1]['messages'])
    assert 'README.md:L1' not in messages
    assert 'без источников' in messages


@pytest.mark.parametrize('failure', ['exception', 'tool_error'])
def test_help_names_mcp_failure(project, failure):
    class FailedClient(GitClient):
        def call_tool(self, tool, arguments):
            if failure == 'exception':
                raise MCPError('offline')
            return MCPCallResult('repo', tool, arguments, 'offline', is_error=True)
    client = FakeClient()
    result = session_for(project, factory=FailedClient, client=client).run_command('/help')
    assert not result.unknown
    assert any('MCP' in line and 'offline' in line for line in result.lines)
    assert 'offline' in '\n'.join(m['content'] for m in client.calls[-1]['messages'])


def test_help_restores_settings_on_api_error(project):
    class FailedModel(FakeClient):
        def ask_with_usage_messages(self, *args, **kwargs):
            raise APIError('model offline')
    session = session_for(project, factory=GitClient, client=FailedModel())
    session.run_command('/rag-code mode off')
    session.run_command('/rag-docs mode on')
    session.run_command('/tool auto on')
    with pytest.raises(APIError, match='model offline'):
        session.run_command('/help change the project')
    assert not session.code_enabled
    assert session.docs_enabled
    assert session._agent.config.auto_tools
    assert not session.task_state.tasks


def test_help_uses_real_repository_mcp(project):
    result = session_for(project).run_command('/help Project structure')
    assert not result.unknown
    assert any('demo' in line for line in result.lines)


def test_help_renders_answer_and_metrics_in_tui(project, recording_console, monkeypatch):
    from ui.tui_app import DevAssistantTUI
    monkeypatch.setenv('OPENCODE_API_KEY', 'sk-test-key')
    session = session_for(project, factory=GitClient)
    tui = DevAssistantTUI(session=session, console=recording_console.console, typing_delay=0)
    tui._handle_command('/help Project structure')
    assert recording_console.contains('Клиент находится в app.')
    assert recording_console.contains('docs/design.md:L1')
    assert recording_console.contains('Токены: 10+5=15')


def test_help_returns_answer_sources_and_metrics_in_web(project):
    from fastapi.testclient import TestClient
    from web.app import create_app
    session = session_for(project, factory=GitClient)
    with TestClient(create_app(session)) as client:
        response = client.post('/api/command', json={'text': '/help Project structure'})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == GROUNDED_ANSWER
    assert body['meta']['total_tokens'] == 15
    assert {s['identifier'] for s in body['sources']} == {'README.md:L1-L1', 'docs/design.md:L1-L1'}
    assert not body['task_run_started']


def test_help_modification_request_stays_read_only(project):
    session = session_for(project, factory=GitClient)
    before = (project / 'README.md').read_bytes()
    result = session.run_command('/help Удали README и измени исходники')
    assert result.answer is not None
    assert not result.task_run
    assert not session.task_state.tasks
    assert (project / 'README.md').read_bytes() == before


def test_help_preserves_real_detached_head(project):
    subprocess.run(['git', 'add', 'README.md'], cwd=project, check=True, capture_output=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], cwd=project, check=True, capture_output=True)
    subprocess.run(['git', 'checkout', '--detach'], cwd=project, check=True, capture_output=True)
    client = FakeClient(GROUNDED_ANSWER)
    result = session_for(project, client=client).run_command('/help Project structure')
    assert result.answer is not None
    messages = '\n'.join(m['content'] for m in client.calls[-1]['messages'])
    assert 'HEAD' in messages
    assert '## demo' not in messages


@pytest.mark.parametrize('edit', ['overwrite', 'append', 'delete', 'add'])
def test_help_detects_external_document_changes(project, edit):
    client = FakeClient(GROUNDED_ANSWER)
    session = session_for(project, factory=GitClient, client=client)
    path = project / 'README.md'
    if edit == 'overwrite':
        path.write_text('Changed structure: other app.\n')
    elif edit == 'append':
        with path.open('a') as file:
            file.write('New structure chapter.\n')
    elif edit == 'delete':
        path.unlink()
    else:
        (project / 'docs' / 'new.md').write_text('New structure chapter.\n')
    result = session.run_command('/help Project structure')
    assert any('stale' in line for line in result.lines)
    assert not session.code_report().fragments
    messages = '\n'.join(m['content'] for m in client.calls[-1]['messages'])
    assert 'Project structure: app contains the client.' not in messages


def test_help_mcp_status_does_not_refresh_git_index(project):
    subprocess.run(['git', 'add', 'README.md'], cwd=project, check=True, capture_output=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], cwd=project, check=True, capture_output=True)
    index = project / '.git/index'
    before = index.read_bytes()
    # Force a stat mismatch without modifying file contents.
    import os
    stat = (project / 'README.md').stat()
    os.utime(project / 'README.md', ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    session_for(project).run_command('/help Project structure')
    assert index.read_bytes() == before


def test_help_structural_index_keeps_unchanged_markdown_available(project):
    session = session_for(project, factory=GitClient, strategy='structural')
    result = session.run_command('/help Project structure')
    assert not any('stale' in line for line in result.lines)
    assert {f.identifier for f in session.code_report().fragments} == {
        'README.md:L1-L1', 'docs/design.md:L1-L1'}
    assert result.answer.text == GROUNDED_ANSWER
