"""Task requests receive discovered files, fresh contents and their own changes only."""
from pathlib import Path
from core import domains, code_index, code_retrieval
from core.task_context import TaskContextProvider
from core.project_changes import ProjectChanges

DOMAIN = domains.load_domain('aurora-qt5')


def test_goal_without_a_path_discovers_source_and_local_conventions(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'authorization.cpp').write_text('bool authorize() { return false; }\n')
    (root / 'README.md').write_text('Use the existing build system.\n')
    database = tmp_path / 'index.sqlite3'
    code_index.build_index(root, DOMAIN.corpus, 'structural', database)
    retrieval = code_retrieval.CodeRetriever(database, DOMAIN.corpus)
    provider = TaskContextProvider(root, DOMAIN, retrieve=lambda q: retrieval.search(q, mode='baseline'))
    context = provider.collect('execution', 'fix authorize')
    assert 'authorization.cpp' in context.paths
    assert 'return false' in '\n'.join(m['content'] for m in context.messages)
    assert 'README.md' in context.paths
    assert context.fingerprints['authorization.cpp']


def test_next_stage_reads_new_text_and_validation_uses_record_paths(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'authorization.cpp').write_text('old\n')
    (root / 'unrelated.cpp').write_text('manual change\n')
    service = ProjectChanges(root, tmp_path / 'changes')
    record = service.prepare('--- a/authorization.cpp\n+++ b/authorization.cpp\n@@ -1 +1 @@\n-old\n+new\n', 1)
    service.apply(record.id, lambda *_: True)
    record = service.get(record.id)
    provider = TaskContextProvider(root, DOMAIN, stale=lambda: True)
    context = provider.collect('validation', 'check work', changes=(record,))
    text = '\n'.join(m['content'] for m in context.messages)
    assert 'new' in text and 'authorization.cpp' in context.paths
    assert 'manual change' not in text
    assert 'unrelated.cpp' not in context.paths
    assert any('stale' in note for note in context.notes)


def test_stale_index_is_not_used_but_current_files_remain_available(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'source.cpp').write_text('current content\n')
    provider = TaskContextProvider(root, DOMAIN, retrieve=lambda _: (_ for _ in ()).throw(AssertionError('stale retrieval')),
                                   stale=lambda: True)
    context = provider.collect('execution', 'update current content')
    assert 'current content' in '\n'.join(m['content'] for m in context.messages)


def test_read_flow_discovers_file_not_named_by_goal(tmp_path):
    from types import SimpleNamespace
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'backend.cpp').write_text('bool ready() { return false; }\n')
    flow = lambda _: (SimpleNamespace(arguments={'path':'backend.cpp'}, text='backend.cpp:1:bool ready()'),)
    context = TaskContextProvider(root, DOMAIN, read_flow=flow).collect('planning', 'исправить авторизацию')
    assert 'backend.cpp' in context.paths
    assert 'return false' in '\n'.join(m['content'] for m in context.messages)


def test_tool_read_range_reaches_task_beyond_file_prefix(tmp_path):
    from types import SimpleNamespace
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'backend.cpp').write_text(('// long header ' + 'x'*100 + '\n')*80 + 'bool TAIL_MARKER() { return false; }\n')
    flow = lambda _: (SimpleNamespace(arguments={'path':'backend.cpp', 'start_line':81, 'end_line':81}, text='81: bool TAIL_MARKER() { return false; }'),)
    context = TaskContextProvider(root, DOMAIN, read_flow=flow).collect('execution', 'fix authorization')
    text = '\n'.join(m['content'] for m in context.messages)
    assert 'TAIL_MARKER' in text
    assert '81:' in text


def test_read_only_mcp_evidence_reaches_model_with_provenance(tmp_path):
    from types import SimpleNamespace
    flow = lambda _: (SimpleNamespace(server='platform-docs', tool='get_document',
        arguments={'query':'API'}, text='DOC_EVIDENCE_MARKER https://docs.example/api'),)
    context = TaskContextProvider(tmp_path, DOMAIN, read_flow=flow).collect('planning', 'fix API')
    text = '\n'.join(m['content'] for m in context.messages)
    assert 'DOC_EVIDENCE_MARKER' in text
    assert 'platform-docs' in text and 'get_document' in text


def test_tool_constraints_are_not_dropped_when_local_files_fill_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core import config
    (tmp_path / 'large.cpp').write_text('x'*10000)
    monkeypatch.setattr(config, 'TOOL_FLOW_CONTEXT_CHARS', 1000)
    monkeypatch.setattr(config, 'CODE_FRAGMENT_MAX_CHARS', 1000)
    flow = lambda _: (SimpleNamespace(server='docs', tool='read', arguments={}, text='MCP_REQUIRED_CONSTRAINT'),)
    context = TaskContextProvider(tmp_path, DOMAIN, read_flow=flow).collect('planning', 'update project')
    text = '\n'.join(m['content'] for m in context.messages)
    assert 'MCP_REQUIRED_CONSTRAINT' in text


def test_prior_patch_does_not_consume_fresh_tool_evidence_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core import config
    monkeypatch.setattr(config, 'TOOL_FLOW_CONTEXT_CHARS', 1000)
    change = SimpleNamespace(id='prior-change', task_number=1, status='applied', paths=(), patch='x'*2000)
    flow = lambda _: (SimpleNamespace(server='docs', tool='read', arguments={}, text='MCP_REQUIRED_CONSTRAINT'),)
    context = TaskContextProvider(tmp_path, DOMAIN, read_flow=flow).collect('execution', 'update project', changes=(change,))
    text = '\n'.join(m['content'] for m in context.messages)
    assert 'MCP_REQUIRED_CONSTRAINT' in text
