"""Commit-bound review must never inspect or mutate the working tree."""

import hashlib
import json
import subprocess
import os
import sys
from pathlib import Path

import pytest

from core import domains
from core.api_client import APIError, AnswerMeta


REVIEW = ('## Потенциальные баги\n'
          'transfer.cpp:L1: Проверка path.valid() удалена, недопустимый путь принят.\n'
          '## Архитектурные проблемы\nНе обнаружены.\n'
          '## Рекомендации\nВернуть проверку согласно docs/conventions.md:L1-L1.\n')


class ReviewClient:
    def __init__(self, text=REVIEW, finish='stop', error=None):
        self.text, self.finish, self.error = text, finish, error
        self.calls = []

    def ask_with_usage_messages(self, messages, **kwargs):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AnswerMeta(self.text, 'test-model', .1, 80, 20, 100, .001, self.finish)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def commit(root):
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'fixture')
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    git(root, 'init', '-q', '-b', 'demo')
    (root / 'README.md').write_text('Project structure: transfer.cpp uploads files.\n')
    (root / 'docs').mkdir()
    (root / 'docs/conventions.md').write_text('Transfer convention: validate the upload path.\n')
    (root / 'transfer.cpp').write_text('bool upload(const Path &path) { return path.valid(); }\n')
    (root / 'old name.cpp').write_text('void old_name() {}\n')
    (root / 'removed.cpp').write_text('void removed() {}\n')
    base = commit(root)
    (root / 'transfer.cpp').write_text('bool upload(const Path &path) { return true; }\n')
    (root / 'old name.cpp').rename(root / 'new name.cpp')
    (root / 'removed.cpp').unlink()
    head = commit(root)
    (root / 'transfer.cpp').write_text('UNCOMMITTED CONTENT MUST NEVER REACH MODEL\n')
    return root, base, head


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_changes_preserve_status_paths_and_committed_content(project):
    from core.pr_review import collect_changes
    root, base, head = project
    before = fingerprints(root)
    result = collect_changes(root, base, head)
    assert result.base == base and result.head == head and result.merge_base == base
    assert {(f.status, f.path, f.old_path) for f in result.files} == {
        ('M', 'transfer.cpp', ''), ('D', 'removed.cpp', ''),
        ('R100', 'new name.cpp', 'old name.cpp')}
    assert '-bool upload' in result.diff and '+bool upload' in result.diff
    assert 'UNCOMMITTED' not in result.diff
    assert fingerprints(root) == before


@pytest.mark.parametrize('revision', ['missing', '--help', 'HEAD:transfer.cpp'])
def test_invalid_revision_is_rejected(project, revision):
    from core.pr_review import ReviewError, collect_changes
    root, base, _ = project
    with pytest.raises(ReviewError):
        collect_changes(root, base, revision)


def test_snapshot_uses_head_and_never_follows_links_or_exports_secrets(project, tmp_path):
    from core.pr_review import collect_changes, prepare_snapshot
    root, base, _ = project
    (root / '.env').write_text('API_KEY=private-value\n')
    (root / 'link.cpp').symlink_to('/etc/passwd')
    (root / 'binary.cpp').write_bytes(b'\0binary')
    (root / 'secret.cpp').write_text('CLIENT_SECRET = "private-value";\n')
    head = commit(root)
    (root / 'transfer.cpp').write_text('WORK TREE\n')
    before = fingerprints(root)
    snapshot, limitations = prepare_snapshot(root, collect_changes(root, base, head),
                                             domains.load_domain('aurora-qt5').corpus,
                                             tmp_path / 'snapshot')
    assert (snapshot / 'transfer.cpp').read_text() == 'UNCOMMITTED CONTENT MUST NEVER REACH MODEL\n'
    assert not (snapshot / '.env').exists()
    assert not (snapshot / 'link.cpp').exists()
    assert not (snapshot / 'binary.cpp').exists()
    assert 'private-value' not in (snapshot / 'secret.cpp').read_text()
    assert {x['path'] for x in limitations} >= {'.env', 'link.cpp', 'binary.cpp', 'secret.cpp'}
    assert fingerprints(root) == before


def test_snapshot_fails_explicitly_on_size_limit(project, tmp_path):
    from core.pr_review import Limits, ReviewError, collect_changes, prepare_snapshot
    root, base, head = project
    with pytest.raises(ReviewError, match='предел'):
        prepare_snapshot(root, collect_changes(root, base, head),
                         domains.load_domain('aurora-qt5').corpus, tmp_path / 'snapshot',
                         Limits(max_snapshot_bytes=10))


def test_rag_delivers_documentation_and_code_with_source_lines(project, tmp_path):
    from core.pr_review import collect_changes, prepare_snapshot, retrieve_context
    root, base, head = project
    corpus = domains.load_domain('aurora-qt5').corpus
    changes = collect_changes(root, base, head)
    snapshot, _ = prepare_snapshot(root, changes, corpus, tmp_path / 'snapshot')
    sources, limitations = retrieve_context(snapshot, corpus, changes, tmp_path / 'index')
    assert {s['kind'] for s in sources} == {'code', 'documentation'}
    assert any(s['path'] == 'docs/conventions.md' for s in sources)
    assert any(s['path'] == 'transfer.cpp' and 'return true' in s['text'] for s in sources)
    assert all(s['start_line'] == 1 and s['end_line'] == 1 for s in sources)
    assert not limitations


def test_rag_names_missing_documentation(project, tmp_path):
    from core.pr_review import collect_changes, prepare_snapshot, retrieve_context
    root, base, head = project
    corpus = domains.load_domain('aurora-qt5').corpus
    changes = collect_changes(root, base, head)
    snapshot, _ = prepare_snapshot(root, changes, corpus, tmp_path / 'snapshot')
    for path in snapshot.rglob('*.md'):
        path.unlink()
    sources, limitations = retrieve_context(snapshot, corpus, changes, tmp_path / 'index')
    assert sources and all(s['kind'] == 'code' for s in sources)
    assert any('documentation' in str(x) for x in limitations)


def test_review_request_and_reports_are_grounded_and_read_only(project, tmp_path):
    from core.pr_review import run_review
    root, base, head = project
    before = fingerprints(root)
    client = ReviewClient()
    output = tmp_path / 'review'
    report = run_review(root, base, head, domains.load_domain('aurora-qt5'), output,
                        client, 'test-model')
    request = json.loads(client.calls[0][-1]['content'])
    assert request['head'] == head
    assert 'return true' in request['diff']
    assert request['changed_content']['transfer.cpp'].startswith('bool upload')
    assert 'UNCOMMITTED' not in json.dumps(request)
    assert request['sources'] and {s['kind'] for s in request['sources']} == {'code', 'documentation'}
    assert report['status'] == 'ok' and report['usage']['total_tokens'] == 100
    assert all(s['status'] == 'ok' for s in report['stages'])
    assert REVIEW in (output / 'review.md').read_text()
    assert json.loads((output / 'report.json').read_text())['head'] == head
    assert fingerprints(root) == before
    assert not (tmp_path / 'state/history.json').exists()


@pytest.mark.parametrize('client', [ReviewClient(''), ReviewClient('incomplete'),
                                  ReviewClient(REVIEW, finish='length'),
                                  ReviewClient(error=APIError('SENSITIVE upstream detail'))])
def test_bad_model_result_fails_with_report_and_no_success_review(project, tmp_path, client):
    from core.pr_review import ReviewError, run_review
    root, base, head = project
    output = tmp_path / 'review'
    with pytest.raises(ReviewError):
        run_review(root, base, head, domains.load_domain('aurora-qt5'), output,
                   client, 'test-model')
    report_text = (output / 'report.json').read_text()
    assert 'SENSITIVE' not in report_text
    report = json.loads(report_text)
    assert report['status'] == 'failed' and report['stages'][-1]['status'] == 'failed'
    assert not (output / 'review.md').exists()


def test_review_refuses_output_inside_project_and_context_overflow(project, tmp_path):
    from core.pr_review import Limits, ReviewError, run_review
    root, base, head = project
    domain, client = domains.load_domain('aurora-qt5'), ReviewClient()
    with pytest.raises(ReviewError, match='вне проекта'):
        run_review(root, base, head, domain, root / 'results', client, 'test-model')
    assert not (root / 'results').exists()
    with pytest.raises(ReviewError, match='предел'):
        run_review(root, base, head, domain, tmp_path / 'review', client, 'test-model',
                   Limits(max_context_chars=20))
    assert not client.calls


def test_review_redacts_credentials_in_diff_context_and_model_output(project, tmp_path):
    from core.pr_review import run_review
    root, base, _ = project
    (root / 'secret.cpp').write_text('const char *CLIENT_SECRET = "credential-fixture";\n')
    (root / '.env').write_text('DO_NOT_SEND=credential-fixture\n')
    head = commit(root)
    client = ReviewClient(REVIEW + '\nCLIENT_SECRET = "credential-fixture";')
    output = tmp_path / 'review'
    run_review(root, base, head, domains.load_domain('aurora-qt5'), output, client, 'test-model')
    assert 'credential-fixture' not in json.dumps(client.calls)
    assert 'DO_NOT_SEND' not in json.dumps(client.calls)
    assert 'credential-fixture' not in (output / 'review.md').read_text()


def test_empty_sections_and_reused_output_are_rejected(project, tmp_path):
    from core.pr_review import ReviewError, run_review
    root, base, head = project
    domain, output = domains.load_domain('aurora-qt5'), tmp_path / 'review'
    with pytest.raises(ReviewError):
        run_review(root, base, head, domain, output,
                   ReviewClient('## Потенциальные баги\n## Архитектурные проблемы\n## Рекомендации\n'),
                   'test-model')
    before = (output / 'report.json').read_bytes()
    with pytest.raises(ReviewError):
        run_review(root, base, head, domain, output, ReviewClient(), 'test-model')
    assert (output / 'report.json').read_bytes() == before


def test_cli_runs_real_http_pipeline_and_returns_failure_code(project, tmp_path):
    from tests.e2e.stub_api import StubAPI, answer
    root, base, head = project
    entry = Path(__file__).resolve().parents[2] / 'ff-ai-review.py'
    api = StubAPI().always(answer(REVIEW))
    api_url = api.start()
    try:
        env = dict(os.environ, OPENCODE_API_KEY='test-key', OPENCODE_API_URL=api_url,
                   FFAI_NO_REEXEC='1', FFAI_LLAMA_AUTOSTART='0')
        output = tmp_path / 'cli-review'
        args = [sys.executable, str(entry), '--repo', str(root), '--domain', 'aurora-qt5',
                '--base', base, '--head', head, '--output', str(output)]
        result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=30)
        assert result.returncode == 0, result.stderr
        assert (output / 'review.md').exists()
        assert json.loads((output / 'report.json').read_text())['status'] == 'ok'
        from tests.e2e.stub_api import Reply
        api.always(Reply(raw_body='{"choices":[{"message":{"content":null}}]}'))
        malformed_output = tmp_path / 'malformed-review'
        malformed = subprocess.run([*args[:-1], str(malformed_output)], capture_output=True,
                                   text=True, env=env, timeout=30)
        assert malformed.returncode == 1 and 'Traceback' not in malformed.stderr
        assert json.loads((malformed_output / 'report.json').read_text())['status'] == 'failed'
        result = subprocess.run([*args[:-1], str(tmp_path / 'bad-review'), '--head', 'missing'],
                                capture_output=True, text=True, env=env, timeout=30)
        assert result.returncode == 1
        assert json.loads((tmp_path / 'bad-review/report.json').read_text())['status'] == 'failed'
    finally:
        api.stop()


def test_partial_key_diff_is_redacted_and_snapshot_preserves_original_lines(project, tmp_path):
    from core.pr_review import run_review
    root, _, _ = project
    key = ('-----BEGIN PRIVATE KEY-----\n' + '\n'.join(f'SENSITIVE_BODY_{n}' for n in range(20))
           + '\n-----END PRIVATE KEY-----\nvoid after_key() {}\n')
    (root / 'key.conf').write_text(key)
    base = commit(root)
    (root / 'key.conf').write_text(key.replace('SENSITIVE_BODY_10', 'NEW_SENSITIVE_BODY_10'))
    head = commit(root)
    client = ReviewClient()
    run_review(root, base, head, domains.load_domain('aurora-qt5'), tmp_path / 'review', client, 'test-model')
    request = json.loads(client.calls[-1][-1]['content'])
    assert 'SENSITIVE_BODY' not in json.dumps(request)
    assert len(request['changed_content']['key.conf'].splitlines()) == 23
    assert request['changed_content']['key.conf'].splitlines()[22] == 'void after_key() {}'


def test_subdirectory_input_cannot_write_elsewhere_in_repository(project):
    from core.pr_review import ReviewError, run_review
    root, base, head = project
    with pytest.raises(ReviewError, match='корн'):
        run_review(root / 'docs', base, head, domains.load_domain('aurora-qt5'),
                   root / 'results', ReviewClient(), 'test-model')
    assert not (root / 'results').exists()


@pytest.mark.parametrize('client', [ReviewClient(None), ReviewClient(error=ValueError('SENSITIVE malformed API'))])
def test_malformed_api_content_records_failed_stage(project, tmp_path, client):
    from core.pr_review import ReviewError, run_review
    root, base, head = project
    output = tmp_path / 'review'
    with pytest.raises(ReviewError):
        run_review(root, base, head, domains.load_domain('aurora-qt5'), output, client, 'test-model')
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed' and report['stages'][-1]['status'] == 'failed'
    assert 'SENSITIVE' not in json.dumps(report)


def test_diff_separates_changed_lines_without_final_newline(project, tmp_path):
    from core.pr_review import run_review
    root, _, _ = project
    (root / 'transfer.cpp').write_text('return false;')
    base = commit(root)
    (root / 'transfer.cpp').write_text('return true;')
    head = commit(root)
    client = ReviewClient()
    run_review(root, base, head, domains.load_domain('aurora-qt5'), tmp_path / 'review', client, 'test-model')
    diff = json.loads(client.calls[-1][-1]['content'])['diff']
    assert '-return false;\n' in diff and '+return true;\n' in diff
    assert '\\ No newline at end of file' in diff


def test_review_receives_file_mode_changes(project, tmp_path):
    from core.pr_review import run_review
    root, _, _ = project
    script = root / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(0o755)
    base = commit(root)
    script.chmod(0o644)
    head = commit(root)
    client = ReviewClient()
    run_review(root, base, head, domains.load_domain('aurora-qt5'), tmp_path / 'review', client, 'test-model')
    diff = json.loads(client.calls[-1][-1]['content'])['diff']
    assert '100755' in diff and '100644' in diff


def test_documentation_quota_is_not_consumed_by_one_long_readme(project, tmp_path):
    from core.pr_review import collect_changes, prepare_snapshot, retrieve_context
    root, base, _ = project
    (root / 'README.md').write_text(('Project structure: upload transfer path valid operation.\n' * 500))
    head = commit(root)
    corpus = domains.load_domain('aurora-qt5').corpus
    changes = collect_changes(root, base, head)
    snapshot, _ = prepare_snapshot(root, changes, corpus, tmp_path / 'snapshot')
    sources, _ = retrieve_context(snapshot, corpus, changes, tmp_path / 'index')
    documents = [s for s in sources if s['kind'] == 'documentation']
    assert any(s['path'] == 'docs/conventions.md' for s in documents)
    assert len([s for s in documents if s['path'] == 'README.md']) == 1
