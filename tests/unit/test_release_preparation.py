"""Release preparation is tied to immutable commits and never writes to its source."""
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core import domains


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def commit(root):
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    git(root, 'init', '-q', '-b', 'test')
    (root / 'app.cpp').write_text('bool validate() { return true; }\n')
    (root / 'README.md').write_text('Validate input before upload.\n')
    (root / 'package.meta').write_text('Name: example\nVersion: 1\n')
    (root / 'resource.bin').write_bytes(b'\x00\xff\x01')
    base = commit(root)
    (root / 'app.cpp').write_text('bool validate() { return false; }\n')
    head = commit(root)
    loaded = domains.load_domain('aurora-qt5')
    release = NS(metadata=(NS(glob='*.meta', required=(r'^Name:', r'^Version:'),
                              identity_pattern=r'^Name:\s*(\S+)'),),
                 build_required=True, tests_required=False, test_command=())
    domain = NS(id='test', corpus=loaded.corpus, checks=NS(forbidden=()),
                ops=loaded.ops, release=release)
    return root, base, head, domain


class Model:
    def __init__(self, review=None, notes=None, finish='stop', mutate=None):
        self.review = review or {'findings': []}
        self.notes = notes
        self.finish = finish
        self.calls = []
        self.mutate = mutate

    def __call__(self, role, prompt):
        self.calls.append((role, prompt))
        if self.mutate:
            self.mutate()
        sources = json.loads(prompt)['context']['sources']
        data = self.review if role == 'release_review' else self.notes or {
            'summary': 'Release draft', 'items': [{'text': 'Validation changed', 'sources': [sources[0]['id']]}]}
        return {'content': json.dumps(data), 'finish_reason': self.finish,
                'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}}


def run(project, tmp_path, model=None, build=None):
    from core.release_preparation import run_preparation
    root, base, head, domain = project
    return run_preparation(root, base, head, domain, tmp_path / 'exports', model or Model(),
                           build=build or (lambda *_: {'status': 'passed', 'reason': 'built'}))


def test_ready_is_commit_bound_isolated_and_persistent(project, tmp_path):
    from core.release_preparation import load_status, status_lines
    root, base, head, domain = project
    before = git(root, 'status', '--porcelain')
    model = Model()
    def build(copy, *_):
        assert copy != root and not (copy / '.git').exists()
        assert (copy / 'resource.bin').read_bytes() == b'\x00\xff\x01'
        assert (copy / 'app.cpp').read_text() == (root / 'app.cpp').read_text()
        package = copy / 'output.pkg'
        package.write_bytes(b'actual package')
        return {'status': 'passed', 'reason': 'built', 'package': package, 'log': 'password=abc123'}
    report = run(project, tmp_path, model, build)
    assert report['status'] == 'ready_for_manual_release'
    assert (report['base'], report['head']) == (base, head)
    assert [c['status'] for c in report['checks'] if c['id'] == 'tests'] == ['not_configured']
    assert len(model.calls) == 2
    assert report['artifacts'][0]['sha256'] == hashlib.sha256(b'actual package').hexdigest()
    directory = Path(report['directory'])
    assert (directory / 'report.md').exists() and (directory / 'release-notes.md').exists()
    assert 'abc123' not in (directory / 'build.log').read_text()
    assert git(root, 'status', '--porcelain') == before
    assert load_status(tmp_path / 'exports')['status'] == report['status']
    assert any(head in line for line in status_lines(report))


@pytest.mark.parametrize('bad', ['dirty', 'same', 'unknown', 'head', 'inside', 'domain'])
def test_invalid_input_preserves_latest_and_never_calls_model(project, tmp_path, bad):
    from core.release_preparation import ReleaseError, run_preparation
    root, base, head, domain = project
    run(project, tmp_path)
    latest = (tmp_path / 'exports/releases/latest.json').read_bytes()
    output = tmp_path / 'exports'
    if bad == 'dirty': (root / 'untracked').write_text('mine')
    if bad == 'same': base = head
    if bad == 'unknown': base = 'unknown-ref'
    if bad == 'head': head = base
    if bad == 'inside': output = root / 'exports'
    if bad == 'domain': domain.release = None
    def forbidden(*_): pytest.fail('must reject before action')
    with pytest.raises(ReleaseError):
        run_preparation(root, base, head, domain, output, forbidden, build=forbidden)
    assert (tmp_path / 'exports/releases/latest.json').read_bytes() == latest


@pytest.mark.parametrize('status,expected', [('failed', 'blocked'), ('unavailable', 'incomplete')])
def test_build_status_cannot_be_overridden_by_ai(project, tmp_path, status, expected):
    report = run(project, tmp_path, build=lambda *_: {'status': status, 'reason': 'actual result'})
    assert report['status'] == expected
    assert report['ai']['release_notes']['status'] == 'passed'


@pytest.mark.parametrize('review,finish', [({'findings': [{'severity': 'blocker', 'text': 'risk', 'sources': ['fake']}]}, 'stop'),
    ({'findings': [], 'extra': 1}, 'stop'), ({'findings': []}, 'length')])
def test_malformed_model_does_not_become_empty_review(project, tmp_path, review, finish):
    model = Model(review=review, finish=finish)
    report = run(project, tmp_path, model)
    assert report['status'] == 'incomplete'
    assert report['ai']['release_review']['status'] == 'error'
    assert report['ai']['release_notes']['status'] == 'not_run'
    assert len(model.calls) == 1


def test_full_metadata_and_unchanged_forbidden_content_checked(project, tmp_path):
    root, base, head, domain = project
    (root / 'package.meta').write_text('Name: example\n')
    (root / 'legacy.cpp').write_text('forbidden_construct();\n')
    project = root, base, commit(root), domain
    domain.checks = NS(forbidden=(domains.ForbiddenRule('forbidden_construct', 'unsafe'),))
    report = run(project, tmp_path)
    assert report['status'] == 'blocked'
    checks = {c['id']: c for c in report['checks']}
    assert checks['metadata']['status'] == checks['domain']['status'] == 'failed'


def test_secrets_never_enter_ai_or_reports_and_original_is_unchanged(project, tmp_path):
    root, base, head, domain = project
    (root / 'app.cpp').write_text('const char *api_key = "sensitive-value";\n')
    project = root, base, commit(root), domain
    model = Model()
    report = run(project, tmp_path, model)
    assert all('sensitive-value' not in prompt for _, prompt in model.calls)
    for path in Path(report['directory']).glob('*'):
        if path.is_file(): assert b'sensitive-value' not in path.read_bytes()
    assert 'sensitive-value' in (root / 'app.cpp').read_text()


def test_stale_and_interrupted_reports_are_incomplete(project, tmp_path):
    from core.release_preparation import load_status
    root, *_ = project
    report = run(project, tmp_path, Model(mutate=lambda: (root / 'untracked').write_text('changed')))
    assert report['status'] == 'incomplete' and report['stale']
    path = Path(report['directory']) / 'checks.json'
    data = json.loads(path.read_text())
    data['status'] = 'running'
    path.write_text(json.dumps(data))
    assert load_status(tmp_path / 'exports')['status'] == 'incomplete'


def test_latest_does_not_follow_arbitrary_paths(tmp_path):
    from core.release_preparation import load_status
    directory = tmp_path / 'exports/releases'
    directory.mkdir(parents=True)
    (directory / 'latest.json').write_text(json.dumps({'run_id': '../../private'}))
    assert load_status(tmp_path / 'exports') is None


def test_recursive_metadata_glob_includes_root_files_and_optional_identity(project, tmp_path):
    root, base, head, domain = project
    domain.release.metadata = (NS(glob='**/*.meta', required=(r'^Name:',), identity_pattern=''),)
    assert run(project, tmp_path)['status'] == 'ready_for_manual_release'


def test_complete_diff_includes_changed_files_outside_corpus(project, tmp_path):
    root, base, head, domain = project
    (root / 'settings.customconfig').write_text('enabled=yes\n')
    project = root, base, commit(root), domain
    model = Model()
    report = run(project, tmp_path, model)
    prompt = json.loads(model.calls[0][1])
    diff = prompt['context']['sources'][0]['text']
    assert 'settings.customconfig' in diff and '+enabled=yes' in diff
    assert any('base_sha' in source for source in report['sources'])


def test_tui_status_includes_real_results_and_usage(project, tmp_path):
    from core.release_preparation import status_lines
    report = run(project, tmp_path)
    text = '\n'.join(status_lines(report))
    assert 'Релиз: ready_for_manual_release' in text
    assert 'Каталог результатов:' in text
    assert 'tests: not_configured' in text
    assert 'Release draft' in text
    assert '60' in text and 'токен' in text


@pytest.mark.parametrize('payload', [[], {'schema': 1, 'run_id': '0'*32, 'status': 'ready_for_manual_release', 'checks': [], 'ai': {}}])
def test_malformed_saved_report_is_not_exposed(tmp_path, payload):
    from core.release_preparation import load_status
    releases = tmp_path / 'exports/releases'
    directory = releases / ('0'*32)
    directory.mkdir(parents=True)
    (releases / 'latest.json').write_text(json.dumps({'run_id': '0'*32}))
    (directory / 'checks.json').write_text(json.dumps(payload))
    assert load_status(tmp_path / 'exports') is None


def test_model_blocker_has_priority_over_unavailable_build(project, tmp_path):
    def model(role, prompt):
        if role == 'release_review':
            return {'content': json.dumps({'findings': [{'severity': 'blocker', 'text': 'unsafe', 'sources': ['diff']}]}), 'finish_reason': 'stop'}
        return Model()(role, prompt)
    report = run(project, tmp_path, model, lambda *_: {'status': 'unavailable', 'reason': 'no tool'})
    assert report['status'] == 'blocked'


def test_interruption_persists_and_does_not_run_author(project, tmp_path):
    from core.release_preparation import load_status
    def interrupted(*_): raise KeyboardInterrupt()
    report = run(project, tmp_path, interrupted)
    assert report['status'] == 'incomplete' and report['interrupted']
    assert load_status(tmp_path / 'exports')['status'] == 'incomplete'


def test_required_tests_are_actual_check_and_optional_tests_do_not_block(project, tmp_path):
    import sys
    root, base, head, domain = project
    domain.release.test_command = (sys.executable, '-c', 'print("test run"); raise SystemExit(1)')
    domain.release.tests_required = True
    report = run(project, tmp_path)
    assert report['status'] == 'blocked'
    assert (Path(report['directory']) / 'tests.log').read_text().strip() == 'test run'
    domain.release.tests_required = False
    assert run(project, tmp_path)['status'] == 'ready_for_manual_release'


def test_diverged_base_is_rejected_instead_of_merge_base(project, tmp_path):
    from core.release_preparation import ReleaseError, run_preparation
    root, base, head, domain = project
    git(root, 'checkout', '-qb', 'other', base)
    (root / 'other.cpp').write_text('unrelated();\n')
    diverged = commit(root)
    git(root, 'checkout', '-q', 'test')
    with pytest.raises(ReleaseError, match='предком'):
        run_preparation(root, diverged, head, domain, tmp_path / 'exports', Model())


def test_model_secrets_are_redacted_without_breaking_valid_json(project, tmp_path):
    def model(role, prompt):
        if role == 'release_review':
            return {'content': json.dumps({'findings': [{'severity': 'recommendation', 'text': 'password=secret-literal', 'sources': ['diff']}]}), 'finish_reason': 'stop'}
        return {'content': json.dumps({'summary': 'token report', 'items': [{'text': 'api_key=other-literal', 'sources': ['diff']}]}), 'finish_reason': 'stop'}
    report = run(project, tmp_path, model)
    assert report['status'] == 'ready_for_manual_release'
    assert 'secret-literal' not in json.dumps(report)
    assert 'other-literal' not in json.dumps(report)
    saved = json.loads((Path(report['directory']) / 'checks.json').read_text())
    assert saved['status'] == report['status']


def test_deleted_source_uses_base_sha_and_key_middle_hunk_is_hidden(project, tmp_path):
    root, base, head, domain = project
    key = '-----BEGIN PRIVATE KEY-----\n' + ''.join(f'private-material-{i}\n' for i in range(30)) + '-----END PRIVATE KEY-----\n'
    (root / 'keys.cpp').write_text(key)
    base = commit(root)
    (root / 'keys.cpp').write_text(key.replace('private-material-15', 'different-private-content'))
    (root / 'app.cpp').unlink()
    project = root, base, commit(root), domain
    model = Model()
    report = run(project, tmp_path, model)
    assert report['status'] == 'ready_for_manual_release'
    assert all('private-material-' not in prompt and 'different-private-content' not in prompt for _, prompt in model.calls)
    deleted = next(source for source in report['sources'] if source['kind'] == 'deleted')
    assert deleted['sha'] == base and deleted['path'] == 'app.cpp'
    assert 'bool validate()' in deleted['text']


def test_build_copy_omits_symlinks_and_sensitive_files(project, tmp_path):
    root, base, head, domain = project
    (root / '.env').write_text('SECRET_VALUE=private\n')
    (root / 'link.cpp').symlink_to('/etc/passwd')
    project = root, base, commit(root), domain
    report = run(project, tmp_path)
    assert report['status'] == 'incomplete'
    assert any(item.get('path') == 'link.cpp' for item in report['limitations'] if isinstance(item, dict))
    assert not (root / 'build').exists()


def test_metadata_identity_mismatch_is_blocker(project, tmp_path):
    root, base, head, domain = project
    (root / 'other.meta').write_text('Name: different\nVersion: 1\n')
    report = run((root, base, commit(root), domain), tmp_path)
    assert report['status'] == 'blocked'
    assert 'не совпадают' in next(c['reason'] for c in report['checks'] if c['id'] == 'metadata')


@pytest.mark.parametrize('shadow', [False, True])
def test_real_build_adapter_applies_target_locally_and_preserves_logs(project, tmp_path, monkeypatch, shadow):
    from core import release_preparation as release
    root, _, _, domain = project
    tool = tmp_path / 'sdk'
    tool.touch()
    tool.chmod(0o700)
    monkeypatch.setenv(domain.ops.tool_env, str(tool))
    if shadow:
        domain.release.target_pattern = domain.ops.target_pattern.removesuffix('.default')
        domain.release.build_command = ('build', '{project}')
    pattern = getattr(domain.release, 'target_pattern', '') or domain.ops.target_pattern
    target = pattern.replace('*', '1').replace('{arch}', domain.ops.default_architecture)
    calls = []
    def runner(argv, cwd, timeout):
        calls.append((argv[1:], cwd))
        if argv[1:] == ['tools', 'list']:
            return 0, target, .1
        if argv[1:] == ['--version']:
            return 0, 'SDK_RELEASE=1', .1
        if argv[1:2] == ['build']:
            assert timeout == 180
            package = cwd / 'RPMS' / 'out.rpm'
            package.parent.mkdir()
            package.write_bytes(b'package')
        return 0, 'real command output', .1
    monkeypatch.setattr(release, '_run', runner)
    result = release._build(root, domain, domain.ops.default_architecture)
    assert result['status'] == 'passed'
    configuration = [(args, cwd) for args, cwd in calls if args[:1] == ['config']]
    assert configuration == [(['config', 'target=' + target], root / ('build_' + domain.ops.default_architecture))]
    assert 'real command output' in result['log']
    assert target in result['log']
    assert any(args == (['build', str(root)] if shadow else ['build']) for args, _ in calls)


def test_sdk_timeout_names_limit_and_keeps_partial_output(project, tmp_path, monkeypatch):
    from core import release_preparation as release
    root, _, _, domain = project
    tool = tmp_path / 'sdk'
    tool.touch()
    tool.chmod(0o700)
    monkeypatch.setenv(domain.ops.tool_env, str(tool))
    def timeout(argv, cwd, limit):
        raise subprocess.TimeoutExpired(argv, limit, output=b'engine did not respond')
    monkeypatch.setattr(release, '_run', timeout)
    result = release._build(root, domain, domain.ops.default_architecture)
    assert result['status'] == 'unavailable'
    assert '60' in result['reason']
    assert 'engine did not respond' in result['log']


@pytest.mark.parametrize('damage', ['reason', 'data', 'error', 'usage', 'cost', 'source', 'huge_metric'])
def test_corrupt_saved_render_fields_are_rejected(project, tmp_path, damage):
    from core.release_preparation import load_status
    report = run(project, tmp_path)
    path = Path(report['directory']) / 'checks.json'
    if damage == 'reason':
        report['ai']['release_review'] = {'status': 'error', 'reason': 123}
    elif damage == 'data':
        report['ai']['release_review'] = {'status': 'not_run', 'data': {'findings': 'broken'}}
    elif damage == 'error':
        report['error'] = {'payload': 'broken'}
    elif damage == 'usage':
        report['ai']['release_review']['usage']['total_tokens'] = float('nan')
    elif damage == 'cost':
        report['ai']['release_review']['usage']['cost_usd'] = 'broken'
    elif damage == 'huge_metric':
        report['ai']['release_review']['usage']['total_tokens'] = 10 ** 400
    else:
        report['sources'][0]['sha'] = 123
    path.write_text(json.dumps(report))
    assert load_status(tmp_path / 'exports') is None
