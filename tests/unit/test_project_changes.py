"""Real temporary repositories prove journal boundaries and working-tree preservation."""
from pathlib import Path
import subprocess
import pytest
from core.project_changes import ProjectChanges, ChangeError
from core import patches


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    (root / 'notes.txt').write_text('base\n')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'base')
    return root, ProjectChanges(root, tmp_path / 'changes')


def edit(name='notes.txt', old='base', new='agent'):
    return f'--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-{old}\n+{new}\n'


def test_apply_dirty_multifile_keeps_staged_and_private_snapshots(project):
    root, service = project
    (root / 'notes.txt').write_text('staged\n')
    git(root, 'add', 'notes.txt')
    (root / 'notes.txt').write_text('manual\n')
    staged = git(root, 'diff', '--cached')
    (root / 'deleted.txt').write_text('delete\n')
    patch = edit(old='manual') + '--- /dev/null\n+++ b/docs/new.md\n@@ -0,0 +1 @@\n+hello\n' + '--- a/deleted.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-delete\n'
    record = service.prepare(patch, task_number=7)
    assert record.status == 'prepared'
    assert (root / 'notes.txt').read_text() == 'manual\n'
    seen = []
    result = service.apply(record.id, lambda summary, diff: seen.append((summary, diff)) or True)
    assert result.ok and result.record.status == 'applied'
    assert seen[0][1] == patch
    assert (root / 'notes.txt').read_text() == 'agent\n'
    assert (root / 'docs/new.md').read_text() == 'hello\n'
    assert not (root / 'deleted.txt').exists()
    assert git(root, 'diff', '--cached') == staged
    assert service.get(record.id).before['notes.txt'].bytes == b'manual\n'
    assert service.changes_dir.stat().st_mode & 0o077 == 0
    assert next(service.changes_dir.glob('*.json')).stat().st_mode & 0o077 == 0


def test_approval_race_preserves_manual_change(project):
    root, service = project
    record = service.prepare(edit())
    def confirm(summary, diff):
        (root / 'notes.txt').write_text('user changed during approval\n')
        return True
    result = service.apply(record.id, confirm)
    assert not result.ok and result.record.status == 'conflict'
    assert (root / 'notes.txt').read_text() == 'user changed during approval\n'


def test_no_confirmation_no_write(project):
    root, service = project
    record = service.prepare(edit())
    assert not service.apply(record.id, None).ok
    assert (root / 'notes.txt').read_text() == 'base\n'


@pytest.mark.parametrize('patch', [
    edit('../outside'), edit('.git/config'), edit('/tmp/outside'),
    '--- a/../outside\n+++ b/notes.txt\n@@ -1 +1 @@\n-base\n+agent\n',
    'diff --git a/../outside b/notes.txt\n' + edit(),
    'diff --git a/notes.txt b/notes.txt\nnew mode 120000\n' + edit(),
    'rename from notes.txt\nrename to renamed.txt\n',
    'GIT binary patch\nliteral 5\nabc\n',
])
def test_reject_unsafe_raw_headers_and_unsupported_operations(project, patch):
    root, service = project
    with pytest.raises(ChangeError):
        service.prepare(patch)
    assert (root / 'notes.txt').read_text() == 'base\n'


def test_quoted_spaces_and_symlinks(project):
    root, service = project
    (root / 'a file.md').write_text('base\n')
    patch = 'diff --git "a/a file.md" "b/a file.md"\n' + edit('a file.md').replace('a/a file.md', '"a/a file.md"').replace('b/a file.md', '"b/a file.md"')
    assert service.apply(service.prepare(patch).id, lambda *_: True).ok
    (root / 'link').symlink_to(root, target_is_directory=True)
    with pytest.raises(ChangeError):
        service.prepare(edit('link/notes.txt'))


def test_state_directory_inside_project_is_rejected(project):
    root, _ = project
    with pytest.raises(ChangeError):
        ProjectChanges(root, root / '.state')


@pytest.mark.parametrize('state', ['before', 'after', 'mixed', 'unknown'])
def test_recover_interrupted_apply(project, state):
    root, service = project
    (root / 'second.txt').write_text('base\n')
    record = service.prepare(edit() + edit('second.txt'))
    record.status = 'applying'
    service._save(record)
    if state in ('after', 'mixed', 'unknown'):
        (root / 'notes.txt').write_text('agent\n')
    if state == 'after':
        (root / 'second.txt').write_text('agent\n')
    if state == 'unknown':
        (root / 'second.txt').write_text('manual\n')
    recovered = ProjectChanges(root, service.changes_dir).recover()[0]
    assert recovered.status == {'before': 'failed', 'after': 'applied', 'mixed': 'failed', 'unknown': 'conflict'}[state]
    if state == 'mixed':
        assert (root / 'notes.txt').read_text() == 'base\n'
    if state == 'unknown':
        assert (root / 'second.txt').read_text() == 'manual\n'


def test_partial_apply_error_restores_known_states(project, monkeypatch):
    root, service = project
    record = service.prepare(edit())
    def fail(root, patch):
        (root / 'notes.txt').write_text('agent\n')
        return 'simulated apply failure'
    monkeypatch.setattr(patches, 'apply', fail)
    assert not service.apply(record.id, lambda *_: True).ok
    assert (root / 'notes.txt').read_text() == 'base\n'


def test_domain_rechecked_after_approval(project):
    root, service = project
    rules = {'reason': ''}
    service.validate = lambda _: rules['reason']
    record = service.prepare(edit())
    def confirm(*_):
        rules['reason'] = 'domain rule changed'
        return True
    result = service.apply(record.id, confirm)
    assert not result.ok and 'domain rule changed' in result.reason
    assert (root / 'notes.txt').read_text() == 'base\n'


def test_symlink_inserted_after_prepare_becomes_conflict(project, tmp_path):
    root, service = project
    record = service.prepare(edit())
    external = tmp_path / 'external.txt'
    external.write_text('outside\n')
    (root / 'notes.txt').unlink()
    (root / 'notes.txt').symlink_to(external)
    result = service.apply(record.id, lambda *_: True)
    assert not result.ok and result.record.status == 'conflict'
    assert external.read_text() == 'outside\n'


def test_undo_exact_dirty_content_modes_and_existence_after_restart(project):
    root, service = project
    (root / 'notes.txt').write_text('staged\n')
    git(root, 'add', 'notes.txt')
    staged = git(root, 'diff', '--cached')
    (root / 'notes.txt').write_text('manual\n')
    (root / 'notes.txt').chmod(0o755)
    (root / 'deleted.txt').write_text('keep me\n')
    patch = edit(old='manual') + '--- /dev/null\n+++ b/docs/new.md\n@@ -0,0 +1 @@\n+hello\n' + '--- a/deleted.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-keep me\n'
    record = service.prepare(patch, 3)
    assert service.apply(record.id, lambda *_: True).ok
    restarted = ProjectChanges(root, service.changes_dir)
    seen = []
    result = restarted.undo(record.id, lambda summary, diff: seen.append(diff) or True)
    assert result.ok and result.record.status == 'undone'
    assert '-agent' in seen[0] and '+manual' in seen[0]
    assert (root / 'notes.txt').read_text() == 'manual\n'
    assert (root / 'notes.txt').stat().st_mode & 0o777 == 0o755
    assert (root / 'deleted.txt').read_text() == 'keep me\n'
    assert not (root / 'docs').exists()
    assert git(root, 'diff', '--cached') == staged
    assert not restarted.undo(record.id, lambda *_: pytest.fail('repeat confirmation')).ok


def test_undo_manual_conflict_preserves_entire_package(project):
    root, service = project
    (root / 'second.txt').write_text('base\n')
    record = service.prepare(edit() + edit('second.txt'))
    assert service.apply(record.id, lambda *_: True).ok
    (root / 'second.txt').write_text('user\n')
    assert not service.undo(record.id, lambda *_: True).ok
    assert (root / 'notes.txt').read_text() == 'agent\n'
    assert (root / 'second.txt').read_text() == 'user\n'


def test_undo_approval_race_and_no_confirmation(project):
    root, service = project
    record = service.prepare(edit())
    assert service.apply(record.id, lambda *_: True).ok
    assert not service.undo(record.id, None).ok
    def confirm(*_):
        (root / 'notes.txt').write_text('user\n')
        return True
    assert not service.undo(record.id, confirm).ok
    assert (root / 'notes.txt').read_text() == 'user\n'


def test_undo_dependencies_require_reverse_order(project):
    root, service = project
    first = service.prepare(edit())
    service.apply(first.id, lambda *_: True)
    second = service.prepare(edit(old='agent', new='base'))
    service.apply(second.id, lambda *_: True)
    # Even if later content accidentally matches the first postimage, dependency is explicit.
    third = service.prepare(edit())
    service.apply(third.id, lambda *_: True)
    assert not service.undo(first.id, lambda *_: True).ok
    assert service.undo(third.id, lambda *_: True).ok
    assert service.undo(second.id, lambda *_: True).ok
    assert service.undo(first.id, lambda *_: True).ok


@pytest.mark.parametrize('state', ['before', 'after', 'mixed', 'unknown'])
def test_recover_interrupted_undo(project, state):
    root, service = project
    (root / 'second.txt').write_text('base\n')
    record = service.prepare(edit() + edit('second.txt'))
    service.apply(record.id, lambda *_: True)
    record.status = 'undoing'
    service._save(record)
    if state in ('before', 'mixed', 'unknown'):
        (root / 'notes.txt').write_text('base\n')
    if state == 'before':
        (root / 'second.txt').write_text('base\n')
    if state == 'unknown':
        (root / 'second.txt').write_text('manual\n')
    recovered = ProjectChanges(root, service.changes_dir).recover()[0]
    assert recovered.status == {'before':'undone', 'after':'applied', 'mixed':'undone', 'unknown':'conflict'}[state]
    if state == 'unknown':
        assert (root / 'second.txt').read_text() == 'manual\n'
