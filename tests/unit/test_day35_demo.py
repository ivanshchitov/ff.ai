"""The live release demonstration isolates history and proves its two inputs."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('day35_demo', Path(__file__).parents[2] / 'scripts/day35_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(autouse=True)
def preserve_mcp_environment(monkeypatch):
    import os
    for key in ('FFAI_MCP_COMMAND', 'FFAI_MCP_ARGS'):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)


def source_repo(tmp_path):
    source = tmp_path / 'original'
    source.mkdir()
    demo.git(source, 'init', '-q')
    controller = source / demo.CONTROLLER
    controller.parent.mkdir(parents=True)
    controller.write_text('void FilesAndFoldersActionsController::initCreateFolderOperation()\n{\n'
                          '    auto operation = new CreateFolderOperation(m_settings, m_path);\n}\n')
    (source / 'rpm').mkdir()
    (source / 'rpm/app.spec').write_text('Name: app\nVersion: 1\nRelease: 1\nLicense: BSD\n')
    (source / '.env').write_text('API_TOKEN=private-demo-value')
    (source / 'untracked.txt').write_text('personal')
    demo.git(source, 'add', demo.CONTROLLER, 'rpm/app.spec', '.env')
    return source


def test_history_requires_explicit_demo_only_consent(tmp_path):
    source = source_repo(tmp_path)
    workspace = tmp_path / 'demo'
    workspace.mkdir()
    with pytest.raises(ValueError, match='approve-demo-history'):
        demo.prepare_scenarios(source, workspace, approved=False)
    assert not list(workspace.iterdir())


def test_two_clean_revisions_per_scenario_and_original_unchanged(tmp_path):
    source = source_repo(tmp_path)
    before = demo.source_state(source)
    workspace = tmp_path / 'demo'
    workspace.mkdir()
    scenarios = demo.prepare_scenarios(source, workspace, approved=True)
    for scenario in scenarios:
        project = scenario['project']
        assert demo.git(project, 'rev-list', '--count', 'HEAD').strip() == '2'
        assert not demo.git(project, 'status', '--porcelain')
        assert scenario['base'] != demo.git(project, 'rev-parse', 'HEAD').strip()
        assert 'connectionInProgress()' in (project / demo.CONTROLLER).read_text()
        assert not (project / '.env').exists()
        assert not (project / 'untracked.txt').exists()
        metadata = (project / 'rpm/app.spec').read_text()
        assert ('Version:' in metadata) == (scenario['name'] == 'good')
    assert demo.source_state(source) == before


def test_workspace_cannot_overlap_original(tmp_path):
    source = source_repo(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        demo.prepare_scenarios(source, source / 'copy', approved=True)


def test_completion_requires_new_rendered_result_not_command_echo():
    previous = 'Релиз: blocked\nКаталог результатов: /old'
    assert not demo.fresh_result(previous, previous + '\nВы: /release prepare --base HEAD~1 --head HEAD')
    current = previous + '\nРелиз: incomplete\nКаталог результатов: /new'
    assert demo.fresh_result(previous, current)


def test_status_contract_and_sanitized_report(tmp_path):
    assert demo.result_status(['Релиз: blocked']) == 'blocked'
    with pytest.raises(RuntimeError):
        demo.result_status(['/release prepare --base main --head HEAD'])
    output = demo.sanitize({'path': str(tmp_path / 'exports'), 'token': 'Bearer abc1234567890'},
                           {tmp_path: '/path/to/demo'})
    assert output['path'] == '/path/to/demo/exports'
    assert 'abc1234567890' not in str(output)


def test_failure_writes_sanitized_artifact_and_nonzero(tmp_path, monkeypatch):
    import json
    source = source_repo(tmp_path)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    output = tmp_path / 'result.json'
    monkeypatch.setattr(demo.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(demo.sys, 'argv', ['day35_demo.py', '--source', str(source),
                        '--workspace', str(workspace), '--output', str(output), '--approve-demo-history'])
    def fail(*args, **kwargs):
        raise RuntimeError('API_TOKEN=private-demo-token')
    monkeypatch.setattr(demo, 'prepare_scenarios', fail)
    assert demo.main() == 1
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['original_unchanged']
    assert 'private-demo-token' not in output.read_text()


def test_corrupt_partial_checks_do_not_hide_demo_failure(tmp_path):
    path = tmp_path / 'bad/state/exports/releases/run/checks.json'
    path.parent.mkdir(parents=True)
    path.write_text('unfinished')
    assert demo.saved_checks(tmp_path)[0]['error'] == 'Saved checks unreadable'
