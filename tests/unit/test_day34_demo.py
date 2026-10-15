"""The live demo cannot claim success without file evidence and scoped changes."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('day34_demo',Path(__file__).parents[2]/'scripts/day34_demo.py')
demo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_demo_scope_rejects_source_changes_during_documentation():
    patch='--- /dev/null\n+++ b/core/other.cpp\n@@ -0,0 +1 @@\n+changed\n'
    assert not demo.allowed_patch(patch,document_only=True)
    assert not demo.allowed_patch(patch)
    docs=patch.replace('core/other.cpp','docs/create-folder.md')
    assert demo.allowed_patch(docs,document_only=True)
    source=patch.replace('core/other.cpp','core/controllers/filesandfoldersactionscontroller.cpp')
    assert demo.allowed_patch(source)
    assert not demo.allowed_patch(source,document_only=True)


def test_behavior_probe_detects_duplicate_start_and_success_error_release(tmp_path):
    source=tmp_path/'controller.cpp'
    source.write_text('''
void FilesAndFoldersActionsController::initCreateFolderOperation()
{
    auto operation = new CreateFolderOperation(m_settings, m_path);
    setConnectionInProgress(true);
    operation->run();
}
void FilesAndFoldersActionsController::handleCreatingFolderOperation(const bool success, const QString errorString)
{
    setConnectionInProgress(false);
    if (success) { operationSuccess(); } else { m_errorMessage=errorString; operationError(); }
}
''')
    assert not demo.verify_behavior(source,tmp_path/'before')['ok']
    text=source.read_text().replace('    auto operation', '    if (connectionInProgress()) return;\n    auto operation')
    source.write_text(text)
    assert demo.verify_behavior(source,tmp_path/'after')['ok']
    source.write_text(text.replace('    setConnectionInProgress(false);',''))
    assert not demo.verify_behavior(source,tmp_path/'broken')['ok']


def test_fingerprint_detects_changes_in_untracked_disposable_copy(tmp_path):
    import subprocess
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    path=tmp_path/'source.cpp'
    path.write_text('before')
    before=demo.fingerprint(tmp_path,include_untracked=True)
    path.write_text('after')
    assert before and before!=demo.fingerprint(tmp_path,include_untracked=True)


def test_scenario_gates_require_document_then_code_test_and_doc_update(tmp_path):
    (tmp_path/'docs').mkdir()
    sources=['core/a.cpp','core/b.h','core/c.cpp']
    before={}
    doc=tmp_path/'docs/folders.md'
    doc.write_text('\n'.join(sources))
    current={'docs/folders.md':'new'}
    assert demo.check_scenario(tmp_path,before,current,sources)['document']=='docs/folders.md'
    with pytest.raises(RuntimeError):
        demo.check_scenario(tmp_path,current,dict(current,**{'core/controllers/filesandfoldersactionscontroller.cpp':'new'}),sources,'docs/folders.md')
    updated=dict(current,**{'docs/folders.md':'updated',
                           'core/controllers/filesandfoldersactionscontroller.cpp':'new','tests/test_folder.cpp':'new'})
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_folder.cpp').write_text('// actual behavioral test')
    assert demo.check_scenario(tmp_path,current,updated,sources,'docs/folders.md')['test_paths']==['tests/test_folder.cpp']


def test_visible_trace_accepts_real_rendered_ranged_read(tmp_path):
    from core.mcp_tools import ToolStep
    (tmp_path/'source.cpp').write_text('evidence')
    step=ToolStep(1,1,'repository','repo_read',{'path':'source.cpp','start_line':1,'end_line':20},{},'1: evidence')
    assert demo.read_paths_from_screen(step.journal_line(),tmp_path)==['source.cpp']


def test_demo_write_goals_are_not_vetoed_by_no_write_guard():
    from core.request_routing import write_forbidden
    assert write_forbidden(demo.INVESTIGATE)
    assert not write_forbidden(demo.DOCUMENT)
    assert not write_forbidden(demo.DEVELOP)


def test_demo_mcp_cannot_start_host_build(tmp_path):
    import json,os,subprocess
    source=tmp_path/'source'
    source.mkdir()
    subprocess.run(['git','init','-q',str(source)],check=True)
    (source/'README.md').write_text('demo')
    subprocess.run(['git','-C',str(source),'add','README.md'],check=True)
    workspace=tmp_path/'workspace';workspace.mkdir()
    project=demo.prepare(source,workspace)
    tools=json.loads((workspace/'tools.json').read_text())
    assert tools['run']['allowed']==[]
    assert str(project) in os.environ['FFAI_MCP_ARGS']


def test_demo_refuses_output_inside_original_source(tmp_path,monkeypatch,capsys):
    source=tmp_path/'original';source.mkdir()
    monkeypatch.setattr(demo.sys,'argv',['day34_demo.py','--source',str(source),'--output',str(source/'result.json')])
    with pytest.raises(SystemExit):
        demo.main()
    assert 'Output must be outside' in capsys.readouterr().err
    assert not (source/'result.json').exists()
