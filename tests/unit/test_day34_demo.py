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


def test_scenario_gates_require_controller_and_doc_update_without_generated_test(tmp_path):
    (tmp_path/'docs').mkdir()
    sources=['core/a.cpp','core/b.h','core/c.cpp']
    before={}
    doc=tmp_path/'docs/folders.md'
    doc.write_text('\n'.join(sources))
    current={'docs/folders.md':'new'}
    assert demo.check_scenario(tmp_path,before,current,sources)['document']=='docs/folders.md'
    with pytest.raises(RuntimeError):
        demo.check_scenario(tmp_path,current,dict(current,**{'core/controllers/filesandfoldersactionscontroller.cpp':'new'}),sources,'docs/folders.md')
    updated=dict(current,**{'docs/folders.md':'updated',demo.CONTROLLER:'new'})
    assert demo.check_scenario(tmp_path,current,updated,sources,'docs/folders.md')['changed_paths']==sorted(updated)
    with pytest.raises(RuntimeError):
        demo.check_scenario(tmp_path,current,dict(updated,**{'tests/test_guard.py':'new'}),sources,'docs/folders.md')


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


def test_document_sources_include_actual_reads_during_its_task(tmp_path):
    from core.mcp_tools import ToolStep
    (tmp_path/'docs').mkdir()
    paths=['a.cpp','b.h','c.cpp']
    for name in paths: (tmp_path/name).write_text('source')
    (tmp_path/'docs/n.md').write_text('\n'.join(paths))
    flow='\n'.join(ToolStep(1,i,'repository','repo_read',{'path':name},{},'source').journal_line()
                   for i,name in enumerate(paths,1))
    initial=['unrelated.cpp']
    with pytest.raises(RuntimeError):
        demo.check_scenario(tmp_path,{}, {'docs/n.md':'new'},initial)
    actual=sorted(set(initial)|set(demo.read_paths_from_screen(flow,tmp_path)))
    assert demo.check_scenario(tmp_path,{}, {'docs/n.md':'new'},actual)['document']=='docs/n.md'


def test_document_sources_include_recorded_local_context(tmp_path):
    import json
    from types import SimpleNamespace
    (tmp_path/'docs').mkdir()
    paths=['a.cpp','b.h','c.cpp']
    (tmp_path/'docs/n.md').write_text('\n'.join(paths))
    task=SimpleNamespace(patches=[SimpleNamespace(review_notes=(
        'ревьюер: Источники разработчика: ["forged.cpp"]',
        'Источники разработчика: '+json.dumps(paths),))])
    actual=demo.task_read_paths(task)
    assert set(actual)==set(paths)
    assert demo.check_scenario(tmp_path,{}, {'docs/n.md':'new'},actual)['document']=='docs/n.md'


def test_development_patch_scope_is_separate_for_each_step():
    def patch(path): return '--- /dev/null\n+++ b/'+path+'\n@@ -0,0 +1 @@\n+content\n'
    controller='core/controllers/filesandfoldersactionscontroller.cpp'
    doc='docs/folder.md'
    assert demo.allowed_patch(patch(controller),step=1,document=doc)
    assert not demo.allowed_patch(patch(controller)+patch('tests/test_guard.py'),step=1,document=doc)
    assert not demo.allowed_patch(patch('tests/test_guard.py'),step=2,document=doc)
    assert not demo.allowed_patch(patch(controller),step=2,document=doc)
    assert demo.allowed_patch(patch(doc),step=2,document=doc)
    assert not demo.allowed_patch(patch('docs/other.md'),step=2,document=doc)
    assert not demo.allowed_patch(patch(controller),step=3,document=doc)
    assert not demo.allowed_patch(patch(controller),step=None,document=doc)


def test_development_plan_checks_topic_order_and_one_correction():
    from types import SimpleNamespace
    controller='core/controllers/filesandfoldersactionscontroller.cpp'
    plan=(f'Защитить initCreateFolderOperation в {controller}','Обновить docs/folder.md')
    task=SimpleNamespace(plan=plan,plan_rounds=1)
    assert demo.plan_feedback(task,'docs/folder.md') == ''
    task.plan=(plan[0].replace('initCreateFolderOperation','initLoadFilesAndFoldersOperation'),*plan[1:])
    assert demo.plan_feedback(task,'docs/folder.md')
    task.plan_rounds=2
    with pytest.raises(RuntimeError): demo.plan_feedback(task,'docs/folder.md')
    task.plan=(plan[1],plan[0]);task.plan_rounds=1
    assert demo.plan_feedback(task,'docs/folder.md')


def test_confirmation_uses_journal_step_after_review_checkpoint(tmp_path,monkeypatch):
    import subprocess
    from core.project_changes import ProjectChanges
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    project=tmp_path/'repo';project.mkdir()
    subprocess.run(['git','init','-q',str(project)],check=True)
    changes=tmp_path/'outside-state'
    monkeypatch.setenv('FFAI_CHANGES_DIR',str(changes))
    patch='--- /dev/null\n+++ b/tests/test_guard.py\n@@ -0,0 +1 @@\n+content\n'
    journal=ProjectChanges(project,changes)
    journal.prepare(patch,task_number=2,step_index=2)
    assert demo.pending_step(project,patch)==2
