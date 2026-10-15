"""Run three live project-file scenarios on a disposable WebDavClient copy.

No model doubles. All state and artifacts belong to a fresh workspace.
Use --approve-demo to authorize only the displayed, narrowly scoped demo patches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

FFAI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FFAI))
CONTROLLER = 'core/controllers/filesandfoldersactionscontroller.cpp'
INVESTIGATE = ('Найди реализацию CreateFolderOperation и её вызов из контроллера создания папки. '
               'Пути не угадывай: сначала получи их поиском, затем прочитай найденные файлы. '
               'Ничего не меняй. Прочитай реализацию, объявление и вызывающий код. '
               'Назови связи и источники с путями и строками. Ответ краткий.')
DOCUMENT = ('Создай в docs один краткий Markdown-документ о создании папки через CreateFolderOperation: '
            'путь от initCreateFolderOperation вызывающего контроллера до сетевой операции, '
            'результат и обработка ошибок. '
            'Сам найди исходники, дай ссылки минимум на три прочитанных файла. '
            'Изменения ограничь ровно одним новым файлом в docs; содержимое существующих файлов сохраняется. '
            'Достаточно одной подзадачи.')
DEVELOP = ('Защити initCreateFolderOperation от повторного запуска: при connectionInProgress() вернись. '
           'Сохрани сброс после успеха/ошибки. Ровно 2 подзадачи: '
           '1) только контроллер, укажи его путь в плане; '
           '2) только дополни {document} описанием защиты. '
           'Проверку выполнит внешняя C++-проба; тесты и сборку не менять. Каждый патч отдельно.')


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args],
                                   stderr=subprocess.PIPE).decode()


def fingerprint(root, include_untracked=False):
    flags=['--cached','--others','--exclude-standard'] if include_untracked else ['--cached']
    names = git(root, 'ls-files', *flags, '-z').split('\0')
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in names if name and (root/name).is_file() and not (root/name).is_symlink()}


def isolate(workspace):
    for key in list(os.environ):
        if key.startswith('FFAI_'):
            del os.environ[key]
    state = workspace/'state'
    os.environ.update(FFAI_STATE_DIR=str(state), FFAI_CACHE_DIR=str(workspace/'cache'),
                      FFAI_LLAMA_AUTOSTART='0', FFAI_NO_REEXEC='1', FFAI_AUTO_TOOLS='1',
                      FFAI_TYPING_DELAY='0', FFAI_MAX_VALIDATION_ATTEMPTS='2',
                      FFAI_MAX_PLAN_ITEMS='3', FFAI_TASK_GOAL_MAX_CHARS='1000',
                      FFAI_REQUEST_TIMEOUT='90', FFAI_TASK_PATCH_MAX_WORDS='2000',
                      FFAI_TASK_PLAN_MAX_WORDS='1500',
                      FFAI_REASONING_EFFORT='low', FFAI_CODE_FRAGMENT_MAX_CHARS='12000', FFAI_CODE_CANDIDATES_BEFORE='6',
                      FFAI_TOOL_FLOW_MAX_ROUNDS='3', FFAI_TOOL_FLOW_MAX_STEPS='8',
                      FFAI_TASK_VALIDATE_MAX_WORDS='1500', FFAI_TOOL_FLOW_CONTEXT_CHARS='80000')
    for key,name in {'HISTORY_FILE':'history.json','MEMORY_FILE':'memory.json',
                     'PROFILE_FILE':'profile.json','TASK_FILE':'task.json',
                     'SCHEDULE_FILE':'schedule.json','TASKS_DIR':'tasks',
                     'EXPORTS_DIR':'exports','CHANGES_DIR':'changes',
                     'INDEX_FILE':'index.sqlite'}.items():
        os.environ['FFAI_'+key]=str(state/name)


def prepare(source, workspace):
    from core.pr_review import redact
    project=workspace/'WebDavClient'
    project.mkdir()
    for name in fingerprint(source):
        path=Path(name)
        if any(part.startswith('.env') or part.lower() in ('credentials','secrets') for part in path.parts):
            continue
        data=(source/path).read_bytes()
        try:
            data=redact(data.decode('utf-8')).encode('utf-8')
        except UnicodeError:
            pass
        destination=project/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
    git(project,'init','-q')
    # Local MCP is sufficient: the goals concern repository behavior, not platform facts.
    tools=json.loads((FFAI/'domains/aurora-qt5/tools.json').read_text())
    tools['run']['allowed']=[]
    (workspace/'tools.json').write_text(json.dumps(tools))
    os.environ['FFAI_MCP_COMMAND']=sys.executable
    os.environ['FFAI_MCP_ARGS']=f'{FFAI}/mcp_server/repo_server.py --root {project} --tools {workspace}/tools.json'
    return project


def allowed_patch(patch, document_only=False, *, step=None, document=None):
    from core.patches import parse
    names=parse(patch).paths
    if not document_only and (step is not None or document is not None):
        if step == 1:
            return names == (CONTROLLER,)
        if step == 2:
            return names == (document,)
        return False
    return bool(names) and all(
        (name.startswith('docs/') and name.endswith('.md')) or
        (not document_only and name == CONTROLLER)
        for name in names)



def plan_feedback(task, document=None):
    """Approve a bounded plan or allow one clarification before any writes."""
    plan=task.plan
    valid=(len(plan)==1 if document is None else
           len(plan)==2 and CONTROLLER in plan[0] and 'initCreateFolderOperation' in plan[0]
           and document in plan[1])
    if valid:
        return ''
    if task.plan_rounds>=2:
        raise RuntimeError('План повторно не соответствует цели; запись не подтверждена')
    if document is None:
        return 'Одна подзадача: создать один краткий Markdown-документ в docs.'
    return ('Ровно две подзадачи по цели создания папки: 1) initCreateFolderOperation в найденном '
            'исходнике контроллера, укажи путь; 2) обновление '+document+'. '
            'Тесты не создавать: проверка поведения выполняется готовой внешней C++-пробой.')


def pending_step(project, patch):
    """The journal keeps the actual step even during a review correction checkpoint."""
    from core.project_changes import ProjectChanges
    for entry in reversed(ProjectChanges(project,Path(os.environ['FFAI_CHANGES_DIR'])).list()):
        if entry.status=='prepared' and entry.patch==patch:
            return entry.step_index
    return None


def task_read_paths(task):
    """Actual local files supplied by the application, not model-declared citations."""
    from core.task_pipeline import CONTEXT_PATHS_PREFIX
    paths=set()
    for patch in task.patches:
        for note in patch.review_notes:
            if note.startswith(CONTEXT_PATHS_PREFIX):
                paths.update(json.loads(note[len(CONTEXT_PATHS_PREFIX):]))
    return paths


def verify_behavior(source, output):
    """Compile actual controller method bodies against small Qt/operation doubles.

    This proves controller transitions, not Qt signal delivery, networking or a full SDK build.
    """
    compiler=shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        return {'ok':False,'available':False,'reason':'C++ compiler unavailable'}
    text=Path(source).read_text()
    methods=[]
    for name in ('initCreateFolderOperation','handleCreatingFolderOperation'):
        marker='void FilesAndFoldersActionsController::'+name+'('
        start=text.find(marker)
        if start<0:
            return {'ok':False,'available':False,'reason':'Controller method missing: '+name}
        opening=text.find('{',start)
        depth=0
        for end in range(opening,len(text)):
            depth += (text[end]=='{')-(text[end]=='}')
            if depth==0:
                methods.append(text[start:end+1])
                break
        else:
            return {'ok':False,'available':False,'reason':'Unbalanced method body'}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    harness=r'''
#include <string>
#include <iostream>
#define emit
using QString=std::string;
namespace Qt { enum { UniqueConnection }; }
struct QObject { static QString tr(const char *s) { return s; } };
int starts=0;
struct CreateFolderOperation {
    CreateFolderOperation(void *, QString) {}
    void run() { ++starts; }
    void operationFinished(bool, QString) {}
};
template<class... T> void connect(T...) {}
struct FilesAndFoldersActionsController {
    void *m_settings=nullptr;
    QString m_path, m_msgTitle, m_errorMessage;
    bool m_connectionInProgress=false;
    bool connectionInProgress() { return m_connectionInProgress; }
    void setConnectionInProgress(bool value) { m_connectionInProgress=value; }
    void operationSuccess() {}
    void operationError() {}
    void initCreateFolderOperation();
    void handleCreatingFolderOperation(const bool success, const QString errorString);
};
'''
    harness+='\n'.join(methods)+r'''
int main() {
    FilesAndFoldersActionsController c;
    c.initCreateFolderOperation();
    c.initCreateFolderOperation();
    if(starts!=1) { std::cout<<"FAIL duplicate start\n"; return 1; }
    c.handleCreatingFolderOperation(true, "");
    if(c.connectionInProgress()) { std::cout<<"FAIL success release\n"; return 2; }
    c.initCreateFolderOperation();
    if(starts!=2) { std::cout<<"FAIL restart after success\n"; return 3; }
    c.handleCreatingFolderOperation(false, "error");
    if(c.connectionInProgress()) { std::cout<<"FAIL error release\n"; return 4; }
    c.initCreateFolderOperation();
    if(starts!=3) { std::cout<<"FAIL restart after error\n"; return 5; }
    std::cout<<"PASS duplicate start; restart after success; restart after error\n";
}
'''
    cpp=output/'probe.cpp'
    cpp.write_text(harness)
    environment={key:value for key,value in os.environ.items()
                 if not any(marker in key.upper() for marker in ('KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
    compile_result=subprocess.run([compiler,'-std=c++11',str(cpp),'-o',str(output/'probe')],
        capture_output=True,text=True,timeout=30,env=environment)
    if compile_result.returncode:
        return {'ok':False,'available':False,'reason':'Controller probe compile failed',
                'output':compile_result.stderr[:4000]}
    run=subprocess.run([str(output/'probe')],capture_output=True,text=True,timeout=10,env=environment)
    return {'ok':run.returncode==0,'available':True,'output':run.stdout+run.stderr,
            'scope':'actual method bodies with Qt/operation doubles; not full application build'}


def read_paths_from_screen(screen,project):
    paths=set(re.findall(r'\.repo_read\s*\([^)]*?\bpath=["\']?([^,\s)"\']+)',screen.replace('\n','')))
    return sorted(path for path in paths if (project/path).is_file())


def check_scenario(project,before,after,read_paths,document=None):
    changed={name for name in set(before)|set(after) if before.get(name)!=after.get(name)}
    if document is None:
        created=[name for name in changed if name not in before and name.startswith('docs/') and name.endswith('.md')]
        if len(created)!=1 or changed!=set(created):
            raise RuntimeError('Цель документации должна создать ровно один Markdown-файл')
        document=created[0]
    else:
        if changed != {document,CONTROLLER}:
            raise RuntimeError('Нужны только правка контроллера и обновление документа')
    content=(project/document).read_text()
    cited={path for path in read_paths if path in content or Path(path).name in content}
    if len(cited)<3:
        raise RuntimeError('Документ не ссылается на три прочитанных исходника')
    return {'document':document,'changed_paths':sorted(changed)}


def run_facade(project, result, approve):
    from core.session import AssistantSession, PhaseChanged
    from core.pr_review import redact
    session=AssistantSession(project,domain_id='aurora-qt5')
    session.settings=session.settings.with_temperature(0)
    session.model=os.environ['DAY34_DEMO_MODEL']
    def event(item):
        if isinstance(item,PhaseChanged):
            print('Фаза: '+item.phase.value,flush=True)
    unsubscribe=session.subscribe(event)
    session.connect_mcp_servers()
    result['index']=list(session.run_command('/rag-code index').lines)
    session.run_command('/rag-code retrieval baseline')
    session.run_command('/rag-docs mode off')
    session.run_command('/rag-code mode off')
    before=fingerprint(project,include_untracked=True)
    print(INVESTIGATE,flush=True)
    answer=session.submit(INVESTIGATE)
    if answer.kind!='answer' or answer.answer is None:
        raise RuntimeError('Исследование не вернуло ответ: '+answer.message)
    result['investigation']={'answer':redact(answer.answer.text), 'flow':list(session.tool_flow_lines())}
    print(redact(answer.answer.text),flush=True)
    read_paths={step.arguments.get('path') for step in session.tool_flow_report().steps
                if step.tool=='repo_read'}
    result['investigation']['read_paths']=sorted(path for path in read_paths if path)
    if len(result['investigation']['read_paths'])<3 or answer.answer.text.startswith('Не знаю'):
        raise RuntimeError('Исследование не подтверждено чтением трёх файлов и содержательным ответом')
    if before!=fingerprint(project,include_untracked=True):
        raise RuntimeError('Исследование изменило исходники')
    session.run_command('/rag-code mode on')
    document=None
    read_paths=set(result['investigation']['read_paths'])
    for number,goal in enumerate((DOCUMENT,DEVELOP),1):
        if number==2:
            goal=goal.format(document=document)
            session.run_command('/tool auto off')
        print(goal,flush=True)
        scenario_before=fingerprint(project,include_untracked=True)
        def confirm(summary,patch):
            print('Патч к подтверждению: '+redact(summary)+'\n'+redact(patch),flush=True)
            if not allowed_patch(patch,document_only=number==1,step=pending_step(project,patch),document=document):
                print('Отказ: патч вне цели демо.',flush=True)
                return False
            if approve:
                return True
            import select
            print('Применить этот патч в копии проекта? [y/N] (30 секунд)',flush=True)
            ready,_,_=select.select([sys.stdin],[],[],30)
            return bool(ready) and sys.stdin.readline().strip().lower()=='y'
        session.set_patch_confirmation(confirm)
        session.run_command('/task add '+goal)
        submitted=session.run_command('/task run')
        if not submitted.task_run:
            raise RuntimeError('Цель не запустила задачу')
        steps=[]
        for _ in range(16):
            report=session.task_step()
            if report is None:
                break
            steps.append({'label':report.label,'notes':list(report.notes)})
            print(report.label,flush=True)
            print('\n'.join(report.notes),flush=True)
            if report.awaiting_edits:
                session.task_answer_edits(plan_feedback(session.task_state.current,document))
            if report.finished or any(p.reason and not p.applied for p in session.task_state.current.patches):
                break
        task=session.task_state.current
        result.setdefault('tasks',[]).append({'goal':goal,'steps':steps,'task':asdict(task),
            'flow':list(session.tool_flow_lines())})
        if task.status!='done' or not task.applied_patches() or session.task_paused or task.issues:
            raise RuntimeError('Задача не завершилась проверенным патчем: '+task.title)
        read_paths.update(read_paths_from_screen('\n'.join(session.tool_flow_lines()),project))
        read_paths.update(task_read_paths(task))
        result['tasks'][-1]['read_paths']=sorted(read_paths)
        acceptance=check_scenario(project,scenario_before,fingerprint(project,include_untracked=True),
                                  read_paths,document)
        document=acceptance['document']
        result['tasks'][-1]['acceptance']=acceptance
    result['usage']=list(session.usage_lines())
    result['changes']=list(session.project_changes_report())
    result['diff']='\n'.join(patch.text for task in session.task_state.tasks for patch in task.applied_patches())
    unsubscribe()


def run_visible(project,result,approve):
    from tests.e2e.harness import AppSession
    from core.task_state import TaskStore
    def mirror(data):
        sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
    size=shutil.get_terminal_size((110,42))
    state=Path(os.environ['FFAI_STATE_DIR'])
    store=TaskStore(state/'task.json')
    with AppSession(history_file=state/'history.json',state_dir=state,repo=project,
                    domain='aurora-qt5',api_key=None,mcp_command=os.environ['FFAI_MCP_COMMAND'],
                    mcp_args=os.environ['FFAI_MCP_ARGS'],auto_tools=True,cols=size.columns,
                    rows=size.lines,extra_env=dict(os.environ),mirror=mirror) as app:
        def wait_input():
            deadline=time.monotonic()+20
            while not app.screen_text().rstrip().endswith('Вы:'):
                if time.monotonic()>deadline:
                    raise TimeoutError('Нет приглашения TUI перед командой')
                time.sleep(.05)
        def command(text,expected):
            wait_input()
            count=app.scrollback().count(expected)
            app.send_line(text)
            deadline=time.monotonic()+40
            while app.scrollback().count(expected)<=count:
                if time.monotonic()>deadline:
                    raise TimeoutError('Нет нового результата команды '+text)
                time.sleep(.05)
            wait_input()
        from core import config
        model=os.environ['DAY34_DEMO_MODEL']
        app.send_line('/models');app.wait_for('↑/↓',timeout=20)
        time.sleep(.4)
        for _ in range(config.model_choices().index(model)):
            app.send_key(b'\x1b[B')
        app.send_key(b'\r');time.sleep(.5)
        app.send_line('/settings');app.wait_for('Температура',timeout=20)
        time.sleep(.4)
        from ui import settings_screen
        for _ in range(settings_screen.ROW_TEMPERATURE):
            app.send_key(b'\x1b[B')
            app.read_for(.1)
        for _ in range(len(f'{config.TEMPERATURE:.1f}')):
            app.send_key(b'\x7f')
            app.read_for(.1)
        app.send_key(b'0');app.read_for(.2)
        app.send_key(b'\x1b')
        settings_deadline=time.monotonic()+20
        while not app.screen_text().rstrip().endswith('Вы:'):
            if time.monotonic()>settings_deadline:
                raise TimeoutError('Экран настроек не вернулся к вводу команды')
            time.sleep(.05)
        command('/rag-code index','Файлов:')
        command('/rag-code retrieval baseline','Режим отбора:')
        command('/rag-docs mode off','Поиск по документации выключен.')
        command('/rag-code mode off','Поиск по корпусу кода выключен.')
        before=fingerprint(project,include_untracked=True)
        app.ask(INVESTIGATE,timeout=180)
        command('/tool flow','Раундов:')
        screen=app.scrollback()
        paths=read_paths_from_screen(screen,project)
        from core.history_manager import HistoryManager
        history=HistoryManager(path=state/'history.json')
        answer=history.dialogues[-1].get('answer','') if history.dialogues else ''
        result['investigation']={'answer':answer,'read_paths':paths,'flow_screen':screen}
        if len(paths)<3 or not answer or answer.startswith('Не знаю') or before!=fingerprint(project,include_untracked=True):
            raise RuntimeError('Исследование TUI не прошло проверку источников и сохранности файлов')
        app.read_for(5)
        command('/rag-code mode on','Поиск по корпусу кода включён.')
        plans=confirmations=0
        document=None
        for number,goal in enumerate((DOCUMENT,DEVELOP),1):
            if number==2:
                goal=goal.format(document=document)
                command('/tool auto off','Автовызов инструментов выключен.')
            scenario_before=fingerprint(project,include_untracked=True)
            app.send_line('/task add '+goal)
            add_deadline=time.monotonic()+20
            while not any(t.number==number for t in store.load().tasks):
                if time.monotonic()>add_deadline:
                    raise TimeoutError('Цель не добавлена в очередь')
                app.screen_text();time.sleep(.05)
            time.sleep(.4)
            app.send_line('/task run')
            deadline=time.monotonic()+300
            last_plan=last_patch=None
            while time.monotonic()<deadline:
                app.screen_text()
                snapshot=store.load()
                task=snapshot.current
                if task and task.number==number:
                    if task.awaiting_edits and last_plan!=(task.plan_rounds,task.plan):
                        marker=app.scrollback().count('План построен. Правки')
                        if marker>plans:
                            time.sleep(5)
                            app.send_line(plan_feedback(task,document))
                            plans=marker;last_plan=(task.plan_rounds,task.plan)
                    screen=app.scrollback()
                    marker=screen.count('Применить патч?')
                    if marker>confirmations:
                        from core.project_changes import ProjectChanges
                        entries=ProjectChanges(project,state/'changes').list()
                        pending=[entry for entry in entries if entry.status=='prepared']
                        entry=pending[-1] if pending else None
                        if entry and last_patch!=entry.id:
                            if not approve:
                                raise RuntimeError('--visible requires --approve-demo for bounded confirmation')
                            permitted=allowed_patch(entry.patch,number==1,step=entry.step_index,document=document)
                            time.sleep(5)
                            app.send_key(b'y' if permitted else b'n')
                            last_patch=entry.id;confirmations=marker
                    if task.status in ('done','failed'):
                        result.setdefault('tasks',[]).append(asdict(task))
                        if task.status=='failed' or task.issues or not task.applied_patches():
                            raise RuntimeError('Задача завершилась с проблемами')
                        break
                    if any(p.reason and not p.applied and not p.review_pending for p in task.patches):
                        raise RuntimeError('Подзадача отклонена: '+next(p.reason for p in task.patches if p.reason and not p.applied))
                    if snapshot.paused:
                        raise RuntimeError('Прогон на паузе: см. результат субагентов')
                time.sleep(.15)
            else:
                raise TimeoutError('Истекло время ожидания задачи')
            if number==1:
                command('/tool flow','Раундов:')
                paths=sorted(set(paths)|set(read_paths_from_screen(app.scrollback(),project)))
            paths=sorted(set(paths)|task_read_paths(task))
            result['tasks'][-1]['read_paths']=paths
            acceptance=check_scenario(project,scenario_before,fingerprint(project,include_untracked=True),paths,document)
            document=acceptance['document']
            result['tasks'][-1]['acceptance']=acceptance
            wait_input()
            app.read_for(5)
        for text,expected in (('/task','Очередь: задач'),('/changes','— applied; задача'),
                              ('/usage','Сессия: запросов')):
            command(text,expected);app.read_for(4)
        app.send_line('/exit');app.wait_exit()
        result['screen']=app.scrollback()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=FFAI.parent/'WebDavClient')
    parser.add_argument('--workspace',type=Path)
    parser.add_argument('--visible',action='store_true')
    parser.add_argument('--approve-demo',action='store_true')
    parser.add_argument('--model',default='glm-5.3-flash')
    parser.add_argument('--reasoning-effort',choices=('', 'low', 'high', 'max'))
    parser.add_argument('--output',type=Path,default=FFAI/'videos/day34-dry-run.json')
    args=parser.parse_args()
    source=args.source.resolve()
    if args.output.resolve().is_relative_to(source):
        parser.error('Output must be outside the original source')
    if args.output.exists():
        parser.error('Output already exists; use a fresh path')
    workspace=args.workspace or Path(tempfile.mkdtemp(prefix='ffai-day34-'))
    if args.workspace and (not workspace.is_dir() or workspace.is_symlink() or any(workspace.iterdir())):
        parser.error('Workspace must be an empty regular directory')
    workspace=workspace.resolve()
    before=fingerprint(source,include_untracked=True),git(source,'status','--porcelain')
    isolate(workspace)
    if args.reasoning_effort is not None:
        os.environ['FFAI_REASONING_EFFORT']=args.reasoning_effort
    elif args.model != 'glm-5.3-flash':
        os.environ['FFAI_REASONING_EFFORT']=''
    os.environ['DAY34_DEMO_MODEL']=args.model
    from core import config
    if args.model not in config.model_choices():
        parser.error('Unknown model')
    from core.pr_review import redact
    result={'status':'running','workspace':'/path/to/demo','mode':'tui' if args.visible else 'facade'}
    start=time.monotonic()
    def cancel(signum,frame):
        raise RuntimeError('Демонстрация прервана')
    signal.signal(signal.SIGTERM,cancel)
    signal.signal(signal.SIGINT,cancel)
    try:
        project=prepare(source,workspace)
        controller=project/'core/controllers/filesandfoldersactionscontroller.cpp'
        result['behavior_before']=verify_behavior(controller,workspace/'probe-before')
        print('Проверка до изменений: '+json.dumps(result['behavior_before'],ensure_ascii=False),flush=True)
        (run_visible if args.visible else run_facade)(project,result,args.approve_demo)
        result['behavior_after']=verify_behavior(controller,workspace/'probe-after')
        print(json.dumps(result['behavior_after'],ensure_ascii=False),flush=True)
        if not result['behavior_after']['ok']:
            raise RuntimeError('Проверка поведения контроллера не пройдена')
        result['status']='ok'
        if args.visible:
            time.sleep(6)
    except Exception as error:
        result['status']='failed';result['error']=redact(str(error))
        print('Демо не завершено: '+result['error'],file=sys.stderr,flush=True)
        if Path(os.environ['FFAI_TASK_FILE']).exists():
            from core.task_state import TaskStore
            result['task_snapshot']=asdict(TaskStore(Path(os.environ['FFAI_TASK_FILE'])).load())
    finally:
        result['original_unchanged']=before==(fingerprint(source,include_untracked=True),git(source,'status','--porcelain'))
        if not result['original_unchanged']:
            result['status']='failed'
        result['elapsed_seconds']=time.monotonic()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        encoded=redact(json.dumps(result,ensure_ascii=False,indent=2,default=str))
        for location in (str(workspace),str(source),str(FFAI)):
            encoded=encoded.replace(location,'/path/to/'+('demo' if location==str(workspace) else 'source'))
        args.output.write_text(encoded+'\n')
        print('Артефакты: '+str(args.output),flush=True)
        print('Временная копия сохранена: '+str(workspace),flush=True)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
