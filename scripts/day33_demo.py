"""Live support demonstration with disposable product data and isolated FFAI state.

Default: dry run through the actual facade, saved answers/trace/usage.
--visible: drive ff-ai.py TUI in a terminal; record that terminal with a screen recorder.
"""
from __future__ import annotations
from contextlib import nullcontext
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

FFAI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FFAI))
QUESTIONS = [
    '/support TICKET-101 Почему не получается войти и что мне сделать?',
    '/support TICKET-102 Почему не получается войти и что мне сделать?',
    '/support TICKET-103 Что выбрать, чтобы файл остался на сервере?',
    '/support TICKET-102 Подключение уже работает. Как обновить список файлов?',
]

def prepare(workspace):
    project = workspace / 'WebDavClient-support'
    shutil.copytree(FFAI / 'examples/support-webdavclient', project)
    subprocess.run(['git','init','-q',str(project)], check=True)
    # Clear inherited explicit paths before importing config, then cover all state paths.
    for key in list(os.environ):
        if key.startswith('FFAI_'): os.environ.pop(key)
    state = workspace / 'state'
    os.environ.update(FFAI_STATE_DIR=str(state), FFAI_CACHE_DIR=str(workspace/'cache'),
                      FFAI_LLAMA_AUTOSTART='0', FFAI_NO_REEXEC='1')
    for key, name in {'HISTORY_FILE':'history.json','MEMORY_FILE':'memory.json',
                      'PROFILE_FILE':'profile.json','TASK_FILE':'task.json',
                      'SCHEDULE_FILE':'schedule.json','TASKS_DIR':'tasks',
                      'EXPORTS_DIR':'exports','CHANGES_DIR':'changes',
                      'INDEX_FILE':'index.sqlite'}.items():
        os.environ['FFAI_'+key] = str(state/name)
    return project

def resolve_ticket(project):
    path=project/'support.json'
    data=json.loads(path.read_text())
    ticket=data['tickets'][1]
    ticket.update(status='resolved',error_code='',subject='Обновить список файлов',
                  description='Повторил добавление с /dav/, учётная запись создана, список файлов открылся. Как обновить список после изменения на другом устройстве?')
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))

class DemoCancelled(RuntimeError):
    """Cancellation unwinds the TUI context before the recording workspace is removed."""


def install_signal_handlers():
    def cancel(signum, frame):
        raise DemoCancelled('Демонстрация прервана.')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)


def workspace_context(args):
    if args.workspace is not None:
        path=args.workspace
        if not args.visible or not path.is_dir() or path.is_symlink() or any(path.iterdir()):
            raise ValueError('Рабочий каталог записи должен быть пустым обычным каталогом; нужен --visible.')
        return nullcontext(str(path.resolve()))
    if args.visible:
        return nullcontext(tempfile.mkdtemp(prefix='ffai-day33-'))
    return tempfile.TemporaryDirectory(prefix='ffai-day33-')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--visible',action='store_true')
    parser.add_argument('--workspace',type=Path,help='Empty workspace allocated by the recording launcher; kept until recorder stops.')
    parser.add_argument('--output',type=Path,default=FFAI/'videos/day33-dry-run.json')
    args=parser.parse_args()
    install_signal_handlers()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        parser.error('Output already exists; choose a fresh --output path.')
    results={'status':'running','answers':[],'mode':'tui' if args.visible else 'facade'}
    started=time.monotonic()
    try:
        context=workspace_context(args)
    except ValueError as error:
        parser.error(str(error))
    with context as name:
        project=prepare(Path(name))
        # config loads the local key without printing it; no user state is touched.
        from core import config
        from core.pr_review import redact
        try:
            if args.visible:
                from tests.e2e.harness import AppSession
                def mirror(data):
                    sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
                size=shutil.get_terminal_size(fallback=(110,45))
                with AppSession(history_file=Path(os.environ['FFAI_HISTORY_FILE']),
                                state_dir=Path(os.environ['FFAI_STATE_DIR']),
                                repo=project, domain='product-support', api_key=None,
                                mcp_command='',mcp_args='',cols=size.columns,rows=size.lines,
                                extra_env=dict(os.environ), mirror=mirror) as app:
                    app.send_line('/rag-code index');app.wait_for('Файлов:',timeout=30)
                    for index, command in enumerate(QUESTIONS):
                        if index==3: resolve_ticket(project)
                        app.ask(command,timeout=150)
                        time.sleep(7)  # Reading time after confirmed completion, not a completion detector.
                    app.send_line('/support TICKET-999 вопрос');app.wait_for('Тикет не найден',timeout=30)
                    for command in ['/rag-code trace','/usage']:
                        app.send_line(command);time.sleep(7)
                    app.send_line('/exit')
                    app.wait_exit()
                    results['screen']=redact(app.scrollback())
            else:
                from core.session import AssistantSession
                session=AssistantSession(project, domain_id='product-support')
                results['index']=list(session.run_command('/rag-code index').lines)
                for index, command in enumerate(QUESTIONS):
                    if index==3: resolve_ticket(project)
                    print(command,flush=True)
                    result=session.run_command(command)
                    if result.answer is None: raise RuntimeError('Support answer unavailable: '+' '.join(result.lines))
                    meta=result.answer.meta
                    results['answers'].append({'command':command,'answer':redact(result.answer.text),
                        'context':list(result.lines),'trace':list(session.code_trace_lines()),
                        'usage':{'model':meta.model,'prompt_tokens':meta.prompt_tokens,
                                 'completion_tokens':meta.completion_tokens,'total_tokens':meta.total_tokens,
                                 'elapsed_seconds':meta.elapsed_seconds,'cost_usd':meta.cost_usd}})
                    print(redact(result.answer.text),flush=True)
                failed=session.run_command('/support TICKET-999 вопрос')
                if failed.answer is not None: raise RuntimeError('Unknown ticket unexpectedly answered.')
                results['unknown_ticket']=list(failed.lines)
                results['usage']=list(session.usage_lines())
            results['status']='ok'
        except Exception as error:
            results['status']='failed'
            results['error']=redact(str(error))
            print('Демо недоступно: '+results['error'],file=sys.stderr)
        finally:
            results['elapsed_seconds']=time.monotonic()-started
            # Persist artifacts before disposing only this run's own temporary workspace.
            encoded=json.dumps(results,ensure_ascii=False,indent=2)
            for location in (str(Path(name)),str(Path(name).resolve())):
                encoded=encoded.replace(location,'/path/to/demo')
            args.output.write_text(encoded+'\n')
            if args.visible:
                print('После остановки рекордера удалите только временный каталог этого демо: '+name)
    return 0 if results['status']=='ok' else 1

if __name__=='__main__':
    raise SystemExit(main())
