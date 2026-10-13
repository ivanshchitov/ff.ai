"""Reproduce day 32 on an isolated WebDavClient copy with a real model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

FFAI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FFAI))
ARTIFACTS = FFAI / 'videos'
CONTROL = ARTIFACTS / 'tmp/day32'
CHANGE_PATH = 'core/operations/createfolderoperation.cpp'


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args]).decode().strip()


def fingerprint(root):
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).split(b'\0')
    return {name.decode(): hashlib.sha256((root / name.decode()).read_bytes()).hexdigest()
            for name in names if name and (root / name.decode()).is_file()
            and not (root / name.decode()).is_symlink()}


def prepare(source):
    from core.pr_review import redact
    workspace = Path(tempfile.mkdtemp(prefix='ffai-day32-'))
    project = workspace / 'WebDavClient'
    project.mkdir()
    for name in fingerprint(source):
        path = Path(name)
        if any(part.startswith('.env') or part == '.git' for part in path.parts):
            continue
        destination = project / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (source / path).read_bytes()
        try:
            data = redact(data.decode('utf-8')).encode('utf-8')
        except UnicodeError:
            pass
        destination.write_bytes(data)
    git(project, 'init', '-q', '-b', 'day32-demo')
    git(project, 'add', '.')
    git(project, '-c', 'user.name=ff.ai demo', '-c', 'user.email=demo@example.invalid',
        'commit', '-qm', 'WebDavClient sanitized demo baseline')
    base = git(project, 'rev-parse', 'HEAD')
    target = project / CHANGE_PATH
    original = target.read_text()
    marker = 'reply->error() == QNetworkReply::NetworkError::NoError'
    if original.count(marker) != 1:
        raise RuntimeError('Demo needs exactly one original error check.')
    target.write_text(original.replace(marker, marker.replace(' == ', ' != ')))
    git(project, 'add', CHANGE_PATH)
    git(project, '-c', 'user.name=ff.ai demo', '-c', 'user.email=demo@example.invalid',
        'commit', '-qm', 'Introduce demo error-check regression')
    return workspace, project, base, git(project, 'rev-parse', 'HEAD')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=FFAI.parent / 'WebDavClient')
    parser.add_argument('--visible', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    before = fingerprint(source)
    status = git(source, 'status', '--porcelain')
    ARTIFACTS.mkdir(exist_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=True)
    workspace, project, base, head = prepare(source)
    (CONTROL / 'setup.json').write_text(json.dumps(dict(workspace=str(workspace),
                                                      project=str(project), base=base, head=head)))
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    console = Console(width=110)

    def page(title):
        if args.visible:
            console.clear()
        console.rule(title)

    def reading_pause(seconds=6):
        if args.visible:
            time.sleep(seconds)  # Deliberate reading time, never model-completion detection.

    process = None
    try:
        page('ff.ai — День 32. Автоматизация ревью кода')
        console.print('Проект: личный WebDavClient. Изолированная копия. Настоящая модель.')
        console.print('Локальное воспроизведение конвейера; удалённый Actions run ещё не выполнялся.')
        if args.visible:
            (CONTROL / 'ready').touch()
            deadline = time.monotonic() + 300
            while not (CONTROL / 'go').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Recorder did not start.')
                time.sleep(.1)
        reading_pause(3)
        page('1. PR-trigger в GitHub Actions')
        console.print(Syntax((FFAI / 'examples/webdavclient-ai-review.yml').read_text(), 'yaml'))
        reading_pause(8)
        page('2. Diff и содержимое изменённого файла на head')
        console.print('Внесённая регрессия: инверсия проверки ошибки создания папки.')
        console.print(Syntax(git(project, 'diff', base, head, '--', CHANGE_PATH), 'diff'))
        console.print('base:', base, '\nhead:', head)
        reading_pause(9)
        page('3. Реальный запуск ff.ai: changes → retrieval → review')
        env = {k: v for k, v in os.environ.items() if not k.startswith('FFAI_')}
        state, cache = workspace / 'state', workspace / 'cache'
        env.update(FFAI_STATE_DIR=str(state), FFAI_CACHE_DIR=str(cache),
                   FFAI_LLAMA_AUTOSTART='0', FFAI_NO_REEXEC='1')
        for key, name in {'HISTORY_FILE': 'history.json', 'MEMORY_FILE': 'memory.json',
                          'PROFILE_FILE': 'profile.json', 'TASK_FILE': 'task.json',
                          'SCHEDULE_FILE': 'schedule.json', 'TASKS_DIR': 'tasks',
                          'EXPORTS_DIR': 'exports', 'CHANGES_DIR': 'changes',
                          'INDEX_FILE': 'index.sqlite'}.items():
            env['FFAI_' + key] = str(state / name)
        output = workspace / 'review'
        command = [sys.executable, str(FFAI / 'ff-ai-review.py'), '--repo', str(project),
                   '--domain', 'aurora-qt5', '--base', base, '--head', head, '--output', str(output)]
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        # Completion is the CLI process exit and report status, never a fixed sleep.
        deadline = time.monotonic() + 420
        import selectors
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError('Model review exceeded demo limit.')
                for key, _ in selector.select(timeout=.2):
                    line = key.fileobj.readline()
                    if line:
                        console.print(line.rstrip(), markup=False)
            for line in process.stdout:
                console.print(line.rstrip(), markup=False)
        if process.returncode:
            raise RuntimeError('Review process failed; see its report.')
        report = json.loads((output / 'report.json').read_text())
        review = (output / 'review.md').read_text()
        if report['status'] != 'ok' or {s['kind'] for s in report['sources']} != {'code', 'documentation'}:
            raise RuntimeError('Demo did not deliver both RAG corpora.')
        if CHANGE_PATH not in review or not any(word in review.lower() for word in ('инвер', 'ошиб', 'успех')):
            raise RuntimeError('Review did not identify the demonstrated change.')
        mode = 'recorded' if args.visible else 'dry'
        for name in ('review.md', 'report.json'):
            shutil.copy2(output / name, ARTIFACTS / f'day32-{mode}-{name}')
        page('4. RAG: документация и связанный код')
        for item in report['sources']:
            console.print(f'{item["kind"]}: {item["identifier"]}', markup=False)
        console.print('\nОтдельная квота документов и кода. Поиск baseline, токенный.')
        reading_pause(9)
        page('5. Ревью: потенциальные баги')
        bug, rest = review.split('## Архитектурные проблемы', 1)
        console.print(Markdown(bug))
        reading_pause(14)
        page('6. Архитектурные проблемы и рекомендации')
        console.print(Markdown('## Архитектурные проблемы' + rest.split('## Источники RAG')[0]))
        reading_pause(14)
        page('7. Отчёт этапов и фактические метрики')
        console.print(json.dumps({key: report[key] for key in ('status', 'stages', 'usage')},
                                 ensure_ascii=False, indent=2), markup=False)
        console.print('review.md + report.json — те же результаты сохраняет workflow как артефакты.')
        console.print('Исходный WebDavClient сохранён. Следующий шаг: публикация и фактический PR-run.')
        reading_pause(10)
        if fingerprint(source) != before or git(source, 'status', '--porcelain') != status:
            raise RuntimeError('Source project changed during the demonstration.')
    except Exception:
        (CONTROL / 'failure').touch()
        raise
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        (CONTROL / 'finished').touch()
        # The workspace stays available until recorder verification and manual cleanup.


if __name__ == '__main__':
    main()
