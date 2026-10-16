"""Demonstrate release preparation with a live model in disposable Git histories.

--approve-demo-history consents only to init/add/commit in fresh copied projects.
No tags, publication, source edits or synthetic build results are performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time

FFAI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FFAI))
from scripts.day34_demo import CONTROLLER, fingerprint, git, isolate, prepare

GUARD = '    if (connectionInProgress()) return;\n'
METHOD = 'void FilesAndFoldersActionsController::initCreateFolderOperation()\n{\n'
STATUSES = ('blocked', 'incomplete', 'ready_for_manual_release')


def source_state(source):
    """Include refs/index evidence as well as tracked and untracked file bytes."""
    return (fingerprint(source, include_untracked=True), git(source, 'status', '--porcelain'),
            git(source, 'for-each-ref', '--format=%(refname) %(objectname)'), git(source, 'ls-files', '--stage'))


def commit(project, message):
    git(project, 'add', '--all')
    git(project, '-c', 'user.name=ff.ai demo', '-c', 'user.email=demo@example.invalid',
        '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
        'commit', '-q', '-m', message)


def prepare_scenarios(source, workspace, *, approved):
    if not approved:
        raise ValueError('Temporary history requires --approve-demo-history')
    source, workspace = source.resolve(), workspace.resolve()
    if workspace.is_relative_to(source) or source.is_relative_to(workspace):
        raise ValueError('Workspace must be outside the original source')
    scenarios = []
    for name in ('bad', 'good'):
        directory = workspace / name
        directory.mkdir()
        project = prepare(source, directory)
        controller = project / CONTROLLER
        original = controller.read_text()
        if original.count(METHOD) != 1:
            raise RuntimeError('Unknown controller fixture; refusing to invent a change')
        baseline = original.replace(METHOD + GUARD, METHOD)
        controller.write_text(baseline)
        specs = sorted(project.glob('rpm/*.spec'))
        if len(specs) != 1 or not re.search(r'^Version:\s*\S+', specs[0].read_text(), re.M):
            raise RuntimeError('Demo requires exactly one RPM spec with Version')
        commit(project, 'demo: baseline before duplicate-folder guard')
        base = git(project, 'rev-parse', 'HEAD').strip()
        controller.write_text(baseline.replace(METHOD, METHOD + GUARD))
        if name == 'bad':
            specs[0].write_text(re.sub(r'^Version:.*\n?', '', specs[0].read_text(), flags=re.M))
        commit(project, 'demo: protect folder creation' + (' with missing package version' if name == 'bad' else ''))
        scenarios.append({'name': name, 'project': project, 'base': base,
                          'head': git(project, 'rev-parse', 'HEAD').strip()})
    return scenarios


def result_status(lines):
    matches = re.findall(r'^Релиз:\s*(' + '|'.join(STATUSES) + r')\s*$', '\n'.join(lines), re.M)
    if not matches:
        raise RuntimeError('No finalized release result was rendered')
    return matches[-1]


def fresh_result(before, current):
    marker = r'^Релиз:\s*(?:' + '|'.join(STATUSES) + r')\s*$'
    return (len(re.findall(marker, current, re.M)) > len(re.findall(marker, before, re.M))
            and current.count('Каталог результатов:') > before.count('Каталог результатов:'))


def sanitize(value, locations):
    from core.pr_review import redact
    if isinstance(value, dict):
        return {str(key): sanitize(item, locations) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, locations) for item in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    # Sanitize strings before JSON encoding: redaction can itself insert quotes.
    encoded = redact(str(value))
    encoded = re.sub(r'(?i)(\b[\w.-]*token\b\s*[:=]\s*)("[^"\n]*"|\x27[^\x27\n]*\x27|[^\s;,]+)',
                     r'\1[REDACTED]', encoded)
    for location, placeholder in sorted(locations.items(), key=lambda pair: -len(str(pair[0]))):
        encoded = encoded.replace(str(location), placeholder)
    return encoded


def saved_checks(workspace):
    reports = []
    for path in sorted(workspace.glob('*/state/exports/releases/*/checks.json')):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            reports.append({'error': 'Saved checks unreadable', 'path': str(path)})
    return reports


def run_facade(scenario, model, arch):
    from core.session import AssistantSession
    session = AssistantSession(scenario['project'], domain_id='aurora-qt5')
    session.model = model
    session.settings = session.settings.with_temperature(0)
    command = f"/release prepare --base {scenario['base']} --head HEAD" + (f' --arch {arch}' if arch else '')
    print(command, flush=True)
    lines = list(session.run_command(command).lines)
    print('\n'.join(lines), flush=True)
    status = result_status(lines)
    requests = session.session_usage.requests
    if requests < 2:
        raise RuntimeError('Both live model stages must run; inspect release artifacts')
    usage = list(session.usage_lines())
    restarted = AssistantSession(scenario['project'], domain_id='aurora-qt5')
    restored = list(restarted.run_command('/release status').lines)
    if result_status(restored) != status or restarted.session_usage.requests:
        raise RuntimeError('Restart did not restore status without a model request')
    return {'status': status, 'lines': lines, 'usage': usage, 'restart_lines': restored,
            'restart_requests': restarted.session_usage.requests}


def run_visible(scenario, model, arch):
    from core import config
    from tests.e2e.harness import AppSession
    state = Path(os.environ['FFAI_STATE_DIR'])
    size = shutil.get_terminal_size((110, 42))
    captures = []
    def mirror(data):
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    def start():
        return AppSession(history_file=state / 'history.json', state_dir=state,
                          exports_dir=Path(os.environ['FFAI_EXPORTS_DIR']),
                          repo=scenario['project'], domain='aurora-qt5', api_key=None,
                          mcp_command=os.environ['FFAI_MCP_COMMAND'], mcp_args=os.environ['FFAI_MCP_ARGS'],
                          extra_env=dict(os.environ), cols=size.columns, rows=size.lines, mirror=mirror)
    def wait_prompt(app, timeout=30):
        deadline = time.monotonic() + timeout
        while not app.screen_text().rstrip().endswith('Вы:'):
            if time.monotonic() > deadline:
                raise TimeoutError('TUI did not return to input')
            time.sleep(.05)
    def command(app, text, timeout=600):
        wait_prompt(app)
        before = app.scrollback()
        app.send_line(text)
        deadline = time.monotonic() + timeout
        while not fresh_result(before, app.scrollback()):
            if time.monotonic() > deadline:
                raise TimeoutError('No new rendered result for ' + text)
            app.screen_text()
            time.sleep(.05)
        wait_prompt(app)
        time.sleep(2)
        return app.scrollback()
    with start() as app:
        wait_prompt(app)
        app.send_line('/models')
        app.wait_for('↑/↓', timeout=20)
        time.sleep(.4)
        for _ in range(config.model_choices().index(model)):
            app.send_key(b'\x1b[B')
        app.send_key(b'\r')
        wait_prompt(app)
        text = f"/release prepare --base {scenario['base']} --head HEAD" + (f' --arch {arch}' if arch else '')
        output = command(app, text)
        status = result_status(output.splitlines())
        app.send_line('/usage')
        app.wait_for('Сессия: запросов', timeout=20)
        wait_prompt(app)
        usage = app.scrollback()
        counts = re.findall(r'Сессия: запросов\s*—\s*(\d+)', usage)
        if not counts or int(counts[-1]) < 2:
            raise RuntimeError('Both live model stages were not observed in TUI usage')
        captures.append(app.scrollback())
        app.send_line('/exit')
        app.wait_exit()
    with start() as app:
        restored = command(app, '/release status', timeout=30)
        if result_status(restored.splitlines()) != status:
            raise RuntimeError('Restart changed release status')
        app.send_line('/usage')
        app.wait_for('Сессия: запросов', timeout=20)
        wait_prompt(app)
        counts = re.findall(r'Сессия: запросов\s*—\s*(\d+)', app.scrollback())
        if not counts or int(counts[-1]) != 0:
            raise RuntimeError('Reading saved status performed a model request')
        captures.append(app.scrollback())
        app.send_line('/exit')
        app.wait_exit()
    return {'status': status, 'screen': captures, 'restart_requests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=FFAI.parent / 'WebDavClient')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--output', type=Path, default=FFAI / 'videos/day35-dry-run.json')
    parser.add_argument('--approve-demo-history', action='store_true')
    parser.add_argument('--visible', action='store_true')
    parser.add_argument('--model', default='glm-5.3-flash')
    parser.add_argument('--arch')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare fixtures without model or SDK calls')
    args = parser.parse_args()
    if not args.approve_demo_history:
        parser.error('Use --approve-demo-history only after consenting to disposable-copy history')
    source = args.source.resolve()
    output = args.output.resolve()
    if output.is_relative_to(source) or output.exists():
        parser.error('Output must be a fresh path outside the original source')
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix='ffai-day35-'))
    if workspace.is_symlink() or not workspace.is_dir() or any(workspace.iterdir()):
        parser.error('Workspace must be an empty regular directory')
    workspace = workspace.resolve()
    if workspace.is_relative_to(source) or source.is_relative_to(workspace):
        parser.error('Workspace must be outside the original source')
    before = source_state(source)
    result = {'status': 'running', 'mode': 'tui' if args.visible else 'facade', 'scenarios': []}
    start = time.monotonic()
    def cancel(signum, frame):
        raise RuntimeError('Demonstration interrupted')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        scenarios = prepare_scenarios(source, workspace, approved=args.approve_demo_history)
        for scenario in scenarios:
            isolate(workspace / scenario['name'])
            # prepare() configured a local MCP endpoint; retain no external host tools.
            os.environ['FFAI_MCP_COMMAND'] = sys.executable
            os.environ['FFAI_MCP_ARGS'] = (f"{FFAI}/mcp_server/repo_server.py --root {scenario['project']} "
                                          f"--tools {workspace / scenario['name'] / 'tools.json'}")
            from core import config
            if args.model not in config.model_choices():
                raise ValueError('Unknown configured model: ' + args.model)
            item = dict(scenario)
            result['scenarios'].append(item)
            if args.prepare_only:
                item['status'] = 'prepared'
                continue
            project_before = source_state(scenario['project'])
            item.update((run_visible if args.visible else run_facade)(scenario, args.model, args.arch))
            item['project_unchanged'] = source_state(scenario['project']) == project_before
            exports = Path(os.environ['FFAI_EXPORTS_DIR'])
            item['checks'] = [json.loads(path.read_text()) for path in sorted(exports.glob('releases/*/checks.json'))]
            if not item['checks'] or not item['project_unchanged']:
                raise RuntimeError('No saved checks or copied input changed during preparation')
            expected = ('blocked',) if scenario['name'] == 'bad' else ('incomplete', 'ready_for_manual_release')
            if item['status'] not in expected:
                raise RuntimeError('Unexpected ' + scenario['name'] + ' result: ' + item['status'])
        result['status'] = 'prepared' if args.prepare_only else 'ok'
    except Exception as error:
        result.update(status='failed', error=str(error))
        # Keep partial evidence when an AI stage or SDK invocation failed.
        result['partial_checks'] = saved_checks(workspace)
    finally:
        result['original_unchanged'] = source_state(source) == before
        if not result['original_unchanged']:
            result['status'] = 'failed'
        result['elapsed_seconds'] = time.monotonic() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(sanitize(result, {workspace: '/path/to/demo', source: '/path/to/source',
                                                      FFAI: '/path/to/ff.ai', Path.home(): '/path/to/home'}), ensure_ascii=False, indent=2) + '\n')
        print('Артефакты: ' + str(output), flush=True)
        print('Временная копия сохранена: ' + str(workspace), flush=True)
    return 0 if result['status'] in ('ok', 'prepared') else 1


if __name__ == '__main__':
    raise SystemExit(main())
