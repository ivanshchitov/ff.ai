"""Read-only, commit-bound PR review pipeline, independent of interactive state."""

from __future__ import annotations

import os
import json
import difflib
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath

from . import code_index, code_retrieval, config, invariants
from .api_client import APIError


class ReviewError(Exception):
    """A named pipeline failure; never include raw input or API errors."""


@dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str
    old_path: str = ''


@dataclass(frozen=True)
class Changes:
    base: str
    head: str
    merge_base: str
    files: tuple[ChangedFile, ...]
    diff: str


@dataclass(frozen=True)
class Limits:
    max_diff_bytes: int = 200_000
    max_context_chars: int = 120_000
    max_files: int = 100
    max_snapshot_bytes: int = 20_000_000


# Redact literal credentials before indexing, requesting or rendering model output.
_SECRET_ASSIGNMENT = re.compile(
    r'''(?im)(\b[\w.-]*(?:secret|password|passwd|api[_-]?key|access[_-]?token|client[_-]?id)\b["']?\s*[:=]\s*)("[^"\n]*"|'[^'\n]*'|[^\s;,\\]+)''')
_BEARER = re.compile(r'(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]{8,}=*')


def redact(text: str) -> str:
    patterns = json.loads((config.ASSETS_DIR / 'pr_review_redaction.json').read_text())
    clean = _BEARER.sub(r'\1[REDACTED]', _SECRET_ASSIGNMENT.sub(r'\1"[REDACTED]"', text))
    return re.sub(patterns['private_key'],
                  lambda match: '\n'.join('[REDACTED PRIVATE KEY]' for _ in match.group().split('\n')),
                  clean, flags=re.S)


def _allowed(path: str, corpus) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or PurePosixPath(path).is_absolute() or any(
            p in ('.', '..', '.git') or p.startswith('.env') for p in parts):
        return False
    return (PurePosixPath(path).suffix.lower() in corpus.include_extensions
            and not code_index._excluded(path, corpus))


def _git(root: Path, *args: str) -> bytes:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GIT_') and not any(
               word in k.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))}
    env.update(GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0',
               GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS='1', GIT_LITERAL_PATHSPECS='1', LC_ALL='C')
    try:
        result = subprocess.run(
            ['git', '--no-optional-locks', '-C', str(root), *args],
            env=env, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReviewError('Не удалось выполнить чтение git.') from exc
    if result.returncode:
        raise ReviewError('Git не смог прочитать выбранные ревизии; проверьте репозиторий и историю.')
    return result.stdout


def collect_changes(root: Path, base: str, head: str) -> Changes:
    """Resolve commits first, then inspect their merge-base diff with NUL paths."""
    def resolve(revision):
        if not revision or revision.startswith('-'):
            raise ReviewError('Непригодная ревизия git.')
        return _git(root, 'rev-parse', '--verify', '--end-of-options',
                    revision + '^{commit}').decode().strip()

    base_sha, head_sha = resolve(base), resolve(head)
    ancestor = _git(root, 'merge-base', base_sha, head_sha).decode().strip()
    fields = _git(root, 'diff', '--no-ext-diff', '--no-textconv', '--name-status',
                  '-z', '-M', ancestor, head_sha, '--').split(b'\0')
    files = []
    pos = 0
    while pos < len(fields) and fields[pos]:
        status = fields[pos].decode('ascii')
        path = fields[pos + 1].decode('utf-8')
        pos += 2
        old_path = ''
        if status.startswith(('R', 'C')):
            old_path, path = path, fields[pos].decode('utf-8')
            pos += 1
        files.append(ChangedFile(status, path, old_path))
    diff = _git(root, 'diff', '--no-ext-diff', '--no-textconv', '--no-color',
                '-M', ancestor, head_sha, '--').decode('utf-8', errors='replace')
    return Changes(base_sha, head_sha, ancestor, tuple(files), diff)


def prepare_snapshot(root: Path, changes: Changes, corpus, destination: Path,
                     limits: Limits | None = None):
    """Materialize allowed regular blobs only, with bounded total size."""
    limits = limits or Limits()
    destination = Path(destination).resolve()
    if destination.is_relative_to(Path(root).resolve()):
        raise ReviewError('Снимок должен находиться вне проекта.')
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    limitations = []
    total = 0
    tree = _git(root, 'ls-tree', '-r', '-z', '-l', changes.head)
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, oid, size = metadata.split()
        path = raw_path.decode('utf-8')
        reason = ''
        if mode not in (b'100644', b'100755') or kind != b'blob':
            reason = 'не обычный файл'
        elif not _allowed(path, corpus):
            reason = 'вне правил корпуса'
        elif int(size) > corpus.max_file_bytes:
            reason = 'крупнее предела файла'
        if reason:
            limitations.append({'path': path, 'reason': reason})
            continue
        total += int(size)
        if total > limits.max_snapshot_bytes:
            raise ReviewError('Превышен предел размера корпуса.')
        data = _git(root, 'cat-file', 'blob', oid.decode('ascii'))
        try:
            text = data.decode('utf-8')
            if '\0' in text:
                raise UnicodeError()
        except UnicodeError:
            limitations.append({'path': path, 'reason': 'двоичный или не UTF-8 файл'})
            continue
        clean = redact(text)
        if clean != text:
            limitations.append({'path': path, 'reason': 'литеральные учётные данные скрыты'})
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean, encoding='utf-8')
    return destination, limitations


def retrieve_context(snapshot: Path, corpus, changes: Changes, index_dir: Path):
    """Independent baseline searches prevent code from displacing documents."""
    index_dir.mkdir(parents=True, exist_ok=True)
    query = ' '.join(f.path for f in changes.files) + '\n' + redact(changes.diff)
    sources, limitations = [], []
    for kind, extensions, extra in (
        ('documentation', tuple(x for x in corpus.include_extensions if x == '.md'),
         'README project structure conventions architecture'),
        ('code', tuple(x for x in corpus.include_extensions if x != '.md'), '')):
        rules = replace(corpus, include_extensions=extensions)
        database = index_dir / (kind + '.sqlite')
        indexed = code_index.build_index(snapshot, rules, corpus.strategy, database)
        search_query = query + '\n' + extra
        count = 4
        if kind == 'documentation':
            search_query = ' '.join(f.path for f in changes.files) + '\n' + extra
            count = max(4, indexed.chunks)
        report = code_retrieval.CodeRetriever(database, rules, ranking='tokens').search(
            search_query, mode='baseline', before=max(12, count), after=count)
        if not report.fragments:
            limitations.append({'kind': kind, 'reason': 'подходящих RAG-фрагментов нет'})
        seen = set()
        for fragment in report.fragments:
            source_path = re.sub(r':L\d+-L\d+$', '', fragment.source)
            if kind == 'documentation' and source_path in seen:
                continue
            seen.add(source_path)
            sources.append({'kind': kind, 'path': source_path,
                            'identifier': fragment.identifier,
                            'start_line': fragment.start_line, 'end_line': fragment.end_line,
                            'text': fragment.text, 'truncated': fragment.truncated})
            if len([s for s in sources if s['kind'] == kind]) == 4:
                break
    return sources, limitations


def _review_input(root, changes, corpus, snapshot, sources, limitations):
    def original_text(path):
        if not _allowed(path, corpus):
            return ''
        entries = _git(root, 'ls-tree', '-z', '-l', changes.merge_base, '--', path).split(b'\0')
        for entry in entries:
            if not entry:
                continue
            metadata, raw_path = entry.split(b'\t', 1)
            mode, kind, oid, size = metadata.split()
            if raw_path.decode('utf-8') != path:
                continue
            if mode not in (b'100644', b'100755') or kind != b'blob' or int(size) > corpus.max_file_bytes:
                limitations.append({'path': path, 'reason': 'содержимое base недоступно в корпусе'})
                return ''
            data = _git(root, 'cat-file', 'blob', oid.decode('ascii'))
            try:
                text = data.decode('utf-8')
                if '\0' in text:
                    raise UnicodeError()
            except UnicodeError:
                limitations.append({'path': path, 'reason': 'содержимое base не является UTF-8 текстом'})
                return ''
            clean = redact(text)
            if clean != text:
                limitations.append({'path': path, 'reason': 'литеральные учётные данные base скрыты'})
            return clean
        return ''

    diffs = []
    content = {}
    for item in changes.files:
        path = snapshot / item.path
        after = ''
        if _allowed(item.path, corpus) and path.is_file():
            after = content[item.path] = path.read_text(encoding='utf-8')
        elif not item.status.startswith('D'):
            limitations.append({'path': item.path, 'reason': 'содержимое head недоступно в корпусе'})
        before_path = item.old_path or item.path
        before = original_text(before_path)
        if before or after:
            # Redact complete blobs before diffing: a middle hunk of a key has no delimiters.
            diff = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                           for line in difflib.unified_diff(
                               before.splitlines(keepends=True), after.splitlines(keepends=True),
                               fromfile='a/' + before_path, tofile='b/' + item.path))
            diffs.append(f'diff --git {json.dumps("a/" + before_path)} {json.dumps("b/" + item.path)}\n'
                         + f'status: {item.status}\n' + diff)
    allowed_paths = sorted({path for item in changes.files for path in (item.path, item.old_path)
                            if path and _allowed(path, corpus)})
    if allowed_paths:
        metadata = _git(root, 'diff', '--no-ext-diff', '--no-textconv', '--summary', '-M',
                        changes.merge_base, changes.head, '--', *allowed_paths).decode('utf-8')
        if metadata:
            diffs.append('Git metadata:\n' + redact(metadata))
    return dict(base=changes.base, head=changes.head, merge_base=changes.merge_base,
                files=[asdict(f) for f in changes.files], diff=''.join(diffs),
                changed_content=content, sources=sources, limitations=limitations)


def parse_review_findings(content: str, known_lines: dict[str, int]):
    """Validate model data and derive both Markdown and the merge decision."""
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', content.strip(), re.S)
    try:
        data = json.loads(fenced.group(1) if fenced else content)
    except (ValueError, TypeError):
        raise ReviewError('Ответ модели не является JSON ревью.') from None
    names = ('potential_bugs', 'architectural_problems', 'recommendations')
    if not isinstance(data, dict) or set(data) != set(names) or any(
            not isinstance(data[name], list) for name in names):
        raise ReviewError('Ответ модели не содержит списки замечаний ревью.')
    for name in names[:2]:
        for finding in data[name]:
            if (not isinstance(finding, dict)
                or finding.get('severity') not in ('critical', 'high', 'medium', 'low')
                or not isinstance(finding.get('path'), str)
                or finding['path'] not in known_lines
                or type(finding.get('start_line')) is not int
                or type(finding.get('end_line')) is not int
                or not 1 <= finding['start_line'] <= finding['end_line'] <= known_lines[finding['path']]
                or any(not isinstance(finding.get(field), str) or not finding[field].strip()
                       for field in ('description', 'recommendation'))):
                raise ReviewError('Непригодное замечание модели: путь, строки или содержание.')
            finding['description'] = redact(finding['description'])
            finding['recommendation'] = redact(finding['recommendation'])
    if any(not isinstance(item, str) or not item.strip() for item in data['recommendations']):
        raise ReviewError('Непригодный список рекомендаций модели.')
    data['recommendations'] = [redact(item) for item in data['recommendations']]
    blocks = []
    for title, name in zip(('Потенциальные баги', 'Архитектурные проблемы'), names[:2]):
        lines = [f'- **{item["severity"]}** — `{item["path"]}:L{item["start_line"]}-L{item["end_line"]}`: '
                 f'{item["description"]}\n  Исправление: {item["recommendation"]}' for item in data[name]]
        blocks.append('## ' + title + '\n\n' + ('\n\n'.join(lines) or 'Не обнаружены.'))
    blocks.append('## Рекомендации\n\n' + ('\n'.join('- ' + s for s in data['recommendations'])
                                           or 'Нет рекомендаций.'))
    counts = {name: len(data[name]) for name in names[:2]}
    gate = {'status': 'blocked' if any(counts.values()) else 'passed', **counts}
    return data, '\n\n'.join(blocks) + '\n', gate


def run_review(root: Path, base: str, head: str, domain, output: Path, client,
               model: str, limits: Limits | None = None, on_stage=None) -> dict:
    """Execute the same pipeline in CI and locally; reports survive failed stages."""
    limits = limits or Limits()
    root, output = Path(root).resolve(), Path(output).resolve()
    toplevel = Path(_git(root, 'rev-parse', '--show-toplevel').decode('utf-8').strip()).resolve()
    if root != toplevel:
        raise ReviewError('Укажите корневой каталог git-репозитория.')
    if output.is_relative_to(root):
        raise ReviewError('Каталог результатов должен находиться вне проекта.')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(output.iterdir()):
        raise ReviewError('Каталог результатов должен быть пустым: прежнее ревью не перезаписывается.')
    report = dict(status='running', base='', head='', merge_base='', model=model,
                  stages=[], files=[], sources=[], limitations=[], usage=None)
    stage = None
    started = time.monotonic()

    def begin(name):
        nonlocal stage, started
        if stage is not None:
            stage.update(status='ok', elapsed_seconds=round(time.monotonic() - started, 3))
        started = time.monotonic()
        stage = dict(name=name, status='running')
        report['stages'].append(stage)
        if on_stage:
            on_stage(name)

    try:
        begin('changes')
        if domain.corpus is None:
            raise ReviewError('Домен не объявляет корпус проекта.')
        changes = collect_changes(root, base, head)
        report.update(base=changes.base, head=changes.head, merge_base=changes.merge_base,
                      files=[asdict(f) for f in changes.files])
        if len(changes.files) > limits.max_files or len(changes.diff.encode()) > limits.max_diff_bytes:
            raise ReviewError('Превышен предел числа файлов или размера diff.')
        # Put all temporary data alongside the result, never in a user-configured state directory.
        with tempfile.TemporaryDirectory(prefix='.ffai-review-', dir=output.parent) as private:
            work = Path(private)
            snapshot, limitations = prepare_snapshot(root, changes, domain.corpus,
                                                     work / 'snapshot', limits)
            report['limitations'] = limitations
            begin('retrieval')
            # Redacted, allowed diff also serves as the search query.
            context = _review_input(root, changes, domain.corpus, snapshot, [], limitations)
            safe_changes = replace(changes, diff=context['diff'])
            sources, missing = retrieve_context(snapshot, domain.corpus, safe_changes, work / 'index')
            report['sources'] = sources
            limitations.extend(missing)
            context.update(sources=sources)
            payload = json.dumps(context, ensure_ascii=False)
            if len(payload) > limits.max_context_chars:
                raise ReviewError('Превышен предел контекста ревью.')
            begin('review')
            instruction = (config.ASSETS_DIR / 'pr_review_prompt.md').read_text(encoding='utf-8')
            rules = '\n'.join(f'{rule.number}. {rule.rule}' for rule in domain.invariants)
            messages = [{'role': 'system', 'content': domain.prompt('system') + '\n' + rules + '\n' + instruction},
                        {'role': 'user', 'content': payload}]
            try:
                meta = client.ask_with_usage_messages(messages, max_tokens=6000,
                                                       temperature=0.1, model=model)
            except APIError as exc:
                raise ReviewError('API модели недоступен; ревью не выполнено.') from exc
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ReviewError('API модели вернул непригодные данные; ревью не выполнено.') from exc
            if not isinstance(meta.content, str):
                raise ReviewError('API модели вернул непригодный текст; ревью не выполнено.')
            report['usage'] = asdict(meta)
            report['usage'].pop('content')
            if meta.finish_reason not in ('stop', None):
                raise ReviewError('Ответ модели не завершён; ревью не выполнено полностью.')
            known_paths = {item.path for item in changes.files} | {s['path'] for s in sources}
            known_lines = {path: len((snapshot / path).read_text(encoding='utf-8').splitlines())
                           for path in known_paths if (snapshot / path).is_file()}
            findings, text, gate = parse_review_findings(meta.content, known_lines)
            report.update(findings=findings, gate=gate)
            if invariants.check_answer(text, domain.invariants):
                raise ReviewError('Ответ модели нарушает правила домена.')
            source_lines = '\n'.join(f'- {s["kind"]}: {s["identifier"]}' for s in sources)
            result = (text + '\n\n## Источники RAG\n\n' + (source_lines or 'Фрагментов нет.')
                      + '\n\nHead: `' + changes.head + '`\n')
            (output / 'review.md').write_text(result, encoding='utf-8')
            report['status'] = 'ok'
    except (ReviewError, OSError, UnicodeError, code_index.CodeIndexError) as exc:
        report['status'] = 'failed'
        message = str(exc) if isinstance(exc, ReviewError) else 'Не удалось прочитать или сохранить данные ревью.'
        report['error'] = message
        if stage:
            stage['status'] = 'failed'
        raise ReviewError(message) from exc
    finally:
        if stage:
            if report['status'] == 'ok':
                stage['status'] = 'ok'
            stage['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
