"""Read-only, commit-bound PR review pipeline, independent of interactive state."""

from __future__ import annotations

import os
import json
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
    return re.sub(patterns['private_key'], '[REDACTED PRIVATE KEY]', clean, flags=re.S)


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
        code_index.build_index(snapshot, rules, corpus.strategy, database)
        report = code_retrieval.CodeRetriever(database, rules, ranking='tokens').search(
            query + '\n' + extra, mode='baseline', before=12, after=4)
        if not report.fragments:
            limitations.append({'kind': kind, 'reason': 'подходящих RAG-фрагментов нет'})
        for fragment in report.fragments:
            sources.append({'kind': kind, 'path': re.sub(r':L\d+-L\d+$', '', fragment.source),
                            'identifier': fragment.identifier,
                            'start_line': fragment.start_line, 'end_line': fragment.end_line,
                            'text': fragment.text, 'truncated': fragment.truncated})
    return sources, limitations


def _review_input(root, changes, corpus, snapshot, sources, limitations):
    paths = sorted({path for f in changes.files for path in (f.path, f.old_path)
                    if path and _allowed(path, corpus)})
    diff = ''
    if paths:
        diff = redact(_git(root, 'diff', '--no-ext-diff', '--no-textconv', '--no-color',
                           '-M', changes.merge_base, changes.head, '--', *paths)
                      .decode('utf-8', errors='replace'))
    content = {}
    for item in changes.files:
        path = snapshot / item.path
        if _allowed(item.path, corpus) and path.is_file():
            content[item.path] = path.read_text(encoding='utf-8')
        elif not item.status.startswith('D'):
            limitations.append({'path': item.path, 'reason': 'содержимое head недоступно в корпусе'})
    return dict(base=changes.base, head=changes.head, merge_base=changes.merge_base,
                files=[asdict(f) for f in changes.files], diff=diff,
                changed_content=content, sources=sources, limitations=limitations)


def run_review(root: Path, base: str, head: str, domain, output: Path, client,
               model: str, limits: Limits | None = None, on_stage=None) -> dict:
    """Execute the same pipeline in CI and locally; reports survive failed stages."""
    limits = limits or Limits()
    root, output = Path(root).resolve(), Path(output).resolve()
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
            report['usage'] = asdict(meta)
            report['usage'].pop('content')
            text = redact(meta.content.strip())
            headings = ('Потенциальные баги', 'Архитектурные проблемы', 'Рекомендации')
            for heading in headings:
                section = re.search(r'^#{1,3}[ \t]+' + heading +
                                    r'[ \t]*\n(.*?)(?=^#{1,3}[ \t]+|\Z)', text, re.M | re.S)
                if section is None or not section.group(1).strip():
                    raise ReviewError('Ответ модели не содержит обязательные разделы ревью.')
            if meta.finish_reason not in ('stop', None):
                raise ReviewError('Ответ модели не завершён; ревью не выполнено полностью.')
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
