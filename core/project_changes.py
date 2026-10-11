"""Confirmed text changes, private durable snapshots and conservative recovery."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from . import config, patches


class ChangeError(Exception):
    """Invalid change, journal or project boundary."""


@dataclass(frozen=True)
class FileState:
    exists: bool
    bytes: bytes = b''
    mode: int = 0

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(str((self.exists, self.mode)).encode() + self.bytes).hexdigest()


@dataclass
class ChangeRecord:
    id: str
    task_number: Optional[int]
    root_identity: str
    patch: str
    paths: Tuple[str, ...]
    before: Dict[str, FileState]
    after: Dict[str, FileState]
    created_dirs: Tuple[str, ...] = ()
    status: str = 'prepared'
    reason: str = ''
    validation: Tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ChangeResult:
    ok: bool
    record: ChangeRecord
    reason: str = ''


class ProjectChanges:
    def __init__(self, root: Path, changes_dir: Optional[Path] = None,
                 validate: Optional[Callable[[str], str]] = None):
        self.root = Path(root).resolve()
        self.changes_dir = Path(changes_dir or config.project_paths(self.root)['changes_dir']).resolve()
        if self.changes_dir == self.root or self.changes_dir.is_relative_to(self.root):
            raise ChangeError('журнал изменений должен быть вне проекта')
        self.root_identity = hashlib.sha256(str(self.root).encode()).hexdigest()
        self.validate = validate

    @staticmethod
    def _read(path: Path) -> FileState:
        if not path.exists():
            return FileState(False)
        content = path.read_bytes()
        if b'\0' in content:
            raise ChangeError('binary-файл не поддерживается')
        try:
            content.decode('utf-8')
        except UnicodeError as error:
            raise ChangeError('нужен UTF-8 текст') from error
        return FileState(True, content, stat.S_IMODE(path.stat().st_mode))

    def _states(self, names) -> Dict[str, FileState]:
        try:
            return {name: self._read(patches.safe_path(self.root, name)) for name in names}
        except (OSError, patches.PatchError) as error:
            raise ChangeError(str(error)) from error

    def _check(self, text: str):
        try:
            parsed = patches.validate_text_patch(text, self.root)
        except patches.PatchError as error:
            raise ChangeError(str(error)) from error
        reason = self.validate(text) if self.validate else patches.check(self.root, text)
        if reason:
            raise ChangeError(reason)
        return parsed

    def prepare(self, patch_text: str, task_number=None) -> ChangeRecord:
        parsed = self._check(patch_text)
        names = tuple(dict.fromkeys(parsed.paths))
        before = self._states(names)
        created_dirs = set()
        for name in names:
            parent = (self.root / name).parent
            while parent != self.root and not parent.exists():
                created_dirs.add(parent.relative_to(self.root).as_posix())
                parent = parent.parent
        # Compute exact postimage without writing the target or its Git index.
        with tempfile.TemporaryDirectory(prefix='ffai-patch-') as directory:
            sandbox = Path(directory)
            for name, state in before.items():
                if state.exists:
                    path = sandbox / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(state.bytes)
                    path.chmod(state.mode)
            reason = patches.apply(sandbox, patch_text)
            if reason:
                raise ChangeError(reason)
            after = {name: self._read(sandbox / name) for name in names}
        record = ChangeRecord(uuid.uuid4().hex[:16], task_number, self.root_identity,
                              patch_text, names, before, after,
                              tuple(sorted(created_dirs, key=lambda p: (-len(Path(p).parts), p))))
        self._save(record)
        return record

    def _file(self, change_id: str) -> Path:
        if len(change_id) != 16 or any(c not in '0123456789abcdef' for c in change_id):
            raise ChangeError('неизвестный id изменения')
        return self.changes_dir / (change_id + '.json')

    def _save(self, record: ChangeRecord) -> None:
        self.changes_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.changes_dir.chmod(0o700)
        def encode(states):
            return {name: {'exists': s.exists, 'bytes': base64.b64encode(s.bytes).decode(),
                           'mode': s.mode, 'fingerprint': s.fingerprint} for name, s in states.items()}
        raw = dict(record.__dict__, before=encode(record.before), after=encode(record.after))
        fd, filename = tempfile.mkstemp(prefix='.record-', dir=self.changes_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(raw, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(filename, self._file(record.id))
            directory = os.open(self.changes_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(filename).unlink(missing_ok=True)

    def get(self, change_id: str) -> ChangeRecord:
        try:
            raw = json.loads(self._file(change_id).read_text(encoding='utf-8'))
            if raw['root_identity'] != self.root_identity or raw['id'] != change_id:
                raise ChangeError('изменение принадлежит другому проекту')
            for key in ('before', 'after'):
                states = {}
                for name, state in raw[key].items():
                    relative = Path(name)
                    if not name or relative.is_absolute() or any(p in ('..', '.git') for p in relative.parts):
                        raise ChangeError('недопустимый путь снимка')
                    value = FileState(bool(state['exists']), base64.b64decode(state['bytes'], validate=True), int(state['mode']))
                    if state['fingerprint'] != value.fingerprint:
                        raise ChangeError('снимок повреждён')
                    states[name] = value
                raw[key] = states
            raw['paths'] = tuple(raw['paths'])
            if set(raw['paths']) != set(raw['before']) or set(raw['paths']) != set(raw['after']):
                raise ChangeError('пути снимков не совпадают')
            raw['created_dirs'] = tuple(raw['created_dirs'])
            for name in raw['created_dirs']:
                if not name or name == '.' or Path(name).is_absolute() or '..' in Path(name).parts or '.git' in Path(name).parts:
                    raise ChangeError('недопустимый каталог')
            raw['validation'] = tuple(raw.get('validation', ()))
            return ChangeRecord(**raw)
        except (OSError, ValueError, KeyError, TypeError, patches.PatchError) as error:
            raise ChangeError('журнал не прочитан: ' + str(error)) from error

    def list(self) -> Tuple[ChangeRecord, ...]:
        records = [self.get(p.stem) for p in self.changes_dir.glob('*.json')]
        return tuple(sorted(records, key=lambda r: (r.created_at, r.id)))

    def _result(self, record, ok, status, reason=''):
        record.status, record.reason = status, reason
        self._save(record)
        return ChangeResult(ok, record, reason)

    def apply(self, change_id: str, confirm=None) -> ChangeResult:
        record = self.get(change_id)
        if record.status != 'prepared':
            return ChangeResult(False, record, 'изменение уже обработано: ' + record.status)
        if confirm is None:
            return ChangeResult(False, record, 'применение не подтверждено')
        if not confirm(patches.parse(record.patch).summary(), record.patch):
            return ChangeResult(False, record, 'пользователь отказался применять патч')
        try:
            if self._states(record.paths) != record.before:
                return self._result(record, False, 'conflict', 'файлы изменились после подготовки')
            self._check(record.patch)
        except ChangeError as error:
            return self._result(record, False, 'conflict', str(error))
        record.status = 'applying'
        self._save(record)  # Durable intent with both postimages before the first write.
        try:
            reason = patches.apply(self.root, record.patch)
            if reason:
                self._restore_known(record, record.before)
                return self._result(record, False, 'failed', reason)
            if self._states(record.paths) != record.after:
                return self._result(record, False, 'conflict', 'результат не совпадает со снимком')
            return self._result(record, True, 'applied')
        except ChangeError as error:
            return self._result(record, False, 'conflict', str(error))
        except OSError as error:
            # Keep applying if journal I/O fails: recovery will determine actual state.
            return ChangeResult(False, record, str(error))

    def _restore_known(self, record, target):
        current = self._states(record.paths)
        if any(current[p] not in (record.before[p], record.after[p]) for p in record.paths):
            raise ChangeError('неизвестные правки: восстановление остановлено')
        for name, state in target.items():
            path = patches.safe_path(self.root, name)
            if state.exists:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.ffai-restore-', dir=path.parent)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(state.bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.chmod(temporary, state.mode)
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            else:
                path.unlink(missing_ok=True)
        if target == record.before:
            for directory in record.created_dirs:
                path = self.root / directory
                if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                    raise ChangeError('каталог восстановления изменился')
                try:
                    path.rmdir()
                except OSError:
                    pass  # Only remove empty directories created by this record.

    def recover(self) -> Tuple[ChangeRecord, ...]:
        recovered = []
        for record in self.list():
            if record.status not in ('applying', 'undoing'):
                continue
            try:
                current = self._states(record.paths)
                if record.status == 'applying' and current == record.after:
                    self._result(record, True, 'applied', 'восстановлено после сбоя')
                elif record.status == 'applying' and current == record.before:
                    self._result(record, False, 'failed', 'запись не произошла')
                elif record.status == 'undoing' and current == record.after:
                    self._result(record, True, 'applied', 'откат не произошёл')
                else:
                    self._restore_known(record, record.before)
                    self._result(record, record.status == 'undoing',
                                 'undone' if record.status == 'undoing' else 'failed',
                                 'восстановлено по журналу')
            except ChangeError as error:
                self._result(record, False, 'conflict', str(error))
            recovered.append(record)
        return tuple(recovered)
