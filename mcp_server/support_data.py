"""Bounded read-only ticket projection; unknown fields never leave the server."""
from __future__ import annotations
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from core.pr_review import redact

MAX_BYTES = 1024 * 1024
MAX_RESULT = 16000
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}')
_TOKEN_ASSIGNMENT = re.compile(r'''(?im)(\b(?:[\w.-]*token|пароль|токен)\b["']?\s*[:=]\s*)("[^"\n]*"|'[^'\n]*'|[^\s;,]+)''')


def redact_support(text: str) -> str:
    return _TOKEN_ASSIGNMENT.sub(r'\1"[REDACTED]"', redact(text))


USER_FIELDS = ('id', 'name', 'experience')
TICKET_FIELDS = ('id', 'user_id', 'status', 'subject', 'description', 'error_code', 'product_version', 'provider')

class SupportDataError(ValueError):
    """Safe error without input payloads."""

def _records(value, name):
    if not isinstance(value, list):
        raise SupportDataError(f'Неверная схема: {name} должен быть списком.')
    result = {}
    for record in value:
        if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not _ID.fullmatch(record['id']):
            raise SupportDataError(f'Неверная схема идентификатора в {name}.')
        if record['id'] in result:
            raise SupportDataError(f'Повтор идентификатора в {name}.')
        result[record['id']] = record
    return result

def _project(record, fields):
    result = {}
    for key in fields:
        if key in record:
            if not isinstance(record[key], str):
                raise SupportDataError('Неверный тип поля записи.')
            result[key] = redact_support(record[key])
    return result

def get_ticket(root: Path, data_path: str, ticket_id: str) -> dict:
    if not isinstance(ticket_id, str) or not _ID.fullmatch(ticket_id):
        raise SupportDataError('Недопустимый идентификатор тикета.')
    relative = PurePosixPath(data_path)
    if not data_path or relative.is_absolute() or any(p in ('.', '..') for p in data_path.split('/')) or '\\' in data_path:
        raise SupportDataError('Источник должен быть относительным путём внутри проекта.')
    try:
        # Pin each directory by fd: concurrent symlink replacement cannot redirect reads.
        directory = os.open(Path(root).resolve(), os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in relative.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
        finally:
            os.close(directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                raise SupportDataError('Источник не обычный файл или превышает лимит.')
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise SupportDataError('Источник превышает лимит.')
        data = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, SupportDataError):
            raise
        raise SupportDataError('Источник недоступен или содержит некорректный JSON.') from None
    if not isinstance(data, dict) or type(data.get('schema')) is not int or data['schema'] != 1:
        raise SupportDataError('Неподдерживаемая схема источника.')
    users = _records(data.get('users'), 'users')
    tickets = _records(data.get('tickets'), 'tickets')
    for ticket in tickets.values():
        if not isinstance(ticket.get('user_id'), str) or ticket['user_id'] not in users:
            raise SupportDataError('Связанный пользователь отсутствует.')
        _project(ticket, TICKET_FIELDS)
    for user in users.values():
        _project(user, USER_FIELDS)
    if ticket_id not in tickets:
        raise SupportDataError('Тикет не найден.')
    ticket = tickets[ticket_id]
    result = {'source': data_path, 'ticket': _project(ticket, TICKET_FIELDS),
              'user': _project(users[ticket['user_id']], USER_FIELDS)}
    if len(json.dumps(result, ensure_ascii=False)) > MAX_RESULT:
        raise SupportDataError('Контекст тикета превышает лимит.')
    return result
