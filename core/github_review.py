"""Publish a current review in one bot-owned GitHub PR timeline comment."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

COMMENT_MARKER = '<!-- ff.ai-pr-review -->'
_BOT = 'github-actions[bot]'
_MAX_BODY_BYTES = 60_000


class CommentError(Exception):
    """Safe failure description without response bodies or credentials."""


class GitHubComments:
    def __init__(self, token: str, api_url: str = 'https://api.github.com'):
        if not token:
            raise CommentError('Токен GitHub недоступен для публикации комментария.')
        self.token = token
        self.api_url = api_url.rstrip('/')

    def request(self, path, method='GET', payload=None):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        request = urllib.request.Request(
            self.api_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
            headers={'Authorization': 'Bearer ' + self.token,
                     'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json',
                     'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'ff.ai-pr-review'},
            method=method)
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise CommentError(f'GitHub отклонил публикацию комментария: HTTP {exc.code}.') from None
        except (urllib.error.URLError, ValueError, OSError):
            raise CommentError('GitHub недоступен или вернул непригодные данные комментария.') from None


def publish_review(client, repository: str, pr: int, review: str, report: dict,
                   run_url: str) -> dict:
    """Update our bot's marked comment; never publish failed or obsolete results."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or not isinstance(pr, int) or pr < 1:
        raise CommentError('Непригодный репозиторий или номер PR.')
    head = report.get('head', '')
    if report.get('status') != 'ok' or not re.fullmatch(r'(?:[a-f0-9]{40}|[a-f0-9]{64})', head):
        raise CommentError('Успешное ревью с head SHA недоступно.')
    if not isinstance(review, str) or not review.strip():
        raise CommentError('Текст ревью недоступен.')
    expected = 'https://github.com/' + repository + '/actions/runs/'
    if not run_url.startswith(expected) or not run_url[len(expected):].isdigit():
        raise CommentError('Непригодная ссылка на запуск Actions.')
    prefix = f'/repos/{repository}'
    pull = client.request(prefix + f'/pulls/{pr}')
    if pull['state'] != 'open' or pull['head']['sha'] != head:
        return {'status': 'stale', 'reason': 'PR закрыт или head изменился; комментарий не опубликован.'}
    intro = f'{COMMENT_MARKER}\n## ff.ai — AI-ревью\n\nHead: `{head}` · [Запуск Actions]({run_url})\n\n'
    gate = report.get('gate', {})
    if gate.get('status') == 'blocked':
        intro += '**Merge заблокирован:** устраните потенциальные баги и архитектурные проблемы ' \
                 'ниже и дождитесь нового успешного AI-ревью.\n\n'
    elif gate.get('status') == 'passed':
        intro += '**Блокирующих замечаний нет.**\n\n'
    suffix = f'\n\nПолный результат и отчёт: [артефакты Actions]({run_url}).\n'
    body = intro + review + suffix
    if len(body.encode('utf-8')) > _MAX_BODY_BYTES:
        suffix = '\n\nТекст сокращён из-за предела комментария.' + suffix
        budget = _MAX_BODY_BYTES - len((intro + suffix).encode('utf-8'))
        body = intro + review.encode('utf-8')[:budget].decode('utf-8', errors='ignore') + suffix
    existing = None
    page = 1
    while existing is None:
        comments = client.request(prefix + f'/issues/{pr}/comments?per_page=100&page={page}')
        existing = next((c for c in comments if COMMENT_MARKER in c.get('body', '')
                         and c.get('user', {}).get('login') == _BOT
                         and c.get('user', {}).get('type') == 'Bot'), None)
        if existing is not None or len(comments) < 100:
            break
        page += 1
    # Check again after pagination; do not post a result invalidated while listing comments.
    pull = client.request(prefix + f'/pulls/{pr}')
    if pull['state'] != 'open' or pull['head']['sha'] != head:
        return {'status': 'stale', 'reason': 'Head изменился перед записью; комментарий не опубликован.'}
    if existing is not None:
        comment = client.request(prefix + f'/issues/comments/{existing["id"]}', 'PATCH', {'body': body})
        status = 'updated'
    else:
        comment = client.request(prefix + f'/issues/{pr}/comments', 'POST', {'body': body})
        status = 'created'
    return {'status': status, 'url': comment['html_url']}
