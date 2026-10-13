"""PR comments must be current, bounded and update only our bot's result."""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


HEAD = 'a' * 40
RUN = 'https://github.com/owner/project/actions/runs/42'
REVIEW = ('## Потенциальные баги\nОшибка проверки.\n'
          '## Архитектурные проблемы\nНе обнаружены.\n'
          '## Рекомендации\nВернуть условие.\n')


@pytest.fixture
def github():
    state = dict(head=HEAD, comments=[], calls=[], status=200)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state['calls'].append(('GET', self.path, None))
            if state['status'] != 200:
                self.reply(state['status'], {'message': 'private-error-detail'})
            elif self.path.endswith('/pulls/1'):
                self.reply(200, {'state': 'open', 'head': {'sha': state['head']}})
            elif '/issues/1/comments' in self.path:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                page = int(query.get('page', ['1'])[0])
                self.reply(200, state['comments'][(page - 1) * 100:page * 100])
            else:
                self.reply(404, {})

        def write_comment(self, method):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['calls'].append((method, self.path, body))
            self.reply(201 if method == 'POST' else 200,
                       {'html_url': 'https://github.com/owner/project/pull/1#issuecomment-7'})

        def do_POST(self):
            self.write_comment('POST')

        def do_PATCH(self):
            self.write_comment('PATCH')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state['url'] = f'http://127.0.0.1:{server.server_port}'
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def publish(github, text=REVIEW, report=None):
    from core.github_review import GitHubComments, publish_review
    client = GitHubComments('credential-fixture', api_url=github['url'])
    return publish_review(client, 'owner/project', 1, text,
                          report or {'status': 'ok', 'head': HEAD}, RUN)


def test_success_posts_full_review_head_and_run_link(github):
    result = publish(github)
    method, path, payload = github['calls'][-1]
    assert (method, path) == ('POST', '/repos/owner/project/issues/1/comments')
    assert REVIEW in payload['body'] and HEAD in payload['body'] and RUN in payload['body']
    assert 'credential-fixture' not in payload['body']
    assert result['status'] == 'created'


def test_existing_bot_comment_is_updated_and_user_comment_preserved(github):
    from core.github_review import COMMENT_MARKER
    github['comments'] = [
        {'id': 1, 'body': COMMENT_MARKER, 'user': {'login': 'owner', 'type': 'User'}},
        {'id': 2, 'body': COMMENT_MARKER, 'user': {'login': 'another[bot]', 'type': 'Bot'}},
        {'id': 7, 'body': COMMENT_MARKER, 'user': {'login': 'github-actions[bot]', 'type': 'Bot'}},
    ]
    result = publish(github)
    assert result['status'] == 'updated'
    assert github['calls'][-1][:2] == ('PATCH', '/repos/owner/project/issues/comments/7')


def test_stale_head_does_not_publish(github):
    github['head'] = 'b' * 40
    assert publish(github)['status'] == 'stale'
    assert all(method == 'GET' for method, _, _ in github['calls'])


def test_failed_review_does_not_publish(github):
    from core.github_review import CommentError
    with pytest.raises(CommentError):
        publish(github, report={'status': 'failed', 'head': HEAD})
    assert not github['calls']


def test_long_review_is_explicitly_shortened_with_full_run_link(github):
    publish(github, 'A' * 80_000)
    body = github['calls'][-1][2]['body']
    assert len(body.encode('utf-8')) < 65_536
    assert 'сокращён' in body and RUN in body


def test_github_error_does_not_expose_api_details(github):
    from core.github_review import CommentError
    github['status'] = 403
    with pytest.raises(CommentError, match='403') as caught:
        publish(github)
    assert 'private-error-detail' not in str(caught.value)


def test_bot_comment_on_later_page_is_updated(github):
    from core.github_review import COMMENT_MARKER
    # Pages are exercised through a narrow HTTP fixture in this test.
    github['comments'] = [{'id': n, 'body': 'user', 'user': {'login': 'owner', 'type': 'User'}}
                          for n in range(100)] + [
        {'id': 7, 'body': COMMENT_MARKER, 'user': {'login': 'github-actions[bot]', 'type': 'Bot'}}]
    assert publish(github)['status'] == 'updated'
    assert any('page=2' in path for _, path, _ in github['calls'])


def test_comment_explains_merge_block_and_resolution(github):
    publish(github, report={'status': 'ok', 'head': HEAD, 'gate': {'status': 'blocked'}})
    body = github['calls'][-1][2]['body']
    assert 'Merge заблокирован' in body and 'нового успешного AI-ревью' in body
