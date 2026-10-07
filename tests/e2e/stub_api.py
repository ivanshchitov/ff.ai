"""Локальный stub облачного API для e2e-прогонов.

Сервер не только отвечает вместо настоящей модели, но и записывает все полученные запросы:
проверка того, что реально ушло в API (системное сообщение выбранного формата, лимиты в
user-промпте, отсутствие поля `stop`, число попыток при таймауте), даёт куда больше, чем
одна лишь проверка нарисованного экрана.

Модуль зависит только от стандартной библиотеки: ни ядра, ни интерфейса приложения.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class Reply:
    """Что сервер сделает с очередным запросом."""

    def __init__(
        self,
        content: Optional[str] = None,
        status: int = 200,
        delay: float = 0.0,
        raw_body: Optional[str] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        self.content = content
        self.status = status
        self.delay = delay
        self.raw_body = raw_body
        self.finish_reason = finish_reason

    def body(self) -> str:
        if self.raw_body is not None:
            return self.raw_body
        if self.status != 200:
            return json.dumps({"error": {"message": "stub error"}})
        choice = {"message": {"content": self.content or ""}}
        if self.finish_reason is not None:
            choice["finish_reason"] = self.finish_reason
        return json.dumps(
            {
                "choices": [choice],
                # Фиксированные значения, не зависящие от длины вопроса/ответа — этого достаточно,
                # чтобы e2e-тесты проверили сам факт проброса usage от API до экрана.
                "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            }
        )


def answer(content: str, finish_reason: Optional[str] = None) -> Reply:
    return Reply(content=content, finish_reason=finish_reason)


def failure(status: int) -> Reply:
    return Reply(status=status)


def hang(seconds: float = 3.0) -> Reply:
    """Ответ дольше REQUEST_TIMEOUT — клиент увидит таймаут и пойдёт на повтор."""
    return Reply(content="слишком поздно", delay=seconds)


def malformed() -> Reply:
    return Reply(raw_body="<html>502 Bad Gateway</html>")


class StubAPI:
    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self._replies: List[Reply] = []
        self._default = answer("Ответ stub-сервера.")
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # --- программирование поведения ---

    def always(self, reply: Reply) -> "StubAPI":
        self._default = reply
        return self

    def sequence(self, *replies: Reply) -> "StubAPI":
        """Ответы по порядку; когда очередь кончится, снова действует `always`."""
        with self._lock:
            self._replies = list(replies)
        return self

    # --- наблюдение ---

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def last_payload(self) -> Dict[str, Any]:
        return self.requests[-1]["payload"]

    def payload_at(self, index: int) -> Dict[str, Any]:
        return self.requests[index]["payload"]

    def system_messages(self) -> List[str]:
        return [r["payload"]["messages"][0]["content"] for r in self.requests]

    def user_messages(self) -> List[str]:
        """Первое user-сообщение каждого запроса: ищем по роли, а не по индексу.

        В запросе перед ходами диалога идут системные сообщения (контекст стратегии),
        поэтому индекс 1 не гарантированно user-ход.
        """
        return [
            next(m["content"] for m in r["payload"]["messages"] if m["role"] == "user")
            for r in self.requests
        ]

    def reset(self) -> None:
        self.requests.clear()

    # --- жизненный цикл ---

    def _next_reply(self) -> Reply:
        with self._lock:
            if self._replies:
                return self._replies.pop(0)
            return self._default

    def start(self) -> str:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - имя задано базовым классом
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                stub.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "payload": json.loads(raw.decode("utf-8")),
                    }
                )
                reply = stub._next_reply()
                if reply.delay:
                    time.sleep(reply.delay)
                body = reply.body().encode("utf-8")
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except BrokenPipeError:
                    # Клиент уже отвалился по таймауту — это штатная часть сценария с ретраями.
                    pass

            def log_message(self, *args):  # тишина в выводе тестов
                pass

        # Многопоточный: пока один запрос намеренно «висит», клиент уже уходит по таймауту
        # и шлёт повтор — однопоточный сервер придержал бы его в очереди.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1/chat/completions"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
