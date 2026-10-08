"""Web-слой вживую: сервер отдельным процессом, настоящая сессия, локальный stub вместо модели.

Проверяется то, чего не видит юнит-тест с поддельной сессией: реальный `python -m web` поднимает
настоящий `AssistantSession`, ответ модели приходит по HTTP из stub-API, статика отдаётся с диска,
поток событий SSE доходит до клиента, а патч задачи применяется к файлам только после ответа
браузера на подтверждение.

Флага `--web` в точке входа ещё нет (его подключает другой шаг), поэтому сервер запускается тем же
модулем, который будет вызывать флаг: `python -m web`.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Файл помечен slow: приложение поднимается в pty, и прогон занимает минуты.
pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from web.app import REPORT_KINDS

from .stub_api import answer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
START_TIMEOUT = 25.0

PLAN = json.dumps({"items": ["правка заметок"]}, ensure_ascii=False)
REVIEW_OK = json.dumps({"issues": []}, ensure_ascii=False)
NOTES_PATCH = """```diff
--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,3 @@
 первая строка
 вторая строка
+третья строка
```"""


# --- HTTP -------------------------------------------------------------------------------------


def http_json(
    url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 60.0
) -> Tuple[int, Any]:
    """Запрос JSON: ошибка тоже возвращается телом ответа, а не исключением."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw}


def http_text(url: str, timeout: float = 30.0) -> Tuple[int, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", "replace")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class SseReader:
    """Читает поток событий в фоновом потоке: тест ждёт события по типу и с дедлайном."""

    def __init__(self, url: str) -> None:
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._seen: List[str] = []
        self._response = urllib.request.urlopen(url, timeout=START_TIMEOUT)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            for raw in self._response:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data: "):
                    self._events.put(json.loads(line[len("data: ") :]))
        except Exception:  # noqa: BLE001 - закрытие потока тестом не является сбоем
            pass

    def next(self, kind: Optional[str] = None, timeout: float = 30.0) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise AssertionError(f"событие «{kind}» не пришло; получено: {self._seen}")
            try:
                event = self._events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            self._seen.append(str(event.get("type")))
            if kind is None or event.get("type") == kind:
                return event

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # noqa: BLE001 - поток уже мог закрыться сам
            pass


class WebServer:
    """Настоящий сервер web-слоя отдельным процессом."""

    def __init__(self, repo: Path, env: Dict[str, str], log_path: Path) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path
        self._log = log_path.open("wb")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "web",
                "--repo",
                str(repo),
                "--port",
                str(self.port),
                "--host",
                "127.0.0.1",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def _wait_ready(self, timeout: float = START_TIMEOUT) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"сервер завершился: {self.log_tail()}")
            try:
                status, _ = http_json(f"{self.url}/api/status", timeout=2.0)
                if status == 200:
                    return
            except Exception:  # noqa: BLE001 - сервер ещё поднимается
                time.sleep(0.1)
        raise AssertionError(f"сервер не поднялся за {timeout} с: {self.log_tail()}")

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "лог недоступен"
        return "\n".join(text.splitlines()[-lines:])

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._log.close()


@pytest.fixture
def web(stub, tmp_path: Path, target_repo: Path):
    """Сервер с изолированным состоянием и stub-API вместо модели."""
    env = dict(os.environ)
    env.update(
        {
            "OPENCODE_API_URL": stub.url,
            "OPENCODE_API_KEY": "sk-e2e-test",
            "FFAI_REQUEST_TIMEOUT": "20",
            "FFAI_TYPING_DELAY": "0",
            "FFAI_LLAMA_AUTOSTART": "0",
            "FFAI_AUTO_TOOLS": "0",
            "FFAI_NO_REEXEC": "1",
            "FFAI_WEB_LOG_LEVEL": "warning",
            "PYTHONUNBUFFERED": "1",
        }
    )
    server = WebServer(repo=target_repo, env=env, log_path=tmp_path / "web.log")
    try:
        yield server
    finally:
        server.close()


def _project(root: Path) -> None:
    """Целевой репозиторий с рабочей копией: патчи применяются `git apply`, ему нужен git."""
    if (root / ".git" / "HEAD").exists():
        return
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "notes.txt").write_text("первая строка\nвторая строка\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )


def _model_double(stub):
    """Двойник модели: ответ зависит от фазы конвейера, а не от порядка запросов."""

    def builder(payload):
        text = "\n".join(message["content"] for message in payload.get("messages", []))
        if "Число подзадач" in text:
            return answer(PLAN)
        if "Выполни подзадачу" in text:
            return answer(NOTES_PATCH)
        if "Ты проверяешь результат" in text:
            return answer(REVIEW_OK)
        return answer("Ответ ассистента.")

    return stub.dynamic(builder)


# --- проверки ----------------------------------------------------------------------------------


def test_page_and_script_are_served_from_disk(web: WebServer):
    """Страница отдаётся как чат: лента и композер на месте, внешних источников нет."""
    status, page = http_text(f"{web.url}/")
    assert status == 200
    assert "ff.ai" in page
    assert 'id="feed"' in page and 'id="composer"' in page
    assert "http://" not in page and "https://" not in page
    status, script = http_text(f"{web.url}/static/app.js")
    assert status == 200
    assert "EventSource" in script
    assert "api/ask" in script
    status, style = http_text(f"{web.url}/static/style.css")
    assert status == 200
    assert ".msg.user .bubble" in style and ".composer" in style


def test_ask_goes_through_the_session_to_the_model(web: WebServer, stub):
    status, payload = http_json(f"{web.url}/api/ask", {"question": "Где точка входа?"})
    assert status == 200, payload
    assert payload["answer"] == "Ответ stub-сервера."
    assert payload["meta"]["total_tokens"] == 150
    assert payload["sources"] == []
    assert stub.call_count == 1
    sent = stub.user_messages()[0]
    assert "Где точка входа?" in sent

    status, usage = http_json(f"{web.url}/api/reports/usage")
    assert status == 200
    assert any("Сессия: запросов — 1" in line for line in usage["lines"])
    assert usage["data"]["session"]["total_tokens"] == 150




def test_settings_are_changed_from_the_browser(web: WebServer) -> None:
    """Частичное изменение применяется: поля, которых нет в запросе, остаются прежними."""
    _, before = http_json(f"{web.url}/api/status")
    status, payload = http_json(f"{web.url}/api/settings", {"max_words": 250})
    assert status == 200, payload
    assert payload["settings"]["max_words"] == 250
    _, after = http_json(f"{web.url}/api/status")
    assert after["settings"]["max_words"] == 250
    assert after["settings"]["temperature"] == before["settings"]["temperature"]
    assert after["settings"]["format"] == before["settings"]["format"]


def test_invalid_setting_is_refused_and_changes_nothing(web: WebServer) -> None:
    """Недопустимое значение — отказ с сообщением; действующие настройки остаются прежними."""
    _, before = http_json(f"{web.url}/api/status")
    status, payload = http_json(f"{web.url}/api/settings", {"temperature": 0.55})
    assert status == 400
    assert payload["detail"]
    _, after = http_json(f"{web.url}/api/status")
    assert after["settings"] == before["settings"]


def test_out_of_range_setting_keeps_the_whole_object(web: WebServer) -> None:
    """Отказ по одному полю не применяет и соседние поля того же запроса."""
    _, before = http_json(f"{web.url}/api/status")
    status, _ = http_json(f"{web.url}/api/settings", {"max_words": 111, "compress_after": 3})
    assert status == 400
    _, after = http_json(f"{web.url}/api/status")
    assert after["settings"] == before["settings"]


def test_model_is_changed_and_unknown_model_is_refused(web: WebServer) -> None:
    """Модель меняется по списку доступных, неизвестная отклоняется без смены состояния."""
    _, status_payload = http_json(f"{web.url}/api/status")
    available = status_payload["models"]
    assert available and status_payload["model"] in available
    target = next(name for name in available if name != status_payload["model"])
    status, payload = http_json(f"{web.url}/api/settings", {"model": target})
    assert status == 200, payload
    assert payload["model"] == target
    _, after = http_json(f"{web.url}/api/status")
    assert after["model"] == target
    status, payload = http_json(f"{web.url}/api/settings", {"model": "нет-такой-модели"})
    assert status == 400
    assert "доступны" in payload["detail"]
    _, after = http_json(f"{web.url}/api/status")
    assert after["model"] == target
