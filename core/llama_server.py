"""Жизненный цикл локального llama-server: поднять, дождаться готовности, остановить.

Локальная модель — это режим работы без облачного ключа и без сети, поэтому сервер поднимает
само приложение, а не пользователь вручную: без этого «локальные пресеты» в списке моделей
остаются именами, по которым никто не отвечает. Решение запускать (`FFAI_LLAMA_AUTOSTART`)
принимается здесь же: точка входа только спрашивает об этом и печатает результат.

Слушателя ищем до запуска, потому что сервер мог быть поднят раньше — прежней сессией или
пользователем; тогда он переиспользуется и останавливается вместе с приложением (осознанное
решение: сервер, которым управляет приложение, приложение же и убирает). Чужой процесс на
этом порту не трогается: остановить его значило бы сломать чужую работу, поэтому занятый
порт — это ошибка с названной причиной, а не повод для сигнала.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import requests

from . import config

# Настройки окружения: имена объявлены здесь, потому что запускает сервер приложение, а не
# вызывающий код; имена попадают в `.env.example` (тест `test_config` сверяет наборы).
AUTOSTART_ENV = "FFAI_LLAMA_AUTOSTART"
START_TIMEOUT_ENV = "FFAI_LLAMA_START_TIMEOUT"

DEFAULT_START_TIMEOUT = 120.0
# Готовность опрашивается часто, а сам запрос — короткий: ждать ответа health дольше, чем
# между попытками, значит растянуть проверку на её же таймаут.
HEALTH_TIMEOUT = 1.0
HEALTH_POLL_SECONDS = 0.1
STOP_TIMEOUT = 10.0
LISTENER_TIMEOUT = 5.0
LOG_TAIL_CHARS = 1200
# Дочерний `llama-server` и все модели, которые он поднимет, живут в одной группе процессов.
SERVER_COMMAND = "llama-server"


def is_autostart_enabled() -> bool:
    """Автозапуск включён, пока переменная явно не просит обратного.

    Значение читается при обращении, а не при импорте: точка входа спрашивает об этом уже
    после разбора аргументов, и тест может переопределить переменную на лету.
    """
    value = os.getenv(AUTOSTART_ENV, "1").strip().lower()
    return value not in ("0", "false", "no", "нет")


def start_timeout() -> float:
    """Бюджет готовности в секундах: на медленном диске загрузка весов идёт минутами."""
    value = os.getenv(START_TIMEOUT_ENV, "").strip()
    if not value:
        return DEFAULT_START_TIMEOUT
    try:
        seconds = float(value)
    except ValueError as error:
        raise RuntimeError(f"{START_TIMEOUT_ENV} должно быть числом секунд: {value!r}") from error
    if seconds <= 0:
        raise RuntimeError(f"{START_TIMEOUT_ENV} должно быть положительным: {value!r}")
    return seconds


def _base_url_from_local_api(local_api_url: Optional[str] = None) -> str:
    """Адрес сервера без пути API — тот же порт, куда уходят запросы локальных пресетов.

    Порт берётся из `FFAI_LOCAL_API_URL`, а не задаётся вторым числом: сервер обязан слушать
    там, куда приложение потом обращается, иначе «локальная модель» молча уходит в пустоту.
    """
    parts = urlsplit(local_api_url or config.LOCAL_API_URL)
    if not parts.scheme or not parts.netloc:
        raise RuntimeError(
            "Не удалось определить адрес локального API: ожидался адрес вида "
            "http://127.0.0.1:9999/v1/chat/completions."
        )
    return f"{parts.scheme}://{parts.netloc}"


class LlamaServer:
    """Процесс `llama-server`: запуск по скрипту, ожидание health и остановка группы."""

    def __init__(
        self,
        script: Optional[Path] = None,
        base_url: Optional[str] = None,
        log_path: Optional[Path] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.script = Path(script) if script is not None else config.BASE_DIR / "llama_server" / "server.sh"
        # Пути и адрес берутся в момент вызова, а не в сигнатуре: значения по умолчанию из
        # конфигурации замерзают на импорте и игнорируют переменные окружения.
        self.base_url = base_url or _base_url_from_local_api()
        self.log_path = Path(log_path) if log_path is not None else config.BASE_DIR / "llama_server" / "server.log"
        self.timeout = timeout if timeout is not None else start_timeout()
        self.process: Optional[subprocess.Popen] = None
        self._existing_pid: Optional[int] = None

    # --- слушатель порта ------------------------------------------------------------------

    def _listener_pid(self) -> Optional[int]:
        """PID слушателя порта, если это llama-server; иначе None или ошибка с причиной."""
        port = urlsplit(self.base_url).port
        if port is None:
            raise RuntimeError(f"В адресе сервера нет порта: {self.base_url}")
        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=LISTENER_TIMEOUT,
            )
            pids = set(result.stdout.split())
            if not pids:
                # `lsof` возвращает 1, когда слушателя нет, — это не сбой проверки.
                if result.returncode not in (0, 1):
                    raise RuntimeError("Не удалось проверить слушателя локального порта.")
                return None
            if len(pids) != 1:
                raise RuntimeError(f"Порт {port} занят несколькими процессами, не llama-server.")
            pid = int(pids.pop())
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True, text=True, timeout=LISTENER_TIMEOUT, check=True,
            ).stdout.strip()
            if Path(command).name != SERVER_COMMAND:
                raise RuntimeError(
                    f"Порт {port} занят процессом не llama-server; он не будет остановлен."
                )
            return pid
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise RuntimeError(f"Не удалось проверить llama-server: {error}") from error

    def _ready(self) -> bool:
        """Готовность по `/health`: до неё сервер принимает порт, но не отвечает на вопросы."""
        try:
            response = requests.get(self.base_url + "/health", timeout=HEALTH_TIMEOUT)
            return response.status_code == 200 and response.json().get("status") == "ok"
        except (requests.RequestException, ValueError):
            return False

    # --- запуск и остановка ---------------------------------------------------------------

    def start(self) -> None:
        """Запускает сервер или переиспользует уже слушающий; при сбое ничего не оставляет."""
        if not is_autostart_enabled():
            return
        try:
            self._existing_pid = self._listener_pid()
            if self._existing_pid is None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("w", encoding="utf-8") as log:
                    self.process = subprocess.Popen(
                        ["bash", str(self.script)], stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True,
                    )
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    detail = self.log_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(
                        f"llama-server завершился при запуске. {detail[-LOG_TAIL_CHARS:].strip()}"
                    )
                if self._ready():
                    return
                time.sleep(HEALTH_POLL_SECONDS)
            raise RuntimeError(
                f"Не дождались готовности llama-server за {self.timeout:g} с. Лог: {self.log_path}"
            )
        except BaseException:
            # Прерывание (в том числе Ctrl+C) не должно оставлять загруженную модель в памяти.
            self.stop()
            raise

    def stop(self) -> None:
        """Останавливает поднятую группу процессов или переиспользованного слушателя."""
        if self.process is not None:
            # У router-сервера есть дочерние модели; группа сохраняется, даже если родитель
            # уже упал, поэтому завершаем всю созданную нами группу целиком.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=STOP_TIMEOUT)
            return

        if self._existing_pid is None:
            return
        pid, self._existing_pid = self._existing_pid, None
        # Проверка перед сигналом защищает от повторного использования PID: если порт сменил
        # владельца, сигнал ушёл бы постороннему процессу.
        try:
            current_pid = self._listener_pid()
        except RuntimeError:
            return
        if current_pid != pid:
            return
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + STOP_TIMEOUT
            while time.monotonic() < deadline:
                state = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "stat="],
                    capture_output=True, text=True, timeout=LISTENER_TIMEOUT,
                ).stdout.strip()
                if not state or state.startswith("Z"):
                    return
                time.sleep(HEALTH_POLL_SECONDS)
            # Сервер мог перестать слушать порт, но остаться в процессе выхода: health-проверка
            # в этот момент уже не отвечает, а память держит он.
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True, text=True, timeout=LISTENER_TIMEOUT,
            ).stdout.strip()
            if Path(command).name == SERVER_COMMAND:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
