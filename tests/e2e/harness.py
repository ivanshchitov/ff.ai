"""Запуск настоящего приложения в псевдотерминале с эмуляцией экрана.

Сырой поток из pty читать бесполезно: `rich.Live` перерисовывает кадр десятки раз в секунду
курсорными последовательностями, поэтому ответ модели окажется размазан по сотням фрагментов
вперемешку с управляющими кодами. Байты скармливаются в `pyte`, и тесты смотрят на
отрисованный экран — ровно то, что видит человек.

Ждать «тишины» в потоке тоже нельзя (по той же причине — кадры идут постоянно даже без
изменений), поэтому ожидание построено на опросе отрисованного экрана: `wait_for` ищет
подстроку, а `read_for` просто читает фиксированное время там, где проверяется отсутствие
чего-либо.

Приложение стартует из корня ff.ai (абсолютные пути от `Path(__file__)`), а рабочим каталогом
ему даётся временный целевой репозиторий: точка входа ищет git-корень от текущего каталога,
а `--repo` называет тот же каталог явно. Все пути состояния уводятся переменными `FFAI_*`
в `tmp_path` теста, поэтому прогон не читает и не пишет файлы пользователя.
"""

import errno
import fcntl
import os
import pty
import re
import signal
import struct
import sys
import tempfile
import termios
import time
from pathlib import Path
from typing import Callable, List, Optional

import pyte

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENTRY_POINT = REPO_ROOT / "ff-ai.py"

# Приглашение ввода: `ui/tui_app.INPUT_PROMPT`. По нему видно, что приложение готово к вводу.
PROMPT = "Вы:"
DEFAULT_TIMEOUT = 10.0

# Строка метрик запроса печатается после каждого ответа — это признак завершённого обмена.
METRICS_MARKER = "⏱"
# Значение stub-сервера: по нему обмен считается завершённым, даже если ответа в кадре нет.
STUB_USAGE_LINE = "Токены: 50+100=150"  # числа stub-сервера — для тестов, ожидающих именно их

# Клавиши как их присылает настоящий терминал.
KEY_UP = b"\x1b[A"
KEY_DOWN = b"\x1b[B"
KEY_RIGHT = b"\x1b[C"
KEY_LEFT = b"\x1b[D"
KEY_ESC = b"\x1b"
KEY_BACKSPACE = b"\x7f"
KEY_ENTER = b"\r"
KEY_TAB = b"\t"
CTRL_C = b"\x03"
CTRL_D = b"\x04"


def _collapse(text: str) -> str:
    return " ".join(text.split())


class AppSession:
    """Живой процесс приложения, подключённый к псевдотерминалу."""

    def __init__(
        self,
        history_file: Path,
        state_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        profile_file: Optional[Path] = None,
        task_file: Optional[Path] = None,
        schedule_file: Optional[Path] = None,
        tasks_dir: Optional[Path] = None,
        exports_dir: Optional[Path] = None,
        index_file: Optional[Path] = None,
        repo: Optional[Path] = None,
        domain: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = "sk-e2e-test",
        mcp_command: Optional[str] = None,
        mcp_args: Optional[str] = None,
        mcp_url: Optional[str] = None,
        mcp_name: Optional[str] = None,
        auto_tools: bool = False,
        cols: int = 100,
        rows: int = 40,
        extra_env: Optional[dict] = None,
        args: Optional[List[str]] = None,
        mirror: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        """Запускает приложение в псевдотерминале.

        `api_url=None` оставляет приложению его собственный адрес — то есть настоящий
        облачный сервис; `api_key=None` не задаёт ключ в окружении, и приложение читает его
        само из `.env`. Обе заглушки по умолчанию включены, живой режим включается явно.

        `repo` — целевой репозиторий (рабочий каталог приложения); без него создаётся
        временный: точка входа отказывается работать вне git-репозитория.

        Все пути состояния выводятся из `state_dir`/`cache_dir` и пробрасываются явно —
        так ни один прогон не касается реальных файлов пользователя. `mirror` получает
        каждый прочитанный из терминала кусок байт — через него вывод приложения
        транслируется в терминал, где запущены тесты (режим `--show-tui=live`).
        """
        self._mirror = mirror
        self.cols = cols
        self.rows = rows
        self.repo = Path(repo) if repo is not None else self.create_target_repo()
        state = Path(state_dir) if state_dir is not None else Path(history_file).parent
        cache = Path(cache_dir) if cache_dir is not None else state.parent / "cache"
        # HistoryScreen хранит уехавшие вверх строки: главный цикл приложения append-only,
        # и прошлые обмены обязаны оставаться в скроллбэке.
        self.screen = pyte.HistoryScreen(cols, rows, history=2000, ratio=0.5)
        self.stream = pyte.ByteStream(self.screen)

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "TERM": "xterm-256color",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "COLUMNS": str(cols),
            "LINES": str(rows),
            # Точка входа рядом с `.venv` перезапускает себя интерпретатором проекта: под
            # тестами уже запущен он же, лишний exec только мешал бы следить за процессом.
            "FFAI_NO_REEXEC": "1",
            # Анимация печати выключена всегда: она не влияет на итоговый текст, но растягивает
            # прогон. Таймаут сжимается только для stub-сервера — живой модели нужны десятки
            # секунд, поэтому там остаётся значение приложения по умолчанию.
            "FFAI_TYPING_DELAY": "0",
            # Автозапуск локальной модели выключен: llama-server установлен на машине, и прогон
            # с автозапуском поднимал бы настоящую модель на каждом тесте.
            "FFAI_LLAMA_AUTOSTART": "0",
            # Автовызов инструментов выключен по умолчанию: иначе каждый тест, считающий запросы
            # или читающий первый запрос, мерил бы запрос выбора вместо ответа.
            "FFAI_AUTO_TOOLS": "1" if auto_tools else "0",
            # Пути состояния: без явного переопределения прогон читал бы и писал реальные
            # файлы пользователя и видел бы записи прошлых сессий.
            "FFAI_STATE_DIR": str(state),
            "FFAI_CACHE_DIR": str(cache),
            "FFAI_HISTORY_FILE": str(history_file),
            "FFAI_MEMORY_FILE": str(memory_file if memory_file is not None else state / "memory.json"),
            "FFAI_PROFILE_FILE": str(profile_file if profile_file is not None else state / "profile.json"),
            "FFAI_TASK_FILE": str(task_file if task_file is not None else state / "task.json"),
            "FFAI_SCHEDULE_FILE": str(schedule_file if schedule_file is not None else state / "schedule.json"),
            "FFAI_TASKS_DIR": str(tasks_dir if tasks_dir is not None else state / "tasks"),
            "FFAI_EXPORTS_DIR": str(exports_dir if exports_dir is not None else state / "reports"),
            "FFAI_INDEX_FILE": str(index_file if index_file is not None else cache / "index.sqlite3"),
        }
        if domain is not None:
            # Домен берётся из пакета по имени, а не по маркерам временного репозитория.
            env["FFAI_DOMAIN"] = domain
        if api_key is not None:
            env["OPENCODE_API_KEY"] = api_key
        if api_url is not None:
            env["OPENCODE_API_URL"] = api_url
            env["FFAI_REQUEST_TIMEOUT"] = "1"
        # Реестр MCP подменяется локальной заглушкой: без этого каждый прогон — тест в том
        # числе — поднимал бы сервер документации портала и ходил в сеть.
        if mcp_name is not None:
            # Имя сервера-заглушки: домен ищет свой корпус документации по имени записи реестра.
            env["FFAI_MCP_NAME"] = mcp_name
        if mcp_command == "":
            # Пустая строка — явная просьба не подменять реестр: так запускается живая проверка
            # с настоящими серверами (документация портала и собственный сервер репозитория).
            env.pop("FFAI_MCP_COMMAND", None)
            env.pop("FFAI_MCP_ARGS", None)
        elif mcp_url is not None:
            env["FFAI_MCP_URL"] = mcp_url
            env.pop("FFAI_MCP_COMMAND", None)
        else:
            env["FFAI_MCP_COMMAND"] = mcp_command or sys.executable
            env["FFAI_MCP_ARGS"] = (
                mcp_args
                if mcp_args is not None
                else str(Path(__file__).resolve().parent.parent / "fake_mcp_server.py")
            )
        env.update(extra_env or {})

        argv = [sys.executable, str(ENTRY_POINT), "--repo", str(self.repo)]
        argv.extend(args or [])

        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # дочерний процесс
            try:
                os.chdir(self.repo)
            except OSError:  # pragma: no cover - каталог исчез между проверкой и запуском
                os._exit(1)
            os.execve(sys.executable, argv, env)
            os._exit(1)  # pragma: no cover

        self._set_window_size(cols, rows)
        os.set_blocking(self.fd, False)
        self._closed = False

    # --- целевой репозиторий ---

    @staticmethod
    def create_target_repo(base: Optional[Path] = None, name: str = "target-repo") -> Path:
        """Временный целевой репозиторий: каталог с `.git`, чтобы проверка корня прошла.

        Точке входа нужен git-родитель (иначе она отказывается работать), а содержимое
        репозитория тесту не важно — роль играет только сам факт наличия `.git`.
        """
        root = Path(base) if base is not None else Path(tempfile.mkdtemp(prefix="ff-ai-e2e-"))
        repo = root if root.name == name else root / name
        (repo / ".git").mkdir(parents=True, exist_ok=True)
        return repo

    def write_repo_file(self, name: str, text: str) -> Path:
        """Кладёт файл в целевой репозиторий — то, что ассистент увидит как рабочую копию."""
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # --- служебное ---

    def _set_window_size(self, cols: int, rows: int) -> None:
        """Фиксированный размер окна: иначе rich перенесёт строки иначе и ассерты поплывут."""
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in (errno.EIO, errno.EBADF):  # процесс закрыл терминал
                    self._closed = True
                    return
                raise
            if not data:
                self._closed = True
                return
            self.stream.feed(data)
            if self._mirror is not None:
                self._mirror(data)

    # --- чтение экрана ---

    def screen_text(self) -> str:
        self._drain()
        return "\n".join(self.screen.display)

    def scrollback(self) -> str:
        """Экран вместе с уехавшими вверх строками."""
        self._drain()
        top = [_render(line, self.cols) for line in self.screen.history.top]
        return "\n".join(top + list(self.screen.display))

    def read_for(self, seconds: float) -> str:
        """Читает фиксированное время. Для проверок «этого на экране быть не должно»."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._drain()
            time.sleep(0.02)
        return self.screen_text()

    def wait_for(self, needle: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Ждёт появления подстроки в скроллбэке, опрашивая отрисованный экран."""
        target = _collapse(needle)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.scrollback()
            if target in _collapse(text):
                return text
            time.sleep(0.02)
        raise AssertionError(
            f"Не дождались {needle!r} за {timeout} с.\n--- экран ---\n{self.scrollback()}"
        )

    def wait_for_prompt(self, timeout: float = DEFAULT_TIMEOUT) -> str:
        return self.wait_for(PROMPT, timeout=timeout)

    def wait_on_screen(self, needle: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Ждёт подстроку в текущем экране, игнорируя скроллбэк.

        Нужно там, где искомое уже встречалось выше по логу: поиск по скроллбэку в таком
        случае срабатывает мгновенно на старом вхождении и ничего не проверяет.
        """
        target = _collapse(needle)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.screen_text()
            if target in _collapse(text):
                return text
            time.sleep(0.02)
        raise AssertionError(
            f"Не дождались {needle!r} на экране за {timeout} с.\n--- экран ---\n{self.screen_text()}"
        )

    def wait_until_gone(self, needle: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Ждёт, пока подстрока исчезнет с текущего экрана (например, закроется панель)."""
        target = _collapse(needle)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.screen_text()
            if target not in _collapse(text):
                return text
            time.sleep(0.02)
        raise AssertionError(
            f"{needle!r} не исчезло с экрана за {timeout} с.\n--- экран ---\n{self.screen_text()}"
        )

    def contains(self, needle: str) -> bool:
        return _collapse(needle) in _collapse(self.scrollback())

    def metrics_count(self) -> int:
        """Сколько ответов завершилось: строк метрик запроса в скроллбэке.

        Считается общий маркер `⏱`, а не строка токенов конкретного ответа: у живого сервиса
        числа каждый раз другие, и привязка к «50+100=150» от stub-сервера означала бы ноль
        всегда, когда отвечает настоящая модель.
        """
        return _collapse(self.scrollback()).count(METRICS_MARKER)

    # --- ввод ---

    def _write_all(self, data: bytes, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Дописывает буфер целиком.

        Один os.write в pty записывает лишь столько, сколько влезло в буфер терминала
        (обычно несколько килобайт). Длинный ввод без дозаписи потерял бы хвост вместе с
        завершающим Enter, и приложение просто ждало бы конца строки.
        """
        deadline = time.monotonic() + timeout
        while data:
            try:
                written = os.write(self.fd, data)
            except BlockingIOError:
                written = 0
            data = data[written:]
            if data:
                if time.monotonic() > deadline:
                    raise AssertionError("Не удалось дописать ввод в терминал за отведённое время")
                # Дать приложению вычитать накопившееся, освободив место в буфере.
                time.sleep(0.01)
                self._drain()

    def send_line(self, text: str) -> None:
        self._write_all(text.encode("utf-8") + b"\r")

    def send_key(self, key: bytes, count: int = 1) -> None:
        self._write_all(key * count)

    def send_keys(self, *keys: bytes) -> None:
        for key in keys:
            self._write_all(key)
            # Клавиши панелей отправляются по одной с зазором: приложение читает их
            # по байту, а пачка escape-последовательностей в один write — отдельный сценарий.
            time.sleep(0.02)

    def ask(self, question: str, expect: str = METRICS_MARKER, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Задать вопрос и дождаться ответа — самый частый шаг сценария.

        По умолчанию ждём **новую** строку метрик запроса: она печатается после каждого ответа.
        Просто искать `⏱` в скроллбэке нельзя — строка прошлого ответа никуда не исчезает, и
        ожидание срабатывало бы мгновенно, не дожидаясь нынешнего запроса. Поэтому считается
        число строк метрик до отправки и ожидается его рост.
        """
        # Таймаут вопроса распространяется и на ожидание приглашения: стартовый обход реестра
        # с настоящими (пусть и локальными) серверами занимает секунды.
        self.wait_for_prompt(timeout=timeout)
        if expect is not METRICS_MARKER:
            self.send_line(question)
            return self.wait_for(expect, timeout=timeout)
        before = self.metrics_count()
        self.send_line(question)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.metrics_count() > before:
                return self.scrollback()
            # Ошибку запроса видно сразу: ждать полного таймаута на ответ, которого не будет,
            # бессмысленно — лучше показать экран с причиной.
            if "Ошибка запроса:" in self.scrollback():
                raise AssertionError(
                    f"Запрос {question!r} завершился ошибкой.\n--- экран ---\n{self.scrollback()}"
                )
            time.sleep(0.02)
        raise AssertionError(
            f"Ответ на {question!r} не появился за {timeout} с.\n--- экран ---\n{self.scrollback()}"
        )

    # --- завершение ---

    def wait_exit(self, timeout: float = DEFAULT_TIMEOUT) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._drain()
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self._drain()
                return os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
            time.sleep(0.02)
        raise AssertionError(
            f"Приложение не завершилось за {timeout} с.\n--- экран ---\n{self.scrollback()}"
        )

    def close(self) -> None:
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass

    def __enter__(self) -> "AppSession":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _render(line, cols: int) -> str:
    """Собирает строку скроллбэка pyte в текст."""
    return "".join(line[x].data for x in range(cols)).rstrip()


_MARKDOWN_CHARS = re.compile(r"[*_`#>]")


def plain_tail(markdown_text: str, words: int = 5) -> str:
    """Хвост ответа в том виде, в каком он окажется на экране.

    Искать по сырому markdown нельзя: rich отрисовывает `**жирный**` жирным начертанием,
    и звёздочек в тексте экрана уже нет.
    """
    cleaned = _MARKDOWN_CHARS.sub("", markdown_text)
    return " ".join(cleaned.split()[-words:])


def wait_for_answers(
    session: AppSession, number: int, marker: str = METRICS_MARKER,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Ждёт завершение обмена номер N: N строк метрик запроса в скроллбэке.

    Счётчика диалогов в статус-баре нет; признак завершённого обмена — строка метрик, которая
    печатается после каждого ответа. Маркер общий (`⏱`), а не строка токенов stub-сервера:
    иначе ожидание работало бы только против заглушки.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _collapse(session.scrollback()).count(marker) >= number:
            return session.scrollback()
        time.sleep(0.02)
    raise AssertionError(
        f"Не дождались {number} обменов за {timeout} с.\n--- экран ---\n{session.scrollback()}"
    )


def tui_display(request):
    """Режим показа вывода приложения (`--show-tui`) плюс сам транслятор потока.

    В режиме `live` байты из псевдотерминала пишутся прямо в stdout процесса тестов, поэтому
    экран приложения выглядит так же, как при обычном запуске — со всеми перерисовками
    `rich.Live`. Писать нужно именно в дескриптор: приложение шлёт готовые управляющие
    последовательности, а не текст.
    """
    mode = request.config.getoption("--show-tui", default="off")
    if mode != "off" and request.config.getoption("capture", default="fd") != "no":
        import warnings

        warnings.warn(
            "--show-tui без -s не покажет ничего: pytest перехватывает вывод. "
            "Запускайте как `pytest -s --show-tui=live ...`",
            stacklevel=2,
        )

    def mirror(data: bytes) -> None:
        stream = sys.__stdout__
        try:
            stream.buffer.write(data)
            stream.buffer.flush()
        except (ValueError, OSError):  # поток закрыт — показывать больше некуда
            pass

    return mode, (mirror if mode == "live" else None)


def dump_screens(mode: str, sessions, title: str) -> None:
    """Печатает итоговый экран каждой сессии — режим `--show-tui=screen`."""
    if mode != "screen":
        return
    for index, session in enumerate(sessions, start=1):
        header = f" {title} — сессия {index} " if len(sessions) > 1 else f" {title} "
        print("\n" + header.center(100, "="))
        print(session.scrollback())
        print("=" * 100)
