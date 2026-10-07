"""Чтение одиночных нажатий клавиш из терминала (для экрана /settings)."""

import contextlib
import os
import sys

try:
    import select
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    select = None
    termios = None
    tty = None

ESC = "ESC"
UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
ENTER = "ENTER"
BACKSPACE = "BACKSPACE"

_ARROW_CODES = {"A": UP, "B": DOWN, "C": RIGHT, "D": LEFT}


@contextlib.contextmanager
def raw_mode():
    """Переключает stdin в cbreak-режим на время экрана настроек (Unix; на Windows — no-op).

    Режим должен включаться один раз на весь экран, а не на каждый read_key(): если
    переключать его туда-обратно между отдельными нажатиями, байты, пришедшие пачкой
    (несколько Backspace подряд, вставка стрелки), могут застать stdin в canonical-режиме
    между вызовами — тогда драйвер терминала обработает их сам как штатное стирание строки,
    и часть нажатий до приложения не дойдёт.
    """
    if termios is None:
        yield
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key() -> str:
    """Блокирующее чтение одной клавиши внутри `with raw_mode():`. Возвращает спецкод
    (ESC/UP/.../ENTER/BACKSPACE) либо сам символ."""
    if termios is None:
        return _read_key_windows()
    return _read_key_unix()


def read_key_nowait() -> str:
    """Неблокирующее чтение клавиши внутри `with raw_mode():`; пустая строка — ввода нет.

    Прогон задачи перерисовывает панель и опрашивает клавишу паузы между операциями: ждать
    нажатия здесь нельзя, иначе пауза срабатывала бы только после следующего нажатия вообще,
    то есть через одну операцию, а не сразу после текущей.
    """
    if termios is None:
        return _read_key_windows_nowait()
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], 0)[0]:
        return ""
    return _read_key_unix()


def read_char() -> str:
    """Читает один символ (в том числе многобайтный) внутри `with raw_mode():`.

    `read_key()` отдаёт один *байт*: для стрелок, Enter и одиночных латинских клавиш этого
    хватает, а для текста — нет. Кириллица в UTF-8 занимает два байта, и по байту она
    рассыпается на «заменяющие» символы, поэтому набранный ответ пользователя доходил бы до
    модели мусором. Здесь читаем ведущий байт, добираем продолжение по его длине и декодируем
    целый символ; спецклавиши возвращаются теми же кодами, что у `read_key()`.
    """
    if termios is None:
        return _read_key_windows()
    fd = sys.stdin.fileno()
    first = os.read(fd, 1)
    if first == b"\x1b":
        if select.select([fd], [], [], 0.05)[0]:
            second = os.read(fd, 1)
            if second == b"[" and select.select([fd], [], [], 0.05)[0]:
                return _ARROW_CODES.get(os.read(fd, 1).decode(errors="replace"), ESC)
        return ESC
    if first in (b"\r", b"\n"):
        return ENTER
    if first in (b"\x7f", b"\x08"):
        return BACKSPACE
    lead = first[0]
    if lead & 0xE0 == 0xC0:
        length = 2
    elif lead & 0xF0 == 0xE0:
        length = 3
    elif lead & 0xF8 == 0xF0:
        length = 4
    else:
        length = 1
    data = first + b"".join(os.read(fd, 1) for _ in range(length - 1))
    return data.decode(errors="replace")


def _read_key_unix() -> str:
    fd = sys.stdin.fileno()
    # os.read() на самом fd, а не sys.stdin.read(): буферизованный TextIOWrapper может
    # одним system-вызовом забрать из pty сразу все байты escape-последовательности и
    # придержать их в своём внутреннем буфере — тогда select() ниже их не увидит и решит,
    # что байт был всего один (то есть одиночный Esc), а не начало ESC [ A/B/C/D.
    ch = os.read(fd, 1).decode(errors="replace")
    if ch == "\x1b":
        # Стрелки приходят как ESC [ A/B/C/D тремя байтами подряд; одиночный Esc —
        # только один байт, поэтому ждём остаток последовательности с таймаутом.
        if select.select([fd], [], [], 0.05)[0]:
            ch2 = os.read(fd, 1).decode(errors="replace")
            if ch2 == "[" and select.select([fd], [], [], 0.05)[0]:
                ch3 = os.read(fd, 1).decode(errors="replace")
                return _ARROW_CODES.get(ch3, ESC)
        return ESC
    if ch in ("\r", "\n"):
        return ENTER
    if ch in ("\x7f", "\x08"):
        return BACKSPACE
    return ch


def _read_key_windows() -> str:  # pragma: no cover - Windows
    import msvcrt

    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        ch2 = msvcrt.getwch()
        return {"H": UP, "P": DOWN, "K": LEFT, "M": RIGHT}.get(ch2, "")
    if ch == "\x1b":
        return ESC
    if ch == "\r":
        return ENTER
    if ch == "\x08":
        return BACKSPACE
    return ch


def _read_key_windows_nowait() -> str:  # pragma: no cover - Windows
    import msvcrt

    if not msvcrt.kbhit():
        return ""
    return _read_key_windows()
