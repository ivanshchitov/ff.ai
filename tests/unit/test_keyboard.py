"""Чтение клавиш из настоящего псевдотерминала.

Моки здесь бесполезны: оба известных бага (буферизация TextIOWrapper и переключение
cbreak-режима вокруг каждого чтения) проявляются только на живом tty, потому что их суть —
во взаимодействии с драйвером терминала и с ядром, а не в логике Python.
"""

import os
import pty
import sys
import termios

import pytest

from ui import keyboard

pytestmark = pytest.mark.pty


@pytest.fixture
def tty_stdin(monkeypatch):
    """Подменяет sys.stdin на slave-конец pty и отдаёт функцию записи в master-конец."""
    master, slave = pty.openpty()
    stdin = os.fdopen(slave, "rb", buffering=0)
    monkeypatch.setattr(sys, "stdin", stdin)

    def write(data: bytes) -> None:
        os.write(master, data)

    try:
        yield write
    finally:
        stdin.close()
        os.close(master)


@pytest.fixture
def cbreak_tty(tty_stdin):
    """То же, но с включённым на всё время теста cbreak-режимом — как на экране настроек."""
    with keyboard.raw_mode():
        yield tty_stdin


# --- отдельные клавиши ------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence,expected",
    [
        (b"\x1b[A", keyboard.UP),
        (b"\x1b[B", keyboard.DOWN),
        (b"\x1b[C", keyboard.RIGHT),
        (b"\x1b[D", keyboard.LEFT),
    ],
)
def test_arrow_sequences(cbreak_tty, sequence, expected):
    cbreak_tty(sequence)
    assert keyboard.read_key() == expected


def test_lone_escape_is_not_mistaken_for_an_arrow(cbreak_tty):
    """Одиночный Esc — выход с экрана настроек, его нельзя спутать с началом стрелки."""
    cbreak_tty(b"\x1b")
    assert keyboard.read_key() == keyboard.ESC


def test_unknown_escape_sequence_falls_back_to_escape(cbreak_tty):
    cbreak_tty(b"\x1b[Z")  # Shift+Tab
    assert keyboard.read_key() == keyboard.ESC


@pytest.mark.parametrize("byte", [b"\r", b"\n"])
def test_enter_variants(cbreak_tty, byte):
    cbreak_tty(byte)
    assert keyboard.read_key() == keyboard.ENTER


@pytest.mark.parametrize("byte", [b"\x7f", b"\x08"])
def test_backspace_variants(cbreak_tty, byte):
    cbreak_tty(byte)
    assert keyboard.read_key() == keyboard.BACKSPACE


@pytest.mark.parametrize("char", ["1", "9", "a", "/"])
def test_plain_characters_pass_through(cbreak_tty, char):
    cbreak_tty(char.encode())
    assert keyboard.read_key() == char


def test_digits_are_recognised_as_digits(cbreak_tty):
    """Числовые поля экрана настроек опираются на key.isdigit()."""
    cbreak_tty(b"7")
    assert keyboard.read_key().isdigit()


# --- регрессы, найденные на живом терминале ----------------------------------------------


def test_arrow_written_as_single_burst(cbreak_tty):
    """Все три байта стрелки приходят одним write — так их и присылает реальный терминал."""
    cbreak_tty(b"\x1b[A")
    assert keyboard.read_key() == keyboard.UP


def test_two_arrows_in_one_burst_are_read_separately(cbreak_tty):
    """Регресс на буферизацию: sys.stdin.read(1) утащил бы обе последовательности в свой
    внутренний буфер, и select() ниже решил бы, что байтов больше нет."""
    cbreak_tty(b"\x1b[A\x1b[B")
    assert keyboard.read_key() == keyboard.UP
    assert keyboard.read_key() == keyboard.DOWN


def test_burst_of_backspaces_is_not_eaten_by_the_terminal(cbreak_tty):
    """Регресс на переключение режима: в canonical-режиме драйвер съел бы Backspace сам,
    обработав их как штатное стирание строки, и до приложения дошла бы не вся пачка."""
    cbreak_tty(b"\x7f\x7f\x7f")
    assert [keyboard.read_key() for _ in range(3)] == [keyboard.BACKSPACE] * 3


def test_mixed_burst_keeps_order(cbreak_tty):
    cbreak_tty(b"5\x1b[B\x7f\x1b")
    assert [keyboard.read_key() for _ in range(4)] == [
        "5",
        keyboard.DOWN,
        keyboard.BACKSPACE,
        keyboard.ESC,
    ]


def test_escape_followed_later_by_a_digit(cbreak_tty):
    """Esc и следующая клавиша, разнесённые во времени, не должны склеиваться."""
    cbreak_tty(b"\x1b")
    assert keyboard.read_key() == keyboard.ESC
    cbreak_tty(b"3")
    assert keyboard.read_key() == "3"


# --- режим терминала ----------------------------------------------------------------------

# PENDIN («ввод требует перепечатки») выставляет само ядро при смене режима — это его
# внутреннее состояние, а не то, что задавало приложение. Побитовое сравнение атрибутов
# без этой маски даёт ложное срабатывание на BSD/macOS.
_VOLATILE_LFLAGS = termios.PENDIN


def _stable_attrs(fd):
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~_VOLATILE_LFLAGS
    return attrs


def _is_canonical(fd) -> bool:
    return bool(termios.tcgetattr(fd)[3] & termios.ICANON)


def test_raw_mode_switches_out_of_canonical_and_back(tty_stdin):
    """Смысл режима: внутри клавиши приходят по одной, снаружи терминал снова строчный."""
    fd = sys.stdin.fileno()
    assert _is_canonical(fd)
    with keyboard.raw_mode():
        assert not _is_canonical(fd)
    assert _is_canonical(fd)


def test_raw_mode_restores_terminal_settings(tty_stdin):
    fd = sys.stdin.fileno()
    before = _stable_attrs(fd)
    with keyboard.raw_mode():
        assert _stable_attrs(fd) != before
    assert _stable_attrs(fd) == before


def test_raw_mode_restores_settings_after_exception(tty_stdin):
    """Ctrl+C на экране настроек не должен оставить терминал в cbreak-режиме."""
    fd = sys.stdin.fileno()
    before = _stable_attrs(fd)
    with pytest.raises(KeyboardInterrupt):
        with keyboard.raw_mode():
            raise KeyboardInterrupt
    assert _stable_attrs(fd) == before
    assert _is_canonical(fd)


def test_raw_mode_is_reentrant_across_screens(tty_stdin):
    """Экран настроек можно открыть несколько раз за сессию."""
    fd = sys.stdin.fileno()
    before = _stable_attrs(fd)
    for _ in range(3):
        with keyboard.raw_mode():
            pass
    assert _stable_attrs(fd) == before


# --- неблокирующее чтение (прогон задачи опрашивает клавишу паузы) -----------------------


def test_read_key_nowait_returns_empty_without_input(cbreak_tty):
    assert keyboard.read_key_nowait() == ""


def test_read_key_nowait_returns_the_key(cbreak_tty):
    cbreak_tty(b"p")

    assert keyboard.read_key_nowait() == "p"
    assert keyboard.read_key_nowait() == ""


def test_read_key_nowait_reads_a_special_key(cbreak_tty):
    cbreak_tty(b"\x1b[A")

    assert keyboard.read_key_nowait() == keyboard.UP


def test_read_key_nowait_does_not_swallow_a_burst(cbreak_tty):
    """Пачка байтов не теряется: каждый вызов отдаёт ровно одну клавишу.

    Ctrl+C в проверку не входит: cbreak-режим сохраняет ISIG, поэтому он приходит сигналом,
    а не байтом, — пауза по Ctrl+C приходит в приложение исключением, а не клавишей.
    """
    cbreak_tty(b"pq")

    assert keyboard.read_key_nowait() == "p"
    assert keyboard.read_key_nowait() == "q"
    assert keyboard.read_key_nowait() == ""


# --- чтение символа (строка правок в прогоне задачи) -------------------------------------


def test_read_char_reads_cyrillic_letters(cbreak_tty):
    """Кириллица в UTF-8 — два байта: по байту она рассыпается, поэтому читаем символ целиком."""
    cbreak_tty("привет".encode("utf-8"))

    assert "".join(keyboard.read_char() for _ in range(6)) == "привет"


def test_read_char_reads_latin_letters(cbreak_tty):
    cbreak_tty(b"ok")

    assert keyboard.read_char() == "o"
    assert keyboard.read_char() == "k"


def test_read_char_maps_special_keys(cbreak_tty):
    cbreak_tty(b"\r\x7f\x1b[A")

    assert keyboard.read_char() == keyboard.ENTER
    assert keyboard.read_char() == keyboard.BACKSPACE
    assert keyboard.read_char() == keyboard.UP


def test_read_char_does_not_swallow_a_burst(cbreak_tty):
    cbreak_tty("да\n".encode("utf-8"))

    assert keyboard.read_char() == "д"
    assert keyboard.read_char() == "а"
    assert keyboard.read_char() == keyboard.ENTER
