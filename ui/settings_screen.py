"""Состояние экрана /settings, отделённое от терминала.

Здесь живёт вся логика реакции на клавиши и применения введённых значений к
`AnswerSettings`. Отрисовка (`rich.Panel`) и чтение клавиш остаются в `ui/tui_app.py` и
`ui/keyboard.py` — так поведение экрана проверяется обычными тестами, без псевдотерминала.
"""

from dataclasses import dataclass, replace
from typing import List, Tuple
import re

from core import config

from core.answer_settings import (
    AnswerFormat,
    AnswerSettings,
    AnswerSettingsError,
    ContextStrategy,
)

from . import keyboard

ROW_FORMAT = 0
ROW_STRATEGY = 1
ROW_MAX_WORDS = 2
ROW_LIST_LIMIT = 3
ROW_TEMPERATURE = 4
ROW_COMPRESS_AFTER = 5
ROW_MAX_SESSION_TOKENS = 6
ROWS_COUNT = 7

FORMAT_VALUES: List[AnswerFormat] = list(AnswerFormat)
# Порядок переключения стратегий стрелками: тот же список, что в конфиге (и в enum).
STRATEGY_VALUES: List[ContextStrategy] = [ContextStrategy(name) for name in config.CONTEXT_STRATEGIES]

_TEMPERATURE_RE = re.compile(r"^(\d+)(?:\.(\d))?$")


@dataclass(frozen=True)
class SettingsScreenState:
    """Снимок экрана настроек: выбранная строка, позиция формата и содержимое числовых полей.

    Числовые поля хранятся строками, а не числами: пользователь может стереть поле в пустоту
    или временно набрать значение вне допустимого диапазона, и экран обязан это показывать —
    к числу и к диапазону они приводятся только на выходе, в `apply_to_settings`.
    """

    row: int = ROW_FORMAT
    format_index: int = 0
    strategy_index: int = 0
    max_words_input: str = ""
    list_limit_input: str = ""
    temperature_input: str = ""
    compress_after_input: str = ""
    max_session_tokens_input: str = ""

    @property
    def selected_format(self) -> AnswerFormat:
        return FORMAT_VALUES[self.format_index]

    @property
    def selected_strategy(self) -> ContextStrategy:
        return STRATEGY_VALUES[self.strategy_index]


def initial_state(settings: AnswerSettings) -> SettingsScreenState:
    return SettingsScreenState(
        row=ROW_FORMAT,
        format_index=FORMAT_VALUES.index(settings.format),
        strategy_index=STRATEGY_VALUES.index(settings.context_strategy),
        max_words_input=str(settings.max_words),
        list_limit_input=str(settings.list_limit),
        temperature_input=_format_temperature(settings.temperature),
        compress_after_input=str(settings.compress_after),
        max_session_tokens_input=str(settings.max_session_tokens),
    )


def _format_temperature(value: float) -> str:
    """Отображение значения температуры одной десятичной дробью: 0.7 → «0.7», 2 → «2.0»."""
    return f"{value:.1f}"


def apply_key(state: SettingsScreenState, key: str) -> SettingsScreenState:
    """Возвращает новое состояние экрана после нажатия клавиши.

    Нераспознанные клавиши (и клавиши, неприменимые к текущей строке) возвращают состояние
    без изменений — экран не должен реагировать, например, на цифры на строке формата.
    """
    if key in (keyboard.UP, keyboard.DOWN):
        step = -1 if key == keyboard.UP else 1
        return replace(state, row=(state.row + step) % ROWS_COUNT)

    if state.row == ROW_FORMAT and key in (keyboard.LEFT, keyboard.RIGHT):
        step = -1 if key == keyboard.LEFT else 1
        return replace(state, format_index=(state.format_index + step) % len(FORMAT_VALUES))

    if state.row == ROW_STRATEGY and key in (keyboard.LEFT, keyboard.RIGHT):
        step = -1 if key == keyboard.LEFT else 1
        return replace(
            state, strategy_index=(state.strategy_index + step) % len(STRATEGY_VALUES)
        )

    if state.row == ROW_MAX_WORDS:
        if key == keyboard.BACKSPACE:
            return replace(state, max_words_input=state.max_words_input[:-1])
        if len(key) == 1 and key.isdigit():
            return replace(state, max_words_input=state.max_words_input + key)

    if state.row == ROW_LIST_LIMIT:
        if key == keyboard.BACKSPACE:
            return replace(state, list_limit_input=state.list_limit_input[:-1])
        if len(key) == 1 and key.isdigit():
            return replace(state, list_limit_input=state.list_limit_input + key)

    if state.row == ROW_TEMPERATURE:
        if key == keyboard.BACKSPACE:
            return replace(state, temperature_input=state.temperature_input[:-1])
        if len(key) == 1 and _temperature_key_accepted(state.temperature_input, key):
            return replace(state, temperature_input=state.temperature_input + key)

    if state.row == ROW_COMPRESS_AFTER:
        if key == keyboard.BACKSPACE:
            return replace(state, compress_after_input=state.compress_after_input[:-1])
        if len(key) == 1 and key.isdigit():
            return replace(state, compress_after_input=state.compress_after_input + key)

    if state.row == ROW_MAX_SESSION_TOKENS:
        if key == keyboard.BACKSPACE:
            return replace(
                state, max_session_tokens_input=state.max_session_tokens_input[:-1]
            )
        if len(key) == 1 and key.isdigit():
            return replace(
                state, max_session_tokens_input=state.max_session_tokens_input + key
            )

    return state


def _temperature_key_accepted(current: str, key: str) -> bool:
    """В поле температуры нельзя ввести вторую точку или второй знак после точки.

    Ограничение на вводе, а не на валидации: значение с двумя знаками после точки через
    экран получить невозможно, ошибка формата у пользователя не возникает.
    """
    if key == ".":
        return "." not in current
    if key.isdigit():
        if "." in current:
            return current.rsplit(".", 1)[1] == ""
        return True
    return False


def apply_to_settings(
    state: SettingsScreenState, settings: AnswerSettings
) -> Tuple[AnswerSettings, List[str]]:
    """Применяет состояние экрана к настройкам.

    Возвращает новые настройки и список сообщений об ошибках. Невалидное поле (пустое или вне
    диапазона) не применяется, но и не отменяет остальные: настройка сохраняет прежнее значение,
    а вызывающий код показывает сообщение — это осознанный отказ от тихого клампинга.
    """
    errors: List[str] = []
    result = settings.with_format(state.selected_format).with_context_strategy(
        state.selected_strategy
    )

    if state.max_words_input.isdigit():
        try:
            result = result.with_max_words(int(state.max_words_input))
        except AnswerSettingsError as exc:
            errors.append(str(exc))
    else:
        errors.append("Максимальный объём ответа: введите число слов.")

    if state.list_limit_input.isdigit():
        try:
            result = result.with_list_limit(int(state.list_limit_input))
        except AnswerSettingsError as exc:
            errors.append(str(exc))
    else:
        errors.append("Лимит вариантов в списке: введите число.")

    result = _apply_temperature(state.temperature_input, result, errors)

    if state.compress_after_input.isdigit():
        try:
            result = result.with_compress_after(int(state.compress_after_input))
        except AnswerSettingsError as exc:
            errors.append(str(exc))
    else:
        errors.append("Порог сжатия: введите число сообщений.")

    if state.max_session_tokens_input.isdigit():
        try:
            result = result.with_max_session_tokens(int(state.max_session_tokens_input))
        except AnswerSettingsError as exc:
            errors.append(str(exc))
    else:
        errors.append("Потолок контекста: введите число токенов.")

    return result, errors


def _apply_temperature(
    raw_value: str, settings: AnswerSettings, errors: List[str]
) -> AnswerSettings:
    """Разбор и применение значения температуры; ошибки идут отдельным сообщением."""
    match = _TEMPERATURE_RE.match(raw_value)
    if match is None:
        errors.append("Температура: введите число с не более чем одним знаком после точки.")
        return settings
    value = float(match.group(1))
    if match.group(2) is not None:
        value += int(match.group(2)) / 10
    try:
        return settings.with_temperature(round(value, 1))
    except AnswerSettingsError as exc:
        errors.append(str(exc))
        return settings
