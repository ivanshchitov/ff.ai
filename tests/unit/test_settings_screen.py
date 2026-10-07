"""Логика экрана /settings без терминала."""

import pytest

from core import config
from core.answer_settings import AnswerFormat, AnswerSettings, ContextStrategy
from ui import keyboard, settings_screen
from ui.settings_screen import STRATEGY_VALUES, SettingsScreenState


def press(state: SettingsScreenState, *keys: str) -> SettingsScreenState:
    for key in keys:
        state = settings_screen.apply_key(state, key)
    return state


def at_row(state: SettingsScreenState, row: int, *keys: str) -> SettingsScreenState:
    """Наводит курсор на строку экрана и нажимает клавиши.

    Число строк экрана — деталь раскладки, а не суть проверки: тесты ходят по строкам
    по их именам, поэтому добавление новой строки не переписывает весь файл.
    """
    while state.row != row:
        state = settings_screen.apply_key(state, keyboard.DOWN)
    return press(state, *keys)


@pytest.fixture
def state() -> SettingsScreenState:
    return settings_screen.initial_state(AnswerSettings())


# --- начальное состояние -----------------------------------------------------------------


def test_initial_state_mirrors_current_settings():
    settings = AnswerSettings(max_words=123, format=AnswerFormat.JSON, list_limit=4)
    state = settings_screen.initial_state(settings)
    assert state.row == settings_screen.ROW_FORMAT
    assert state.selected_format == AnswerFormat.JSON
    assert state.max_words_input == "123"
    assert state.list_limit_input == "4"
    assert state.temperature_input == str(config.TEMPERATURE)


def test_initial_state_shows_temperature_with_one_decimal():
    settings = AnswerSettings().with_temperature(1.2)
    state = settings_screen.initial_state(settings)
    assert state.temperature_input == "1.2"


# --- навигация ----------------------------------------------------------------------------


def test_down_moves_through_rows_and_wraps(state):
    """Стрелка вниз проходит строки в том порядке, в котором они нарисованы на экране."""
    order = [
        settings_screen.ROW_STRATEGY,
        settings_screen.ROW_MAX_WORDS,
        settings_screen.ROW_LIST_LIMIT,
        settings_screen.ROW_TEMPERATURE,
        settings_screen.ROW_COMPRESS_AFTER,
        settings_screen.ROW_MAX_SESSION_TOKENS,
    ]
    for steps, expected in enumerate(order, start=1):
        assert press(state, *[keyboard.DOWN] * steps).row == expected
    assert press(state, *[keyboard.DOWN] * settings_screen.ROWS_COUNT).row == (
        settings_screen.ROW_FORMAT
    )


def test_up_moves_backwards(state):
    """Стрелка вверх должна идти вверх, а не повторять поведение стрелки вниз."""
    assert press(state, keyboard.UP).row == settings_screen.ROW_MAX_SESSION_TOKENS  # последняя строка
    assert press(state, keyboard.DOWN, keyboard.UP).row == settings_screen.ROW_FORMAT


def test_up_and_down_cancel_each_other(state):
    assert press(state, keyboard.DOWN, keyboard.DOWN, keyboard.UP, keyboard.UP) == state


# --- переключение формата -----------------------------------------------------------------


def test_right_cycles_formats_forward(state):
    values = settings_screen.FORMAT_VALUES
    start = state.format_index
    assert press(state, keyboard.RIGHT).selected_format == values[(start + 1) % len(values)]
    assert press(state, keyboard.RIGHT, keyboard.RIGHT).selected_format == values[
        (start + 2) % len(values)
    ]


def test_format_wraps_around_in_both_directions(state):
    values = settings_screen.FORMAT_VALUES
    forward = press(state, *[keyboard.RIGHT] * len(values))
    assert forward.selected_format == state.selected_format
    backward = press(state, keyboard.LEFT)
    assert backward.selected_format == values[(state.format_index - 1) % len(values)]


def test_format_arrows_ignored_on_numeric_rows(state):
    on_words = at_row(state, settings_screen.ROW_MAX_WORDS)
    assert press(on_words, keyboard.RIGHT, keyboard.LEFT) == on_words


# --- числовые поля --------------------------------------------------------------------------


def test_digits_append_to_the_selected_field(state):
    result = at_row(
        state, settings_screen.ROW_MAX_WORDS, keyboard.BACKSPACE, keyboard.BACKSPACE, keyboard.BACKSPACE, "5", "0"
    )
    assert result.max_words_input == "50"
    assert result.list_limit_input == str(config.DEFAULT_LIST_LIMIT)


def test_digits_go_only_to_the_current_row(state):
    """Цифра, набранная на строке формата, не должна попасть в числовое поле."""
    assert press(state, "7") == state


def test_backspace_erases_one_character(state):
    result = at_row(state, settings_screen.ROW_MAX_WORDS, keyboard.BACKSPACE)
    assert result.max_words_input == str(config.DEFAULT_MAX_WORDS)[:-1]


def test_field_can_be_emptied_completely(state):
    result = at_row(state, settings_screen.ROW_MAX_WORDS, *[keyboard.BACKSPACE] * 10)
    assert result.max_words_input == ""


def test_backspace_on_empty_field_is_harmless(state):
    emptied = at_row(state, settings_screen.ROW_MAX_WORDS, *[keyboard.BACKSPACE] * 10)
    assert press(emptied, keyboard.BACKSPACE) == emptied


def test_list_limit_field_is_edited_independently(state):
    result = at_row(state, settings_screen.ROW_LIST_LIMIT, keyboard.BACKSPACE, "8")
    assert result.list_limit_input == "8"
    assert result.max_words_input == str(config.DEFAULT_MAX_WORDS)


@pytest.mark.parametrize("key", [keyboard.ENTER, "a", "/", "Ж", "-", "."])
def test_unrelated_keys_do_nothing(state, key):
    on_words = at_row(state, settings_screen.ROW_MAX_WORDS)
    assert press(on_words, key) == on_words


def test_state_is_immutable(state):
    at_row(state, settings_screen.ROW_MAX_WORDS, "5")
    assert state.row == settings_screen.ROW_FORMAT
    assert state.max_words_input == str(config.DEFAULT_MAX_WORDS)


def test_temperature_row_edits_independently(state):
    """Строка температуры редактируется сама, не задевая объём и лимит."""
    result = at_row(
        state,
        settings_screen.ROW_TEMPERATURE,
        keyboard.BACKSPACE,
        keyboard.BACKSPACE,
        keyboard.BACKSPACE,
        "1",
        ".",
        "2",
    )
    assert result.temperature_input == "1.2"
    assert result.max_words_input == str(config.DEFAULT_MAX_WORDS)
    assert result.list_limit_input == str(config.DEFAULT_LIST_LIMIT)


def test_temperature_second_dot_is_ignored(state):
    on_temp = at_row(
        state, settings_screen.ROW_TEMPERATURE, *[keyboard.BACKSPACE] * 3, "1", ".",
    )
    assert press(on_temp, ".").temperature_input == "1."


def test_temperature_accepts_only_one_digit_after_dot(state):
    """Второй знак после точки ввести невозможно — лишняя цифра игнорируется."""
    on_temp = at_row(
        state, settings_screen.ROW_TEMPERATURE, *[keyboard.BACKSPACE] * 3, "0", ".", "5",
    )
    assert press(on_temp, "5").temperature_input == "0.5"


def test_temperature_more_digits_before_dot_are_fine(state):
    on_temp = at_row(
        state, settings_screen.ROW_TEMPERATURE, *[keyboard.BACKSPACE] * 3,
    )
    result = press(on_temp, "1", "2", ".", "5")
    assert result.temperature_input == "12.5"


def test_temperature_backspace_erases_dot_too(state):
    on_temp = at_row(state, settings_screen.ROW_TEMPERATURE, "0", ".", "5")
    result = press(on_temp, keyboard.BACKSPACE, keyboard.BACKSPACE)
    assert result.temperature_input == "0"
    assert press(result, ".", "7").temperature_input == "0.7"


@pytest.mark.parametrize("key", [keyboard.ENTER, "a", "/", "Ж", "-", ","])
def test_temperature_rejects_non_numeric_keys(state, key):
    on_temp = at_row(state, settings_screen.ROW_TEMPERATURE)
    assert press(on_temp, key) == on_temp


# --- применение к настройкам --------------------------------------------------------------


def test_valid_input_is_applied():
    state = at_row(
        press(settings_screen.initial_state(AnswerSettings()), keyboard.RIGHT),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 3,
        "5",
        "0",
    )
    state = at_row(state, settings_screen.ROW_LIST_LIMIT, keyboard.BACKSPACE, "7")
    settings, errors = settings_screen.apply_to_settings(state, AnswerSettings())
    values = settings_screen.FORMAT_VALUES
    next_format = values[(values.index(AnswerSettings().format) + 1) % len(values)]
    assert errors == []
    assert settings.max_words == 50
    assert settings.list_limit == 7
    assert settings.format == next_format


def test_out_of_range_value_keeps_previous_and_reports():
    """Отказ вместо клампинга: настройка сохраняет старое значение, пользователь видит ошибку."""
    original = AnswerSettings().with_max_words(120)
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 4,
        "9",
        "9",
        "9",
        "9",
    )
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert settings.max_words == 120
    assert len(errors) == 1
    assert str(config.MAX_MAX_WORDS) in errors[0]


def test_empty_field_reports_its_own_message():
    original = AnswerSettings()
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 5,
    )
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert settings.max_words == original.max_words
    assert errors == ["Максимальный объём ответа: введите число слов."]


def test_one_bad_field_does_not_block_the_other():
    original = AnswerSettings(max_words=200, format=AnswerFormat.FREE, list_limit=3)
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 4,
    )
    state = at_row(state, settings_screen.ROW_LIST_LIMIT, keyboard.BACKSPACE, "9")
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert settings.list_limit == 9  # валидное поле применилось
    assert settings.max_words == 200  # невалидное осталось прежним
    assert len(errors) == 1


def test_both_fields_can_fail_at_once():
    original = AnswerSettings()
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 5,
    )
    state = at_row(state, settings_screen.ROW_LIST_LIMIT, *[keyboard.BACKSPACE] * 5)
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert len(errors) == 2
    assert settings == original


def test_format_is_applied_even_when_numbers_are_invalid():
    """Формат меняется стрелками и не зависит от валидности числовых полей."""
    original = AnswerSettings()
    state = at_row(
        press(settings_screen.initial_state(original), keyboard.RIGHT),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 5,
    )
    settings, errors = settings_screen.apply_to_settings(state, original)
    values = settings_screen.FORMAT_VALUES
    assert settings.format == values[(values.index(original.format) + 1) % len(values)]
    assert errors


def test_untouched_screen_leaves_settings_unchanged():
    """Открыть /settings и сразу нажать Esc — ничего не должно измениться."""
    original = AnswerSettings(max_words=77, format=AnswerFormat.COMPACT, list_limit=6)
    settings, errors = settings_screen.apply_to_settings(
        settings_screen.initial_state(original), original
    )
    assert settings == original
    assert errors == []


def test_leading_zeros_are_parsed_as_numbers():
    original = AnswerSettings()
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 5,
        "0",
        "5",
        "0",
    )
    settings, _ = settings_screen.apply_to_settings(state, original)
    assert settings.max_words == 50


# --- температура: применение ---------------------------------------------------------------


def _state_on_temperature_row(*keys: str) -> SettingsScreenState:
    """Строка температуры с заранее очищенным полем (дефолт «0.7» стирается)."""
    return at_row(
        settings_screen.initial_state(AnswerSettings()),
        settings_screen.ROW_TEMPERATURE,
        *[keyboard.BACKSPACE] * 3,
        *keys,
    )


def test_temperature_value_is_applied():
    state = _state_on_temperature_row("1", ".", "2")
    settings, errors = settings_screen.apply_to_settings(state, AnswerSettings())
    assert errors == []
    assert settings.temperature == 1.2


def test_temperature_zero_is_applied():
    """Граница 0.0 из задания дня: набирается как «0.0», а не пустотой."""
    state = _state_on_temperature_row("0", ".", "0")
    settings, errors = settings_screen.apply_to_settings(state, AnswerSettings())
    assert errors == []
    assert settings.temperature == 0.0


def test_temperature_out_of_range_reports_and_keeps_previous():
    original = AnswerSettings().with_temperature(0.7)
    state = _state_on_temperature_row("2", ".", "5")
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert settings.temperature == 0.7
    assert len(errors) == 1
    assert f"{config.MIN_TEMPERATURE}..{config.MAX_TEMPERATURE}" in errors[0]


def test_temperature_trailing_dot_reports_and_keeps_previous():
    original = AnswerSettings().with_temperature(0.3)
    state = _state_on_temperature_row("0", ".")
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert settings.temperature == 0.3
    assert len(errors) == 1
    assert "Температура" in errors[0]


def test_bad_temperature_does_not_block_other_fields():
    original = AnswerSettings()
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 3,
        "5",
        "0",
    )
    state = at_row(
        state, settings_screen.ROW_TEMPERATURE, *[keyboard.BACKSPACE] * 3, "9", ".", "9"
    )
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert settings.max_words == 50  # валидное поле применилось
    assert settings.temperature == original.temperature  # невалидное осталось прежним
    assert len(errors) == 1


def test_untouched_temperature_row_keeps_value():
    original = AnswerSettings().with_temperature(1.1)
    settings, errors = settings_screen.apply_to_settings(
        settings_screen.initial_state(original), original
    )
    assert settings == original
    assert errors == []


# --- день 10: строка стратегии управления контекстом -------------------------------------


def test_initial_state_shows_the_active_strategy():
    settings = AnswerSettings().with_context_strategy(ContextStrategy.STICKY_FACTS)
    state = settings_screen.initial_state(settings)
    assert state.strategy_index == STRATEGY_VALUES.index(ContextStrategy.STICKY_FACTS)
    assert state.selected_strategy is ContextStrategy.STICKY_FACTS


def test_strategy_row_sits_right_after_the_format_row():
    """Стратегия — такая же переключаемая стрелками настройка, как формат, и стоит рядом."""
    assert settings_screen.ROW_STRATEGY == settings_screen.ROW_FORMAT + 1
    assert settings_screen.ROWS_COUNT == 7


def test_right_cycles_strategies_forward(state):
    on_strategy = press(state, keyboard.DOWN)
    assert press(on_strategy, keyboard.RIGHT).selected_strategy is STRATEGY_VALUES[1]
    assert press(on_strategy, keyboard.RIGHT, keyboard.RIGHT).selected_strategy is (
        STRATEGY_VALUES[2]
    )


def test_strategy_wraps_around_in_both_directions(state):
    on_strategy = press(state, keyboard.DOWN)
    assert press(on_strategy, *[keyboard.RIGHT] * len(STRATEGY_VALUES)).selected_strategy is (
        on_strategy.selected_strategy
    )
    assert press(on_strategy, keyboard.LEFT).selected_strategy is STRATEGY_VALUES[-1]


def test_digits_on_the_strategy_row_are_ignored(state):
    on_strategy = press(state, keyboard.DOWN)
    assert press(on_strategy, "2", keyboard.BACKSPACE) == on_strategy


def test_navigation_walks_all_seven_rows(state):
    rows = {press(state, *[keyboard.DOWN] * step).row for step in range(settings_screen.ROWS_COUNT)}
    assert rows == set(range(settings_screen.ROWS_COUNT))


def test_selected_strategy_is_applied_on_exit():
    state = press(
        settings_screen.initial_state(AnswerSettings()), keyboard.DOWN, keyboard.RIGHT
    )
    settings, errors = settings_screen.apply_to_settings(state, AnswerSettings())
    assert errors == []
    assert settings.context_strategy is STRATEGY_VALUES[1]


def test_strategy_is_applied_alongside_valid_numeric_fields():
    original = AnswerSettings()
    state = at_row(
        settings_screen.initial_state(original),
        settings_screen.ROW_MAX_WORDS,
        *[keyboard.BACKSPACE] * 3,
        "5",
        "0",
    )
    state = press(state, keyboard.UP, keyboard.LEFT)  # строкой выше — стратегия
    settings, errors = settings_screen.apply_to_settings(state, original)
    assert errors == []
    assert settings.max_words == 50
    assert settings.context_strategy is STRATEGY_VALUES[-1]
