"""Валидация и неизменяемость AnswerSettings."""

import pytest

from core import config
from core.answer_settings import (
    AnswerFormat,
    AnswerSettings,
    AnswerSettingsError,
    ContextStrategy,
)


def test_defaults_match_config():
    settings = AnswerSettings()
    assert settings.max_words == config.DEFAULT_MAX_WORDS
    assert settings.format == AnswerFormat.FREE
    assert settings.list_limit == config.DEFAULT_LIST_LIMIT
    assert settings.temperature == config.TEMPERATURE
    assert settings.compress_after == config.DEFAULT_COMPRESS_AFTER
    assert settings.max_session_tokens == config.DEFAULT_MAX_SESSION_TOKENS
    assert settings.context_strategy == ContextStrategy.SUMMARY


@pytest.mark.parametrize("value", list(ContextStrategy))
def test_with_context_strategy_accepts_every_member(value):
    assert AnswerSettings().with_context_strategy(value).context_strategy == value


def test_context_strategy_values_match_the_config_list():
    """Список стратегий в конфиге и enum настроек обязаны совпадать: конфиг задаёт порядок
    переключения стрелками и значение по умолчанию, enum — типы."""
    assert [strategy.value for strategy in ContextStrategy] == config.CONTEXT_STRATEGIES


def test_context_strategy_default_is_the_config_default():
    assert ContextStrategy(config.DEFAULT_CONTEXT_STRATEGY) == AnswerSettings().context_strategy


def test_with_context_strategy_rejects_unknown_value():
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_context_strategy("нет такой стратегии")


def test_with_context_strategy_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_context_strategy(ContextStrategy.SLIDING_WINDOW)
    assert (
        updated.context_strategy,
        updated.max_words,
        updated.format,
        updated.list_limit,
        updated.temperature,
        updated.compress_after,
        updated.max_session_tokens,
    ) == (
        ContextStrategy.SLIDING_WINDOW,
        100,
        AnswerFormat.JSON,
        7,
        config.TEMPERATURE,
        config.DEFAULT_COMPRESS_AFTER,
        config.DEFAULT_MAX_SESSION_TOKENS,
    )


def test_rejected_context_strategy_leaves_original_untouched():
    settings = AnswerSettings().with_context_strategy(ContextStrategy.BRANCHING)
    with pytest.raises(AnswerSettingsError):
        settings.with_context_strategy("окно")
    assert settings.context_strategy == ContextStrategy.BRANCHING


@pytest.mark.parametrize(
    "value", [config.MIN_COMPRESS_AFTER, 10, 30, config.MAX_COMPRESS_AFTER]
)
def test_with_compress_after_accepts_range_including_edges(value):
    assert AnswerSettings().with_compress_after(value).compress_after == value


@pytest.mark.parametrize(
    "value", [config.MIN_COMPRESS_AFTER - 1, config.MAX_COMPRESS_AFTER + 1, 0, 999]
)
def test_with_compress_after_rejects_out_of_range(value):
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_compress_after(value)


@pytest.mark.parametrize("value", [10.5, 20.0, 10.0001])
def test_with_compress_after_rejects_non_integer(value):
    """Порог сжатия задаётся только целым: без округления и усечения."""
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_compress_after(value)


@pytest.mark.parametrize(
    "value", [config.MIN_MAX_SESSION_TOKENS, 20000, config.MAX_MAX_SESSION_TOKENS]
)
def test_with_max_session_tokens_accepts_range_including_edges(value):
    assert AnswerSettings().with_max_session_tokens(value).max_session_tokens == value

@pytest.mark.parametrize("value", [10.5, 10.0001])
def test_with_compress_after_rejects_non_integer(value):
    """Порог сжатия задаётся только целым: без округления и усечения (20.0 допустим)."""
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_compress_after(value)


@pytest.mark.parametrize(
    "value", [config.MIN_MAX_SESSION_TOKENS - 1, config.MAX_MAX_SESSION_TOKENS + 1, 0, -5, 100_000]
)
def test_with_max_session_tokens_rejects_out_of_range(value):
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_max_session_tokens(value)


@pytest.mark.parametrize("value", [20000.5, 6000.25])
def test_with_max_session_tokens_rejects_non_integer(value):
    """Потолок токенов задаётся только целым: ошибка, не округление."""
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_max_session_tokens(value)


def test_with_compress_after_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_compress_after(20)
    assert (
        updated.compress_after,
        updated.max_words,
        updated.format,
        updated.list_limit,
        updated.max_session_tokens,
    ) == (20, 100, AnswerFormat.JSON, 7, config.DEFAULT_MAX_SESSION_TOKENS)


def test_with_max_session_tokens_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_max_session_tokens(6000)
    assert (
        updated.max_session_tokens,
        updated.max_words,
        updated.format,
        updated.list_limit,
        updated.compress_after,
    ) == (6000, 100, AnswerFormat.JSON, 7, config.DEFAULT_COMPRESS_AFTER)


def test_compress_after_rejected_value_leaves_original_untouched():
    settings = AnswerSettings().with_compress_after(15)
    with pytest.raises(AnswerSettingsError):
        settings.with_compress_after(999)
    assert settings.compress_after == 15


@pytest.mark.parametrize("value", [config.MIN_TEMPERATURE, 0.7, 1.2, config.MAX_TEMPERATURE])
def test_with_temperature_accepts_range_with_one_decimal(value):
    assert AnswerSettings().with_temperature(value).temperature == value


@pytest.mark.parametrize(
    "value", [config.MAX_TEMPERATURE + 0.5, config.MIN_TEMPERATURE - 0.1, 2.5, -0.1]
)
def test_with_temperature_rejects_out_of_range(value):
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_temperature(value)


@pytest.mark.parametrize("value", [0.55, 1.15, 0.01, 1.999])
def test_with_temperature_rejects_more_than_one_decimal_digit(value):
    """Инвариант ядра: программный вызов с двумя знаками — ошибка, не округление."""
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_temperature(value)


def test_with_temperature_rejected_value_leaves_original_untouched():
    settings = AnswerSettings().with_temperature(1.2)
    with pytest.raises(AnswerSettingsError):
        settings.with_temperature(3.0)
    assert settings.temperature == 1.2


def test_with_temperature_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_temperature(0.0)
    assert (updated.temperature, updated.max_words, updated.format, updated.list_limit) == (
        0.0,
        100,
        AnswerFormat.JSON,
        7,
    )


@pytest.mark.parametrize("value", [config.MIN_MAX_WORDS, 200, config.MAX_MAX_WORDS])
def test_with_max_words_accepts_range_including_edges(value):
    assert AnswerSettings().with_max_words(value).max_words == value


@pytest.mark.parametrize(
    "value", [config.MIN_MAX_WORDS - 1, config.MAX_MAX_WORDS + 1, 0, -5, 10_000]
)
def test_with_max_words_rejects_out_of_range(value):
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_max_words(value)


@pytest.mark.parametrize("value", [config.MIN_LIST_LIMIT, 3, config.MAX_LIST_LIMIT])
def test_with_list_limit_accepts_range_including_edges(value):
    assert AnswerSettings().with_list_limit(value).list_limit == value


@pytest.mark.parametrize(
    "value", [config.MIN_LIST_LIMIT - 1, config.MAX_LIST_LIMIT + 1, -1, 999]
)
def test_with_list_limit_rejects_out_of_range(value):
    with pytest.raises(AnswerSettingsError):
        AnswerSettings().with_list_limit(value)


def test_rejected_value_leaves_original_untouched():
    """Отказ вместо тихого клампинга: прежнее значение должно уцелеть."""
    settings = AnswerSettings().with_max_words(120)
    with pytest.raises(AnswerSettingsError):
        settings.with_max_words(9999)
    assert settings.max_words == 120


@pytest.mark.parametrize("fmt", list(AnswerFormat))
def test_with_format_accepts_every_enum_member(fmt):
    assert AnswerSettings().with_format(fmt).format == fmt


def test_with_methods_return_new_instance():
    original = AnswerSettings()
    assert original.with_max_words(50) is not original
    assert original.max_words == config.DEFAULT_MAX_WORDS


def test_with_max_words_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_max_words(300)
    assert (updated.max_words, updated.format, updated.list_limit) == (300, AnswerFormat.JSON, 7)


def test_with_format_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_format(AnswerFormat.COMPACT)
    assert (updated.max_words, updated.format, updated.list_limit) == (
        100,
        AnswerFormat.COMPACT,
        7,
    )


def test_with_list_limit_preserves_other_fields():
    settings = AnswerSettings(max_words=100, format=AnswerFormat.JSON, list_limit=7)
    updated = settings.with_list_limit(2)
    assert (updated.max_words, updated.format, updated.list_limit) == (
        100,
        AnswerFormat.JSON,
        2,
    )


def test_with_methods_chain():
    settings = (
        AnswerSettings()
        .with_max_words(42)
        .with_format(AnswerFormat.COMPACT)
        .with_list_limit(5)
    )
    assert (settings.max_words, settings.format, settings.list_limit) == (
        42,
        AnswerFormat.COMPACT,
        5,
    )


def test_answer_settings_error_is_value_error():
    """Тип ошибки — часть контракта: вызывающий код ловит её как ValueError."""
    assert issubclass(AnswerSettingsError, ValueError)


def test_answer_format_values_are_stable_strings():
    """Значения enum попадают в имена файлов ассетов, менять их нельзя молча."""
    assert {f.value for f in AnswerFormat} == {"compact", "json", "patch", "free"}
