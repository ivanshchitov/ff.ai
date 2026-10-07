"""Настройки формата и объёма ответа (общие для сессии TUI)."""

from dataclasses import dataclass, replace
from enum import Enum

from . import config


class AnswerFormat(str, Enum):
    """Формат ответа, выбираемый на экране настроек."""

    COMPACT = "compact"
    JSON = "json"
    PATCH = "patch"
    FREE = "free"


class ContextStrategy(str, Enum):
    """Стратегия управления контекстом сессии, выбираемая на экране /settings."""

    SUMMARY = "summary"
    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


class AnswerSettingsError(ValueError):
    """Невалидное значение настройки (без тихого клампинга)."""


@dataclass
class AnswerSettings:
    max_words: int = config.DEFAULT_MAX_WORDS
    format: AnswerFormat = AnswerFormat.FREE
    list_limit: int = config.DEFAULT_LIST_LIMIT
    temperature: float = config.TEMPERATURE
    compress_after: int = config.DEFAULT_COMPRESS_AFTER
    max_session_tokens: int = config.DEFAULT_MAX_SESSION_TOKENS
    context_strategy: ContextStrategy = ContextStrategy(config.DEFAULT_CONTEXT_STRATEGY)

    def with_max_words(self, value: int) -> "AnswerSettings":
        if not config.MIN_MAX_WORDS <= value <= config.MAX_MAX_WORDS:
            raise AnswerSettingsError(
                f"Значение должно быть в диапазоне {config.MIN_MAX_WORDS}..{config.MAX_MAX_WORDS}."
            )
        return replace(self, max_words=value)

    def with_format(self, value: AnswerFormat) -> "AnswerSettings":
        return replace(self, format=value)

    def with_list_limit(self, value: int) -> "AnswerSettings":
        if not config.MIN_LIST_LIMIT <= value <= config.MAX_LIST_LIMIT:
            raise AnswerSettingsError(
                f"Значение должно быть в диапазоне {config.MIN_LIST_LIMIT}..{config.MAX_LIST_LIMIT}."
            )
        return replace(self, list_limit=value)

    def with_temperature(self, value: float) -> "AnswerSettings":
        if not config.MIN_TEMPERATURE <= value <= config.MAX_TEMPERATURE:
            raise AnswerSettingsError(
                f"Температура должна быть в диапазоне "
                f"{config.MIN_TEMPERATURE}..{config.MAX_TEMPERATURE}."
            )
        # Пользовательский ввод ограничен одним знаком после точки ещё на экране
        # настроек; здесь та же проверка служит инвариантом для программных вызовов.
        if round(value, 1) != value:
            raise AnswerSettingsError(
                "Температура задаётся числом не более чем с одним знаком после точки."
            )
        return replace(self, temperature=value)

    def with_compress_after(self, value: int) -> "AnswerSettings":
        if not config.MIN_COMPRESS_AFTER <= value <= config.MAX_COMPRESS_AFTER:
            raise AnswerSettingsError(
                f"Порог сжатия должен быть в диапазоне "
                f"{config.MIN_COMPRESS_AFTER}..{config.MAX_COMPRESS_AFTER}."
            )
        # Пользовательский ввод ограничен целыми ещё на экране настроек; здесь та же
        # проверка служит инвариантом для программных вызовов.
        if round(value) != value:
            raise AnswerSettingsError("Порог сжатия задаётся целым числом сообщений.")
        return replace(self, compress_after=value)

    def with_max_session_tokens(self, value: int) -> "AnswerSettings":
        if not config.MIN_MAX_SESSION_TOKENS <= value <= config.MAX_MAX_SESSION_TOKENS:
            raise AnswerSettingsError(
                f"Потолок контекста должен быть в диапазоне "
                f"{config.MIN_MAX_SESSION_TOKENS}..{config.MAX_MAX_SESSION_TOKENS}."
            )
        if round(value) != value:
            raise AnswerSettingsError("Потолок контекста задаётся целым числом токенов.")
        return replace(self, max_session_tokens=value)

    def with_context_strategy(self, value: ContextStrategy) -> "AnswerSettings":
        """Смена стратегии управления контекстом; неизвестное значение — ошибка, не дефолт."""
        try:
            strategy = ContextStrategy(value)
        except ValueError:
            raise AnswerSettingsError(
                "Стратегия контекста: допустимы "
                + ", ".join(config.CONTEXT_STRATEGIES)
                + "."
            )
        return replace(self, context_strategy=strategy)
