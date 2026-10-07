"""Расчёт стоимости и учёт токенов: запрос, сессия, сохранённая история."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from . import config


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    pricing = config.MODEL_PRICING.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Приближённая клиентская оценка числа токенов в тексте (без обращения к API).

    Эвристика «символы / ESTIMATED_CHARS_PER_TOKEN»: точность не гарантируется, в отчётах
    помечается как оценка («≈»). Кириллица тратит заметно больше токена на слово, чем
    английский текст, поэтому константа консервативно занижает размер.
    """
    return max(1, math.ceil(len(text) / config.ESTIMATED_CHARS_PER_TOKEN))


@dataclass
class SessionUsage:
    """Снимок расхода: число запросов, токены, стоимость, средний размер запроса.

    cost_usd равен None, если расход ещё не считался или стоимость хоть одного
    учтённого запроса неизвестна (для модели нет цены).
    """

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    average_prompt_tokens: float = 0.0


class SessionLedger:
    """Накопитель метрик использования за текущую сессию агента.

    Каждая успешная запись — один фактический запрос к API. reset() привязан к
    очистке контекста (/clear): статистика сессии начинается с нуля вместе с ним.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._usage = SessionUsage()
        self._cost_total = 0.0
        self._cost_known = True

    def record(self, meta: Any) -> None:
        """Учитывает метрики одного успешного запроса (достаточно полей AnswerMeta)."""
        usage = self._usage
        usage.requests += 1
        usage.prompt_tokens += meta.prompt_tokens
        usage.completion_tokens += meta.completion_tokens
        usage.total_tokens += meta.total_tokens
        if meta.cost_usd is None:
            self._cost_known = False
        else:
            self._cost_total += meta.cost_usd
        usage.cost_usd = self._cost_total if self._cost_known else None
        usage.average_prompt_tokens = usage.prompt_tokens / usage.requests

    @property
    def usage(self) -> SessionUsage:
        return self._usage


def sum_usage(dialogues: Iterable[Dict[str, Any]]) -> SessionUsage:
    """Суммирует метрики использования по записям истории.

    Записи без блока usage (старый формат файла) пропускаются; стоимость с null у
    хоть одной учтённой записи делает итоговую стоимость неизвестной. Файл истории не
    вытесняется, поэтому итог — расход всего сохранённого диалога.
    """
    usage = SessionUsage()
    cost_total = 0.0
    cost_known = True
    for item in dialogues:
        record_usage = item.get("usage")
        if not isinstance(record_usage, dict):
            continue
        usage.requests += 1
        usage.prompt_tokens += record_usage.get("prompt_tokens", 0)
        usage.completion_tokens += record_usage.get("completion_tokens", 0)
        usage.total_tokens += record_usage.get("total_tokens", 0)
        cost = record_usage.get("cost_usd")
        if cost is None:
            cost_known = False
        else:
            cost_total += cost
    if usage.requests:
        usage.average_prompt_tokens = usage.prompt_tokens / usage.requests
        usage.cost_usd = cost_total if cost_known else None
    return usage
