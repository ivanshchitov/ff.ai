"""Расчёт стоимости запроса по таблице цен моделей."""

from core import usage


def test_estimate_cost_known_model():
    cost = usage.estimate_cost("deepseek-v4.1-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == 0.30 + 1.20


def test_estimate_cost_unknown_model_returns_none():
    cost = usage.estimate_cost("no-such-model", prompt_tokens=100, completion_tokens=100)
    assert cost is None


"""--- День 8: подсчёт токенов сессии и истории ---"""

from types import SimpleNamespace

from core.usage import SessionLedger, estimate_tokens, sum_usage


def _request(prompt_tokens=10, completion_tokens=5, cost_usd=0.5):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd,
    )


def test_estimate_tokens_counts_nonempty_text():
    assert estimate_tokens("abcdef") == 2


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("abcd") == 2  # 4 символа / 3 = 1.33 -> 2


def test_estimate_tokens_never_below_one():
    assert estimate_tokens("") == 1


def test_ledger_accumulates_requests():
    ledger = SessionLedger()
    ledger.record(_request(prompt_tokens=10, completion_tokens=5, cost_usd=0.5))
    ledger.record(_request(prompt_tokens=20, completion_tokens=7, cost_usd=0.25))
    usage = ledger.usage
    assert usage.requests == 2
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 12
    assert usage.total_tokens == 42
    assert usage.cost_usd == 0.75
    assert usage.average_prompt_tokens == 15.0


def test_ledger_unknown_cost_poisons_session_total():
    ledger = SessionLedger()
    ledger.record(_request(cost_usd=0.5))
    ledger.record(_request(cost_usd=None))
    assert ledger.usage.cost_usd is None


def test_ledger_without_requests_has_no_cost():
    ledger = SessionLedger()
    assert ledger.usage.requests == 0
    assert ledger.usage.total_tokens == 0
    assert ledger.usage.cost_usd is None


def test_ledger_reset_starts_from_zero():
    ledger = SessionLedger()
    ledger.record(_request())
    ledger.reset()
    assert ledger.usage.requests == 0
    assert ledger.usage.total_tokens == 0


def test_sum_usage_skips_records_without_usage():
    records = [
        {"question": "q1", "answer": "a1", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.5}},
        {"question": "q2", "answer": "a2"},
    ]
    usage = sum_usage(records)
    assert usage.requests == 1
    assert usage.prompt_tokens == 10
    assert usage.total_tokens == 15
    assert usage.cost_usd == 0.5


def test_sum_usage_unknown_cost_poisons_total():
    records = [{"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost_usd": None}}]
    assert sum_usage(records).cost_usd is None


def test_sum_usage_of_empty_history():
    usage = sum_usage([])
    assert usage.requests == 0
    assert usage.total_tokens == 0
    assert usage.cost_usd is None
