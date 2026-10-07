"""Чистый модуль компрессии: порог сжатия, разделение хода, сборка сообщений."""

import pytest

from core import context_compressor


# --- порог сжатия ------------------------------------------------------------------------


def test_threshold_in_messages_fires_on_reach():
    assert context_compressor.needs_compression(10, 10) is True
    assert context_compressor.needs_compression(11, 10) is True


def test_threshold_not_fired_below_reach():
    assert context_compressor.needs_compression(9, 10) is False
    assert context_compressor.needs_compression(0, 10) is False

def test_split_keeps_last_exchange_intact():
    turns = []
    for n in range(1, 7):
        turns.append({"role": "user" if n % 2 else "assistant", "content": f"Ход {n}"})

    digest, tail = context_compressor.split_for_digest(turns)

    assert len(digest) == 4 and len(tail) == 2
    assert tail[0]["content"] == "Ход 5" and tail[1]["content"] == "Ход 6"


def test_split_rounds_digest_to_pairs():
    turns = [{"role": "user", "content": str(i)} for i in range(5)]  # нечётная длина
    digest, tail = context_compressor.split_for_digest(turns, keep_last=2)

    assert len(digest) % 2 == 0
    assert len(tail) == 3  # нечётный хвост остаётся в стеке, не сворачивается


def test_split_with_nothing_to_digest_returns_empty_digest():
    turns = [{"role": "user", "content": "Один"}, {"role": "assistant", "content": "Два"}]
    digest, tail = context_compressor.split_for_digest(turns, keep_last=2)

    assert digest == [] and tail == turns


# --- сборка сообщений запроса ------------------------------------------------------------


def test_context_messages_system_first_then_turns():
    turns = [{"role": "user", "content": "В"}, {"role": "assistant", "content": "О"}]

    messages = context_compressor.context_messages("SYSTEM", None, turns)

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == "SYSTEM"


def test_context_messages_inserts_summary_after_system():
    turns = [{"role": "user", "content": "В"}]

    messages = context_compressor.context_messages("SYSTEM", "Старое резюме", turns)

    assert [m["role"] for m in messages] == ["system", "system", "user"]
    assert "Старое резюме" in messages[1]["content"]


def test_context_messages_without_summary_has_one_system():
    turns = [{"role": "user", "content": "В"}]

    messages = context_compressor.context_messages("SYSTEM", None, turns)

    assert [m["role"] for m in messages] == ["system", "user"]
