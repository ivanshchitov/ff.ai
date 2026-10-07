"""Стратегии управления контекстом: окно, факты, ветки — чистая логика без HTTP."""

import pytest

from core import config
from core import context_strategies as strategies


def _turns(count: int):
    """Ходы стека: `count` обменов user/assistant."""
    turns = []
    for index in range(1, count + 1):
        turns.append({"role": "user", "content": f"Вопрос {index}"})
        turns.append({"role": "assistant", "content": f"Ответ {index}"})
    return turns


# --- окно -------------------------------------------------------------------------------


def test_window_keeps_last_whole_exchanges():
    turns = _turns(3)

    window = strategies.window_messages(turns, 5)

    assert [m["content"] for m in window] == ["Вопрос 2", "Ответ 2", "Вопрос 3", "Ответ 3"]


def test_window_of_young_dialog_sends_everything():
    turns = _turns(2)

    assert strategies.window_messages(turns, 10) == turns


def test_window_never_drops_the_last_exchange():
    turns = _turns(4)

    assert strategies.window_messages(turns, 1) == turns[-2:]


def test_window_of_empty_stack_is_empty():
    assert strategies.window_messages([], 10) == []


# --- факты ------------------------------------------------------------------------------


def test_merge_replaces_value_of_existing_key():
    merged = strategies.merge_facts({"цель": "собираем ТЗ"}, {"цель": "собираем ТЗ по игре"})

    assert merged == {"цель": "собираем ТЗ по игре"}


def test_merge_keeps_order_of_first_appearance():
    merged = strategies.merge_facts({"цель": "ТЗ"}, {"ограничение": "10 сообщений"})

    assert list(merged) == ["цель", "ограничение"]


def test_merge_evicts_earliest_keys_over_the_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_FACTS_KEYS", 2)

    merged = strategies.merge_facts({"a": "1", "b": "2"}, {"c": "3"})

    assert merged == {"b": "2", "c": "3"}


def test_block_over_the_limit_keeps_the_newest_keys():
    facts = {f"ключ {index}": str(index) for index in range(config.MAX_FACTS_KEYS + 3)}

    merged = strategies.merge_facts(facts, {})

    assert len(merged) == config.MAX_FACTS_KEYS
    assert "ключ 0" not in merged


def test_render_facts_lists_pairs():
    text = strategies.render_facts({"цель": "ТЗ", "жанр": "настольные игры"})

    assert "цель: ТЗ" in text and "жанр: настольные игры" in text


def test_render_facts_of_empty_block_is_empty():
    assert strategies.render_facts({}) == ""


def test_parse_direct_json_object():
    assert strategies.parse_facts_response('{"цель": "ТЗ"}') == {"цель": "ТЗ"}


def test_parse_json_inside_code_block():
    text = 'Вот факты:\n```json\n{"жанр": "карточная"}\n```\n'

    assert strategies.parse_facts_response(text) == {"жанр": "карточная"}


def test_parse_of_empty_object_is_an_empty_block_not_a_failure():
    assert strategies.parse_facts_response("{}") == {}


def test_parse_returns_none_for_non_json():
    assert strategies.parse_facts_response("Фактов не нашёл, всё уже известно.") is None


def test_parse_skips_nested_values():
    parsed = strategies.parse_facts_response('{"цель": "ТЗ", "детали": {"a": 1}}')

    assert parsed == {"цель": "ТЗ"}


def test_parse_coerces_scalar_values():
    parsed = strategies.parse_facts_response('{"игроков": 4, "договорились": true}')

    assert parsed == {"игроков": "4", "договорились": "True"}


def test_facts_request_carries_block_and_pending_messages():
    messages = strategies.build_facts_messages({"цель": "ТЗ"}, ["Добавь ограничение"])

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == strategies.facts_instruction()
    assert "цель: ТЗ" in messages[1]["content"]
    assert "Добавь ограничение" in messages[1]["content"]


def test_facts_request_without_block_marks_it_empty():
    messages = strategies.build_facts_messages({}, ["Первый вопрос"])

    assert "Первый вопрос" in messages[1]["content"]
    assert "цель" not in messages[1]["content"]


# --- ветки ------------------------------------------------------------------------------


def _tree_with_exchanges(count: int):
    """Дерево веток поверх лога из `count` обменов, каждый обмен учтён в активной ветке."""
    log = []
    tree = strategies.BranchTree(log)
    for index in range(1, count + 1):
        log.append({"role": "user", "content": f"Вопрос {index}"})
        log.append({"role": "assistant", "content": f"Ответ {index}"})
        tree.add_exchange()
    return tree, log


def test_restored_log_becomes_the_single_branch():
    log = _turns(3)
    tree = strategies.BranchTree(log)

    assert tree.branches() == ((strategies.DEFAULT_BRANCH_NAME, 3),)
    assert tree.active_name == strategies.DEFAULT_BRANCH_NAME
    assert tree.active_messages() == log


def test_new_branch_starts_from_checkpoint():
    tree, _ = _tree_with_exchanges(2)
    tree.checkpoint()

    name = tree.new_branch()

    assert name == "ветка 2"
    assert tree.active_name == "ветка 2"
    assert [m["content"] for m in tree.active_messages()] == [
        "Вопрос 1",
        "Ответ 1",
        "Вопрос 2",
        "Ответ 2",
    ]


def test_new_branch_without_checkpoint_starts_empty():
    tree, _ = _tree_with_exchanges(2)

    tree.new_branch()

    assert tree.active_messages() == []


def test_exchange_lands_only_in_the_active_branch():
    tree, log = _tree_with_exchanges(2)
    tree.checkpoint()
    tree.new_branch()
    log.append({"role": "user", "content": "Вопрос ветки"})
    log.append({"role": "assistant", "content": "Ответ ветки"})
    tree.add_exchange()

    assert [m["content"] for m in tree.active_messages()][-2:] == [
        "Вопрос ветки",
        "Ответ ветки",
    ]

    assert tree.switch(strategies.DEFAULT_BRANCH_NAME) is True
    assert [m["content"] for m in tree.active_messages()] == [
        "Вопрос 1",
        "Ответ 1",
        "Вопрос 2",
        "Ответ 2",
    ]


def test_switch_to_unknown_branch_keeps_the_active_one():
    tree, _ = _tree_with_exchanges(1)

    assert tree.switch("нет такой ветки") is False
    assert tree.active_name == strategies.DEFAULT_BRANCH_NAME


def test_branches_report_names_with_exchange_counts():
    tree, _ = _tree_with_exchanges(2)
    tree.checkpoint()
    tree.new_branch()

    assert tree.branches() == ((strategies.DEFAULT_BRANCH_NAME, 2), ("ветка 2", 2))


def test_reset_returns_to_a_single_empty_branch():
    tree, _ = _tree_with_exchanges(2)
    tree.checkpoint()
    tree.new_branch()

    tree.reset()

    assert tree.branches() == ((strategies.DEFAULT_BRANCH_NAME, 0),)
    assert tree.active_messages() == []
    assert tree.switch("ветка 2") is False


# --- пакет сообщений для извлекателя ------------------------------------------------------


def test_facts_batch_takes_everything_within_budget():
    pending = ["Первое сообщение", "Второе сообщение"]

    assert strategies.facts_batch({}, pending, config.MAX_MAX_SESSION_TOKENS) == pending


def test_facts_batch_always_takes_at_least_one_message():
    """Иначе очередь не сдвинется: первое сообщение принимается даже при крошечном бюджете."""
    pending = ["Очень длинное сообщение " * 500, "Второе"]

    assert strategies.facts_batch({}, pending, 1) == pending[:1]


def test_facts_batch_stops_before_the_message_that_breaks_the_budget():
    pending = [f"Сообщение {index} " + "х" * 400 for index in range(5)]

    batch = strategies.facts_batch({}, pending, 300)

    assert 0 < len(batch) < len(pending)
    assert batch == pending[: len(batch)]


def test_facts_batch_accounts_for_the_current_block():
    """Текущий блок входит в запрос, поэтому сокращает место для новых сообщений."""
    pending = ["Сообщение " + "х" * 300 for _ in range(4)]
    facts = {f"ключ {index}": "значение " + "у" * 200 for index in range(10)}

    with_block = strategies.facts_batch(facts, pending, 400)

    assert len(with_block) < len(pending)


def test_facts_batch_of_empty_queue_is_empty():
    assert strategies.facts_batch({}, [], 1000) == []
