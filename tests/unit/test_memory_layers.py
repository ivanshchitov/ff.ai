"""Слои памяти: правила домена, маршрутизация реплик, слияние рабочей памяти и сообщение модели."""

import pytest

from core import domains, memory_layers
from core.domains import MemoryRule

# Пример реплики на каждый образец домена: правило без примера в тесте — правило, которое никто не
# проверял, поэтому соответствие «образец → реплика → ключ» объявлено таблицей и проверяется целиком.
EXAMPLES = {
    r"\bзапомни\b|\bвпредь\b|\bна будущее\b": (
        "Запомни: сборка идёт только через qmake.",
        "решения по проекту",
        memory_layers.LONG_TERM,
    ),
    r"\bдоговорились\b|\bрешили\b|\bусловились\b": (
        "Решили, что .spec правим только вручную.",
        "решения по проекту",
        memory_layers.LONG_TERM,
    ),
    r"в\s+этом\s+проекте": (
        "В этом проекте только qmake.",
        "решения по проекту",
        memory_layers.LONG_TERM,
    ),
    r"\bмы\s+используем\b|\bпользуемся\b|\bу\s+нас\s+принято\b|\bв\s+нашем\s+проекте\b": (
        "Мы используем CMake и ninja.",
        "практика проекта",
        memory_layers.LONG_TERM,
    ),
    r"\bпредпочитаю\b|\bмне\s+нравится\b|\bне\s+люблю\b": (
        "Предпочитаю короткие ответы с примерами кода.",
        "предпочтения",
        memory_layers.LONG_TERM,
    ),
    r"\bновичок\b|\bя\s+опытн\w*|\bопыт\s+в\b": (
        "Я новичок в QML, зато знаю C++.",
        "опыт",
        memory_layers.LONG_TERM,
    ),
    r"\bсобери\w*|\bсобрать\b|\bпочини\w*|\bисправ\w*|\bдобав\w*|\bреализуй\w*|\bсделай\b|\bпомоги\b": (
        "Собери отчёт о сборке пакета.",
        "цель",
        memory_layers.WORKING,
    ),
    r"\bне\s+используй\w*|\bне\s+меняй\w*|\bне\s+предлагай\w*|\bне\s+более\b|\bне\s+больше\b": (
        "Не меняй состав SDK.",
        "ограничения",
        memory_layers.WORKING,
    ),
    r"\bбез\s+(?:qtwidgets|qt6|интернета|сети|зависимост|сборочных\s+скриптов)\w*": (
        "Сделай сборку без интернета.",
        "ограничения",
        memory_layers.WORKING,
    ),
}


@pytest.fixture
def rules() -> tuple:
    domain = domains.load_domain("aurora-qt5")
    assert domain.memory is not None
    return domain.memory.rules


def test_every_rule_has_an_example_in_the_test_table(rules):
    """Новое правило домена обязано прийти вместе с примером — иначе тест краснеет."""
    assert {rule.pattern for rule in rules} == set(EXAMPLES)


def test_examples_route_to_the_declared_key_and_layer(rules):
    for rule in rules:
        message, key, layer = EXAMPLES[rule.pattern]
        routed = {record.key: record for record in memory_layers.route(message, rules)}
        assert key in routed, f"образец {rule.pattern!r} не сработал на своей реплике"
        assert routed[key].layer == layer
        assert routed[key].category == rule.category


def test_question_without_turns_writes_nothing(rules):
    assert memory_layers.route("Что такое moc?", rules) == ()
    assert memory_layers.route("Почему падает валидатор пакета?", rules) == ()


def test_free_phrase_does_not_route_anything(rules):
    """Свободные обороты не должны попадать в слои: правило ловит темы, а не любое слово."""
    assert memory_layers.route("Вопрос без ответа", rules) == ()


def test_recognition_is_case_insensitive_and_inside_sentence(rules):
    records = memory_layers.route("Слушай, РЕШИЛИ перейти на CMake.", rules)

    assert [record.key for record in records] == ["решения по проекту"]


def test_value_is_the_sentence_with_the_match(rules):
    records = memory_layers.route(
        "Предпочитаю короткие ответы. Собери отчёт по сборке, не предлагай Qt 6.", rules
    )

    by_key = {record.key: record.value for record in records}
    assert by_key["предпочтения"] == "Предпочитаю короткие ответы"
    assert by_key["цель"] == "Собери отчёт по сборке, не предлагай Qt 6"
    assert by_key["ограничения"] == "Собери отчёт по сборке, не предлагай Qt 6"


def test_goal_starts_a_new_sentence(rules):
    """Значение — предложение целиком, а не хвост после предыдущего: точка отделяет реплики."""
    records = memory_layers.route("Так. Собери отчёт.", rules)

    assert [record.value for record in records] == ["Собери отчёт"]


def test_one_record_per_key_even_with_several_matching_patterns(rules):
    records = memory_layers.route("Запомни: решили перейти на CMake.", rules)

    assert [record.key for record in records] == ["решения по проекту"]


def test_value_is_truncated_to_the_limit(rules):
    records = memory_layers.route("Собери " + "очень длинный запрос " * 20, rules)

    assert len(records) == 1
    assert len(records[0].value) == memory_layers.MEMORY_VALUE_MAX_CHARS
    assert records[0].value.endswith("…")


def test_pattern_order_decides_the_value_of_a_repeated_key(rules):
    """Правило, идущее ниже, заменяет значение того же ключа — порядок таблицы значим."""
    message = "Запомни про сборку. В этом проекте только qmake."

    forward = memory_layers.route(message, rules)
    backward = memory_layers.route(message, tuple(reversed(rules)))

    assert [record.key for record in forward] == ["решения по проекту"]
    assert [record.key for record in backward] == ["решения по проекту"]
    assert forward[0].value == "В этом проекте только qmake"
    assert backward[0].value == "Запомни про сборку"


def test_records_from_block_keep_the_stored_values():
    records = memory_layers.records_from_block({"цель": "собрать пакет", "ограничения": "без сети"})

    assert [record.key for record in records] == ["цель", "ограничения"]
    assert all(record.layer == memory_layers.WORKING for record in records)
    assert all(record.value for record in records)


def test_merge_working_replaces_by_key_and_keeps_the_rest():
    merged = memory_layers.merge_working(
        {"цель": "старая цель", "ограничения": "без сети"},
        [memory_layers.working_record("цель", "новая цель")],
    )

    assert merged == {"цель": "новая цель", "ограничения": "без сети"}


def test_merge_working_ignores_records_of_other_layers():
    merged = memory_layers.merge_working(
        {"цель": "старая цель"},
        [memory_layers.MemoryRecord(memory_layers.LONG_TERM, "решения", "решения", "текст")],
    )

    assert merged == {"цель": "старая цель"}


def test_merge_working_drops_the_earliest_keys_past_the_limit():
    records = [memory_layers.working_record(f"ключ {index}", "значение") for index in range(3)]

    merged = memory_layers.merge_working({"первый": "значение"}, records, limit=2)

    assert list(merged) == ["ключ 1", "ключ 2"]


def test_message_is_empty_when_layers_are_empty():
    assert memory_layers.memory_message(()) is None


def test_long_term_memory_goes_before_working():
    message = memory_layers.memory_message(
        (
            memory_layers.working_record("цель", "Собрать отчёт по сборке"),
            memory_layers.MemoryRecord(
                memory_layers.LONG_TERM, "решения", "решения по проекту", "только qmake"
            ),
        )
    )

    assert message is not None
    assert message.index("только qmake") < message.index("Собрать отчёт по сборке")
    assert "решения" in message
    assert "цель" in message


def test_message_carries_the_instruction_asset():
    message = memory_layers.memory_message(
        (memory_layers.working_record("цель", "Собрать отчёт"),)
    )

    assert message is not None
    assert memory_layers.memory_instruction()[:30] in message


def test_message_marks_the_category_of_every_record():
    message = memory_layers.memory_message(
        (
            memory_layers.MemoryRecord(memory_layers.LONG_TERM, "опыт", "опыт", "знаю C++"),
            memory_layers.MemoryRecord(
                memory_layers.LONG_TERM, "решения", "решения по проекту", "только qmake"
            ),
        )
    )

    assert message is not None
    assert "опыт · опыт" in message
    assert "решения · решения по проекту" in message


def test_rules_description_covers_every_rule(rules):
    described = memory_layers.describe_rules(rules)

    assert len(described) == len(rules)
    for rule in rules:
        assert any(f"«{rule.pattern}»" in line for line in described)
        assert any(memory_layers.LAYER_LABELS[rule.layer] in line for line in described)
    assert len(set(described)) == len(rules)


def test_describe_rules_marks_the_layer_of_a_record():
    line = memory_layers.describe_rules(
        (MemoryRule(memory_layers.LONG_TERM, "решения", "ключ", "образец", "опись правила"),)
    )

    expected = f"«образец» → {memory_layers.LAYER_LABELS[memory_layers.LONG_TERM]} память"
    assert line[0].startswith(expected)
    assert "решения · ключ" in line[0]
    assert "опись правила" in line[0]


def test_every_layer_has_a_label_lifetime_and_store_hint():
    """Модель слоёв полна: у каждого слоя есть подпись, срок жизни и место хранения."""
    for layer in (memory_layers.SHORT_TERM, memory_layers.WORKING, memory_layers.LONG_TERM):
        assert memory_layers.LAYER_LABELS[layer]
        assert memory_layers.LAYER_LIFETIME[layer]
        assert memory_layers.LAYER_STORE_HINT[layer]
    assert memory_layers.STORABLE_LAYERS == (memory_layers.LONG_TERM, memory_layers.WORKING)


def test_clip_value_keeps_short_values_and_marks_truncated_ones():
    assert memory_layers.clip_value("коротко", limit=10) == "коротко"
    assert memory_layers.clip_value("длинное значение", limit=5) == "длин…"


def test_routing_uses_only_the_given_rules(rules):
    """Правила — аргумент, а не константа модуля: чужой домен маршрутизируется своими правилами."""
    foreign = (
        MemoryRule(memory_layers.LONG_TERM, "заметка", "чужой ключ", r"\bпривет\b", "чужое правило"),
    )

    assert [record.key for record in memory_layers.route("Привет, мир.", foreign)] == ["чужой ключ"]
    assert memory_layers.route("Привет, мир.", rules) == ()
