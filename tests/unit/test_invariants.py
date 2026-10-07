"""Инварианты домена: правила в запросе, детерминированная проверка, отказ и проверка аргументов.

Положительный и отрицательный пример берётся на каждое правило домена: у правила с запрещёнными
оборотами это текст с оборотом и текст с отрицанием, у правила без оборотов — наличие правила в
сообщении модели и отсутствие ложного срабатывания на его собственном тексте.
"""

import pytest

from core import domains, invariants
from core.invariants import (
    REFUSAL_PREFIX,
    REFUSAL_WINDOW,
    Violation,
    arguments_refusal_text,
    check_answer,
    check_arguments,
    invariants_message,
    is_refusal,
    refusal_text,
    retry_prompt,
)


@pytest.fixture
def rules() -> tuple:
    return domains.load_domain("aurora-qt5").invariants


@pytest.fixture
def terms(rules) -> dict:
    """Оборот правила — по одному на каждое правило, у которого обороты объявлены."""
    described = {rule.number: rule.forbidden[0] for rule in rules if rule.forbidden}
    assert described, "у домена должны быть правила, проверяемые кодом"
    return described


# --- таблица правил ---


def test_table_comes_from_the_domain_and_is_numbered_in_order(rules):
    assert rules
    assert [rule.number for rule in rules] == list(range(1, len(rules) + 1))
    assert all(rule.rule.strip() and rule.source.strip() for rule in rules)


def test_every_rule_is_either_checked_by_code_or_prompt_only(rules):
    """Двойная защита там, где нарушение видно в тексте; остальное держит только промпт."""
    for rule in rules:
        assert isinstance(rule.forbidden, tuple)
        assert all(term == term.casefold() for term in rule.forbidden), (
            "оборот сравнивается с текстом в нижнем регистре"
        )


def test_model_only_rules_are_carried_by_the_message(rules):
    """Правило без оборотов проверяется промптом: код его не судит, но сообщение обязано назвать."""
    message = invariants_message(rules)
    model_only = [rule for rule in rules if not rule.forbidden]

    assert model_only, "у домена должны быть правила, проверяемые только промптом"
    for rule in model_only:
        assert f"{rule.number}. {rule.rule}" in message
        assert check_answer(rule.rule, rules) == (), "текст самого правила не должен ловиться кодом"


# --- проверка ответа: положительный и отрицательный пример на каждое правило ---


def test_each_rule_with_terms_catches_its_own_violation(rules, terms):
    for number, term in terms.items():
        violations = check_answer(f"Проверено: {term} — вот что предлагаю.", rules)

        assert {violation.number for violation in violations} == {number}


def test_each_rule_with_terms_passes_the_negated_mention(rules, terms):
    for number, term in terms.items():
        text = f"Обойдёмся без {term}, всё в порядке."

        assert check_answer(text, rules) == ()


def test_clean_answer_has_no_violations(rules):
    assert check_answer("Соберите пакет командой из документации и приложите ссылку.", rules) == ()


def test_violation_carries_number_rule_and_the_table_term(rules, terms):
    number, term = next(iter(terms.items()))
    rule = next(rule for rule in rules if rule.number == number)

    violations = check_answer(f"Предлагаю {term}.", rules)

    assert violations == (Violation(number=number, rule=rule.rule, term=term),)


def test_two_rules_give_two_violations(rules, terms):
    first, second = list(terms.items())[:2]

    violations = check_answer(f"Сначала {first[1]}, потом {second[1]}.", rules)

    assert [violation.number for violation in violations] == [first[0], second[0]]


def test_one_rule_reports_a_single_violation_even_with_several_terms(rules):
    rule = next(rule for rule in rules if len(rule.forbidden) > 1)
    text = " ".join(rule.forbidden)

    violations = check_answer(text, rules)

    assert [violation.number for violation in violations] == [rule.number]
    assert violations[0].term == rule.forbidden[0]


def test_negation_before_a_plain_mention_does_not_hide_it(rules):
    rule = next(rule for rule in rules if rule.forbidden)
    text = f"Без этого не обойтись: возьмите {rule.forbidden[0]}."

    assert [violation.number for violation in check_answer(text, rules)] == [rule.number]


# --- отказ модели ---


def test_refusal_prefix_within_the_window_skips_the_check(rules, terms):
    """Отказ называет запрещённое и решения не предлагает."""
    number, term = next(iter(terms.items()))
    answer = f"{REFUSAL_PREFIX}: просьба противоречит инварианту {number} — {term} под запретом."

    assert is_refusal(answer)
    assert check_answer(answer, rules) == ()


def test_refusal_inside_a_json_block_is_still_a_refusal(rules, terms):
    term = next(iter(terms.values()))
    answer = f'```json\n{{"error": "Не могу предложить: {term} под запретом"}}\n```'

    assert is_refusal(answer)
    assert check_answer(answer, rules) == ()


def test_refusal_prefix_is_case_insensitive():
    assert is_refusal(f"{REFUSAL_PREFIX.upper()} такое: правило под запретом.")


def test_refusal_phrase_beyond_the_window_does_not_protect_a_solution(rules, terms):
    number, term = next(iter(terms.items()))
    answer = f"Вот решение с {term}: " + "и ещё много слов " * 20 + f"{REFUSAL_PREFIX} иначе."

    assert len(answer) > REFUSAL_WINDOW
    assert not is_refusal(answer)
    assert [violation.number for violation in check_answer(answer, rules)] == [number]


# --- сообщение модели и тексты ---


def test_message_lists_every_rule_with_its_number_and_the_instruction(rules):
    message = invariants_message(rules)

    for rule in rules:
        assert f"{rule.number}. {rule.rule}" in message
    assert REFUSAL_PREFIX in message
    assert invariants.invariants_instruction() in message


def test_message_carries_only_the_rules_and_the_instruction(rules):
    """Список оборотов модели не отправляется — только тексты правил и инструкция."""
    rule_lines = [f"{rule.number}. {rule.rule}" for rule in rules]
    expected = "\n".join(
        [
            "Инварианты домена — правила, которые нельзя нарушать ни при каких условиях:",
            *rule_lines,
            "",
            invariants.invariants_instruction(),
        ]
    )

    assert invariants_message(rules) == expected


def test_message_does_not_send_the_terms_of_a_rule_as_a_list(rules):
    """Обороты живут в данных и в коде: списком они подсказывали бы синонимы вместо соблюдения."""
    listed = [", ".join(rule.forbidden) for rule in rules if len(rule.forbidden) > 1]
    message = invariants_message(rules)

    assert listed, "у домена должно быть правило с несколькими оборотами"
    for group in listed:
        assert group not in message


def test_refusal_text_names_number_rule_and_term(rules, terms):
    number, term = next(iter(terms.items()))
    text = refusal_text(check_answer(f"Предлагаю {term}.", rules))

    assert text.startswith(REFUSAL_PREFIX)
    assert f"инвариант {number}" in text
    assert "«{0}»".format(term) in text
    assert is_refusal(text)


def test_retry_prompt_lists_violations_and_asks_to_rewrite_or_refuse(rules, terms):
    first, second = list(terms.items())[:2]

    text = retry_prompt(check_answer(f"Сначала {first[1]}, потом {second[1]}.", rules))

    assert f"инвариант {first[0]}" in text and f"инвариант {second[0]}" in text
    assert "перепиши" in text.casefold()
    assert "откажи" in text.casefold()
    assert REFUSAL_PREFIX in text


# --- проверка аргументов инструментов до вызова ---


def test_clean_arguments_pass(rules):
    assert check_arguments({"path": "src/main.cpp", "text": "// правка"}, rules) == ()
    assert check_arguments("src/main.cpp", rules) == ()


def test_arguments_carrying_a_forbidden_term_are_refused(rules, terms):
    number, term = next(iter(terms.items()))

    violations = check_arguments({"path": f"keys/{term}", "text": "вставь как есть"}, rules)

    assert [violation.number for violation in violations] == [number]
    assert violations[0].term == term


def test_nested_argument_values_are_checked(rules, terms):
    number, term = next(iter(terms.items()))

    violations = check_arguments({"patch": {"body": {"text": term}}}, rules)

    assert [violation.number for violation in violations] == [number]


def test_negation_does_not_excuse_an_argument(rules, terms):
    """Отрицание — эвристика для прозы: аргумент уходит в репозиторий независимо от соседних слов."""
    number, term = next(iter(terms.items()))

    assert [violation.number for violation in check_arguments(f"без {term}", rules)] == [number]


def test_arguments_of_a_foreign_type_are_still_checked(rules, terms):
    """Значение без JSON-представления проверяется своим текстом, а не пропускается."""
    number, term = next(iter(terms.items()))

    class Payload:
        def __str__(self) -> str:
            return term

    assert [violation.number for violation in check_arguments(Payload(), rules)] == [number]


def test_arguments_refusal_text_names_the_rule_and_the_place(rules, terms):
    number, term = next(iter(terms.items()))
    text = arguments_refusal_text(check_arguments(term, rules))

    assert f"инвариант {number}" in text
    assert f"«{term}»" in text
    assert "в аргументах" in text
    assert not is_refusal(text), "вызов не состоялся — это не ответ модели с отказом"
