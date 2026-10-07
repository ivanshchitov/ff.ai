"""Инварианты домена: правила, которые агент не имеет права нарушать.

Инварианты — не диалог, не память и не профиль: таблица правил приходит из пакета домена
(`Domain.invariants`), уходит модели отдельным системным сообщением в каждый запрос, и по ней же
проверяются ответ и аргументы вызова инструментов. Двойная защита: инструкция модели — сверить
просьбу с каждым правилом и отказать, если просьба им противоречит; проверка кодом — запрещённые
формулировки правила в ответе или в аргументах означают нарушение, и такое действие до пользователя
не доходит. Сам текст правил и запрещённые обороты — данные домена, поэтому ядро не знает ни одного
правила и ни одного запрещённого слова.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

from . import config
from .domains import Invariant

INVARIANTS_PROMPT_ASSET = "invariants_prompt.md"

# Фиксированное начало отказа модели: ответ с этими словами в первых REFUSAL_WINDOW символах —
# отказ, а не решение, и на запрещённые слова не проверяется: отказ обязан назвать, что именно
# запрещено, иначе он ничего не объясняет. Окно, а не «начинается с»: в JSON-формате отказ лежит
# внутри {"error": "..."} в блоке кода.
REFUSAL_PREFIX = "Не могу предложить"
REFUSAL_WINDOW = 200

# Отрицание перед запрещённым словом — соблюдение правила, а не нарушение: «без приложения»,
# «не используем это». Маркеры ищутся как начала слов в окне символов перед вхождением.
# Эвристика, не грамматика: «нельзя без приложения» пропустится, «обязательно приложение» —
# поймается; промпт остаётся первой линией защиты, проверка — второй.
NEGATIONS = ("без", "не", "нет", "никаких", "ни", "запрещ", "отсутств")
NEGATION_WINDOW = 20


@dataclass(frozen=True)
class Violation:
    """Нарушение правила: номер, текст правила и оборот таблицы (не фрагмент ответа)."""

    number: int
    rule: str
    term: str


def is_refusal(answer: str) -> bool:
    """Отказ модели — фиксированное начало отказа в первых REFUSAL_WINDOW символах ответа."""
    return REFUSAL_PREFIX.casefold() in answer[:REFUSAL_WINDOW].casefold()


def check_answer(
    answer: str, invariants: Sequence[Invariant]
) -> Tuple[Violation, ...]:
    """Нарушения правил в ответе: по одному на правило, с первым найденным оборотом.

    Сравнение регистронезависимое, по вхождению; отказ модели не проверяется — он решения не
    предлагает, а назвать запрещённое в объяснении отказа допустимо.
    """
    if is_refusal(answer):
        return ()
    return _scan(answer.casefold(), invariants)


def check_arguments(
    arguments: object, invariants: Sequence[Invariant]
) -> Tuple[Violation, ...]:
    """Нарушения правил в аргументах вызова инструмента: проверка до запуска процесса.

    Проверка строже проверки ответа: оговорки об отрицании здесь нет. Отрицание — эвристика для
    прозы, а аргумент инструмента — факт: путь с ключом подписи попадёт в репозиторий или в журнал
    независимо от того, что стоит в тексте рядом. Причина отказа называется так же, как для ответа.
    """
    return _scan(_arguments_text(arguments).casefold(), invariants, allow_negation=False)


def _arguments_text(arguments: object) -> str:
    """Текст аргументов для проверки: строки как есть, структуры — их JSON-представлением.

    Значения вложенных структур тоже проверяются, поэтому аргументы не «уплощаются» до ключей:
    секрет может лежать в значении любого уровня.
    """
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(arguments)


def _scan(
    text: str, invariants: Sequence[Invariant], allow_negation: bool = True
) -> Tuple[Violation, ...]:
    violations: List[Violation] = []
    for invariant in invariants:
        for term in invariant.forbidden:
            if _mentions(text, term.casefold(), allow_negation):
                violations.append(Violation(invariant.number, invariant.rule, term))
                break
    return tuple(violations)


def _mentions(text: str, term: str, allow_negation: bool) -> bool:
    """Есть ли вхождение оборота — с оговоркой об отрицании или без неё."""
    start = text.find(term)
    while start != -1:
        before = text[max(0, start - NEGATION_WINDOW) : start]
        if not allow_negation or not _negated(before):
            return True
        start = text.find(term, start + 1)
    return False


def _negated(before: str) -> bool:
    words = before.replace("—", " ").replace("-", " ").split()
    return any(word.strip("«»\"'(),.:;!?").startswith(NEGATIONS) for word in words)


@lru_cache(maxsize=None)
def invariants_instruction() -> str:
    """Инструкция работы с инвариантами из ассета assets/invariants_prompt.md."""
    return (config.ASSETS_DIR / INVARIANTS_PROMPT_ASSET).read_text(encoding="utf-8").strip()


def invariants_message(invariants: Sequence[Invariant]) -> str:
    """Системное сообщение с правилами домена для запроса к модели.

    Модели уходят правила с номерами и инструкция; запрещённые обороты остаются в коде и в данных —
    список слов провоцировал бы обход синонимами вместо соблюдения правила.
    """
    lines: List[str] = ["Инварианты домена — правила, которые нельзя нарушать ни при каких условиях:"]
    lines.extend(f"{invariant.number}. {invariant.rule}" for invariant in invariants)
    lines.append("")
    lines.append(invariants_instruction())
    return "\n".join(lines)


def _describe(violation: Violation, where: str = "в ответе") -> str:
    return f"инвариант {violation.number} «{violation.rule}» ({where}: «{violation.term}»)"


def refusal_text(violations: Sequence[Violation]) -> str:
    """Отказ приложения вместо отклонённого ответа: называет правило и найденный оборот."""
    described = "; ".join(_describe(violation) for violation in violations)
    return f"{REFUSAL_PREFIX}: ответ нарушает {described}."


def arguments_refusal_text(violations: Sequence[Violation]) -> str:
    """Отказ до вызова инструмента: называет правило и оборот, найденный в аргументах.

    Отдельный текст, а не `refusal_text`: отказ модели «Не могу предложить» здесь был бы неправдой —
    ничего не предлагалось, вызов просто не состоялся, и причина у него своя.
    """
    described = "; ".join(
        _describe(violation, where="в аргументах") for violation in violations
    )
    return f"Вызов отменён: аргументы нарушают {described}."


def retry_prompt(violations: Sequence[Violation]) -> str:
    """Ход пользователя для повторного запроса: перечень нарушений и просьба переписать или отказать."""
    lines = ["Твой ответ нарушает инварианты домена:"]
    lines.extend(f"- {_describe(violation)}" for violation in violations)
    lines.append("")
    lines.append(
        "Перепиши ответ так, чтобы он не нарушал ни один инвариант, или откажи по инструкции об "
        f"инвариантах (начни с «{REFUSAL_PREFIX}», назови номер и текст правила и объясни причину)."
    )
    return "\n".join(lines)
