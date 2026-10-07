"""Слои памяти агента: что попадает в какой слой, по какому правилу и в каком виде это уходит модели.

Три слоя: краткосрочная память (ход текущего диалога), рабочая (данные текущей задачи) и
долговременная (сведения о пользователе между сессиями). Модуль — чистая логика без ввода-вывода:
он собирает системное сообщение со слоями для запроса к модели и описывает правила для отчёта
интерфейса. Хранилищами владеют агент (рабочая память лежит в конверте истории) и
`core/long_term_memory` (отдельный файл).

Таблицы правил здесь нет и быть не должно: маршрутизация — данные пакета домена
(`DomainMemory`), потому что у другой области применимости другие слои, категории и ключи.
Ядро знает только модель слоёв и форму записи.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from . import config

if TYPE_CHECKING:  # правила приходят данными; импорт нужен только проверке типов
    from .domains import MemoryRule

# Идентификаторы слоёв: короткие машинные имена, подписи — в LAYER_LABELS.
SHORT_TERM = "short_term"
WORKING = "working"
LONG_TERM = "long_term"

# Слои, которые может заполнить правило маршрутизации. Краткосрочный слой — это сами реплики
# диалога, он собирается из обменов, а не из правил; таблица допустимых значений нужна загрузчику
# домена, чтобы опечатка в слое останавливала старт, а не теряла запись молча.
STORABLE_LAYERS: Tuple[str, ...] = (LONG_TERM, WORKING)

# Срок жизни слоя — часть модели памяти, а не деталь отчёта: краткосрочная и рабочая живут текущий
# диалог (команда /clear их опустошает), долговременная — между сессиями и /clear её не трогает.
LAYER_LABELS = {
    SHORT_TERM: "краткосрочная",
    WORKING: "рабочая",
    LONG_TERM: "долговременная",
}

LAYER_LIFETIME = {
    SHORT_TERM: "текущий диалог, очищается /clear",
    WORKING: "текущая задача, очищается /clear",
    LONG_TERM: "между сессиями, /clear не трогает",
}

# Хранилище слоя — тоже часть модели: слои лежат отдельно друг от друга.
LAYER_STORE_HINT = {
    SHORT_TERM: "конверт истории, блок dialogues",
    WORKING: "конверт истории, блок working",
    LONG_TERM: "отдельный файл долговременной памяти",
}

# Категория записи для всего, что записано вручную: у записи правила категория приходит из данных,
# а у свободной заметки её нет.
CATEGORY_NOTE = "заметка"

MEMORY_PROMPT_ASSET = "memory_prompt.md"

# Предел длины записи: записи уходят в каждый запрос, поэтому длинная реплика в слой не помещается
# целиком. Предел числа ключей нужен рабочему слою: его ключи задаёт реплика, а не таблица.
MEMORY_VALUE_MAX_CHARS = 120
MAX_WORKING_KEYS = 10

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

MESSAGE_HEADER = "Память ассистента (слои памяти, от долговременной к текущей задаче):"

_LAYER_HEADER = {
    LONG_TERM: "Долговременная память (сведения о пользователе между сессиями):",
    WORKING: "Рабочая память (данные текущей задачи этого диалога):",
}

_TRUNCATION_MARK = "…"


def clip_value(value: str, limit: int = MEMORY_VALUE_MAX_CHARS) -> str:
    """Обрезает значение до предела длины записи; многоточие входит в предел."""
    if len(value) <= limit:
        return value
    return value[: limit - 1] + _TRUNCATION_MARK


@dataclass(frozen=True)
class MemoryRecord:
    """Запись слоя памяти: слой, категория, ключ и значение.

    Ключ задан правилом домена или командой пользователя; повторная запись по тому же ключу
    заменяет значение, а не добавляет вторую запись.
    """

    layer: str
    category: str
    key: str
    value: str


def working_record(key: str, value: str) -> MemoryRecord:
    """Запись рабочего слоя: категорией служит сам ключ, набор которого задают правила домена."""
    return MemoryRecord(layer=WORKING, category=key, key=key, value=value)


def records_from_block(block: Mapping[str, str]) -> Tuple[MemoryRecord, ...]:
    """Записи рабочего слоя из плоского блока «ключ — значение» (блок working конверта истории)."""
    return tuple(working_record(key, value) for key, value in block.items())


def merge_working(
    block: Mapping[str, str],
    records: Sequence[MemoryRecord],
    limit: int = MAX_WORKING_KEYS,
) -> Dict[str, str]:
    """Новый плоский блок рабочей памяти: замена по ключу и вытеснение самых ранних ключей.

    Рабочая память — один блок на диалог, поэтому повторная запись по ключу обновляет значение,
    а не удлиняет список; предел нужен потому, что ключи приходят из реплик пользователя.
    """
    merged: Dict[str, str] = dict(block)
    for record in records:
        if record.layer != WORKING:
            continue
        merged[record.key] = record.value
    while len(merged) > limit:
        del merged[next(iter(merged))]
    return merged


def _sentence_with(text: str, start: int) -> str:
    """Предложение реплики, в котором сработал образец, — оно и становится значением записи.

    Значение — то, что сказал пользователь, а не оборот правила: иначе в памяти лежало бы
    «мне нравится» без того, что именно нравится.
    """
    begin = max(text.rfind(separator, 0, start) for separator in ".!?…\n") + 1
    end = len(text)
    for separator in ".!?…\n":
        found = text.find(separator, start)
        if found != -1:
            end = min(end, found)
    return text[begin:end].strip()


def route(message: str, rules: Sequence["MemoryRule"]) -> Tuple[MemoryRecord, ...]:
    """Разбирает реплику пользователя правилами домена и возвращает записи по одной на ключ.

    Ни одного запроса к модели: решение принимает таблица правил, поэтому маршрут детерминирован,
    не тратит токены и не зависит от ответа модели. Реплика без распознанных оборотов не даёт
    записей — она остаётся только в краткосрочном слое как ход диалога.
    """
    found: Dict[str, MemoryRecord] = {}
    for rule in rules:
        matched = re.search(rule.pattern, message, re.IGNORECASE)
        if matched is None:
            continue
        found[rule.key] = MemoryRecord(
            layer=rule.layer,
            category=rule.category,
            key=rule.key,
            value=clip_value(_sentence_with(message, matched.start())),
        )
    return tuple(found.values())


@lru_cache(maxsize=None)
def memory_instruction() -> str:
    """Инструкция работы со слоями памяти из ассета assets/memory_prompt.md."""
    return (config.ASSETS_DIR / MEMORY_PROMPT_ASSET).read_text(encoding="utf-8").strip()


def memory_message(records: Sequence[MemoryRecord]) -> Optional[str]:
    """Системное сообщение со слоями памяти для запроса к модели; пустые слои сообщения не дают.

    Слои печатаются от долговременной памяти к рабочей, а записи — с меткой категории: модель
    должна различать сведения о пользователе и данные текущей задачи, чтобы свежая реплика могла
    их перевесить. Инструкция взаимодействия со слоями — в ассете.
    """
    if not records:
        return None
    lines: List[str] = [MESSAGE_HEADER]
    for layer in STORABLE_LAYERS:
        layer_records = [record for record in records if record.layer == layer]
        if not layer_records:
            continue
        lines.append("")
        lines.append(_LAYER_HEADER[layer])
        for record in layer_records:
            lines.append(f"- {record.category} · {record.key}: {record.value}")
    lines.append("")
    lines.append(memory_instruction())
    return "\n".join(lines)


def describe_rules(rules: Sequence["MemoryRule"]) -> Tuple[str, ...]:
    """Правила маршрутизации текстом — для отчёта интерфейса, без обращения к модели."""
    described: List[str] = []
    for rule in rules:
        layer = LAYER_LABELS.get(rule.layer, rule.layer)
        described.append(
            f"«{rule.pattern}» → {layer} память ({rule.category} · {rule.key}): {rule.description}"
        )
    return tuple(described)
