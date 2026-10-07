"""Чистая логика стратегий управления контекстом: окно сообщений, факты, ветки диалога.

Без HTTP и без rich, по образцу `core/context_compressor`: агент оркестрирует запросы, а
модуль решает чистую часть — какие ходы попадают в запрос, как сливается блок фактов и как
устроены ветки. Сжатие истории живёт по той же схеме в `core/context_compressor.py`.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence

from . import config
from .context_compressor import LAST_EXCHANGE_MESSAGES
from .usage import estimate_tokens

FACTS_PROMPT_ASSET = "facts_prompt.md"

FACTS_MESSAGE_TEMPLATE = (
    "Известные факты текущего диалога (ключ — значение), обновляй их при новых сведениях:\n{facts}"
)

FACTS_USER_TEMPLATE = (
    "Текущий блок фактов:\n{facts}\n\n---\n\nНовые сообщения пользователя:\n{messages}"
)

_PLACEHOLDER_NO_FACTS = "Фактов пока нет."

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

DEFAULT_BRANCH_NAME = "ветка 1"

_SCALARS = (str, int, float, bool)


# --- окно -----------------------------------------------------------------------------------


def window_messages(turns: List[Dict[str, str]], window: int) -> List[Dict[str, str]]:
    """Последние сообщения стека в границе окна, целыми обменами.

    Окно — размер в сообщениях; ходы берутся парой user/assistant, поэтому фактический
    размер округляется вниз до чётного. Последний обмен попадает в запрос всегда: без него
    модели нечего отвечать, даже если окно задано в одно сообщение.
    """
    keep = min(len(turns), max(window - window % 2, LAST_EXCHANGE_MESSAGES))
    return list(turns[-keep:]) if keep else []


# --- факты ----------------------------------------------------------------------------------


def merge_facts(current: Dict[str, str], fresh: Dict[str, str]) -> Dict[str, str]:
    """Сливает блок фактов: значение существующего ключа заменяется, новые ключи дописываются.

    Порядок первого появления ключа сохраняется, поэтому вытеснение за предел (см.
    `config.MAX_FACTS_KEYS`) убирает самые старые по времени появления ключи.
    """
    merged = dict(current)
    merged.update(fresh)
    while len(merged) > config.MAX_FACTS_KEYS:
        merged.pop(next(iter(merged)))
    return merged


def render_facts(facts: Dict[str, str]) -> str:
    """Текст блока фактов; пустой блок — пустая строка (сообщение не добавляется)."""
    if not facts:
        return ""
    return "\n".join(f"- {key}: {value}" for key, value in facts.items())


def parse_facts_response(text: str) -> Optional[Dict[str, str]]:
    """Разбирает ответ извлекателя в блок фактов; None — ответ не JSON-объект пар.

    Отличается от «пустой блок»: `{}` — валидный ответ «новых фактов нет» (обновление
    прошло, очередь сообщений очищается), а None — сбой, при котором блок не меняется.
    Вложенные структуры пропускаются: блок — плоский словарь строк.
    """
    data = _load_json_object(text)
    if data is None:
        return None
    parsed: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, _SCALARS) and isinstance(value, _SCALARS):
            parsed[str(key).strip()] = str(value).strip()
    return parsed


def _load_json_object(text: str) -> Optional[dict]:
    candidates = [text.strip()]
    fenced = _FENCED_JSON_RE.search(text)
    if fenced is not None:
        candidates.append(fenced.group(1))
    embedded = _JSON_OBJECT_RE.search(text)
    if embedded is not None:
        candidates.append(embedded.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


@lru_cache(maxsize=None)
def facts_instruction() -> str:
    """Инструкция извлекателя из ассета assets/facts_prompt.md."""
    return (config.ASSETS_DIR / FACTS_PROMPT_ASSET).read_text(encoding="utf-8").strip()


def facts_user_text(facts: Dict[str, str], pending_messages: Sequence[str]) -> str:
    """User-сообщение извлекателя: текущий блок (или явное отсутствие) + новые сообщения."""
    return FACTS_USER_TEMPLATE.format(
        facts=render_facts(facts) or _PLACEHOLDER_NO_FACTS,
        messages="\n\n".join(pending_messages),
    )


def build_facts_messages(
    facts: Dict[str, str], pending_messages: Sequence[str]
) -> List[Dict[str, str]]:
    """Запрос извлекателя: system — инструкция, user — блок фактов и новые сообщения."""
    return [
        {"role": "system", "content": facts_instruction()},
        {"role": "user", "content": facts_user_text(facts, pending_messages)},
    ]


def estimate_facts_call_tokens(
    facts: Dict[str, str], pending_messages: Sequence[str]
) -> int:
    """Приближённая оценка промпта извлекателя (для бюджета пакета и отчёта)."""
    return estimate_tokens(facts_instruction()) + estimate_tokens(
        facts_user_text(facts, pending_messages)
    )


def facts_batch(
    facts: Dict[str, str], pending_messages: Sequence[str], budget_tokens: int
) -> List[str]:
    """Первые сообщения очереди, чья оценка запроса извлекателя не превышает бюджет.

    Извлекатель видит весь накопленный диалог, поэтому первую активацию стратегии на длинном
    диалоге обрабатывают пакетами: сообщения берутся с начала очереди, пока запрос помещается
    в бюджет. Первое сообщение принимается всегда — иначе очередь не сдвинулась бы с места.
    """
    selected: List[str] = []
    for message in pending_messages:
        if selected and estimate_facts_call_tokens(facts, selected + [message]) > budget_tokens:
            break
        selected.append(message)
    return selected


def facts_message(facts: Dict[str, str]) -> str:
    """Текст системного сообщения с блоком фактов."""
    return FACTS_MESSAGE_TEMPLATE.format(facts=render_facts(facts))


# --- ветки ----------------------------------------------------------------------------------


@dataclass
class Branch:
    """Ветка диалога: имя и индексы ходов в общем логе сессии."""

    name: str
    indices: List[int] = field(default_factory=list)

    @property
    def exchanges(self) -> int:
        return len(self.indices) // 2


class BranchTree:
    """Ветки диалога поверх одного лога ходов.

    Лог — стек сессии агента: он только растёт, а ветка хранит индексы своих ходов, поэтому
    ветки не дублируют сообщения и не расходятся между собой. Чекпоинт — позиция в активной
    ветке, от которой создаётся новая ветка; без чекпоинта новая ветка пуста.
    """

    def __init__(self, log: List[Dict[str, str]]) -> None:
        self._log = log
        self._branches: List[Branch] = [
            Branch(DEFAULT_BRANCH_NAME, list(range(len(log))))
        ]
        self._active = 0
        self._checkpoint = len(self._branches[0].indices)

    @property
    def active_name(self) -> str:
        return self._branches[self._active].name

    @property
    def active_exchanges(self) -> int:
        return self._branches[self._active].exchanges

    def branches(self) -> tuple:
        """Снимок веток для отчёта: (имя, число обменов) в порядке создания."""
        return tuple((branch.name, branch.exchanges) for branch in self._branches)

    def active_messages(self) -> List[Dict[str, str]]:
        """Ходы активной ветки в порядке следования."""
        return [self._log[index] for index in self._branches[self._active].indices]

    def add_exchange(self) -> None:
        """Учитывает обмен, только что дописанный в лог последней парой ходов."""
        start = len(self._log) - 2
        self._branches[self._active].indices.extend((start, start + 1))

    def checkpoint(self) -> None:
        """Отмечает текущую позицию активной ветки для будущей ветки."""
        self._checkpoint = len(self._branches[self._active].indices)

    def new_branch(self) -> str:
        """Создаёт ветку от чекпоинта и делает её активной; возвращает имя."""
        active = self._branches[self._active]
        name = f"ветка {len(self._branches) + 1}"
        self._branches.append(Branch(name, list(active.indices[: self._checkpoint])))
        self._active = len(self._branches) - 1
        return name

    def switch(self, name: str) -> bool:
        """Делает ветку с таким именем активной; False — ветки нет."""
        for index, branch in enumerate(self._branches):
            if branch.name == name:
                self._active = index
                self._checkpoint = len(branch.indices)
                return True
        return False

    def reset(self) -> None:
        """Возвращает дерево к одной пустой ветке (команда /clear)."""
        self._branches = [Branch(DEFAULT_BRANCH_NAME)]
        self._active = 0
        self._checkpoint = 0
