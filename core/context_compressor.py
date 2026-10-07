"""Чистая логика сжатия контекста: порог, разделение хода, сборка сообщений.

Без HTTP и без rich, по образцу core/usage: агент оркестрирует, модуль решает
чистую часть — какой пакет сворачивается и как собирается запрос.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import config
from .usage import estimate_tokens

# Хвост стека, который не сворачивается: последний обмен (пара user/assistant).
LAST_EXCHANGE_MESSAGES = 2

SUMMARY_PROMPT_ASSET = "summary_prompt.md"

SUMMARY_MESSAGE_TEMPLATE = (
    "Краткое резюме более ранних обменов текущего диалога "
    "(их дословный текст в этот запрос не входит):\n{summary}"
)

_PLACEHOLDER_NO_SUMMARY = "Пока ни один обмен не свёрнут в резюме."


def needs_compression(raw_messages: int, threshold: int) -> bool:
    """Порог сжатия в сообщениях: срабатывание на достижении."""
    return raw_messages >= threshold


def split_for_digest(
    turns: List[Dict[str, str]], keep_last: int = LAST_EXCHANGE_MESSAGES
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Разделяет ходы на пакет сворачиваемых и неотжатый хвост.

    Пакет кратен парам user/assistant; последний обмен всегда остаётся дословным,
    нечётный хвост не сворачивается (остаётся в стеке до образования пары).
    """
    droppable = max(0, len(turns) - keep_last)
    digest_count = droppable - droppable % 2
    return list(turns[:digest_count]), list(turns[digest_count:])


def context_messages(
    system: str, summary: Optional[str], turns: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Собирает запрос: system, затем при наличии одно сообщение-резюме, затем ходы."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if summary:
        messages.append({"role": "system", "content": summary_message(summary)})
    messages.extend(turns)
    return messages


def summary_message(summary: str) -> str:
    """Текст сообщения-резюме: обёртка вокруг дайджеста."""
    return SUMMARY_MESSAGE_TEMPLATE.format(summary=summary)


@lru_cache(maxsize=None)
def summary_instruction() -> str:
    """Инструкция суммаризатора из ассета assets/summary_prompt.md."""
    return (config.ASSETS_DIR / SUMMARY_PROMPT_ASSET).read_text(encoding="utf-8").strip()


def summary_user_text(
    existing_summary: Optional[str], digest_turns: List[Dict[str, str]]
) -> str:
    """User-сообщение суммаризатора: прежнее резюме (или явное отсутствие) + пакет дословно."""
    prefix = existing_summary if existing_summary else _PLACEHOLDER_NO_SUMMARY
    turns_text = "\n\n".join(
        f"{turn['role']}: {turn['content']}" for turn in digest_turns
    )
    return f"{prefix}\n\n---\n\n{turns_text}"


def build_summary_messages(
    existing_summary: Optional[str], digest_turns: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Запрос суммаризатора: system — инструкция, user — прежнее резюме и пакет ходов."""
    return [
        {"role": "system", "content": summary_instruction()},
        {"role": "user", "content": summary_user_text(existing_summary, digest_turns)},
    ]


def estimate_summary_call_tokens(
    existing_summary: Optional[str], batch_turns: List[Dict[str, str]]
) -> int:
    """Приближённая оценка промпта суммаризатора (для бюджета пакета)."""
    return estimate_tokens(summary_instruction()) + estimate_tokens(
        summary_user_text(existing_summary, batch_turns)
    )


def batch_within_budget(
    digest_turns: List[Dict[str, str]], existing_summary: Optional[str], budget_tokens: int
) -> List[Dict[str, str]]:
    """Первые ходы пакета (кратно парам), чья оценка суммаризатор-запроса не превышает бюджет.

    Ходы накапливаются, пока оценка (инструкция + прежнее резюме + накопленный пакет) не
    превысит бюджет; первый обмен принимается всегда, чтобы прогресс не застревал.
    """
    selected: List[Dict[str, str]] = []
    for index in range(0, len(digest_turns), 2):
        pair = digest_turns[index : index + 2]
        if selected and estimate_summary_call_tokens(existing_summary, selected + pair) > budget_tokens:
            break
        selected.extend(pair)
    return selected


def estimate_context_tokens(
    system: str,
    summary: Optional[str],
    turns: List[Dict[str, str]],
    user_prompt: str,
) -> int:
    """Приближённая оценка собираемого запроса (system + резюме + ходы + новый user-ход)."""
    total = estimate_tokens(system)
    if summary:
        total += estimate_tokens(summary_message(summary))
    total += sum(estimate_tokens(turn["content"]) for turn in turns)
    total += estimate_tokens(user_prompt)
    return total
