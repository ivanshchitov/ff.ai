"""Вторая ступень отбора: оценка найденных кандидатов моделью сессии.

Модуль чистый: он строит сообщения для оценщика, строго разбирает его ответ и отбирает кандидатов
по порогу. Ни HTTP, ни состояния сессии здесь нет — запрос делает агент, и расход с фазой остаются
там же, где у прочих вспомогательных обращений.

Строгость разбора — часть контракта: неполный ответ, чужой идентификатор, повтор или оценка вне
диапазона означают, что отбор не состоялся. Доставить в этом случае сырые кандидаты значило бы
вернуть ровно тот шум, ради борьбы с которым ступень и появилась.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from . import config

RERANK_ASSET = "docs_rerank_prompt.md"

MODE_ENHANCED = "enhanced"
MODE_BASELINE = "baseline"
MODES: Tuple[str, ...] = (MODE_ENHANCED, MODE_BASELINE)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class RerankError(Exception):
    """Ответ оценщика негоден: причина показывается пользователю, доставки не будет."""


@dataclass(frozen=True)
class CandidateRating:
    """Оценка одного кандидата: идентификатор, число 0..1 и короткая причина."""

    identifier: int
    score: float
    reason: str = ""


@dataclass(frozen=True)
class Reranked:
    """Результат отбора: прошедшие порог, отсеянные и все оценки."""

    kept: Tuple[Any, ...]
    dropped: Tuple[Any, ...]
    ratings: Tuple[CandidateRating, ...]
    threshold: float

    @property
    def empty(self) -> bool:
        return not self.kept


@lru_cache(maxsize=None)
def instruction() -> str:
    """Инструкция оценщику из ассета: одна на все вопросы, поэтому кэшируется."""
    path = config.ASSETS_DIR / RERANK_ASSET
    return path.read_text(encoding="utf-8").strip()


def candidate_payload(identifier: int, candidate: Any) -> Mapping[str, Any]:
    """Кандидат в том виде, в каком его видит оценщик: без служебных полей и лишнего текста."""
    return {
        "id": identifier,
        "path": str(getattr(candidate, "path", "")),
        "title": str(getattr(candidate, "title", "")),
        "section": str(getattr(candidate, "section", "")),
        "snippet": str(getattr(candidate, "snippet", ""))[: config.DOCS_SNIPPET_CHARS],
    }


def build_messages(question: str, candidates: Sequence[Any]) -> List[dict]:
    """Сообщения вспомогательного запроса: инструкция в системе, данные — в user-сообщении."""
    payload = {
        "question": question,
        "candidates": [
            candidate_payload(index, candidate)
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    return [
        {"role": "system", "content": instruction()},
        {
            "role": "user",
            "content": "Оцени кандидатов и верни только JSON.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _loads_object(text: str) -> Any:
    """Первый JSON-объект из ответа: reasoning-модели часто оборачивают его в прозу или фенс."""
    for candidate in _FENCED_JSON.findall(text) + [text]:
        match = _JSON_OBJECT.search(candidate)
        if match is None:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def parse_response(text: str, candidates: Sequence[Any]) -> Tuple[CandidateRating, ...]:
    """Строгий разбор оценок: все идентификаторы по одному разу и оценки в диапазоне 0..1."""
    data = _loads_object(text or "")
    if not isinstance(data, Mapping):
        raise RerankError("ответ оценщика не разобран как JSON-объект")
    raw = data.get("results")
    if not isinstance(raw, list):
        raise RerankError("в ответе оценщика нет списка «results»")

    expected = set(range(1, len(candidates) + 1))
    ratings: List[CandidateRating] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise RerankError("элемент списка оценок не объект")
        identifier = item.get("id")
        if not isinstance(identifier, int) or isinstance(identifier, bool):
            raise RerankError(f"идентификатор кандидата не целое число: {identifier!r}")
        if identifier not in expected:
            raise RerankError(f"оценщик вернул неизвестный идентификатор {identifier}")
        if identifier in seen:
            raise RerankError(f"идентификатор {identifier} повторяется")
        seen.add(identifier)
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RerankError(f"оценка кандидата {identifier} не число: {score!r}")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RerankError(f"оценка кандидата {identifier} вне диапазона 0..1: {value}")
        reason = item.get("reason", "")
        ratings.append(
            CandidateRating(
                identifier=identifier,
                score=value,
                reason=str(reason).strip() if isinstance(reason, str) else "",
            )
        )
    missing = sorted(expected - seen)
    if missing:
        raise RerankError(
            "оценщик не оценил кандидатов: " + ", ".join(str(item) for item in missing)
        )
    ratings.sort(key=lambda rating: rating.identifier)
    return tuple(ratings)


def select(
    candidates: Sequence[Any],
    ratings: Sequence[CandidateRating],
    threshold: float,
) -> Reranked:
    """Отбор по порогу: порядок кандидатов сохраняется, граница порога включается."""
    by_id = {rating.identifier: rating for rating in ratings}
    kept: List[Any] = []
    dropped: List[Any] = []
    for index, candidate in enumerate(candidates, start=1):
        rating = by_id.get(index)
        (kept if rating is not None and rating.score >= threshold else dropped).append(candidate)
    return Reranked(
        kept=tuple(kept), dropped=tuple(dropped), ratings=tuple(ratings), threshold=threshold
    )


def mark(candidates: Sequence[Any], ratings: Sequence[CandidateRating]) -> Tuple[Any, ...]:
    """Копии кандидатов с проставленными оценками и причинами — для снимка и отчёта.

    Кандидаты в ретривере неизменяемы, поэтому снимок получает новые объекты: иначе оценка
    протекла бы в данные поиска, и отчёт показывал бы не то, что было на самом деле.
    """
    by_id = {rating.identifier: rating for rating in ratings}
    marked: List[Any] = []
    for index, candidate in enumerate(candidates, start=1):
        rating = by_id.get(index)
        if rating is None:
            marked.append(candidate)
            continue
        replaced = _with_score(candidate, rating)
        marked.append(replaced if replaced is not None else candidate)
    return tuple(marked)


def _with_score(candidate: Any, rating: CandidateRating) -> Any:
    """Кандидат с оценкой: работает и с dataclass ретривера, и с заглушкой в тестах."""
    try:
        return type(candidate)(**{**candidate.__dict__, "score": rating.score, "reason": rating.reason})
    except (AttributeError, TypeError):
        return None


def rating_lines(ratings: Iterable[CandidateRating], paths: Mapping[int, str] = None) -> Tuple[str, ...]:
    """Строки отчёта: оценка, путь и причина — по каждому кандидату."""
    paths = paths or {}
    lines: List[str] = []
    for rating in ratings:
        path = paths.get(rating.identifier, "")
        where = f"{path} — " if path else ""
        reason = f": {rating.reason}" if rating.reason else ""
        lines.append(f"    {rating.score:.2f} — {where}{reason.lstrip(': ') or 'без пояснения'}")
    return tuple(lines)
