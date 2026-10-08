"""Поиск по корпусу кода: векторы фрагментов, две ступени отбора, снимок поиска.

Первый этап — близость вопроса и фрагмента: по векторам, когда они есть в индексе для текущей
векторной модели (тогда вопрос на русском находит английский код без переформулировки), иначе по
совпадению токенов. Токенный путь остаётся явным режимом: он точнее на идентификаторах
(`Q_PROPERTY`, `%files`), и выбирает способ пользователь, а не приложение за него. Способ, которым
поиск шёл, и причина, по которой он пошёл другим, всегда в снимке: безвекторный результат не
выдаётся за векторный.

Вторая ступень повторяет отбор документации: `baseline` отдаёт первые результаты как есть, `enhanced`
переформулирует вопрос, оценивает кандидатов моделью сессии и применяет порог. Различается только
инструкция из ассетов — механизм оценки общий (`core/reranking.py`). Переформулировка нужна там, где
векторов нет: она была лекарством от несовпадения языков, а векторный поиск лечит причину.

Приложение не выбрасывает сырые кандидаты: если оценка не удалась или все оценки ниже порога, это
видимое состояние снимка, а не «сойдёт и так». Ответ по памяти в такой ситуации — ровно то, от чего
корпус и защищает.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import code_index, config, embeddings, reranking

MODE_BASELINE = reranking.MODE_BASELINE
MODE_ENHANCED = reranking.MODE_ENHANCED
MODES = (MODE_BASELINE, MODE_ENHANCED)

STATUS_OK = "ok"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_NO_MATCHES = "no_matches"
STATUS_RERANK_FAILED = "rerank_failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DISABLED = "disabled"

CODE_QUERY_ASSET = "code_query_prompt.md"
CODE_RERANK_ASSET = "code_rerank_prompt.md"

# Способы ранжирования первого этапа: явный токенный режим и векторная близость.
RANKING_TOKENS = "tokens"
RANKING_VECTORS = "vectors"
RANKINGS = (RANKING_TOKENS, RANKING_VECTORS)

# Токены кода: слова, идентификаторы и числа. Идентификаторы дополнительно режутся по регистру и
# подчёркиваниям — «ModelListLoader» должно находиться и по «model», и по «loader».
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-ZА-ЯЁ]+(?![a-zа-яё])|[A-ZА-ЯЁ][a-zа-яё0-9]+")
MIN_TOKEN_CHARS = 3


@dataclass(frozen=True)
class CodeCandidate:
    """Кандидат поиска: фрагмент индекса со своей близостью к запросу.

    Поля `title`/`section` пусты, но объявлены: вторая ступень работает с кандидатами обоих
    корпусов одинаково и читает их через атрибуты.
    """

    identifier: str
    path: str
    snippet: str
    text: str
    start_line: int
    end_line: int
    chunk_id: str
    score: float = 0.0
    reason: str = ""
    title: str = ""
    section: str = ""


@dataclass(frozen=True)
class CodeFragment:
    """Фрагмент, доставленный в запрос: идентификатор `путь:L10-L40` и дословный текст."""

    identifier: str
    source: str
    text: str
    start_line: int
    end_line: int
    truncated: bool = False


@dataclass(frozen=True)
class CodeReport:
    """Снимок поиска: что искали, как отбирали, что доставили и чем это кончилось."""

    question: str
    query: str
    mode: str
    threshold: float
    before: int
    after: int
    database: Path
    status: str
    candidates: Tuple[CodeCandidate, ...] = ()
    fragments: Tuple[CodeFragment, ...] = ()
    rated: bool = False
    chunks: int = 0
    error: str = ""
    # Способ ранжирования первого этапа и почему он такой: пусто — ранжирования не было.
    ranking: str = ""
    ranking_note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def selected(self) -> Tuple[CodeCandidate, ...]:
        """Кандидаты, попавшие в доставку, — по идентификатору доставленных фрагментов."""
        delivered = {fragment.identifier for fragment in self.fragments}
        return tuple(
            candidate for candidate in self.candidates if candidate.identifier in delivered
        )

    def dropped(self) -> Tuple[CodeCandidate, ...]:
        """Кандидаты, отсеянные порогом: в `baseline` порога нет, поэтому список пуст."""
        delivered = {fragment.identifier for fragment in self.fragments}
        return tuple(
            candidate for candidate in self.candidates if candidate.identifier not in delivered
        )


def _reason(error: BaseException) -> str:
    """Читаемая причина сбоя вспомогательного запроса: SDK заворачивает их в группы."""
    current: BaseException = error
    while isinstance(current, BaseExceptionGroup) and current.exceptions:
        current = current.exceptions[0]
    return str(current) or type(current).__name__


# --- векторы --------------------------------------------------------------------------------


def tokenize(text: str) -> Tuple[str, ...]:
    """Токены текста или кода: слова, части идентификаторов по регистру и подчёркиваниям."""
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        word = raw.casefold()
        parts = [word]
        if any(character.isupper() for character in raw[1:]):
            parts.extend(part.casefold() for part in _CAMEL_RE.findall(raw) if part)
        if "_" in raw:
            parts.extend(part.casefold() for part in raw.split("_") if part)
        for part in parts:
            if len(part) >= MIN_TOKEN_CHARS and part not in tokens:
                tokens.append(part)
    return tuple(tokens)


def _vector(tokens: Sequence[str], weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Разреженный вектор текста: частота токена, умноженная на вес (для запроса вес — idf)."""
    counts = Counter(tokens)
    return {
        token: count * (1.0 if weights is None else weights.get(token, 0.0))
        for token, count in counts.items()
    }


def cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    """Косинусная близость разреженных векторов; нулевые векторы близки нулю, а не друг другу."""
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    if not common:
        return 0.0
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _idf(documents: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Обратная частота: редкий в корпусе токен весит больше, поэтому «int» не забивает «ModelList»."""
    total = len(documents)
    seen: Counter = Counter()
    for tokens in documents:
        for token in set(tokens):
            seen[token] += 1
    return {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in seen.items()}


class CodeRetriever:
    """Поиск по индексу: читает фрагменты, считает близость и применяет выбранную ступень."""

    def __init__(
        self,
        database: Path,
        corpus=None,
        snippet_chars: int = None,
        embedding=None,
        ranking: str = None,
    ) -> None:
        """`embedding` — векторная модель; без неё берётся модель по умолчанию.

        По умолчанию спрашивается `embeddings.code_provider`: тот же выбор, что и при сборке
        индекса, — иначе поиск считал бы вектор вопроса моделью, которой нет в индексе.

        `ranking` — способ ранжирования: `tokens`, `vectors` или None (векторы, когда они есть
        в индексе, иначе токены с названной причиной). Явный режим нужен и пользователю (на
        идентификаторах токены точнее), и проверке: смешение способов не должно быть незаметным.
        """
        self.database = Path(database)
        self.corpus = corpus
        self._snippet_chars = snippet_chars
        self._embedding = embedding
        self._choice: Optional[embeddings.EmbeddingChoice] = None
        self.ranking = ranking

    @property
    def embedding_choice(self) -> embeddings.EmbeddingChoice:
        """Векторная модель и причина, если её нет: спрашивается один раз и запоминается."""
        if self._choice is None:
            self._choice = (
                embeddings.EmbeddingChoice(provider=self._embedding)
                if self._embedding is not None
                else embeddings.code_provider()
            )
        return self._choice

    @property
    def snippet_chars(self) -> int:
        return config.CODE_SNIPPET_CHARS if self._snippet_chars is None else self._snippet_chars

    def search(
        self,
        question: str,
        mode: str = None,
        rewrite: Optional[Callable[[str], str]] = None,
        rate: Optional[Callable[[str, Sequence[CodeCandidate]], str]] = None,
        threshold: float = None,
        before: int = None,
        after: int = None,
        ranking: str = None,
    ) -> CodeReport:
        """Ищет фрагменты по вопросу и возвращает снимок поиска.

        Вспомогательные обращения выполняет вызывающий: ретривер получает готовые функции
        `rewrite` (поисковый запрос) и `rate` (оценка кандидатов), поэтому локальный поиск
        остаётся проверяемым без сети и без модели. Переформулировка не спрашивается, когда
        первый этап идёт по векторам: вопрос на русском и английский код сближает сама модель,
        и отдельный запрос за неё платить нечем.
        """
        mode = mode or config.CODE_RETRIEVAL_MODE
        ranking = self.ranking if ranking is None else ranking
        threshold = config.CODE_RELEVANCE_THRESHOLD if threshold is None else threshold
        before = config.CODE_CANDIDATES_BEFORE if before is None else before
        after = config.CODE_FRAGMENTS_AFTER if after is None else after
        settings = dict(
            question=question,
            mode=mode,
            threshold=threshold,
            before=before,
            after=after,
            database=self.database,
        )
        if mode not in MODES:
            return CodeReport(
                query=question,
                status=STATUS_DISABLED,
                error=f"неизвестный режим: {mode}",
                **settings,
            )
        if ranking is not None and ranking not in RANKINGS:
            return CodeReport(
                query=question,
                status=STATUS_DISABLED,
                error=f"неизвестный способ ранжирования: {ranking}",
                **settings,
            )

        chunks = code_index.read_chunks(self.database)
        if not chunks:
            return CodeReport(
                query=question,
                status=STATUS_UNAVAILABLE,
                error="индекс корпуса кода не читается или пуст",
                **settings,
            )

        # Векторы спрашиваются у индекса один раз: и решение о переформулировке, и ранжирование
        # идут по одному и тому же набору, иначе причина отказа разошлась бы с тем, чем искали.
        want_vectors = ranking != RANKING_TOKENS
        vector_set = code_index.read_vectors(self.database) if want_vectors else code_index.VectorSet()
        choice = self.embedding_choice
        vectors_ready, vectors_note = (
            self._vector_readiness(chunks, vector_set, choice) if want_vectors else (False, "")
        )
        if want_vectors and not vectors_ready and ranking == RANKING_VECTORS:
            # Строгий режим: молчаливая подмена векторов токенами — ровно то, чего просили не делать.
            return CodeReport(
                query=question,
                status=STATUS_UNAVAILABLE,
                error=vectors_note,
                chunks=len(chunks),
                ranking=RANKING_VECTORS,
                ranking_note=vectors_note,
                **settings,
            )

        query = question
        if mode == MODE_ENHANCED and not vectors_ready:
            if rewrite is None:
                return CodeReport(
                    query=query,
                    status=STATUS_RERANK_FAILED,
                    error="режим enhanced без переформулировки вопроса",
                    chunks=len(chunks),
                    **settings,
                )
            try:
                rewritten = (rewrite(question) or "").strip()
            except Exception as error:  # noqa: BLE001 - сбой вспомогательного запроса не отменяет вопрос
                return CodeReport(
                    query=query,
                    status=STATUS_RERANK_FAILED,
                    error=_reason(error),
                    chunks=len(chunks),
                    **settings,
                )
            if not rewritten:
                return CodeReport(
                    query=query,
                    status=STATUS_RERANK_FAILED,
                    error="переформулировка вернула пустой запрос",
                    chunks=len(chunks),
                    **settings,
                )
            query = rewritten

        candidates: Optional[Tuple[CodeCandidate, ...]] = None
        used_ranking = RANKING_TOKENS
        ranking_note = vectors_note or "выбран явный токенный режим"
        if vectors_ready:
            candidates, vectors_note = self._vector_candidates(
                query, chunks, before, vector_set, choice
            )
            if candidates is not None:
                used_ranking = RANKING_VECTORS
                ranking_note = f"векторная модель {vector_set.model}"
        if candidates is None:
            candidates = self._token_candidates(query, chunks, before)
            ranking_note = vectors_note or ranking_note
        settings["ranking"] = used_ranking
        settings["ranking_note"] = ranking_note
        if not candidates:
            return CodeReport(
                query=query, status=STATUS_NO_CANDIDATES, chunks=len(chunks), **settings
            )

        rated = False
        if mode == MODE_ENHANCED:
            if rate is None:
                return CodeReport(
                    query=query,
                    status=STATUS_RERANK_FAILED,
                    error="режим enhanced без оценки кандидатов",
                    candidates=candidates,
                    chunks=len(chunks),
                    **settings,
                )
            try:
                ratings = reranking.parse_response(rate(question, candidates), candidates)
            except Exception as error:  # noqa: BLE001 - негодная оценка это состояние поиска
                return CodeReport(
                    query=query,
                    status=STATUS_RERANK_FAILED,
                    error=_reason(error),
                    candidates=candidates,
                    chunks=len(chunks),
                    **settings,
                )
            candidates = tuple(reranking.mark(candidates, ratings))
            selected = reranking.select(candidates, ratings, threshold).kept
            rated = True
            if not selected:
                return CodeReport(
                    query=query,
                    status=STATUS_NO_MATCHES,
                    candidates=candidates,
                    rated=True,
                    chunks=len(chunks),
                    **settings,
                )
        else:
            selected = candidates[:after]

        fragments = tuple(self._fragment(candidate) for candidate in selected[:after])
        return CodeReport(
            query=query,
            status=STATUS_OK,
            candidates=candidates,
            fragments=fragments,
            rated=rated,
            chunks=len(chunks),
            **settings,
        )

    def _vector_readiness(
        self,
        chunks: Sequence[code_index.CodeChunk],
        vector_set: code_index.VectorSet,
        choice: embeddings.EmbeddingChoice,
    ) -> Tuple[bool, str]:
        """Можно ли ранжировать векторами: (готовность, причина отказа).

        Отказ — не исключение, а состояние с причиной: чужое пространство и неполный набор
        векторов не дают оценки, которую можно было бы назвать близостью.
        """
        if not vector_set.model:
            return False, vector_set.note or "в индексе нет векторов"
        if not choice.available:
            return False, (
                f"векторы модели {vector_set.model} в индексе есть, но посчитать вектор вопроса "
                f"нечем: {choice.reason}"
            )
        if choice.model != vector_set.model:
            return False, (
                f"в индексе векторы модели {vector_set.model}, доступная модель векторов — "
                f"{choice.model}: чужое пространство не используется"
            )
        missing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in vector_set.vectors]
        if missing:
            return False, (
                f"векторы есть не у всех фрагментов: нет у {len(missing)} из {len(chunks)} — "
                "пересоберите индекс"
            )
        return True, ""

    def _vector_candidates(
        self,
        query: str,
        chunks: Sequence[code_index.CodeChunk],
        before: int,
        vector_set: code_index.VectorSet,
        choice: embeddings.EmbeddingChoice,
    ) -> Tuple[Optional[Tuple[CodeCandidate, ...]], str]:
        """Кандидаты по близости векторов: (None, причина), если векторами ранжировать нельзя.

        Сбой модели на векторе вопроса — та же названная причина, а не повод выдать токенный
        результат за векторный: способ ранжирования обязан быть виден в снимке.
        """
        try:
            query_vector = choice.provider.embed_query(query)
            scored = [
                (
                    embeddings.cosine(query_vector, vector_set.vectors[chunk.chunk_id]),
                    chunk.chunk_id,
                    chunk,
                )
                for chunk in chunks
            ]
        except embeddings.EmbeddingsError as error:
            return None, f"вектор вопроса не посчитан: {error}"
        return self._candidates_from(scored, before), ""

    def _token_candidates(
        self, query: str, chunks: Sequence[code_index.CodeChunk], before: int
    ) -> Tuple[CodeCandidate, ...]:
        """Кандидаты по совпадению токенов: idf по корпусу, косинус по фрагментам."""
        documents = [tokenize(chunk.text) for chunk in chunks]
        weights = _idf(documents)
        query_vector = _vector(tokenize(query), weights)
        scored = [
            (cosine(query_vector, _vector(tokens)), chunk.chunk_id, chunk)
            for chunk, tokens in zip(chunks, documents)
        ]
        return self._candidates_from(scored, before)

    def _candidates_from(
        self, scored: Sequence[Tuple[float, str, code_index.CodeChunk]], before: int
    ) -> Tuple[CodeCandidate, ...]:
        """Кандидаты по близости: нулевая близость означает «ничего общего», а порядок устойчив.

        Правило одно для обоих способов: фрагмент без общей близости с запросом — не кандидат,
        и платить модели за его оценку не за что.
        """
        ranked = [item for item in scored if item[0] > 0.0]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(self._candidate(chunk, score) for score, _, chunk in ranked[: max(1, before)])

    def _candidate(self, chunk: code_index.CodeChunk, score: float) -> CodeCandidate:
        label = f"{chunk.path}:L{chunk.start_line}-L{chunk.end_line}"
        return CodeCandidate(
            identifier=label,
            path=label,
            snippet=chunk.text[: self.snippet_chars],
            text=chunk.text,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_id=chunk.chunk_id,
            score=score,
        )

    def _fragment(self, candidate: CodeCandidate) -> CodeFragment:
        """Доставка: текст фрагмента целиком, но не длиннее предела корпуса кода."""
        limit = config.CODE_FRAGMENT_MAX_CHARS
        text = candidate.text
        truncated = len(text) > limit
        return CodeFragment(
            identifier=candidate.identifier,
            source=candidate.path,
            text=text[:limit] if truncated else text,
            start_line=candidate.start_line,
            end_line=candidate.end_line,
            truncated=truncated,
        )


def query_instruction() -> str:
    """Инструкция переформулировки вопроса из ассета."""
    return (config.ASSETS_DIR / CODE_QUERY_ASSET).read_text(encoding="utf-8").strip()


def rerank_instruction() -> str:
    """Инструкция оценки кандидатов кода: тот же механизм, что у документации."""
    return reranking.instruction(CODE_RERANK_ASSET)


def query_messages(question: str) -> List[dict]:
    """Вспомогательный запрос переформулировки: инструкция и вопрос пользователя."""
    return [
        {"role": "system", "content": query_instruction()},
        {"role": "user", "content": question},
    ]


def rerank_messages(question: str, candidates: Sequence[CodeCandidate]) -> List[dict]:
    """Вспомогательный запрос оценки кандидатов кода."""
    return reranking.build_messages(
        question, candidates, asset=CODE_RERANK_ASSET, snippet_chars=config.CODE_SNIPPET_CHARS
    )


def rate_lines(candidates: Sequence[CodeCandidate]) -> Tuple[str, ...]:
    """Строки отчёта об оценках: кандидат, оценка и причина."""
    lines: List[str] = []
    for candidate in candidates:
        lines.append(
            f"    {candidate.score:.2f} — {candidate.path}"
            + (f": {candidate.reason}" if candidate.reason else "")
        )
    return tuple(lines)


def dropped_by_threshold(candidates: Sequence[CodeCandidate], threshold: float) -> Tuple[CodeCandidate, ...]:
    """Кандидаты ниже порога: нужны для строки «отсеяно», даже если оценка состоялась."""
    return tuple(candidate for candidate in candidates if candidate.score < threshold)
