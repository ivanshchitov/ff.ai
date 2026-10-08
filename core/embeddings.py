"""Локальные embeddings через llama-server: векторы для индекса и поиска по коду.

Смысл модуля — режим без облачного ключа: и индекс, и поисковый запрос считает модель на том
же локальном сервере, что отвечает на вопросы. Векторы хеш-функции (`core/code_retrieval.py`)
остаются там, где они уже есть: этот модуль не подменяет их молча, а даёт вызывающему вторую
ветку и **отдельный тип ошибки** — сбой embedding-модели обязан быть видимым состоянием, а не
тихим откатом к мусорному поиску.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import requests

from . import config, llama_server

# Модель считает вектор фрагмента целиком: бюджет щедрый, потому что первый запрос ещё и
# загружает веса в память сервера.
EMBED_TIMEOUT = 120.0
CHAT_COMPLETIONS_SUFFIX = "/chat/completions"

# Qwen3-Embedding обучен на инструкции только для поискового запроса: без неё близость
# «вопрос ↔ код» считается по другому распределению, чем близость «код ↔ код». Формулировка
# нейтральна к предметной области: ядро не знает, по какому репозиторию идёт поиск.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a source code repository, "
    "retrieve relevant code fragments.\nQuery: "
)


class EmbeddingsError(Exception):
    """Ошибка локальной embedding-модели: транспорт, формат ответа или размерность."""


class LocalEmbeddings:
    """Одна embedding-модель llama-server: нормализованные векторы и их размерность.

    Размерность запоминается на первом векторе: сравнивать её не с чем, кроме самой себя, но
    расхождение между фрагментами одного индекса означает, что модель сменилась под теми же
    весами, — и поискать это стоит на входе, а не после выдачи несопоставимых оценок.
    """

    def __init__(self, model: str, url: str, timeout: Optional[float] = None) -> None:
        self.model = model
        self.url = url
        self.timeout = EMBED_TIMEOUT if timeout is None else timeout
        self._dimensions: Optional[int] = None

    @property
    def dimensions(self) -> Optional[int]:
        """Размерность первого полученного вектора; до первого обращения — None."""
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        """Вектор текста, приведённый к единичной длине."""
        vector = self._request_vector(text)
        norm = math.hypot(*vector)
        if not norm or not math.isfinite(norm):
            raise EmbeddingsError(
                f"Локальная embedding-модель {self.model}: нулевая или некорректная длина вектора."
            )
        if self._dimensions is None:
            self._dimensions = len(vector)
        elif len(vector) != self._dimensions:
            raise EmbeddingsError(
                f"Локальная embedding-модель {self.model}: несовместимые размерности "
                f"векторов ({self._dimensions} и {len(vector)}). Пересоберите индекс: "
                "веса пресета сменились."
            )
        return [value / norm for value in vector]

    def embed_query(self, query: str) -> List[float]:
        """Вектор поискового запроса — с инструкцией, в отличие от векторов фрагментов."""
        return self.embed(QUERY_INSTRUCTION + query)

    def _request_vector(self, text: str) -> List[float]:
        try:
            response = requests.post(
                self.url, json={"model": self.model, "input": text}, timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
            if len(data) != 1 or data[0]["index"] != 0:
                raise ValueError("неверный состав элементов ответа")
            vector = data[0]["embedding"]
        except (requests.RequestException, ValueError, KeyError, TypeError, OverflowError) as error:
            raise EmbeddingsError(
                f"Локальная embedding-модель {self.model}: {error}"
            ) from error
        if not isinstance(vector, list) or not vector:
            raise EmbeddingsError(f"Локальная embedding-модель {self.model}: пустой вектор.")
        for value in vector:
            # bool — подкласс int, а строка и NaN до арифметики не доходят: близость по ним
            # была бы молчаливым мусором.
            if type(value) not in (int, float) or not math.isfinite(value):
                raise EmbeddingsError(
                    f"Локальная embedding-модель {self.model}: некорректный embedding-вектор."
                )
        return [float(value) for value in vector]


def embeddings_url(chat_url: Optional[str] = None) -> str:
    """Адрес `/v1/embeddings` того же сервера, что обслуживает локальные пресеты чата."""
    local_url = chat_url or config.LOCAL_API_URL
    if not local_url.endswith(CHAT_COMPLETIONS_SUFFIX):
        raise EmbeddingsError(
            "Локальный адрес не похож на chat-completions: ожидался путь, оканчивающийся на "
            f"{CHAT_COMPLETIONS_SUFFIX}, получено {local_url!r}. Проверьте FFAI_LOCAL_API_URL."
        )
    return local_url[: -len(CHAT_COMPLETIONS_SUFFIX)] + "/embeddings"


def for_model(chat_model: str, presets=None) -> Optional[LocalEmbeddings]:
    """Провайдер векторов для выбранной модели или None, если модель облачная.

    Для локального режима нужен ровно один embedding-пресет: два означали бы выбор за
    пользователя, ноль — что локальные векторы посчитать нечем. И то и другое — ошибка с
    причиной, а не тихое переключение на хеш-векторы.

    Набор пресетов берётся тем же снимком, что и список моделей (`config.LOCAL_MODELS`), иначе
    приложение показало бы одну модель, а векторы считало бы другой. Путь к файлу можно назвать
    явно — этим пользуются тесты.
    """
    if chat_model not in config.LOCAL_MODELS:
        return None
    embedding_models = (
        config.LOCAL_EMBEDDING_MODELS if presets is None else config.local_embedding_models(presets)
    )
    if len(embedding_models) != 1:
        raise EmbeddingsError(
            "Для локальных embeddings нужен ровно один пресет с embedding = true в "
            f"llama_server/models.ini; найдено {len(embedding_models)}."
        )
    return LocalEmbeddings(
        model=embedding_models[0],
        url=embeddings_url(config.api_url_for_model(chat_model)),
    )


@dataclass(frozen=True)
class EmbeddingChoice:
    """Провайдер векторов и названная причина, когда его нет.

    Пара «провайдер + причина» вместо голого `Optional`: вызывающий обязан показать причину.
    Токенный поиск без векторов законен, а молчаливая подмена — нет, поэтому отсутствие модели
    всегда приходит текстом, а не пустым значением.
    """

    provider: Optional[LocalEmbeddings] = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.provider is not None

    @property
    def model(self) -> str:
        return "" if self.provider is None else self.provider.model


def local_provider(presets=None) -> EmbeddingChoice:
    """Векторная модель по умолчанию: единственный embedding-пресет, независимо от чат-модели.

    Векторы считает embedding-модель, а не выбранная чат-модель: иначе облачный чат требовал бы
    облачных векторов, и режим без ключа терял бы поиск. Ноль пресетов и несколько — состояния
    с причиной: выбирать модель за пользователя нельзя.
    """
    embedding_models = (
        config.LOCAL_EMBEDDING_MODELS if presets is None else config.local_embedding_models(presets)
    )
    if not embedding_models:
        return EmbeddingChoice(
            reason="в llama_server/models.ini нет пресета с embedding = true"
        )
    if len(embedding_models) > 1:
        return EmbeddingChoice(
            reason=(
                "в llama_server/models.ini несколько embedding-пресетов "
                f"({', '.join(embedding_models)}): модель векторов не выбрана"
            )
        )
    return EmbeddingChoice(
        provider=LocalEmbeddings(
            model=embedding_models[0], url=embeddings_url(config.LOCAL_API_URL)
        )
    )


def code_provider(presets=None) -> EmbeddingChoice:
    """Векторная модель для корпуса кода: локальный сервер должен быть под управлением приложения.

    Локальный llama-server поднимает и убирает сама точка входа, и только при включённом
    автозапуске (`core/llama_server.py`). При выключенном сервера может не быть вовсе: тогда
    векторы не считаются, а причина возвращается текстом — иначе `/code index` падал бы там, где
    достаточно токенного индекса, а поиск молча выдавал бы токенный результат за векторный.
    """
    if not llama_server.is_autostart_enabled():
        return EmbeddingChoice(
            reason=(
                "автозапуск локального сервера выключен (FFAI_LLAMA_AUTOSTART): сервера может "
                "не быть, векторы не считаются"
            )
        )
    return local_provider(presets)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Косинусная близость векторов: размерности обязаны совпадать, иначе — ошибка.

    Векторы приходят нормализованными, поэтому здесь только скалярное произведение; проверка
    длины нужна, чтобы расхождение моделей не превратилось в оценку «примерно ноль».
    """
    if len(left) != len(right):
        raise EmbeddingsError(
            f"Несовместимые размерности векторов запроса и фрагментов: {len(left)} и {len(right)}."
        )
    return sum(a * b for a, b in zip(left, right))
