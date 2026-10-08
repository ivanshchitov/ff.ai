"""Локальные embeddings: пресеты, HTTP-контракт, нормализация, размерность и ошибки.

Модель векторов берётся на том же локальном сервере, что и чат, поэтому контракт проверяется
против поддельного HTTP (`responses`): без сети и без загруженных весов.
"""

from __future__ import annotations

import json

import pytest
import requests
import responses

from core import config
from core.embeddings import (
    EmbeddingsError,
    LocalEmbeddings,
    code_provider,
    cosine,
    embeddings_url,
    for_model,
    local_provider,
)

EMBEDDINGS_URL = "http://localhost:9999/v1/embeddings"


def test_repository_presets_keep_chat_and_embedding_apart():
    """Файл пресетов репозитория: embedding-модель ровно одна и в списке чата её нет."""
    chat = config.local_models()
    embedding = config.local_embedding_models()

    assert chat, "в llama_server/models.ini нет ни одного чат-пресета"
    assert len(embedding) == 1, "для локального поиска нужен ровно один embedding-пресет"
    assert not set(chat) & set(embedding)
    assert set(config.LOCAL_MODELS) == set(chat)


def test_embedding_presets_are_excluded_from_chat(tmp_path):
    """Разбор файла пресетов: признак `embedding` делит пресеты, отсутствие файла — пусто."""
    presets = tmp_path / "models.ini"
    presets.write_text("[chat]\ntemp = 0.7\n\n[embed]\nembedding = true\n", encoding="utf-8")

    assert config.local_models(presets) == ["chat"]
    assert config.local_embedding_models(presets) == ["embed"]
    assert config.local_embedding_models(tmp_path / "missing.ini") == []


@pytest.mark.parametrize("names", [[], ["one", "two"]])
def test_local_embedding_configuration_must_be_unambiguous(monkeypatch, names):
    """Ноль или два embedding-пресета — ошибка с причиной, а не выбор за пользователя."""
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODELS", names)

    assert for_model(config.DEFAULT_MODEL) is None  # облачной модели векторы не нужны
    with pytest.raises(EmbeddingsError, match="embedding"):
        for_model(config.LOCAL_MODELS[0])


def test_provider_targets_the_local_chat_server(monkeypatch):
    """Адрес векторов выводится из адреса локального чата: тот же сервер, без второй настройки."""
    monkeypatch.setattr(
        config, "LOCAL_API_URL", "http://localhost:1234/v1/chat/completions"
    )
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODELS", ["embed"])

    provider = for_model(config.LOCAL_MODELS[0])

    assert provider is not None
    assert provider.model == "embed"
    assert provider.url == "http://localhost:1234/v1/embeddings"
    assert embeddings_url() == "http://localhost:1234/v1/embeddings"


def test_not_a_chat_completions_url_is_reported(monkeypatch):
    """Адрес без пути chat-completions не даёт догадаться, где embeddings: лучше ошибка."""
    monkeypatch.setattr(config, "LOCAL_API_URL", "http://localhost:1234/v1")

    with pytest.raises(EmbeddingsError, match="FFAI_LOCAL_API_URL"):
        embeddings_url()


@responses.activate
def test_http_contract_normalization_and_query_instruction():
    """Запрос — model + input, ответ нормализуется, у поискового запроса своя инструкция."""
    responses.post(
        EMBEDDINGS_URL, json={"data": [{"index": 0, "embedding": [3, 4]}], "model": "embed"}
    )
    provider = LocalEmbeddings("embed", EMBEDDINGS_URL)

    assert provider.embed("class ModelList : public QObject") == pytest.approx([0.6, 0.8])
    assert provider.embed_query("Где читается список моделей?") == pytest.approx([0.6, 0.8])
    assert provider.dimensions == 2

    first, second = responses.calls
    assert json.loads(first.request.body) == {
        "model": "embed",
        "input": "class ModelList : public QObject",
    }
    query_body = json.loads(second.request.body)["input"]
    assert "Instruct:" in query_body
    assert "Где читается список моделей?" in query_body
    # Локальному серверу облачный заголовок не отправляется: ключа у него нет.
    assert "Authorization" not in first.request.headers
    assert first.request.req_kwargs["timeout"] > 0


@responses.activate
def test_changed_dimensions_are_reported():
    """Смена весов под тем же пресетом видна на входе, а не после несопоставимых оценок."""
    responses.post(EMBEDDINGS_URL, json={"data": [{"index": 0, "embedding": [1, 0]}]})
    responses.post(EMBEDDINGS_URL, json={"data": [{"index": 0, "embedding": [1, 0, 0]}]})
    provider = LocalEmbeddings("embed", EMBEDDINGS_URL)

    assert provider.embed("первый фрагмент") == pytest.approx([1.0, 0.0])
    with pytest.raises(EmbeddingsError, match="размерности"):
        provider.embed("второй фрагмент")


@responses.activate
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": []},
        {"data": [{"index": 1, "embedding": [1]}]},
        {"data": [{"index": 0}]},
        {"data": [{"index": 0, "embedding": []}]},
        {"data": [{"index": 0, "embedding": [0, 0]}]},
        {"data": [{"index": 0, "embedding": [True, 2]}]},
        {"data": [{"index": 0, "embedding": ["1", 2]}]},
        {"data": [{"index": 0, "embedding": [float("nan"), 2]}]},
    ],
)
def test_invalid_embeddings_are_user_facing_errors(payload):
    """Неполный, чужой, пустой, нулевой или нечисловой вектор — ошибка с типом, не мусор."""
    responses.post(EMBEDDINGS_URL, json=payload)

    with pytest.raises(EmbeddingsError):
        LocalEmbeddings("embed", EMBEDDINGS_URL).embed("текст")


@responses.activate
@pytest.mark.parametrize(
    "failure", [requests.Timeout("timeout"), requests.ConnectionError("offline")]
)
def test_transport_failure_is_user_facing(failure):
    """Сервер не отвечает — это состояние локального режима, а не повод подменить векторы."""
    responses.post(EMBEDDINGS_URL, body=failure)

    with pytest.raises(EmbeddingsError, match="embedding"):
        LocalEmbeddings("embed", EMBEDDINGS_URL).embed("текст")


def test_cosine_refuses_mismatched_dimensions():
    """Скалярное произведение векторов разной длины — молчаливый мусор, поэтому ошибка."""
    assert cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(5.0)
    with pytest.raises(EmbeddingsError, match="размерности"):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


# --- выбор векторной модели -----------------------------------------------------------------


def test_default_provider_is_the_only_embedding_preset(monkeypatch):
    """Векторная модель не зависит от чат-модели: это embedding-пресет и его адрес."""
    monkeypatch.setattr(config, "LOCAL_API_URL", "http://localhost:1234/v1/chat/completions")
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODELS", ["embed"])

    choice = local_provider()

    assert choice.available is True
    assert choice.model == "embed"
    assert choice.provider.url == "http://localhost:1234/v1/embeddings"
    assert choice.reason == ""


@pytest.mark.parametrize("names", [[], ["one", "two"]])
def test_ambiguous_presets_are_a_named_reason(monkeypatch, names):
    """Ноль и два пресета — названная причина, а не исключение: поиску есть чем ответить."""
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODELS", names)

    choice = local_provider()

    assert choice.available is False
    assert "embedding = true" in choice.reason or "embedding-пресетов" in choice.reason


def test_code_provider_needs_a_managed_local_server(monkeypatch):
    """Автозапуск выключен — приложение сервером не управляет, и векторы не считаются."""
    monkeypatch.setenv("FFAI_LLAMA_AUTOSTART", "0")
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODELS", ["embed"])

    choice = code_provider()

    assert choice.available is False
    assert "FFAI_LLAMA_AUTOSTART" in choice.reason


def test_code_provider_uses_the_preset_when_the_server_is_managed(monkeypatch):
    """Автозапуск включён — сервер поднимает точка входа, и пресет становится моделью векторов."""
    monkeypatch.setenv("FFAI_LLAMA_AUTOSTART", "1")
    monkeypatch.setattr(config, "LOCAL_API_URL", "http://localhost:1234/v1/chat/completions")
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODELS", ["embed"])

    choice = code_provider()

    assert choice.available is True
    assert choice.model == "embed"
