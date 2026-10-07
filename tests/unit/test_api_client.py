"""Клиент API: разбор ответов, ошибки, ретраи, состав запроса."""

import json

import pytest
import requests
import responses

from core import config
from core.api_client import (
    API_KEY_CHARSET_ERROR,
    APIClient,
    APIError,
    is_valid_api_key,
    is_valid_json_answer,
)


@pytest.fixture
def client() -> APIClient:
    return APIClient("sk-test")


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _completion_with_usage(content: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# --- успешный путь --------------------------------------------------------------------


@responses.activate
def test_returns_stripped_content(client):
    responses.add(responses.POST, config.API_URL, json=_completion("  Ответ модели  "), status=200)
    assert client.ask("system", "user") == "Ответ модели"


@responses.activate
def test_request_payload_carries_messages_and_settings(client):
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("СИСТЕМА", "ВОПРОС", max_tokens=777)

    payload = json.loads(responses.calls[0].request.body)
    assert payload["model"] == config.DEFAULT_MODEL
    assert payload["temperature"] == config.TEMPERATURE
    assert payload["max_tokens"] == 777
    assert payload["messages"] == [
        {"role": "system", "content": "СИСТЕМА"},
        {"role": "user", "content": "ВОПРОС"},
    ]


@responses.activate
def test_model_parameter_reaches_payload(client):
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("system", "user", model="kimi-k3")
    payload = json.loads(responses.calls[0].request.body)
    assert payload["model"] == "kimi-k3"


@responses.activate
def test_temperature_parameter_reaches_payload(client):
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("system", "user", temperature=1.2)
    payload = json.loads(responses.calls[0].request.body)
    assert payload["temperature"] == 1.2


@responses.activate
def test_temperature_defaults_to_config_value(client):
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("system", "user")
    payload = json.loads(responses.calls[0].request.body)
    assert payload["temperature"] == config.TEMPERATURE


@responses.activate
def test_payload_never_contains_stop_field(client):
    """Регресс: `stop` обрывает reasoning-модель на reasoning_content с пустым content."""
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("system", "user")
    assert "stop" not in json.loads(responses.calls[0].request.body)


@responses.activate
def test_authorization_header_carries_key(client):
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("system", "user")
    assert responses.calls[0].request.headers["Authorization"] == "Bearer sk-test"


@responses.activate
def test_default_max_tokens_matches_default_word_limit(client):
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    client.ask("system", "user")
    payload = json.loads(responses.calls[0].request.body)
    assert payload["max_tokens"] == config.max_tokens_for_words(config.DEFAULT_MAX_WORDS)


# --- метрики использования --------------------------------------------------------------


@responses.activate
def test_ask_with_usage_returns_answer_meta(client):
    responses.add(
        responses.POST,
        config.API_URL,
        json=_completion_with_usage("Ответ", prompt_tokens=100, completion_tokens=200),
        status=200,
    )
    meta = client.ask_with_usage("system", "user", model="deepseek-v4.1-flash")

    assert meta.content == "Ответ"
    assert meta.model == "deepseek-v4.1-flash"
    assert meta.elapsed_seconds > 0
    assert meta.prompt_tokens == 100
    assert meta.completion_tokens == 200
    assert meta.total_tokens == 300
    assert meta.cost_usd == pytest.approx((100 * 0.30 + 200 * 1.20) / 1_000_000)


@responses.activate
def test_ask_with_usage_unknown_model_has_no_cost(client):
    responses.add(
        responses.POST,
        config.API_URL,
        json=_completion_with_usage("Ответ", prompt_tokens=10, completion_tokens=20),
        status=200,
    )
    meta = client.ask_with_usage("system", "user", model="no-such-model")
    assert meta.cost_usd is None


# --- ошибки ---------------------------------------------------------------------------


@responses.activate
def test_unauthorized_reports_bad_key(client):
    responses.add(responses.POST, config.API_URL, json={"error": "nope"}, status=401)
    with pytest.raises(APIError, match="Неверный API-ключ"):
        client.ask("system", "user")


@responses.activate
def test_server_error_is_wrapped(client):
    responses.add(responses.POST, config.API_URL, json={"error": "boom"}, status=500)
    with pytest.raises(APIError, match="Ошибка API"):
        client.ask("system", "user")


@responses.activate
@pytest.mark.parametrize(
    "payload",
    [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]}],
)
def test_malformed_success_payload_is_wrapped(client, payload):
    responses.add(responses.POST, config.API_URL, json=payload, status=200)
    with pytest.raises(APIError, match="Некорректный ответ"):
        client.ask("system", "user")


@responses.activate
def test_non_json_body_is_wrapped(client):
    responses.add(responses.POST, config.API_URL, body="<html>502</html>", status=200)
    with pytest.raises(APIError, match="Некорректный ответ"):
        client.ask("system", "user")


@responses.activate
def test_connection_error_is_not_retried(client, no_sleep):
    """Проблема с сетью считается устойчивой — повтор бессмысленен, в отличие от таймаута."""
    responses.add(responses.POST, config.API_URL, body=requests.exceptions.ConnectionError())
    with pytest.raises(APIError, match="Ошибка соединения"):
        client.ask("system", "user")
    assert len(responses.calls) == 1
    assert no_sleep == []


# --- ретраи по таймауту ----------------------------------------------------------------


@responses.activate
def test_timeout_is_retried_until_success(client, no_sleep):
    responses.add(responses.POST, config.API_URL, body=requests.exceptions.Timeout())
    responses.add(responses.POST, config.API_URL, body=requests.exceptions.Timeout())
    responses.add(responses.POST, config.API_URL, json=_completion("Наконец-то"), status=200)

    assert client.ask("system", "user") == "Наконец-то"
    assert len(responses.calls) == 3


@responses.activate
def test_timeout_gives_up_after_max_retries(client, no_sleep):
    for _ in range(config.MAX_RETRIES):
        responses.add(responses.POST, config.API_URL, body=requests.exceptions.Timeout())

    with pytest.raises(APIError, match="Превышено время ожидания"):
        client.ask("system", "user")
    assert len(responses.calls) == config.MAX_RETRIES


@responses.activate
def test_backoff_is_exponential(client, no_sleep):
    for _ in range(config.MAX_RETRIES):
        responses.add(responses.POST, config.API_URL, body=requests.exceptions.Timeout())

    with pytest.raises(APIError):
        client.ask("system", "user")
    # Пауза перед каждой повторной попыткой, но не после последней.
    assert no_sleep == [2**i for i in range(config.MAX_RETRIES - 1)]


@responses.activate
def test_http_error_is_not_retried(client, no_sleep):
    """Ретраятся только таймауты: 500 отдаётся пользователю сразу."""
    responses.add(responses.POST, config.API_URL, json={}, status=500)
    with pytest.raises(APIError):
        client.ask("system", "user")
    assert len(responses.calls) == 1


@responses.activate
@pytest.mark.parametrize("status", [502, 503, 504])
def test_temporary_unavailability_is_retried_until_success(client, no_sleep, status):
    """Временная недоступность сервиса — переходный сбой, как таймаут (живой 503 на демо дня 20)."""
    responses.add(responses.POST, config.API_URL, json={}, status=status)
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    assert client.ask("system", "user") == "ok"
    assert len(responses.calls) == 2


@responses.activate
def test_temporary_unavailability_gives_up_after_max_retries(client, no_sleep):
    for _ in range(config.MAX_RETRIES):
        responses.add(responses.POST, config.API_URL, json={}, status=503)
    with pytest.raises(APIError, match="503"):
        client.ask("system", "user")
    assert len(responses.calls) == config.MAX_RETRIES


@responses.activate
def test_request_timeout_value_is_passed(client):
    captured = {}

    def callback(request):
        captured["timeout"] = request.req_kwargs.get("timeout")
        return (200, {}, json.dumps(_completion("ok")))

    responses.add_callback(responses.POST, config.API_URL, callback=callback)
    client.ask("system", "user")
    assert captured["timeout"] == config.REQUEST_TIMEOUT


# --- пригодность ключа ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["sk-abc123", "sk-A1_b2-c3.d4", "x" * 200])
def test_ascii_keys_are_accepted(key):
    assert is_valid_api_key(key) is True


@pytest.mark.parametrize("key", ["", "sk-ключ", "sk-clé", "sk-키", "sk-abc\u00a0def"])
def test_non_ascii_or_empty_keys_are_rejected(key):
    assert is_valid_api_key(key) is False


@responses.activate
def test_non_ascii_key_fails_with_a_readable_message():
    """Раньше здесь вылетал UnicodeEncodeError из недр requests — падение с traceback."""
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    with pytest.raises(APIError, match="проверьте раскладку клавиатуры"):
        APIClient("sk-ключ-в-кириллице").ask("system", "user")


@responses.activate
def test_bad_key_is_rejected_before_any_request():
    """Заведомо непригодный ключ не должен приводить к обращению к API."""
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    with pytest.raises(APIError):
        APIClient("sk-кириллица").ask("system", "user")
    assert len(responses.calls) == 0


@responses.activate
def test_empty_key_is_reported_too():
    responses.add(responses.POST, config.API_URL, json=_completion("ok"), status=200)
    with pytest.raises(APIError, match=API_KEY_CHARSET_ERROR[:20]):
        APIClient("").ask("system", "user")


# --- проверка JSON-ответа ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"name_ru": "Каркассон"}',
        '  {"a": 1}  ',
        '```json\n{"a": 1}\n```',
        '```json\n{\n  "name_ru": "Catan",\n  "genre": "евро"\n}\n```',
        "[1, 2, 3]",
    ],
)
def test_valid_json_answers(text):
    assert is_valid_json_answer(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Каркассон — отличная игра",
        '{"name_ru": "Каркассон"',
        "```json\nне json\n```",
        "Вот ответ: {броken}",
    ],
)
def test_invalid_json_answers(text):
    assert is_valid_json_answer(text) is False


def test_json_block_wins_over_surrounding_prose():
    """Модель иногда добавляет текст вокруг блока — блок всё равно должен быть распознан."""
    assert is_valid_json_answer('Вот карточка:\n```json\n{"a": 1}\n```\nГотово!') is True


# --- день 8: причина завершения генерации ---------------------------------------------


@responses.activate
def test_finish_reason_is_captured_from_response(client):
    body = _completion("Обрезанный ответ")
    body["choices"][0]["finish_reason"] = "length"
    responses.add(responses.POST, config.API_URL, json=body, status=200)
    meta = client.ask_with_usage("system", "user")
    assert meta.finish_reason == "length"


@responses.activate
def test_missing_finish_reason_is_none(client):
    responses.add(responses.POST, config.API_URL, json=_completion("Ответ"), status=200)
    meta = client.ask_with_usage("system", "user")
    assert meta.finish_reason is None


@responses.activate
def test_empty_content_with_length_finish_reason_is_still_a_meta(client):
    """Известный излом reasoning-моделей: бюджет исчерпан, content пуст, ошибки нет."""
    body = _completion("")
    body["choices"][0]["finish_reason"] = "length"
    body["usage"] = {"prompt_tokens": 100, "completion_tokens": 4050, "total_tokens": 4150}
    responses.add(responses.POST, config.API_URL, json=body, status=200)
    meta = client.ask_with_usage("system", "user")
    assert meta.content == ""
    assert meta.finish_reason == "length"
    assert meta.completion_tokens == 4050
