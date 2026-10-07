"""Клиент для обращения к Chat Completions API."""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from . import config, usage

_JSON_CODE_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def is_valid_json_answer(text: str) -> bool:
    """Проверяет, что ответ — валидный JSON (в т.ч. внутри блока ```json)."""
    match = _JSON_CODE_BLOCK_RE.search(text)
    candidate = match.group(1) if match else text
    try:
        json.loads(candidate)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_api_key(api_key: str) -> bool:
    """Проверяет, что ключ пригоден для HTTP-заголовка.

    Заголовки кодируются latin-1, поэтому ключ с кириллицей (частая опечатка — набор в русской
    раскладке) обрывал бы запрос UnicodeEncodeError глубоко внутри requests, то есть падением с
    traceback вместо понятного сообщения. Проверяем заранее и сами.
    """
    return bool(api_key) and api_key.isascii()


API_KEY_CHARSET_ERROR = (
    "API-ключ содержит недопустимые символы. Допустимы только латинские буквы, цифры и знаки "
    "пунктуации — проверьте раскладку клавиатуры."
)


# Статусы временной недоступности: повторяются, как таймаут.
TRANSIENT_STATUSES = frozenset({502, 503, 504})


class APIError(Exception):
    """Ошибка при обращении к API."""


@dataclass
class AnswerMeta:
    """Ответ модели вместе с метриками запроса: время, токены, стоимость, причина завершения."""

    content: str
    model: str
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Optional[float]
    # Значение choices[0].finish_reason из ответа API; None, если поле не пришло.
    finish_reason: Optional[str] = None


class APIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: Optional[float],
        model: str,
    ) -> tuple[Dict[str, Any], float]:
        """Единый HTTP-путь: messages — готовый список ролей; None-температура = дефолт."""
        local = model in config.LOCAL_MODELS
        if not local and not is_valid_api_key(self.api_key):
            raise APIError(API_KEY_CHARSET_ERROR + " Добавьте OPENCODE_API_KEY в .env для облачных моделей.")

        if local:
            # Шаблон Qwen разрешает system только в начале. Стек агента содержит
            # несколько system-блоков: сохраняем их текст и порядок в одном блоке.
            instructions = [item["content"] for item in messages if item["role"] == "system"]
            conversation = [item for item in messages if item["role"] != "system"]
            messages = ([{"role": "system", "content": "\n\n".join(instructions)}]
                        if instructions else []) + conversation

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else config.TEMPERATURE,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if not local:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_timeout: Optional[Exception] = None
        start = time.perf_counter()

        for attempt in range(config.MAX_RETRIES):
            try:
                response = requests.post(
                    config.api_url_for_model(model), json=payload, headers=headers,
                    timeout=config.REQUEST_TIMEOUT
                )
            except requests.exceptions.Timeout as exc:
                last_timeout = exc
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(2**attempt)
                    continue
                raise APIError(
                    "Превышено время ожидания ответа от API."
                ) from last_timeout
            except requests.exceptions.ConnectionError as exc:
                raise APIError(
                    "Ошибка соединения с локальным API. Проверьте llama-server на порту 9999."
                    if local else "Ошибка соединения с API. Проверьте интернет-соединение."
                ) from exc

            if response.status_code == 401:
                raise APIError("Неверный API-ключ.")
            # Временная недоступность сервиса — переходный сбой, как таймаут: живой 503 от
            # OpenCode Zen посреди флоу инструментов оборвал демо дня 20. Последняя попытка
            # отдаёт ошибку со статусом, как прочие HTTP-ошибки.
            if response.status_code in TRANSIENT_STATUSES and attempt < config.MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                raise APIError(f"Ошибка API: {exc}") from exc

            elapsed = time.perf_counter() - start
            try:
                data = response.json()
                data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError) as exc:
                raise APIError("Некорректный ответ от API.") from exc
            return data, elapsed

        raise APIError("Превышено время ожидания ответа от API.")

    def ask(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = config.max_tokens_for_words(config.DEFAULT_MAX_WORDS),
        temperature: float = config.TEMPERATURE,
        model: str = config.DEFAULT_MODEL,
    ) -> str:
        data, _ = self._request(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens,
            temperature,
            model,
        )
        return data["choices"][0]["message"]["content"].strip()

    def ask_with_usage(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = config.max_tokens_for_words(config.DEFAULT_MAX_WORDS),
        temperature: float = config.TEMPERATURE,
        model: str = config.DEFAULT_MODEL,
    ) -> AnswerMeta:
        data, elapsed = self._request(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens,
            temperature,
            model,
        )
        return self._meta_from(data, elapsed, model)

    def ask_with_usage_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = config.max_tokens_for_words(config.DEFAULT_MAX_WORDS),
        temperature: Optional[float] = None,
        model: str = config.DEFAULT_MODEL,
    ) -> AnswerMeta:
        """Полный список сообщений (стек агента); None-температура — клиентский дефолт."""
        data, elapsed = self._request(messages, max_tokens, temperature, model)
        return self._meta_from(data, elapsed, model)

    @staticmethod
    def _meta_from(data: Dict[str, Any], elapsed: float, model: str) -> AnswerMeta:
        content = data["choices"][0]["message"]["content"].strip()
        finish_reason = data["choices"][0].get("finish_reason")
        request_usage = data.get("usage") or {}
        prompt_tokens = request_usage.get("prompt_tokens", 0)
        completion_tokens = request_usage.get("completion_tokens", 0)
        total_tokens = request_usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost_usd = usage.estimate_cost(model, prompt_tokens, completion_tokens)
        return AnswerMeta(
            content=content,
            model=model,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
        )
