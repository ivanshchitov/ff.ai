"""Клиент MCP: подключение по stdio или по адресу, список инструментов и вызов инструмента.

Приложение синхронное (главный цикл — обычный ``input()``), а пакет ``mcp`` асинхронный: граница
двух миров спрятана здесь, наружу модуль отдаёт обычные методы, а ``asyncio.run`` живёт внутри.
Перевод всего приложения на асинхронность стоил бы правки интерфейса и конвейера ради двух команд.

Соединение поднимается на время операции и закрывается сразу после неё: держать процесс сервера
всю сессию означало бы фоновый поток с собственным событийным циклом и корректное завершение
при выходе — для отчёта и ручного вызова это лишняя сложность.

Модуль ничего не знает о конкретных серверах: имена, описания и схемы инструментов приходят
от сервера, а способ подключения — из записи реестра (`core/mcp_registry.py`).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import config
from .mcp_registry import MCPServerSpec, TRANSPORT_HTTP

HEALTH_PROBE_MESSAGE = "ff.ai проверка соединения"

_SESSION_TERMINATION_FILTER_INSTALLED = False


class _SessionTerminationFilter(logging.Filter):
    """Гасит одно известное сообщение SDK.

    Сервер портала не поддерживает завершение сессии: на DELETE он отвечает `403` уже после того,
    как данные получены. Для приложения это не ошибка, но SDK печатает предупреждение прямо на
    экран пользователя — глушим только эту строку, остальные сообщения библиотеки остаются.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Session termination failed" not in record.getMessage()


def _install_log_filter() -> None:
    global _SESSION_TERMINATION_FILTER_INSTALLED
    if _SESSION_TERMINATION_FILTER_INSTALLED:
        return
    logging.getLogger("mcp.client.streamable_http").addFilter(_SessionTerminationFilter())
    _SESSION_TERMINATION_FILTER_INSTALLED = True


class MCPError(Exception):
    """Сбой подключения: процесс не запустился, рукопожатие не прошло, нет ответа."""


@dataclass(frozen=True)
class MCPParameter:
    """Параметр схемы инструмента в удобном для отчёта виде."""

    name: str
    type: str = ""
    description: str = ""
    required: bool = False
    allowed: Tuple[str, ...] = ()
    default: Optional[str] = None

    def render(self) -> str:
        parts = [self.name]
        if self.type:
            parts.append(f": {self.type}")
        if self.required:
            parts.append(" (обязательный)")
        if self.allowed:
            parts.append(f" [{', '.join(self.allowed)}]")
        if self.default is not None:
            parts.append(f" = {self.default}")
        return "".join(parts)


@dataclass(frozen=True)
class MCPTool:
    """Инструмент так, как его объявил сервер: имя, описание и схема входных параметров."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def parameters(self) -> List[MCPParameter]:
        """Параметры схемы; чужая схема может быть какой угодно, поэтому разбор защитный."""
        if not isinstance(self.input_schema, dict):
            return []
        properties = self.input_schema.get("properties")
        if not isinstance(properties, dict):
            return []
        required = self.input_schema.get("required") or []
        parameters: List[MCPParameter] = []
        for name, definition in properties.items():
            definition = definition if isinstance(definition, dict) else {}
            parameters.append(
                MCPParameter(
                    name=str(name),
                    type=str(definition.get("type", "")),
                    description=str(definition.get("description", "")),
                    required=name in required,
                    allowed=tuple(str(value) for value in definition.get("enum", []) or []),
                    default=None
                    if definition.get("default") is None
                    else str(definition["default"]),
                )
            )
        return parameters


@dataclass(frozen=True)
class MCPConnection:
    """Снимок подключения: кто ответил, по какому протоколу, что предлагает или чем не ответил."""

    server_name: str = ""
    server_version: str = ""
    protocol_version: str = ""
    tools: Tuple[MCPTool, ...] = ()
    error: Optional[str] = None
    transport: str = ""
    target: str = ""

    @property
    def available(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class MCPCallResult:
    """Результат вызова инструмента: отказ сервера — данные, а не исключение."""

    server: str
    tool: str
    arguments: Dict[str, Any]
    text: str
    is_error: bool = False


class MCPClient:
    """Клиент одного сервера: подключиться, спросить инструменты, вызвать инструмент."""

    def __init__(
        self,
        spec: MCPServerSpec,
        env: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.spec = spec
        self._env = dict(env) if env is not None else None
        self.timeout = float(timeout if timeout is not None else config.MCP_TIMEOUT)

    # --- публичные операции ---------------------------------------------------------------

    def connect(self) -> MCPConnection:
        """Поднимает соединение, забирает список инструментов и подтверждает его проверкой.

        Сбой не поднимается исключением наружу: недоступный сервер — обычная ситуация, и отчёт
        обязан её показать, а не оборвать обход остальных серверов.
        """
        try:
            return asyncio.run(self._connect())
        except MCPError as error:
            return self._failed(str(error))
        except Exception as error:  # noqa: BLE001 - чужая библиотека, причину показываем текстом
            return self._failed(_describe(error))

    def call_tool(self, tool: str, arguments: Optional[Dict[str, Any]] = None) -> MCPCallResult:
        """Вызывает инструмент; сбой подключения — `MCPError`, отказ инструмента — данные."""
        arguments = dict(arguments or {})
        try:
            return asyncio.run(self._call(tool, arguments))
        except MCPError:
            raise
        except Exception as error:  # noqa: BLE001
            raise MCPError(_describe(error)) from error

    # --- внутреннее -----------------------------------------------------------------------

    def _failed(self, reason: str) -> MCPConnection:
        return MCPConnection(
            error=reason,
            transport=self.spec.transport,
            target=self.spec.target,
        )

    def _child_env(self) -> Optional[Dict[str, str]]:
        """Окружение запускаемого процесса: своё плюс значения объявленных ключей.

        Ключи из записи реестра — это секреты конкретного сервера; если переменной нет,
        она просто не передаётся.
        """
        if not self.spec.env_keys and self._env is None:
            return None
        env = dict(os.environ if self._env is None else self._env)
        for key in self.spec.env_keys:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _target(self) -> Any:
        if self.spec.transport == TRANSPORT_HTTP:
            return self.spec.url
        from mcp import StdioServerParameters

        return StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args),
            env=self._child_env(),
        )

    def _client(self):
        from mcp.client.client import Client

        if self.spec.transport == TRANSPORT_HTTP:
            _install_log_filter()
        return Client(self._target(), read_timeout_seconds=self.timeout)

    async def _connect(self) -> MCPConnection:
        async with self._client() as client:
            listing = await client.list_tools()
            tools = tuple(
                MCPTool(
                    name=str(getattr(item, "name", "")),
                    description=str(getattr(item, "description", "") or ""),
                    # Модели mcp 2.x — snake_case (`input_schema`, `is_error`); на проводе
                    # они отдаются как inputSchema, но библиотека переводит их сама.
                    input_schema=dict(getattr(item, "input_schema", None) or {}),
                )
                for item in getattr(listing, "tools", ()) or ()
            )
            info = getattr(client, "server_info", None)
            connection = MCPConnection(
                server_name=str(getattr(info, "name", "") or ""),
                server_version=str(getattr(info, "version", "") or ""),
                protocol_version=str(getattr(client, "protocol_version", "") or ""),
                tools=tools,
                transport=self.spec.transport,
                target=self.spec.target,
            )
            return await self._health_check(client, connection)

    async def _health_check(self, client, connection: MCPConnection) -> MCPConnection:
        """Проверка соединения вызовом: имя инструмента проверки приходит из записи реестра."""
        if not self.spec.health_tool:
            return connection
        if self.spec.health_tool not in {tool.name for tool in connection.tools}:
            return MCPConnection(
                server_name=connection.server_name,
                server_version=connection.server_version,
                protocol_version=connection.protocol_version,
                tools=connection.tools,
                transport=connection.transport,
                target=connection.target,
                error=(
                    f"инструмент проверки «{self.spec.health_tool}» не объявлен сервером, "
                    "соединение не подтверждено"
                ),
            )
        result = await client.call_tool(
            self.spec.health_tool, {"message": HEALTH_PROBE_MESSAGE}
        )
        text = _content_text(result)
        if getattr(result, "is_error", False):
            return MCPConnection(
                server_name=connection.server_name,
                server_version=connection.server_version,
                protocol_version=connection.protocol_version,
                tools=connection.tools,
                transport=connection.transport,
                target=connection.target,
                error=f"проверка соединения не прошла: {text or 'сервер вернул ошибку'}",
            )
        return connection

    async def _call(self, tool: str, arguments: Dict[str, Any]) -> MCPCallResult:
        async with self._client() as client:
            result = await client.call_tool(tool, arguments)
            return MCPCallResult(
                server=self.spec.name,
                tool=tool,
                arguments=arguments,
                text=_content_text(result),
                is_error=bool(getattr(result, "is_error", False)),
            )


def _content_text(result: Any) -> str:
    """Текст результата: сервер отдаёт список блоков, текстовые из них склеиваем."""
    chunks: List[str] = []
    for item in getattr(result, "content", ()) or ():
        text = getattr(item, "text", None)
        if text is not None:
            chunks.append(str(text))
    return "\n".join(chunks).strip()


def _describe(error: BaseException) -> str:
    """Читаемая причина: SDK заворачивает сбои задач в группу исключений."""
    seen: List[str] = []
    current: Optional[BaseException] = error
    while current is not None and len(seen) < 4:
        text = f"{type(current).__name__}: {current}".strip()
        if text not in seen:
            seen.append(text)
        nested = getattr(current, "exceptions", None)
        current = nested[0] if nested else current.__cause__
    return " → ".join(seen) if seen else "неизвестная ошибка"
