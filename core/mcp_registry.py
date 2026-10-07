"""Реестр MCP-серверов: записи как данные и сборка реестра на запуск.

Запись объявляет имя, транспорт (`stdio` или `http`), способ подключения — команду запуска либо
адрес, — описание, ключи окружения с секретами и, необязательно, имя инструмента проверки
соединения. Серверы, относящиеся к области применимости, объявляет пакет домена; собственный
сервер репозитория — часть инструмента.

Имён инструментов конкретных серверов ядро не знает: они приходят от серверов при подключении.
Именно поэтому замена сервера остаётся правкой данных, а не кода.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import config

TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"
TRANSPORTS = (TRANSPORT_STDIO, TRANSPORT_HTTP)

REPO_SERVER_NAME = "repo"
REPO_SERVER_DESCRIPTION = (
    "Инструменты целевого репозитория: перечень файлов, чтение, поиск, история git, "
    "разрешённые команды сборки и сохранение выгрузки. Репозиторий не изменяет."
)


class MCPSpecError(Exception):
    """Запись реестра неполна или противоречива."""


@dataclass(frozen=True)
class MCPServerSpec:
    """Сервер так, как его объявили данные: способ подключения и метаданные."""

    name: str
    transport: str = TRANSPORT_STDIO
    description: str = ""
    command: str = ""
    args: Tuple[str, ...] = ()
    url: str = ""
    env_keys: Tuple[str, ...] = ()
    health_tool: str = ""
    source: str = ""

    def validate(self) -> None:
        """Проверка записи: без неё сломанный реестр проявился бы как «сервер молчит»."""
        if not self.name.strip():
            raise MCPSpecError("у записи сервера нет имени")
        if self.transport not in TRANSPORTS:
            raise MCPSpecError(
                f"{self.name}: неизвестный транспорт {self.transport!r}, "
                f"допустимы {', '.join(TRANSPORTS)}"
            )
        if self.transport == TRANSPORT_STDIO and not self.command.strip():
            raise MCPSpecError(f"{self.name}: для транспорта stdio нужна команда запуска")
        if self.transport == TRANSPORT_HTTP:
            if not self.url.strip():
                raise MCPSpecError(f"{self.name}: для транспорта http нужен адрес")
            if not self.url.startswith(("http://", "https://")):
                raise MCPSpecError(f"{self.name}: адрес должен начинаться с http:// или https://")

    @property
    def target(self) -> str:
        """Строка, по которой сервер узнаётся в отчёте: команда или адрес."""
        if self.transport == TRANSPORT_HTTP:
            return self.url
        return " ".join([self.command, *self.args]).strip()

    @classmethod
    def from_data(cls, data: Mapping[str, Any], source: str) -> "MCPServerSpec":
        """Запись из файла данных: словарь с полями, описанными в документации пакета."""
        if not isinstance(data, Mapping):
            raise MCPSpecError("запись сервера должна быть объектом")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MCPSpecError("у записи сервера нет обязательного поля «name»")

        def _strings(key: str) -> Tuple[str, ...]:
            raw = data.get(key, [])
            if raw is None:
                return ()
            if isinstance(raw, str) or not isinstance(raw, Sequence):
                raise MCPSpecError(f"{name}: поле «{key}» должно быть списком строк")
            values = []
            for item in raw:
                if not isinstance(item, str):
                    raise MCPSpecError(f"{name}: поле «{key}» должно быть списком строк")
                values.append(item)
            return tuple(values)

        transport = data.get("transport", TRANSPORT_STDIO)
        if not isinstance(transport, str):
            raise MCPSpecError(f"{name}: поле «transport» должно быть строкой")
        spec = cls(
            name=name.strip(),
            transport=transport.strip(),
            description=str(data.get("description", "")),
            command=str(data.get("command", "")),
            args=_strings("args"),
            url=str(data.get("url", "")),
            env_keys=_strings("env_keys"),
            health_tool=str(data.get("health_tool", "")),
            source=source,
        )
        spec.validate()
        return spec


def repo_server_spec(root: Path, tools_path: Optional[Path] = None) -> MCPServerSpec:
    """Запись собственного сервера репозитория: корень и файл белых списков — аргументы запуска."""
    script = Path(__file__).resolve().parent.parent / "mcp_server" / "repo_server.py"
    args = [str(script), "--root", str(Path(root))]
    if tools_path is not None:
        args += ["--tools", str(tools_path)]
    spec = MCPServerSpec(
        name=REPO_SERVER_NAME,
        transport=TRANSPORT_STDIO,
        description=REPO_SERVER_DESCRIPTION,
        command=sys.executable,
        args=tuple(args),
        # Каталог выгрузок читает серверный процесс, а не приложение: без этой переменной в
        # окружении сервера выгрузка уходила бы в каталог состояния по умолчанию, а не туда,
        # куда просит запуск (ловилось живым прогоном: отчёт оказался в домашнем каталоге).
        env_keys=("FFAI_EXPORTS_DIR",),
        source="инструмент",
    )
    spec.validate()
    return spec


def _override_from_environment() -> Optional[Tuple[MCPServerSpec, ...]]:
    """Тестовое и демонстрационное переопределение реестра: один сервер вместо всего реестра.

    Без него любой прогон — тест в том числе — поднимал бы реальный сервер документации
    портала и собственный сервер репозитория.
    """
    url = os.getenv("FFAI_MCP_URL", "").strip()
    command = os.getenv("FFAI_MCP_COMMAND", "").strip()
    if url:
        spec = MCPServerSpec(
            name=config.MCP_OVERRIDE_NAME,
            transport=TRANSPORT_HTTP,
            url=url,
            description="сервер, заданный переменной FFAI_MCP_URL",
            source="окружение",
        )
        spec.validate()
        return (spec,)
    if command:
        # Несколько наборов аргументов, разделённых « | », дают несколько серверов: сквозные тесты
        # автовызова проверяют маршрутизацию шага между серверами, а переопределение окружением
        # заменяет реестр целиком — значит, серверов должно быть столько, сколько нужно тесту.
        sets = [
            tuple(part for part in chunk.split(" ") if part)
            for chunk in os.getenv("FFAI_MCP_ARGS", "").split(" | ")
        ]
        specs = []
        for index, args in enumerate(sets, start=1):
            name = config.MCP_OVERRIDE_NAME if len(sets) == 1 else f"{config.MCP_OVERRIDE_NAME} {index}"
            spec = MCPServerSpec(
                name=name,
                transport=TRANSPORT_STDIO,
                command=command,
                args=args,
                description="сервер, заданный переменными FFAI_MCP_COMMAND и FFAI_MCP_ARGS",
                source="окружение",
            )
            spec.validate()
            specs.append(spec)
        return tuple(specs)
    return None


def registry(domain: Any = None, root: Optional[Path] = None) -> Tuple[MCPServerSpec, ...]:
    """Реестр на запуск: серверы домена, затем собственный сервер репозитория.

    Переопределение окружением заменяет реестр целиком — так тесты и демонстрации не поднимают
    ни сервер портала, ни сервер репозитория.
    """
    override = _override_from_environment()
    if override is not None:
        return override
    specs = list(getattr(domain, "servers", ()) or ())
    if root is not None:
        tools_path = getattr(domain, "tools_path", None)
        specs.append(repo_server_spec(Path(root), tools_path))
    return tuple(specs)


def describe(specs: Sequence[MCPServerSpec]) -> str:
    """Короткая подпись реестра для журнальной строки: имена и транспорты."""
    return ", ".join(f"{spec.name} ({spec.transport})" for spec in specs)
