"""Фасад сессии: единственная точка входа для интерфейсов.

Терминальный интерфейс (и будущий web-фронтенд) не владеет агентом, настройками, моделью или
расходом: он вызывает операции сессии и подписывается на события. Это то, что позволяет
держать два интерфейса над одним ядром и оставить `core/` без зависимости от терминала.

Операции сериализованы мьютексом: агент синхронный и блокирующий, а «одна операция за раз» —
его собственный контракт (один вопрос — один запрос). Ошибка в подписчике не должна ломать
операцию: интерфейс — наблюдатель, а не участник.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import config, domains, mcp_registry
from .agent import CompressionReport, ContextReport, FactsReport, RepoAgent, RequestPhase
from .answer_settings import AnswerSettings
from .api_client import APIClient, APIError, AnswerMeta, is_valid_api_key
from .domains import Domain, DomainError, DomainSelection
from .history_manager import HistoryManager
from .mcp_client import MCPCallResult, MCPClient, MCPConnection, MCPError, MCPTool
from .mcp_registry import MCPServerSpec
from .usage import SessionUsage


# --- события ------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseChanged:
    """Сменилась фаза операции — интерфейс показывает её в спиннере."""

    phase: RequestPhase


@dataclass(frozen=True)
class JournalLine:
    """Строка журнала: то, что произошло по ходу операции и заслуживает упоминания на экране."""

    text: str


@dataclass(frozen=True)
class AnswerReady:
    """Ответ на вопрос готов и уже записан в историю."""

    question: str
    answer: str
    meta: Optional[AnswerMeta]


@dataclass(frozen=True)
class DomainSelected:
    """Активный домен выбран (при старте или сменой командой)."""

    selection: DomainSelection


def _split_command_line(text: str) -> List[str]:
    """Разбор строки команды с поддержкой кавычек; незакрытая кавычка — не повод падать."""
    import shlex

    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def parse_tool_arguments(tokens: Sequence[str]) -> Dict[str, str]:
    """Аргументы ручного вызова: `ключ=значение`, как их удобно набирать в строке.

    Значения не разбираются на типы: сервер объявляет схему и сам решает, что делать с текстом.
    """
    arguments: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise MCPError(f"аргумент «{token}» задан не как ключ=значение")
        key, _, value = token.partition("=")
        if not key.strip():
            raise MCPError(f"аргумент «{token}» без имени ключа")
        arguments[key.strip()] = value
    return arguments


@dataclass(frozen=True)
class Answer:
    """Результат операции «задать вопрос»."""

    question: str
    text: str
    meta: Optional[AnswerMeta]


@dataclass(frozen=True)
class CommandResult:
    """Результат команды: строки для печати и признаки, важные интерфейсу."""

    command: str
    lines: Tuple[str, ...] = ()
    exit_requested: bool = False
    unknown: bool = False
    domain_changed: bool = False


class AssistantSession:
    """Сессия ассистента: агент, домен, настройки ответа, модель и расход."""

    def __init__(
        self,
        root: Path,
        domain_id: Optional[str] = None,
        client: Optional[APIClient] = None,
        domains_dir: Optional[Path] = None,
        history: Optional[HistoryManager] = None,
        settings: Optional[AnswerSettings] = None,
        mcp_client_factory: Optional[Callable[[MCPServerSpec], MCPClient]] = None,
    ) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[object], None]] = []
        self._exit_requested = False
        self._history = history if history is not None else HistoryManager()
        self._selection = domains.select_domain(
            root=self.root, explicit=domain_id, domains_dir=domains_dir
        )
        self._agent = RepoAgent(
            domain=self._selection.domain,
            client=client,
            root=self.root,
            settings=settings if settings is not None else AnswerSettings(),
            history=self._history,
        )
        self._domains_dir = domains_dir
        # Фабрика клиентов MCP: тесты подменяют её, чтобы не поднимать серверные процессы.
        self._mcp_client_factory = mcp_client_factory or MCPClient
        self._mcp_specs: Tuple[MCPServerSpec, ...] = ()
        self._mcp_connections: Tuple[MCPConnection, ...] = ()

    # --- доступ к состоянию ---------------------------------------------------------------

    @property
    def domain(self) -> Domain:
        return self._selection.domain

    @property
    def domain_selection(self) -> DomainSelection:
        return self._selection

    @property
    def history(self) -> HistoryManager:
        return self._history

    @property
    def settings(self) -> AnswerSettings:
        return self._agent.settings

    @settings.setter
    def settings(self, value: AnswerSettings) -> None:
        self._agent.settings = value

    @property
    def model(self) -> str:
        return self._agent.model

    @model.setter
    def model(self, value: str) -> None:
        self._agent.model = value

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    @property
    def last_result(self) -> Optional[AnswerMeta]:
        return self._agent.last_result

    @property
    def last_compression(self) -> Optional[CompressionReport]:
        return self._agent.last_compression

    @property
    def last_facts(self) -> Optional[FactsReport]:
        return self._agent.last_facts

    @property
    def session_usage(self) -> SessionUsage:
        return self._agent.session_usage

    @property
    def client(self) -> Optional[APIClient]:
        return self._agent.client

    @client.setter
    def client(self, value: Optional[APIClient]) -> None:
        self._agent.client = value

    def set_api_key(self, api_key: str) -> None:
        """Ключ, введённый пользователем вручную, становится ключом процесса и клиента сессии."""
        config.set_api_key_runtime(api_key)
        self._agent.client = APIClient(api_key)

    def has_usable_api_key(self) -> bool:
        return is_valid_api_key(config.get_api_key() or "")

    # --- события --------------------------------------------------------------------------

    def subscribe(self, listener: Callable[[object], None]) -> Callable[[], None]:
        """Подписка на события; возвращает функцию отписки."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: object) -> None:
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception:
                # Подписчик — наблюдатель: его сбой не должен ломать операцию и мешать другим.
                continue

    # --- операции -------------------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Задать вопрос. Операция блокирующая: следующий вызов ждёт завершения этого."""
        with self._lock:
            self._require_client()
            meta = self._agent.ask(
                question, on_phase=lambda phase: self._emit(PhaseChanged(phase))
            )
            self._emit_answer_journal()
            self._emit(AnswerReady(question=question, answer=meta.content, meta=meta))
            return Answer(question=question, text=meta.content, meta=meta)

    def _require_client(self) -> APIClient:
        """Клиент создаётся из ключа при первом вопросе: интерфейс спрашивает ключ заранее.

        Без ключа вопрос отправить нечем, поэтому это ожидаемая ошибка запроса, а не сбой
        программы: интерфейс покажет её пользователю ровно так же, как отказ API.
        """
        if self._agent.client is None:
            api_key = config.get_api_key()
            if not is_valid_api_key(api_key or ""):
                raise APIError(
                    "Нет пригодного ключа OPENCODE_API_KEY: облачная модель без него "
                    "недоступна (локальной модели ключ не нужен)."
                )
            self._agent.client = APIClient(api_key)
        return self._agent.client

    def run_command(self, text: str) -> CommandResult:
        """Выполнить команду. Отчёты не обращаются к модели и не меняют расход."""
        command, _, argument = text.strip().partition(" ")
        command = command.strip()
        argument = argument.strip()
        with self._lock:
            if command in ("/exit", "/quit"):
                self._exit_requested = True
                return CommandResult(command="/exit", lines=("Выход.",), exit_requested=True)
            if command == "/clear":
                self._agent.reset()
                return CommandResult(command="/clear", lines=("Диалог очищен.",))
            if command == "/context":
                return CommandResult(command="/context", lines=self.context_lines())
            if command == "/usage":
                return CommandResult(command="/usage", lines=self.usage_lines())
            if command == "/domain":
                return self._domain_command(argument)
            if command == "/mcp":
                return self._mcp_command(argument)
            if command == "/tool":
                return self._tool_command(argument)
            if command == "/commands":
                return CommandResult(
                    command="/commands",
                    lines=("Список команд открывается панелью интерфейса.",),
                )
            return CommandResult(
                command=command or text.strip(),
                lines=(f"Неизвестная команда «{text.strip()}». Список — /commands.",),
                unknown=True,
            )

    def context_report(self) -> ContextReport:
        with self._lock:
            return self._agent.context_report()

    # --- MCP: серверы, отчёт, вызов инструмента -------------------------------------------

    def connect_mcp_servers(self) -> Tuple[MCPConnection, ...]:
        """Обходит реестр и запоминает снимки подключений.

        Соединение каждого сервера живёт ровно этот вызов: наружу уходят только снимки, поэтому
        отчёт печатается мгновенно и ничего не поднимает заново. Сбой одного сервера остаётся
        его собственной строкой отчёта и не мешает остальным.
        """
        with self._lock:
            specs = mcp_registry.registry(self.domain, self.root)
            self._mcp_specs = specs
            connections = []
            for spec in specs:
                self._emit(PhaseChanged(RequestPhase.MCP_CONNECT))
                connections.append(self._mcp_client_factory(spec).connect())
            self._mcp_connections = tuple(connections)
            return self._mcp_connections

    def mcp_report(self) -> Tuple[MCPConnection, ...]:
        with self._lock:
            return self._mcp_connections

    def mcp_specs(self) -> Tuple[MCPServerSpec, ...]:
        with self._lock:
            return self._mcp_specs

    def mcp_tools(self) -> Tuple[Tuple[str, MCPTool], ...]:
        """Инструменты всех подключённых серверов: (имя сервера, инструмент)."""
        return tuple(
            (connection.server_name or spec_name, tool)
            for spec_name, connection in zip(
                (spec.name for spec in self._mcp_specs), self._mcp_connections
            )
            for tool in connection.tools
        )

    def mcp_summary(self) -> str:
        """Строка стартовой сводки: сколько серверов ответило и сколько инструментов доступно."""
        total = len(self._mcp_specs)
        available = sum(1 for connection in self._mcp_connections if connection.available)
        tools = sum(len(connection.tools) for connection in self._mcp_connections)
        line = f"MCP: {available}/{total} серверов, {tools} инструментов — подробности: /mcp"
        if total and available < total:
            line += f", {total - available} недоступно"
        return line

    def call_mcp_tool(self, tool: str, arguments: Optional[Dict[str, str]] = None) -> MCPCallResult:
        """Ручной вызов инструмента: сервер ищется среди подключённых, процессы не поднимаются.

        Если одно имя объявили несколько серверов, берётся первый в порядке реестра; явное
        указание сервера появится вместе с автовызовом, где выбор делает модель (фаза P5).
        """
        with self._lock:
            spec = self._find_server_with_tool(tool)
            connection = next(
                (
                    item
                    for item, declared in zip(self._mcp_connections, self._mcp_specs)
                    if declared.name == spec.name
                ),
                None,
            )
            if connection is None:
                raise MCPError(f"сервер «{spec.name}» не подключён — обновите отчёт командой /mcp refresh")
            self._emit(PhaseChanged(RequestPhase.MCP_TOOL))
            return self._mcp_client_factory(spec).call_tool(tool, dict(arguments or {}))

    def _find_server_with_tool(self, tool: str) -> MCPServerSpec:
        declaring = [
            spec
            for spec, connection in zip(self._mcp_specs, self._mcp_connections)
            if tool in {item.name for item in connection.tools}
        ]
        if not declaring:
            known = ", ".join(sorted({item.name for _, item in self.mcp_tools()})) or "нет ни одного"
            raise MCPError(f"инструмент «{tool}» не объявлен ни одним сервером реестра; известны: {known}")
        return declaring[0]

    def mcp_lines(self) -> Tuple[str, ...]:
        """Строки отчёта о подключениях: по серверу — имя, версия, протокол, инструменты, ошибка."""
        if not self._mcp_connections:
            return ("Подключения не проверялись: обход реестра не выполнялся.",)
        lines: List[str] = []
        for spec, connection in zip(self._mcp_specs, self._mcp_connections):
            title = spec.name
            if connection.server_name:
                title += f" («{connection.server_name}»"
                title += f" {connection.server_version})" if connection.server_version else ")"
            lines.append(f"{title} — {spec.transport}: {connection.target}")
            if not connection.available:
                lines.append(f"    недоступен: {connection.error}")
                continue
            lines.append(f"    протокол: {connection.protocol_version or 'н/д'}; инструментов: {len(connection.tools)}")
            for tool in connection.tools:
                parameters = ", ".join(item.render() for item in tool.parameters())
                lines.append(f"    • {tool.name}: {tool.description}")
                if parameters:
                    lines.append(f"        параметры: {parameters}")
        return tuple(lines)

    def tool_lines(self) -> Tuple[str, ...]:
        """Перечень инструментов всех серверов — то, что можно вызвать вручную."""
        tools = self.mcp_tools()
        if not tools:
            return ("Инструментов нет: серверы не подключены или ничего не объявляют.",)
        lines: List[str] = []
        for server, tool in tools:
            parameters = ", ".join(item.render() for item in tool.parameters())
            lines.append(f"{server}.{tool.name}: {tool.description}")
            if parameters:
                lines.append(f"    параметры: {parameters}")
        return tuple(lines)

    # --- ветки диалога --------------------------------------------------------------------

    def branches(self) -> Tuple[Tuple[str, int], ...]:
        with self._lock:
            return self._agent.branches()

    @property
    def active_branch(self) -> str:
        with self._lock:
            return self._agent.active_branch

    def switch_branch(self, name: str) -> bool:
        with self._lock:
            return self._agent.switch_branch(name)

    def set_checkpoint(self) -> None:
        with self._lock:
            self._agent.set_checkpoint()

    def new_branch(self) -> str:
        with self._lock:
            return self._agent.new_branch()

    def context_lines(self) -> Tuple[str, ...]:
        """Отчёт о контексте: стратегия, границы, память стратегии, оценка токенов."""
        report = self.context_report()
        lines = [
            f"Стратегия: {report.strategy.value}; окно: {report.window} сообщений; "
            f"потолок: {report.max_session_tokens} токенов",
            f"Обменов в диалоге: {report.log_exchanges}; "
            f"ходов в ближайшем запросе: {report.request_turns}",
            f"Резюме: {'есть' if report.has_summary else 'нет'}, "
            f"покрыто обменов: {report.summary_covers}",
        ]
        if report.facts:
            facts = "; ".join(f"{key} — {value}" for key, value in report.facts.items())
            lines.append(f"Факты: {facts}")
        if len(report.branches) > 1:
            branches = ", ".join(
                f"{name} ({exchanges})" for name, exchanges in report.branches
            )
            lines.append(f"Активная ветка: {report.branch}; ветки: {branches}")
        lines.append(f"Оценка запроса: ≈{report.tokens_estimate} токенов")
        return tuple(lines)

    def usage_lines(self) -> Tuple[str, ...]:
        """Отчёт о расходе: последний запрос, итоги сессии и расход всей истории."""
        lines: List[str] = []
        meta = self.last_result
        if meta is not None:
            lines.append(
                f"Последний запрос: {meta.model}, {meta.elapsed_seconds:.2f} с, "
                f"{meta.prompt_tokens}+{meta.completion_tokens}={meta.total_tokens} токенов"
            )
        usage = self.session_usage
        lines.append(
            f"Сессия: запросов — {usage.requests}, входных — {usage.prompt_tokens}, "
            f"выходных — {usage.completion_tokens}, всего — {usage.total_tokens} токенов"
        )
        total = self._history.total_usage()
        lines.append(
            f"Вся история ({self._history.count()} обменов): "
            f"запросов — {total.requests}, всего — {total.total_tokens} токенов"
        )
        return tuple(lines)

    def _mcp_command(self, argument: str) -> CommandResult:
        """`/mcp` — отчёт о подключениях, `/mcp refresh` — обойти реестр заново."""
        if argument.strip() == "refresh":
            self.connect_mcp_servers()
            return CommandResult(
                command="/mcp",
                lines=(f"Реестр обойдён заново — {self.mcp_summary()}",) + self.mcp_lines(),
            )
        return CommandResult(command="/mcp", lines=self.mcp_lines())

    def _tool_command(self, argument: str) -> CommandResult:
        """`/tool` — перечень инструментов, `/tool call <имя> ключ=значение …` — ручной вызов.

        Строка разбирается как командная: значение с пробелами берётся в кавычки —
        `/tool call <инструмент> query="поисковый запрос"`. Имена инструментов ядру неизвестны:
        они приходят от серверов, поэтому пример здесь обезличен.
        """
        tokens = _split_command_line(argument)
        if not tokens:
            return CommandResult(command="/tool", lines=self.tool_lines())
        if tokens[0] != "call":
            return CommandResult(
                command="/tool",
                lines=(
                    "Форма команды: /tool — перечень, "
                    "/tool call <инструмент> ключ=значение …",
                ),
            )
        if len(tokens) < 2:
            return CommandResult(
                command="/tool", lines=("Укажите инструмент: /tool call <инструмент> …",)
            )
        tool = tokens[1]
        try:
            arguments = parse_tool_arguments(tokens[2:])
            result = self.call_mcp_tool(tool, arguments)
        except MCPError as error:
            return CommandResult(command="/tool", lines=(f"Инструмент не вызван: {error}",))
        head = f"🔧 {result.server}.{result.tool}"
        if result.arguments:
            rendered = ", ".join(f"{key}={value}" for key, value in result.arguments.items())
            head += f" ({rendered})"
        if result.is_error:
            return CommandResult(
                command="/tool",
                lines=(f"{head}: инструмент вернул ошибку", result.text or "(пустой ответ)"),
            )
        return CommandResult(command="/tool", lines=(head + ":", result.text or "(пустой ответ)"))

    def _domain_command(self, argument: str) -> CommandResult:
        """`/domain` без аргумента — отчёт, `/domain switch <id>` — смена активного домена."""
        if argument.startswith("switch"):
            requested = argument[len("switch") :].strip()
            if not requested:
                return CommandResult(
                    command="/domain",
                    lines=("Укажите домен: /domain switch <id>.",),
                )
            try:
                selection = domains.select_domain(
                    root=self.root, explicit=requested, domains_dir=self._domains_dir
                )
            except DomainError as error:
                return CommandResult(command="/domain", lines=(str(error),))
            self._selection = selection
            self._agent.domain = selection.domain
            self._emit(DomainSelected(selection))
            return CommandResult(
                command="/domain",
                lines=self.domain_lines(),
                domain_changed=True,
            )
        return CommandResult(command="/domain", lines=self.domain_lines())

    def domain_lines(self) -> Tuple[str, ...]:
        domain = self.domain
        selection = self._selection
        lines = [
            f"Домен: {domain.title} ({domain.id})",
            f"Платформа: {domain.platform}; документация: {domain.docs_version or 'н/д'}; "
            f"локальный SDK: {domain.local_sdk_version or 'н/д'}",
            f"Выбран: {selection.source}"
            + (f"; маркеры: {', '.join(selection.evidence)}" if selection.evidence else ""),
            f"Инвариантов домена: {len(domain.invariants)}; "
            f"корень репозитория: {self.root}",
        ]
        return tuple(lines)

    # --- вспомогательное ------------------------------------------------------------------

    def _emit_answer_journal(self) -> None:
        """Строки журнала о том, что стратегия сделала перед ответом."""
        if self._history.last_error:
            self._emit(
                JournalLine(f"История не сохранена: {self._history.last_error}")
            )
        compression = self.last_compression
        if compression is not None:
            self._emit(
                JournalLine(
                    f"Сжатие: свёрнуто {compression.exchanges} обменов "
                    f"({compression.messages} сообщений)."
                )
            )
        facts = self.last_facts
        if facts is not None:
            if facts.updated:
                self._emit(JournalLine(f"Факты обновлены: {facts.keys} ключей."))
            else:
                self._emit(JournalLine(f"Факты не обновлены: {facts.error}"))
