"""Фасад сессии: единственная точка входа для интерфейсов.

Терминальный интерфейс (и будущий web-фронтенд) не владеет агентом, настройками, моделью или
расходом: он вызывает операции сессии и подписывается на события. Это то, что позволяет
держать два интерфейса над одним ядром и оставить `core/` без зависимости от терминала.

Операции сериализованы мьютексом: агент синхронный и блокирующий, а «одна операция за раз» —
его собственный контракт (один вопрос — один запрос). Ошибка в подписчике не должна ломать
операцию: интерфейс — наблюдатель, а не участник.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import (
    aurora_ops,
    code_index,
    code_retrieval,
    config,
    domain_checks,
    domains,
    mcp_registry,
    mcp_tools,
    repo,
    long_term_memory,
    memory_layers,
    schedule_store,
    user_profile,
    patches,
    reranking,
    task_pipeline,
    task_state,
)
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
    # Прогон задачи ведёт интерфейс: команда лишь сообщает, что его надо начать.
    task_run: bool = False
    # Опросник профиля — диалог с пользователем, поэтому его ведёт интерфейс.
    profile_setup: bool = False


# Отличие «ретривер не передан» от «ретривера нет»: первое означает «собери по данным домена»,
# второе — «поиска по документации в этой сессии не будет». Тестам нужен именно второй случай.
RETRIEVER_UNSET = object()


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
        docs_retriever: object = RETRIEVER_UNSET,
        code_retriever: object = RETRIEVER_UNSET,
    ) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[object], None]] = []
        self._exit_requested = False
        # Состояние проекта: история, память, задачи, расписание и отчёты живут в поддереве,
        # привязанном к пути целевого репозитория, — инструмент запускают из корня проекта, и
        # показывать в нём состояние другого проекта нельзя.
        self._project_paths = config.project_paths(self.root)
        self._history = (
            history if history is not None else HistoryManager(path=self._project_paths["history"])
        )
        self._selection = domains.select_domain(
            root=self.root, explicit=domain_id, domains_dir=domains_dir
        )
        self._domains_dir = domains_dir
        # Фабрика клиентов MCP: тесты подменяют её, чтобы не поднимать серверные процессы.
        self._mcp_client_factory = mcp_client_factory or MCPClient
        # Ретривер по документации собирается по данным домена и до агента: агент получает его
        # готовым, иначе пришлось бы менять ретривер у уже созданного агента. Явный `None`
        # означает «поиска не будет» — так тесты и прогоны без сети отключают корпус целиком.
        self._docs_retriever = (
            self._build_docs_retriever()
            if docs_retriever is RETRIEVER_UNSET
            else docs_retriever
        )
        # Корпус кода: индекс читается по пути кэша, поиск локальный. Явный `None` означает
        # «поиска по коду в этой сессии не будет» — так тесты отключают корпус целиком.
        self._code_retriever = (
            self._build_code_retriever()
            if code_retriever is RETRIEVER_UNSET
            else code_retriever
        )
        self._mcp_specs: Tuple[MCPServerSpec, ...] = ()
        self._mcp_connections: Tuple[MCPConnection, ...] = ()
        # Снимок последней сборки корпуса кода: нужен, чтобы отчёт печатался без повторного чтения.
        self._last_code_report: Optional[code_index.IndexReport] = None
        self._agent = RepoAgent(
            domain=self._selection.domain,
            client=client,
            root=self.root,
            settings=settings if settings is not None else AnswerSettings(),
            history=self._history,
            docs_retriever=self._docs_retriever,
            code_retriever=self._code_retriever,
            tool_hub=self._tool_hub(),
            task_provider=lambda: self._pipeline.state,
            long_term=long_term_memory.LongTermMemory(self._project_paths["memory"]),
            schedule=schedule_store.ScheduleStore(self._project_paths["schedule"]),
        )
        self._task_store = task_state.TaskStore(self._project_paths["task"])
        self._pipeline = task_pipeline.TaskPipeline(
            domain=self._selection.domain,
            root=self.root,
            store=self._task_store,
            ask=self._task_ask,
            prepare=self._task_prepare,
            build=self._task_build,
            # Отчёты задачи пишутся в поддерево проекта, а не в общий каталог состояния.
            tasks_dir=self._project_paths["tasks_dir"],
        )
        # Каталог выгрузок читает серверный процесс: он получает его из окружения приложения.
        os.environ.setdefault("FFAI_EXPORTS_DIR", str(self._project_paths["exports_dir"]))
        # Подтверждение записи в репозиторий даёт интерфейс; без него патчи не применяются.
        self._patch_confirmation: Optional[Callable[[str, str], bool]] = None
        # Конвейер операций: исполнитель команд и подтверждение необратимых шагов приходят
        # снаружи — так тесты проверяют решения, не запуская SDK.
        self._ops_runner: Optional[Callable] = None
        self._ops_confirmation: Optional[Callable[[str, str], bool]] = None
        self._ops: Optional[aurora_ops.AuroraOps] = None
        # Опросник профиля и подтверждения — то, что умеет только интерфейс.
        self._profile_interview: Optional[Callable[[], None]] = None

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
                if not config.is_local_model(self._agent.model):
                    raise APIError(
                        "Нет пригодного ключа OPENCODE_API_KEY: облачная модель без него "
                        "недоступна (локальной модели ключ не нужен)."
                    )
                # Локальному пресету ключ не нужен: клиент для него не проверяет ключ и не
                # отправляет заголовок авторизации.
                api_key = ""
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
            if command == "/rag-docs":
                return self._docs_command(argument)
            if command == "/rag-code":
                return self._code_command(argument)
            if command == "/mcp":
                return self._mcp_command(argument)
            if command == "/tool":
                return self._tool_command(argument)
            if command == "/task":
                return self._task_command(argument)
            if command == "/ops":
                return self._ops_command(argument)
            if command == "/memory":
                return self._memory_command(argument)
            if command == "/profile":
                return self._profile_command(argument)
            if command == "/schedule":
                return CommandResult(command="/schedule", lines=self.schedule_lines())
            if command == "/invariants":
                return CommandResult(command="/invariants", lines=self.invariants_lines())
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

    def _tool_hub(self) -> Optional[object]:
        """Доступ к инструментам по текущим снимкам подключений.

        Пока обход реестра не выполнялся, снимков нет — и автовызов не может предложить модели ни
        одного инструмента. После `/mcp refresh` хаб собирается заново: каталог обязан отражать
        то, что серверы ответили сейчас.
        """
        if not self._mcp_specs:
            return None
        return mcp_tools.ToolHub(self._mcp_specs, self._mcp_connections, self._mcp_client_factory)

    def _build_code_retriever(self) -> Optional[object]:
        """Собирает поиск по корпусу кода: индекс берётся из кэша состояния.

        Индекса может не быть — это законный случай: поиск тогда не выполняется, и агент не
        добавляет ни фрагментов, ни сообщений о состоянии.
        """
        corpus = getattr(self.domain, "corpus", None)
        if corpus is None:
            return None
        return code_retrieval.CodeRetriever(config.repo_index_file(self.root), corpus)

    def _build_docs_retriever(self) -> Optional[object]:
        """Собирает ретривер документации по данным домена; нет корпуса — нет ретривера.

        Домен без внешнего корпуса документации — законный случай: тогда агент не ищет ничего
        и не добавляет сообщений о поиске.
        """
        docs = getattr(self.domain, "docs", None)
        if docs is None:
            return None
        spec = next(
            (
                candidate
                for candidate in mcp_registry.registry(self.domain, self.root)
                if candidate.name == docs.server
            ),
            None,
        )
        if spec is None:
            return None
        from .docs_retrieval import DocsRetriever

        return DocsRetriever(
            self._mcp_client_factory,
            spec,
            docs,
            sdk_version=str(getattr(self.domain, "local_sdk_version", "") or ""),
        )

    def docs_report(self) -> Optional[object]:
        """Снимок последнего поиска по документации (или None, если поиска не было)."""
        with self._lock:
            return self._agent.docs_report()

    @property
    def last_citations(self):
        with self._lock:
            return self._agent.last_citations

    def tool_flow_report(self):
        """Снимок последнего флоу автовызова (или None, если флоу не выполнялся)."""
        with self._lock:
            return self._agent.tool_flow_report()

    def code_report(self) -> Optional[object]:
        """Снимок последнего поиска по корпусу кода (или None, если поиска не было)."""
        with self._lock:
            return self._agent.code_report()

    @property
    def code_enabled(self) -> bool:
        return self._agent.config.code_enabled

    @property
    def code_retrieval(self) -> str:
        return self._agent.config.code_retrieval

    @property
    def code_threshold(self) -> float:
        return self._agent.config.code_threshold

    @property
    def docs_enabled(self) -> bool:
        return self._agent.config.docs_enabled

    @property
    def docs_retrieval(self) -> str:
        return self._agent.config.docs_retrieval

    @property
    def docs_threshold(self) -> float:
        return self._agent.config.docs_threshold

    def docs_lines(self) -> Tuple[str, ...]:
        """Отчёт о поиске по документации: запрос, разделы, версия, доставленные фрагменты."""
        if not self.docs_enabled:
            return ("Поиск по документации выключен (/rag-docs mode on — включить).",)
        settings = f"Режим отбора: {self.docs_retrieval}; порог: {self.docs_threshold:.2f}"
        report = self.docs_report()
        if report is None:
            return (
                settings,
                "Поиска ещё не было: он выполняется перед каждым вопросом.",
            )
        lines = [
            f"Запрос: {getattr(report, 'query', '')}",
            settings,
            f"Разделы: {', '.join(getattr(report, 'sections', ()) or ()) or 'н/д'}; "
            f"версия документации: {getattr(report, 'version', '') or 'н/д'}"
            + (
                f"; локальный SDK: {getattr(report, 'sdk_version', '')}"
                if getattr(report, "sdk_version", "")
                else ""
            ),
            f"Состояние: {getattr(report, 'status', '')}"
            + (f" ({getattr(report, 'error', '')})" if getattr(report, "error", "") else ""),
        ]
        candidates = getattr(report, "candidates", ()) or ()
        if candidates:
            rated = bool(getattr(report, "rated", False))
            lines.append(
                f"Найдено кандидатов: {len(candidates)}"
                + (" (оценены)" if rated else " (оценка не выполнялась)")
            )
            for candidate in candidates[:5]:
                score = float(getattr(candidate, "score", 0.0) or 0.0)
                reason = str(getattr(candidate, "reason", "") or "")
                mark = f"{score:.2f} — " if rated else ""
                tail = f": {reason}" if reason else ""
                lines.append(
                    f"    {mark}{getattr(candidate, 'path', '')} — "
                    f"{getattr(candidate, 'title', '')} ({getattr(candidate, 'section', '')}){tail}"
                )
        fragments = getattr(report, "fragments", ()) or ()
        lines.append(f"Доставлено фрагментов: {len(fragments)}")
        for fragment in fragments:
            lines.append(
                f"    {getattr(fragment, 'identifier', '')} — раздел "
                f"{getattr(fragment, 'section', '')}, {len(getattr(fragment, 'text', '') or '')} симв."
                + (" (обрезан)" if getattr(fragment, "truncated", False) else "")
            )
        return tuple(lines)

    def _docs_command(self, argument: str) -> CommandResult:
        """`/rag-docs` — отчёт, `mode on|off`, `version <версия>`, `retrieval`, `threshold`, `trace`.

        `trace` — тот же отчёт: подробный вывод уже включает кандидатов, их оценки и причины.
        """
        argument = argument.strip()
        if argument.startswith("retrieval"):
            value = argument[len("retrieval") :].strip().lower()
            if value not in reranking.MODES:
                return CommandResult(
                    command="/rag-docs",
                    lines=(f"Форма: /rag-docs retrieval {'|'.join(reranking.MODES)}",),
                )
            self._agent.config.docs_retrieval = value
            note = (
                "кандидаты оцениваются моделью"
                if value == reranking.MODE_ENHANCED
                else "кандидаты доставляются без оценки"
            )
            return CommandResult(
                command="/rag-docs", lines=(f"Режим отбора: {value} — {note}.",) + self.docs_lines()
            )
        if argument.startswith("threshold"):
            value = argument[len("threshold") :].strip().replace(",", ".")
            try:
                parsed = float(value)
            except ValueError:
                return CommandResult(
                    command="/rag-docs", lines=("Форма: /rag-docs threshold <число от 0 до 1>",)
                )
            if not 0.0 <= parsed <= 1.0:
                return CommandResult(
                    command="/rag-docs",
                    lines=(f"Порог должен быть от 0 до 1, а не {parsed}.",),
                )
            self._agent.config.docs_threshold = parsed
            return CommandResult(command="/rag-docs", lines=(f"Порог отбора: {parsed:.2f}.",))
        if argument.startswith("mode"):
            value = argument[len("mode") :].strip().lower()
            if value not in ("on", "off"):
                return CommandResult(command="/rag-docs", lines=("Форма: /rag-docs mode on|off",))
            self._agent.config.docs_enabled = value == "on"
            state = "включён" if self.docs_enabled else "выключен"
            return CommandResult(
                command="/rag-docs",
                lines=(f"Поиск по документации {state}.",)
                + (self.docs_lines() if self.docs_enabled else ()),
            )
        if argument.startswith("version"):
            value = argument[len("version") :].strip()
            if self._docs_retriever is None:
                return CommandResult(
                    command="/rag-docs", lines=("Домен не объявляет корпус документации.",)
                )
            self._docs_retriever.version = value
            self._docs_retriever.refresh_versions()
            chosen = value or "актуальная по данным сервера"
            return CommandResult(command="/rag-docs", lines=(f"Версия документации: {chosen}.",))
        return CommandResult(command="/rag-docs", lines=self.docs_lines())

    # --- память, профиль и правила домена -----------------------------------------------------

    def set_profile_interview(self, runner) -> None:
        """Опросник профиля ведёт интерфейс: он задаёт вопросы и читает ответы."""
        self._profile_interview = runner

    def profile_questions(self) -> Tuple[Tuple[str, str, str], ...]:
        """Скрипт опросника: (машинное имя, вопрос, значение по умолчанию) — интерфейс печатает его."""
        domain_profile = getattr(self.domain, "profile", None)
        if domain_profile is None:
            return ()
        return tuple(
            (question.field, question.prompt, question.default)
            for question in user_profile.questions(domain_profile)
        )

    def save_profile(self, answers: Sequence[Tuple[str, str]]) -> str:
        """Сохраняет профиль из ответов опросника и делает его активным.

        Пустой ответ оставляет раздел как был (для нового профиля — как задал домен), поэтому
        повторная настройка — это правка, а не потеря данных.
        """
        domain_profile = getattr(self.domain, "profile", None)
        if domain_profile is None:
            return ""
        return self._agent.save_profile(domain_profile, answers)

    def agent_routing(self) -> tuple:
        """Записи памяти, сделанные последней репликой: журнальные строки интерфейса."""
        with self._lock:
            return self._agent.last_routing

    def agent_invariants(self):
        """Нарушения правил домена в последнем ответе (или None)."""
        with self._lock:
            return self._agent.last_invariants

    def memory_report(self) -> Dict[str, object]:
        with self._lock:
            return self._agent.memory_report()

    def profile_report(self) -> Dict[str, object]:
        with self._lock:
            return self._agent.profile_report()

    def invariants_report(self) -> Dict[str, object]:
        with self._lock:
            return self._agent.invariants_report()

    @property
    def project_name(self) -> str:
        """Имя каталога целевого проекта: по нему пользователь понимает, с чем работает."""
        return self.root.name

    @property
    def branch(self) -> str:
        """Активная ветка git (пустая строка — ветки нет)."""
        return repo.current_branch(self.root)

    def schedule_report(self):
        """Снимок расписания: задания, прогоны и объём собранного — без запуска заданий."""
        with self._lock:
            return self._agent.schedule_report()

    def schedule_startup_line(self) -> str:
        """Строка старта: сколько заданий и прогонов уже в файле расписания."""
        report = self.schedule_report()
        return f"🗓 Планировщик: {len(report.jobs)} заданий, {len(report.runs)} прогонов"

    def schedule_announcement(self) -> Tuple[str, ...]:
        """Строки объявления о прогонах, которых пользователь ещё не видел."""
        report = self._agent.last_schedule
        if report is None or not report.fresh:
            return ()
        lines = [
            f"🗓 Планировщик: прогонов {report.fresh_count}, "
            f"сбоев {report.failed}; собрано всего: {report.collected_total}"
        ]
        for run in report.fresh:
            mark = "✅" if getattr(run, "ok", True) else "⛔"
            summary = str(getattr(run, "summary", "") or "")
            lines.append(f"    {mark} задание {getattr(run, 'number', '')}: {summary}")
        return tuple(lines)

    def schedule_lines(self) -> Tuple[str, ...]:
        """Отчёт о расписании: задания, последние прогоны и накопленное."""
        report = self.schedule_report()
        lines = [
            f"Заданий: {len(report.jobs)}; прогонов: {len(report.runs)}; "
            f"накоплено записей: {report.collected_total}",
            f"Хранилище: {report.path or 'нет пути'}",
        ]
        if report.jobs:
            lines.append("Задания:")
            for job in report.jobs:
                lines.append(
                    f"    {job.number}. {job.tool} каждые {job.every_minutes} мин; "
                    f"прогонов {job.runs}"
                )
        else:
            lines.append("Заданий нет: их ставит модель автовызовом или /tool call.")
        if report.runs:
            lines.append("Последние прогоны:")
            for run in report.runs[-5:]:
                mark = "✅" if getattr(run, "ok", True) else "⛔"
                lines.append(
                    f"    {mark} задание {run.number}: "
                    f"{getattr(run, 'summary', '')} (+{getattr(run, 'fresh', 0)})"
                )
        return tuple(lines)

    def memory_lines(self) -> Tuple[str, ...]:
        """Отчёт о слоях памяти: обмены, рабочая память, долговременные записи и правила."""
        report = self.memory_report()
        lines = [f"Обменов в диалоге: {report['exchanges']}"]
        working = report["working"]
        lines.append("Рабочая память задачи:")
        if working:
            lines.extend(f"    {key}: {value}" for key, value in sorted(working.items()))
        else:
            lines.append("    пусто")
        long_term = report["long_term"]
        lines.append(f"Долговременная память ({report['long_term_path'] or 'нет пути'}):")
        if long_term:
            lines.extend(
                f"    {record.key} [{record.category}]: {record.value}" for record in long_term
            )
        else:
            lines.append("    пусто")
        lines.append("Правила маршрутизации:")
        lines.extend(f"    {rule}" for rule in report["rules"])
        return tuple(lines)

    def _memory_command(self, argument: str) -> CommandResult:
        """`/memory` — отчёт, `goal|remember|forget` — операции со свободным текстом."""
        tokens = argument.split(maxsplit=1)
        action = tokens[0].lower() if tokens else ""
        rest = tokens[1].strip() if len(tokens) > 1 else ""
        if not action:
            return CommandResult(command="/memory", lines=self.memory_lines())
        if action == "goal":
            if not rest:
                return CommandResult(command="/memory", lines=("Нужна цель: /memory goal <текст>",))
            self._agent.set_goal(rest)
            return CommandResult(
                command="/memory", lines=(f"Цель задачи: {rest}",) + self.memory_lines()
            )
        if action == "remember":
            if not rest:
                return CommandResult(command="/memory", lines=("Нужен текст: /memory remember <текст>",))
            record = self._agent.remember(rest)
            where = memory_layers.LAYER_LABELS.get(record.layer, record.layer)
            return CommandResult(
                command="/memory",
                lines=(f"Запомнено ({where}, {record.key}): {record.value}",),
            )
        if action == "forget":
            if not rest:
                return CommandResult(
                    command="/memory", lines=("Нужен ключ или all: /memory forget <ключ|all>",)
                )
            removed = self._agent.forget(rest)
            return CommandResult(
                command="/memory",
                lines=(f"Удалено записей: {removed}.",) + self.memory_lines(),
            )
        return CommandResult(
            command="/memory",
            lines=("Форма: /memory, /memory goal <текст>, /memory remember <текст>, /memory forget <ключ|all>",),
        )

    def profile_lines(self) -> Tuple[str, ...]:
        """Отчёт о профилях: активный, его разделы и имена сохранённых профилей."""
        report = self.profile_report()
        name = report["active"] or "нет"
        lines = [f"Активный профиль: {name}", f"Хранилище: {report['path'] or 'нет пути'}"]
        if report["sections"]:
            lines.append("Разделы:")
            lines.extend(f"    {label}: {value or '—'}" for label, value in report["sections"])
        names = ", ".join(report["names"]) if report["names"] else "нет"
        lines.append(f"Сохранённые профили: {names}")
        return tuple(lines)

    def _profile_command(self, argument: str) -> CommandResult:
        """`/profile` — отчёт, `setup` — опросник, `use <имя>`, `forget <имя>`."""
        tokens = argument.split(maxsplit=1)
        action = tokens[0].lower() if tokens else ""
        rest = tokens[1].strip() if len(tokens) > 1 else ""
        if not action:
            return CommandResult(command="/profile", lines=self.profile_lines())
        if action == "setup":
            if self._profile_interview is None:
                return CommandResult(
                    command="/profile",
                    lines=("Опросник профиля ведёт интерфейс: в этом режиме он недоступен.",),
                )
            self._profile_interview()
            return CommandResult(command="/profile", lines=("Профиль настроен.",) + self.profile_lines())
        if action == "use":
            if not rest:
                return CommandResult(command="/profile", lines=("Нужно имя: /profile use <имя>",))
            if not self._agent.profiles.use(rest):
                return CommandResult(
                    command="/profile",
                    lines=(f"Профиля «{rest}» нет: выбор несуществующего профиля его не создаёт.",),
                )
            return CommandResult(
                command="/profile", lines=(f"Активный профиль: {rest}.",) + self.profile_lines()
            )
        if action == "forget":
            if not rest:
                return CommandResult(command="/profile", lines=("Нужно имя: /profile forget <имя>",))
            removed = self._agent.profiles.forget(rest)
            return CommandResult(
                command="/profile",
                lines=(
                    (f"Профиль «{rest}» удалён." if removed else f"Профиля «{rest}» нет."),
                )
                + self.profile_lines(),
            )
        return CommandResult(
            command="/profile",
            lines=("Форма: /profile, /profile setup, /profile use <имя>, /profile forget <имя>",),
        )

    def invariants_lines(self) -> Tuple[str, ...]:
        """Отчёт о правилах домена: таблица правил и результат последней проверки."""
        report = self.invariants_report()
        rules = report["rules"]
        lines = [f"Правил домена: {len(rules)}"]
        for rule in rules:
            forbid = ", ".join(rule.forbidden) if rule.forbidden else "проверяет промпт"
            lines.append(f"    {rule.number}. {rule.rule} (запрещено: {forbid})")
        check = report["check"]
        if check:
            lines.append("Последняя проверка: нарушения —")
            lines.extend(f"    {item.number}. {item.rule} («{item.term}»)" for item in check)
        else:
            lines.append("Последняя проверка: нарушений не было.")
        return tuple(lines)

    # --- конвейер операций: SDK, цель, сборка, подпись, установка, запуск --------------------

    def set_ops_runner(self, runner) -> None:
        """Интерфейс передаёт исполнителя команд: в приложении это `subprocess`, в тестах — заглушка."""
        self._ops_runner = runner
        self._ops = None

    def set_ops_confirmation(self, callback: Optional[Callable[[str, str], bool]]) -> None:
        """Подтверждение необратимых шагов: подпись, установка и запуск без него не выполняются."""
        self._ops_confirmation = callback
        self._ops = None

    def ops_pipeline(self) -> Optional[aurora_ops.AuroraOps]:
        """Конвейер операций по правилам домена; домен без правил — конвейера нет."""
        rules = getattr(self.domain, "ops", None)
        if rules is None:
            return None
        if self._ops is None:
            self._ops = aurora_ops.AuroraOps(
                root=self.root,
                ops=rules,
                run=self._ops_runner,
                confirm=self._ops_confirmation,
                env=dict(os.environ),
                docs_version=getattr(self.domain, "docs_version", ""),
                app_id=self._app_id(),
            )
        return self._ops

    def _app_id(self) -> str:
        """Идентификатор приложения: имя файла по образцам домена, иначе имя каталога проекта."""
        rules = getattr(self.domain, "ops", None)
        patterns = getattr(rules, "app_id_globs", ()) if rules is not None else ()
        for pattern in patterns:
            found = sorted(self.root.glob(pattern))
            if found:
                return found[0].stem
        return self.root.name

    def _ops_command(self, argument: str) -> CommandResult:
        """`/ops` — состояние, `target <архитектура>`, `build|sign|verify|install|run`, `refresh`."""
        ops = self.ops_pipeline()
        if ops is None:
            return CommandResult(command="/ops", lines=("Домен не объявляет конвейер операций.",))
        tokens = argument.split()
        action = tokens[0].lower() if tokens else ""
        if not action or action == "status":
            return CommandResult(command="/ops", lines=ops.lines())
        if action == "refresh":
            status = ops.status(refresh=True)
            return CommandResult(command="/ops", lines=("Список целей обновлён.",) + ops.lines(status))
        if action == "target":
            value = tokens[1] if len(tokens) > 1 else ""
            report = ops.select_target(value)
            lines = (report.line(),)
            if not value and not report.reason:
                lines = ("Выбрана цель по архитектуре домена.",)
            return CommandResult(command="/ops", lines=lines + ops.lines())
        steps = {
            "build": ops.build,
            "sign": ops.sign,
            "verify": ops.verify,
            "install": ops.install,
            "run": ops.run_app,
        }
        if action not in steps:
            return CommandResult(
                command="/ops",
                lines=(
                    "Форма: /ops [status|refresh], /ops target <архитектура>, "
                    "/ops build|sign|verify|install|run",
                ),
            )
        report = steps[action]()
        return CommandResult(command="/ops", lines=(report.line(),) + ops.lines())

    def ops_report(self):
        """Снимок конвейера: состояние и шаги — интерфейс печатает его, не читая внутренности."""
        return self.ops_pipeline()

    # --- задача: прогон, подтверждение патчей, отчёты ---------------------------------------

    def set_patch_confirmation(self, callback: Optional[Callable[[str, str], bool]]) -> None:
        """Интерфейс передаёт сюда свой способ спросить пользователя о записи в репозиторий."""
        self._patch_confirmation = callback

    def _task_ask(self, messages, max_words: int, phase: str) -> AnswerMeta:
        """Один запрос этапа задачи: та же модель сессии, свой предел длины, общий учёт расхода."""
        client = self._require_client()
        try:
            self._emit(PhaseChanged(RequestPhase(phase)))
        except ValueError:
            pass
        meta = client.ask_with_usage_messages(
            messages,
            max_tokens=config.max_tokens_for_words(max_words),
            temperature=None,
            model=self._agent.model,
        )
        self._agent.record_usage(meta)
        return meta

    def _task_build(self, command: str) -> Tuple[bool, str]:
        """Команда сборки выполняется инструментом сервера репозитория: белый список — там.

        Своего раннера команд приложение не заводит: два места с белым списком разошлись бы, и
        второе было бы дырой.
        """
        pipeline = self.ops_pipeline()
        if pipeline is not None and pipeline.state.target and self._ops_runner is not None:
            # Сборка идёт конвейером операций: отдельный каталог сборки и настоящий SDK.
            report = pipeline.build()
            if report.status == aurora_ops.STATUS_UNAVAILABLE:
                raise domain_checks.BuildUnavailable(report.reason)
            return report.ok, report.output or report.reason
        checks = getattr(self.domain, "checks", None)
        tool_name = getattr(checks, "build_tool", "") if checks is not None else ""
        if not tool_name:
            raise domain_checks.BuildUnavailable("домен не объявил инструмент сборки")
        hub = self._tool_hub()
        if hub is None:
            raise domain_checks.BuildUnavailable("серверы инструментов не подключены")
        try:
            route = mcp_tools.route(hub.views(), "", tool_name)
            if route.error:
                raise domain_checks.BuildUnavailable(route.error)
            result = hub.call(route.server, tool_name, {"command": command})
            return (not result.is_error), result.text
        except domain_checks.BuildUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 - причину показываем текстом
            raise domain_checks.BuildUnavailable(str(error)) from error
        finally:
            hub.close()

    def _task_prepare(self, index: int, patch_text: str) -> task_state.TaskPatch:
        """Готовит патч к записи: проверки домена, подтверждение пользователя, применение.

        Порядок именно такой: непроверенный патч не показывается как «можно применять», а
        непринятый не трогает файлы вовсе.
        """
        checks = getattr(self.domain, "checks", None)
        summary = ""
        try:
            summary = patches.parse(patch_text).summary()
        except patches.PatchError as error:
            return task_state.TaskPatch(
                index=index, text=patch_text, summary="", applied=False, reason=str(error)
            )
        if checks is not None:
            # На этапе выполнения проверяются правила содержимого и применение: сборку запускает
            # этап проверки, а не каждый патч.
            result = domain_checks.check_patch(
                patch_text, self.root, domain_checks.without_build(checks)
            )
            if not result.ok:
                return task_state.TaskPatch(
                    index=index,
                    text=patch_text,
                    summary=summary,
                    applied=False,
                    reason="; ".join(issue.line() for issue in result.issues),
                )
        if self._patch_confirmation is None:
            return task_state.TaskPatch(
                index=index,
                text=patch_text,
                summary=summary,
                applied=False,
                reason="применение не подтверждено",
            )
        if not self._patch_confirmation(summary, patch_text):
            return task_state.TaskPatch(
                index=index,
                text=patch_text,
                summary=summary,
                applied=False,
                reason="пользователь отказался применять патч",
            )
        reason = patches.apply(self.root, patch_text)
        return task_state.TaskPatch(
            index=index,
            text=patch_text,
            summary=summary,
            applied=not reason,
            reason=reason,
        )

    def _task_command(self, argument: str) -> CommandResult:
        """`/task` — отчёт, `/task add <цель>`, `/task run`, `/task stage <этап>`, `/task stop`."""
        tokens = argument.split(maxsplit=1)
        action = tokens[0].lower() if tokens else ""
        rest = tokens[1].strip() if len(tokens) > 1 else ""
        if not action:
            return CommandResult(command="/task", lines=self.task_lines())
        if action == "add":
            if not rest:
                return CommandResult(
                    command="/task", lines=("Нужна цель: /task add <что сделать>",)
                )
            self._pipeline.add(rest)
            return CommandResult(
                command="/task", lines=("Задача добавлена.",) + self.task_lines()
            )
        if action == "run":
            if not self._pipeline.unfinished():
                return CommandResult(
                    command="/task",
                    lines=("Очередь пуста — выполнять нечего. Добавьте задачу: /task add <цель>",),
                )
            return CommandResult(
                command="/task", lines=("Запускаю прогон задачи.",), task_run=True
            )
        if action == "stage":
            if not rest:
                return CommandResult(
                    command="/task",
                    lines=("Нужен этап: /task stage planning|execution|validation|done",),
                )
            result = self._pipeline.request_stage(rest)
            if result.accepted:
                return CommandResult(
                    command="/task",
                    lines=(f"Этап: {task_state.STAGE_TITLES[result.stage]}.",) + self.task_lines(),
                )
            allowed = ", ".join(
                task_state.STAGE_TITLES[stage] for stage in result.allowed
            )
            return CommandResult(
                command="/task",
                lines=(
                    f"Переход не выполнен: {result.reason}.",
                    f"Допустимые переходы: {allowed or 'нет'}.",
                ),
            )
        if action == "stop":
            self._pipeline.stop()
            return CommandResult(command="/task", lines=("Очередь задач остановлена.",))
        return CommandResult(
            command="/task",
            lines=(
                "Форма: /task, /task add <цель>, /task run, /task stage <этап>, /task stop",
            ),
        )

    def task_lines(self) -> Tuple[str, ...]:
        """Отчёт о задаче и очереди: этап, шаг, ожидаемое действие, план с отметками, замечания."""
        state = self._pipeline.state
        if not state.tasks:
            return (
                "Задач нет. Добавить: /task add <цель>; выполнить: /task run.",
                f"Состояние: {self._task_store.path}",
            )
        lines: List[str] = [
            f"Очередь: задач {len(state.tasks)}, незавершённых {len(state.unfinished())}; "
            + ("прогон на паузе" if state.paused else "прогон не на паузе")
        ]
        task = state.current
        if task is not None:
            lines.append(
                f"Задача {task.number} ({task.status}): {task.goal}"
            )
            lines.append(f"Этап: {task.title}; шаг: {task.current_step}")
            lines.append(f"Ожидается: {task.expected_action}")
            for index, item in enumerate(task.plan, start=1):
                patch = task.patch_for(index)
                mark = "x" if patch is not None and patch.applied else " "
                note = ""
                if patch is not None and not patch.applied:
                    note = f" — {patch.reason or 'не применён'}"
                lines.append(f"    [{mark}] {index}. {item}{note}")
            if task.issues:
                lines.append("Замечания:")
                lines.extend(f"    {issue.line()}" for issue in task.issues)
            if task.transitions:
                lines.append("Журнал переходов:")
                lines.extend(f"    {entry.line()}" for entry in task.transitions[-5:])
            if task.result_path:
                lines.append(f"Отчёт: {task.result_path}")
            if task.fail_reason:
                lines.append(f"Причина неудачи: {task.fail_reason}")
        if state.paused:
            lines.append("Продолжить: /task run.")
        return tuple(lines)

    def task_step(self):
        """Одна операция прогона: интерфейс вызывает её и рисует панель."""
        with self._lock:
            return self._pipeline.step()

    def task_answer_edits(self, text: str) -> None:
        with self._lock:
            self._pipeline.answer_edits(text)

    def task_set_paused(self, paused: bool) -> None:
        with self._lock:
            self._pipeline.set_paused(paused)

    @property
    def task_paused(self) -> bool:
        return self._pipeline.state.paused

    @property
    def task_state(self):
        return self._pipeline.state

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
            self._agent.tool_hub = self._tool_hub()
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

    # --- корпус кода: сборка, состояние, сравнение стратегий --------------------------------

    def _code_command(self, argument: str) -> CommandResult:
        """`/rag-code index [стратегия]`, `/rag-code status`, `/rag-code compare` — без обращений к модели.

        Сборка синхронная: индекс небольшого проекта собирается мгновенно, а большой проект
        пользователь собирает осознанно. Ошибка сборки не затирает прежний индекс — она просто
        видна строкой отчёта.
        """
        tokens = argument.split()
        action = tokens[0] if tokens else "status"
        corpus = getattr(self.domain, "corpus", None)
        if corpus is None:
            return CommandResult(
                command="/rag-code", lines=("Домен не объявляет корпус кода.",)
            )
        if action == "retrieval":
            return self._code_retrieval_command(tokens[1:])
        if action == "threshold":
            return self._code_threshold_command(tokens[1:])
        if action == "tune":
            return self._code_tune_command(tokens[1:])
        if action == "mode":
            return self._code_mode_command(tokens[1:])
        if action == "trace":
            return CommandResult(command="/rag-code", lines=self.code_trace_lines())
        if action == "index":
            strategy = tokens[1] if len(tokens) > 1 else corpus.strategy
            try:
                report = code_index.build_index(
                    self.root, corpus, strategy, config.repo_index_file(self.root)
                )
            except code_index.CodeIndexError as error:
                return CommandResult(
                    command="/rag-code",
                    lines=(
                        f"Индекс не собран: {error}",
                        "Прежний индекс (если был) остался на месте.",
                    ),
                )
            self._last_code_report = report
            return CommandResult(command="/rag-code", lines=self.code_lines())
        if action == "compare":
            return CommandResult(command="/rag-code", lines=self._code_compare_lines(corpus))
        if action != "status":
            return CommandResult(
                command="/rag-code",
                lines=(
                    "Форма команды: /rag-code index [fixed|structural], /rag-code status, /rag-code compare, "
                    "/rag-code retrieval baseline|enhanced, /rag-code threshold <0..1>, "
                    "/rag-code tune before=<n> after=<n>, /rag-code mode on|off, /rag-code trace",
                ),
            )
        return CommandResult(command="/rag-code", lines=self.code_lines())

    def _code_retrieval_command(self, tokens: List[str]) -> CommandResult:
        """`/rag-code retrieval baseline|enhanced` — режим отбора кандидатов корпуса кода."""
        value = tokens[0].lower() if tokens else ""
        if value not in reranking.MODES:
            return CommandResult(
                command="/rag-code", lines=(f"Форма: /rag-code retrieval {'|'.join(reranking.MODES)}",)
            )
        self._agent.config.code_retrieval = value
        note = (
            "вопрос переформулируется, кандидаты оцениваются моделью"
            if value == reranking.MODE_ENHANCED
            else "поиск по исходному вопросу, первые результаты без оценки"
        )
        return CommandResult(
            command="/rag-code", lines=(f"Режим отбора: {value} — {note}.",) + self.code_trace_lines()
        )

    def _code_threshold_command(self, tokens: List[str]) -> CommandResult:
        """`/rag-code threshold <число от 0 до 1>` — порог отбора кандидатов."""
        raw = (tokens[0].replace(",", ".") if tokens else "")
        try:
            parsed = float(raw)
        except ValueError:
            return CommandResult(
                command="/rag-code", lines=("Форма: /rag-code threshold <число от 0 до 1>",)
            )
        if not 0.0 <= parsed <= 1.0:
            return CommandResult(
                command="/rag-code", lines=(f"Порог должен быть от 0 до 1, а не {parsed}.",)
            )
        self._agent.config.code_threshold = parsed
        return CommandResult(command="/rag-code", lines=(f"Порог отбора: {parsed:.2f}.",))

    def _code_tune_command(self, tokens: List[str]) -> CommandResult:
        """`/rag-code tune before=<n> after=<n>` — пулы кандидатов и фрагментов, целиком атомарно.

        Негодная пара не меняет ни одно поле: половинчатые настройки опаснее отказа, потому что
        выглядят применёнными.
        """
        form = "Форма: /rag-code tune before=<кандидатов> after=<фрагментов>"
        values = {}
        for token in tokens:
            key, separator, value = token.partition("=")
            if not separator or key not in ("before", "after"):
                return CommandResult(command="/rag-code", lines=(form,))
            try:
                values[key] = int(value)
            except ValueError:
                return CommandResult(command="/rag-code", lines=(form,))
        if set(values) != {"before", "after"}:
            return CommandResult(command="/rag-code", lines=(form,))
        before, after = values["before"], values["after"]
        if before < 1 or after < 1 or after > before:
            return CommandResult(
                command="/rag-code",
                lines=(
                    f"Пул кандидатов {before}, фрагментов {after}: нужно before ≥ 1, "
                    "1 ≤ after ≤ before. Настройки не изменены.",
                ),
            )
        self._agent.config.code_before = before
        self._agent.config.code_after = after
        return CommandResult(
            command="/rag-code",
            lines=(f"Пулы: до {before} кандидатов, до {after} фрагментов.",) + self.code_trace_lines(),
        )

    def _code_mode_command(self, tokens: List[str]) -> CommandResult:
        """`/rag-code mode on|off` — поиск по корпусу кода целиком."""
        value = tokens[0].lower() if tokens else ""
        if value not in ("on", "off"):
            return CommandResult(command="/rag-code", lines=("Форма: /rag-code mode on|off",))
        self._agent.config.code_enabled = value == "on"
        state = "включён" if self._agent.config.code_enabled else "выключен"
        return CommandResult(
            command="/rag-code",
            lines=(f"Поиск по корпусу кода {state}.",)
            + (self.code_trace_lines() if self._agent.config.code_enabled else ()),
        )

    def code_trace_lines(self) -> Tuple[str, ...]:
        """Отчёт о последнем поиске по корпусу кода: запрос, режим, оценки, доставленное."""
        config_ = self._agent.config
        if not config_.code_enabled:
            return ("Поиск по корпусу кода выключен (/rag-code mode on — включить).",)
        settings = (
            f"Режим отбора: {config_.code_retrieval}; порог: {config_.code_threshold:.2f}; "
            f"пулы: до {config_.code_before} кандидатов, до {config_.code_after} фрагментов"
        )
        report = self._agent.code_report()
        if report is None:
            return (settings, "Поиска ещё не было: он выполняется перед каждым вопросом.")
        lines = [
            f"Вопрос: {report.question}",
            f"Поисковый запрос: {report.query}",
            settings,
            f"Индекс: {report.database}; фрагментов в индексе: {report.chunks}",
            f"Ранжирование: {report.ranking or 'не выполнялось'}"
            + (f" — {report.ranking_note}" if report.ranking_note else ""),
            f"Состояние: {report.status}"
            + (f" ({report.error})" if report.error else ""),
            f"Кандидатов: {len(report.candidates)}"
            + (" (оценены)" if report.rated else " (оценка не выполнялась)"),
        ]
        lines.extend(code_retrieval.rate_lines(report.candidates[:5]))
        lines.append(f"Доставлено фрагментов: {len(report.fragments)}")
        for fragment in report.fragments:
            lines.append(
                f"    {fragment.identifier} — {len(fragment.text)} симв."
                + (" (обрезан)" if fragment.truncated else "")
            )
        return tuple(lines)

    def code_lines(self) -> Tuple[str, ...]:
        """Отчёт о корпусе кода: путь индекса, стратегия, время сборки, файлы, фрагменты, пропуски."""
        if getattr(self.domain, "corpus", None) is None:
            return ("Домен не объявляет корпус кода.",)
        report = self._last_code_report
        database = config.repo_index_file(self.root)
        if report is None or report.database != database:
            report = code_index.read_index(database)
        if report is None:
            return (
                f"Индекса нет. Путь: {database}",
                "Собрать: /rag-code index [fixed|structural]; сравнить стратегии: /rag-code compare",
            )
        lines = [
            f"Индекс: {report.database}",
            f"Стратегия: {report.strategy}; собран: {report.built_at_text}; "
            f"список файлов: {report.source}",
            f"Файлов: {report.files}; фрагментов: {report.chunks}; "
            f"строк во фрагментах: {report.lines}; объём: {report.size_bytes} симв.",
        ]
        reasons = report.skip_reasons()
        if reasons:
            summary = ", ".join(f"{reason} — {count}" for reason, count in sorted(reasons.items()))
            lines.append(f"Пропущено: {summary}")
        return tuple(lines)

    def _code_compare_lines(self, corpus) -> Tuple[str, ...]:
        """Сравнение стратегий разбиения: число фрагментов и примеры границ, без записи индекса."""
        lines: List[str] = ["Сравнение стратегий разбиения:"]
        for strategy in code_index.STRATEGIES:
            chunks = code_index.collect(self.root, corpus, strategy)
            lines.append(f"    {strategy}: фрагментов — {len(chunks)}")
            for label in code_index.sample_labels(chunks):
                lines.append(f"        {label}")
        lines.append("Индекс не изменялся: /rag-code index <стратегия> собирает выбранную.")
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

    def tool_flow_lines(self) -> Tuple[str, ...]:
        """Отчёт о последнем флоу автовызова: раунды, шаги с объёмами передачи, причина остановки.

        Строится из снимка: ни модели, ни серверных процессов. Журнальные строки шагов здесь те же,
        что печатает интерфейс, — формат у них один.
        """
        state = "включён" if self._agent.config.auto_tools else "выключен"
        lines: List[str] = [f"Автовызов инструментов: {state}"]
        report = self._agent.tool_flow_report()
        if report is None:
            lines.append(
                "Флоу не выполнялся: он идёт перед каждым вопросом, когда есть что выбирать."
            )
            return tuple(lines)
        lines.append(f"Вопрос: {report.question}")
        lines.append(
            f"Раундов: {report.rounds}; запросов выбора: {report.choice_requests}; "
            f"шагов: {report.executed}"
        )
        if report.dropped:
            lines.append(f"Отброшено шагов сверх предела: {report.dropped}")
        lines.append(f"Остановка: {report.stop_reason}")
        lines.extend(f"    {step.journal_line()}" for step in report.steps)
        return tuple(lines)

    def _tool_command(self, argument: str) -> CommandResult:
        """`/tool` — перечень, `/tool call` — ручной вызов, `/tool auto` и `/tool flow` — автовызов.

        Строка разбирается как командная: значение с пробелами берётся в кавычки —
        `/tool call <инструмент> query="поисковый запрос"`. Имена инструментов ядру неизвестны:
        они приходят от серверов, поэтому пример здесь обезличен.
        """
        tokens = _split_command_line(argument)
        if not tokens:
            return CommandResult(command="/tool", lines=self.tool_lines())
        if tokens[0] == "auto":
            value = tokens[1].lower() if len(tokens) > 1 else ""
            if value not in ("on", "off"):
                return CommandResult(command="/tool", lines=("Форма: /tool auto on|off",))
            self._agent.config.auto_tools = value == "on"
            state = "включён" if self._agent.config.auto_tools else "выключен"
            return CommandResult(
                command="/tool",
                lines=(
                    f"Автовызов инструментов {state}.",
                    "Выключенный автовызов не делает ни запроса выбора, ни вызовов инструментов.",
                ),
            )
        if tokens[0] == "flow":
            return CommandResult(command="/tool", lines=self.tool_flow_lines())
        if tokens[0] != "call":
            return CommandResult(
                command="/tool",
                lines=(
                    "Форма команды: /tool — перечень, /tool call <инструмент> ключ=значение …, "
                    "/tool auto on|off, /tool flow",
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
