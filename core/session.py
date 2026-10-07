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
from typing import Callable, List, Optional, Tuple

from . import config, domains
from .agent import CompressionReport, ContextReport, FactsReport, RepoAgent, RequestPhase
from .answer_settings import AnswerSettings
from .api_client import APIClient, APIError, AnswerMeta, is_valid_api_key
from .domains import Domain, DomainError, DomainSelection
from .history_manager import HistoryManager
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
