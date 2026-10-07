"""Агент: единственное место, где решается, что и как уходит модели.

Агент владеет логом ходов сессии (append-only: его не чистит даже сжатие — стратегия лишь
выбирает, что из лога отправить), памятью активной стратегии (резюме, блок фактов, ветки),
конвертом истории на диске и накопителем расхода сессии. Терминальный слой не заглядывает
внутрь: он получает снимки (`context_report`) и метрики (`last_result`).

Предметной области здесь нет: роль и правила приходят пакетом домена, адрес и модель — из
конфигурации. Инварианты домена подключаются в фазе P8, извлечение знаний о платформе —
в P2–P4; этот модуль занимается контекстом и запросом.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import (
    citations,
    code_retrieval,
    config,
    mcp_tools,
    context_compressor,
    context_strategies,
    invariants,
    long_term_memory,
    memory_layers,
    prompts,
    reranking,
    schedule_store,
    task_state,
    user_profile,
)
from .answer_settings import AnswerFormat, AnswerSettings, ContextStrategy
from .api_client import APIClient, APIError, AnswerMeta
from .domains import Domain
from .history_manager import HistoryManager
from .mcp_client import MCPError
from .usage import SessionLedger, SessionUsage, estimate_tokens


class RequestPhase(Enum):
    """Фаза запроса — для спиннера интерфейса: что именно сейчас происходит."""

    REQUEST = "request"
    COMPRESSION = "compression"
    FACTS_UPDATE = "facts_update"
    MCP_CONNECT = "mcp_connect"
    MCP_TOOL = "mcp_tool"
    DOCS_RERANK = "docs_rerank"
    CODE_QUERY = "code_query"
    CODE_RERANK = "code_rerank"
    TOOL_CHOICE = "tool_choice"
    TASK_PLAN = "task_plan"
    TASK_EXECUTE = "task_execute"
    TASK_VALIDATE = "task_validate"


@dataclass
class CompressionReport:
    """Что сделало сжатие: сколько сообщений и обменов свёрнуто и какой это был запрос."""

    messages: int
    exchanges: int
    meta: Optional[AnswerMeta] = None


@dataclass
class ScheduleReport:
    """Снимок расписания: задания, прогоны, объём собранного и прогоны, о которых не говорили."""

    jobs: Tuple[object, ...] = ()
    runs: Tuple[object, ...] = ()
    collected_total: int = 0
    path: str = ""
    fresh: Tuple[object, ...] = ()

    @property
    def fresh_count(self) -> int:
        return len(self.fresh)

    @property
    def failed(self) -> int:
        return sum(1 for run in self.fresh if not getattr(run, "ok", True))


@dataclass
class FactsReport:
    """Итог обновления блока фактов: удалось ли, сколько ключей, какая ошибка."""

    updated: bool
    keys: int
    error: Optional[str] = None


@dataclass
class ContextReport:
    """Снимок состояния контекста для отчётов интерфейса."""

    strategy: ContextStrategy
    window: int
    max_session_tokens: int
    log_exchanges: int
    request_turns: int
    summary_covers: int
    has_summary: bool
    facts: Dict[str, str]
    branch: str
    branches: Tuple[Tuple[str, int], ...]
    tokens_estimate: int


@dataclass
class AgentConfig:
    """Конфиг сессии: настройки ответа и выбранная модель."""

    settings: AnswerSettings = field(default_factory=AnswerSettings)
    model: str = config.DEFAULT_MODEL
    # Поиск по документации портала — решение сессии (как модель и настройки ответа): выключенный
    # режим не делает ни одного обращения к серверу и не добавляет ни одного сообщения.
    docs_enabled: bool = True
    # Режим отбора кандидатов документации: enhanced оценивает их моделью, baseline — нет.
    docs_retrieval: str = config.DOCS_RETRIEVAL_MODE
    docs_threshold: float = config.DOCS_RELEVANCE_THRESHOLD
    # Корпус кода: поиск локальный, поэтому выключение убирает доставку фрагментов и сообщений,
    # а режим с порогом управляют только отбором.
    code_enabled: bool = True
    code_retrieval: str = config.CODE_RETRIEVAL_MODE
    code_threshold: float = config.CODE_RELEVANCE_THRESHOLD
    code_before: int = config.CODE_CANDIDATES_BEFORE
    code_after: int = config.CODE_FRAGMENTS_AFTER
    # Автовызов инструментов: одно вспомогательное обращение на вопрос, когда есть что выбирать.
    auto_tools: bool = config.AUTO_TOOLS

    @property
    def format(self):
        return self.settings.format

    @property
    def strategy(self) -> ContextStrategy:
        return self.settings.context_strategy

    @property
    def max_words(self) -> int:
        return self.settings.max_words

    @property
    def list_limit(self) -> int:
        return self.settings.list_limit

    @property
    def temperature(self) -> float:
        return self.settings.temperature


class RepoAgent:
    """Вопрос -> контекст по активной стратегии -> сборка сообщений -> модель -> метрики."""

    def __init__(
        self,
        domain: Domain,
        client: Optional[APIClient],
        root: Path,
        settings: Optional[AnswerSettings] = None,
        model: Optional[str] = None,
        history: Optional[HistoryManager] = None,
        docs_retriever: Optional[object] = None,
        code_retriever: Optional[object] = None,
        tool_hub: Optional[object] = None,
        task_provider: Optional[Callable[[], object]] = None,
        long_term: Optional[object] = None,
        profiles: Optional[object] = None,
        schedule: Optional[object] = None,
    ) -> None:
        self.domain = domain
        self.root = Path(root)
        self.client = client
        self.config = AgentConfig(
            settings=settings if settings is not None else AnswerSettings(),
            model=model if model is not None else config.DEFAULT_MODEL,
        )
        self.history = history if history is not None else HistoryManager()
        self._ledger = SessionLedger()
        # Лог ходов сессии: пары user/assistant успешных обменов, append-only.
        self._turns: List[Dict[str, str]] = []
        # Память стратегии резюме: дайджест, число покрытых им обменов файла и число обменов
        # лога, которые резюме накрывает сейчас (после рестарта лог содержит только несвёрнутый
        # хвост файла, поэтому счётчики расходятся).
        self._summary: Optional[str] = None
        self._summary_covers: int = 0
        self._log_covered: int = 0
        # Память стратегии фактов: блок «ключ — значение», очередь не переработанных сообщений
        # пользователя (сбой не должен их терять) и число обменов лога, уже отданных извлекателю.
        self._facts: Dict[str, str] = {}
        self._facts_pending: List[str] = []
        self._facts_covered: int = 0
        self._branches = context_strategies.BranchTree(self._turns)
        self._last_result: Optional[AnswerMeta] = None
        self._last_compression: Optional[CompressionReport] = None
        self._last_facts: Optional[FactsReport] = None
        # Поиск по документации: сам поиск делает ретривер (данные домена + сервер портала),
        # агент лишь решает, когда его звать, и что из найденного уходит в запрос.
        self.docs_retriever = docs_retriever
        self._last_docs_report: Optional[object] = None
        self._docs_fragments: tuple = ()
        # Корпус кода: ретривер получает готовый индекс, агент решает, когда искать, и что из
        # найденного уходит в запрос.
        self.code_retriever = code_retriever
        self._last_code_report: Optional[object] = None
        self._code_fragments: tuple = ()
        # Автовызов инструментов: каталог приходит из снимков подключений, флоу исполняет агент.
        self.tool_hub = tool_hub
        self._last_tool_flow: Optional[mcp_tools.ToolFlowReport] = None
        self._tool_steps: tuple = ()
        # Состояние задачи приходит функцией: агент не владеет конвейером и не читает его файл.
        self.task_provider = task_provider
        # Долговременная память и профили — свои хранилища: очистка диалога их не касается.
        self.long_term = long_term if long_term is not None else long_term_memory.LongTermMemory()
        self.profiles = profiles if profiles is not None else user_profile.ProfileStore()
        self._last_routing: tuple = ()
        self._last_invariants: Optional[object] = None
        # Расписание пишет фоновый исполнитель, приложение только читает: курсор объявленных
        # прогонов начинается с того, что уже лежит в файле, поэтому старт не выдаёт историю за новость.
        self.schedule = schedule if schedule is not None else schedule_store.ScheduleStore()
        self._announced = self.schedule.cursor()
        self._last_schedule: Optional[ScheduleReport] = None
        self._last_citations: Optional[citations.CitationsCheck] = None
        self.restore_context()

    # --- доступ к состоянию ---------------------------------------------------------------

    @property
    def settings(self) -> AnswerSettings:
        return self.config.settings

    @settings.setter
    def settings(self, value: AnswerSettings) -> None:
        self.config.settings = value

    @property
    def model(self) -> str:
        return self.config.model

    @model.setter
    def model(self, value: str) -> None:
        self.config.model = value

    @property
    def last_result(self) -> Optional[AnswerMeta]:
        return self._last_result

    @property
    def last_compression(self) -> Optional[CompressionReport]:
        return self._last_compression

    @property
    def last_facts(self) -> Optional[FactsReport]:
        return self._last_facts

    @property
    def session_usage(self) -> SessionUsage:
        return self._ledger.usage

    @property
    def exchanges(self) -> int:
        return len(self._turns) // 2

    # --- основной путь --------------------------------------------------------------------

    def ask(
        self,
        question: str,
        on_phase: Optional[Callable[[RequestPhase], None]] = None,
    ) -> AnswerMeta:
        """Задаёт вопрос с контекстом сессии; при ошибке API поднимает `APIError`.

        Перед вопросом собирается контекст активной стратегии: суммаризатор сворачивает
        старейшие ходы в резюме, извлекатель обновляет блок фактов. Сбой извлекателя вопрос
        не отменяет — он уходит с прежним блоком фактов.
        """
        user_prompt = prompts.build_user_prompt(question, self.config.settings)
        self._route_memory(question)
        self._retrieve_docs(question, on_phase)
        self._retrieve_code(question, on_phase)
        self._tool_steps = self._run_tool_flow(question, on_phase)
        skip = self._prepare_context(question, user_prompt, on_phase)
        self._signal(on_phase, RequestPhase.REQUEST)
        messages = self._build_messages(user_prompt, skip)
        meta = self._ask_question(messages)
        self._refresh_schedule()
        meta = self._enforce_invariants(messages, meta, on_phase)
        meta = self._enforce_citations(messages, meta, on_phase)
        self._remember(user_prompt, meta.content)
        self.history.add(question, meta.content, usage=self._usage_block(meta))
        return meta

    # --- документация портала и проверка ссылок -------------------------------------------

    def _retrieve_docs(self, question: str, on_phase=None) -> None:
        """Ищет фрагменты документации перед вопросом; сбой поиска вопрос не отменяет.

        Поиск идёт до сборки запроса: найденное — данные именно этого вопроса. Выключенный режим
        и отсутствующий ретривер означают одно: поиска нет и сообщений о нём тоже.
        """
        self._last_citations = None
        self._docs_fragments = ()
        if not self.config.docs_enabled or self.docs_retriever is None:
            self._last_docs_report = None
            return
        rate = None
        if self.config.docs_retrieval == reranking.MODE_ENHANCED:
            rate = lambda text, candidates: self._rate_candidates(text, candidates, on_phase)
        # Версия документации — свойство ретривера: её задаёт пользователь командой, а в снимок
        # она попадает так, как её выбрал сервер или пользователь.
        self._last_docs_report = self.docs_retriever.search(
            question, rate=rate, threshold=self.config.docs_threshold
        )
        self._docs_fragments = tuple(getattr(self._last_docs_report, "fragments", ()) or ())

    def _rate_candidates(self, question: str, candidates, on_phase=None) -> str:
        """Оценка кандидатов моделью сессии: вспомогательный запрос, как у стратегий.

        Расход учитывается в общей сессии, а снимок «последнего запроса» остаётся за ответом:
        пользователю в метриках нужен ответ, а не служебная оценка.
        """
        self._signal(on_phase, RequestPhase.DOCS_RERANK)
        meta = self.client.ask_with_usage_messages(
            reranking.build_messages(question, candidates),
            max_tokens=config.max_tokens_for_words(config.DOCS_RERANK_MAX_WORDS),
            temperature=None,
            model=self.config.model,
        )
        self._ledger.record(meta)
        return meta.content

    # --- автовызов инструментов ---------------------------------------------------------------

    def _run_tool_flow(
        self, question: str, on_phase: Optional[Callable[[RequestPhase], None]]
    ) -> Tuple[mcp_tools.ToolStep, ...]:
        """Выбирает инструменты моделью и выполняет шаги; возвращает выполненные шаги.

        Флоу — один запрос выбора на раунд: продолжение запрашивается только пометкой `more`, иначе
        разговор с моделью о инструментах стоил бы двух запросов на каждый вопрос. Ошибка выбора,
        шага или исчерпание лимита останавливают флоу с названной причиной, и вопрос всё равно
        уходит к модели — но уже без данных, которые получить не удалось.
        """
        hub = self.tool_hub
        self._last_tool_flow = None
        if not self.config.auto_tools or hub is None:
            return ()
        catalog = hub.catalog()
        if not catalog:
            return ()
        try:
            return self._flow_rounds(hub, question, on_phase)
        finally:
            # Соединения флоу закрываются вместе с вопросом: держать серверные процессы между
            # вопросами значило бы держать их до конца сессии.
            hub.close()

    def _flow_rounds(
        self, hub, question: str, on_phase: Optional[Callable[[RequestPhase], None]]
    ) -> Tuple[mcp_tools.ToolStep, ...]:
        """Раунды флоу: запрос выбора, шаги, при `more` — следующий раунд."""
        views = hub.views()
        steps: List[mcp_tools.ToolStep] = []
        results: List[str] = []
        rounds = 0
        choice_requests = 0
        dropped = 0
        stop_reason = mcp_tools.FLOW_NO_TOOL
        messages = mcp_tools.build_choice_messages(views, question)
        while rounds < config.TOOL_FLOW_MAX_ROUNDS:
            rounds += 1
            self._signal(on_phase, RequestPhase.TOOL_CHOICE)
            try:
                meta = self.client.ask_with_usage_messages(
                    messages,
                    max_tokens=config.max_tokens_for_words(config.TOOL_CHOICE_MAX_WORDS),
                    temperature=None,
                    model=self.config.model,
                )
            except APIError as error:
                stop_reason = mcp_tools.FLOW_CHOICE_FAILED.format(round=rounds, error=error)
                break
            self._ledger.record(meta)
            choice_requests += 1
            parsed = mcp_tools.parse_round(meta.content)
            if parsed is mcp_tools.UNPARSED:
                stop_reason = mcp_tools.FLOW_CHOICE_FAILED.format(
                    round=rounds, error="ответ выбора не разобран"
                )
                break
            if parsed is None:
                stop_reason = mcp_tools.FLOW_DONE if steps else mcp_tools.FLOW_NO_TOOL
                break
            dropped += parsed.dropped
            failed = False
            for choice in parsed.steps:
                if len(steps) >= config.TOOL_FLOW_MAX_STEPS:
                    stop_reason = mcp_tools.FLOW_STEPS_LIMIT.format(
                        limit=config.TOOL_FLOW_MAX_STEPS
                    )
                    failed = True
                    break
                step = self._execute_tool_step(choice, views, hub, results, steps, rounds, on_phase)
                if step is None:
                    stop_reason = mcp_tools.FLOW_STEP_FAILED.format(number=len(steps) + 1)
                    failed = True
                    break
                steps.append(step)
                results.append(step.text)
            if failed:
                break
            if not parsed.more:
                stop_reason = mcp_tools.FLOW_DONE
                break
            messages = mcp_tools.build_round_messages(views, question, steps)
        else:
            stop_reason = mcp_tools.FLOW_ROUNDS_LIMIT.format(limit=config.TOOL_FLOW_MAX_ROUNDS)

        self._last_tool_flow = mcp_tools.ToolFlowReport(
            question=question,
            rounds=rounds,
            choice_requests=choice_requests,
            steps=tuple(steps),
            stop_reason=stop_reason,
            dropped=dropped,
        )
        return tuple(steps)

    def _execute_tool_step(
        self,
        choice: mcp_tools.ToolChoice,
        views,
        hub,
        results: List[str],
        steps: List[mcp_tools.ToolStep],
        round_number: int,
        on_phase: Optional[Callable[[RequestPhase], None]],
    ) -> Optional[mcp_tools.ToolStep]:
        """Один шаг флоу: подстановка ссылок, маршрут по снимкам, вызов и проверка отказа.

        Любая неудача — не исключение наружу, а причина остановки: данные, которых нет, не должны
        превращаться в данные, которые «как-то» получились.
        """
        try:
            arguments, sources = mcp_tools.resolve_references(choice.arguments, list(results))
        except mcp_tools.ReferenceFailure:
            return None
        route = mcp_tools.route(views, choice.server, choice.tool)
        if route.error:
            return None
        rules = tuple(getattr(self.domain, "invariants", ()) or ())
        bad_arguments = invariants.check_arguments(arguments, rules) if rules else ()
        if bad_arguments:
            # Аргументы проверяются до запуска процесса: секрет не должен дойти ни до сервера,
            # ни до журнала вызовов.
            return None
        self._signal(on_phase, RequestPhase.MCP_TOOL)
        try:
            result = hub.call(route.server, choice.tool, arguments)
        except (MCPError, KeyError):
            return None
        if result.is_error:
            return None
        return mcp_tools.ToolStep(
            step=len(steps) + 1,
            round=round_number,
            server=route.server,
            tool=choice.tool,
            arguments=arguments,
            sources=sources,
            text=result.text,
            rerouted_from=route.rerouted_from,
        )

    def tool_flow_report(self) -> Optional[mcp_tools.ToolFlowReport]:
        """Снимок последнего флоу: раунды, шаги, число запросов выбора, причина остановки."""
        return self._last_tool_flow

    def _tool_message(self) -> Optional[Dict[str, str]]:
        """Служебное сообщение с результатами инструментов: один вызов или цепочка шагов."""
        if not self._tool_steps:
            return None
        if len(self._tool_steps) == 1:
            step = self._tool_steps[0]
            content = mcp_tools.tool_result_message(
                step.server, step.tool, step.arguments, step.text
            )
        else:
            content = mcp_tools.tool_chain_message(step.as_tuple() for step in self._tool_steps)
        return {"role": "system", "content": content}

    def _route_memory(self, question: str) -> None:
        """Раскладывает реплику пользователя по слоям памяти правилами домена.

        Роутинг идёт до сборки запроса: запись, сделанная текущим сообщением, должна быть видна
        модели в этом же запросе. Модель память не заполняет: источник — только слова пользователя.
        """
        self._last_routing = ()
        rules = getattr(self.domain, "memory", None)
        if rules is None:
            return
        records = memory_layers.route(question, rules.rules)
        if not records:
            return
        working = dict(self.history.working)
        for record in records:
            if record.layer == memory_layers.LONG_TERM:
                self.long_term.remember(record.key, record.value, record.category)
                continue
            working[record.key] = record.value
        if working != dict(self.history.working):
            merged = memory_layers.merge_working(
                self.history.working, memory_layers.records_from_block(working)
            )
            self.history.set_working(merged)
        self._last_routing = tuple(records)

    def set_goal(self, text: str) -> None:
        """Ставит цель текущей задачи в рабочую память (`/memory goal`)."""
        rules = getattr(self.domain, "memory", None)
        key = getattr(rules, "goal_key", "") if rules is not None else ""
        if not key:
            return
        merged = memory_layers.merge_working(
            self.history.working,
            (
                memory_layers.MemoryRecord(
                    layer=memory_layers.WORKING,
                    category=getattr(rules, "goal_category", ""),
                    key=key,
                    value=memory_layers.clip_value(text),
                ),
            ),
        )
        self.history.set_working(merged)

    def remember(self, text: str) -> object:
        """Запоминает текст: правило домена задаёт слой и ключ, иначе это заметка с новым ключом."""
        rules = getattr(self.domain, "memory", None)
        found = memory_layers.route(text, getattr(rules, "rules", ()) or ())
        if found:
            record = found[0]
            if record.layer == memory_layers.LONG_TERM:
                self.long_term.remember(record.key, record.value, record.category)
            else:
                self.set_goal(record.value) if record.key == getattr(rules, "goal_key", "") else None
                merged = memory_layers.merge_working(self.history.working, (record,))
                self.history.set_working(merged)
            return record
        key = self._next_note_key()
        return memory_layers.MemoryRecord(
            layer=memory_layers.LONG_TERM,
            category=memory_layers.CATEGORY_NOTE,
            key=key,
            value=memory_layers.clip_value(text),
        )

    def _next_note_key(self) -> str:
        """Ключ для заметки без правила: последовательный номер в долговременной памяти."""
        taken = {record.key for record in self.long_term.records()}
        number = 1
        while f"заметка-{number}" in taken:
            number += 1
        return f"заметка-{number}"

    def forget(self, key: str) -> int:
        """Удаляет записи: `all` — вся долговременная память, иначе ключ в обоих слоях."""
        removed = 0
        working = dict(self.history.working)
        if key == "all":
            removed += len(working)
            self.history.set_working({})
            removed += len(self.long_term.records())
            self.long_term.clear()
            return removed
        if key in working:
            working.pop(key)
            self.history.set_working(working)
            removed += 1
        removed += self.long_term.forget(key)
        return removed

    def save_profile(self, domain_profile, answers) -> str:
        """Собирает профиль из ответов опросника и делает его активным.

        Имя и значения приходят от интерфейса: сам опросник — диалог с пользователем, а не логика
        памяти. Пустое значение раздела не перезаписывает уже сохранённое.
        """
        name = ""
        values: Dict[str, str] = {}
        for field, value in answers:
            value = (value or "").strip()
            if not value:
                continue
            if field == user_profile.NAME_FIELD:
                name = value
                continue
            values[field] = value
        if not name:
            name = self.profiles.next_name(domain_profile.name_prefix)
        profile = user_profile.UserProfile(name=name)
        for field, value in values.items():
            profile = profile.with_value(field, value)
        self.profiles.save(profile)
        return name

    def _memory_message(self) -> Optional[Dict[str, str]]:
        """Сообщение памяти: долговременные записи, рабочая память задачи и инструкция."""
        rules = getattr(self.domain, "memory", None)
        if rules is None:
            return None
        # Долговременная память идёт первой: свежая реплика и рабочая память важнее того, что
        # агент знал раньше.
        records = tuple(self.long_term.records()) + memory_layers.records_from_block(
            self.history.working
        )
        content = memory_layers.memory_message(records)
        return {"role": "system", "content": content} if content else None

    def _profile_message(self) -> Optional[Dict[str, str]]:
        """Сообщение профиля: он отвечает на вопрос «как отвечать», поэтому идёт сразу после системы."""
        domain_profile = getattr(self.domain, "profile", None)
        if domain_profile is None:
            return None
        active = self.profiles.active()
        if active is None:
            return None
        content = user_profile.profile_message(active, domain_profile.sections)
        return {"role": "system", "content": content} if content else None

    def _invariants_message(self) -> Optional[Dict[str, str]]:
        """Сообщение правил домена: они идут в каждый вопрос и стоят выше профиля и памяти."""
        rules = tuple(getattr(self.domain, "invariants", ()) or ())
        if not rules:
            return None
        return {"role": "system", "content": invariants.invariants_message(rules)}

    def _retrieve_code(self, question: str, on_phase=None) -> None:
        """Ищет фрагменты исходников перед вопросом; сбой поиска вопрос не отменяет.

        Поиск локальный, поэтому единственное, что мешает ему идти — выключенный корпус или
        отсутствие индекса: тогда нет ни фрагментов, ни сообщений о поиске.
        """
        self._code_fragments = ()
        if not self.config.code_enabled or self.code_retriever is None:
            self._last_code_report = None
            return
        # Отсутствие индекса — не состояние поиска, а его невозможность: тогда в запрос не уходит
        # ничего. Нечитаемый (испорченный) индекс — уже состояние, и о нём модель узнаёт.
        database = getattr(self.code_retriever, "database", None)
        if database is not None and not Path(database).is_file():
            self._last_code_report = None
            return
        rewrite = None
        if self.config.code_retrieval == code_retrieval.MODE_ENHANCED:
            rewrite = lambda text: self._code_query(text, on_phase)
        rate = None
        if self.config.code_retrieval == code_retrieval.MODE_ENHANCED:
            rate = lambda text, candidates: self._code_rerank(text, candidates, on_phase)
        self._last_code_report = self.code_retriever.search(
            question,
            mode=self.config.code_retrieval,
            rewrite=rewrite,
            rate=rate,
            threshold=self.config.code_threshold,
            before=self.config.code_before,
            after=self.config.code_after,
        )
        self._code_fragments = tuple(
            getattr(self._last_code_report, "fragments", ()) or ()
        )

    def _code_query(self, question: str, on_phase=None) -> str:
        """Переформулировка вопроса в поисковый запрос: вспомогательный запрос сессии."""
        self._signal(on_phase, RequestPhase.CODE_QUERY)
        meta = self.client.ask_with_usage_messages(
            code_retrieval.query_messages(question),
            max_tokens=config.max_tokens_for_words(config.CODE_QUERY_MAX_WORDS),
            temperature=None,
            model=self.config.model,
        )
        self._ledger.record(meta)
        return meta.content

    def _code_rerank(self, question: str, candidates, on_phase=None) -> str:
        """Оценка кандидатов кода: тот же механизм, что у документации, свой ассет."""
        self._signal(on_phase, RequestPhase.CODE_RERANK)
        meta = self.client.ask_with_usage_messages(
            code_retrieval.rerank_messages(question, candidates),
            max_tokens=config.max_tokens_for_words(config.CODE_RERANK_MAX_WORDS),
            temperature=None,
            model=self.config.model,
        )
        self._ledger.record(meta)
        return meta.content

    @property
    def last_routing(self) -> tuple:
        """Записи памяти, сделанные текущей репликой: их печатает журнал."""
        return self._last_routing

    @property
    def last_invariants(self):
        return self._last_invariants

    def memory_report(self) -> Dict[str, object]:
        """Снимок слоёв памяти: обмены, рабочая память и долговременные записи с путями."""
        rules = getattr(self.domain, "memory", None)
        working = dict(self.history.working)
        goal_key = getattr(rules, "goal_key", "") if rules is not None else ""
        return {
            "exchanges": self.exchanges,
            "working": working,
            # Цель отдаётся отдельным полем: интерфейсу не нужно знать её ключ из пакета домена.
            "goal": working.get(goal_key, "") if goal_key else "",
            "long_term": tuple(self.long_term.records()),
            "long_term_path": str(getattr(self.long_term, "path", "")),
            "rules": memory_layers.describe_rules(getattr(rules, "rules", ()) or ()),
            "last_routing": tuple(self._last_routing),
        }

    def profile_report(self) -> Dict[str, object]:
        """Снимок профилей: активный, его разделы и имена всех профилей."""
        domain_profile = getattr(self.domain, "profile", None)
        # Секции берутся из пакета домена, а не из скрипта опросника: опросник добавляет вопрос об
        # имени, и в отчёте он был бы лишним разделом.
        sections = getattr(domain_profile, "sections", ()) if domain_profile is not None else ()
        active = self.profiles.active()
        return {
            "active": active.name if active is not None else "",
            "sections": tuple(
                (
                    section.label,
                    (active.value(section.id) if active is not None else "") or section.default,
                )
                for section in sections
            ),
            "names": self.profiles.names(),
            "path": str(getattr(self.profiles, "path", "")),
        }

    def invariants_report(self) -> Dict[str, object]:
        """Снимок правил домена и результата последней проверки ответа."""
        rules = tuple(getattr(self.domain, "invariants", ()) or ())
        return {"rules": rules, "check": self._last_invariants}

    def schedule_report(self) -> "ScheduleReport":
        """Снимок расписания: задания, прогоны, объём собранного и путь хранилища."""
        self.schedule.reload()
        return ScheduleReport(
            jobs=self.schedule.jobs(),
            runs=self.schedule.runs(),
            collected_total=self.schedule.collected_total(),
            path=str(getattr(self.schedule, "path", "")),
        )

    def _refresh_schedule(self) -> None:
        """Ищет прогоны, которых пользователь ещё не видел, и сдвигает курсор."""
        self.schedule.reload()
        fresh = self.schedule.fresh_runs(self._announced)
        if not fresh:
            return
        self._last_schedule = ScheduleReport(
            jobs=self.schedule.jobs(),
            runs=self.schedule.runs(),
            collected_total=self.schedule.collected_total(),
            path=str(getattr(self.schedule, "path", "")),
            fresh=fresh,
        )
        self._announced = self.schedule.cursor()

    @property
    def last_schedule(self) -> Optional["ScheduleReport"]:
        return self._last_schedule

    def record_usage(self, meta: AnswerMeta) -> None:
        """Учитывает расход запроса, выполненного вне пути вопроса (шаги конвейера задачи)."""
        self._ledger.record(meta)

    def _task_message(self) -> Optional[Dict[str, str]]:
        """Сообщение состояния задачи: этап, шаг и допустимые переходы — для контекста запроса."""
        if self.task_provider is None:
            return None
        try:
            state = self.task_provider()
        except Exception:  # noqa: BLE001 - состояние задачи не должно ломать вопрос
            return None
        content = task_state.task_message(state) if state is not None else None
        return {"role": "system", "content": content} if content else None

    def code_report(self) -> Optional[object]:
        """Снимок последнего поиска по корпусу кода."""
        return self._last_code_report

    def docs_report(self) -> Optional[object]:
        """Снимок последнего поиска: что искали, где, в какой версии, что доставили."""
        return self._last_docs_report

    @property
    def last_citations(self) -> Optional[citations.CitationsCheck]:
        return self._last_citations

    def _citations_enabled(self) -> bool:
        """Проверка имеет смысл только при доставленных фрагментах и свободном формате.

        У форматов JSON, компактного и диффа свои контракты: обязательные блоки цитат им бы
        противоречили, поэтому источники им печатает терминальный слой.
        """
        return bool(self._docs_fragments or self._code_fragments) and (
            self.config.format is AnswerFormat.FREE
        )

    def _citations_check(self, answer: str) -> citations.CitationsCheck:
        """Проверка ответа против доставленных корпусов: подтверждение любого из них достаточно."""
        return citations.check_groups(
            answer,
            (
                (self._docs_fragments, config.DOCS_CITATION_MIN_CHARS),
                (self._code_fragments, config.CODE_CITATION_MIN_CHARS),
            ),
        )

    def _enforce_citations(
        self,
        messages: List[Dict[str, str]],
        meta: AnswerMeta,
        on_phase: Optional[Callable[[RequestPhase], None]],
    ) -> AnswerMeta:
        """Проверка ссылок кодом: повтор с перечнем нарушений, затем замена ответа.

        Как у инвариантов: повтор — продолжение того же диалога, а если повтор тоже не подтверждён,
        до пользователя доходит только фиксированный текст, и именно он идёт в лог и историю.
        """
        if not self._citations_enabled():
            return meta
        check = self._citations_check(meta.content)
        if check.confirmed:
            self._last_citations = check
            return meta
        final = check.violations
        for _ in range(config.DOCS_CITATION_RETRIES):
            self._signal(on_phase, RequestPhase.REQUEST)
            retry = messages + [
                {"role": "assistant", "content": meta.content},
                {"role": "user", "content": citations.retry_prompt(final)},
            ]
            meta = self._ask_question(retry)
            repeated = self._citations_check(meta.content)
            if repeated.confirmed:
                self._last_citations = citations.CitationsCheck(
                    violations=check.violations, retried=True
                )
                return meta
            final = repeated.violations
        self._last_citations = citations.CitationsCheck(
            violations=check.violations, retried=True, replaced=True, final_violations=final
        )
        return AnswerMeta(
            content=citations.disclaimer_text(),
            model=meta.model,
            elapsed_seconds=meta.elapsed_seconds,
            prompt_tokens=meta.prompt_tokens,
            completion_tokens=meta.completion_tokens,
            total_tokens=meta.total_tokens,
            cost_usd=meta.cost_usd,
            finish_reason=meta.finish_reason,
        )

    def _enforce_invariants(
        self,
        messages: List[Dict[str, str]],
        meta: AnswerMeta,
        on_phase: Optional[Callable[[RequestPhase], None]],
    ) -> AnswerMeta:
        """Проверка ответа правилами домена: повтор, затем замена текстом отказа.

        Ответ-отказ не проверяется: он называет запрещённое, но ничего не предлагает. В историю
        уходит то, что увидел пользователь, — отклонённый ответ не должен становиться примером
        для следующего запроса.
        """
        rules = tuple(getattr(self.domain, "invariants", ()) or ())
        if not rules or invariants.is_refusal(meta.content):
            self._last_invariants = None
            return meta
        found = invariants.check_answer(meta.content, rules)
        if not found:
            self._last_invariants = None
            return meta
        for _ in range(config.INVARIANT_RETRIES):
            self._signal(on_phase, RequestPhase.REQUEST)
            retry = messages + [
                {"role": "assistant", "content": meta.content},
                {"role": "user", "content": invariants.retry_prompt(found)},
            ]
            meta = self._ask_question(retry)
            if invariants.is_refusal(meta.content):
                self._last_invariants = None
                return meta
            again = invariants.check_answer(meta.content, rules)
            if not again:
                self._last_invariants = None
                return meta
            found = again
        text = invariants.refusal_text(found)
        self._last_invariants = found
        return AnswerMeta(
            content=text,
            model=meta.model,
            elapsed_seconds=meta.elapsed_seconds,
            prompt_tokens=meta.prompt_tokens,
            completion_tokens=meta.completion_tokens,
            total_tokens=meta.total_tokens,
            cost_usd=meta.cost_usd,
            finish_reason=meta.finish_reason,
        )

    def reset(self) -> None:
        """Опустошает диалог и его память (команда /clear). Файлы других слоёв не трогает."""
        self._turns.clear()
        self._summary = None
        self._summary_covers = 0
        self._log_covered = 0
        self._facts = {}
        self._facts_pending = []
        self._facts_covered = 0
        self._branches.reset()
        self._last_compression = None
        self._last_facts = None
        self._last_result = None
        # Снимок проверки относится к последнему ответу — он сбрасывается вместе с диалогом.
        self._last_citations = None
        self._last_routing = ()
        self._last_invariants = None
        # Снимок поиска по документации не сбрасывается: это результат уже выполненной операции,
        # и /clear его не отменяет (то же правило, что у снимка сжатия).
        self.history.clear_dialogues()
        self._ledger.reset()

    def restore_context(self) -> None:
        """Засеивает контекст из истории: резюме, блок фактов и несвёрнутый хвост обменов.

        Ход пользователя собирается текущими настройками (в файле хранится сырой вопрос),
        ход ассистента переносится дословно. Непокрытые обмены догоняются суммаризатором при
        первом вопросе — тем же механизмом, что и живая сессия, без отдельного кода рестарта.
        """
        self._summary = self.history.summary
        self._summary_covers = self.history.summary_covers
        self._log_covered = 0
        self._facts = dict(self.history.facts)
        dialogues = self.history.dialogues
        tail = max(0, len(dialogues) - self._summary_covers)
        for item in dialogues[-tail:] if tail else []:
            self._remember(
                prompts.build_user_prompt(item["question"], self.config.settings),
                item["answer"],
            )
        # Восстановленные сообщения уже отражены в блоке из файла: извлекатель их не переспрашивает.
        self._facts_covered = len(self._turns) // 2

    def context_report(self) -> ContextReport:
        """Снимок контекста без обращения к модели: стратегия, границы, память, оценка токенов."""
        turns = self._view_turns()
        return ContextReport(
            strategy=self.config.strategy,
            window=self.config.settings.compress_after,
            max_session_tokens=self.config.settings.max_session_tokens,
            log_exchanges=len(self._turns) // 2,
            request_turns=len(turns),
            summary_covers=self._summary_covers,
            has_summary=bool(self._summary),
            facts=dict(self._facts),
            branch=self._branches.active_name,
            branches=self._branches.branches(),
            tokens_estimate=self._context_tokens(),
        )

    # --- ветки диалога ---------------------------------------------------------------------

    def branches(self) -> Tuple[Tuple[str, int], ...]:
        """Снимок веток: (имя, число обменов) в порядке создания."""
        return self._branches.branches()

    @property
    def active_branch(self) -> str:
        return self._branches.active_name

    def switch_branch(self, name: str) -> bool:
        return self._branches.switch(name)

    def set_checkpoint(self) -> None:
        """Отмечает текущую позицию: от неё создаётся следующая ветка."""
        self._branches.checkpoint()

    def new_branch(self) -> str:
        return self._branches.new_branch()

    # --- сборка запроса -------------------------------------------------------------------

    def _build_messages(self, user_prompt: str, skip: int = 0) -> List[Dict[str, str]]:
        """Сборка запроса: системное сообщение домена и формата, память стратегии, ходы, новый ход."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": prompts.build_system_message(self.domain, self.config.format)}
        ]
        for message in (
            self._profile_message(),
            self._invariants_message(),
            self._docs_message(),
            self._code_message(),
            self._tool_message(),
            self._task_message(),
            self._memory_message(),
        ):
            if message is not None:
                messages.append(message)
        if self._citations_enabled():
            messages.append({"role": "system", "content": citations.citations_message()})
        strategy_memory = self._strategy_memory_message()
        if strategy_memory is not None:
            messages.append(strategy_memory)
        messages.extend(self._view_turns(skip))
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _docs_message(self) -> Optional[Dict[str, str]]:
        """Сообщение о доставленных фрагментах или о состоянии поиска.

        Есть фрагменты — уходит блок с заголовками (путь, раздел, версия). Фрагментов нет, но
        поиск был — уходит инструкция о том, что источника нет: молчание модели читалось бы как
        разрешение ответить по памяти.
        """
        report = self._last_docs_report
        if report is None:
            return None
        if self._docs_fragments:
            note = prompts.docs_note(report)
            content = citations.context_message(self._docs_fragments, note=note)
            return {"role": "system", "content": content} if content else None
        state_message = prompts.docs_state_message(report)
        return {"role": "system", "content": state_message} if state_message else None

    def _code_message(self) -> Optional[Dict[str, str]]:
        """Сообщение о фрагментах исходников или о состоянии их поиска.

        Есть фрагменты — уходит блок с идентификаторами `путь:L10-L40`. Фрагментов нет, но поиск
        был — уходит инструкция о состоянии: молчание читалось бы как разрешение назвать метод по
        памяти, а код этого проекта проверяем.
        """
        report = self._last_code_report
        if report is None:
            return None
        if self._code_fragments:
            note = prompts.code_note(report)
            content = citations.context_message(self._code_fragments, note=note)
            return {"role": "system", "content": content} if content else None
        state_message = prompts.code_state_message(report)
        return {"role": "system", "content": state_message} if state_message else None

    def _strategy_memory_message(self) -> Optional[Dict[str, str]]:
        """Сообщение памяти стратегии: резюме или блок фактов, если они непусты."""
        if self.config.strategy is ContextStrategy.STICKY_FACTS and self._facts:
            return {"role": "system", "content": context_strategies.facts_message(self._facts)}
        if self.config.strategy is ContextStrategy.SUMMARY and self._summary:
            return {"role": "system", "content": context_compressor.summary_message(self._summary)}
        return None

    def _view_turns(self, skip: int = 0) -> List[Dict[str, str]]:
        """Ходы, которые активная стратегия отправляет в запрос (без обрезки по потолку)."""
        strategy = self.config.strategy
        if strategy is ContextStrategy.BRANCHING:
            turns = self._branches.active_messages()
        elif strategy is ContextStrategy.SUMMARY:
            turns = self._turns[self._log_covered * 2 :]
        else:
            turns = list(self._turns)
        if strategy in (ContextStrategy.SLIDING_WINDOW, ContextStrategy.STICKY_FACTS):
            turns = context_strategies.window_messages(turns, self.config.settings.compress_after)
        return turns[skip * 2 :]

    def _prepare_context(
        self,
        question: str,
        user_prompt: str,
        on_phase: Optional[Callable[[RequestPhase], None]],
    ) -> int:
        """Готовит контекст и возвращает, сколько старейших обменов отброшено из вида запроса."""
        strategy = self.config.strategy
        if strategy is ContextStrategy.STICKY_FACTS:
            self._update_facts(question, on_phase)
        if strategy is ContextStrategy.SUMMARY:
            self._digest_before_request(user_prompt, on_phase)
            return 0
        return self._shrink_to_ceiling(user_prompt)

    def _shrink_to_ceiling(self, user_prompt: str) -> int:
        """Сужает вид стратегии, пока оценка запроса выше потолка токенов сессии."""
        skip = 0
        while len(self._view_turns(skip)) > 2 and not self._fits_ceiling(user_prompt, skip):
            skip += 1
        return skip

    def _fits_ceiling(self, user_prompt: str, skip: int = 0) -> bool:
        return self._estimate(self._build_messages(user_prompt, skip)) <= (
            self.config.settings.max_session_tokens
        )

    def _context_tokens(self) -> int:
        return self._estimate(self._build_messages(""))

    @staticmethod
    def _estimate(messages: List[Dict[str, str]]) -> int:
        return sum(estimate_tokens(message["content"]) for message in messages)

    def _remember(self, user_content: str, assistant_content: str) -> None:
        self._turns.append({"role": "user", "content": user_content})
        self._turns.append({"role": "assistant", "content": assistant_content})
        self._branches.add_exchange()

    def _signal(
        self,
        on_phase: Optional[Callable[[RequestPhase], None]],
        phase: RequestPhase,
    ) -> None:
        if on_phase is not None:
            on_phase(phase)

    def _ask_question(self, messages: List[Dict[str, str]]) -> AnswerMeta:
        """Запрос вопроса с настройками сессии; расход учтён, метрики — в `last_result`."""
        meta = self.client.ask_with_usage_messages(
            messages,
            max_tokens=config.max_tokens_for_words(self.config.max_words),
            temperature=self.config.temperature,
            model=self.config.model,
        )
        self._last_result = meta
        self._ledger.record(meta)
        return meta

    # --- вспомогательные запросы стратегий -------------------------------------------------

    def _update_facts(
        self,
        question: str,
        on_phase: Optional[Callable[[RequestPhase], None]],
    ) -> None:
        """Обновляет блок фактов по сообщениям пользователя, ещё не отданным извлекателю.

        Очередь пополняется из лога, поэтому переключение на факты посреди диалога даёт
        полный блок, а не «слепой». Успешный ответ (в том числе пустой объект — «новых фактов
        нет») снимает обработанные сообщения с очереди; сбой оставляет блок и очередь как были,
        и вопрос всё равно уходит.
        """
        if not self._facts_pending:
            self._facts_pending = self._logged_user_messages()
        self._facts_pending.append(question)
        answered_exchanges = len(self._turns) // 2
        while self._facts_pending:
            batch = context_strategies.facts_batch(
                self._facts, self._facts_pending, self.config.settings.max_session_tokens
            )
            self._signal(on_phase, RequestPhase.FACTS_UPDATE)
            try:
                meta = self.client.ask_with_usage_messages(
                    context_strategies.build_facts_messages(self._facts, batch),
                    max_tokens=config.max_tokens_for_words(config.FACTS_MAX_WORDS),
                    temperature=None,
                    model=self.config.model,
                )
            except APIError as error:
                self._last_facts = FactsReport(
                    updated=False, keys=len(self._facts), error=str(error)
                )
                return
            self._last_result = meta
            self._ledger.record(meta)
            parsed = context_strategies.parse_facts_response(meta.content)
            if parsed is None:
                self._last_facts = FactsReport(
                    updated=False,
                    keys=len(self._facts),
                    error="Извлекатель фактов вернул пустой ответ или не JSON.",
                )
                return
            self._facts = context_strategies.merge_facts(self._facts, parsed)
            del self._facts_pending[: len(batch)]
            self.history.set_facts(self._facts)
            self._last_facts = FactsReport(updated=True, keys=len(self._facts))
        self._facts_covered = answered_exchanges + 1

    def _logged_user_messages(self) -> List[str]:
        """Сообщения пользователя из лога, ещё не отданные извлекателю фактов."""
        answered = len(self._turns) // 2
        return [
            self._turns[index * 2]["content"] for index in range(self._facts_covered, answered)
        ]

    def _digest_before_request(
        self,
        user_prompt: str,
        on_phase: Optional[Callable[[RequestPhase], None]],
    ) -> None:
        """Сворачивает старые ходы в резюме перед вопросом (стратегия резюме).

        Два повода: оценка запроса выше потолка токенов или длина несвёрнутого хвоста достигла
        порога сжатия. Свёрнутые ходы остаются в логе — растёт лишь счётчик покрытых обменов,
        поэтому переключение стратегии снова делает их отправимыми.

        Пустой ответ суммаризатора — сбой, а не резюме: принять его значит поднять счётчик,
        ничего не подставив в запрос. Запрос уже потрачен, поэтому расход учтён, а память
        остаётся прежней — следующий вопрос повторит сжатие.
        """
        settings = self.config.settings
        digested_messages = 0
        digested_exchanges = 0
        last_meta: Optional[AnswerMeta] = None
        while True:
            uncovered = self._turns[self._log_covered * 2 :]
            if self._fits_ceiling(user_prompt) and len(uncovered) < settings.compress_after:
                break
            digest, _ = context_compressor.split_for_digest(uncovered)
            if not digest:
                break
            batch = context_compressor.batch_within_budget(
                digest, self._summary, settings.max_session_tokens
            )
            self._signal(on_phase, RequestPhase.COMPRESSION)
            meta = self.client.ask_with_usage_messages(
                context_compressor.build_summary_messages(self._summary, batch),
                max_tokens=config.max_tokens_for_words(config.SUMMARY_MAX_WORDS),
                temperature=None,
                model=self.config.model,
            )
            self._last_result = meta
            self._ledger.record(meta)
            if not meta.content.strip():
                reason = (
                    " (finish_reason=length: модель израсходовала бюджет на рассуждение)"
                    if meta.finish_reason == "length"
                    else ""
                )
                raise APIError(
                    "Суммаризатор вернул пустой ответ — контекст не сжат" + reason + "."
                )
            self._summary = meta.content
            self._summary_covers += len(batch) // 2
            self._log_covered += len(batch) // 2
            self.history.set_summary(self._summary, self._summary_covers)
            digested_messages += len(batch)
            digested_exchanges += len(batch) // 2
            last_meta = meta
        if digested_exchanges:
            self._last_compression = CompressionReport(
                messages=digested_messages, exchanges=digested_exchanges, meta=last_meta
            )

    @staticmethod
    def _usage_block(meta: AnswerMeta) -> Dict[str, Any]:
        """Метрики запроса для записи в историю (cost_usd — None, если цены нет)."""
        return {
            "prompt_tokens": meta.prompt_tokens,
            "completion_tokens": meta.completion_tokens,
            "total_tokens": meta.total_tokens,
            "cost_usd": meta.cost_usd,
        }
