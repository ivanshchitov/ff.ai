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
    context_compressor,
    context_strategies,
    prompts,
    reranking,
)
from .answer_settings import AnswerFormat, AnswerSettings, ContextStrategy
from .api_client import APIClient, APIError, AnswerMeta
from .domains import Domain
from .history_manager import HistoryManager
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


@dataclass
class CompressionReport:
    """Что сделало сжатие: сколько сообщений и обменов свёрнуто и какой это был запрос."""

    messages: int
    exchanges: int
    meta: Optional[AnswerMeta] = None


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
        self._retrieve_docs(question, on_phase)
        self._retrieve_code(question, on_phase)
        skip = self._prepare_context(question, user_prompt, on_phase)
        self._signal(on_phase, RequestPhase.REQUEST)
        messages = self._build_messages(user_prompt, skip)
        meta = self._ask_question(messages)
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
        # Снимок поиска по документации не сбрасывается: это результат уже выполненной операции,
        # и /clear его не отменяет (то же правило, что у снимка сжатия).
        self.history.clear()
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
        for message in (self._docs_message(), self._code_message()):
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
