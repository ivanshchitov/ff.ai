"""Терминальный интерфейс: главный цикл, панели, журнал и строка ввода.

Оркестрации здесь нет: агент, настройки, модель, расход и ветки живут в фасаде
(`core/session.py`), а этот модуль читает ввод, подписывается на события сессии и рисует
снимки. Экран — append-only: приложение не очищает терминал, поэтому прошлые ответы, строки
журнала и подсказки остаются в прокрутке.

Приглашение ввода — обычная строка без rich-разметки: разметка внутри промпта ломает подсчёт
видимой ширины в readline, и строка портится при стирании.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

try:  # pragma: no cover - зависит от сборки Python
    import readline
except ImportError:  # pragma: no cover - Windows без pyreadline
    readline = None  # type: ignore[assignment]

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from core import config
from core.agent import CompressionReport, FactsReport, RequestPhase
from core.answer_settings import AnswerFormat, ContextStrategy
from core.api_client import APIError, AnswerMeta, is_valid_api_key, is_valid_json_answer
from core.session import AssistantSession, JournalLine, PhaseChanged

from . import branches_screen, commands_screen, keyboard, models_screen, settings_screen

INPUT_PROMPT = "Вы: "
PHASE_LABELS = {
    RequestPhase.REQUEST: "Отправка вопроса...",
    RequestPhase.COMPRESSION: "Суммаризация контекста...",
    RequestPhase.FACTS_UPDATE: "Обновление фактов...",
    RequestPhase.MCP_CONNECT: "Подключение к MCP-серверам...",
    RequestPhase.MCP_TOOL: "Вызов инструмента...",
    RequestPhase.DOCS_RERANK: "Отбор фрагментов документации...",
    RequestPhase.CODE_QUERY: "Поисковый запрос по исходникам...",
    RequestPhase.CODE_RERANK: "Отбор фрагментов кода...",
}
FORMAT_LABELS = {
    AnswerFormat.FREE: "свободный",
    AnswerFormat.COMPACT: "компактный",
    AnswerFormat.JSON: "json",
    AnswerFormat.PATCH: "дифф",
}
STRATEGY_LABELS = {
    ContextStrategy.SUMMARY: "резюме",
    ContextStrategy.SLIDING_WINDOW: "окно",
    ContextStrategy.STICKY_FACTS: "факты",
    ContextStrategy.BRANCHING: "ветки",
}
STATUS_COMMANDS = ["/exit", "/commands"]
API_KEY_CHARSET_ERROR = (
    "Ключ содержит символы, недопустимые в HTTP-заголовке: "
    "проверьте раскладку клавиатуры и введите ключ заново."
)
GOODBYE_MESSAGE = "До связи! История диалога сохранена."
TRUNCATED_EMPTY_WARNING = (
    "Модель исчерпала бюджет max_tokens и не вернула ответ — попробуйте ещё раз "
    "или уменьшите объём ответа."
)
TRUNCATED_WARNING = "Ответ мог быть обрезан: модель упёрлась в бюджет max_tokens."


class DevAssistantTUI:
    """Главный цикл: вопрос, команда, панели, статус-бар."""

    def __init__(
        self,
        session: AssistantSession,
        console: Optional[Console] = None,
        typing_delay: Optional[float] = None,
    ) -> None:
        self.session = session
        self.console = console if console is not None else Console()
        self.typing_delay = (
            typing_delay
            if typing_delay is not None
            else float(os.getenv("FFAI_TYPING_DELAY", "0.004"))
        )
        self._journal: List[str] = []
        self._printed_compression: Optional[CompressionReport] = None
        self._printed_facts: Optional[FactsReport] = None
        self._register_autocomplete()

    # --- основной цикл -------------------------------------------------------------------

    def run(self) -> int:
        self._print_welcome()
        self._print_mcp_summary()
        self._replay_history()
        try:
            while not self.session.exit_requested:
                try:
                    line = input(INPUT_PROMPT)
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    self._handle_command(line)
                else:
                    self._ask(line)
                if self.session.exit_requested:
                    break
                self._print_status_bar()
        except KeyboardInterrupt:
            # Ctrl+C приходит и во время ввода, и во время ожидания ответа модели: в обоих
            # случаях приложение завершается штатно, а не необработанным traceback-ом.
            self.console.print()
        self._exit()
        return 0

    def _handle_command(self, line: str) -> None:
        command = line.split(maxsplit=1)[0]
        if command == "/commands":
            self._open_commands_screen()
            return
        if command == "/settings":
            self._open_settings_screen()
            return
        if command == "/models":
            self._open_models_screen()
            return
        if command == "/branches":
            self._open_branches_screen()
            return
        result = self.session.run_command(line)
        for text in result.lines:
            self.console.print(escape(text))
        if result.domain_changed:
            self.console.print(
                f"[dim]Активный домен: {escape(self.session.domain.title)}[/dim]"
            )

    def _ask(self, question: str) -> None:
        if not self._ensure_api_key():
            return
        try:
            answer = self._ask_with_spinner(question)
        except APIError as error:
            self.console.print(f"[bold red]Ошибка запроса: {escape(str(error))}[/bold red]")
            return
        self._print_answer(answer.text)
        self._print_journal()
        self._print_sources()
        self._print_code_sources()
        self._print_docs_journal()
        self._print_code_journal()
        self._print_warnings(answer.meta)
        self._print_usage_meta(answer.meta)

    def _ask_with_spinner(self, question: str):
        """Вопрос под спиннером: фазы приходят событиями сессии, журнал собирается на экран."""
        self._journal = []
        with self.console.status(PHASE_LABELS[RequestPhase.REQUEST], spinner="dots") as status:

            def listener(event: object) -> None:
                if isinstance(event, PhaseChanged):
                    status.update(PHASE_LABELS.get(event.phase, "Запрос..."))
                elif isinstance(event, JournalLine):
                    self._journal.append(event.text)

            unsubscribe = self.session.subscribe(listener)
            try:
                return self.session.ask(question)
            finally:
                unsubscribe()

    # --- печать ---------------------------------------------------------------------------

    def _print_welcome(self) -> None:
        selection = self.session.domain_selection
        body = Text()
        body.append("ff.ai — ассистент по разработке\n", style="bold")
        body.append(f"Домен: {selection.domain.title} ({selection.domain.id})\n")
        body.append(f"Корень репозитория: {self.session.root}\n")
        body.append(f"Домен выбран: {selection.source}")
        if selection.evidence:
            body.append(f" — маркеры: {', '.join(selection.evidence)}")
        body.append(f"\nМодель: {self.session.model}\n")
        body.append("Команды: /commands, /settings, /models, /domain, /context, /exit")
        self.console.print(Panel(body, title="ff.ai", style="cyan"))

    def _print_mcp_summary(self) -> None:
        """Обход реестра при старте и одна строка сводки: подробности — по команде `/mcp`."""
        with self._status(PHASE_LABELS[RequestPhase.MCP_CONNECT]) as status:

            def listener(event: object) -> None:
                if isinstance(event, PhaseChanged):
                    status.update(PHASE_LABELS.get(event.phase, PHASE_LABELS[RequestPhase.MCP_CONNECT]))

            unsubscribe = self.session.subscribe(listener)
            try:
                self.session.connect_mcp_servers()
            finally:
                unsubscribe()
        self.console.print(f"[dim]{escape(self.session.mcp_summary())}[/dim]")

    @contextmanager
    def _status(self, label: str):
        with self.console.status(label, spinner="dots") as status:
            yield status

    def _replay_history(self) -> None:
        """Показывает, что контекст восстановлен: число обменов и последний из них."""
        history = self.session.history
        if history.count() == 0:
            return
        self.console.print(
            f"[dim]Восстановлено из истории: {history.count()} обменов "
            f"(контекст подхвачен, можно продолжать диалог).[/dim]"
        )
        last = history.dialogues[-1]
        self._print_exchange(last["question"], last["answer"])

    def _print_exchange(self, question: str, answer: str) -> None:
        self.console.print(f"[bold cyan]Вы:[/bold cyan] {escape(question)}")
        self.console.print(f"[bold green]ff.ai:[/bold green] {escape(answer)}")

    def _print_answer(self, answer: str) -> None:
        if self.typing_delay <= 0:
            self.console.print(Markdown(answer))
            return
        # Печать «по живому»: тот же Markdown дорисовывается по мере поступления символов.
        with Live(Markdown(""), console=self.console, refresh_per_second=20) as live:
            shown = ""
            for index in range(0, len(answer), 3):
                shown = answer[: index + 3]
                live.update(Markdown(shown))
                time.sleep(self.typing_delay)
            live.update(Markdown(answer))

    def _print_journal(self) -> None:
        for text in self._journal:
            self.console.print(f"[dim]{escape(text)}[/dim]")
        self._journal = []

    def _print_sources(self) -> None:
        """Строка источников: доставленные фрагменты документации, на которые ответ мог опереться.

        Печатается по снимку поиска, поэтому ничего не перезапрашивает и не зависит от того,
        сослался ли на них ответ — это то, что приложение реально передало модели.
        """
        report = self.session.docs_report()
        fragments = getattr(report, "fragments", ()) or ()
        if not fragments:
            return
        rendered = "; ".join(
            f"{escape(str(getattr(item, 'identifier', '')))} — раздел "
            f"{escape(str(getattr(item, 'section', '')))}"
            for item in fragments
        )
        version = escape(str(getattr(report, "version", "") or "н/д"))
        self.console.print(f"[dim]📚 Источники (документация {version}): {rendered}[/dim]")

    def _print_code_sources(self) -> None:
        """Строка источников по коду: файл и строки каждого доставленного фрагмента."""
        report = self.session.code_report()
        fragments = getattr(report, "fragments", ()) or ()
        if not fragments:
            return
        rendered = "; ".join(escape(str(getattr(item, "identifier", ""))) for item in fragments)
        self.console.print(f"[dim]🧩 Код: {rendered}[/dim]")

    def _print_code_journal(self) -> None:
        """Строки журнала о поиске по корпусу кода: только отклонения от нормы."""
        if not self.session.code_enabled:
            return
        report = self.session.code_report()
        status = str(getattr(report, "status", "") or "")
        if status == "unavailable":
            reason = escape(str(getattr(report, "error", "") or "причина неизвестна"))
            self.console.print(
                f"[bold yellow]🧩 Корпус кода недоступен: {reason} — соберите индекс /code index[/bold yellow]"
            )
        elif status == "no_candidates":
            self.console.print("[dim]🧩 В исходниках по этому вопросу ничего не найдено.[/dim]")
        elif status == "no_matches":
            found = len(getattr(report, "candidates", ()) or ())
            threshold = self.session.code_threshold
            self.console.print(
                f"[dim]🧩 Найдено кандидатов: {found}, но ни один не прошёл порог "
                f"{threshold:.2f} — отвечаю без исходников.[/dim]"
            )
        elif status == "rerank_failed":
            reason = escape(str(getattr(report, "error", "") or "причина неизвестна"))
            self.console.print(f"[bold yellow]🧩 Отбор фрагментов кода не удался: {reason}[/bold yellow]")

    def _print_docs_journal(self) -> None:
        """Строки журнала о поиске и проверке ссылок: только отклонения от нормы."""
        report = self.session.docs_report()
        status = str(getattr(report, "status", "") or "")
        if status == "unavailable":
            reason = escape(str(getattr(report, "error", "") or "причина неизвестна"))
            self.console.print(f"[bold yellow]📚 Документация портала недоступна: {reason}[/bold yellow]")
        elif status == "no_candidates":
            self.console.print(
                "[dim]📚 В документации портала по этому вопросу ничего не найдено.[/dim]"
            )
        elif status == "no_matches":
            found = len(getattr(report, "candidates", ()) or ())
            self.console.print(
                f"[dim]📚 Найдено кандидатов: {found}, но ни один не прошёл порог "
                f"{self.session.docs_threshold:.2f} — отвечаю без документации.[/dim]"
            )
        elif status == "rerank_failed":
            reason = escape(str(getattr(report, "error", "") or "причина неизвестна"))
            self.console.print(
                f"[bold yellow]📚 Оценка фрагментов не удалась: {reason}[/bold yellow]"
            )
        check = self.session.last_citations
        if check is None:
            return
        if check.opted_out:
            self.console.print(
                "[dim]📚 Ответ без ссылки: документация к вопросу не относится.[/dim]"
            )
        elif check.replaced:
            self.console.print(
                "[bold yellow]📚 Ссылки не подтверждены: ответ заменён — источников нет.[/bold yellow]"
            )
        elif check.retried:
            self.console.print("[dim]📚 Ответ не подтверждён — повторный запрос.[/dim]")

    def _print_warnings(self, meta: Optional[AnswerMeta]) -> None:
        if meta is None:
            return
        if meta.finish_reason == "length":
            warning = TRUNCATED_EMPTY_WARNING if not meta.content.strip() else TRUNCATED_WARNING
            self.console.print(f"[bold yellow]⚠ {warning}[/bold yellow]")
        if self.session.settings.format is AnswerFormat.JSON and not is_valid_json_answer(
            meta.content
        ):
            self.console.print(
                "[bold yellow]⚠ Модель не вернула валидный JSON — ответ показан как есть.[/bold yellow]"
            )

    def _print_usage_meta(self, meta: Optional[AnswerMeta]) -> None:
        if meta is None:
            return
        cost = f"${meta.cost_usd:.6f}" if meta.cost_usd is not None else "неизвестно"
        speed = (
            f"{meta.completion_tokens / meta.elapsed_seconds:.2f} ток/сек"
            if meta.elapsed_seconds > 0 and meta.completion_tokens > 0
            else "н/д"
        )
        self.console.print(
            f"[dim]⏱ {meta.elapsed_seconds:.2f}с  |  "
            f"Токены: {meta.prompt_tokens}+{meta.completion_tokens}={meta.total_tokens}  |  "
            f"Стоимость: {cost}  |  Средняя скорость: {speed}[/dim]"
        )

    def _print_status_bar(self) -> None:
        usage = self.session.session_usage
        cost = f"${usage.cost_usd:.6f}" if usage.cost_usd is not None else "неизвестно"
        settings = self.session.settings
        self.console.print(
            f"[dim]Модель: {escape(self.session.model)}  |  "
            f"Домен: {escape(self.session.domain.id)}  |  "
            f"Формат: {FORMAT_LABELS[settings.format]}  |  "
            f"Стратегия: {STRATEGY_LABELS[settings.context_strategy]}  |  "
            f"Объём: {settings.max_words} слов  |  "
            f"Команды: {' '.join(STATUS_COMMANDS)}  |  "
            f"Сессия: {usage.total_tokens} ток., {cost}[/dim]"
        )
        self.console.print(Rule(style="dim"))

    def _exit(self) -> None:
        self.console.print(f"[bold yellow]{GOODBYE_MESSAGE}[/bold yellow]")

    # --- клавиатура и автодополнение ------------------------------------------------------

    def _register_autocomplete(self) -> None:
        if readline is None or not hasattr(readline, "set_completer"):
            return
        commands = [command for command, _ in commands_screen.COMMAND_OPTIONS]

        def complete(text: str, state: int) -> Optional[str]:
            matches = [command for command in commands if command.startswith(text)]
            return matches[state] if state < len(matches) else None

        readline.set_completer(complete)
        readline.set_completer_delims(" \t\n")
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")

    def _ensure_api_key(self) -> bool:
        """Ключ проверяется и когда он пришёл из .env: непригодный оборвал бы первый запрос."""
        if self.session.has_usable_api_key():
            return True
        while True:
            self.console.print(
                "[bold red]Ключа нет или он непригоден. Добавьте OPENCODE_API_KEY в .env "
                "или введите его здесь (для локальных моделей ключ не нужен).[/bold red]"
            )
            self.console.print("[bold]OPENCODE_API_KEY:[/bold] ")
            line = sys.stdin.readline()
            if line == "":
                raise EOFError
            entered = line.strip()
            if not entered:
                return False
            if is_valid_api_key(entered):
                self.session.set_api_key(entered)
                return True
            self.console.print(f"[bold red]{API_KEY_CHARSET_ERROR}[/bold red]")

    # --- панели ---------------------------------------------------------------------------

    def _open_commands_screen(self) -> None:
        """Панель команд: ↑/↓ — выбор, Enter — выполнить, Esc — отмена.

        Выбранная команда исполняется тем же диспетчером, что и ручной набор: панель не
        подставляет текст в строку ввода (libedit такую подстановку игнорирует).
        """
        command = "/commands"
        while command == "/commands":
            state = commands_screen.initial_state()
            with Live(
                console=self.console, refresh_per_second=30, transient=True
            ) as live, keyboard.raw_mode():
                live.update(self._render_commands_panel(state))
                while True:
                    key = keyboard.read_key()
                    state = commands_screen.apply_key(state, key)
                    if state.confirmed or state.cancelled:
                        break
                    live.update(self._render_commands_panel(state))
            if not state.confirmed:
                return
            command = state.selected[0]
        self._handle_command(command)

    def _render_commands_panel(self, state: commands_screen.CommandsScreenState) -> Panel:
        lines = []
        for index, (command, description) in enumerate(commands_screen.COMMAND_OPTIONS):
            if index == state.selected_index:
                lines.append(f"➤ [reverse bold]{command} — {description}[/reverse bold]")
            else:
                lines.append(f"  {command} — {description}")
        body = (
            "\n".join(lines) + "\n\n[dim]↑/↓ — выбор, Enter — выполнить, Esc — отмена[/dim]"
        )
        return Panel(body, title="Команды", style="cyan")

    def _open_models_screen(self) -> None:
        """Панель выбора модели: ↑/↓ — выбор, Enter — применить, Esc — отмена."""
        state = models_screen.initial_state(self.session.model)
        with Live(
            console=self.console, refresh_per_second=30, transient=True
        ) as live, keyboard.raw_mode():
            live.update(self._render_models_panel(state))
            while True:
                key = keyboard.read_key()
                state = models_screen.apply_key(state, key)
                if state.confirmed or state.cancelled:
                    break
                live.update(self._render_models_panel(state))
        if state.confirmed:
            self.session.model = state.selected

    def _render_models_panel(self, state: models_screen.ModelSelectionState) -> Panel:
        lines = []
        for index, model in enumerate(state.available):
            marker = "➤ " if index == state.selected_index else "  "
            highlight = "[reverse bold]" if index == state.selected_index else ""
            reset = "[/reverse bold]" if index == state.selected_index else ""
            suffix = " (текущая)" if model == state.current else ""
            if model in config.LOCAL_MODELS:
                suffix += " (локальная)"
            lines.append(f"{marker}{highlight}{model}{reset}{suffix}")
        body = "\n".join(lines) + "\n\n[dim]↑/↓ — выбор, Enter — применить, Esc — отмена[/dim]"
        return Panel(body, title="Модель", style="cyan")

    def _open_branches_screen(self) -> None:
        """Панель веток: ↑/↓ — выбор, Enter — переключить, «c» — чекпоинт, «n» — новая ветка."""
        state = branches_screen.initial_state(self.session.branches(), self.session.active_branch)
        with Live(
            console=self.console, refresh_per_second=30, transient=True
        ) as live, keyboard.raw_mode():
            live.update(self._render_branches_panel(state))
            while True:
                key = keyboard.read_key()
                state = branches_screen.apply_key(state, key)
                if state.finished:
                    break
                live.update(self._render_branches_panel(state))
        if state.checkpoint_requested:
            self.session.set_checkpoint()
            self.console.print("[dim]Чекпоинт поставлен в активной ветке.[/dim]")
        elif state.new_branch_requested:
            name = self.session.new_branch()
            self.console.print(f"[bold green]Создана ветка {escape(name)}.[/bold green]")
        elif state.switched:
            self.session.switch_branch(state.selected_name)

    def _render_branches_panel(self, state: branches_screen.BranchesScreenState) -> Panel:
        lines = []
        for index, (name, exchanges) in enumerate(state.branches):
            if index == state.selected_index:
                line = f"➤ [reverse bold]{name}[/reverse bold]"
            else:
                line = f"  {name}"
            suffix = " (активная)" if name == state.active else ""
            lines.append(f"{line} — обменов: {exchanges}{suffix}")
        body = (
            "\n".join(lines)
            + "\n\n[dim]↑/↓ — выбор, Enter — переключить, c — чекпоинт, "
            "n — новая ветка, Esc — закрыть[/dim]"
        )
        return Panel(body, title="Ветки диалога", style="cyan")

    def _open_settings_screen(self) -> None:
        """Экран настроек: ↑/↓ — поле, ←/→ — значения, цифры — ввод, Esc — сохранить и выйти."""
        state = settings_screen.initial_state(self.session.settings)
        with Live(
            console=self.console, refresh_per_second=30, transient=True
        ) as live, keyboard.raw_mode():
            live.update(self._render_settings_panel(state))
            while True:
                key = keyboard.read_key()
                if key == keyboard.ESC:
                    break
                state = settings_screen.apply_key(state, key)
                live.update(self._render_settings_panel(state))
        self.session.settings, errors = settings_screen.apply_to_settings(
            state, self.session.settings
        )
        for error in errors:
            self.console.print(f"[bold red]{escape(error)}[/bold red]")

    def _render_settings_panel(self, state: settings_screen.SettingsScreenState) -> Panel:
        format_line = "   ".join(
            f"[reverse bold]{FORMAT_LABELS[value]}[/reverse bold]"
            if index == state.format_index
            else FORMAT_LABELS[value]
            for index, value in enumerate(settings_screen.FORMAT_VALUES)
        )
        strategy_line = "   ".join(
            f"[reverse bold]{STRATEGY_LABELS[value]}[/reverse bold]"
            if index == state.strategy_index
            else STRATEGY_LABELS[value]
            for index, value in enumerate(settings_screen.STRATEGY_VALUES)
        )
        rows = [
            (settings_screen.ROW_FORMAT, "Формат ответа", format_line),
            (settings_screen.ROW_STRATEGY, "Стратегия контекста", strategy_line),
            (settings_screen.ROW_MAX_WORDS, "Объём ответа, слов", state.max_words_input),
            (settings_screen.ROW_LIST_LIMIT, "Лимит списка", state.list_limit_input),
            (settings_screen.ROW_TEMPERATURE, "Температура", state.temperature_input),
            (
                settings_screen.ROW_COMPRESS_AFTER,
                "Порог сжатия, сообщений",
                state.compress_after_input,
            ),
            (
                settings_screen.ROW_MAX_SESSION_TOKENS,
                "Потолок запроса, токенов",
                state.max_session_tokens_input,
            ),
        ]
        lines = []
        for row, label, value in rows:
            marker = "➤ " if state.row == row else "  "
            lines.append(f"{marker}{label}: {value}")
        body = (
            "\n".join(lines)
            + "\n\n[dim]↑/↓ — поле, ←/→ — формат и стратегия, цифры — ввод, "
            "Esc — сохранить и выйти[/dim]"
        )
        return Panel(body, title="Настройки", style="cyan")
