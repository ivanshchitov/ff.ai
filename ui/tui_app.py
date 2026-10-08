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
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from core import config, task_state
from core.agent import CompressionReport, FactsReport, RequestPhase
from core.answer_settings import AnswerFormat, ContextStrategy
from core.api_client import APIError, AnswerMeta, is_valid_api_key, is_valid_json_answer
from core import invariants, memory_layers
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
    RequestPhase.TOOL_CHOICE: "Выбор инструментов...",
    RequestPhase.TASK_PLAN: "Планирование задачи...",
    RequestPhase.TASK_EXECUTE: "Выполнение подзадачи...",
    RequestPhase.TASK_VALIDATE: "Проверка артефакта...",
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
        self.session.set_profile_interview(self._run_profile_setup)
        self._run_phase: str = ""
        # Подтверждение записи в репозиторий даёт только интерфейс: без него патчи не применяются.
        self.session.set_patch_confirmation(self._confirm_patch)
        # Конвейер операций: процессы запускает приложение, необратимые шаги подтверждает оно же.
        self.session.set_ops_runner(self._run_process)
        self.session.set_ops_confirmation(self._confirm_ops_step)
        self._register_autocomplete()

    # --- основной цикл -------------------------------------------------------------------

    def run(self) -> int:
        self._print_welcome()
        self._print_mcp_summary()
        self._print_schedule_line()
        self._replay_history()
        self._print_status_bar()
        try:
            while not self.session.exit_requested:
                try:
                    line = input(INPUT_PROMPT)
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    # Пустой Enter — тоже ход цикла: статусная строка должна оставаться на экране.
                    self._print_status_bar()
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
        if result.task_run:
            self._run_task()

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
        self._print_tool_journal()
        self._print_memory_journal()
        self._print_warnings(answer.meta)
        self._print_usage_meta(answer.meta)
        self._print_schedule_announcement()

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

    def _print_schedule_line(self) -> None:
        """Строка расписания при старте: что уже стоит в файле, без запуска заданий."""
        self.console.print(f"[dim]{escape(self.session.schedule_startup_line())}[/dim]")

    def _print_schedule_announcement(self) -> None:
        """Объявление о прогонах планировщика: только о тех, которых пользователь ещё не видел."""
        for line in self.session.schedule_announcement():
            self.console.print(f"[dim]{escape(line)}[/dim]")

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

    def _print_tool_journal(self) -> None:
        """Строки журнала о флоу инструментов: по строке на шаг, плюс причина остановки.

        Ничего не печатается, если модель решила, что инструменты не нужны: это обычный исход, а не
        событие. Объёмы переданного текста в строке шага — чтобы передачу данных можно было
        проверить глазами.
        """
        report = self.session.tool_flow_report()
        if report is None or not report.steps:
            return
        for step in report.steps:
            self.console.print(f"[dim]{escape(step.journal_line())}[/dim]")
        failure_markers = ("сбой", "исчерпан")
        if any(marker in report.stop_reason for marker in failure_markers):
            self.console.print(
                f"[bold yellow]🔧 Флоу остановлен: {escape(report.stop_reason)}[/bold yellow]"
            )

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
                f"[bold yellow]🧩 Корпус кода недоступен: {reason} — соберите индекс /rag-code index[/bold yellow]"
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

    # --- прогон задачи ---------------------------------------------------------------------

    def _run_task(self) -> None:
        """Прогон задачи: операции по одной под панелью, пауза на границе операции.

        Пауза (клавиша p или Ctrl+C) останавливает прогон между операциями: текущий запрос
        завершается, состояние задачи уже записано, и продолжение возможно и здесь, и после
        перезапуска приложения.
        """
        paused = False
        try:
            self.session.task_set_paused(False)
            with keyboard.raw_mode():
                with Live(
                    self._task_frame(""),
                    console=self.console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    while True:
                        live.update(self._task_frame(""))
                        key = self._drain_pause_key()
                        if key == "pause":
                            paused = True
                            break
                        report = self.session.task_step()
                        if report is None:
                            break
                        for text in report.notes:
                            self.console.print(escape(text))
                        live.update(self._task_frame(report.label))
                        if report.awaiting_edits:
                            answer = self._ask_plan_edits(live)
                            self.session.task_answer_edits(answer)
                            continue
                        if report.finished:
                            self._print_task_result()
                            break
        except KeyboardInterrupt:
            # Прерывание во время прогона — это пауза, а не выход: задача не должна теряться.
            paused = True
        finally:
            if paused:
                self.session.task_set_paused(True)
                self.console.print(
                    "[bold yellow]Прогон остановлен на паузе: состояние сохранено, "
                    "продолжить — /task run[/bold yellow]"
                )
            self._print_status_bar()

    def _drain_pause_key(self) -> str:
        """Читает клавиши, накопленные с прошлой операции: пауза по p или Ctrl+C."""
        while True:
            try:
                key = keyboard.read_key_nowait()
            except KeyboardInterrupt:
                return "pause"
            if not key:
                return ""
            if key in ("p", "P", "з", "З") or key == keyboard.ESC:
                return "pause"

    def _task_frame(self, label: str) -> Group:
        """Панель прогона: этапы, шаг, ожидаемое действие, план с отметками и подсказка о паузе."""
        state = self.session.task_state
        task = state.current
        if task is None:
            return Group(Panel("Очередь задач пуста.", title="Задача"))
        stages = " → ".join(
            f"[bold]{title}[/bold]" if stage is task.stage else title
            for stage, title in (
                (task_state.Stage.PLANNING, "Планирование"),
                (task_state.Stage.EXECUTION, "Выполнение"),
                (task_state.Stage.VALIDATION, "Проверка"),
                (task_state.Stage.DONE, "Завершено"),
            )
        )
        lines = [stages, ""]
        lines.append(f"Шаг: {escape(task.current_step)}")
        lines.append(f"Ожидается: {escape(task.expected_action)}")
        if label:
            lines.append(f"Сейчас: {escape(label)}")
        if task.plan:
            lines.append("")
            for index, item in enumerate(task.plan, start=1):
                patch = task.patch_for(index)
                mark = "[x]" if patch is not None and patch.applied else "[ ]"
                lines.append(f"{escape(mark)} {index}. {escape(item)}")
        if task.issues:
            lines.append("")
            lines.extend(f"⛔ {escape(issue.line())}" for issue in task.issues)
        lines.append("")
        lines.append("Пауза — клавиша p или Ctrl+C; Enter — выполнить введённое")
        title = f"Задача {task.number} из {len(state.tasks)}"
        return Group(
            Panel("\n".join(lines), title=title, border_style="cyan"),
            Rule(style="dim"),
            self._status_line(),
            Rule(style="dim"),
        )

    def _status_line(self) -> Text:
        """Та же строка состояния, но как элемент кадра прогона."""
        return Text.from_markup(self._status_bar_text())

    def _ask_plan_edits(self, live: Live) -> str:
        """Вопрос о правках плана набирается посимвольно: панель показывает набираемое.

        `readline` внутри прогона не годится: терминал уже в cbreak, эха нет, и набранный текст
        был бы невидим; останавливать панель ради вопроса — противоречит «панель показывает
        состояние всегда».
        """
        typed: List[str] = []
        self.console.print("[bold]План построен. Правки (пустая строка — выполнять):[/bold]")
        while True:
            live.update(self._task_frame("Правки к плану: " + "".join(typed)))
            try:
                key = keyboard.read_char()
            except KeyboardInterrupt:
                return ""
            if key == keyboard.ENTER:
                return "".join(typed)
            if key == keyboard.BACKSPACE:
                if typed:
                    typed.pop()
                continue
            if key and key not in (keyboard.ESC, keyboard.UP, keyboard.DOWN):
                typed.append(key)

    def _run_process(self, argv, cwd, timeout):
        """Запуск процесса конвейера: списком аргументов, без оболочки, со снимком времени."""
        import subprocess

        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 124, f"превышено время ожидания ({timeout} с)", time.monotonic() - started
        except OSError as error:
            return 127, str(error), time.monotonic() - started
        output = (completed.stdout or "") + (completed.stderr or "")
        return completed.returncode, output, time.monotonic() - started

    def _confirm_ops_step(self, step: str, description: str) -> bool:
        """Подтверждение необратимого шага конвейера: один ответ, чтение вне построчного ввода."""
        self.console.print(f"[bold]Шаг «{escape(step)}»: {escape(description)}[/bold]")
        self.console.print("Выполнить? y — да, n — нет")
        try:
            with keyboard.raw_mode():
                key = keyboard.read_char().strip().casefold()
        except KeyboardInterrupt:
            return False
        return key in ("y", "н", "д")

    def _confirm_patch(self, summary: str, patch_text: str) -> bool:
        """Подтверждение записи в репозиторий: показывает патч и читает один ответ."""
        self.console.print(f"[bold]Патч к применению: {escape(summary)}[/bold]")
        preview = patch_text.strip().splitlines()[:20]
        for line in preview:
            self.console.print(f"[dim]{escape(line)}[/dim]")
        if len(patch_text.strip().splitlines()) > len(preview):
            self.console.print("[dim]… патч показан не целиком[/dim]")
        self.console.print("Применить патч? y — да, n — нет")
        try:
            key = keyboard.read_char().strip().casefold()
        except KeyboardInterrupt:
            return False
        return key in ("y", "н", "д")

    def _print_task_result(self) -> None:
        state = self.session.task_state
        task = state.current
        if task is None:
            return
        self.console.print(
            f"[bold green]Задача {task.number}: {escape(task.title)}[/bold green]"
        )
        if task.result_path:
            self.console.print(f"[dim]Отчёт: {escape(task.result_path)}[/dim]")

    def _status_bar_text(self) -> str:
        """Текст строки состояния: один источник и для печатной полосы, и для кадра прогона."""
        usage = self.session.session_usage
        cost = f"${usage.cost_usd:.6f}" if usage.cost_usd is not None else "неизвестно"
        branch = self.session.branch
        return (
            f"[dim]Модель: {escape(self.session.model)}  |  "
            f"Домен: {escape(self.session.domain.id)}  |  "
            f"Проект: {escape(self.session.project_name)}  |  "
            + (f"Ветка: {escape(branch)}  |  " if branch else "")
            + self._task_status_fragment()
            + self._memory_status_fragment()
            + f"Команды: {' '.join(STATUS_COMMANDS)}  |  "
            f"Сессия: {usage.total_tokens} ток., {cost}[/dim]"
        )

    def _print_status_bar(self) -> None:
        """Строка состояния полосой: линия сверху закрывает предыдущий блок, линия снизу отделяет
        её от приглашения — иначе строка читается хвостом вывода.
        """
        self.console.print(Rule(style="dim"))
        self.console.print(self._status_bar_text())
        self.console.print(Rule(style="dim"))

    def _memory_status_fragment(self) -> str:
        """Цель задачи и имя активного профиля — только когда они есть: иначе layout не меняется."""
        parts = []
        goal = str(self.session.memory_report().get("goal", "") or "")
        if goal:
            parts.append(f"Цель: «{escape(goal)}»")
        name = str(self.session.profile_report().get("active", "") or "")
        if name:
            parts.append(f"Профиль: {escape(name)}")
        return ("  |  " + "  |  ".join(parts)) if parts else ""

    def _print_memory_journal(self) -> None:
        """Журнал памяти и правил: строки только об отклонениях и о сделанных записях."""
        for record in self.session.agent_routing():
            layer = memory_layers.LAYER_LABELS.get(record.layer, record.layer)
            self.console.print(
                f"[dim]🧠 Память ({escape(layer)}, {escape(record.key)}): "
                f"{escape(record.value)}[/dim]"
            )
        violations = self.session.agent_invariants()
        if violations:
            for item in violations:
                self.console.print(
                    f"[bold red]⛔ Инвариант {item.number} нарушен («{escape(item.term)}»): "
                    f"ответ заменён[/bold red]"
                )
            self.console.print(f"[dim]{escape(invariants.refusal_text(violations))}[/dim]")

    def _run_profile_setup(self) -> None:
        """Опросник профиля: вопросы печатаются, ответы читаются построчно.

        `readline`, а не `input()`: прерывание не должно обрывать приложение — оно отменяет только
        настройку профиля.
        """
        questions = self.session.profile_questions()
        if not questions:
            self.console.print("[bold yellow]Домен не объявляет разделов профиля.[/bold yellow]")
            return
        answers: list = []
        self.console.print(
            "[dim]Настройка профиля: пустой ответ оставляет раздел как был, Ctrl+C отменяет.[/dim]"
        )
        try:
            for field, prompt, default in questions:
                suffix = f" [{escape(default)}]" if default else ""
                self.console.print(f"{escape(prompt)}{suffix}")
                line = sys.stdin.readline()
                if not line:
                    break
                answers.append((field, line.strip()))
        except KeyboardInterrupt:
            self.console.print("[bold yellow]Настройка профиля отменена.[/bold yellow]")
            return
        name = self.session.save_profile(answers)
        self.console.print(f"[bold green]Профиль «{escape(name)}» настроен.[/bold green]")

    def _task_status_fragment(self) -> str:
        """Строка задачи в статус-баре: только пока задача незавершена — иначе layout не меняется."""
        state = self.session.task_state
        task = state.current
        if task is None or not state.unfinished():
            return ""
        stage = task_state.STAGE_TITLES[task.stage]
        return f"Задача: {escape(str(task.number))} ({escape(stage)})  |  "

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
        if self.session.has_usable_api_key() or config.is_local_model(self.session.model):
            # Локальному пресету ключ не нужен: запрос уходит на llama.cpp без авторизации.
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
