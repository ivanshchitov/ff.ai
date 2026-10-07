"""Конвейер задачи: одна операция на вызов — план, подзадача, проверка.

Конвейер не знает ни терминала, ни HTTP: запрос приходит функцией `ask`, подготовка патча —
функцией `prepare` (в приложении она проверяет патч правилами домена и спрашивает подтверждение),
сборка — функцией `build`. Благодаря этому каждая ветка этапов проверяется на поддельных запросах,
а пауза безопасна: состояние записывается до следующего запроса.

Модель поставляет содержание — план, патч, замечания — и никогда не выбирает переход: этапами
управляет таблица `core/task_state.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import config, domain_checks, patches, task_state
from .api_client import AnswerMeta, APIError
from .domains import Domain
from .task_state import Stage, TaskIssue, TaskItem, TaskPatch, TaskState

PLAN_ASSET = "task_plan_prompt.md"
EXECUTE_ASSET = "task_execute_prompt.md"
VALIDATE_ASSET = "task_validate_prompt.md"

_FENCED = re.compile(r"```(?:diff|patch)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_DIFF_START = re.compile(r"^(---|\+\+\+|diff --git) ", re.MULTILINE)

EXECUTE_FAILED_EMPTY = "модель вернула пустой ответ"
EXECUTE_FAILED_NO_PATCH = "в ответе нет патча"
EXECUTE_FAILED_TRUNCATED = "ответ обрезан техническим потолком запроса"


@dataclass(frozen=True)
class TaskStepReport:
    """Итог одной операции: этап, подпись для панели, журнальные строки и расход запроса."""

    stage: Stage
    label: str
    notes: Tuple[str, ...] = ()
    meta: Optional[AnswerMeta] = None
    finished: bool = False
    awaiting_edits: bool = False

    @property
    def lines(self) -> Tuple[str, ...]:
        return self.notes


def asset_text(name: str) -> str:
    return (config.ASSETS_DIR / name).read_text(encoding="utf-8").strip()


def parse_plan_response(text: str) -> Tuple[str, ...]:
    """Подзадачи из ответа: JSON-объект с `items`, иначе — строки списка.

    Разбор клиентский и терпимый: рассуждающие модели любят обрамлять ответ пояснениями, а строгий
    формат здесь не несёт смысла — список подзадач одинаково читается и как JSON, и как перечень.
    """
    data = _first_json(text)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = [str(item).strip() for item in data["items"] if str(item).strip()]
        if items:
            return tuple(items)
    if isinstance(data, list):
        items = [str(item).strip() for item in data if str(item).strip()]
        if items:
            return tuple(items)
    numbered = re.findall(r"^\s*(?:\d+[.)]|[-*])\s+(.+)$", text or "", re.MULTILINE)
    if numbered:
        return tuple(item.strip() for item in numbered if item.strip())
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return tuple(lines[: config.MAX_PLAN_ITEMS])


def parse_validation_response(text: str) -> Tuple[Tuple[TaskIssue, ...], Tuple[str, ...]]:
    """Замечания и новая работа из ответа проверки.

    Замечание без номера подзадачи относится к задаче целиком: так общее замечание («в артефакте
    нет тестов») не теряется из-за отсутствия номера.
    """
    data = _first_json(text)
    if not isinstance(data, dict):
        issues = [
            TaskIssue(index=None, text=line.strip())
            for line in (text or "").splitlines()
            if line.strip()
        ]
        return tuple(issues[:10]), ()
    issues: List[TaskIssue] = []
    for item in data.get("issues", []) or []:
        if isinstance(item, dict):
            index = item.get("item")
            text_value = str(item.get("text", "")).strip()
            if not text_value:
                continue
            issues.append(
                TaskIssue(
                    index=int(index) if isinstance(index, int) and not isinstance(index, bool) else None,
                    text=text_value,
                )
            )
        elif isinstance(item, str) and item.strip():
            issues.append(TaskIssue(index=None, text=item.strip()))
    new_items = [
        str(item).strip() for item in data.get("items", []) or [] if str(item).strip()
    ]
    return tuple(issues), tuple(new_items)


def parse_patch_response(text: str) -> str:
    """Патч из ответа модели: блок ```, иначе всё, что начинается с заголовков diff-а."""
    for candidate in _FENCED.findall(text or ""):
        if _DIFF_START.search(candidate):
            return candidate.strip() + "\n"
    match = _DIFF_START.search(text or "")
    if match:
        return (text or "")[match.start() :].strip() + "\n"
    return ""


def _first_json(text: str) -> object:
    for candidate in [*_FENCED.findall(text or ""), text or ""]:
        match = _JSON_OBJECT.search(candidate)
        if match is None:
            continue
        try:
            return json.loads(match.group(0))
        except ValueError:
            continue
    return None


class TaskPipeline:
    """Одна операция за вызов: планирование, выполнение подзадачи, проверка.

    `ask(messages, max_words, phase)` отдаёт ответ модели, `prepare(index, patch)` готовит патч к
    записи (проверки плюс подтверждение пользователя), `build(command)` выполняет команду сборки,
    `on_change(state)` получает каждое новое состояние — им пользуется панель прогона.
    """

    def __init__(
        self,
        domain: Domain,
        root: Path,
        store: Optional[task_state.TaskStore] = None,
        ask: Optional[Callable[[List[Dict[str, str]], int, str], AnswerMeta]] = None,
        prepare: Optional[Callable[[int, str], TaskPatch]] = None,
        build: Optional[Callable[[str], Tuple[bool, str]]] = None,
        tasks_dir: Optional[Path] = None,
    ) -> None:
        self.domain = domain
        self.root = Path(root)
        self.store = store if store is not None else task_state.TaskStore()
        self._ask = ask
        self._prepare = prepare
        self._build = build
        self.tasks_dir = Path(tasks_dir or config.TASKS_DIR)
        self.state = self.store.load()

    # --- состояние --------------------------------------------------------------------------

    def _save(self, state: TaskState) -> TaskState:
        self.state = state
        self.store.save(state)
        return state

    def add(self, goal: str) -> TaskState:
        return self._save(task_state.add_task(self.state, goal))

    def stop(self) -> TaskState:
        return self._save(task_state.drop_all(self.state))

    def set_paused(self, paused: bool) -> TaskState:
        return self._save(task_state.set_paused(self.state, paused))

    def request_stage(self, name: str) -> task_state.StageRequestResult:
        result = task_state.request_stage(self.state, name)
        if result.state is not None:
            self._save(result.state)
        return result

    def answer_edits(self, text: str) -> TaskState:
        return self._save(task_state.edits_response(self.state, text))

    @property
    def paused(self) -> bool:
        return self.state.paused

    def unfinished(self) -> Tuple[TaskItem, ...]:
        return self.state.unfinished()

    # --- одна операция ----------------------------------------------------------------------

    def step(self) -> Optional[TaskStepReport]:
        """Выполняет ровно одну операцию текущего этапа; `None` — очередь пуста."""
        state = self.state
        if state.current is None or state.current.status != task_state.TASK_STATUS_RUNNING:
            state = self._save(task_state.begin_task(state))
        task = state.current
        if task is None:
            return None
        if task.stage is Stage.PLANNING:
            return self._planning(task, state)
        if task.stage is Stage.EXECUTION:
            return self._execution(task, state)
        if task.stage is Stage.VALIDATION:
            return self._validation(task, state)
        self._finish(task)
        return TaskStepReport(
            stage=Stage.DONE, label=f"Задача {task.number} завершена", finished=True
        )

    def _planning(self, task: TaskItem, state: TaskState) -> TaskStepReport:
        if task.awaiting_edits:
            # Вопрос задаётся терминальным слоем: конвейер только сообщает, что ждёт ответа.
            return TaskStepReport(
                stage=Stage.PLANNING,
                label="План построен — жду правок",
                notes=(f"План ({len(task.plan)}): " + "; ".join(task.plan),),
                awaiting_edits=True,
            )
        messages = [
            {"role": "system", "content": task_state.task_instruction(self.domain)},
            {"role": "user", "content": self._plan_request(task, state)},
        ]
        meta = self._request(messages, config.TASK_PLAN_MAX_WORDS, "task_plan")
        plan = parse_plan_response(meta.content)
        if not plan:
            state = self._save(task_state.fail_task(state, EXECUTE_FAILED_EMPTY))
            return TaskStepReport(
                stage=Stage.PLANNING, label="План не построен", notes=(EXECUTE_FAILED_EMPTY,), meta=meta
            )
        state = self._save(task_state.plan_built(self.state, plan))
        return TaskStepReport(
            stage=Stage.PLANNING,
            label=f"План: подзадач {len(state.current.plan)}",
            notes=tuple(
                f"{index}. {item}" for index, item in enumerate(state.current.plan, start=1)
            ),
            meta=meta,
            awaiting_edits=True,
        )

    def _execution(self, task: TaskItem, state: TaskState) -> TaskStepReport:
        index = task_state.next_item_index(task)
        if index is None:
            moved = task_state.transition(state, Stage.VALIDATION, "все подзадачи выполнены")
            self._save(moved.state)
            return TaskStepReport(
                stage=Stage.VALIDATION, label="Все подзадачи выполнены — проверяю"
            )
        messages = [
            {"role": "system", "content": task_state.task_instruction(self.domain)},
            {"role": "user", "content": self._execute_request(task, index)},
        ]
        meta = self._request(messages, config.TASK_PATCH_MAX_WORDS, "task_execute")
        if not (meta.content or "").strip():
            reason = EXECUTE_FAILED_EMPTY
            if meta.finish_reason == "length":
                reason = EXECUTE_FAILED_TRUNCATED
            return TaskStepReport(
                stage=Stage.EXECUTION,
                label=f"Подзадача {index}: {reason}",
                notes=(reason,),
                meta=meta,
            )
        patch_text = parse_patch_response(meta.content)
        if not patch_text:
            prepared = TaskPatch(
                index=index, text="", summary="", applied=False, reason=EXECUTE_FAILED_NO_PATCH
            )
            self._save(task_state.record_patch(self.state, prepared))
            return TaskStepReport(
                stage=Stage.EXECUTION,
                label=f"Подзадача {index}: патча нет",
                notes=(EXECUTE_FAILED_NO_PATCH,),
                meta=meta,
            )
        prepared = (
            self._prepare(index, patch_text)
            if self._prepare is not None
            else self._default_prepare(index, patch_text)
        )
        state = self._save(task_state.record_patch(self.state, prepared))
        note = (
            f"Патч применён: {prepared.summary}"
            if prepared.applied
            else f"Патч не применён: {prepared.reason or 'причина не названа'}"
        )
        return TaskStepReport(
            stage=Stage.EXECUTION,
            label=f"Подзадача {index}/{len(task.plan)}: {task.plan[index - 1]}",
            notes=(note,),
            meta=meta,
        )

    def _validation(self, task: TaskItem, state: TaskState) -> TaskStepReport:
        artifact_issues, skipped, build_output = self._check_artifact(task)
        notes = tuple(f"⛔ {issue.line()}" for issue in artifact_issues)
        notes += tuple(f"⏭ {item}" for item in skipped)
        if build_output:
            notes += (f"🔨 {build_output}",)

        messages = [
            {"role": "system", "content": task_state.task_instruction(self.domain)},
            {"role": "user", "content": self._validate_request(task, artifact_issues)},
        ]
        meta = self._request(messages, config.TASK_VALIDATE_MAX_WORDS, "task_validate", tolerant=True)
        review_issues: Tuple[TaskIssue, ...] = ()
        new_items: Tuple[str, ...] = ()
        if meta is None:
            notes += ("⏭ модельная проверка недоступна — вердикт проверок в силе",)
        else:
            review_issues, new_items = parse_validation_response(meta.content)
            notes += tuple(f"📝 {issue.line()}" for issue in review_issues)

        issues = artifact_issues + review_issues
        final = task_state.validation_verdict(self.state, issues, new_items)
        done = final.current is not None and final.current.stage is Stage.DONE
        state = self._save(final)
        if done:
            self._finish(state.current)
        return TaskStepReport(
            stage=Stage.DONE if done else Stage.EXECUTION,
            label=(
                f"Проверка: замечаний {len(issues)}"
                if issues
                else "Проверка: замечаний нет"
            ),
            notes=notes,
            meta=meta,
            finished=done,
        )

    def _finish(self, task: TaskItem) -> Optional[Path]:
        """Пишет отчёт о задаче и запоминает его путь: ошибка записи прогон не прерывает."""
        if task is None or task.result_path:
            return None
        path = task_state.write_result(self.tasks_dir, task.number, task)
        if path is None:
            return None
        updated = task_state._put_active(self.state, replace(task, result_path=str(path)))
        self._save(updated)
        return path

    # --- проверки артефакта ------------------------------------------------------------------

    def _check_artifact(self, task: TaskItem) -> Tuple[Tuple[TaskIssue, ...], Tuple[str, ...], str]:
        """Проверяет всё, что прогон изменил в проекте: патчи применены — значит и проверять надо их.

        Проверка идёт по расхождению рабочего дерева с HEAD, а не по текстам патчей: только так
        видно результат, а не намерение.
        """
        checks = getattr(self.domain, "checks", None)
        if checks is None:
            return (), ("домен не объявляет проверок",), ""
        # Источник истины — применённые патчи прогона: они лежат в состоянии задачи и не зависят
        # от того, что ещё успел изменить пользователь в рабочем дереве. Расхождение с HEAD
        # добавляется к проверке: правки, сделанные мимо конвейера, тоже часть артефакта.
        artifact = "\n".join(patch.text for patch in task.applied_patches())
        diff = self._working_tree_diff()
        if diff.strip():
            artifact = (artifact + "\n" + diff).strip()
        if not artifact.strip():
            return (), (), ""
        result = domain_checks.check_patch(
            artifact, self.root, checks, build=self._build, include_apply=False
        )
        issues = tuple(TaskIssue(index=None, text=issue.line()) for issue in result.issues)
        return issues, tuple(result.skipped), result.build_output

    def _working_tree_diff(self) -> str:
        import subprocess

        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "diff", "--no-color", "--no-ext-diff"],
                capture_output=True,
                timeout=patches.GIT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout.decode("utf-8", errors="replace")

    # --- запросы -----------------------------------------------------------------------------

    def _default_prepare(self, index: int, patch_text: str) -> TaskPatch:
        """Готовит патч без подтверждения: применяет, если детерминированные проверки прошли.

        Так конвейер остаётся работоспособным без терминала (тесты и библиотечное использование);
        приложение подставляет сюда проверку с вопросом пользователю.
        """
        checks = getattr(self.domain, "checks", None)
        if checks is not None:
            result = domain_checks.check_patch(patch_text, self.root, checks)
            if not result.ok:
                return TaskPatch(
                    index=index,
                    text=patch_text,
                    summary=patches.parse(patch_text).summary(),
                    applied=False,
                    reason="; ".join(issue.line() for issue in result.issues),
                )
        reason = patches.apply(self.root, patch_text)
        summary = patches.parse(patch_text).summary()
        return TaskPatch(
            index=index,
            text=patch_text,
            summary=summary,
            applied=not reason,
            reason=reason,
        )

    def _request(
        self,
        messages: List[Dict[str, str]],
        max_words: int,
        phase: str,
        tolerant: bool = False,
    ) -> Optional[AnswerMeta]:
        """Один запрос этапа. `tolerant` — для проверки: её сбой не отменяет вердикт проверок."""
        if self._ask is None:
            if tolerant:
                return None
            raise APIError("конвейер задачи не получил источник запросов")
        try:
            return self._ask(messages, max_words, phase)
        except APIError:
            if tolerant:
                return None
            raise

    def _plan_request(self, task: TaskItem, state: TaskState) -> str:
        asset = asset_text(PLAN_ASSET)
        limit = task.plan_limit
        return (
            f"{asset}\n\n"
            f"Цель задачи: {task.goal}\n"
            f"Число подзадач: от 3 до {limit}\n"
            f"Формулировка подзадачи — одна строка, до {config.TASK_GOAL_MAX_CHARS} символов\n"
            f"Объём ответа — не больше {config.TASK_PLAN_MAX_WORDS} слов"
        )

    def _execute_request(self, task: TaskItem, index: int) -> str:
        asset = asset_text(EXECUTE_ASSET)
        issues = "\n".join(
            issue.line() for issue in task.issues if issue.index in (None, index)
        )
        parts = [
            asset,
            f"Цель задачи: {task.goal}",
            "План:\n"
            + "\n".join(
                f"{number}. {'готово' if (task.patch_for(number) and task.patch_for(number).applied) else 'не выполнено'}: {item}"
                for number, item in enumerate(task.plan, start=1)
            ),
            f"Выполни подзадачу {index}: {task.plan[index - 1]}",
        ]
        if issues:
            parts.append("Замечания проверки, которые нужно устранить:\n" + issues)
        parts.append(f"Объём патча — не больше {config.TASK_PATCH_MAX_WORDS} слов")
        return "\n\n".join(parts)

    def _validate_request(self, task: TaskItem, artifact_issues: Sequence[TaskIssue]) -> str:
        asset = asset_text(VALIDATE_ASSET)
        parts = [
            asset,
            f"Цель задачи: {task.goal}",
            "План и состояние:\n"
            + "\n".join(
                f"{number}. {'выполнено' if (task.patch_for(number) and task.patch_for(number).applied) else 'не выполнено'}: {item}"
                for number, item in enumerate(task.plan, start=1)
            ),
        ]
        if artifact_issues:
            parts.append(
                "Замечания проверок (уже найдены, не повторяй их):\n"
                + "\n".join(issue.line() for issue in artifact_issues)
            )
        parts.append(f"Объём ответа — не больше {config.TASK_VALIDATE_MAX_WORDS} слов")
        return "\n\n".join(parts)
