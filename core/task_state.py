"""Задача как процесс: этапы, переходы, план, патчи, попытки и отчёт.

Здесь живёт вся логика задачи без терминала, модели и сети: этапы меняются только через таблицу
переходов и её шлюз, план и патчи лежат в состоянии, журнал переходов едет вместе с задачей.
Конвейер (`core/task_pipeline.py`) вызывает эти функции по одной операции за раз, а терминальный
слой печатает снимки — так ветки процесса проверяются без терминала.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config
from .domains import Domain

TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"

# Причины отказа перехода — данные: их видит пользователь и хранит журнал задачи.
REASON_NO_TASK = "незавершённой задачи нет — менять нечего"
REASON_TERMINAL = "задача завершена: переходов из этого этапа нет"
REASON_SAME_STAGE = "задача уже на этапе {stage}"
REASON_NOT_ALLOWED = "переход {source} → {target} не разрешён таблицей"
REASON_PLAN_NOT_APPROVED = "план не утверждён: выполнение до утверждённого плана запрещено"
REASON_NEEDS_VALIDATION = "финал без проверки: в завершение можно только из проверки"


class Stage(str, Enum):
    """Этапы задачи: планирование, выполнение, проверка, завершение."""

    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


STAGE_TITLES = {
    Stage.PLANNING: "Планирование",
    Stage.EXECUTION: "Выполнение",
    Stage.VALIDATION: "Проверка",
    Stage.DONE: "Завершено",
}

STAGE_ORDER = (Stage.PLANNING, Stage.EXECUTION, Stage.VALIDATION, Stage.DONE)

# Таблица переходов: единственное описание жизненного цикла задачи.
TRANSITIONS: Dict[Stage, Tuple[Stage, ...]] = {
    Stage.PLANNING: (Stage.EXECUTION,),
    Stage.EXECUTION: (Stage.VALIDATION, Stage.PLANNING),
    Stage.VALIDATION: (Stage.DONE, Stage.EXECUTION),
    Stage.DONE: (),
}


class TaskStateError(Exception):
    """Состояние задачи не может быть прочитано или записано."""


@dataclass(frozen=True)
class Transition:
    """Запись журнала: откуда, куда, почему и принято ли."""

    source: str
    target: str
    reason: str
    accepted: bool

    def line(self) -> str:
        mark = "→" if self.accepted else "✗"
        return f"{mark} {self.source} → {self.target}: {self.reason}"


@dataclass(frozen=True)
class TaskPatch:
    """Патч подзадачи: текст, подпись, применён ли и почему нет."""

    index: int
    text: str
    summary: str
    applied: bool = False
    reason: str = ""

    @property
    def recorded(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True)
class TaskIssue:
    """Замечание проверки: к какой подзадаче относится и в чём оно."""

    index: Optional[int]
    text: str

    def line(self) -> str:
        where = f"подзадача {self.index}" if self.index else "задача"
        return f"{where}: {self.text}"


@dataclass(frozen=True)
class StageRequestResult:
    """Итог просьбы пользователя сменить этап: принято или причина отказа с допустимыми переходами."""

    accepted: bool
    stage: Stage
    reason: str = ""
    allowed: Tuple[Stage, ...] = ()
    state: Optional["TaskState"] = None


@dataclass(frozen=True)
class TaskItem:
    """Задача: цель, этап, план, патчи, замечания и журнал переходов."""

    number: int
    goal: str
    stage: Stage = Stage.PLANNING
    status: str = TASK_STATUS_PENDING
    plan: Tuple[str, ...] = ()
    plan_limit: int = config.MAX_PLAN_ITEMS
    plan_rounds: int = 0
    awaiting_edits: bool = False
    replan: bool = False
    patches: Tuple[TaskPatch, ...] = ()
    issues: Tuple[TaskIssue, ...] = ()
    attempts: int = 0
    transitions: Tuple[Transition, ...] = ()
    result_path: str = ""
    fail_reason: str = ""

    @property
    def title(self) -> str:
        return STAGE_TITLES[self.stage]

    def patch_for(self, index: int) -> Optional[TaskPatch]:
        return next((item for item in self.patches if item.index == index), None)

    def applied_patches(self) -> Tuple[TaskPatch, ...]:
        return tuple(item for item in self.patches if item.applied)

    def open_issues(self) -> Tuple[TaskIssue, ...]:
        return self.issues

    @property
    def current_step(self) -> str:
        """Текущий шаг — производное от этапа и плана, а не хранимое поле: ему нечем разойтись."""
        if self.stage is Stage.PLANNING:
            if self.awaiting_edits:
                return "ожидание правок пользователя"
            return f"построение плана (раунд {self.plan_rounds + 1} из {config.MAX_PLAN_ROUNDS})"
        if self.stage is Stage.EXECUTION:
            index = next_item_index(self)
            if index is None:
                return "выполнение завершено"
            return f"подзадача {index}: {self.plan[index - 1]}"
        if self.stage is Stage.VALIDATION:
            return f"проверка артефакта (попытка {self.attempts + 1} из {config.MAX_VALIDATION_ATTEMPTS})"
        return "задача завершена"

    @property
    def expected_action(self) -> str:
        """Ожидаемое действие — подсказка, что произойдёт на следующей операции."""
        if self.stage is Stage.PLANNING:
            return (
                "ответить на вопрос о правках плана"
                if self.awaiting_edits
                else "построить план по цели"
            )
        if self.stage is Stage.EXECUTION:
            return "выполнить подзадачу патчем"
        if self.stage is Stage.VALIDATION:
            return "проверить артефакт и назвать замечания"
        return "—"


@dataclass(frozen=True)
class TaskState:
    """Очередь задач и активная задача: то, что сохраняется в `task.json`."""

    tasks: Tuple[TaskItem, ...] = ()
    active: int = 0
    paused: bool = False
    next_number: int = 1

    @property
    def current(self) -> Optional[TaskItem]:
        if not self.tasks:
            return None
        if 0 <= self.active < len(self.tasks):
            return self.tasks[self.active]
        return None

    def unfinished(self) -> Tuple[TaskItem, ...]:
        return tuple(
            task
            for task in self.tasks
            if task.status in (TASK_STATUS_PENDING, TASK_STATUS_RUNNING)
        )


# --- переходы -------------------------------------------------------------------------------


def allowed_transitions(stage: Stage) -> Tuple[Stage, ...]:
    return TRANSITIONS.get(stage, ())


def parse_stage(name: str) -> Optional[Stage]:
    """Этап по имени, как его пишет пользователь: англ. имя этапа или русское название."""
    value = (name or "").strip().casefold()
    for stage in STAGE_ORDER:
        if value == stage.value or value == STAGE_TITLES[stage].casefold():
            return stage
    return None


def transition(
    state: TaskState, target: Stage, reason: str = "", enforce_gate: bool = True
) -> TransitionResult:
    """Единственный путь смены этапа: таблица плюс шлюз предусловий.

    Отказ — данные, а не исключение: он объясняется пользователю и попадает в журнал задачи вместе
    с принятыми переходами. Предусловия живут здесь, а не у вызывающего: «выполнение только по
    утверждённому плану» и «завершение только из проверки» — свойства модели задачи.
    """
    task = state.current
    if task is None:
        return TransitionResult(accepted=False, state=state, reason=REASON_NO_TASK, stage=None)

    def refused(reason: str, stage: Optional[Stage] = None) -> TransitionResult:
        """Отказ тоже остаётся в журнале: попытка перехода — часть истории задачи."""
        logged = _put_active(
            state,
            replace(
                task,
                transitions=_log_transition(task, task.stage, target, reason, False),
            ),
        )
        return TransitionResult(
            accepted=False, state=logged, reason=reason, stage=stage or task.stage
        )

    if task.stage is Stage.DONE:
        return refused(REASON_TERMINAL)
    if target is task.stage:
        return refused(REASON_SAME_STAGE.format(stage=target.value), stage=target)
    if target not in allowed_transitions(task.stage):
        return refused(
            REASON_NOT_ALLOWED.format(source=task.stage.value, target=target.value)
        )
    if enforce_gate:
        if target is Stage.EXECUTION and not _plan_approved(task):
            return refused(REASON_PLAN_NOT_APPROVED)
        if target is Stage.DONE and task.stage is not Stage.VALIDATION:
            return refused(REASON_NEEDS_VALIDATION)
    moved = _replace_active(
        state, stage=target, transitions=_log_transition(task, task.stage, target, reason, True)
    )
    return TransitionResult(accepted=True, state=moved, reason=reason, stage=target)


@dataclass(frozen=True)
class TransitionResult:
    """Итог перехода этапа: принят ли, новое состояние и причина."""

    accepted: bool
    state: TaskState
    reason: str = ""
    stage: Optional[Stage] = None


def _log_transition(
    task: TaskItem, source: Stage, target: Stage, reason: str, accepted: bool
) -> Tuple[Transition, ...]:
    entry = Transition(
        source=STAGE_TITLES[source], target=STAGE_TITLES[target], reason=reason, accepted=accepted
    )
    return (task.transitions + (entry,))[-config.MAX_TRANSITION_LOG :]


def _plan_approved(task: TaskItem) -> bool:
    """План утверждён, когда он непуст, правок не ждём и перестройки не требуется."""
    return bool(task.plan) and not task.awaiting_edits and not task.replan


def request_stage(state: TaskState, name: str) -> StageRequestResult:
    """Просьба пользователя сменить этап: идёт через тот же шлюз, что и конвейер."""
    stage = parse_stage(name)
    if stage is None:
        return StageRequestResult(
            accepted=False,
            stage=Stage.PLANNING,
            reason="этап не распознан: назовите planning, execution, validation или done",
        )
    result = transition(state, stage, reason="просьба пользователя")
    task = result.state.current
    allowed = allowed_transitions(task.stage) if task else ()
    return StageRequestResult(
        accepted=result.accepted,
        stage=task.stage if task else stage,
        reason=result.reason,
        allowed=allowed,
        state=result.state if result.accepted else None,
    )


# --- очередь и план -------------------------------------------------------------------------


def clip_goal(goal: str, limit: int = config.TASK_GOAL_MAX_CHARS) -> str:
    text = " ".join((goal or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def add_task(state: TaskState, goal: str) -> TaskState:
    """Добавляет задачу в очередь: цель обрезается, номер выдаётся по порядку."""
    text = clip_goal(goal)
    if not text:
        return state
    task = TaskItem(number=state.next_number, goal=text)
    return replace(state, tasks=state.tasks + (task,), next_number=state.next_number + 1)


def drop_all(state: TaskState) -> TaskState:
    """Останавливает прогон: очередь опустошается, активная задача сбрасывается."""
    return replace(state, tasks=(), active=0, paused=False)


def set_paused(state: TaskState, paused: bool) -> TaskState:
    return replace(state, paused=paused)


def begin_task(state: TaskState) -> TaskState:
    """Берёт первую незавершённую задачу: помечает её выполняющейся и ставит на планирование."""
    for index, task in enumerate(state.tasks):
        if task.status in (TASK_STATUS_PENDING, TASK_STATUS_RUNNING):
            updated = replace(task, status=TASK_STATUS_RUNNING, stage=Stage.PLANNING)
            return replace(state, active=index, tasks=state.tasks[:index] + (updated,) + state.tasks[index + 1 :])
    return state


def plan_built(state: TaskState, plan: Sequence[str]) -> TaskState:
    """План первой версии: подзадачи обрезаются потолком задачи, раунд считается."""
    task = state.current
    if task is None:
        return state
    items = tuple(item.strip() for item in plan if item and item.strip())
    limit = task.plan_limit if task.replan else min(task.plan_limit, config.MAX_PLAN_ITEMS)
    built = items[:limit] if not task.replan else items
    updated = replace(
        task,
        plan=built,
        plan_rounds=task.plan_rounds + 1,
        awaiting_edits=True,
        replan=False,
    )
    return _put_active(state, updated)


def edits_response(state: TaskState, text: str) -> TaskState:
    """Ответ пользователя на вопрос о правках: непустой ответ перестраивает план, пустой утверждает."""
    task = state.current
    if task is None or not task.awaiting_edits:
        return state
    if (text or "").strip():
        if task.plan_rounds >= config.MAX_PLAN_ROUNDS:
            # Последний раунд: план берётся как есть, иначе правки зациклятся.
            return _put_active(state, replace(task, awaiting_edits=False, replan=False, plan_limit=task.plan_limit))
        # План растёт вместе с работой, которую назвал пользователь.
        limit = min(config.MAX_PLAN_ITEMS * 2, task.plan_limit + config.MAX_PLAN_ITEMS)
        return _put_active(
            state, replace(task, awaiting_edits=False, replan=True, plan_limit=limit)
        )
    approved = transition(
        _put_active(state, replace(task, awaiting_edits=False, replan=False)),
        Stage.EXECUTION,
        reason="план утверждён",
    )
    return approved.state if approved.accepted else state


def look_ahead(state: TaskState, items: Sequence[str]) -> TaskState:
    """Дополняет план работой, названной проверкой: она выполняется, а не теряется."""
    task = state.current
    if task is None or not items:
        return state
    extra = tuple(item.strip() for item in items if item and item.strip())
    merged = task.plan + tuple(item for item in extra if item not in task.plan)
    return _put_active(
        state, replace(task, plan=merged, plan_limit=max(task.plan_limit, len(merged)))
    )


def next_item_index(task: TaskItem) -> Optional[int]:
    """Номер подзадачи, которую выполнять следующей: первая, к которой ещё не было попытки.

    Отклонённый патч тоже считается попыткой: иначе конвейер возвращался бы к той же подзадаче
    бесконечно. Неудачная попытка остаётся в снимке и попадает в замечания проверки.
    """
    for index, _ in enumerate(task.plan, start=1):
        if task.patch_for(index) is None:
            return index
    return None


def incomplete_indexes(task: TaskItem) -> Tuple[int, ...]:
    """Незавершённые подзадачи: без патча, с неприменённым патчем или пустые."""
    if task.stage is Stage.EXECUTION and task.replan:
        return tuple(range(1, len(task.plan) + 1))
    indexes: List[int] = []
    for index, _ in enumerate(task.plan, start=1):
        patch = task.patch_for(index)
        if patch is None or not patch.applied:
            indexes.append(index)
    return tuple(indexes)


def record_patch(state: TaskState, patch: TaskPatch) -> TaskState:
    """Записывает патч подзадачи: применённый или отклонённый с причиной."""
    task = state.current
    if task is None:
        return state
    kept = tuple(item for item in task.patches if item.index != patch.index)
    return _put_active(state, replace(task, patches=(kept + (patch,))))


def fail_task(state: TaskState, reason: str) -> TaskState:
    """Помечает активную задачу неудачной: прогон переходит к следующей."""
    task = state.current
    if task is None:
        return state
    return _put_active(
        state, replace(task, status=TASK_STATUS_FAILED, fail_reason=reason, stage=Stage.DONE)
    )


def validation_verdict(
    state: TaskState, issues: Sequence[TaskIssue], new_items: Sequence[str] = ()
) -> TaskState:
    """Итог проверки: без замечаний — завершение, с замечаниями — исправление незавершённых.

    Готовое не переделывается: замечание об уже выполненной подзадаче остаётся в списке, но её
    выполнение не перезапускается — пользовательское правило «законченное не переделывать».
    """
    task = state.current
    if task is None:
        return state
    attempts = task.attempts + 1
    if not issues:
        finished = replace(task, issues=(), attempts=attempts, stage=Stage.VALIDATION)
        done = transition(
            _put_active(state, finished), Stage.DONE, reason="проверка без замечаний"
        )
        if done.accepted:
            task_done = done.state.current
            return _put_active(
                done.state, replace(task_done, status=TASK_STATUS_DONE)
            )
        return _put_active(state, finished)

    kept = tuple(issues)
    if attempts >= config.MAX_VALIDATION_ATTEMPTS:
        # Попытки исчерпаны: задача завершается с перечнем неустранённых замечаний.
        finished = replace(task, issues=kept, attempts=attempts)
        done = transition(_put_active(state, finished), Stage.DONE, reason="попытки исчерпаны")
        final = done.state if done.accepted else _put_active(state, finished)
        task_done = final.current
        return _put_active(final, replace(task_done, status=TASK_STATUS_DONE))

    # Исправление получает свежую попытку по непринятым подзадачам: записи об отклонённых патчах
    # снимаются, применённые остаются — готовое не переделывается.
    repaired = _put_active(
        state,
        replace(
            task,
            issues=kept,
            attempts=attempts,
            awaiting_edits=False,
            replan=False,
            patches=tuple(item for item in task.patches if item.applied),
        ),
    )
    if new_items:
        repaired = look_ahead(repaired, new_items)
    back = transition(repaired, Stage.EXECUTION, reason="замечания проверки")
    return back.state if back.accepted else repaired


def _advance(state: TaskState, target: Stage, reason: str) -> TaskState:
    result = transition(state, target, reason=reason)
    return result.state if result.accepted else state


def _replace_active(state: TaskState, **changes: Any) -> TaskState:
    task = state.current
    if task is None:
        return state
    return _put_active(state, replace(task, **changes))


def _put_active(state: TaskState, task: TaskItem) -> TaskState:
    index = state.active
    tasks = state.tasks[:index] + (task,) + state.tasks[index + 1 :]
    return replace(state, tasks=tasks)


# --- сообщение для запроса ------------------------------------------------------------------


def task_message(state: TaskState) -> Optional[str]:
    """Сообщение состояния задачи для запроса: этап, шаг и допустимые переходы.

    Названы только допустимые переходы: перечислять запрещённые — значит подсказывать, как обойти
    модель процесса.
    """
    task = state.current
    if task is None or task.status == TASK_STATUS_DONE:
        return None
    allowed = ", ".join(STAGE_TITLES[stage] for stage in allowed_transitions(task.stage))
    plan = "\n".join(
        f"{index}. {item}" for index, item in enumerate(task.plan, start=1)
    )
    parts = [
        "Состояние задачи текущей сессии (выполняется конвейером, а не тобой).",
        f"Цель: {task.goal}",
        f"Этап: {task.title}; текущий шаг: {task.current_step}",
        f"Допустимые переходы этапа: {allowed or 'нет'}",
    ]
    if plan:
        parts.append("План:\n" + plan)
    if task.issues:
        parts.append("Открытые замечания:\n" + "\n".join(issue.line() for issue in task.issues))
    return "\n".join(parts)


def task_instruction(domain: Domain) -> str:
    """Инструкция домена о том, что этапами управляет приложение, а не модель."""
    return domain.prompt("task")


# --- отчёт ----------------------------------------------------------------------------------


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def result_slug(goal: str, limit: int = config.TASK_RESULT_SLUG_MAX_CHARS) -> str:
    """Латинский слог из цели: имя файла не должно зависеть от локали и раскладки."""
    letters: List[str] = []
    for character in (goal or "").casefold():
        if character in _TRANSLIT:
            letters.append(_TRANSLIT[character])
        elif character.isalnum() and character.isascii():
            letters.append(character)
        elif letters and letters[-1] != "-":
            letters.append("-")
    slug = re.sub(r"-+", "-", "".join(letters)).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rstrip("-")
    return slug or "task"


def result_file_name(task: TaskItem, number: int) -> str:
    return f"{number}-{result_slug(task.goal)}.md"


def result_markdown(task: TaskItem) -> str:
    """Отчёт без русских числительных: склонения — дело терминального слоя, а не файла."""
    lines = [f"# Задача {task.number}: {task.goal}", ""]
    summary = f"Итог: {task.title}"
    if task.status == TASK_STATUS_FAILED:
        summary = f"Итог: не удалось — {task.fail_reason or 'причина не названа'}"
    lines += [summary, f"Подзадач: {len(task.plan)}; применённых патчей: {len(task.applied_patches())}", ""]
    for index, item in enumerate(task.plan, start=1):
        patch = task.patch_for(index)
        lines.append(f"## {index}. {item}")
        if patch and patch.applied:
            lines.append(f"Патч применён: {patch.summary}")
        elif patch and patch.recorded:
            reason = patch.reason or "не применён"
            lines.append(f"Патч не применён: {reason}")
        else:
            lines.append("Работа не выполнена.")
        lines.append("")
    if task.issues:
        lines.append("## Неустранённые замечания")
        lines.extend(f"- {issue.line()}" for issue in task.issues)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_result(directory: Path, number: int, task: TaskItem) -> Optional[Path]:
    """Пишет отчёт о задаче; ошибка записи не прерывает прогон."""
    path = Path(directory) / result_file_name(task, number)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result_markdown(task), encoding="utf-8")
    except OSError:
        return None
    return path


# --- хранилище ------------------------------------------------------------------------------


class TaskStore:
    """`task.json`: очередь, план, патчи и журнал переходов. Чтение терпимое, запись — сразу."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or config.TASK_FILE)
        self.last_error = ""

    def load(self) -> TaskState:
        """Читает состояние: отсутствующий или испорченный файл — пустая очередь, не ошибка."""
        self.last_error = ""
        if not self.path.is_file():
            return TaskState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.last_error = f"состояние задачи не прочитано: {error}"
            return TaskState()
        if not isinstance(raw, dict):
            self.last_error = "состояние задачи не прочитано: ожидался объект"
            return TaskState()
        return _state_from_dict(raw)

    def save(self, state: TaskState) -> None:
        """Пишет состояние; ошибка записи не должна ронять прогон."""
        self.last_error = ""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(_state_to_dict(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            self.last_error = f"состояние задачи не сохранено: {error}"


def _state_to_dict(state: TaskState) -> Dict[str, object]:
    return {
        "schema": 1,
        "active": state.active,
        "paused": state.paused,
        "next_number": state.next_number,
        "tasks": [_task_to_dict(task) for task in state.tasks],
    }


def _task_to_dict(task: TaskItem) -> Dict[str, object]:
    return {
        "number": task.number,
        "goal": task.goal,
        "stage": task.stage.value,
        "status": task.status,
        "plan": list(task.plan),
        "plan_limit": task.plan_limit,
        "plan_rounds": task.plan_rounds,
        "awaiting_edits": task.awaiting_edits,
        "replan": task.replan,
        "patches": [
            {
                "index": patch.index,
                "text": patch.text,
                "summary": patch.summary,
                "applied": patch.applied,
                "reason": patch.reason,
            }
            for patch in task.patches
        ],
        "issues": [
            {"index": issue.index, "text": issue.text} for issue in task.issues
        ],
        "attempts": task.attempts,
        "transitions": [
            {
                "source": item.source,
                "target": item.target,
                "reason": item.reason,
                "accepted": item.accepted,
            }
            for item in task.transitions
        ],
        "result_path": task.result_path,
        "fail_reason": task.fail_reason,
    }


def _state_from_dict(raw: Dict[str, object]) -> TaskState:
    tasks = raw.get("tasks")
    items: List[TaskItem] = []
    if isinstance(tasks, list):
        for entry in tasks:
            if isinstance(entry, dict):
                items.append(_task_from_dict(entry))
    return TaskState(
        tasks=tuple(items),
        active=_as_int(raw.get("active")),
        paused=bool(raw.get("paused", False)),
        next_number=max(_as_int(raw.get("next_number"), 1), len(items) + 1),
    )


def _task_from_dict(raw: Dict[str, object]) -> TaskItem:
    plan = tuple(str(item) for item in raw.get("plan", []) if str(item).strip())
    patches = []
    for entry in raw.get("patches", []) or []:
        if isinstance(entry, dict):
            patches.append(
                TaskPatch(
                    index=_as_int(entry.get("index")),
                    text=str(entry.get("text", "")),
                    summary=str(entry.get("summary", "")),
                    applied=bool(entry.get("applied", False)),
                    reason=str(entry.get("reason", "")),
                )
            )
    issues = []
    for entry in raw.get("issues", []) or []:
        if isinstance(entry, dict):
            index = entry.get("index")
            issues.append(
                TaskIssue(
                    index=_as_int(index) if index is not None else None,
                    text=str(entry.get("text", "")),
                )
            )
    transitions = []
    for entry in raw.get("transitions", []) or []:
        if isinstance(entry, dict):
            transitions.append(
                Transition(
                    source=str(entry.get("source", "")),
                    target=str(entry.get("target", "")),
                    reason=str(entry.get("reason", "")),
                    accepted=bool(entry.get("accepted", False)),
                )
            )
    return TaskItem(
        number=_as_int(raw.get("number"), 1),
        goal=str(raw.get("goal", "")),
        stage=_stage(raw.get("stage")),
        status=str(raw.get("status", TASK_STATUS_PENDING)),
        plan=plan,
        plan_limit=_as_int(raw.get("plan_limit"), config.MAX_PLAN_ITEMS),
        plan_rounds=_as_int(raw.get("plan_rounds")),
        awaiting_edits=bool(raw.get("awaiting_edits", False)),
        replan=bool(raw.get("replan", False)),
        patches=tuple(patches),
        issues=tuple(issues),
        attempts=_as_int(raw.get("attempts")),
        transitions=tuple(transitions)[-config.MAX_TRANSITION_LOG :],
        result_path=str(raw.get("result_path", "")),
        fail_reason=str(raw.get("fail_reason", "")),
    )


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return default


def _stage(value: object) -> Stage:
    try:
        return Stage(str(value))
    except ValueError:
        return Stage.PLANNING
