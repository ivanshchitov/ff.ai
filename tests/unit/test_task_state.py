"""Состояние задачи: переходы, план, патчи, попытки, отчёт и хранилище."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config, task_state
from core.task_state import (
    STAGE_TITLES,
    Stage,
    TaskIssue,
    TaskItem,
    TaskPatch,
    TaskState,
    TaskStore,
)


def _state_with_task(goal: str = "добавить экран") -> TaskState:
    return task_state.add_task(TaskState(), goal)


def _running(goal: str = "добавить экран") -> TaskState:
    return task_state.begin_task(_state_with_task(goal))


def _planned(plan=("правка сборки", "новый экран")) -> TaskState:
    state = task_state.plan_built(_running(), plan)
    return task_state.edits_response(state, "")


# --- переходы -------------------------------------------------------------------------------


def test_table_forbids_skipping_stages():
    assert task_state.allowed_transitions(Stage.PLANNING) == (Stage.EXECUTION,)
    assert task_state.allowed_transitions(Stage.DONE) == ()


def test_parse_stage_accepts_russian_names():
    assert task_state.parse_stage("execution") is Stage.EXECUTION
    assert task_state.parse_stage("Выполнение") is Stage.EXECUTION
    assert task_state.parse_stage("нет такого") is None


def test_transition_without_task_is_refused():
    result = task_state.transition(TaskState(), Stage.EXECUTION)
    assert result.accepted is False
    assert result.reason == task_state.REASON_NO_TASK


def test_transition_to_same_stage_is_refused():
    state = _running()
    result = task_state.transition(state, Stage.PLANNING)
    assert result.accepted is False
    assert "уже на этапе" in result.reason


def test_transition_outside_the_table_is_refused():
    state = _running()
    result = task_state.transition(state, Stage.VALIDATION)
    assert result.accepted is False
    assert "не разрешён таблицей" in result.reason


def test_execution_requires_an_approved_plan():
    state = task_state.plan_built(_running(), ("пункт",))
    result = task_state.transition(state, Stage.EXECUTION)
    assert result.accepted is False
    assert result.reason == task_state.REASON_PLAN_NOT_APPROVED


def test_done_is_unreachable_from_execution():
    """Прямой переход в завершение отсекается таблицей, а не шлюзом."""
    state = _planned()
    direct = task_state.transition(state, Stage.DONE, "хочу финал")
    assert direct.accepted is False
    assert "не разрешён таблицей" in direct.reason


def test_journal_keeps_attempts_both_accepted_and_refused():
    """В журнал попадают и принятые переходы, и отказы: попытка — часть истории задачи."""
    state = _planned()
    refused = task_state.transition(state, Stage.DONE, "хочу финал")
    assert refused.accepted is False
    journal = refused.state.current.transitions
    assert journal and journal[-1].accepted is False
    assert "✗" in journal[-1].line()

    accepted = task_state.transition(state, Stage.VALIDATION, "проверяем")
    assert accepted.accepted is True
    assert accepted.state.current.transitions[-1].accepted is True


def test_transition_journal_is_capped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "MAX_TRANSITION_LOG", 2)
    state = _planned()
    for _ in range(4):
        state = task_state.transition(state, Stage.DONE, "мимо").state
    assert len(state.current.transitions) == 2


def test_done_is_terminal():
    state = _planned()
    state = task_state.transition(state, Stage.VALIDATION, "проверяем").state
    state = task_state.validation_verdict(state, ())
    assert state.current.stage is Stage.DONE
    result = task_state.transition(state, Stage.PLANNING, "ещё раз")
    assert result.accepted is False
    assert result.reason == task_state.REASON_TERMINAL


def test_request_stage_goes_through_the_gate():
    state = _running()
    refused = task_state.request_stage(state, "done")
    assert refused.accepted is False
    assert Stage.EXECUTION in refused.allowed
    assert refused.state is None

    planned = task_state.edits_response(task_state.plan_built(state, ("пункт",)), "")
    accepted = task_state.request_stage(planned, "validation")
    assert accepted.accepted is True
    assert accepted.state.current.stage is Stage.VALIDATION


def test_request_stage_unknown_name():
    result = task_state.request_stage(_running(), "куда-нибудь")
    assert result.accepted is False
    assert "не распознан" in result.reason


# --- очередь и план -------------------------------------------------------------------------


def test_queue_numbers_and_clips():
    state = task_state.add_task(TaskState(), "x" * (config.TASK_GOAL_MAX_CHARS + 50))
    assert state.current.number == 1
    assert len(state.tasks[0].goal) <= config.TASK_GOAL_MAX_CHARS
    state = task_state.add_task(state, "вторая")
    assert [task.number for task in state.tasks] == [1, 2]
    assert state.next_number == 3


def test_empty_goal_is_not_a_task():
    assert task_state.add_task(TaskState(), "   ").tasks == ()


def test_begin_task_takes_the_first_unfinished():
    state = task_state.add_task(task_state.add_task(TaskState(), "первая"), "вторая")
    started = task_state.begin_task(state)
    assert started.current.number == 1
    assert started.current.status == task_state.TASK_STATUS_RUNNING
    assert started.current.stage is Stage.PLANNING


def test_begin_task_skips_finished_tasks():
    state = _state_with_task()
    finished = task_state._put_active(
        state, _task_with_stage(state.current, stage=Stage.DONE, status=task_state.TASK_STATUS_DONE)
    )
    state = task_state.add_task(finished, "вторая")
    started = task_state.begin_task(state)
    assert started.current.number == 2


def _task_with_stage(task: TaskItem, **changes) -> TaskItem:
    from dataclasses import replace

    return replace(task, **changes)


def test_drop_all_clears_the_queue():
    state = task_state.begin_task(_state_with_task())
    cleared = task_state.drop_all(state)
    assert cleared.tasks == ()
    assert cleared.current is None


def test_plan_limit_and_rounds():
    state = task_state.plan_built(_running(), [f"пункт {number}" for number in range(1, 30)])
    assert len(state.current.plan) == config.MAX_PLAN_ITEMS
    assert state.current.plan_rounds == 1
    assert state.current.awaiting_edits is True


def test_edits_rebuild_the_plan_and_grow_the_limit():
    state = task_state.plan_built(_running(), ("правка сборки",))
    asked = task_state.edits_response(state, "добавь экран и тесты")
    assert asked.current.replan is True
    assert asked.current.plan_limit > config.MAX_PLAN_ITEMS - 1
    rebuilt = task_state.plan_built(asked, ("правка сборки", "экран", "тесты"))
    assert rebuilt.current.plan == ("правка сборки", "экран", "тесты")
    assert rebuilt.current.plan_rounds == 2


def test_empty_answer_approves_the_plan():
    state = task_state.plan_built(_running(), ("пункт",))
    approved = task_state.edits_response(state, "   ")
    assert approved.current.stage is Stage.EXECUTION
    assert approved.current.awaiting_edits is False


def test_last_plan_round_is_taken_as_is(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "MAX_PLAN_ROUNDS", 1)
    state = task_state.plan_built(_running(), ("пункт",))
    answered = task_state.edits_response(state, "ещё правка")
    assert answered.current.replan is False
    assert answered.current.awaiting_edits is False


def test_look_ahead_appends_new_items():
    state = _planned()
    extended = task_state.look_ahead(state, ("тесты",))
    assert extended.current.plan[-1] == "тесты"
    again = task_state.look_ahead(extended, ("тесты",))
    assert again.current.plan.count("тесты") == 1


# --- патчи и попытки ------------------------------------------------------------------------


def _applied(index: int, text: str = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n") -> TaskPatch:
    return TaskPatch(index=index, text=text, summary=f"x (+1/-1)", applied=True)


def test_next_item_index_follows_the_plan():
    state = _planned()
    assert task_state.next_item_index(state.current) == 1
    state = task_state.record_patch(state, _applied(1))
    assert task_state.next_item_index(state.current) == 2
    state = task_state.record_patch(state, _applied(2))
    assert task_state.next_item_index(state.current) is None


def test_rejected_patch_keeps_the_item_incomplete():
    state = _planned()
    rejected = TaskPatch(index=1, text="патч", summary="x", applied=False, reason="путь вне репозитория")
    state = task_state.record_patch(state, rejected)
    assert task_state.incomplete_indexes(state.current) == (1, 2)
    assert state.current.patch_for(1).reason == "путь вне репозитория"


def test_record_patch_replaces_by_index():
    state = _planned()
    state = task_state.record_patch(state, TaskPatch(index=1, text="a", summary="a"))
    state = task_state.record_patch(state, TaskPatch(index=1, text="b", summary="b", applied=True))
    assert len(state.current.patches) == 1
    assert state.current.patch_for(1).text == "b"


def test_fail_task_marks_it_failed():
    state = task_state.fail_task(_running(), "запрос не прошёл")
    assert state.current.status == task_state.TASK_STATUS_FAILED
    assert state.current.fail_reason == "запрос не прошёл"


def test_validation_without_issues_finishes_the_task():
    state = _planned()
    state = task_state.transition(state, Stage.VALIDATION, "проверяем").state
    finished = task_state.validation_verdict(state, ())
    assert finished.current.stage is Stage.DONE
    assert finished.current.status == task_state.TASK_STATUS_DONE


def test_validation_issues_send_the_task_back_to_execution():
    state = _planned()
    state = task_state.transition(state, Stage.VALIDATION, "проверяем").state
    back = task_state.validation_verdict(state, (TaskIssue(index=1, text="патч не применён"),))
    assert back.current.stage is Stage.EXECUTION
    assert back.current.attempts == 1
    assert back.current.issues[0].text == "патч не применён"


def test_exhausted_attempts_finish_with_open_issues(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "MAX_VALIDATION_ATTEMPTS", 1)
    state = _planned()
    state = task_state.transition(state, Stage.VALIDATION, "проверяем").state
    finished = task_state.validation_verdict(state, (TaskIssue(index=1, text="сборка падает"),))
    assert finished.current.stage is Stage.DONE
    assert finished.current.issues[0].text == "сборка падает"


def test_validation_new_items_are_appended():
    state = _planned()
    state = task_state.transition(state, Stage.VALIDATION, "проверяем").state
    back = task_state.validation_verdict(
        state, (TaskIssue(index=None, text="нет тестов"),), ("добавить тесты",)
    )
    assert back.current.plan[-1] == "добавить тесты"
    assert back.current.stage is Stage.EXECUTION


# --- сообщение и отчёт ----------------------------------------------------------------------


def test_task_message_names_stage_and_allowed_transitions():
    message = task_state.task_message(_planned())
    assert "Этап: Выполнение" in message
    assert "Проверка" in message and "Планирование" in message
    assert "Цель: добавить экран" in message


def test_task_message_is_empty_without_a_task():
    assert task_state.task_message(TaskState()) is None


def test_result_slug_transliterates():
    assert task_state.result_slug("Добавить экран!") == "dobavit-ekran"
    assert task_state.result_slug("!!!") == "task"


def test_result_markdown_lists_plan_and_issues():
    state = _planned()
    state = task_state.record_patch(state, _applied(1))
    state = task_state._put_active(
        state, _task_with_stage(state.current, issues=(TaskIssue(index=2, text="нет патча"),))
    )
    text = task_state.result_markdown(state.current)
    assert "# Задача 1: добавить экран" in text
    assert "## 1. правка сборки" in text
    assert "Патч применён" in text
    assert "Работа не выполнена." in text
    assert "нет патча" in text


def test_write_result_creates_the_file(tmp_path: Path):
    state = _planned()
    path = task_state.write_result(tmp_path, 3, state.current)
    assert path is not None
    assert path.name.startswith("3-")
    assert path.read_text(encoding="utf-8").startswith("# Задача 1")


# --- хранилище ------------------------------------------------------------------------------


def test_store_round_trip(tmp_path: Path):
    store = TaskStore(tmp_path / "task.json")
    state = _planned()
    state = task_state.record_patch(state, _applied(1))
    state = task_state.set_paused(state, True)
    store.save(state)

    restored = store.load()
    assert restored.paused is True
    assert restored.current.plan == state.current.plan
    assert restored.current.patch_for(1).applied is True
    assert restored.current.stage is Stage.EXECUTION


def test_store_reads_missing_and_corrupt_files_as_empty(tmp_path: Path):
    store = TaskStore(tmp_path / "task.json")
    assert store.load().tasks == ()
    (tmp_path / "task.json").write_text("{это не json", encoding="utf-8")
    assert store.load().tasks == ()
    assert store.last_error


def test_store_tolerates_partial_records(tmp_path: Path):
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps({"tasks": [{"goal": "цель"}], "active": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    state = TaskStore(path).load()
    assert state.current.goal == "цель"
    assert state.current.stage is Stage.PLANNING
    assert state.current.plan == ()


def test_store_write_error_is_visible(tmp_path: Path):
    store = TaskStore(tmp_path / "нет-каталога" / "task.json")
    store.path.parent.write_text("файл вместо каталога", encoding="utf-8")
    store.save(_planned())
    assert "не сохранено" in store.last_error
