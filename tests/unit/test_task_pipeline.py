"""Конвейер задачи: операции этапов, патчи, проверки, попытки и отчёт."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core import config, task_pipeline, task_state
from core.api_client import APIError, AnswerMeta
from core.domains import load_domain
from core.task_state import Stage, TaskIssue, TaskPatch, TaskStore

PLAN = json.dumps({"items": ["правка сборки", "новый экран"]}, ensure_ascii=False)
PATCH = (
    "```diff\n--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,3 @@\n первая строка\n вторая строка\n+третья строка\n```"
)
REVIEW_OK = json.dumps({"issues": []}, ensure_ascii=False)
REVIEW_ISSUE = json.dumps(
    {"issues": [{"item": 2, "text": "экран не добавлен"}], "items": ["тесты экрана"]},
    ensure_ascii=False,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "notes.txt").write_text("первая строка\nвторая строка\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    return root


class FakeAsk:
    """Источник запросов: отдаёт ответы по программе теста и помнит, что у него спросили."""

    def __init__(self, *answers: str, fail_phases: tuple = (), truncated: bool = False) -> None:
        self.answers = list(answers)
        self.calls: list = []
        self.fail_phases = fail_phases
        self.truncated = truncated

    def __call__(self, messages, max_words, phase):
        self.calls.append((phase, messages, max_words))
        if phase in self.fail_phases:
            raise APIError("модель недоступна")
        content = self.answers.pop(0) if self.answers else ""
        return AnswerMeta(
            content=content,
            model="test",
            elapsed_seconds=0.1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0,
            finish_reason="length" if self.truncated else "stop",
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _repo(tmp_path)


def _pipeline(repo: Path, tmp_path: Path, ask, **kwargs) -> task_pipeline.TaskPipeline:
    return task_pipeline.TaskPipeline(
        domain=load_domain("aurora-qt5"),
        root=repo,
        store=TaskStore(tmp_path / "task.json"),
        ask=ask,
        tasks_dir=tmp_path / "tasks",
        **kwargs,
    )


def _approved(pipeline: task_pipeline.TaskPipeline) -> None:
    """Доводит задачу до выполнения: добавить, построить план, утвердить пустым ответом."""
    pipeline.add("добавить экран")
    pipeline.step()
    pipeline.answer_edits("")


# --- разбор ответов -------------------------------------------------------------------------


def test_parse_plan_from_json_numbered_and_plain():
    assert task_pipeline.parse_plan_response(PLAN) == ("правка сборки", "новый экран")
    assert task_pipeline.parse_plan_response("1. первое\n2. второе") == ("первое", "второе")
    assert task_pipeline.parse_plan_response("- одно\n- другое") == ("одно", "другое")
    assert task_pipeline.parse_plan_response("правка сборки") == ("правка сборки",)


def test_parse_patch_from_fenced_and_plain():
    assert task_pipeline.parse_patch_response(PATCH).startswith("--- a/notes.txt")
    plain = "вот патч:\n--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-a\n+b\n"
    assert task_pipeline.parse_patch_response(plain).startswith("--- a/notes.txt")
    assert task_pipeline.parse_patch_response("просто текст") == ""


def test_parse_validation_response_reads_issues_and_items():
    issues, items = task_pipeline.parse_validation_response(REVIEW_ISSUE)
    assert issues[0].index == 2 and issues[0].text == "экран не добавлен"
    assert items == ("тесты экрана",)
    plain, _ = task_pipeline.parse_validation_response("сборка не проходит")
    assert plain[0].text == "сборка не проходит"


# --- планирование ---------------------------------------------------------------------------


def test_plan_step_builds_and_asks_for_edits(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN)
    pipeline = _pipeline(repo, tmp_path, ask)
    pipeline.add("добавить экран")

    report = pipeline.step()
    assert report.stage is Stage.PLANNING
    assert report.awaiting_edits is True
    assert pipeline.state.current.plan == ("правка сборки", "новый экран")
    assert pipeline.state.current.awaiting_edits is True
    assert ask.calls[0][0] == "task_plan"


def test_awaiting_edits_makes_no_request(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN)
    pipeline = _pipeline(repo, tmp_path, ask)
    pipeline.add("цель")
    pipeline.step()
    before = len(ask.calls)

    report = pipeline.step()
    assert report.awaiting_edits is True
    assert len(ask.calls) == before


def test_empty_plan_fails_the_task(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(""))
    pipeline.add("цель")
    report = pipeline.step()
    assert pipeline.state.current.status == task_state.TASK_STATUS_FAILED
    assert "пустой ответ" in pipeline.state.current.fail_reason
    assert report.notes


def test_empty_answer_approves_and_moves_to_execution(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN))
    _approved(pipeline)
    assert pipeline.state.current.stage is Stage.EXECUTION


# --- выполнение -----------------------------------------------------------------------------


def test_execution_applies_patch_and_moves_to_validation(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN, PATCH, PATCH)
    pipeline = _pipeline(repo, tmp_path, ask)
    _approved(pipeline)

    first = pipeline.step()
    assert first.stage is Stage.EXECUTION
    assert pipeline.state.current.patch_for(1).applied is True
    assert "третья строка" in (repo / "notes.txt").read_text(encoding="utf-8")

    ask.answers = [
        "```diff\n--- /dev/null\n+++ b/src/screen.qml\n@@ -0,0 +1,2 @@\n"
        "+import Sailfish.Silica 1.0\n+ApplicationWindow { id: window }\n```"
    ]
    second = pipeline.step()
    assert pipeline.state.current.patch_for(2).applied is True
    assert second.stage is Stage.EXECUTION

    third = pipeline.step()
    assert third.stage is Stage.VALIDATION
    assert pipeline.state.current.stage is Stage.VALIDATION


def test_prepare_function_is_used_and_can_reject(repo: Path, tmp_path: Path):
    seen: list = []

    def prepare(index: int, text: str) -> TaskPatch:
        seen.append(index)
        return TaskPatch(index=index, text=text, summary="x", applied=False, reason="путь вне репозитория")

    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN, PATCH, PATCH), prepare=prepare)
    _approved(pipeline)
    report = pipeline.step()

    assert seen == [1]
    assert pipeline.state.current.patch_for(1).applied is False
    assert "путь вне репозитория" in report.notes[0]
    assert "третья строка" not in (repo / "notes.txt").read_text(encoding="utf-8")


def test_rejected_patch_does_not_loop_on_the_same_item(repo: Path, tmp_path: Path):
    def prepare(index: int, text: str) -> TaskPatch:
        return TaskPatch(index=index, text=text, summary="x", applied=False, reason="не годится")

    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN, PATCH, PATCH), prepare=prepare)
    _approved(pipeline)
    pipeline.step()
    report = pipeline.step()
    assert report.label.startswith("Подзадача 2"), "конвейер идёт дальше, а не повторяет подзадачу 1"
    assert pipeline.step().stage is Stage.VALIDATION


def test_answer_without_patch_is_recorded(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN, "просто текст"))
    _approved(pipeline)
    report = pipeline.step()
    assert task_pipeline.EXECUTE_FAILED_NO_PATCH in report.notes[0]
    assert pipeline.state.current.patch_for(1).reason == task_pipeline.EXECUTE_FAILED_NO_PATCH


def test_truncated_answer_is_named(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN, "", truncated=True))
    _approved(pipeline)
    report = pipeline.step()
    assert task_pipeline.EXECUTE_FAILED_TRUNCATED in report.notes[0]


def test_request_failure_marks_the_task_failed(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN, fail_phases=("task_execute",)))
    _approved(pipeline)
    with pytest.raises(APIError):
        pipeline.step()


# --- проверка -------------------------------------------------------------------------------


def _to_validation(pipeline: task_pipeline.TaskPipeline) -> None:
    for _ in range(4):
        if pipeline.state.current.stage is Stage.VALIDATION:
            return
        pipeline.step()


def test_validation_without_issues_finishes_and_writes_the_report(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN, PATCH, PATCH, REVIEW_OK)
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda command: (True, "собрано"))
    _approved(pipeline)
    _to_validation(pipeline)

    report = pipeline.step()
    assert report.finished is True
    assert pipeline.state.current.status == task_state.TASK_STATUS_DONE
    assert pipeline.state.current.result_path.startswith(str(tmp_path / "tasks"))
    path = Path(pipeline.state.current.result_path)
    assert path.is_file()
    assert "Патч применён" in path.read_text(encoding="utf-8")


def test_validation_returns_to_execution_on_issues(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN, PATCH, PATCH, REVIEW_ISSUE)
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda command: (True, "собрано"))
    _approved(pipeline)
    _to_validation(pipeline)

    report = pipeline.step()
    assert report.stage is Stage.EXECUTION
    assert pipeline.state.current.attempts == 1
    assert pipeline.state.current.plan[-1] == "тесты экрана"
    assert pipeline.state.current.patch_for(1) is not None, "применённое остаётся"
    assert pipeline.state.current.patch_for(2) is None, "незавершённая получила свежую попытку"


def test_failed_build_is_an_issue(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN, PATCH, PATCH, REVIEW_OK)
    pipeline = _pipeline(
        repo, tmp_path, ask, build=lambda command: (command != "make", "ошибка сборки")
    )
    _approved(pipeline)
    _to_validation(pipeline)

    report = pipeline.step()
    assert report.stage is Stage.EXECUTION
    assert any("сборка" in note for note in report.notes)


def test_unavailable_review_keeps_the_artifact_verdict(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN, PATCH, PATCH, fail_phases=("task_validate",))
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda command: (command != "make", "ошибка"))
    _approved(pipeline)
    _to_validation(pipeline)

    report = pipeline.step()
    assert report.stage is Stage.EXECUTION
    assert any("модельная проверка недоступна" in note for note in report.notes)
    assert any("сборка" in note for note in report.notes)


def test_attempts_are_exhausted_and_task_finishes_with_issues(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "MAX_VALIDATION_ATTEMPTS", 1)
    ask = FakeAsk(PLAN, PATCH, PATCH, REVIEW_ISSUE)
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda command: (True, "ок"))
    _approved(pipeline)
    _to_validation(pipeline)

    report = pipeline.step()
    assert report.finished is True
    assert pipeline.state.current.status == task_state.TASK_STATUS_DONE
    assert pipeline.state.current.issues


def test_forbidden_construction_in_the_artifact_is_caught(repo: Path, tmp_path: Path):
    """Вторая линия защиты: проверка этапа ловит запрещённое даже в применённом патче."""
    bad = (
        "```diff\n--- /dev/null\n+++ b/src/window.cpp\n@@ -0,0 +1,1 @@\n"
        "+#include <QtWidgets/QApplication>\n```"
    )
    ask = FakeAsk(PLAN, bad, bad, REVIEW_OK)
    # Лояльная подготовка: патч применяется мимо проверок выполнения — так проверяется именно этап.
    pipeline = _pipeline(
        repo,
        tmp_path,
        ask,
        prepare=lambda index, text: _apply_without_checks(repo, index, text),
        build=lambda command: (True, "ок"),
    )
    _approved(pipeline)
    _to_validation(pipeline)

    report = pipeline.step()
    assert report.stage is Stage.EXECUTION
    assert any("запрещённая конструкция" in note for note in report.notes)


def _apply_without_checks(root: Path, index: int, patch_text: str) -> TaskPatch:
    from core import patches as patches_module

    text = task_pipeline.parse_patch_response(patch_text)
    reason = patches_module.apply(root, text)
    return TaskPatch(
        index=index, text=text, summary=patches_module.parse(text).summary(), applied=not reason, reason=reason
    )


# --- очередь, пауза и отчёт -----------------------------------------------------------------


def test_step_without_tasks_returns_none(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk())
    assert pipeline.step() is None


def test_second_task_starts_after_the_first(repo: Path, tmp_path: Path):
    ask = FakeAsk(PLAN, PATCH, PATCH, REVIEW_OK, PLAN, PATCH, PATCH, REVIEW_OK)
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda command: (True, "ок"))
    _approved(pipeline)
    pipeline.add("второе изменение")
    _to_validation(pipeline)
    pipeline.step()
    assert pipeline.state.current.status == task_state.TASK_STATUS_DONE

    report = pipeline.step()
    assert report.stage is Stage.PLANNING, "прогон переходит к следующей задаче"


def test_pause_persists(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN))
    pipeline.add("цель")
    pipeline.set_paused(True)
    assert pipeline.paused is True
    assert TaskStore(tmp_path / "task.json").load().paused is True


def test_request_stage_saves_accepted_state(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN))
    _approved(pipeline)
    result = pipeline.request_stage("validation")
    assert result.accepted is True
    assert TaskStore(tmp_path / "task.json").load().current.stage is Stage.VALIDATION


def test_stop_clears_the_queue(repo: Path, tmp_path: Path):
    pipeline = _pipeline(repo, tmp_path, FakeAsk(PLAN))
    pipeline.add("цель")
    assert pipeline.stop().tasks == ()
    assert TaskStore(tmp_path / "task.json").load().tasks == ()


def test_applied_item_keeps_its_mark_after_a_review_round(repo: Path, tmp_path: Path):
    """Применённая подзадача не теряет отметку и не выполняется заново после замечаний."""
    bad_review = json.dumps(
        {"issues": [{"text": "в артефакте нет описания"}]}, ensure_ascii=False
    )
    ask = FakeAsk(PLAN, PATCH, PATCH, bad_review, REVIEW_OK)
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda command: (True, "ок"))
    _approved(pipeline)
    pipeline.step()
    applied_index = next(
        index for index, _ in enumerate(pipeline.state.current.plan, start=1)
        if pipeline.state.current.patch_for(index) and pipeline.state.current.patch_for(index).applied
    )
    _to_validation(pipeline)
    pipeline.step()  # замечание возвращает задачу в выполнение

    assert pipeline.state.current.patch_for(applied_index) is not None
    assert pipeline.state.current.patch_for(applied_index).applied is True
    assert task_state.next_item_index(pipeline.state.current) != applied_index
