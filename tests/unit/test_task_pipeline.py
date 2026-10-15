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
        content = REVIEW_OK if phase == "task_review" else (self.answers.pop(0) if self.answers else "")
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
        confirm=lambda *_: True,
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
    # Explicitly invalid context: a repeated short insertion is a valid minimal diff.
    invalid = PATCH.replace(" вторая строка", " отсутствующий контекст")
    ask = FakeAsk(PLAN, PATCH, invalid, REVIEW_ISSUE)
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
    # Seed a previously applied unsafe artifact: execution now rejects it before prepare.
    pipeline = _pipeline(repo, tmp_path, FakeAsk(REVIEW_OK), build=lambda command: (True, "ок"))
    pipeline.add("добавить экран")
    pipeline._save(task_state.begin_task(pipeline.state))
    pipeline._save(task_state.plan_built(pipeline.state, ("новый экран",)))
    pipeline.answer_edits("")
    seeded = _apply_without_checks(repo, 1, task_pipeline.parse_patch_response(bad))
    pipeline._save(task_state.record_patch(pipeline.state, seeded))
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


def test_standalone_pipeline_requires_explicit_patch_confirmation(repo, tmp_path):
    pipeline = task_pipeline.TaskPipeline(domain=load_domain('aurora-qt5'), root=repo,
                                         store=TaskStore(tmp_path / 'standalone.json'),
                                         ask=FakeAsk(PLAN, PATCH), tasks_dir=tmp_path / 'tasks')
    before = (repo / 'notes.txt').read_bytes()
    _approved(pipeline)
    pipeline.step()
    assert not pipeline.state.current.patch_for(1).applied
    assert pipeline.state.current.patch_for(1).change_id
    assert (repo / 'notes.txt').read_bytes() == before


def test_validation_ignores_unrelated_user_work_and_sees_check_results(repo, tmp_path):
    from core.task_context import TaskContextProvider
    (repo / 'unrelated.cpp').write_text('int value;\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'unrelated.cpp'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@example.com', 'commit', '-qm', 'extra'], check=True)
    (repo / 'unrelated.cpp').write_text('#include <QtWidgets>\n')
    ask = FakeAsk(PLAN, PATCH, '--- /dev/null\n+++ b/docs/new.md\n@@ -0,0 +1 @@\n+documentation\n', REVIEW_OK)
    pipeline = _pipeline(repo, tmp_path, ask, build=lambda _: (True, 'build output marker'))
    _approved(pipeline)
    pipeline.step()
    pipeline.step()
    pipeline.step()
    pipeline.step()
    validation = next(messages for phase, messages, _ in ask.calls if phase == 'task_validate')
    text = '\n'.join(m['content'] for m in validation)
    assert 'build output marker' in text
    assert 'unrelated.cpp' not in text
    assert 'третья строка' in text
    assert pipeline.state.current.status == task_state.TASK_STATUS_DONE


def test_subagents_review_before_confirmation_with_isolated_context(repo, tmp_path):
    calls, approvals = [], []
    patch = task_pipeline.parse_patch_response(PATCH)
    def ask(messages, words, phase):
        calls.append((phase, messages))
        if phase == 'task_plan':
            content = '{"items":["update notes"]}'
        elif phase == 'task_execute':
            content = 'PRIVATE_BUILDER_REASONING\n' + patch
        else:
            assert phase == 'task_review'
            assert (repo / 'notes.txt').read_text() == 'первая строка\nвторая строка\n'
            assert not approvals
            context = '\n'.join(m['content'] for m in messages)
            assert 'PRIVATE_BUILDER_REASONING' not in context
            assert patch.strip() in context
            assert 'первая строка' in context
            content = '{"issues":[]}'
        return AnswerMeta(content, 'test', .01, 10, 10, 20, 0)
    pipeline = _pipeline(repo, tmp_path, ask)
    pipeline._confirm = lambda *_: approvals.append('confirmed') or True
    _approved(pipeline)
    report = pipeline.step()
    assert [phase for phase, _ in calls] == ['task_plan', 'task_execute', 'task_review']
    assert approvals == ['confirmed']
    assert pipeline.state.current.patches[0].applied
    assert any('ревьюер' in note.lower() for note in report.notes)


@pytest.mark.parametrize('response,finish', [('not json','stop'), ('{}','stop'),
    ('{"issues":[]}', 'length'), ('{"issues":"none"}','stop')])
def test_unavailable_or_invalid_subagent_review_blocks_write(repo, tmp_path, response, finish):
    approvals=[]
    class Ask(FakeAsk):
        def __call__(self, messages, words, phase):
            if phase == 'task_review':
                return AnswerMeta(response, 'test', .01, 1, 1, 2, 0, finish)
            return super().__call__(messages, words, phase)
    pipeline = _pipeline(repo, tmp_path, Ask(PLAN, PATCH))
    pipeline._confirm=lambda *_: approvals.append(True) or True
    _approved(pipeline)
    pipeline.step()
    assert not pipeline.state.current.patches[0].applied
    assert not approvals
    assert 'третья строка' not in (repo / 'notes.txt').read_text()


def test_subagents_correct_once_and_keep_final_rejection(repo, tmp_path):
    phases, corrections = [], []
    class Ask(FakeAsk):
        def __call__(self, messages, words, phase):
            phases.append(phase)
            if phase == 'task_review':
                return AnswerMeta('{"issues":["MISSING_CHECK"]}', 'test', .01, 1, 1, 2, 0)
            if phase == 'task_execute' and phases.count(phase) == 2:
                corrections.extend(m['content'] for m in messages)
            return super().__call__(messages, words, phase)
    pipeline=_pipeline(repo, tmp_path, Ask(PLAN, PATCH, PATCH))
    _approved(pipeline)
    pipeline.step()
    assert phases.count('task_execute') == 2 and phases.count('task_review') == 2
    assert 'MISSING_CHECK' in '\n'.join(corrections)
    assert not pipeline.state.current.patches[0].applied
    assert 'MISSING_CHECK' in pipeline.state.current.patches[0].reason


def test_deterministic_rejection_precedes_subagent_review(repo, tmp_path):
    ask=FakeAsk(PLAN, '--- a/absent.txt\n+++ b/absent.txt\n@@ -1 +1 @@\n-no\n+yes\n')
    pipeline=_pipeline(repo,tmp_path,ask)
    _approved(pipeline)
    pipeline.step()
    assert not pipeline.state.current.patches[0].applied
    assert 'task_review' not in [phase for phase, *_ in ask.calls]


def test_review_api_failure_is_recorded_without_confirmation(repo, tmp_path):
    ask=FakeAsk(PLAN, PATCH, fail_phases=('task_review',))
    pipeline=_pipeline(repo,tmp_path,ask)
    _approved(pipeline)
    pipeline.step()
    assert not pipeline.state.current.patches[0].applied
    assert 'ревьюер' in pipeline.state.current.patches[0].reason


def test_subagent_successful_correction_and_audit_survive_restart(repo, tmp_path):
    reviews=[]
    class Ask(FakeAsk):
        def __call__(self, messages, words, phase):
            if phase == 'task_review':
                reviews.append(messages)
                content = '{"issues":["explain change"]}' if len(reviews)==1 else REVIEW_OK
                return AnswerMeta(content,'test',.01,3,2,5,0)
            return super().__call__(messages,words,phase)
    pipeline=_pipeline(repo,tmp_path,Ask(PLAN,PATCH,PATCH))
    _approved(pipeline)
    pipeline.step()
    stored=pipeline.store.load().current.patches[0]
    assert stored.applied
    assert any('explain change' in line for line in stored.review_notes)
    assert any('Ревьюер' in line and '5' in line for line in stored.review_notes)


def test_single_item_plan_limit_has_consistent_request(repo, tmp_path):
    pipeline=_pipeline(repo,tmp_path,FakeAsk())
    task=task_state.TaskItem(1,'atomic code and docs change',plan_limit=1)
    assert 'от 1 до 1' in pipeline._plan_request(task,pipeline.state)


def test_correction_api_failure_keeps_rejection_audit_and_pauses(repo,tmp_path):
    phases=[]
    class Ask(FakeAsk):
        def __call__(self,messages,words,phase):
            phases.append(phase)
            if phase=='task_review':
                return AnswerMeta('{"issues":["FIX_REQUIRED"]}','test',.01,3,2,5,0)
            if phase=='task_execute' and phases.count(phase)==2:
                raise APIError('unavailable')
            return super().__call__(messages,words,phase)
    pipeline=_pipeline(repo,tmp_path,Ask(PLAN,PATCH))
    _approved(pipeline)
    pipeline.step()
    task=pipeline.store.load().current
    assert not task.patches[0].applied
    assert any('FIX_REQUIRED' in note for note in task.patches[0].review_notes)
    assert pipeline.paused
    assert pipeline.step() is None


def test_empty_truncated_developer_does_not_repeat_without_user(repo,tmp_path):
    ask=FakeAsk(PLAN,'',truncated=True)
    pipeline=_pipeline(repo,tmp_path,ask)
    _approved(pipeline)
    pipeline.step()
    assert pipeline.paused
    assert pipeline.step() is None
    assert pipeline.store.load().current.patches[0].reason==task_pipeline.EXECUTE_FAILED_TRUNCATED


def test_review_json_substring_cannot_authorize_write(repo,tmp_path):
    class Ask(FakeAsk):
        def __call__(self,messages,words,phase):
            if phase=='task_review':
                return AnswerMeta('{"issues":[]} review unavailable','test',.01,1,1,2,0)
            return super().__call__(messages,words,phase)
    pipeline=_pipeline(repo,tmp_path,Ask(PLAN,PATCH))
    _approved(pipeline)
    pipeline.step()
    assert pipeline.paused and not pipeline.state.current.patches[0].applied


def test_interrupted_correction_is_durable_without_transient_pause(repo,tmp_path):
    pipeline=None
    class Ask(FakeAsk):
        executions=0
        def __call__(self,messages,words,phase):
            if phase=='task_review':
                return AnswerMeta('{"issues":["FIX"]}','test',.01,1,1,2,0)
            if phase=='task_execute':
                self.executions+=1
                if self.executions==2:
                    snapshot=pipeline.store.load()
                    assert snapshot.current.patches[0].review_pending
                    assert not snapshot.paused
                    raise KeyboardInterrupt()
            return super().__call__(messages,words,phase)
    pipeline=_pipeline(repo,tmp_path,Ask(PLAN,PATCH))
    _approved(pipeline)
    with pytest.raises(KeyboardInterrupt):
        pipeline.step()
    restarted=_pipeline(repo,tmp_path,FakeAsk())
    assert restarted.paused
    restarted.set_paused(False)
    assert restarted.step().notes
    assert restarted.paused
    assert 'третья строка' not in (repo/'notes.txt').read_text()


def test_model_diff_recounts_hunks_without_changing_content(repo):
    from core import patches
    raw = ("diff --git a/notes.txt b/notes.txt\n--- a/notes.txt\n+++ b/notes.txt\n"
           "@@ -1,99 +1,99 @@\n первая строка\n вторая строка\n+третья строка\n"
           "diff --git a/new.txt b/new.txt\nnew file mode 100644\n--- /dev/null\n+++ b/new.txt\n"
           "@@ -0,0 +1,99 @@\n+--- a/decoy.txt\n++++ b/decoy.txt\n")
    result = task_pipeline.parse_patch_response(raw)
    parsed = patches.parse(result)
    assert parsed.paths == ('notes.txt', 'new.txt')
    assert '@@ -1,2 +1,3 @@' in result
    assert '@@ -0,0 +1,2 @@' in result
    subprocess.run(['git','apply','--check','-'],cwd=repo,input=result,text=True,check=True)
    subprocess.run(['git','apply','-'],cwd=repo,input=result,text=True,check=True)
    assert (repo/'notes.txt').read_text() == 'первая строка\nвторая строка\nтретья строка\n'
    assert (repo/'new.txt').read_text() == '--- a/decoy.txt\n+++ b/decoy.txt\n'
    assert not (repo/'decoy.txt').exists()


def test_model_diff_recount_preserves_invalid_content_and_unframed_diff():
    from core import patches
    malformed = ('diff --git a/n b/n\n--- a/n\n+++ b/n\n'
                 '@@ -1,3 +1,3 @@\n context\nmissing prefix\n')
    with pytest.raises(patches.PatchError):
        patches.parse(task_pipeline.parse_patch_response(malformed))
    unframed = '--- a/n\n+++ b/n\n@@ -1,99 +1,99 @@\n context\n'
    assert task_pipeline.parse_patch_response(unframed) == unframed


def test_model_diff_recount_preserves_payload_cr_and_unicode_separator():
    text = ('diff --git a/n b/n\n--- a/n\n+++ b/n\n'
            '@@ -1,99 +1,99 @@\n context\n+payload\r\n+one\u2028two\n')
    result = task_pipeline.parse_patch_response(text)
    assert '+payload\r\n+one\u2028two\n' in result
    assert result == text.replace('@@ -1,99 +1,99 @@', '@@ -1,1 +1,3 @@')


def test_model_diff_fence_preserves_trailing_cr_payload():
    body = ('diff --git a/n b/n\n--- a/n\n+++ b/n\n'
            '@@ -1,99 +1,99 @@\n context\n+payload\r\n')
    result = task_pipeline.parse_patch_response('```diff\n' + body + '```')
    assert result == body.replace('@@ -1,99 +1,99 @@', '@@ -1,1 +1,2 @@')


def test_model_diff_fence_keeps_markdown_fences_inside_hunk():
    body = ('diff --git a/readme.md b/readme.md\n--- a/readme.md\n+++ b/readme.md\n'
            '@@ -1,99 +1,99 @@\n-old\n+new\n ```\n context\n+after fence\n'
            'diff --git a/n b/n\n--- a/n\n+++ b/n\n'
            '@@ -1,99 +1,99 @@\n-old\n+new\n')
    expected = body.replace('@@ -1,99 +1,99 @@', '@@ -1,3 +1,4 @@', 1)
    expected = expected.replace('@@ -1,99 +1,99 @@', '@@ -1,1 +1,1 @@')
    assert task_pipeline.parse_patch_response('```diff\n' + body + '```') == expected


def test_small_goal_can_have_one_step_with_three_step_ceiling(repo,tmp_path):
    pipeline=_pipeline(repo,tmp_path,FakeAsk())
    task=task_state.TaskItem(1,'создать короткий документ',plan_limit=3)
    assert 'от 1 до 3' in pipeline._plan_request(task,pipeline.state)
