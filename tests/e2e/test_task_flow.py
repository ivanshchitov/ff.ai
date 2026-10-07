"""Задача в живом приложении: прогон, патчи, проверки, пауза и продолжение.

Приложение запускается в pty с изолированным состоянием, модель отвечает локальным stub-сервером
по фазе запроса. Проверяется то, что уходит в запросы, что попадает в файлы проекта и что видно
на экране.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from .harness import AppSession
from .stub_api import answer

PLAN = json.dumps({"items": ["правка заметок", "новый экран"]}, ensure_ascii=False)
REVIEW_OK = json.dumps({"issues": []}, ensure_ascii=False)
REVIEW_ISSUE = json.dumps(
    {"issues": [{"item": 2, "text": "экран не добавлен"}], "items": ["тесты экрана"]},
    ensure_ascii=False,
)
NOTES_PATCH = """```diff
--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,3 @@
 первая строка
 вторая строка
+третья строка
```"""
SCREEN_PATCH = """```diff
--- /dev/null
+++ b/src/screen.qml
@@ -0,0 +1,2 @@
+import Sailfish.Silica 1.0
+ApplicationWindow { id: window }
+```"""
BAD_PATCH = """```diff
--- /dev/null
+++ b/src/window.cpp
@@ -0,0 +1,1 @@
+#include <QtWidgets/QApplication>
+```"""
LIVE = 60


# Задержка ответов: с мгновенными ответами прогон заканчивается быстрее, чем тест успевает
# нажать паузу или подтвердить патч, и проверка превращается в гонку.
DELAY = 0.4


def _model_double(stub, *, second_patch: str = SCREEN_PATCH, review: str = REVIEW_OK):
    """Двойник модели: ответ зависит от фазы конвейера, а не от порядка запросов."""

    def reply(text: str):
        payload = answer(text)
        payload.delay = DELAY
        return payload

    def builder(payload):
        text = "\n".join(message["content"] for message in payload["messages"])
        if "Число подзадач" in text:
            return reply(PLAN)
        if "Выполни подзадачу" in text:
            # Номер подзадачи берётся из строки запроса: план в запросе перечислен целиком, и по
            # названию пункта фазу не отличить.
            match = re.search(r"Выполни подзадачу (\d+):", text)
            index = int(match.group(1)) if match else 1
            return reply(second_patch if index == 2 else NOTES_PATCH)
        if "Ты проверяешь результат" in text:
            return reply(review)
        return reply("Ответ ассистента.")

    return stub.dynamic(builder)


def _project(root: Path) -> Path:
    """Готовит целевой репозиторий один раз: повторный вызов (перезапуск) ничего не пересоздаёт."""
    if not (root / ".git" / "HEAD").exists():
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "notes.txt").write_text("первая строка\nвторая строка\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=root,
            check=True,
        )
    return root


def _launch(tmp_path: Path, stub, **kwargs) -> AppSession:
    repo = AppSession.create_target_repo(base=tmp_path, name="aurora-project")
    _project(repo)
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        task_file=tmp_path / "state" / "task.json",
        tasks_dir=tmp_path / "state" / "tasks",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=repo,
        api_url=stub.url,
        **kwargs,
    )


def _answer_plan(app: AppSession) -> None:
    """Отвечает на вопрос о правках плана пустой строкой: план утверждается."""
    app.wait_for("Правки к плану", timeout=LIVE)
    app.send_line("")
    app.wait_for("Пауза — клавиша p или Ctrl+C", timeout=LIVE)


def _approve_patch(app: AppSession) -> None:
    """Ждёт вопроса о применении патча и отвечает «да»: запись требует подтверждения."""
    app.wait_for("Применить патч?", timeout=LIVE)
    app.send_key(b"y")


def _phases(stub) -> list:
    """Фазы конвейера в порядке запросов — по подписи в тексте запроса."""
    phases = []
    for request in stub.requests:
        text = "\n".join(message["content"] for message in request["payload"]["messages"])
        if "Число подзадач" in text:
            phases.append("plan")
        elif "Выполни подзадачу" in text:
            phases.append("execute")
        elif "Ты проверяешь результат" in text:
            phases.append("validate")
        else:
            phases.append("answer")
    return phases


@pytest.fixture
def app_with_task(stub, tmp_path: Path):
    _model_double(stub)
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        yield app, stub, tmp_path
    finally:
        app.close()


def test_full_run_applies_patches_and_writes_the_report(app_with_task):
    app, stub, tmp_path = app_with_task
    app.send_line("/task add поправить заметки и добавить экран")
    app.wait_for("Задача добавлена", timeout=LIVE)
    app.send_line("/task run")
    _answer_plan(app)
    _approve_patch(app)
    app.wait_for("Патч применён", timeout=LIVE)
    _approve_patch(app)
    app.wait_for("Задача 1: Завершено", timeout=LIVE)

    repo = app.repo
    assert (repo / "notes.txt").read_text(encoding="utf-8").endswith("третья строка\n")
    assert (repo / "src" / "screen.qml").is_file()
    assert _phases(stub) == ["plan", "execute", "execute", "validate"]

    tasks_dir = tmp_path / "state" / "tasks"
    reports = list(tasks_dir.glob("*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "Патч применён" in text
    assert "поправить заметки" in text


def test_run_state_and_reports_stay_out_of_the_repository(app_with_task):
    app, stub, tmp_path = app_with_task
    app.send_line("/task add поправить заметки")
    app.wait_for("Задача добавлена", timeout=LIVE)
    app.send_line("/task run")
    _answer_plan(app)
    _approve_patch(app)
    app.wait_for("Патч применён", timeout=LIVE)

    names = {path.name for path in app.repo.rglob("*")}
    assert "task.json" not in names
    assert not any(name.endswith(".md") for name in names), "отчётов в репозитории нет"


def test_edits_rebuild_the_plan(app_with_task):
    app, stub, tmp_path = app_with_task
    app.send_line("/task add поправить заметки")
    app.wait_for("Задача добавлена", timeout=LIVE)
    app.send_line("/task run")
    app.wait_for("Правки к плану", timeout=LIVE)
    app.send_line("добавь экран")
    # Вопрос о плане задаётся снова, и пустой ответ утверждает уже перестроенный план; запрос
    # патча появляется только после этого — по нему и видно, что план строился дважды.
    _answer_plan(app)
    _approve_patch(app)

    assert _phases(stub)[:2] == ["plan", "plan"], "правки перестраивают план вторым запросом"


def test_forbidden_construction_is_rejected_before_applying(stub, tmp_path: Path):
    _model_double(stub, second_patch=BAD_PATCH)
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.send_line("/task add добавить окно")
        app.wait_for("Задача добавлена", timeout=LIVE)
        app.send_line("/task run")
        _answer_plan(app)
        _approve_patch(app)
        screen = app.wait_for("Патч не применён", timeout=LIVE)

        assert "запрещённая конструкция" in screen
        assert not (app.repo / "src" / "window.cpp").exists()
    finally:
        app.close()


def test_pause_and_resume(stub, tmp_path: Path):
    _model_double(stub)
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.send_line("/task add поправить заметки")
        app.wait_for("Задача добавлена", timeout=LIVE)
        app.send_line("/task run")
        app.wait_for("Правки к плану", timeout=LIVE)
        app.send_line("")
        _approve_patch(app)
        # Пауза ставится между операциями: текущая завершается, состояние записано.
        app.send_key(b"p")
        app.wait_for("Прогон остановлен на паузе", timeout=LIVE)

        app.send_line("/task")
        assert "прогон на паузе" in app.wait_for("прогон на паузе", timeout=LIVE)

        app.send_line("/task run")
        app.wait_for("Пауза — клавиша p или Ctrl+C", timeout=LIVE)
        _approve_patch(app)
        app.wait_for("Патч применён", timeout=LIVE)
    finally:
        app.close()


def test_restart_continues_from_the_saved_step(stub, tmp_path: Path):
    _model_double(stub)
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.send_line("/task add поправить заметки")
        app.wait_for("Задача добавлена", timeout=LIVE)
        app.send_line("/task run")
        _answer_plan(app)
        _approve_patch(app)
        app.send_key(b"p")
        app.wait_for("Прогон остановлен на паузе", timeout=LIVE)
    finally:
        app.close()

    # Задача уже на этапе выполнения: план сохранён, второй раз его строить не нужно.
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.send_line("/task")
        screen = app.wait_for("Этап:", timeout=LIVE)
        assert "Выполнение" in screen
        assert "Задача добавлена" in screen or "поправить заметки" in screen

        app.send_line("/task run")
        app.wait_for("Пауза — клавиша p или Ctrl+C", timeout=LIVE)
        _approve_patch(app)
        assert "Патч применён" in app.wait_for("Патч применён", timeout=LIVE)
    finally:
        app.close()


def test_clear_keeps_the_queue_and_state_keeps_out_of_the_dialogue(app_with_task):
    app, stub, tmp_path = app_with_task
    app.send_line("/task add поправить заметки")
    app.wait_for("Задача добавлена", timeout=LIVE)
    app.send_line("/clear")
    app.wait_for("Диалог очищен", timeout=LIVE)

    app.send_line("/task")
    assert "поправить заметки" in app.wait_for("поправить заметки", timeout=LIVE)
    history = json.loads((tmp_path / "state" / "history.json").read_text(encoding="utf-8"))
    assert "Состояние задачи" not in json.dumps(history, ensure_ascii=False)


def test_stage_command_goes_through_the_gate(app_with_task):
    app, stub, tmp_path = app_with_task
    app.send_line("/task add поправить заметки")
    app.wait_for("Задача добавлена", timeout=LIVE)

    app.send_line("/task stage validation")
    screen = app.wait_for("Переход не выполнен", timeout=LIVE)
    assert "не разрешён таблицей" in screen

    app.send_line("/task stage execution")
    assert "план не утверждён" in app.wait_for("план не утверждён", timeout=LIVE)
