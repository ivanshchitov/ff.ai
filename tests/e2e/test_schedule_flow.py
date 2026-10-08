"""Планировщик и задания здоровья репозитория в живом приложении.

Реестр MCP здесь подменён не заглушкой, а настоящим собственным сервером репозитория: задания и
планировщик объявляет только он, и ручной вызов идёт ровно тем путём, каким пойдёт задание в
расписании. Середина поэтому не подделывается — проверяется проводка приложения, сервера и файла
расписания.

Пути состояния серверу передаются аргументами (`--schedule-file`, `--index-file`): запуск с
переопределением реестра переменных окружения ему не передаёт — библиотека оставляет процессу
безопасный минимум, — и без аргументов прогон писал бы расписание и индекс в каталог пользователя.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from .harness import REPO_ROOT, AppSession


REPO_SERVER = REPO_ROOT / "mcp_server" / "repo_server.py"
TOOLS_FILE = REPO_ROOT / "domains" / "aurora-qt5" / "tools.json"

JOB_TOOLS = ("target_build", "index_refresh", "public_api_scan")
SCHEDULE_TOOLS = ("schedule_add", "schedule_list", "schedule_run_due", "schedule_summary")


def schedule_file(tmp_path: Path) -> Path:
    return tmp_path / "state" / "schedule.json"


def index_file(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "index.sqlite3"


def _launch(tmp_path: Path, repo: Path, **kwargs) -> AppSession:
    """Приложение против собственного сервера репозитория вместо заглушки реестра."""
    args = (
        f"{REPO_SERVER} --root {repo} --tools {TOOLS_FILE}"
        f" --schedule-file {schedule_file(tmp_path)} --index-file {index_file(tmp_path)}"
    )
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=repo,
        mcp_args=args,
        **kwargs,
    )


@contextmanager
def running(tmp_path: Path, stub, **kwargs):
    repo = AppSession.create_target_repo(base=tmp_path)
    session = _launch(tmp_path, repo, api_url=stub.url, **kwargs)
    try:
        session.wait_for_prompt()
        yield session
    finally:
        session.close()


def read_schedule(tmp_path: Path) -> dict:
    path = schedule_file(tmp_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# Файл помечен slow: приложение поднимается в pty, и прогон занимает минуты.
pytestmark = [pytest.mark.e2e, pytest.mark.slow]

def test_job_is_scheduled_and_run_by_the_app(tmp_path: Path, stub):
    with running(tmp_path, stub) as session:
        session.send_line("/tool call schedule_add tool=public_api_scan every_minutes=1")
        assert "поставлено" in session.wait_for("поставлено")

        session.send_line("/tool call schedule_list")
        assert "public_api_scan" in session.wait_for("Задания планировщика")

        session.write_repo_file("src/main.cpp", "QML_ELEMENT\n")
        session.send_line("/tool call schedule_run_due")
        assert "Выполнено заданий: 1" in session.wait_for("Выполнено заданий")

    data = read_schedule(tmp_path)
    assert [job["tool"] for job in data["jobs"]] == ["public_api_scan"]
    assert data["runs"][0]["ok"] is True
    assert "Проверка публичного API" in data["runs"][0]["summary"]
    assert data["runs"][0]["fresh"] == 1
    assert stub.call_count == 0


def test_schedule_state_survives_a_restart(tmp_path: Path, stub):
    """Состояние планировщика — файл: следующая сессия видит и задание, и накопленное."""
    repo = AppSession.create_target_repo(base=tmp_path)
    first = _launch(tmp_path, repo, api_url=stub.url)
    try:
        first.wait_for_prompt()
        first.write_repo_file("src/main.cpp", "QML_ELEMENT\n")
        first.send_line("/tool call schedule_add tool=public_api_scan every_minutes=1")
        first.wait_for("поставлено")
        first.send_line("/tool call schedule_run_due")
        first.wait_for("Выполнено заданий")
        first.send_line("/exit")
        first.wait_exit()
    finally:
        first.close()

    second = _launch(tmp_path, repo, api_url=stub.url)
    try:
        second.wait_for_prompt()
        second.send_line("/tool call schedule_summary")
        text = second.wait_for("накоплено записей")
    finally:
        second.close()

    assert "public_api_scan" in text
    assert "накоплено записей 1" in text
    assert "прогонов 1" in text


def test_repository_state_is_untouched(tmp_path: Path, stub):
    """Ни приложение, ни сервер не пишут в целевой репозиторий и в состояние пользователя."""
    real = REPO_ROOT / "schedule.json"
    before = real.read_text(encoding="utf-8") if real.exists() else None
    repo = AppSession.create_target_repo(base=tmp_path)
    session = _launch(tmp_path, repo, api_url=stub.url)
    try:
        session.wait_for_prompt()
        session.send_line("/tool call schedule_add tool=public_api_scan every_minutes=1")
        session.wait_for("поставлено")
        session.send_line("/exit")
        session.wait_exit()
    finally:
        session.close()

    after = real.read_text(encoding="utf-8") if real.exists() else None
    assert after == before
    assert not (repo / "schedule.json").exists()
    assert schedule_file(tmp_path).is_file()
    # Обменов с моделью не было — истории тоже: строки инструментов в неё не попадают.
    assert not (tmp_path / "state" / "history.json").exists()
