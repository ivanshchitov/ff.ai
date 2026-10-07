"""Фикстуры сквозных прогонов: локальный stub API и приложение в псевдотерминале."""

from pathlib import Path

import pytest

from .harness import AppSession, tui_display
from .stub_api import StubAPI


@pytest.fixture
def stub() -> StubAPI:
    """Локальный OpenAI-совместимый сервер: отвечает по программе теста и пишет все запросы."""
    server = StubAPI()
    server.url = server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    """Целевой репозиторий: каталог с `.git`, иначе точка входа откажется работать."""
    return AppSession.create_target_repo(base=tmp_path, name="aurora-project")


@pytest.fixture
def app(stub: StubAPI, tmp_path: Path, target_repo: Path, request):
    """Живое приложение в псевдотерминале с изолированным состоянием и stub-сервером."""
    _, mirror = tui_display(request)
    session = AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=target_repo,
        api_url=stub.url,
        mirror=mirror,
    )
    try:
        yield session
    finally:
        session.close()
