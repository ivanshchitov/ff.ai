"""Документация портала в реальном приложении: фрагменты, обязательные ссылки и цитаты.

Прогон идёт против заглушки сервера документации (`tests/fake_docs_server.py`) со своим корпусом:
приложение поднимает её как единственный сервер реестра, но под именем `aurora-docs` — иначе
домен не найдёт свой корпус. Сети здесь нет.
"""

import json
from pathlib import Path

import pytest
from tests.fake_docs_server import MARKERS

from .harness import AppSession

pytestmark = pytest.mark.e2e

FAKE_DOCS = Path(__file__).resolve().parent.parent / "fake_docs_server.py"
# Поиск по документации — это несколько обращений к серверу (версии, разделы, полный текст),
# поэтому ответ по корпусу приходит секундами, а не мгновенно.
DOCS_TIMEOUT = 90.0
POSITIONING = "doc/software_development/guides/cpp_api/positioning"
RELEASE_NOTES = "doc/release_notes/5.2.0"
QUESTION = "как получить координаты устройства на Авроре?"
UNCITED = "Координаты берутся из модуля Qt Positioning, его нужно подключить в проекте."
CITED = (
    "Координаты отдаёт модуль Qt Positioning.\n"
    f"Цитата: «{MARKERS[POSITIONING]}»\n"
    f"Источник: {POSITIONING}"
)


def _launch(tmp_path: Path, stub, **kwargs) -> AppSession:
    """Приложение с корпусом документации под именем сервера домена."""
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=AppSession.create_target_repo(base=tmp_path),
        api_url=stub.url,
        mcp_args=str(FAKE_DOCS),
        mcp_name="aurora-docs",
        **kwargs,
    )


def test_cited_answer_is_accepted_with_sources_line(stub, tmp_path: Path):
    from .stub_api import answer

    stub.always(answer(CITED))
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert stub.call_count == 1, "подтверждённый ответ не требует повтора"
        screen = session.screen_text()
        assert "📚 Источники (документация 5.2.1)" in screen
        assert POSITIONING in screen

        payload = json.dumps(stub.last_payload(), ensure_ascii=False)
        assert MARKERS[POSITIONING] in payload, "фрагмент документа уходит в запрос"
        assert "Ниже — фрагменты" in payload
        assert "дословные фрагменты переданных источников" in payload, "инструкция цитат добавлена"
    finally:
        session.close()


def test_uncited_answer_is_retried_then_replaced(stub, tmp_path: Path):
    from .stub_api import answer

    stub.sequence(answer(UNCITED), answer("Тот же ответ без ссылки и без цитаты."))
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert stub.call_count == 2, "нарушение стоит одного повтора"
        screen = session.screen_text()
        assert "📚 Ссылки не подтверждены" in screen
        assert "Не знаю:" in screen
        history = json.loads(
            (tmp_path / "state" / "history.json").read_text(encoding="utf-8")
        )
        assert history["dialogues"][-1]["answer"].startswith("Не знаю:")
    finally:
        session.close()


def test_disabled_mode_removes_search_and_check(stub, tmp_path: Path):
    from .stub_api import answer

    stub.always(answer(UNCITED))
    session = _launch(tmp_path, stub)
    try:
        session.send_line("/docs mode off")
        session.wait_for("Поиск по документации выключен")
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert stub.call_count == 1, "без проверки повтора быть не должно"
        assert UNCITED[:40] in session.screen_text()
        payload = json.dumps(stub.last_payload(), ensure_ascii=False)
        assert "Ниже — фрагменты" not in payload
        assert "дословные фрагменты переданных источников" not in payload
    finally:
        session.close()


def test_unavailable_docs_server_is_reported_and_the_answer_goes_through(stub, tmp_path: Path):
    """Сервер документации, который не поднимается: приложение говорит об этом и отвечает дальше."""
    from .stub_api import answer

    stub.always(answer(UNCITED))
    session = AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        repo=AppSession.create_target_repo(base=tmp_path),
        api_url=stub.url,
        mcp_args="/нет/такого/сервера.py",
        mcp_name="aurora-docs",
    )
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        screen = session.screen_text()
        assert "📚 Документация портала недоступна" in screen
        assert UNCITED[:40] in screen, "без документации ответ всё равно доходит"
    finally:
        session.close()


def test_version_question_searches_release_notes(stub, tmp_path: Path):
    from .stub_api import answer

    cited = (
        "Фоновый режим появился в 5.2.0.\n"
        f"Цитата: «{MARKERS[RELEASE_NOTES]}»\n"
        f"Источник: {RELEASE_NOTES}"
    )
    stub.always(answer(cited))
    session = _launch(tmp_path, stub)
    try:
        session.ask("в какой версии появился фоновый режим геопозиции?", timeout=DOCS_TIMEOUT)
        session.send_line("/docs")
        lines = session.wait_for("Разделы:")
        assert "release_notes" in lines, "вопрос о версии ищется и в примечаниях к выпуску"
        assert "версия документации: 5.2.1" in lines
    finally:
        session.close()


def test_docs_report_shows_fragments_without_model_requests(stub, tmp_path: Path):
    from .stub_api import answer

    stub.always(answer(CITED))
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        before = stub.call_count
        session.send_line("/docs trace")
        lines = session.wait_for("Доставлено фрагментов:")
        assert "Доставлено фрагментов:" in lines
        assert POSITIONING in lines, "в отчёте видно, какие документы доставлены"
        assert stub.call_count == before, "отчёт не обращается к модели"
    finally:
        session.close()


def test_fragment_block_stays_out_of_history(stub, tmp_path: Path):
    from .stub_api import answer

    stub.always(answer(CITED))
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        raw = (tmp_path / "state" / "history.json").read_text(encoding="utf-8")
        assert "Ниже — фрагменты" not in raw
        assert "📚" not in raw
        assert QUESTION in raw
    finally:
        session.close()
