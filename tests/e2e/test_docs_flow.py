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
from .stub_api import StubAPI, answer

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


def _model_double(stub: StubAPI, *, answer_text: str, score: float = 0.9, rerank: str = None):
    """Заглушка модели для корпуса документации: на оценку кандидатов — JSON, на вопрос — ответ.

    Число и идентификаторы кандидатов знает только приложение, поэтому ответ на запрос оценки
    вычисляется по телу запроса, а не задаётся заранее.
    """
    if rerank is None:

        def rerank(payload):
            data = json.loads(payload["messages"][-1]["content"].split("\n", 1)[1])
            return answer(
                json.dumps(
                    {
                        "results": [
                            {"id": item["id"], "score": score, "reason": "тестовая оценка"}
                            for item in data["candidates"]
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    def builder(payload):
        system = payload["messages"][0]["content"]
        if "оцениваешь" in system:
            return rerank(payload)
        return answer(answer_text)

    return stub.dynamic(builder)


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

    _model_double(stub, answer_text=CITED)
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert stub.call_count == 2, "оценка кандидатов и ответ"
        assert client_answer_calls(stub) == 1, "подтверждённый ответ не требует повтора"
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

    _model_double(stub, answer_text=UNCITED)
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert client_answer_calls(stub) == 2, "нарушение стоит одного повтора"
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

    _model_double(stub, answer_text=UNCITED)
    session = _launch(tmp_path, stub)
    try:
        session.send_line("/rag-docs mode off")
        session.wait_for("Поиск по документации выключен")
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert stub.call_count == 1, "без корпуса нет ни оценки, ни повтора"
        assert UNCITED[:40] in session.screen_text()
        payload = json.dumps(stub.last_payload(), ensure_ascii=False)
        assert "Ниже — фрагменты" not in payload
        assert "дословные фрагменты переданных источников" not in payload
    finally:
        session.close()


def test_unavailable_docs_server_is_reported_and_the_answer_goes_through(stub, tmp_path: Path):
    """Сервер документации, который не поднимается: приложение говорит об этом и отвечает дальше."""
    from .stub_api import answer

    _model_double(stub, answer_text=UNCITED)
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
    _model_double(stub, answer_text=cited)
    session = _launch(tmp_path, stub)
    try:
        session.ask("в какой версии появился фоновый режим геопозиции?", timeout=DOCS_TIMEOUT)
        session.send_line("/rag-docs")
        lines = session.wait_for("Разделы:")
        assert "release_notes" in lines, "вопрос о версии ищется и в примечаниях к выпуску"
        assert "версия документации: 5.2.1" in lines
    finally:
        session.close()


def test_docs_report_shows_fragments_without_model_requests(stub, tmp_path: Path):
    from .stub_api import answer

    _model_double(stub, answer_text=CITED)
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        before = stub.call_count
        session.send_line("/rag-docs trace")
        lines = session.wait_for("Доставлено фрагментов:")
        assert "Доставлено фрагментов:" in lines
        assert POSITIONING in lines, "в отчёте видно, какие документы доставлены"
        assert stub.call_count == before, "отчёт не обращается к модели"
    finally:
        session.close()


def test_fragment_block_stays_out_of_history(stub, tmp_path: Path):
    from .stub_api import answer

    _model_double(stub, answer_text=CITED)
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        raw = (tmp_path / "state" / "history.json").read_text(encoding="utf-8")
        assert "Ниже — фрагменты" not in raw
        assert "📚" not in raw
        assert QUESTION in raw
    finally:
        session.close()


def client_answer_calls(stub: StubAPI) -> int:
    """Сколько запросов было к модели за ответом, без служебного запроса оценки."""
    return sum(
        1
        for request in stub.requests
        if "оцениваешь" not in request["payload"]["messages"][0]["content"]
    )


def test_low_scores_give_no_fragments_and_a_journal_line(stub, tmp_path: Path):
    """Все кандидаты ниже порога: документация не доставляется, ответ всё равно формируется."""
    _model_double(stub, answer_text="В документации портала подходящего ответа нет.", score=0.1)
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        screen = session.screen_text()
        assert "ни один не прошёл порог" in screen
        payload = json.dumps(stub.last_payload(), ensure_ascii=False)
        assert "Ниже — фрагменты" not in payload, "фрагменты не доставляются"
        assert "не признан относящимся" in payload, "модель получает инструкцию"
    finally:
        session.close()


def test_bad_rating_response_is_a_visible_state(stub, tmp_path: Path):
    """Негодный ответ оценщика: видимая причина и ответ без документации."""
    _model_double(stub, answer_text="Отвечаю без документации.", rerank=lambda payload: answer("не JSON"))
    session = _launch(tmp_path, stub)
    try:
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert "Оценка фрагментов не удалась" in session.screen_text()
    finally:
        session.close()


def test_baseline_mode_skips_the_rating_request(stub, tmp_path: Path):
    _model_double(stub, answer_text=CITED)
    session = _launch(tmp_path, stub)
    try:
        session.send_line("/rag-docs retrieval baseline")
        session.wait_for("Режим отбора: baseline")
        session.ask(QUESTION, timeout=DOCS_TIMEOUT)
        assert stub.call_count == 1, "в baseline оценка не запрашивается"
        assert session.contains("Режим отбора: baseline"), "отчёт показывает режим"
    finally:
        session.close()


def test_threshold_command_changes_the_report(stub, tmp_path: Path):
    _model_double(stub, answer_text=CITED)
    session = _launch(tmp_path, stub)
    try:
        session.send_line("/rag-docs threshold 0.9")
        session.wait_for("Порог отбора: 0.90")
        session.send_line("/rag-docs")
        assert "порог: 0.90" in session.wait_for("Режим отбора:")

        session.send_line("/rag-docs threshold 5")
        assert "от 0 до 1" in session.wait_for("от 0 до 1")
    finally:
        session.close()
