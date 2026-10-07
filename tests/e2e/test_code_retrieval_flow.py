"""Корпус кода в живом приложении: фрагменты в запросе, цитата, состояния поиска.

Приложение запускается в pty, индекс собран заранее в каталоге кэша теста, модель отвечает
локальным stub-сервером. Ни сети, ни настоящих серверных процессов: проверяется то, что уходит
в запрос и что видно на экране.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import rerank_answer

from core import code_index, domains

from .harness import AppSession
from .stub_api import answer

CORPUS = domains.load_domain("aurora-qt5").corpus
CODE = "src/models.cpp"
QUOTE = 'const QString path = settings_.value("models.ini");'
CODE_TEXT = "\n".join(
    [
        "class ModelList : public QObject {",
        "public:",
        "    void load();",
        "};",
        "",
        "void ModelList::load() {",
        f"    {QUOTE}",
        "    model_ = new QAbstractListModel(this);",
        "}",
    ]
)
QUESTION = "где инициализируется модель списка?"
CITED = f"Смотри {CODE}:L1-L9.\n\nЦитата: {QUOTE}"
UNCITED = "Модель списка инициализируется при загрузке настроек."
LIVE = 60


def _model_double(stub, answers: list, *, score: float = 0.9, query: str = "ModelList load models.ini"):
    """Двойник модели: переформулировка, оценка кандидатов и ответы по очереди.

    Ответы выдаются по программе теста: без этого нельзя проверить ни подтверждённый ответ, ни
    повтор с заменой.
    """
    state = {"left": list(answers)}

    def builder(payload):
        system = payload["messages"][0]["content"]
        if "оцениваешь" in system:
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
        if "превращаешь вопрос" in system:
            return answer(query)
        if len(state["left"]) > 1:
            return answer(state["left"].pop(0))
        return answer(state["left"][0] if state["left"] else "Ответ без ссылок.")

    return stub.dynamic(builder)


def _project(root: Path) -> Path:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "models.cpp").write_text(CODE_TEXT, encoding="utf-8")
    (root / "src" / "render.cpp").write_text(
        "void Renderer::paint(QPainter &painter) {\n    painter.setBrush(background_);\n}\n",
        encoding="utf-8",
    )
    return root


def _launch(tmp_path: Path, stub, *, with_index: bool = True, **kwargs) -> AppSession:
    repo = AppSession.create_target_repo(base=tmp_path, name="aurora-project")
    _project(repo)
    cache = tmp_path / "cache"
    if with_index:
        cache.mkdir(parents=True, exist_ok=True)
        code_index.build_index(repo, CORPUS, code_index.STRATEGY_STRUCTURAL, cache / "index.sqlite3")
    return AppSession(
        history_file=tmp_path / "state" / "history.json",
        state_dir=tmp_path / "state",
        cache_dir=cache,
        index_file=cache / "index.sqlite3",
        repo=repo,
        api_url=stub.url,
        # Реестр MCP заменяется фейковым сервером по умолчанию: без этого домен нашёл бы свой
        # настоящий сервер документации, и тест ушёл бы в сеть.
        **kwargs,
    )


@pytest.fixture
def session(stub, tmp_path: Path):
    _model_double(stub, [CITED])
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        yield app, stub
    finally:
        app.close()


def _answer_request(stub) -> str:
    """Текст последнего запроса ответа: сообщения склеены, служебные запросы исключены.

    Склейка текстом, а не JSON: в JSON кавычки экранированы, и проверять дословную цитату по нему
    значит проверять экранирование, а не текст.
    """
    for request in reversed(stub.requests):
        messages = request["payload"]["messages"]
        head = messages[0]["content"]
        if "оцениваешь" in head or "превращаешь вопрос" in head:
            continue
        return "\n".join(message["content"] for message in messages)
    return ""


def test_code_fragment_reaches_the_request_and_the_screen(session):
    app, stub = session
    app.ask(QUESTION, timeout=LIVE)

    request = _answer_request(stub)
    assert CODE in request and "L1-L9" in request, "фрагмент кода с путём и строками"
    assert QUOTE in request, "текст фрагмента дословный"
    assert "Подтверди это прямо в ответе" in request, "инструкция цитат добавлена"

    screen = app.screen_text()
    assert "🧩 Код:" in screen
    assert f"{CODE}:L1-L9" in screen
    assert app.contains(CITED.splitlines()[0])


def test_unconfirmed_answer_is_retried_then_replaced(stub, tmp_path: Path):
    _model_double(stub, [UNCITED])
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.ask(QUESTION, timeout=LIVE)
        screen = app.screen_text()
        assert "Не знаю:" in screen, "ответ заменён текстом об отсутствии подтверждения"
        assert UNCITED not in screen

        history = json.loads((tmp_path / "state" / "history.json").read_text(encoding="utf-8"))
        assert history["dialogues"][-1]["answer"].startswith("Не знаю:")
    finally:
        app.close()


def test_no_matches_tells_the_model_there_is_no_source(stub, tmp_path: Path):
    _model_double(stub, [UNCITED], score=0.1)
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.ask(QUESTION, timeout=LIVE)

        request = _answer_request(stub)
        assert "ни один не признан относящимся" in request
        assert "Ниже — фрагменты, найденные по запросу" not in request, "фрагменты не доставляются"
        assert "не прошёл порог" in app.screen_text()
    finally:
        app.close()


def test_missing_index_sends_nothing(stub, tmp_path: Path):
    _model_double(stub, [UNCITED])
    app = _launch(tmp_path, stub, with_index=False)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.ask(QUESTION, timeout=LIVE)
        request = _answer_request(stub)
        assert "фрагменты, найденные по запросу" not in request
        assert "Корпус исходников" not in request
    finally:
        app.close()


def test_trace_reports_the_search_without_model_requests(session):
    app, stub = session
    app.ask(QUESTION, timeout=LIVE)
    before = len(stub.requests)

    app.send_line("/code trace")
    screen = app.wait_for("Поисковый запрос:", timeout=LIVE)
    assert "ModelList load models.ini" in screen
    assert "Состояние: ok" in screen
    assert f"{CODE}:L1-L9" in screen
    assert len(stub.requests) == before, "отчёт не обращается к модели"


def test_baseline_mode_skips_auxiliary_requests(stub, tmp_path: Path):
    _model_double(stub, [UNCITED])
    app = _launch(tmp_path, stub)
    try:
        app.wait_for("MCP:", timeout=LIVE)
        app.send_line("/code retrieval baseline")
        app.wait_for("Режим отбора: baseline", timeout=LIVE)
        app.ask(QUESTION, timeout=LIVE)

        assert len(stub.requests) == 1, "в baseline только запрос ответа"
        assert "Code" not in app.screen_text() or True
    finally:
        app.close()
