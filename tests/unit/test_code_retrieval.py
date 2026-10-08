"""Поиск по корпусу кода: локальные векторы, две ступени отбора и состояния снимка."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import code_index, code_retrieval, config, domains
from core.embeddings import EmbeddingsError

CORPUS = domains.load_domain("aurora-qt5").corpus


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(root: Path) -> Path:
    """Проект с тремя темами: модель списка, подпись пакета и графика."""
    _write(
        root,
        "src/models.cpp",
        "\n".join(
            [
                "#include <QObject>",
                "class ModelList : public QObject {",
                "public:",
                "    void load();",
                "};",
                "",
                "void ModelList::load() {",
                "    const QString path = settings_.value(\"models.ini\");",
                "    model_ = new QAbstractListModel(this);",
                "    model_->setSource(path);",
                "}",
            ]
        ),
    )
    _write(
        root,
        "rpm/app.spec",
        "\n".join(
            [
                "Name: aurora-notes",
                "Version: 0.1",
                "%build",
                "qmake",
                "%files",
                "%{_bindir}/aurora-notes",
            ]
        ),
    )
    _write(
        root,
        "src/render.cpp",
        "\n".join(
            [
                "void Renderer::paint(QPainter &painter) {",
                "    painter.setBrush(background_);",
                "    painter.drawRoundedRect(rect_, 8, 8);",
                "}",
            ]
        ),
    )
    return root


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    return _project(tmp_path / "repo")


@pytest.fixture
def retriever(repository: Path, tmp_path: Path) -> code_retrieval.CodeRetriever:
    database = tmp_path / "cache" / "index.sqlite3"
    code_index.build_index(repository, CORPUS, code_index.STRATEGY_STRUCTURAL, database)
    return code_retrieval.CodeRetriever(database, CORPUS)


def _ratings(candidates, score: float = 0.9) -> str:
    return json.dumps(
        {
            "results": [
                {"id": index, "score": score, "reason": "тестовая оценка"}
                for index in range(1, len(candidates) + 1)
            ]
        },
        ensure_ascii=False,
    )


# --- векторы и порядок ----------------------------------------------------------------------


def test_tokens_split_identifiers():
    tokens = code_retrieval.tokenize("void ModelList::loadFromSpec(const QString &path)")
    assert "model" in tokens and "list" in tokens
    assert "loadfromspec" in tokens and "from" in tokens and "spec" in tokens
    assert "const" in tokens
    assert "то" not in tokens, "слишком короткие токены отбрасываются"


def test_cosine_bounds():
    assert code_retrieval.cosine({}, {"a": 1.0}) == 0.0
    assert code_retrieval.cosine({"a": 1.0}, {"b": 1.0}) == 0.0
    assert code_retrieval.cosine({"a": 1.0}, {"a": 2.0}) == pytest.approx(1.0)


def test_relevant_chunk_ranks_first(retriever):
    report = retriever.search("ModelList load settings models", mode="baseline")
    assert report.status == code_retrieval.STATUS_OK
    assert report.candidates[0].path.startswith("src/models.cpp")
    assert report.fragments[0].identifier.startswith("src/models.cpp:L")


def test_russian_question_needs_the_rewrite(retriever):
    """Вопрос по-русски не пересекается с английским кодом — ровно поэтому нужна вторая ступень."""
    plain = retriever.search("где инициализируется модель списка моделей?", mode="baseline")
    assert plain.status == code_retrieval.STATUS_NO_CANDIDATES

    rewritten = retriever.search(
        "где инициализируется модель списка моделей?",
        mode="enhanced",
        rewrite=lambda q: "ModelList load QAbstractListModel settings",
        rate=lambda q, c: _ratings(c),
    )
    assert rewritten.status == code_retrieval.STATUS_OK
    assert rewritten.candidates[0].path.startswith("src/models.cpp")


def test_zero_similarity_is_not_a_candidate(retriever):
    """Нулевая близость — не кандидат: до оценки дело не доходит, и это видно по состоянию."""
    report = retriever.search("аврора", mode="baseline")
    assert report.status == code_retrieval.STATUS_NO_CANDIDATES
    assert report.candidates == ()


def test_common_words_do_not_beat_identifiers(retriever):
    """idf: редкий идентификатор весит больше, чем «int»/«void», встречающиеся везде."""
    report = retriever.search("Renderer paint", mode="baseline")
    assert report.candidates[0].path == "src/render.cpp:L1-L4"


def test_order_is_stable(retriever):
    first = retriever.search("подпись пакета spec", mode="baseline")
    second = retriever.search("подпись пакета spec", mode="baseline")
    assert [item.identifier for item in first.candidates] == [
        item.identifier for item in second.candidates
    ]


def test_fragment_text_is_verbatim(repository, retriever):
    report = retriever.search("ModelList", mode="baseline")
    for fragment in report.fragments:
        lines = code_index.read_lines(repository / fragment.identifier.split(":")[0])
        body = lines[fragment.start_line - 1 : fragment.end_line]
        assert fragment.text.splitlines()[: len(body)] == body


# --- ступени --------------------------------------------------------------------------------


def test_baseline_makes_no_auxiliary_calls(retriever):
    def forbidden(*args, **kwargs):  # pragma: no cover - вызов означает нарушение
        raise AssertionError("в baseline вспомогательных запросов быть не должно")

    report = retriever.search("ModelList", mode="baseline", rewrite=forbidden, rate=forbidden)
    assert report.status == code_retrieval.STATUS_OK
    assert report.rated is False
    assert report.query == "ModelList"


def test_baseline_delivers_first_results_and_ignores_threshold(retriever):
    """Порога в baseline нет: даже нулевые оценки не мешают доставке первых результатов."""
    report = retriever.search("ModelList qmake spec", mode="baseline", threshold=1.0, after=2)
    assert len(report.fragments) == 2, "порог в baseline не применяется"
    assert report.candidates[0].score > 0


def test_baseline_respects_after(retriever):
    report = retriever.search("qmake spec файлы пакета", mode="baseline", after=1)
    assert len(report.fragments) == 1


def test_enhanced_searches_by_the_rewritten_query(retriever):
    seen: list = []

    def rewrite(question: str) -> str:
        seen.append(question)
        return "painter brush RoundedRect"

    report = retriever.search("как рисуется фон?", mode="enhanced", rewrite=rewrite, rate=lambda q, c: _ratings(c))
    assert seen == ["как рисуется фон?"]
    assert report.query == "painter brush RoundedRect"
    assert report.candidates[0].path == "src/render.cpp:L1-L4"
    assert report.rated is True


def test_enhanced_applies_threshold_and_after(retriever):
    def rate(question, candidates):
        # Первому кандидату — высокая оценка, остальным — низкая.
        return json.dumps(
            {
                "results": [
                    {
                        "id": index,
                        "score": 0.9 if index == 1 else 0.1,
                        "reason": "тест",
                    }
                    for index in range(1, len(candidates) + 1)
                ]
            },
            ensure_ascii=False,
        )

    report = retriever.search(
        "ModelList", mode="enhanced", rewrite=lambda q: q, rate=rate, threshold=0.6
    )
    assert report.status == code_retrieval.STATUS_OK
    assert len(report.fragments) == 1
    assert report.fragments[0].identifier == report.candidates[0].identifier
    assert len(report.dropped()) == len(report.candidates) - 1


def test_enhanced_all_below_threshold_is_no_matches(retriever):
    report = retriever.search(
        "ModelList",
        mode="enhanced",
        rewrite=lambda q: q,
        rate=lambda q, c: _ratings(c, score=0.2),
        threshold=0.6,
    )
    assert report.status == code_retrieval.STATUS_NO_MATCHES
    assert report.fragments == ()
    assert report.rated is True
    assert report.candidates, "кандидаты остаются в снимке — видно, что искали"


def test_before_caps_the_pool(retriever):
    seen: list = []

    def rate(question, candidates):
        seen.append(len(candidates))
        return _ratings(candidates)

    retriever.search("qmake", mode="enhanced", rewrite=lambda q: q, rate=rate, before=1)
    assert seen == [1]


def test_snippet_given_to_the_evaluator_is_bounded(retriever, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "CODE_SNIPPET_CHARS", 20)
    seen: list = []

    def rate(question, candidates):
        seen.extend(len(candidate.snippet) for candidate in candidates)
        return _ratings(candidates)

    retriever.search("ModelList", mode="enhanced", rewrite=lambda q: q, rate=rate)
    assert seen and max(seen) <= 20


# --- состояния ------------------------------------------------------------------------------


def test_missing_index_is_unavailable(tmp_path: Path):
    empty = code_retrieval.CodeRetriever(tmp_path / "нет.sqlite3", CORPUS)
    report = empty.search("ModelList", mode="baseline")
    assert report.status == code_retrieval.STATUS_UNAVAILABLE
    assert report.fragments == ()
    assert report.error


def test_unknown_mode_is_disabled(tmp_path: Path):
    empty = code_retrieval.CodeRetriever(tmp_path / "нет.sqlite3", CORPUS)
    report = empty.search("ModelList", mode="какой-то")
    assert report.status == code_retrieval.STATUS_DISABLED


def test_no_candidates_when_nothing_matches(retriever):
    report = retriever.search("ёёё-несуществующее-слово", mode="baseline")
    assert report.status == code_retrieval.STATUS_NO_CANDIDATES
    assert report.fragments == ()


def test_failed_rewrite_is_visible(retriever):
    report = retriever.search("ModelList", mode="enhanced", rewrite=lambda q: "", rate=lambda q, c: _ratings(c))
    assert report.status == code_retrieval.STATUS_RERANK_FAILED
    assert "пустой" in report.error
    assert report.fragments == ()


def test_exception_in_rewrite_is_visible(retriever):
    def boom(question: str) -> str:
        raise RuntimeError("модель недоступна")

    report = retriever.search("ModelList", mode="enhanced", rewrite=boom, rate=lambda q, c: _ratings(c))
    assert report.status == code_retrieval.STATUS_RERANK_FAILED
    assert "недоступна" in report.error


def test_exception_in_rating_is_visible(retriever):
    def boom(question, candidates):
        raise RuntimeError("оценщик недоступен")

    report = retriever.search("ModelList", mode="enhanced", rewrite=lambda q: q, rate=boom)
    assert report.status == code_retrieval.STATUS_RERANK_FAILED
    assert "недоступен" in report.error
    assert report.candidates, "сырые кандидаты не доставляются"
    assert report.fragments == ()


def test_garbage_rating_is_visible(retriever):
    report = retriever.search("ModelList", mode="enhanced", rewrite=lambda q: q, rate=lambda q, c: "не JSON")
    assert report.status == code_retrieval.STATUS_RERANK_FAILED
    assert report.fragments == ()


def test_enhanced_without_callbacks_is_a_visible_failure(retriever):
    report = retriever.search("ModelList", mode="enhanced")
    assert report.status == code_retrieval.STATUS_RERANK_FAILED
    assert report.error


def test_fragment_is_clipped_with_a_marker(repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "CODE_FRAGMENT_MAX_CHARS", 30)
    database = tmp_path / "cache" / "index.sqlite3"
    code_index.build_index(repository, CORPUS, code_index.STRATEGY_STRUCTURAL, database)
    retriever = code_retrieval.CodeRetriever(database, CORPUS)
    report = retriever.search("ModelList", mode="baseline")
    assert report.fragments[0].truncated is True
    assert len(report.fragments[0].text) == 30


def test_instructions_come_from_assets():
    assert "поисковый запрос" in code_retrieval.query_instruction()
    assert "JSON" in code_retrieval.rerank_instruction()
    messages = code_retrieval.query_messages("где модель списка?")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "где модель списка?"


def test_rerank_messages_carry_snippet_limit(retriever, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "CODE_SNIPPET_CHARS", 15)
    report = retriever.search("ModelList", mode="baseline")
    messages = code_retrieval.rerank_messages("ModelList", report.candidates)
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert payload["candidates"]
    assert all(len(item["snippet"]) <= 15 for item in payload["candidates"])
    assert payload["candidates"][0]["path"].startswith("src/")


def test_rate_lines_show_scores(retriever):
    report = retriever.search("ModelList qmake spec", mode="baseline")
    lines = code_retrieval.rate_lines(report.candidates[:2])
    assert len(lines) == 2
    assert all("—" in line for line in lines)


# --- векторный путь -------------------------------------------------------------------------


class FakeEmbedding:
    """Векторная модель-заглушка: тема текста вместо весов — семантика без сети.

    Пространство трёхмерное и осмысленное: «модель списка», «подпись пакета» и «рисование»
    разнесены по осям, поэтому русский вопрос и английский код одной темы близки, а разных — нет.
    Настоящая модель доказывает то же самое живьём (`docs/roadmap.md`, живой прогон фазы), но
    проверка не должна зависеть от загруженных весов и от сервера.
    """

    TOPICS = (
        ("списк", "modellist", "qabstractlistmodel", "models.ini"),
        ("пакет", "подпись", "spec", "rpm"),
        ("рисуется", "фон", "график", "painter", "renderer", "brush"),
    )

    def __init__(self, model: str = "fake-embed", dimensions: int = None) -> None:
        self.model = model
        self._dimensions = dimensions or len(self.TOPICS)

    def embed(self, text: str):
        return self._vector(text)

    def embed_query(self, text: str):
        return self._vector(text)

    def _vector(self, text: str):
        lowered = (text or "").casefold()
        values = [
            1.0 if any(stem in lowered for stem in stems) else 0.0 for stems in self.TOPICS
        ]
        if self._dimensions != len(self.TOPICS):
            values = values[: self._dimensions]
        return values


class BrokenQueryEmbedding(FakeEmbedding):
    """Модель, которая отвечает на фрагменты, но падает на векторе вопроса."""

    def embed_query(self, text: str):
        raise EmbeddingsError("сервер не отвечает")


def _vector_index(repository: Path, tmp_path: Path, provider=None):
    database = tmp_path / "cache" / "vectors.sqlite3"
    code_index.build_index(
        repository,
        CORPUS,
        code_index.STRATEGY_STRUCTURAL,
        database,
        provider=provider or FakeEmbedding(),
    )
    return database


def test_russian_question_finds_english_code_by_vectors(repository: Path, tmp_path: Path):
    """Приёмка фазы: векторный поиск сближает языки, и переформулировка больше не нужна."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())

    def forbidden(*args, **kwargs):  # pragma: no cover - вызов означает лишний запрос
        raise AssertionError("векторному поиску переформулировка не нужна")

    report = retriever.search(
        "где инициализируется модель списка моделей?",
        mode="baseline",
        rewrite=forbidden,
        rate=forbidden,
        ranking="vectors",
    )

    assert report.status == code_retrieval.STATUS_OK
    assert report.ranking == code_retrieval.RANKING_VECTORS
    assert report.candidates[0].path.startswith("src/models.cpp")
    assert report.fragments[0].identifier.startswith("src/models.cpp")
    assert report.candidates[0].score == pytest.approx(1.0), "оценка — косинус векторов"
    assert report.ranking_note == "векторная модель fake-embed"


def test_tokens_on_the_same_index_find_nothing_in_russian(repository: Path, tmp_path: Path):
    """Контраст к предыдущему: токенам языки мешают, и это видно, а не спрятано."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())

    report = retriever.search("где инициализируется модель списка моделей?", mode="baseline")

    assert report.ranking == code_retrieval.RANKING_VECTORS, (
        "по умолчанию индекс с векторами идёт по векторам"
    )
    assert report.ranking_note.startswith("векторная модель")

    tokens = retriever.search(
        "где инициализируется модель списка моделей?", mode="baseline", ranking="tokens"
    )
    assert tokens.ranking == code_retrieval.RANKING_TOKENS
    assert tokens.ranking_note == "выбран явный токенный режим"
    assert tokens.status == code_retrieval.STATUS_NO_CANDIDATES


def test_explicit_token_mode_still_ranks_by_identifiers(repository: Path, tmp_path: Path):
    """Токенный режим нужен на идентификаторах: `ModelList` он находит по буквам, а не по теме."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())

    report = retriever.search("ModelList load settings", mode="baseline", ranking="tokens")

    assert report.ranking == code_retrieval.RANKING_TOKENS
    assert report.candidates[0].path.startswith("src/models.cpp")


def test_enhanced_with_vectors_skips_the_rewrite(repository: Path, tmp_path: Path):
    """Переформулировка — лекарство от несовпадения языков; с векторами её место занимает модель."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())
    question = "где инициализируется модель списка моделей?"

    def forbidden(question: str) -> str:  # pragma: no cover - вызов означает лишний запрос
        raise AssertionError("переформулировка не нужна, когда вопрос понимает векторная модель")

    report = retriever.search(
        question,
        mode="enhanced",
        rewrite=forbidden,
        rate=lambda q, candidates: _ratings(candidates),
    )

    assert report.ranking == code_retrieval.RANKING_VECTORS
    assert report.query == question, "поисковый запрос — сам вопрос"
    assert report.status == code_retrieval.STATUS_OK
    assert report.rated is True


def test_strict_vector_mode_is_visible_without_vectors(repository: Path, tmp_path: Path):
    """Строгий режим не подменяет векторы токенами молча: он называет причину и не доставляет."""
    database = tmp_path / "cache" / "index.sqlite3"
    code_index.build_index(repository, CORPUS, code_index.STRATEGY_STRUCTURAL, database)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())

    report = retriever.search("ModelList", mode="baseline", ranking="vectors")

    assert report.status == code_retrieval.STATUS_UNAVAILABLE
    assert report.fragments == ()
    assert "вектор" in report.error


def test_vectors_of_another_model_are_not_used(repository: Path, tmp_path: Path):
    """Смешать пространства двух моделей — получить оценку без смысла, поэтому чужое не берётся."""
    database = _vector_index(repository, tmp_path, provider=FakeEmbedding("fake-embed"))
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding("other-embed"))

    report = retriever.search("ModelList load", mode="baseline")

    assert report.ranking == code_retrieval.RANKING_TOKENS
    assert "fake-embed" in report.ranking_note and "other-embed" in report.ranking_note
    assert report.status == code_retrieval.STATUS_OK, "токенный путь остаётся рабочим"


def test_provider_absence_is_named_when_the_index_has_vectors(repository: Path, tmp_path: Path):
    """Локального сервера нет — причина в отчёте, а поиск идёт токенным путём (автозапуск выключен)."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS)  # провайдер берётся по умолчанию

    report = retriever.search("ModelList load", mode="baseline")

    assert report.ranking == code_retrieval.RANKING_TOKENS
    assert "FFAI_LLAMA_AUTOSTART" in report.ranking_note
    assert report.status == code_retrieval.STATUS_OK


def test_index_without_vectors_names_the_reason(repository: Path, tmp_path: Path):
    """Индекс собран без векторов — отчёт говорит, почему, а не молчит о способе."""
    database = tmp_path / "cache" / "index.sqlite3"
    code_index.build_index(repository, CORPUS, code_index.STRATEGY_STRUCTURAL, database)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())

    report = retriever.search("ModelList load", mode="baseline")

    assert report.ranking == code_retrieval.RANKING_TOKENS
    assert "автозапуск" in report.ranking_note


def test_query_embedding_failure_degrades_visibly(repository: Path, tmp_path: Path):
    """Сбой модели на векторе вопроса — названная причина, а не выдача токенного за векторный."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=BrokenQueryEmbedding())

    report = retriever.search("ModelList load", mode="baseline")

    assert report.ranking == code_retrieval.RANKING_TOKENS
    assert "сервер не отвечает" in report.ranking_note


def test_changed_dimensions_degrade_visibly(repository: Path, tmp_path: Path):
    """Чужая размерность вектора запроса обнаруживается на сравнении, а не превращается в ноль."""
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(
        database, CORPUS, embedding=FakeEmbedding("fake-embed", dimensions=1)
    )

    report = retriever.search("ModelList load", mode="baseline")

    assert report.ranking == code_retrieval.RANKING_TOKENS
    assert "размерности" in report.ranking_note


def test_unknown_ranking_is_disabled(repository: Path, tmp_path: Path):
    database = _vector_index(repository, tmp_path)
    retriever = code_retrieval.CodeRetriever(database, CORPUS, embedding=FakeEmbedding())

    report = retriever.search("ModelList", mode="baseline", ranking="какой-то")

    assert report.status == code_retrieval.STATUS_DISABLED
    assert "способ ранжирования" in report.error
