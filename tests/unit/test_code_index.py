"""Корпус кода: отбор файлов, разбиение на фрагменты и индекс в кэше."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core import code_index, config, domains
from core.embeddings import EmbeddingsError

CORPUS = domains.load_domain("aurora-qt5").corpus


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(root: Path) -> Path:
    """Небольшой проект под Аврору: исходник, QML, spec, сборка, мусор рядом."""
    _write(root, "src/models.cpp", "\n".join(f"// строка {number}" for number in range(1, 41)))
    _write(root, "src/main.qml", "import QtQuick 2.0\n\nApplicationWindow {\n    id: window\n}\n")
    _write(root, "rpm/app.spec", "Name: app\nVersion: 1.0\n%build\nqmake\nmake\n")
    _write(root, "build/generated.cpp", "int generated;")
    _write(root, "src/notes.md", "# заметка")
    _write(root, "src/blob.cpp", "int a;\0int b;")
    _write(root, "src/huge.cpp", "// x\n" * 200_000)
    return root


# --- отбор файлов ---------------------------------------------------------------------------


def test_corpus_keeps_sources_and_explains_skips(tmp_path: Path):
    root = _project(tmp_path)
    result = code_index.scan(root, CORPUS)
    kept = sorted(path.relative_to(root).as_posix() for path in result.files)
    assert kept == ["rpm/app.spec", "src/main.qml", "src/models.cpp"]

    reasons = {item.path: item.reason for item in result.skipped}
    assert reasons["src/notes.md"] == code_index.SKIP_EXTENSION
    assert reasons["build/generated.cpp"] == code_index.SKIP_GLOBS
    assert reasons["src/blob.cpp"] == code_index.SKIP_BINARY
    assert reasons["src/huge.cpp"] == code_index.SKIP_TOO_LARGE


def test_scan_without_git_walks_the_tree(tmp_path: Path):
    root = _project(tmp_path)
    result = code_index.scan(root, CORPUS)
    assert result.source == "обход каталога"


def test_gitignore_is_respected(tmp_path: Path):
    root = _project(tmp_path)
    _write(root, "src/ignored.cpp", "int ignored;")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _write(root, ".gitignore", "ignored.cpp\n")
    result = code_index.scan(root, CORPUS)
    assert result.source == "git ls-files"
    kept = {path.relative_to(root).as_posix() for path in result.files}
    assert "src/ignored.cpp" not in kept
    assert "src/models.cpp" in kept


def test_glob_rule_without_slash_matches_any_directory():
    assert code_index._glob_matches("a/build/b.cpp", "build")
    assert code_index._glob_matches("build/b.cpp", "build")
    assert not code_index._glob_matches("builds/b.cpp", "build")
    assert code_index._glob_matches("a/vendor/b.cpp", "vendor/*")
    assert code_index._glob_matches("a/deep/x.lock", "*.lock")


# --- разбиение ------------------------------------------------------------------------------


def test_fixed_chunks_cover_lines_with_overlap():
    lines = [f"строка {number}" for number in range(1, 101)]
    chunks = code_index.chunk_fixed(lines, "src/a.cpp", chunk_lines=40, overlap_lines=10)
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 40),
        (31, 70),
        (61, 100),
    ]
    assert all(chunk.strategy == code_index.STRATEGY_FIXED for chunk in chunks)
    assert chunks[0].label == "src/a.cpp:L1-L40"


def test_fixed_chunks_shorter_file_is_one_chunk():
    lines = ["одна", "две"]
    chunks = code_index.chunk_fixed(lines, "a.cpp", chunk_lines=60, overlap_lines=10)
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 2)]


def test_chunk_text_matches_the_file_lines(tmp_path: Path):
    """Инвариант корпуса: текст фрагмента — ровно те строки файла, что названы диапазоном."""
    root = _project(tmp_path)
    path = root / "src" / "models.cpp"
    lines = code_index.read_lines(path)
    for strategy in code_index.STRATEGIES:
        for chunk in code_index.chunk_file(path, "src/models.cpp", CORPUS, strategy):
            assert chunk.text.splitlines() == lines[chunk.start_line - 1 : chunk.end_line]
            assert chunk.start_line <= chunk.end_line


def test_structural_chunks_start_on_block_boundaries(tmp_path: Path):
    root = tmp_path
    _write(
        root,
        "src/window.cpp",
        "\n".join(
            [
                "#include <QObject>",
                "",
                "class Window : public QObject {",
                "public:",
                "    void open();",
                "};",
                "",
                "void Window::open() {",
                "    ++count_;",
                "}",
            ]
            + [f"// хвост {number}" for number in range(1, 60)]
        ),
    )
    chunks = code_index.chunk_file(root / "src" / "window.cpp", "src/window.cpp", CORPUS, "structural")
    starts = {chunk.start_line for chunk in chunks}
    assert 8 in starts, "определение метода — граница блока"
    assert 1 in starts, "первая строка файла всегда начало блока"
    first = chunks[0]
    assert (first.start_line, first.end_line) == (1, 7)
    assert "class Window" in first.text, "мелкий блок класса склеен с началом файла"
    assert chunks[1].text.splitlines()[0].startswith("void Window::open")


def test_structural_splits_oversized_block(tmp_path: Path):
    body = [f"// строка {number}" for number in range(1, 301)]
    _write(tmp_path, "src/long.cpp", "\n".join(body))
    chunks = code_index.chunk_file(tmp_path / "src" / "long.cpp", "src/long.cpp", CORPUS, "structural")
    assert all(chunk.end_line - chunk.start_line + 1 <= CORPUS.structural_max_lines for chunk in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 300


def test_structural_merges_tiny_blocks(tmp_path: Path):
    _write(tmp_path, "rpm/tiny.spec", "Name: tiny\nVersion: 1\nRelease: 1\n%build\nmake\n")
    chunks = code_index.chunk_file(tmp_path / "rpm" / "tiny.spec", "rpm/tiny.spec", CORPUS, "structural")
    assert len(chunks) == 1, "мелкие блоки склеиваются в один фрагмент"
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 5)


def test_structural_falls_back_to_windows_without_rules(tmp_path: Path):
    _write(tmp_path, "config/app.yaml", "\n".join(f"key{number}: {number}" for number in range(1, 100)))
    chunks = code_index.chunk_file(tmp_path / "config" / "app.yaml", "config/app.yaml", CORPUS, "structural")
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == CORPUS.fixed_chunk_lines


def test_unknown_strategy_is_rejected(tmp_path: Path):
    with pytest.raises(code_index.CodeIndexError):
        code_index.chunk_file(tmp_path / "a.cpp", "a.cpp", CORPUS, "какая-то")


# --- индекс ---------------------------------------------------------------------------------


def _cache(tmp_path: Path) -> Path:
    """Индекс кэша лежит вне целевого репозитория — как и в жизни."""
    return tmp_path.parent / f"{tmp_path.name}-cache" / "code-index.sqlite3"


def test_build_index_and_read_report(tmp_path: Path):
    root = _project(tmp_path)
    database = _cache(tmp_path)
    report = code_index.build_index(root, CORPUS, code_index.STRATEGY_STRUCTURAL, database)
    assert report.files == 3
    assert report.chunks >= 3
    assert report.database == database
    assert database.is_file()

    restored = code_index.read_index(database)
    assert restored is not None
    assert (restored.files, restored.chunks, restored.strategy) == (
        report.files,
        report.chunks,
        report.strategy,
    )
    assert restored.skip_reasons()[code_index.SKIP_EXTENSION] == 1


def test_index_rows_keep_paths_and_ranges(tmp_path: Path):
    root = _project(tmp_path)
    database = _cache(tmp_path)
    code_index.build_index(root, CORPUS, code_index.STRATEGY_FIXED, database)
    import sqlite3

    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT path, start_line, end_line, content FROM chunks ORDER BY path, start_line"
    ).fetchall()
    connection.close()
    assert rows
    for path, start, end, content in rows:
        lines = code_index.read_lines(root / path)
        assert content.splitlines() == lines[start - 1 : end]


def test_index_is_not_written_into_the_repository(tmp_path: Path):
    root = _project(tmp_path)
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    code_index.build_index(root, CORPUS, code_index.STRATEGY_FIXED, _cache(tmp_path))
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert before == after, "в целевом репозитории не появляется ничего"


def test_missing_index_reads_as_none(tmp_path: Path):
    assert code_index.read_index(tmp_path / "нет.sqlite3") is None
    assert not code_index.index_exists(tmp_path / "нет.sqlite3")


def test_garbage_file_is_not_an_index(tmp_path: Path):
    database = tmp_path / "index.sqlite3"
    database.write_text("не база")
    assert code_index.read_index(database) is None


def test_failed_build_keeps_the_previous_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _project(tmp_path)
    database = _cache(tmp_path)
    first = code_index.build_index(root, CORPUS, code_index.STRATEGY_FIXED, database)

    _write(root, "src/second.cpp", "int second;")
    monkeypatch.setattr(code_index.os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("нет места")))
    with pytest.raises(code_index.CodeIndexError):
        code_index.build_index(root, CORPUS, code_index.STRATEGY_STRUCTURAL, database)

    restored = code_index.read_index(database)
    assert restored is not None
    assert restored.chunks == first.chunks, "прежний индекс остался пригодным"
    assert restored.strategy == code_index.STRATEGY_FIXED
    assert not (database.parent / "code-index.sqlite3.tmp").exists(), "временный файл убран"


def test_build_rejects_unknown_strategy(tmp_path: Path):
    with pytest.raises(code_index.CodeIndexError):
        code_index.build_index(tmp_path, CORPUS, "нет", tmp_path / "i.sqlite3")


def test_index_path_isolated_per_repository(tmp_path: Path):
    left = config.repo_index_file(tmp_path / "one")
    right = config.repo_index_file(tmp_path / "two")
    assert left != right
    assert left.name == "index.sqlite3"


def test_index_stays_out_of_the_temporary_index_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    override = tmp_path / "override.sqlite3"
    monkeypatch.setenv("FFAI_INDEX_FILE", str(override))
    assert config.repo_index_file(tmp_path / "any") == override


def test_collect_counts_both_strategies(tmp_path: Path):
    root = _project(tmp_path)
    fixed = code_index.collect(root, CORPUS, code_index.STRATEGY_FIXED)
    structural = code_index.collect(root, CORPUS, code_index.STRATEGY_STRUCTURAL)
    assert fixed and structural
    assert code_index.sample_labels(structural)
    assert all("L" in label for label in code_index.sample_labels(structural))


def test_file_at_the_size_limit_is_kept(tmp_path: Path):
    """Предел размера — включительный: файл ровно по пределу остаётся в корпусе."""
    limit = CORPUS.max_file_bytes
    body = "// " + "x" * (limit - 3)
    assert len(body.encode("utf-8")) == limit
    _write(tmp_path, "src/exact.cpp", body)
    _write(tmp_path, "src/over.cpp", body + "x")
    result = code_index.scan(tmp_path, CORPUS)
    kept = {path.name for path in result.files}
    assert "exact.cpp" in kept
    assert "over.cpp" not in kept


# --- векторы в индексе ----------------------------------------------------------------------


class FakeVectors:
    """Векторная модель-заглушка: вектор заданной длины, без сети и без загруженных весов."""

    def __init__(self, model: str = "fake-embed", dimensions: int = 3) -> None:
        self.model = model
        self.dimensions = dimensions
        self.seen: list = []

    def embed(self, text: str):
        self.seen.append(text)
        return [1.0] + [0.0] * (self.dimensions - 1)


class BrokenVectors(FakeVectors):
    """Модель, отвечающая на первые фрагменты и падающая на одном из следующих."""

    def __init__(self, model: str = "fake-embed", fail_after: int = 1) -> None:
        super().__init__(model)
        self.fail_after = fail_after

    def embed(self, text: str):
        if len(self.seen) >= self.fail_after:
            raise EmbeddingsError("сервер не отвечает")
        return super().embed(text)


class ShiftingVectors(FakeVectors):
    """Модель, у которой сменилась размерность посреди корпуса — признак смены весов пресета."""

    def embed(self, text: str):
        self.seen.append(text)
        width = self.dimensions if len(self.seen) == 1 else self.dimensions - 1
        return [1.0] + [0.0] * (width - 1)


def test_build_with_provider_stores_vectors_for_every_chunk(tmp_path: Path):
    """Вектор есть у каждого фрагмента, и записано, какой моделью он посчитан."""
    root = _project(tmp_path)
    database = _cache(tmp_path)
    provider = FakeVectors()

    report = code_index.build_index(
        root, CORPUS, code_index.STRATEGY_STRUCTURAL, database, provider=provider
    )

    assert report.vector_model == "fake-embed"
    assert report.vectors == report.chunks
    assert report.vector_note == ""
    assert len(provider.seen) == report.chunks

    vector_set = code_index.read_vectors(database)
    assert vector_set.model == "fake-embed"
    assert set(vector_set.vectors) == {
        chunk.chunk_id for chunk in code_index.read_chunks(database)
    }
    assert all(len(vector) == 3 for vector in vector_set.vectors.values())
    assert vector_set.note == ""

    restored = code_index.read_index(database)
    assert restored is not None
    assert (restored.vector_model, restored.vectors) == ("fake-embed", report.chunks)


def test_build_without_provider_keeps_the_token_path_and_names_the_reason(tmp_path: Path):
    """Провайдера нет — индекс собирается токенным, и причина видна, а не спрятана."""
    root = _project(tmp_path)
    database = _cache(tmp_path)

    report = code_index.build_index(root, CORPUS, code_index.STRATEGY_FIXED, database)

    assert (report.vectors, report.vector_model) == (0, "")
    assert "FFAI_LLAMA_AUTOSTART" in report.vector_note
    assert code_index.read_chunks(database), "токенный поиск работает и без векторов"
    assert code_index.read_vectors(database).note == report.vector_note


def test_embedding_failure_keeps_the_previous_index(tmp_path: Path):
    """Сбой модели отменяет сборку целиком: прежний индекс с векторами остаётся пригодным."""
    root = _project(tmp_path)
    database = _cache(tmp_path)
    first = code_index.build_index(
        root, CORPUS, code_index.STRATEGY_FIXED, database, provider=FakeVectors()
    )

    with pytest.raises(code_index.CodeIndexError) as error:
        code_index.build_index(
            root, CORPUS, code_index.STRATEGY_STRUCTURAL, database, provider=BrokenVectors()
        )

    assert "сервер не отвечает" in str(error.value)
    assert not (database.parent / "code-index.sqlite3.tmp").exists(), "временный файл убран"
    restored = code_index.read_index(database)
    assert restored is not None
    assert restored.strategy == code_index.STRATEGY_FIXED
    assert restored.vectors == first.vectors, "прежние векторы не тронуты"
    assert code_index.read_vectors(database).model == "fake-embed"


def test_changed_dimensions_cancel_the_build(tmp_path: Path):
    """Размерность изменилась посреди корпуса — числа несопоставимы, сборка не состоится."""
    root = _project(tmp_path)
    database = _cache(tmp_path)
    code_index.build_index(
        root, CORPUS, code_index.STRATEGY_FIXED, database, provider=FakeVectors()
    )

    with pytest.raises(code_index.CodeIndexError) as error:
        code_index.build_index(
            root, CORPUS, code_index.STRATEGY_STRUCTURAL, database, provider=ShiftingVectors()
        )

    assert "размерность" in str(error.value)
    restored = code_index.read_index(database)
    assert restored is not None and restored.strategy == code_index.STRATEGY_FIXED


def test_old_schema_index_is_not_read_as_usable(tmp_path: Path):
    """Индекс прошлой версии схемы не читается как пригодный: пересборка объявляется, а не подразумевается."""
    import sqlite3

    root = _project(tmp_path)
    database = _cache(tmp_path)
    code_index.build_index(
        root, CORPUS, code_index.STRATEGY_FIXED, database, provider=FakeVectors()
    )
    connection = sqlite3.connect(database)
    connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema'")
    connection.commit()
    connection.close()

    assert code_index.read_index(database) is None
    assert code_index.read_chunks(database) == ()
    assert code_index.read_vectors(database).model == ""
    assert "пересоберите" in code_index.read_vectors(database).note
