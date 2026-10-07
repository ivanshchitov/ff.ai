"""Долговременная память: отдельный файл, категории записей, замена по ключу и предел числа записей."""

import json
from pathlib import Path

import pytest

from core import config, domains, memory_layers
from core.long_term_memory import MAX_ENTRIES, LongTermMemory


def _category(layer: str = memory_layers.LONG_TERM) -> str:
    """Категория из данных домена: тест не должен знать ни одного доменного слова сам."""
    memory = domains.load_domain("aurora-qt5").memory
    assert memory is not None, "у пакета домена должны быть правила памяти"
    return next(rule.category for rule in memory.rules if rule.layer == layer)


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.json"


@pytest.fixture
def category() -> str:
    return _category()


def test_missing_file_gives_empty_memory(memory_path):
    memory = LongTermMemory(path=memory_path)

    assert memory.records() == ()
    assert memory.get("опыт") is None
    assert memory.last_error is None


def test_corrupted_json_is_tolerated(memory_path):
    memory_path.write_text("{не json", encoding="utf-8")

    memory = LongTermMemory(path=memory_path)

    assert memory.records() == ()
    assert memory.last_error, "причина читается отчётом, а не выпадает в лог"


def test_foreign_shape_is_tolerated(memory_path):
    """Чужая форма файла — это отсутствие памяти, а не падение: записывать поверх можно."""
    memory_path.write_text(json.dumps(["не объект"]), encoding="utf-8")

    memory = LongTermMemory(path=memory_path)
    memory.remember(key="ключ", value="значение")

    assert memory.get("ключ") == "значение"


def test_remember_persists_immediately(memory_path, category):
    memory = LongTermMemory(path=memory_path)
    memory.remember(key="опыт", value="знаю C++", category=category)

    saved = json.loads(memory_path.read_text(encoding="utf-8"))
    assert saved["entries"]["опыт"] == {"value": "знаю C++", "category": category}


def test_records_survive_reload(memory_path, category):
    LongTermMemory(path=memory_path).remember(key="решения", value="только qmake", category=category)
    reloaded = LongTermMemory(path=memory_path)

    records = reloaded.records()
    assert [record.key for record in records] == ["решения"]
    assert records[0].layer == memory_layers.LONG_TERM
    assert records[0].category == category
    assert records[0].value == "только qmake"


def test_repeat_keeps_one_entry_with_the_last_value(memory_path, category):
    memory = LongTermMemory(path=memory_path)
    memory.remember(key="опыт", value="новичок", category=category)
    memory.remember(key="опыт", value="знаю C++", category=category)

    assert memory.get("опыт") == "знаю C++"
    assert len(memory.records()) == 1


def test_repeat_moves_the_entry_to_the_end(memory_path, category):
    """Вытесняется самая давняя запись, а не та, к которой только что вернулись."""
    memory = LongTermMemory(path=memory_path)
    for index in range(MAX_ENTRIES):
        memory.remember(key=f"ключ {index}", value="значение", category=category)
    memory.remember(key="ключ 0", value="обновлено", category=category)
    memory.remember(key="последний", value="значение", category=category)

    keys = [record.key for record in memory.records()]
    assert "ключ 0" in keys
    assert "ключ 1" not in keys


def test_number_of_entries_is_bounded(memory_path, category):
    memory = LongTermMemory(path=memory_path)
    for index in range(MAX_ENTRIES + 5):
        memory.remember(key=f"ключ {index}", value="значение", category=category)

    assert len(memory.records()) == MAX_ENTRIES
    assert memory.get("ключ 0") is None
    assert memory.get(f"ключ {MAX_ENTRIES + 4}") == "значение"


def test_forget_removes_one_entry(memory_path, category):
    memory = LongTermMemory(path=memory_path)
    memory.remember(key="опыт", value="знаю C++", category=category)
    memory.remember(key="решения", value="только qmake", category=category)

    assert memory.forget("опыт") is True
    assert memory.forget("нет такого") is False
    assert [record.key for record in LongTermMemory(path=memory_path).records()] == ["решения"]


def test_clear_wipes_the_file(memory_path, category):
    memory = LongTermMemory(path=memory_path)
    memory.remember(key="опыт", value="знаю C++", category=category)
    memory.clear()

    assert memory.records() == ()
    assert LongTermMemory(path=memory_path).records() == ()


def test_entries_without_category_read_as_notes(memory_path):
    """Файл могли править руками: отсутствующее поле не должно терять значение."""
    memory_path.write_text(
        json.dumps({"entries": {"заметка 1": {"value": "сборка через qmake"}}}),
        encoding="utf-8",
    )

    records = LongTermMemory(path=memory_path).records()
    assert records[0].category == memory_layers.CATEGORY_NOTE
    assert records[0].value == "сборка через qmake"


def test_plain_string_entries_read_as_notes(memory_path):
    memory_path.write_text(
        json.dumps({"entries": {"заметка 1": "сборка через qmake"}}), encoding="utf-8"
    )

    records = LongTermMemory(path=memory_path).records()
    assert records[0].category == memory_layers.CATEGORY_NOTE


def test_value_is_clipped_to_the_record_limit(memory_path, category):
    memory = LongTermMemory(path=memory_path)
    long_value = "о" * (memory_layers.MEMORY_VALUE_MAX_CHARS + 50)

    memory.remember(key="опыт", value=long_value, category=category)

    assert len(memory.get("опыт")) == memory_layers.MEMORY_VALUE_MAX_CHARS
    assert memory.get("опыт").endswith("…")


def test_save_survives_unwritable_path(tmp_path, category):
    """Ошибка записи не должна ронять приложение — память просто не сохранится."""
    target = tmp_path / "memory.json"
    target.mkdir()
    memory = LongTermMemory(path=target)

    memory.remember(key="опыт", value="знаю C++", category=category)

    assert memory.get("опыт") == "знаю C++"
    assert memory.last_error, "причина видна отчёту, а не только в логе ошибок"


def test_default_path_is_read_at_creation_time():
    """Путь состояния ленивый: значение по умолчанию не «замерзает» на импорте модуля."""
    assert LongTermMemory().path == config.MEMORY_FILE
