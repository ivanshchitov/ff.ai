"""Загрузка, накопление и сохранение истории диалогов: конверт с резюме, без вытеснения."""

import json

import pytest

from core.history_manager import HistoryManager


def test_missing_file_gives_empty_history(tmp_path):
    manager = HistoryManager(path=tmp_path / "nope.json")
    assert manager.dialogues == []
    assert manager.summary is None
    assert manager.summary_covers == 0
    assert manager.count() == 0


def test_corrupted_json_is_tolerated(history_path):
    """Битый файл истории не должен ронять запуск приложения."""
    history_path.write_text("{не json", encoding="utf-8")
    assert HistoryManager(path=history_path).dialogues == []


def test_dict_without_dialogues_is_ignored(history_path):
    history_path.write_text(json.dumps({"question": "q"}), encoding="utf-8")
    assert HistoryManager(path=history_path).dialogues == []


def test_add_appends_and_persists_immediately(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("Правила Splendor?", "Собирайте фишки.")
    saved = json.loads(history_path.read_text(encoding="utf-8"))
    assert saved == {
        "summary": None,
        "summary_covers": 0,
        "facts": {},
        "working": {},
        "dialogues": [{"question": "Правила Splendor?", "answer": "Собирайте фишки."}],
    }


def test_history_survives_reload(history_path):
    HistoryManager(path=history_path).add("Вопрос", "Ответ")
    reloaded = HistoryManager(path=history_path)
    assert reloaded.dialogues == [{"question": "Вопрос", "answer": "Ответ"}]


def test_add_never_evicts_earlier_records(history_path):
    manager = HistoryManager(path=history_path)
    for i in range(10):
        manager.add(f"q{i}", f"a{i}")
    assert manager.count() == 10


def test_load_keeps_every_record_unchanged(history_path):
    records = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(60)]
    manager = HistoryManager(path=history_path)
    manager.dialogues = records
    manager.save()

    reloaded = HistoryManager(path=history_path)
    assert [d["question"] for d in reloaded.dialogues][:3] == ["q0", "q1", "q2"]
    assert reloaded.count() == 60


def test_old_bare_list_loads_as_envelope_without_summary(history_path):
    history_path.write_text(json.dumps([{"question": "q", "answer": "a"}]), encoding="utf-8")
    manager = HistoryManager(path=history_path)
    assert manager.dialogues == [{"question": "q", "answer": "a"}]
    assert manager.summary is None
    assert manager.summary_covers == 0


def test_set_summary_persists_envelope(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("Ранний вопрос", "Ранний ответ")
    manager.set_summary("Ранний обмен: подбор игры.", 1)

    saved = json.loads(history_path.read_text(encoding="utf-8"))
    assert saved["summary"] == "Ранний обмен: подбор игры."
    assert saved["summary_covers"] == 1
    assert saved["dialogues"] == [{"question": "Ранний вопрос", "answer": "Ранний ответ"}]

    reloaded = HistoryManager(path=history_path)
    assert reloaded.summary == "Ранний обмен: подбор игры."
    assert reloaded.summary_covers == 1
    assert reloaded.dialogues == [{"question": "Ранний вопрос", "answer": "Ранний ответ"}]


def test_clear_writes_empty_envelope(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q", "a")
    manager.set_summary("Дайджест.", 1)
    manager.clear()

    assert manager.dialogues == []
    assert manager.summary is None
    assert manager.summary_covers == 0
    assert json.loads(history_path.read_text(encoding="utf-8")) == {
        "summary": None,
        "summary_covers": 0,
        "facts": {},
        "working": {},
        "dialogues": [],
    }


def test_cyrillic_is_stored_readable(history_path):
    HistoryManager(path=history_path).add("Каркассон", "Мипл 🎲")
    raw = history_path.read_text(encoding="utf-8")
    assert "Каркассон" in raw
    assert "\\u" not in raw


def test_save_survives_unwritable_path(tmp_path):
    """Ошибка записи не должна ронять приложение — история просто не сохранится."""
    path = tmp_path / "history.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    path.chmod(0o000)
    manager = HistoryManager(path=path)
    try:
        manager.add("q", "a")
    finally:
        path.chmod(0o644)
    assert manager.dialogues == [{"question": "q", "answer": "a"}]


def _usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.5):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
    }


def test_add_persists_usage_block(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q", "a", usage=_usage())
    saved = json.loads(history_path.read_text(encoding="utf-8"))["dialogues"]
    assert saved == [{"question": "q", "answer": "a", "usage": _usage()}]


def test_add_without_usage_keeps_old_record_shape(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q", "a")
    saved = json.loads(history_path.read_text(encoding="utf-8"))["dialogues"]
    assert saved == [{"question": "q", "answer": "a"}]


def test_old_format_file_loads_without_usage(history_path):
    history_path.write_text(json.dumps([{"question": "q", "answer": "a"}]), encoding="utf-8")
    manager = HistoryManager(path=history_path)
    assert manager.total_usage().requests == 0


def test_total_usage_sums_usage_across_records(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q1", "a1", usage=_usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.25))
    manager.add("q2", "a2", usage=_usage(prompt_tokens=20, completion_tokens=10, cost_usd=0.5))
    usage = manager.total_usage()
    assert usage.requests == 2
    assert usage.prompt_tokens == 30
    assert usage.cost_usd == 0.75


def test_total_usage_unknown_cost_poisons_total(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q", "a", usage=_usage(cost_usd=None))
    assert manager.total_usage().cost_usd is None


def test_every_record_carries_its_usage_no_eviction(history_path):
    """Записи не вытесняются: расход всего файла считается по всем обменам."""
    manager = HistoryManager(path=history_path)
    for i in range(10):
        manager.add(f"q{i}", f"a{i}", usage=_usage(prompt_tokens=10, completion_tokens=5))
    assert manager.total_usage().prompt_tokens == 100


def test_usage_block_survives_reload(history_path):
    HistoryManager(path=history_path).add("q", "a", usage=_usage())
    reloaded = HistoryManager(path=history_path)
    assert reloaded.dialogues[0]["usage"] == _usage()


# --- день 10: блок фактов в конверте ----------------------------------------------------


def test_facts_are_saved_and_reloaded(history_path):
    manager = HistoryManager(path=history_path)
    manager.set_facts({"цель": "собрать ТЗ", "жанр": "настольные игры"})

    saved = json.loads(history_path.read_text(encoding="utf-8"))
    assert saved["facts"] == {"цель": "собрать ТЗ", "жанр": "настольные игры"}

    reloaded = HistoryManager(path=history_path)
    assert reloaded.facts == {"цель": "собрать ТЗ", "жанр": "настольные игры"}


def test_missing_facts_key_reads_as_an_empty_block(history_path):
    history_path.write_text(
        json.dumps(
            {
                "summary": None,
                "summary_covers": 0,
        "facts": {},
                "dialogues": [{"question": "q", "answer": "a"}],
            }
        ),
        encoding="utf-8",
    )

    manager = HistoryManager(path=history_path)

    assert manager.facts == {}
    assert manager.dialogues == [{"question": "q", "answer": "a"}]


def test_clear_wipes_facts_too(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q", "a")
    manager.set_facts({"цель": "ТЗ"})
    manager.clear()

    assert manager.facts == {}
    assert json.loads(history_path.read_text(encoding="utf-8")) == {
        "summary": None,
        "summary_covers": 0,
        "facts": {},
        "working": {},
        "dialogues": [],
    }


# --- день 11: рабочая память в конверте --------------------------------------------------


def test_working_memory_is_saved_and_reloaded(history_path):
    manager = HistoryManager(path=history_path)
    manager.set_working({"цель": "собрать партию на вечер", "ограничения": "без таймера"})

    reloaded = HistoryManager(path=history_path)

    assert reloaded.working == {
        "цель": "собрать партию на вечер",
        "ограничения": "без таймера",
    }


def test_missing_working_key_reads_as_empty(history_path):
    """Файл дня 10 рабочей памяти не знает — она читается пустой, ошибок нет."""
    history_path.write_text(
        json.dumps(
            {
                "summary": None,
                "summary_covers": 0,
                "facts": {},
                "dialogues": [{"question": "q", "answer": "a"}],
            }
        ),
        encoding="utf-8",
    )

    manager = HistoryManager(path=history_path)

    assert manager.working == {}
    assert manager.dialogues == [{"question": "q", "answer": "a"}]


def test_clear_wipes_working_memory_too(history_path):
    manager = HistoryManager(path=history_path)
    manager.add("q", "a")
    manager.set_working({"цель": "собрать партию"})
    manager.clear()

    assert manager.working == {}
    assert json.loads(history_path.read_text(encoding="utf-8")) == {
        "summary": None,
        "summary_covers": 0,
        "facts": {},
        "working": {},
        "dialogues": [],
    }


def test_envelope_carries_working_beside_facts(history_path):
    """Слои хранятся раздельно: рабочая память — свой блок конверта, долговременная — свой файл."""
    manager = HistoryManager(path=history_path)
    manager.add("q", "a")
    manager.set_facts({"жанр": "евро"})
    manager.set_working({"цель": "собрать партию"})

    saved = json.loads(history_path.read_text(encoding="utf-8"))

    assert saved["working"] == {"цель": "собрать партию"}
    assert saved["facts"] == {"жанр": "евро"}
    assert "опыт" not in json.dumps(saved)
