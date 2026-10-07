"""Хранилище планировщика: задания, журнал прогонов, накопленные данные.

Файл общий для двух процессов, поэтому отдельно проверяется то, чем эта общность держится:
терпимое чтение, сдвиг срока после прогона, предел журнала при неограниченном накопленном и
курсор объявления (`cursor`/`fresh_runs`) — по нему приложение понимает, о чём ещё не сообщало.
"""

import json

from core import config, schedule_store
from core.schedule_store import ScheduleStore


def store(tmp_path):
    return ScheduleStore(path=tmp_path / "schedule.json")


def test_missing_file_reads_as_empty_scheduler(tmp_path):
    empty = store(tmp_path)
    assert empty.jobs() == ()
    assert empty.runs() == ()


def test_broken_file_reads_as_empty_scheduler(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text("{не json", encoding="utf-8")
    broken = ScheduleStore(path=path)
    assert broken.jobs() == ()
    assert broken.runs() == ()


def test_default_path_is_read_at_creation_time():
    """Путь состояния ленивый: значение по умолчанию не «замерзает» на импорте модуля."""
    assert ScheduleStore().path == config.SCHEDULE_FILE


def test_state_directory_is_created_on_the_first_write(tmp_path):
    """Каталога состояния может ещё не быть: первая запись его создаёт, а не падает."""
    path = tmp_path / "нет" / "такого" / "каталога" / "schedule.json"
    scheduler = ScheduleStore(path=path)

    scheduler.add_job(tool="index_refresh", arguments={}, every_minutes=5, next_run=0.0)

    assert path.is_file()


def test_added_job_gets_number_and_survives_reload(tmp_path):
    path = tmp_path / "schedule.json"
    first = ScheduleStore(path=path)
    job = first.add_job(
        tool="index_refresh", arguments={"strategy": "structural"}, every_minutes=5, next_run=100.0
    )
    assert job.number == 1
    assert job.runs == 0

    again = ScheduleStore(path=path)
    assert [item.tool for item in again.jobs()] == ["index_refresh"]
    assert again.jobs()[0].arguments == {"strategy": "structural"}
    assert again.jobs()[0].every_minutes == 5
    assert again.job(1).tool == "index_refresh"
    assert again.job(2) is None


def test_numbers_do_not_repeat(tmp_path):
    scheduler = store(tmp_path)
    scheduler.add_job(tool="index_refresh", arguments={}, every_minutes=5, next_run=0.0)
    second = scheduler.add_job(tool="public_api_scan", arguments={}, every_minutes=5, next_run=0.0)
    assert second.number == 2


def test_due_jobs_are_selected_by_their_time(tmp_path):
    scheduler = store(tmp_path)
    scheduler.add_job(tool="index_refresh", arguments={}, every_minutes=5, next_run=100.0)
    scheduler.add_job(tool="public_api_scan", arguments={}, every_minutes=5, next_run=500.0)

    assert [job.tool for job in scheduler.due_jobs(200.0)] == ["index_refresh"]
    assert [job.tool for job in scheduler.due_jobs(500.0)] == ["index_refresh", "public_api_scan"]


def test_recorded_run_moves_the_next_run_and_counts(tmp_path):
    scheduler = store(tmp_path)
    job = scheduler.add_job(tool="index_refresh", arguments={}, every_minutes=5, next_run=100.0)
    scheduler.record_run(job, at=100.0, ok=True, summary="собрано 3", collected=3, fresh=3)

    saved = scheduler.jobs()[0]
    assert saved.runs == 1
    assert saved.next_run == 100.0 + 5 * 60
    assert [run.summary for run in scheduler.runs()] == ["собрано 3"]
    assert scheduler.runs()[0].fresh == 3
    assert scheduler.last_run(job.number).summary == "собрано 3"


def test_failed_run_is_recorded_too(tmp_path):
    scheduler = store(tmp_path)
    job = scheduler.add_job(tool="target_build", arguments={}, every_minutes=5, next_run=100.0)
    scheduler.record_run(
        job, at=100.0, ok=False, summary="команда завершилась ошибкой", collected=0, fresh=0
    )

    run = scheduler.runs()[0]
    assert run.ok is False
    assert "ошибкой" in run.summary
    assert scheduler.jobs()[0].next_run == 100.0 + 5 * 60


def test_run_log_is_capped_but_collected_data_is_not(tmp_path):
    scheduler = store(tmp_path)
    job = scheduler.add_job(tool="public_api_scan", arguments={}, every_minutes=1, next_run=0.0)
    for index in range(schedule_store.MAX_RUN_LOG + 5):
        scheduler.record_run(
            job, at=float(index), ok=True, summary=f"прогон {index}", collected=1, fresh=0
        )
    scheduler.remember_collected(
        "нарушения", [f"запись-{index}" for index in range(schedule_store.MAX_RUN_LOG + 5)]
    )

    assert len(scheduler.runs()) == schedule_store.MAX_RUN_LOG
    assert scheduler.runs()[-1].summary == f"прогон {schedule_store.MAX_RUN_LOG + 4}"
    assert len(scheduler.collected("нарушения")) == schedule_store.MAX_RUN_LOG + 5


def test_collected_reports_only_names_seen_for_the_first_time(tmp_path):
    scheduler = store(tmp_path)
    fresh = scheduler.remember_collected("нарушения", ["первое", "второе"])
    assert fresh == ("первое", "второе")

    repeated = scheduler.remember_collected("нарушения", ["второе", "третье"])
    assert repeated == ("третье",)
    assert scheduler.collected("нарушения") == ("первое", "второе", "третье")
    assert scheduler.collected_total() == 3


def test_collected_is_kept_per_key(tmp_path):
    scheduler = store(tmp_path)
    scheduler.remember_collected("нарушения", ["лечение"])
    assert scheduler.remember_collected("другое", ["лечение"]) == ("лечение",)
    assert scheduler.collected_total() == 2


def test_cursor_and_fresh_runs_skip_what_is_already_announced(tmp_path):
    """Курсор — пара «время и сколько прогонов на нём»: один проход пишет их одним временем."""
    scheduler = store(tmp_path)
    first = scheduler.add_job(tool="index_refresh", arguments={}, every_minutes=5, next_run=0.0)
    second = scheduler.add_job(tool="public_api_scan", arguments={}, every_minutes=5, next_run=0.0)
    scheduler.record_run(first, at=100.0, ok=True, summary="первый", collected=0, fresh=0)
    scheduler.record_run(second, at=100.0, ok=True, summary="второй", collected=0, fresh=0)

    cursor = scheduler.cursor()
    assert cursor == (100.0, 2)

    scheduler.record_run(first, at=200.0, ok=True, summary="третий", collected=0, fresh=0)
    assert [run.summary for run in scheduler.fresh_runs(cursor)] == ["третий"]
    assert scheduler.fresh_runs(scheduler.cursor()) == ()


def test_fresh_runs_of_an_empty_log_is_empty(tmp_path):
    assert store(tmp_path).fresh_runs((0.0, 0)) == ()


def test_reload_picks_up_another_process_write(tmp_path):
    """Приложение читает тот же файл, что пишет серверный процесс: снимок обновляется перечитыванием."""
    path = tmp_path / "schedule.json"
    reader = ScheduleStore(path=path)
    writer = ScheduleStore(path=path)
    writer.add_job(tool="index_refresh", arguments={}, every_minutes=5, next_run=0.0)

    assert reader.jobs() == ()
    reader.reload()
    assert len(reader.jobs()) == 1


def test_file_is_written_as_json_envelope(tmp_path):
    path = tmp_path / "schedule.json"
    scheduler = ScheduleStore(path=path)
    job = scheduler.add_job(
        tool="public_api_scan", arguments={"modules": "QtCore"}, every_minutes=5, next_run=1.0
    )
    scheduler.record_run(job, at=1.0, ok=True, summary="итог", collected=1, fresh=1)
    scheduler.remember_collected("нарушения", ["файл:1"])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"jobs", "runs", "collected"}
    assert data["jobs"][0]["tool"] == "public_api_scan"
    assert data["collected"]["нарушения"] == ["файл:1"]
