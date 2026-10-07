"""Планировщик и задания здоровья репозитория: расписание, прогоны, сканирование.

Процесс сервера здесь не поднимается: исполнитель вызова инжектируется, поэтому вся логика
расписания проверяется без протокола и без сети — тот же приём, что у редьюсеров панелей.
Задания проверяются на настоящем временном репозитории: их работа — чтение файлов и запуск
разрешённых команд, и подделывать её значило бы проверять не то, что выполняется в прогоне.
"""

from types import SimpleNamespace

import pytest

from core import config, code_index, schedule_store
from core.domains import load_domain
from core.schedule_store import ScheduleStore
from mcp_server import scheduler
from mcp_server.repo_tools import RepoContext, Tools, ToolsError
from mcp_server.scheduler import JOB_TOOL_NAMES, SCHEDULE_TOOL_NAMES, Scheduler


class FakeClock:
    def __init__(self, value: float = 1000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, minutes: float) -> None:
        self.value += minutes * 60


class FakeTools:
    """Исполнитель вызова: записывает обращения, отвечает заданным исходом."""

    def __init__(self, outcome=(True, "Проверка публичного API: файлов 4, нарушений 2 (новых 2).")):
        self.calls = []
        self.outcome = outcome
        self.failures = set()

    def __call__(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name in self.failures:
            return False, "Сборка цели не завершена: команда «make» — код возврата 2."
        return self.outcome


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tools():
    return FakeTools()


@pytest.fixture
def scheduler_fixture(tmp_path, clock, tools):
    return Scheduler(
        store=ScheduleStore(path=tmp_path / "schedule.json"),
        call_tool=tools,
        tool_names=JOB_TOOL_NAMES,
        now=clock,
    )


# --- постановка задания ---


def test_added_job_is_listed_with_its_number(scheduler_fixture):
    text = scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "every_minutes": 5})
    assert "1" in text
    assert "index_refresh" in scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})


def test_job_without_delay_is_due_at_once(scheduler_fixture, tools):
    scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "every_minutes": 5})
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    assert [name for name, _ in tools.calls] == ["index_refresh"]


def test_delayed_job_waits_for_its_first_run(scheduler_fixture, tools, clock):
    scheduler_fixture.call(
        scheduler.SCHEDULE_ADD,
        {"tool": "index_refresh", "every_minutes": 5, "start_in_minutes": 10},
    )
    assert "нечего" in scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    assert tools.calls == []

    clock.advance(10)
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    assert [name for name, _ in tools.calls] == ["index_refresh"]


def test_arguments_of_the_job_reach_the_tool(scheduler_fixture, tools):
    scheduler_fixture.call(
        scheduler.SCHEDULE_ADD,
        {"tool": "public_api_scan", "arguments": {"modules": "QtCore"}, "every_minutes": 5},
    )
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    assert tools.calls[0][1] == {"modules": "QtCore"}


def test_unknown_tool_is_rejected_without_creating_a_job(scheduler_fixture):
    text = scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "которого-нет", "every_minutes": 5})
    assert "index_refresh" in text
    assert "заданий нет" in scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})


def test_scheduler_tool_cannot_be_scheduled(scheduler_fixture):
    """Задание, зовущее сам планировщик, зациклило бы исполнитель — такой вызов отклоняется."""
    text = scheduler_fixture.call(
        scheduler.SCHEDULE_ADD, {"tool": scheduler.SCHEDULE_RUN_DUE, "every_minutes": 5}
    )
    assert "нельзя" in text.lower() or "недопустим" in text.lower()
    assert "заданий нет" in scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})


def test_period_out_of_range_is_rejected(scheduler_fixture):
    too_often = scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "every_minutes": 0})
    too_rare = scheduler_fixture.call(
        scheduler.SCHEDULE_ADD,
        {"tool": "index_refresh", "every_minutes": schedule_store.MAX_EVERY_MINUTES + 1},
    )
    assert str(schedule_store.MAX_EVERY_MINUTES) in too_often
    assert str(schedule_store.MIN_EVERY_MINUTES) in too_rare
    assert "заданий нет" in scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})


def test_delay_out_of_range_is_rejected(scheduler_fixture):
    text = scheduler_fixture.call(
        scheduler.SCHEDULE_ADD,
        {
            "tool": "index_refresh",
            "every_minutes": 5,
            "start_in_minutes": schedule_store.MAX_START_DELAY_MINUTES + 1,
        },
    )
    assert str(schedule_store.MAX_START_DELAY_MINUTES) in text
    assert "заданий нет" in scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})


def test_missing_tool_argument_is_rejected(scheduler_fixture):
    text = scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"every_minutes": 5})
    assert "tool" in text


# --- выполнение просроченных ---


def test_due_run_records_the_run_and_moves_the_next_run(scheduler_fixture, clock):
    scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "every_minutes": 5})
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})

    job = scheduler_fixture.store.jobs()[0]
    assert job.runs == 1
    assert job.next_run == clock.value + 5 * 60
    assert scheduler_fixture.store.runs()[0].ok is True


def test_nothing_is_due(scheduler_fixture, tools, clock):
    scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "every_minutes": 5})
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    tools.calls.clear()

    text = scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    assert "нечего" in text
    assert tools.calls == []


def test_failed_job_does_not_stop_the_others(scheduler_fixture, tools):
    scheduler_fixture.call(
        scheduler.SCHEDULE_ADD, {"tool": "target_build", "arguments": {}, "every_minutes": 5}
    )
    scheduler_fixture.call(
        scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "arguments": {}, "every_minutes": 5}
    )
    tools.failures.add("target_build")

    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})

    assert [name for name, _ in tools.calls] == ["target_build", "index_refresh"]
    runs = {run.number: run for run in scheduler_fixture.store.runs()}
    assert runs[1].ok is False
    assert "код возврата 2" in runs[1].summary
    assert runs[2].ok is True


def test_run_due_without_jobs_is_successful(scheduler_fixture):
    assert "нечего" in scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})


# --- агрегированный отчёт ---


def test_report_without_jobs(scheduler_fixture):
    assert "заданий нет" in scheduler_fixture.call(scheduler.SCHEDULE_SUMMARY, {})


def test_report_names_schedule_runs_and_collected(scheduler_fixture, clock):
    scheduler_fixture.call(
        scheduler.SCHEDULE_ADD, {"tool": "public_api_scan", "arguments": {}, "every_minutes": 5}
    )
    scheduler_fixture.store.remember_collected("нарушения", ["первое", "второе"])
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})

    text = scheduler_fixture.call(scheduler.SCHEDULE_SUMMARY, {})
    assert "public_api_scan" in text
    assert "5" in text
    assert "накоплено" in text.lower()
    assert "2" in text


def test_report_does_not_call_tools(scheduler_fixture, tools):
    scheduler_fixture.call(scheduler.SCHEDULE_ADD, {"tool": "index_refresh", "every_minutes": 5})
    tools.calls.clear()
    scheduler_fixture.call(scheduler.SCHEDULE_SUMMARY, {})
    scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})
    assert tools.calls == []


# --- объявление инструментов ---


def test_scheduler_declares_four_tools():
    assert set(SCHEDULE_TOOL_NAMES) == {
        scheduler.SCHEDULE_ADD,
        scheduler.SCHEDULE_LIST,
        scheduler.SCHEDULE_RUN_DUE,
        scheduler.SCHEDULE_SUMMARY,
    }


def test_three_jobs_are_declared():
    assert set(JOB_TOOL_NAMES) == {
        scheduler.TARGET_BUILD,
        scheduler.INDEX_REFRESH,
        scheduler.PUBLIC_API_SCAN,
    }


def test_unknown_scheduler_tool_is_reported(scheduler_fixture):
    assert "не объявлен" in scheduler_fixture.call("schedule_drop", {})


def test_arguments_may_arrive_as_a_json_string(scheduler_fixture, tools):
    """Ручной вызов /tool call передаёт значения строками, да и модели часто шлют JSON строкой."""
    scheduler_fixture.call(
        scheduler.SCHEDULE_ADD,
        {"tool": "public_api_scan", "arguments": '{"modules": "QtCore"}', "every_minutes": 5},
    )
    scheduler_fixture.call(scheduler.SCHEDULE_RUN_DUE, {})
    assert tools.calls[0][1] == {"modules": "QtCore"}


def test_unparsable_arguments_string_is_rejected(scheduler_fixture):
    text = scheduler_fixture.call(
        scheduler.SCHEDULE_ADD,
        {"tool": "public_api_scan", "arguments": "modules QtCore", "every_minutes": 5},
    )
    assert "arguments" in text
    assert "заданий нет" in scheduler_fixture.call(scheduler.SCHEDULE_LIST, {})


def test_call_result_reports_failure_flag(scheduler_fixture):
    """Сервер ставит признак ошибки по этому флагу: отказ отличим от данных."""
    ok, text = scheduler_fixture.call_result(scheduler.SCHEDULE_ADD, {"tool": ""})
    assert ok is False and text.startswith(scheduler.ARGUMENT_ERROR_PREFIX)
    ok, text = scheduler_fixture.call_result("schedule_drop", {})
    assert ok is False and "не объявлен" in text
    ok, _ = scheduler_fixture.call_result(scheduler.SCHEDULE_LIST, {})
    assert ok is True


# --- задания: репозиторий, правила, хранилище ------------------------------


@pytest.fixture
def repo(tmp_path):
    """Целевой репозиторий с одним исходником корпуса."""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.cpp").write_text(
        "int main() {}\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def context(repo, tmp_path):
    return RepoContext(
        root=repo,
        exports_dir=tmp_path / "reports",
        tools=Tools(run_allowed=("true", "false"), git_allowed=("log",)),
    )


@pytest.fixture
def job_store(tmp_path):
    return ScheduleStore(path=tmp_path / "schedule.json")


@pytest.fixture
def domain():
    return load_domain("aurora-qt5")


def test_run_job_rejects_an_unknown_name(context, domain, job_store):
    with pytest.raises(ToolsError) as error:
        scheduler.run_job("нет-такого", {}, context, domain, job_store)
    assert "index_refresh" in str(error.value)


def test_run_job_rejects_an_unknown_argument(context, domain, job_store):
    """Лишний параметр назван, а не проигнорирован: учтённый и пропущенный выглядят одинаково."""
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.INDEX_REFRESH, {"strategyy": "fixed"}, context, domain, job_store)
    assert "strategyy" in str(error.value)


# target_build


def test_target_build_runs_the_given_steps(context, domain, job_store):
    text = scheduler.run_job(scheduler.TARGET_BUILD, {"steps": "true; true"}, context, domain, job_store)
    assert "шагов 2" in text
    assert text.count("ок") == 2


def test_target_build_stops_at_the_first_failing_step(context, domain, job_store):
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.TARGET_BUILD, {"steps": "true; false"}, context, domain, job_store)
    assert "false" in str(error.value)
    assert "код возврата 1" in str(error.value)
    assert "- true: ок" in str(error.value)


def test_target_build_command_outside_the_whitelist_is_refused(context, domain, job_store):
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.TARGET_BUILD, {"steps": "rm -rf /"}, context, domain, job_store)
    assert "белый список" in str(error.value)


def test_target_build_takes_steps_from_the_domain(context, job_store):
    """Шаги — данные домена: задание без аргумента собирает то, что объявил домен."""
    domain = SimpleNamespace(checks=SimpleNamespace(build_steps=("true",)), invariants=())
    text = scheduler.run_job(scheduler.TARGET_BUILD, {}, context, domain, job_store)
    assert "шагов 1" in text


def test_target_build_without_steps_anywhere_names_the_reason(context, job_store):
    domain = SimpleNamespace(checks=SimpleNamespace(build_steps=()), invariants=())
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.TARGET_BUILD, {}, context, domain, job_store)
    assert "шаги сборки не заданы" in str(error.value)


# index_refresh


def test_index_refresh_builds_the_index_of_the_repository(context, domain, job_store):
    text = scheduler.run_job(scheduler.INDEX_REFRESH, {}, context, domain, job_store)

    assert "Индекс обновлён" in text
    assert code_index.index_exists(config.repo_index_file(context.root))


def test_index_refresh_rejects_an_unknown_strategy(context, domain, job_store):
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.INDEX_REFRESH, {"strategy": "по-строкам"}, context, domain, job_store)
    assert "structural" in str(error.value)


def test_index_refresh_without_a_corpus_names_the_reason(context, job_store):
    domain = SimpleNamespace(corpus=None, checks=None, invariants=())
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.INDEX_REFRESH, {}, context, domain, job_store)
    assert "корпус" in str(error.value)


# public_api_scan


def write_source(context, text: str) -> None:
    (context.root / "src" / "main.cpp").write_text(text, encoding="utf-8")


def test_scan_finds_forbidden_constructs_from_the_domain_rules(context, domain, job_store):
    write_source(context, "int main() {\nQML_ELEMENT\n}\n")

    text = scheduler.run_job(scheduler.PUBLIC_API_SCAN, {}, context, domain, job_store)

    assert "src/main.cpp:2" in text
    assert "новых 1" in text
    assert "Проверка модулей не выполнялась" in text  # список модулей не задан — это сказано


def test_scan_reports_only_new_violations_on_the_second_run(context, domain, job_store):
    """Накопленное живёт между прогонами: вчерашнее нарушение — не новость сегодня."""
    write_source(context, "QML_ELEMENT\n")

    first = scheduler.run_job(scheduler.PUBLIC_API_SCAN, {}, context, domain, job_store)
    second = scheduler.run_job(scheduler.PUBLIC_API_SCAN, {}, context, domain, job_store)

    assert "новых 1" in first
    assert "новых 0" in second
    assert len(job_store.collected(scheduler.SCAN_KEY)) == 1


def test_scan_flags_modules_outside_the_public_api(context, domain, job_store):
    write_source(context, '#include <QtCore/QObject>\n#include <QtQuick/QQuickItem>\n')

    text = scheduler.run_job(
        scheduler.PUBLIC_API_SCAN, {"modules": "QtQuick"}, context, domain, job_store
    )

    flagged = [line for line in text.splitlines() if "вне публичного API" in line]
    assert len(flagged) == 1
    assert "QtCore" in flagged[0]


def test_scan_accepts_a_module_named_by_qmake(context, domain, job_store):
    """`QT += core` и модуль `QtCore` — одно и то же: имя сверяется без регистра и префикса `Qt`."""
    write_source(context, "QT += core quick\n")

    text = scheduler.run_job(
        scheduler.PUBLIC_API_SCAN, {"modules": "QtCore,QtQuick"}, context, domain, job_store
    )

    assert "вне публичного API" not in text


def test_scan_takes_the_stems_of_the_domain_invariants(context, job_store):
    from core.domains import Invariant

    domain = SimpleNamespace(
        corpus=load_domain("aurora-qt5").corpus,
        checks=SimpleNamespace(forbidden=()),
        invariants=(Invariant(number=3, rule="Только Qt 5.6", forbidden=("ecm_feature_summary",), source="qt"),),
    )
    write_source(context, "find_package(ECM)\necm_feature_summary(...)\n")

    text = scheduler.run_job(scheduler.PUBLIC_API_SCAN, {}, context, domain, job_store)

    assert "инвариант 3" in text
    assert "ecm_feature_summary" in text


def test_scan_without_rules_names_the_reason(context, job_store):
    domain = SimpleNamespace(
        corpus=load_domain("aurora-qt5").corpus,
        checks=SimpleNamespace(forbidden=()),
        invariants=(),
    )
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(scheduler.PUBLIC_API_SCAN, {}, context, domain, job_store)
    assert "запрещённых конструкций" in str(error.value)


def test_scan_limit_is_named_in_the_report(context, domain, job_store):
    write_source(context, "QML_ELEMENT\nQML_ELEMENT\nQML_ELEMENT\n")

    text = scheduler.run_job(
        scheduler.PUBLIC_API_SCAN, {"max_results": 1}, context, domain, job_store
    )

    assert "предел max_results" in text
    assert len(job_store.collected(scheduler.SCAN_KEY)) == 1


def test_scan_rejects_a_limit_out_of_range(context, domain, job_store):
    with pytest.raises(ToolsError) as error:
        scheduler.run_job(
            scheduler.PUBLIC_API_SCAN, {"max_results": scheduler.MAX_SCAN_RESULTS + 1}, context, domain, job_store
        )
    assert "max_results" in str(error.value)


# --- правила пакета домена ---


def test_package_domain_is_read_next_to_the_whitelist_file():
    tools_path = config.DOMAINS_DIR / "aurora-qt5" / "tools.json"
    domain = scheduler.load_package_domain(tools_path)
    assert domain is not None
    assert domain.id == "aurora-qt5"


def test_package_domain_is_absent_without_a_package(tmp_path):
    assert scheduler.load_package_domain(None) is None
    assert scheduler.load_package_domain(tmp_path / "tools.json") is None


def test_package_domain_is_absent_for_a_broken_package(tmp_path):
    package = tmp_path / "broken"
    package.mkdir()
    (package / "domain.json").write_text("{ это не json", encoding="utf-8")
    assert scheduler.load_package_domain(package / "tools.json") is None
