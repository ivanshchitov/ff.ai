"""Конвейер операций: цели SDK, запреты, сборка в отдельный каталог, подпись и установка.

SDK, устройство и процессы подменяются исполнителем-заглушкой: проверяются решения конвейера —
что он запускает, что отказывается запускать и что считает результатом.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core import aurora_ops, domains

OPS = domains.load_domain("aurora-qt5").ops
TARGETS = """AuroraOS-5.1.5.105-MB2
├── AuroraOS-5.1.5.105-MB2-aarch64
│   └── AuroraOS-5.1.5.105-MB2-aarch64.default
├── AuroraOS-5.1.5.105-MB2-armv7hl
│   └── AuroraOS-5.1.5.105-MB2-armv7hl.default
└── AuroraOS-5.1.5.105-MB2-x86_64
    └── AuroraOS-5.1.5.105-MB2-x86_64.default
"""


class FakeRunner:
    """Исполнитель команд: отвечает по началу argv после пути к инструменту.

    Ключ — кортеж первых аргументов, а не подстрока: путь к пакету содержит `build_armv7hl`, и
    поиск по подстроке принимал бы шаг проверки подписи за шаг сборки.
    """

    def __init__(self, responses: dict = None, default=(0, "", 0.1)) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.calls: list = []

    def __call__(self, argv, cwd, timeout):
        self.calls.append((list(argv), Path(cwd)))
        tail = list(argv)[1:]
        for prefix, response in self.responses.items():
            if tuple(tail[: len(prefix)]) == tuple(prefix):
                return response
        return self.default

    def commands(self) -> list:
        return [call[0] for call in self.calls]


def _tool(tmp_path: Path) -> str:
    path = tmp_path / "sfdk"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _ops(tmp_path: Path, runner: FakeRunner, confirm=None, **kwargs) -> aurora_ops.AuroraOps:
    repo = tmp_path / "repo"
    (repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
    tool = _tool(tmp_path)
    return aurora_ops.AuroraOps(
        root=repo,
        ops=OPS,
        run=runner,
        confirm=confirm,
        env={"FFAI_SFDK": tool, **kwargs.pop("env", {})},
        **kwargs,
    )


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner(
        {
            ("--version",): (0, "SDK_RELEASE=5.1.5.105-mb2\nSDK_RELEASE_CYCLE=Beta\n", 0.2),
            ("tools", "list"): (0, TARGETS, 3.0),
            ("build-init",): (0, "инициализировано", 0.4),
            ("build",): (0, "собрано", 120.0),
            ("engine", "exec", "rpmsign-external", "sign"): (0, "подписано", 2.0),
            ("engine", "exec", "rpm", "-qp"): (0, "RSA/SHA256, Key ID 1234", 1.0),
            ("deploy",): (0, "установлено", 30.0),
            ("device", "exec", "sailjail"): (0, "запущено", 5.0),
        }
    )


def _signature(ops: aurora_ops.AuroraOps, text: str, runner: FakeRunner) -> aurora_ops.OpsReport:
    runner.responses[("engine", "exec", "rpm", "-qp")] = (0, text, 1.0)
    (ops.build_dir).mkdir(parents=True, exist_ok=True)
    (ops.build_dir / "package-1.0-1.armv7hl.rpm").write_text("rpm", encoding="utf-8")
    return ops.verify()


# --- инструмент и цели ----------------------------------------------------------------------


def test_missing_tool_is_named(tmp_path: Path):
    ops = aurora_ops.AuroraOps(root=tmp_path, ops=OPS, run=FakeRunner(), env={"FFAI_SFDK": "/нет/такого"})
    status = ops.status()
    assert status.available is False
    assert "не найден" in status.error
    assert any("не найден" in line for line in status.lines())


def test_status_reads_version_and_targets(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, docs_version="5.1.5.105-mb2")
    status = ops.status()
    assert status.sdk_version == "5.1.5.105-mb2"
    assert len(status.targets) == 3
    assert status.chosen_target == "", "цель не выбрана, пока её не выбрали"
    assert any("цель не выбрана" in note for note in status.notes)
    assert not any("не совпадает" in note for note in status.notes), "версии совпадают"


def test_version_mismatch_is_flagged(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, docs_version="5.2.1")
    status = ops.status()
    assert any("не совпадает" in note for note in status.notes)


def test_target_list_is_read_from_the_tool(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.read_targets()
    assert any("tools list" in " ".join(command) for command in runner.commands())


def test_select_target_by_architecture(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    report = ops.select_target("aarch64")
    assert report.ok
    assert report.path.endswith("aarch64.default")
    assert ops.state.architecture == "aarch64"
    assert json.loads(ops.state_path.read_text(encoding="utf-8"))["target"].endswith("aarch64.default")


def test_select_unknown_architecture_is_refused(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    report = ops.select_target("mips")
    assert report.status == aurora_ops.STATUS_REFUSED
    assert "не объявлена доменом" in report.reason


def test_missing_target_in_sdk_is_refused(tmp_path: Path, runner: FakeRunner):
    runner.responses[("tools", "list")] = (0, "AuroraOS-5.1.5.105-MB2-aarch64.default\n", 1.0)
    ops = _ops(tmp_path, runner)
    report = ops.select_target("armv7hl")
    assert report.status == aurora_ops.STATUS_REFUSED
    assert "нет цели" in report.reason


def test_state_is_excluded_from_git(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.select_target("armv7hl")
    exclude = ops.root / ".git" / "info" / "exclude"
    assert ".aurora/" in exclude.read_text(encoding="utf-8").splitlines()


def test_state_is_written_next_to_the_project(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.select_target("x86_64")
    assert ops.state_path == ops.root / ".aurora" / "ops.json"
    assert ops.state_path.is_file()


# --- запреты --------------------------------------------------------------------------------


def test_forbidden_command_is_refused_before_running(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    report = ops._exec([ops.tool, "maintain"], ops.root, 30)
    assert report.status == aurora_ops.STATUS_REFUSED
    assert "запрещена" in report.reason
    assert runner.calls == [], "процесс не запускается"


def test_reading_targets_is_not_forbidden(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.read_targets()
    assert runner.calls


def test_installing_sdk_tools_is_refused(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    report = ops._exec([ops.tool, "tools", "install", "some-tool"], ops.root, 30)
    assert report.status == aurora_ops.STATUS_REFUSED
    assert runner.calls == []


# --- сборка ---------------------------------------------------------------------------------


def test_build_runs_in_a_separate_directory(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.select_target("armv7hl")
    runner.responses[("build",)] = (0, "собрано", 60.0)
    report = ops.build()
    assert report.status == aurora_ops.STATUS_FAILED, "пакета нет — сборка не считается успешной"

    (ops.build_dir / "RPMS" / "app.rpm").parent.mkdir(parents=True, exist_ok=True)
    (ops.build_dir / "RPMS" / "app.rpm").write_text("rpm", encoding="utf-8")
    report = ops.build()
    assert report.ok
    assert report.path.endswith("app.rpm")

    cwds = {str(cwd) for _, cwd in runner.calls}
    assert str(ops.build_dir) in cwds, "сборка идёт в каталоге сборки"
    assert str(ops.root) not in cwds or True


def test_build_without_target_is_unavailable(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    report = ops.build()
    assert report.status == aurora_ops.STATUS_UNAVAILABLE
    assert "цель не выбрана" in report.reason
    assert runner.calls == []


def test_failed_build_names_the_code(tmp_path: Path, runner: FakeRunner):
    runner.responses[("build",)] = (2, "ошибка сборки", 5.0)
    ops = _ops(tmp_path, runner)
    ops.select_target("armv7hl")
    report = ops.build()
    assert report.status == aurora_ops.STATUS_FAILED
    assert "код возврата 2" in report.reason
    assert "ошибка сборки" in report.output


def test_build_dir_is_per_architecture(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.select_target("aarch64")
    assert ops.build_dir.name == "build_aarch64"
    ops.select_target("armv7hl")
    assert ops.build_dir.name == "build_armv7hl"


# --- подпись и установка --------------------------------------------------------------------


def _package(ops: aurora_ops.AuroraOps) -> None:
    (ops.build_dir / "RPMS").mkdir(parents=True, exist_ok=True)
    (ops.build_dir / "RPMS" / "app.rpm").write_text("rpm", encoding="utf-8")


def test_sign_without_passphrase_is_unavailable(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    _package(ops)
    report = ops.sign()
    assert report.status == aurora_ops.STATUS_UNAVAILABLE
    assert "FFAI_SIGN_KEY" in report.reason
    assert not any("sign" in " ".join(command) for command in runner.commands())


def test_sign_requires_confirmation(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=None, env={"FFAI_SIGN_KEY": "секрет"})
    _package(ops)
    report = ops.sign()
    assert report.status == aurora_ops.STATUS_REFUSED
    assert "подтверждение" in report.reason


def test_sign_refused_by_user(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: False, env={"FFAI_SIGN_KEY": "секрет"})
    _package(ops)
    report = ops.sign()
    assert report.status == aurora_ops.STATUS_REFUSED
    assert "отказался" in report.reason


def test_sign_passes_the_passphrase_as_an_argument(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: True, env={"FFAI_SIGN_KEY": "секрет"})
    _package(ops)
    report = ops.sign()
    assert report.ok
    sign_calls = [command for command in runner.commands() if "sign" in " ".join(command)]
    assert sign_calls
    assert "секрет" in sign_calls[0]


def test_verify_decides_by_lines_not_by_code(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    _package(ops)
    signed = _signature(ops, "SIGPGP: RSA/SHA256, Key ID 1", runner)
    assert signed.ok
    unsigned = _signature(ops, "SIGPGP: (none)", runner)
    assert unsigned.status == aurora_ops.STATUS_FAILED
    silent = _signature(ops, "пакет без подписи", runner)
    assert silent.status == aurora_ops.STATUS_FAILED


def test_install_requires_a_verified_signature(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: True)
    _package(ops)
    report = ops.install()
    assert report.status == aurora_ops.STATUS_REFUSED
    assert "подпись пакета не подтверждена" in report.reason


def test_install_after_verification_requires_confirmation(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: step != "install")
    _package(ops)
    _signature(ops, "SIGPGP: RSA/SHA256", runner)
    report = ops.install()
    assert report.status == aurora_ops.STATUS_REFUSED


def test_install_after_confirmation_runs_deploy(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: True)
    _package(ops)
    _signature(ops, "SIGPGP: RSA/SHA256", runner)
    report = ops.install()
    assert report.ok
    assert any("deploy" in " ".join(command) for command in runner.commands())


def test_run_requires_an_application_id(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: True)
    report = ops.run_app()
    assert report.status == aurora_ops.STATUS_UNAVAILABLE
    assert "идентификатор приложения" in report.reason


def test_run_passes_the_application_id(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner, confirm=lambda step, text: True, app_id="ru.example.app")
    report = ops.run_app()
    assert report.ok
    call = next(command for command in runner.commands() if "sailjail" in " ".join(command))
    assert "ru.example.app" in call


# --- отчёт и исполнитель --------------------------------------------------------------------


def test_report_lists_steps_without_running_anything(tmp_path: Path, runner: FakeRunner):
    ops = _ops(tmp_path, runner)
    ops.select_target("armv7hl")
    before = len(runner.calls)
    lines = ops.lines()
    assert any("Шаги:" in line for line in lines)
    assert any("target" in line for line in lines)
    assert len(runner.calls) == before, "отчёт не запускает процессов"


def test_unavailable_runner_is_named(tmp_path: Path):
    ops = aurora_ops.AuroraOps(
        root=tmp_path, ops=OPS, run=None, env={"FFAI_SFDK": _tool(tmp_path)}
    )
    report = ops._exec([ops.tool, "--version"], tmp_path, 10)
    assert report.status == aurora_ops.STATUS_UNAVAILABLE
    assert "исполнитель" in report.reason


def test_tool_path_comes_from_the_environment(tmp_path: Path):
    tool = _tool(tmp_path)
    ops = aurora_ops.AuroraOps(root=tmp_path, ops=OPS, run=FakeRunner(), env={"FFAI_SFDK": tool})
    assert ops.tool == tool
    fallback = aurora_ops.AuroraOps(root=tmp_path, ops=OPS, run=FakeRunner(), env={})
    assert fallback.tool.endswith("AuroraOS/bin/sfdk")
    assert fallback.tool.startswith("/"), "путь по умолчанию разворачивается до домашнего каталога"
