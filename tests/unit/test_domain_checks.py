"""Детерминированные проверки артефакта: правила домена, применение и сборка."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core import domain_checks, domains

CHECKS = domains.load_domain("aurora-qt5").checks


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "src").mkdir()
    (root / "src" / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    return root


def _add(path: str, *lines: str) -> str:
    body = "\n".join(f"+{line}" for line in lines)
    return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}\n"


def _edit(path: str, added: str = "", removed: str = "") -> str:
    lines = [f"-{line}" for line in removed.splitlines()]
    lines += [f"+{line}" for line in (removed + ("\n" if removed else "") + added).splitlines()]
    return (
        f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,1 @@\n" + "\n".join(lines) + "\n"
    )


def test_forbidden_construction_is_caught_in_added_lines(tmp_path: Path):
    root = _repo(tmp_path)
    text = _add("src/window.cpp", "#include <QtWidgets/QApplication>")
    result = domain_checks.check_patch(text, root, CHECKS)
    assert not result.ok
    assert any("запрещённая конструкция" in issue.rule for issue in result.issues)
    assert result.issues[0].path == "src/window.cpp"


def test_removing_a_forbidden_construction_is_allowed(tmp_path: Path):
    """Удаление запрещённого — исправление, а не нарушение."""
    root = _repo(tmp_path)
    (root / "src" / "main.cpp").write_text(
        "#include <QtWidgets/QApplication>\nint main() { return 0; }\n", encoding="utf-8"
    )
    text = (
        "--- a/src/main.cpp\n+++ b/src/main.cpp\n@@ -1,2 +1,2 @@\n"
        "-#include <QtWidgets/QApplication>\n+#include <QObject>\n"
        " int main() { return 0; }\n"
    )
    assert domain_checks.check_patch(text, root, CHECKS).issues == ()


def test_allowed_qml_passes(tmp_path: Path):
    root = _repo(tmp_path)
    text = _add("src/main.qml", "import Sailfish.Silica 1.0", "ApplicationWindow { id: window }")
    assert domain_checks.check_patch(text, root, CHECKS).issues == ()


def test_new_spec_without_required_fields_is_caught(tmp_path: Path):
    root = _repo(tmp_path)
    text = _add("rpm/app.spec", "Name: app", "%build", "qmake")
    result = domain_checks.check_patch(text, root, CHECKS)
    assert not result.ok
    assert "описание пакета" in result.issues[0].rule
    assert "Version" in result.issues[0].detail or "%files" in result.issues[0].detail


def test_spec_that_removes_a_required_field_is_caught(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "rpm").mkdir()
    (root / "rpm" / "app.spec").write_text(
        "Name: app\nVersion: 1\n%files\n%{_bindir}/app\n", encoding="utf-8"
    )
    text = "--- a/rpm/app.spec\n+++ b/rpm/app.spec\n@@ -3,2 +3,1 @@\n-%files\n-%{_bindir}/app\n"
    result = domain_checks.check_patch(text, root, CHECKS)
    assert not result.ok
    assert "%files" in result.issues[0].detail


def test_paths_outside_the_repository_are_caught(tmp_path: Path):
    root = _repo(tmp_path)
    text = _add("../вне.cpp", "int x;")
    result = domain_checks.check_patch(text, root, CHECKS)
    assert not result.ok
    assert result.issues[0].rule == "путь вне репозитория"


def test_patch_that_does_not_apply_is_caught(tmp_path: Path):
    root = _repo(tmp_path)
    text = "--- a/src/main.cpp\n+++ b/src/main.cpp\n@@ -1 +1 @@\n-совсем другой текст\n+int x;\n"
    result = domain_checks.check_patch(text, root, CHECKS)
    assert not result.ok
    assert result.issues[0].rule == "применение"


def test_garbage_is_caught(tmp_path: Path):
    root = _repo(tmp_path)
    result = domain_checks.check_patch("не патч", root, CHECKS)
    assert not result.ok
    assert result.issues[0].rule == "разбор"


def test_build_success_is_part_of_the_result(tmp_path: Path):
    root = _repo(tmp_path)
    calls: list = []

    def build(command: str):
        calls.append(command)
        return True, f"выполнено: {command}"

    result = domain_checks.check_patch(_add("src/extra.cpp", "int x;"), root, CHECKS, build=build)
    assert result.ok
    assert calls == list(CHECKS.build_steps)
    assert "выполнено: qmake" in result.build_output


def test_build_failure_is_an_issue(tmp_path: Path):
    root = _repo(tmp_path)

    def build(command: str):
        return (command != "make"), "нет цели" if command == "make" else "ок"

    result = domain_checks.check_patch(_add("src/extra.cpp", "int x;"), root, CHECKS, build=build)
    assert not result.ok
    assert result.issues[0].rule == "сборка"
    assert "make" in result.issues[0].detail


def test_missing_builder_is_named_not_skipped_silently(tmp_path: Path):
    root = _repo(tmp_path)
    result = domain_checks.check_patch(_add("src/extra.cpp", "int x;"), root, CHECKS)
    assert result.ok, "отсутствие исполнителя не делает патч негодным"
    assert result.skipped and "недоступна" in result.skipped[0]
    assert any("недоступна" in line for line in result.lines())


def test_build_is_not_run_for_an_invalid_patch(tmp_path: Path):
    root = _repo(tmp_path)
    calls: list = []

    def build(command: str):
        calls.append(command)
        return True, "ок"

    result = domain_checks.check_patch(
        _add("src/bad.cpp", "#include <QtWidgets/QApplication>"), root, CHECKS, build=build
    )
    assert not result.ok
    assert calls == [], "сборка не запускается для заведомо негодного патча"
