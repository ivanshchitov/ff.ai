"""Патчи: разбор, границы путей, проверка и применение."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core import patches


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "notes.txt").write_text("первая строка\nвторая строка\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    return root


NEW_FILE = """--- /dev/null
+++ b/src/models.cpp
@@ -0,0 +1,3 @@
+#include <QObject>
+class ModelList {};
+// конец
"""

EDIT_FILE = """--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,3 @@
 первая строка
 вторая строка
+третья строка
"""


def test_parse_new_file_counts_added_lines():
    parsed = patches.parse(NEW_FILE)
    assert parsed.paths == ("src/models.cpp",)
    assert parsed.files[0].is_new is True
    assert parsed.files[0].added_lines == 3
    assert "class ModelList {};" in parsed.added_lines()


def test_parse_edit_separates_added_from_context():
    """Контекстные строки не считаются ни добавленными, ни удалёнными."""
    parsed = patches.parse(EDIT_FILE)
    assert parsed.files[0].path == "notes.txt"
    assert parsed.files[0].is_new is False
    assert parsed.files[0].added == ("третья строка",)
    assert parsed.files[0].removed == ()


def test_parse_edit_keeps_removed_lines():
    text = "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,1 @@\n-первая строка\n-вторая строка\n+первая строка\n"
    parsed = patches.parse(text)
    assert parsed.files[0].removed == ("первая строка", "вторая строка")
    assert parsed.files[0].added == ("первая строка",)


def test_parse_several_files():
    parsed = patches.parse(NEW_FILE + EDIT_FILE)
    assert parsed.paths == ("src/models.cpp", "notes.txt")
    assert "src/models.cpp (+3/-0)" in parsed.summary()


def test_empty_and_garbage_are_rejected():
    with pytest.raises(patches.PatchError):
        patches.parse("")
    with pytest.raises(patches.PatchError):
        patches.parse("это не патч, а текст")


def test_paths_outside_root_are_caught(tmp_path: Path):
    root = _repo(tmp_path)
    parsed = patches.parse(
        "--- /dev/null\n+++ b/../вне.txt\n@@ -0,0 +1 @@\n+плохо\n"
    )
    assert patches.outside_root(parsed, root) == ("../вне.txt",)


def test_absolute_path_is_outside(tmp_path: Path):
    root = _repo(tmp_path)
    parsed = patches.parse("--- /dev/null\n+++ /etc/passwd\n@@ -0,0 +1 @@\n+плохо\n")
    assert patches.outside_root(parsed, root) == ("/etc/passwd",)


def test_check_accepts_a_patch_and_apply_changes_the_file(tmp_path: Path):
    root = _repo(tmp_path)
    assert patches.check(root, EDIT_FILE) == ""
    assert patches.apply(root, EDIT_FILE) == ""
    assert "третья строка" in (root / "notes.txt").read_text(encoding="utf-8")
    assert patches.changed_paths(root) == ("notes.txt",)


def test_check_rejects_a_patch_that_does_not_fit(tmp_path: Path):
    root = _repo(tmp_path)
    stale = EDIT_FILE.replace("вторая строка", "другой текст")
    assert patches.check(root, stale) != ""
    assert patches.apply(root, stale) != ""
    assert "третья строка" not in (root / "notes.txt").read_text(encoding="utf-8")


def test_new_file_patch_applies_inside_the_repository(tmp_path: Path):
    root = _repo(tmp_path)
    assert patches.apply(root, NEW_FILE) == ""
    assert (root / "src" / "models.cpp").is_file()
