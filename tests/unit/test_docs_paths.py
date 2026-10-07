"""Сторож путей в документации: полных путей конкретной машины там быть не должно.

Правило простое и потому проверяемое: в документах — относительные пути и шаблоны вида
`/path/to/...`. Абсолютный путь с домашним каталогом автора попадает в документ один раз,
а живёт годами: читатель не может его повторить, а автор считает, что «и так понятно».
"""

import re
from pathlib import Path

import pytest

from core import config

DOCUMENTS = (
    "README.md",
    "AGENTS.md",
)

FORBIDDEN = re.compile(
    r"(?:/Users/|/home/|/private/|/Volumes/|~/|/var/folders/)"
)
TEMPLATE = re.compile(r"/path/to/")


def _documented_files():
    for name in DOCUMENTS:
        yield config.BASE_DIR / name
    for directory in ("docs", "openspec"):
        for path in sorted((config.BASE_DIR / directory).rglob("*.md")):
            yield path
    yield config.BASE_DIR / "openspec" / "config.yaml"


@pytest.mark.parametrize("path", list(_documented_files()), ids=lambda p: str(p).split("/")[-1])
def test_documents_have_no_machine_specific_paths(path: Path):
    text = path.read_text(encoding="utf-8")
    for match in FORBIDDEN.finditer(text):
        line = text[: match.start()].count("\n") + 1
        fragment = text[match.start() : match.start() + 60].split("\n")[0]
        pytest.fail(f"{path.name}:{line}: полный путь машины в документе — {fragment!r}")


def test_template_paths_stay_recognisable():
    """Шаблоны одного вида: если в документе есть путь, он должен быть `/path/to/...`."""
    for path in _documented_files():
        for match in TEMPLATE.finditer(path.read_text(encoding="utf-8")):
            assert match.group(0) == "/path/to/"


def test_documents_are_discoverable():
    paths = list(_documented_files())
    assert len(paths) >= 6, "документация должна быть найдена, а не замолчать"
