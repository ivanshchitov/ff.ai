"""Ядро и интерфейс не должны знать ни одного слова из пакета домена.

Это тот же приём, что у донора («в `core/` и `ui/` нет имён инструментов MCP-сервера»): он
превращает «домен — данные» из намерения в проверяемое свойство. Если тест покраснел — значит
в ядро просочилась предметная область, и добавлять домены станет дороже.
"""

from pathlib import Path

import pytest

from core import config, domains

PACKAGES = ("core", "ui")


def _sources():
    for package in PACKAGES:
        for path in (config.BASE_DIR / package).glob("*.py"):
            yield path, path.read_text(encoding="utf-8").lower()


def test_core_and_ui_do_not_mention_domain_identifiers():
    domain = domains.load_domain("aurora-qt5")
    identifiers = domains.identifiers_of(domain)
    assert identifiers, "у домена должны быть опознаваемые слова"
    for path, text in _sources():
        for word in identifiers:
            assert word not in text, f"{path.name}: в ядре встречается слово домена «{word}»"


def test_core_does_not_import_the_terminal_library():
    for path, text in _sources():
        if path.parent.name != "core":
            continue
        assert "import rich" not in text, f"{path.name}: ядро импортирует rich"
        assert "from rich" not in text, f"{path.name}: ядро импортирует rich"


def test_core_does_not_read_stdin_or_print():
    for path, text in _sources():
        if path.parent.name != "core":
            continue
        assert "sys.stdin" not in text, f"{path.name}: ядро читает стандартный ввод"
        assert "print(" not in text, f"{path.name}: ядро печатает в стандартный вывод"


def test_domain_identifier_helper_includes_platform_and_stems():
    domain = domains.load_domain("aurora-qt5")
    identifiers = domains.identifiers_of(domain)
    assert domain.id in identifiers
    assert "qtwidgets" in identifiers
    assert all(len(word) >= 4 for word in identifiers)


@pytest.mark.parametrize("package", PACKAGES)
def test_packages_exist(package: str):
    assert (config.BASE_DIR / package).is_dir()
