"""Инструменты собственного сервера: границы путей, команд и выгрузки — без процессов.

Логика проверяется прямо на функциях (`repo_tools`), поэтому ни `mcp`, ни запуска сервера здесь
не нужно. Репозиторий — настоящий (`git init`): правила игнорирования и `git ls-files` — часть
контракта, и подделка `.git` проверяла бы не то, что работает в жизни. Отдельный случай с `.git`
без репозитория проверяет запасной обход дерева.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from core import domains
from mcp_server import repo_tools
from mcp_server.repo_tools import RepoContext, Tools, ToolsError

# Интерпретатор теста — разрешённая команда: он есть на любой машине и печатает что угодно.
PYTHON = f'"{sys.executable}"'


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Репозиторий с игнорируемыми файлами, подкаталогом и одним коммитом."""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "build").mkdir()
    (root / ".gitignore").write_text("build/\n*.log\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Проект\nвторая строка\nтретья строка\n", encoding="utf-8"
    )
    (root / "src" / "main.cpp").write_text(
        "int main() {\n  ensure_inside();\n}\n", encoding="utf-8"
    )
    (root / "src" / "notes.txt").write_text("Заметка про ensure_inside\n", encoding="utf-8")
    (root / "build" / "out.o").write_text("мусор", encoding="utf-8")
    (root / "debug.log").write_text("шум", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-qm",
        "первый коммит",
    )
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    secret = tmp_path / "secret.txt"
    secret.write_text("секрет\n", encoding="utf-8")
    return secret


@pytest.fixture
def ctx(repo: Path, tmp_path: Path) -> RepoContext:
    return RepoContext(
        root=repo,
        exports_dir=tmp_path / "reports",
        tools=Tools(
            run_allowed=("echo", "sleep", PYTHON),
            git_allowed=("status", "log", "diff"),
        ),
    )


def _python(command: str) -> str:
    """Разрешённая запуском команда: интерпретатор теста с аргументом."""
    return f"{PYTHON} -c {shlex.quote(command)}"


# --- Перечень файлов --------------------------------------------------------


def test_tree_lists_files_relative_to_root_and_respects_gitignore(ctx: RepoContext):
    lines = repo_tools.repo_tree(ctx).splitlines()
    assert lines[0] == "Файлы: 4 из 4 (pattern «**/*»)"
    assert [line.strip() for line in lines[1:]] == [
        ".gitignore",
        "README.md",
        "src/main.cpp",
        "src/notes.txt",
    ]


def test_tree_filters_by_pattern_and_marks_the_limit(ctx: RepoContext):
    only_cpp = repo_tools.repo_tree(ctx, "src/**/*.cpp")
    assert "src/main.cpp" in only_cpp
    assert "notes.txt" not in only_cpp

    limited = repo_tools.repo_tree(ctx, max_results=2)
    assert "Файлы: 2 из 4" in limited
    assert "показаны первые 2 из 4" in limited
    assert len([line for line in limited.splitlines() if line.startswith("  ")]) == 2


def test_tree_without_a_git_repository_walks_the_tree_excluding_dot_git(tmp_path: Path):
    plain = tmp_path / "plain"
    (plain / ".git").mkdir(parents=True)
    (plain / "sub").mkdir()
    (plain / "a.txt").write_text("x", encoding="utf-8")
    (plain / "sub" / "b.txt").write_text("y", encoding="utf-8")
    (plain / ".git" / "config").write_text("z", encoding="utf-8")

    context = RepoContext(root=plain, exports_dir=tmp_path / "reports", tools=Tools())
    text = repo_tools.repo_tree(context)
    assert "a.txt" in text and "sub/b.txt" in text
    assert ".git/" not in text


def test_invalid_limits_are_refused(ctx: RepoContext):
    with pytest.raises(ToolsError, match="max_results"):
        repo_tools.repo_tree(ctx, max_results=0)
    with pytest.raises(ToolsError, match="max_results"):
        repo_tools.repo_search(ctx, "ensure_inside", max_results=10_000)
    with pytest.raises(ToolsError, match="end_line"):
        repo_tools.repo_read(ctx, "README.md", 3, 1)


# --- Чтение -----------------------------------------------------------------


def test_read_numbers_the_requested_range(ctx: RepoContext):
    text = repo_tools.repo_read(ctx, "README.md", 2, 3)
    lines = text.splitlines()
    assert lines[0] == "README.md: строки 2–3 из 3"
    assert lines[1].endswith("вторая строка")
    assert lines[1].lstrip().startswith("2")
    assert lines[2].endswith("третья строка")
    assert "# Проект" not in text


def test_read_reports_an_empty_file(ctx: RepoContext):
    (ctx.root / "empty.txt").write_text("", encoding="utf-8")
    assert "файл пуст" in repo_tools.repo_read(ctx, "empty.txt")


def test_read_refuses_missing_directory_and_binary(ctx: RepoContext):
    (ctx.root / "data.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolsError, match="не найден"):
        repo_tools.repo_read(ctx, "нет-такого.txt")
    with pytest.raises(ToolsError, match="каталог"):
        repo_tools.repo_read(ctx, "src")
    with pytest.raises(ToolsError, match="двоичный"):
        repo_tools.repo_read(ctx, "data.bin")


def test_paths_outside_the_root_are_refused(ctx: RepoContext, outside: Path):
    link = ctx.root / "link.txt"
    link.symlink_to(outside)
    for escaped in ("../secret.txt", str(outside), "src/../../escape.txt", "link.txt"):
        with pytest.raises(ToolsError, match="за пределы целевого репозитория"):
            repo_tools.repo_read(ctx, escaped)
    with pytest.raises(ToolsError) as error:
        repo_tools.repo_read(ctx, "link.txt")
    assert "секрет" not in str(error.value)


# --- Поиск ------------------------------------------------------------------


def test_search_returns_file_line_and_text(ctx: RepoContext):
    text = repo_tools.repo_search(ctx, "ensure_inside")
    assert "найдено 2" in text
    assert "src/main.cpp:2: ensure_inside();" in text
    assert "src/notes.txt:1: Заметка про ensure_inside" in text


def test_search_ignores_case_and_honours_glob(ctx: RepoContext):
    text = repo_tools.repo_search(ctx, "ENSURE_INSIDE", glob="src/*.txt")
    assert "src/notes.txt" in text
    assert "main.cpp" not in text


def test_search_without_matches_says_so(ctx: RepoContext):
    text = repo_tools.repo_search(ctx, "нет-такого-текста-в-репозитории")
    assert "ничего не найдено" in text


def test_search_marks_the_limit_and_skips_binary_files(ctx: RepoContext):
    limited = repo_tools.repo_search(ctx, "ensure_inside", max_results=1)
    assert "найдено 1" in limited
    assert "предел max_results = 1" in limited

    (ctx.root / "data.bin").write_bytes(b"\x00ensure_inside")
    assert "data.bin" not in repo_tools.repo_search(ctx, "ensure_inside")


# --- Разрешённые команды ----------------------------------------------------


def test_allowed_command_runs_and_returns_code_and_output(ctx: RepoContext):
    text = repo_tools.repo_run(ctx, "echo привет")
    lines = text.splitlines()
    assert lines[0] == "$ echo привет"
    assert lines[1] == "код возврата: 0"
    assert "привет" in lines[2]


def test_nonzero_code_is_a_result_not_a_refusal(ctx: RepoContext):
    text = repo_tools.repo_run(ctx, _python("import sys; sys.exit(3)"))
    assert "код возврата: 3" in text


def test_command_outside_the_whitelist_is_refused_with_the_list(ctx: RepoContext):
    with pytest.raises(ToolsError) as error:
        repo_tools.repo_run(ctx, "rm -rf /")
    message = str(error.value)
    assert "не входит в белый список" in message
    assert "echo, sleep" in message


def test_whitelist_prefix_does_not_open_another_command(ctx: RepoContext):
    with pytest.raises(ToolsError):
        repo_tools.repo_run(ctx, "echoa привет")


def test_whitelist_prefix_boundaries():
    allowed = ("make", "sfdk config target=", "rpmsign-external")
    assert repo_tools._allowed_prefix("make -j4", allowed) == "make"
    assert repo_tools._allowed_prefix("make", allowed) == "make"
    assert repo_tools._allowed_prefix("makeevil", allowed) is None
    assert (
        repo_tools._allowed_prefix("sfdk config target=AuroraOS-x", allowed)
        == "sfdk config target="
    )
    assert repo_tools._allowed_prefix("sfdk config other=1", allowed) is None
    assert repo_tools._allowed_prefix("rpmsign-external --file x", allowed) is not None
    assert repo_tools._allowed_prefix("rpmsign-externally", allowed) is None


def test_command_runs_without_a_shell(ctx: RepoContext, tmp_path: Path):
    marker = tmp_path / "pwned"
    text = repo_tools.repo_run(ctx, f"echo a; touch {marker}")
    assert not marker.exists()
    assert "a; touch" in text


def test_long_output_is_cut_with_a_mark(ctx: RepoContext):
    size = repo_tools.MAX_OUTPUT_CHARS + 100
    text = repo_tools.repo_run(ctx, f"echo {'x' * size}")
    assert "вывод обрезан" in text
    assert f"из {size}" in text
    assert len(text) < size + 500


def test_timeout_is_refused(ctx: RepoContext, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(repo_tools, "COMMAND_TIMEOUT_SECONDS", 0.3)
    with pytest.raises(ToolsError, match="не завершилась за"):
        repo_tools.repo_run(ctx, "sleep 5")


def test_secrets_are_not_passed_to_the_command(
    ctx: RepoContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FFAI_TEST_TOKEN", "секрет")
    text = repo_tools.repo_run(
        ctx, _python("import os; print(os.environ.get('FFAI_TEST_TOKEN', 'нет'))")
    )
    assert "нет" in text
    assert "секрет" not in text


def test_command_with_an_empty_whitelist_explains_itself(repo: Path, tmp_path: Path):
    context = RepoContext(root=repo, exports_dir=tmp_path / "reports", tools=Tools())
    with pytest.raises(ToolsError, match="без файла белых списков"):
        repo_tools.repo_run(context, "make")
    with pytest.raises(ToolsError, match="без файла белых списков"):
        repo_tools.repo_git(context, "log")


# --- Git --------------------------------------------------------------------


def test_git_read_subcommand_returns_its_output(ctx: RepoContext):
    text = repo_tools.repo_git(ctx, "log --oneline -1")
    assert "код возврата: 0" in text
    assert "первый коммит" in text


@pytest.mark.parametrize("args", ["commit -m x", "push", "reset --hard", "checkout -b new"])
def test_git_state_changing_subcommand_is_refused(ctx: RepoContext, args: str):
    with pytest.raises(ToolsError, match="подкоманда git"):
        repo_tools.repo_git(ctx, args)


def test_git_argument_that_writes_a_file_is_refused(ctx: RepoContext):
    with pytest.raises(ToolsError, match="запрещён"):
        repo_tools.repo_git(ctx, "log --output=out.txt")
    with pytest.raises(ToolsError, match="запрещён"):
        repo_tools.repo_git(ctx, "diff --no-index a b")


# --- Выгрузка ---------------------------------------------------------------


def test_save_writes_to_the_exports_dir_and_never_overwrites(ctx: RepoContext):
    first = repo_tools.repo_save(ctx, "report.md", "итог")
    assert str(ctx.exports_dir / "report.md") in first
    assert (ctx.exports_dir / "report.md").read_text(encoding="utf-8") == "итог"

    repo_tools.repo_save(ctx, "report.md", "другой")
    assert (ctx.exports_dir / "report-2.md").read_text(encoding="utf-8") == "другой"
    assert (ctx.exports_dir / "report.md").read_text(encoding="utf-8") == "итог"
    assert not (ctx.root / "report.md").exists()


def test_save_refuses_unsafe_names_and_empty_text(ctx: RepoContext):
    for name in ("../escape.md", "sub/файл.md", "..", "", "выгрузка.md"):
        with pytest.raises(ToolsError, match="имя"):
            repo_tools.repo_save(ctx, name, "текст")
    for text in ("", "   \n"):
        with pytest.raises(ToolsError, match="пустой текст"):
            repo_tools.repo_save(ctx, "ok.md", text)
    assert not ctx.exports_dir.exists()


def test_exports_dir_inside_the_repository_is_refused(repo: Path):
    context = RepoContext(root=repo, exports_dir=repo / "reports", tools=Tools())
    with pytest.raises(ToolsError, match="внутри целевого репозитория"):
        repo_tools.repo_save(context, "report.md", "текст")
    assert not (repo / "reports").exists()


# --- Белые списки -----------------------------------------------------------


def test_tools_load_reads_both_lists(tmp_path: Path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps({"run": {"allowed": ["make", "  mb2  "]}, "git": {"allowed": ["log"]}}),
        encoding="utf-8",
    )
    tools = Tools.load(path)
    assert tools.run_allowed == ("make", "mb2")
    assert tools.git_allowed == ("log",)


def test_tools_without_sections_allow_nothing(tmp_path: Path):
    path = tmp_path / "tools.json"
    path.write_text("{}", encoding="utf-8")
    assert Tools.load(path) == Tools()


def test_broken_tools_file_names_the_file(tmp_path: Path):
    missing = tmp_path / "нет.json"
    with pytest.raises(ToolsError) as error:
        Tools.load(missing)
    assert str(missing) in str(error.value)

    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")
    with pytest.raises(ToolsError) as error:
        Tools.load(broken)
    assert str(broken) in str(error.value)

    shape = tmp_path / "shape.json"
    shape.write_text(json.dumps({"run": {"allowed": "make"}}), encoding="utf-8")
    with pytest.raises(ToolsError) as error:
        Tools.load(shape)
    assert str(shape) in str(error.value)
    assert "allowed" in str(error.value)


def test_domain_package_supplies_the_whitelists():
    """Белые списки — файл пакета домена, и этот файл читается тем же загрузчиком."""
    domain = domains.load_domain(domains.default_domain_id())
    assert domain.tools_path is not None
    tools = Tools.load(domain.tools_path)
    assert tools.run_allowed
    assert tools.git_allowed
