"""Сервер репозитория как отдельный процесс: инструменты, вызов, ошибка и запуск без корня.

Тест поднимает настоящий `repo_server.py` через `mcp.client.client.Client` и говорит с ним по
stdio — это единственная проверка, что процесс собирается, объявляет инструменты и умеет отдавать
ошибку инструмента пометкой, а не падением. Логика самих инструментов проверяется без процессов
(`test_repo_tools.py`); здесь важен транспорт.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="сервер и клиент говорят на MCP: нужен пакет mcp (Python 3.10+)")

from mcp.client.client import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

SERVER = Path(__file__).resolve().parents[2] / "mcp_server" / "repo_server.py"
TOOLS = (
    "repo_tree",
    "repo_read",
    "repo_search",
    "repo_git",
    "repo_run",
    "repo_save",
    # Задания здоровья репозитория и планировщик объявлены тем же сервером — порядок объявления
    # и есть порядок этого списка.
    "target_build",
    "index_refresh",
    "public_api_scan",
    "schedule_add",
    "schedule_list",
    "schedule_run_due",
    "schedule_summary",
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# Проект\nвторая строка\n", encoding="utf-8")
    (root / "src" / "main.cpp").write_text("int main() {}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    return root


@pytest.fixture
def tools_file(tmp_path: Path) -> Path:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps({"run": {"allowed": ["echo"]}, "git": {"allowed": ["log", "status"]}}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def exports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"


def _parameters(
    root: Path, tools_file: Path | None, exports_dir: Path
) -> StdioServerParameters:
    args = [str(SERVER), "--root", str(root)]
    if tools_file is not None:
        args += ["--tools", str(tools_file)]
    return StdioServerParameters(
        command=sys.executable, args=args, env={"FFAI_EXPORTS_DIR": str(exports_dir)}
    )


def test_server_lists_its_tools(repo: Path, tools_file: Path, exports_dir: Path):
    async def scenario():
        async with Client(_parameters(repo, tools_file, exports_dir)) as client:
            return (await client.list_tools()).tools

    tools = asyncio.run(scenario())
    assert [tool.name for tool in tools] == list(TOOLS)
    assert all(tool.description for tool in tools)
    assert all(tool.input_schema.get("type") == "object" for tool in tools)
    read = next(tool for tool in tools if tool.name == "repo_read")
    assert "path" in read.input_schema["required"]


def test_read_call_returns_numbered_lines(repo: Path, tools_file: Path, exports_dir: Path):
    async def scenario():
        async with Client(_parameters(repo, tools_file, exports_dir)) as client:
            return await client.call_tool(
                "repo_read", {"path": "README.md", "start_line": 2, "end_line": 2}
            )

    result = asyncio.run(scenario())
    assert result.is_error is False
    text = result.content[0].text
    assert "README.md: строки 2–2 из 2" in text
    assert "вторая строка" in text
    assert "# Проект" not in text


def test_tool_error_is_marked_and_the_server_keeps_working(
    repo: Path, tools_file: Path, exports_dir: Path
):
    async def scenario():
        async with Client(_parameters(repo, tools_file, exports_dir)) as client:
            refused = await client.call_tool("repo_read", {"path": "../secret.txt"})
            unknown = await client.call_tool("repo_write", {"path": "README.md"})
            ok = await client.call_tool("repo_tree", {})
            return refused, unknown, ok

    refused, unknown, ok = asyncio.run(scenario())
    assert refused.is_error is True
    assert "за пределы целевого репозитория" in refused.content[0].text
    assert unknown.is_error is True
    assert ok.is_error is False
    assert "README.md" in ok.content[0].text


def test_command_outside_the_whitelist_is_an_error_result(
    repo: Path, tools_file: Path, exports_dir: Path
):
    async def scenario():
        async with Client(_parameters(repo, tools_file, exports_dir)) as client:
            return await client.call_tool("repo_run", {"command": "rm -rf /"})

    result = asyncio.run(scenario())
    assert result.is_error is True
    assert "не входит в белый список" in result.content[0].text
    assert "echo" in result.content[0].text


def test_job_tool_refusal_is_marked_through_the_protocol(
    repo: Path, tools_file: Path, exports_dir: Path
):
    """Задание идёт тем же путём, что ручной вызов: отказ белого списка — ошибочный результат.

    Планировщик выполняет задание внутри процесса, поэтому признак ошибки для задания — то же,
    чем он отличает отказ от данных в журнале прогонов.
    """
    async def scenario():
        async with Client(_parameters(repo, tools_file, exports_dir)) as client:
            return await client.call_tool("target_build", {"steps": "rm -rf /"})

    result = asyncio.run(scenario())
    assert result.is_error is True
    assert "белый список" in result.content[0].text


def test_broken_whitelist_file_is_an_error_not_a_crash(
    repo: Path, tmp_path: Path, exports_dir: Path
):
    """Сломанный пакет домена называет себя в инструментах команд; чтение продолжает работать."""
    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")

    async def scenario():
        async with Client(_parameters(repo, broken, exports_dir)) as client:
            command = await client.call_tool("repo_run", {"command": "echo ok"})
            read = await client.call_tool("repo_read", {"path": "README.md"})
            return command, read

    command, read = asyncio.run(scenario())
    assert command.is_error is True
    assert "файл белых списков" in command.content[0].text
    assert str(broken) in command.content[0].text
    assert read.is_error is False
    assert "# Проект" in read.content[0].text


def test_save_writes_outside_the_repository(
    repo: Path, tools_file: Path, exports_dir: Path
):
    async def scenario():
        async with Client(_parameters(repo, tools_file, exports_dir)) as client:
            return await client.call_tool("repo_save", {"name": "report.md", "text": "итог"})

    result = asyncio.run(scenario())
    assert result.is_error is False
    assert (exports_dir / "report.md").read_text(encoding="utf-8") == "итог"
    assert not (repo / "report.md").exists()


def test_start_without_a_root_reports_the_missing_argument():
    done = subprocess.run(
        [sys.executable, str(SERVER)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode != 0
    assert "--root" in done.stderr
    assert "корень" in done.stderr


def test_start_with_a_missing_root_names_the_path(tmp_path: Path):
    missing = tmp_path / "nope"
    done = subprocess.run(
        [sys.executable, str(SERVER), "--root", str(missing)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode != 0
    assert str(missing) in done.stderr
