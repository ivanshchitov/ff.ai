"""Целевой репозиторий: выбор корня, проверка git и границы файловых операций."""

from pathlib import Path

import pytest

from core.repo import (
    PathOutsideRepoError,
    RepoRootError,
    ensure_inside,
    find_git_root,
    relative_to_root,
    resolve_root,
)


def _git_repo(path: Path) -> Path:
    (path / ".git").mkdir(parents=True)
    return path


def test_explicit_argument_wins(tmp_path: Path):
    repo = _git_repo(tmp_path / "explicit")
    other = _git_repo(tmp_path / "other")
    assert resolve_root(explicit=str(repo), env=str(other), cwd=other) == repo


def test_environment_is_next(tmp_path: Path):
    repo = _git_repo(tmp_path / "from-env")
    assert resolve_root(env=str(repo), cwd=tmp_path) == repo


def test_current_directory_is_the_last_resort(tmp_path: Path, monkeypatch):
    repo = _git_repo(tmp_path / "cwd-repo")
    monkeypatch.chdir(repo)
    assert resolve_root() == repo


def test_subdirectory_finds_the_repository_root(tmp_path: Path):
    repo = _git_repo(tmp_path / "project")
    nested = repo / "src" / "qml"
    nested.mkdir(parents=True)
    assert resolve_root(explicit=str(nested)) == repo
    assert find_git_root(nested) == repo


def test_directory_without_git_is_an_error_naming_the_path(tmp_path: Path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    with pytest.raises(RepoRootError) as error:
        resolve_root(explicit=str(plain))
    assert str(plain) in str(error.value)
    assert "git" in str(error.value)


def test_missing_path_is_an_error(tmp_path: Path):
    with pytest.raises(RepoRootError) as error:
        resolve_root(explicit=str(tmp_path / "nope"))
    assert "не существует" in str(error.value)


def test_ensure_inside_accepts_paths_within_the_root(tmp_path: Path):
    repo = _git_repo(tmp_path / "repo")
    (repo / "src").mkdir()
    inside = ensure_inside("src/main.cpp", repo)
    assert inside == (repo / "src" / "main.cpp").resolve()
    assert ensure_inside(repo, repo) == repo.resolve()


def test_ensure_inside_rejects_paths_outside_the_root(tmp_path: Path):
    repo = _git_repo(tmp_path / "repo")
    for escaped in ("../secrets.txt", str(tmp_path / "secrets.txt"), "src/../../escape"):
        with pytest.raises(PathOutsideRepoError):
            ensure_inside(escaped, repo)


def test_ensure_inside_rejects_symlink_pointing_outside(tmp_path: Path):
    repo = _git_repo(tmp_path / "repo")
    outside = tmp_path / "outside.txt"
    outside.write_text("секрет", encoding="utf-8")
    link = repo / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(PathOutsideRepoError):
        ensure_inside(link, repo)


def test_relative_to_root_gives_the_citation_form(tmp_path: Path):
    repo = _git_repo(tmp_path / "repo")
    (repo / "src").mkdir()
    target = repo / "src" / "main.cpp"
    target.write_text("int main() {}\n", encoding="utf-8")
    assert relative_to_root(target, repo) == "src/main.cpp"
