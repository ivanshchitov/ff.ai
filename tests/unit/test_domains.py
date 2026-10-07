"""Домен как данные: загрузка пакета, проверка схемы, выбор активного домена."""

import json
from pathlib import Path

import pytest

from core import domains
from core.domains import (
    DomainError,
    DomainSchemaError,
    UnknownDomainError,
    load_domain,
    select_domain,
)


def _pack(
    root: Path,
    domain_id: str = "probe",
    *,
    schema: int = 1,
    with_title: bool = True,
    markers: list = None,
    invariants: list = None,
    with_prompts: bool = True,
    default: bool = False,
) -> Path:
    path = root / domain_id
    (path / "prompts").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": schema,
        "id": domain_id,
        "title": "Пробный домен" if with_title else None,
        "platform": "Платформа 1.0",
        "docs_version": "1.2.3",
        "default": default,
        "markers": markers if markers is not None else [{"glob": "*.probe"}],
    }
    if not with_title:
        del manifest["title"]
    (path / "domain.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (path / "invariants.json").write_text(
        json.dumps(
            {"invariants": invariants if invariants is not None else [
                {"number": 1, "rule": "Правило", "source": "документ", "forbidden": ["запрет"]}
            ]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if with_prompts:
        (path / "prompts" / "system.md").write_text("Роль домена.", encoding="utf-8")
        (path / "prompts" / "refusal.md").write_text("Текст отказа.", encoding="utf-8")
    return path


def test_bundled_domain_loads():
    domain = load_domain("aurora-qt5")
    assert domain.id == "aurora-qt5"
    assert domain.platform.startswith("Qt 5.6")
    assert domain.docs_version and domain.local_sdk_version
    assert len(domain.invariants) >= 10
    assert domain.prompt("system").startswith("# Роль")
    assert domain.prompt("refusal")


def test_bundled_domain_is_the_default_one():
    assert domains.default_domain_id() == "aurora-qt5"
    assert "aurora-qt5" in domains.available_domains()


def test_broken_json_names_the_file(tmp_path: Path):
    path = _pack(tmp_path)
    (path / "domain.json").write_text("{ это не json", encoding="utf-8")
    with pytest.raises(DomainSchemaError) as error:
        load_domain("probe", tmp_path)
    assert "domain.json" in str(error.value)
    assert "JSON" in str(error.value)


def test_unknown_schema_version_is_refused(tmp_path: Path):
    _pack(tmp_path, schema=2)
    with pytest.raises(DomainSchemaError) as error:
        load_domain("probe", tmp_path)
    assert "схемы" in str(error.value)


def test_missing_field_is_named(tmp_path: Path):
    _pack(tmp_path, with_title=False)
    with pytest.raises(DomainSchemaError) as error:
        load_domain("probe", tmp_path)
    assert "title" in str(error.value)


def test_missing_prompt_is_refused(tmp_path: Path):
    _pack(tmp_path, with_prompts=False)
    with pytest.raises(DomainSchemaError) as error:
        load_domain("probe", tmp_path)
    assert "промпт" in str(error.value)


def test_empty_invariants_are_refused(tmp_path: Path):
    _pack(tmp_path, invariants=[])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("probe", tmp_path)
    assert "invariants" in str(error.value)


def test_marker_matching_reads_the_file_content(tmp_path: Path):
    _pack(tmp_path, markers=[{"glob": "*.pro", "contains": "QT"}])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.pro").write_text("QT += quick\n", encoding="utf-8")
    detected = domains.detect_domain(repo, tmp_path)
    assert detected is not None
    domain, evidence = detected
    assert domain.id == "probe"
    assert evidence == ("app.pro",)


def test_marker_absent_means_no_detection(tmp_path: Path):
    _pack(tmp_path, markers=[{"glob": "*.absent"}])
    repo = tmp_path / "repo"
    repo.mkdir()
    assert domains.detect_domain(repo, tmp_path) is None


def test_selection_priority_argument_then_env_then_markers(tmp_path: Path):
    _pack(tmp_path, "first", markers=[{"glob": "*.none"}])
    _pack(tmp_path, "second", markers=[{"glob": "*.none"}])
    repo = tmp_path / "repo"
    repo.mkdir()

    explicit = select_domain(root=repo, explicit="second", env="first", domains_dir=tmp_path)
    assert explicit.domain.id == "second"
    assert explicit.source == "аргумент"

    from_env = select_domain(root=repo, env="first", domains_dir=tmp_path)
    assert from_env.domain.id == "first"
    assert from_env.source == "переменная окружения"


def test_selection_by_markers_reports_evidence(tmp_path: Path):
    _pack(tmp_path, "by-marker", markers=[{"glob": "*.probe"}])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "thing.probe").write_text("", encoding="utf-8")
    selection = select_domain(root=repo, domains_dir=tmp_path)
    assert selection.domain.id == "by-marker"
    assert selection.source == "маркеры репозитория"
    assert selection.evidence == ("thing.probe",)


def test_selection_falls_back_to_the_default_pack(tmp_path: Path):
    _pack(tmp_path, "fallback", markers=[{"glob": "*.absent"}], default=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    selection = select_domain(root=repo, domains_dir=tmp_path)
    assert selection.domain.id == "fallback"
    assert selection.source == "по умолчанию"


def test_unknown_domain_is_refused_with_the_list(tmp_path: Path):
    _pack(tmp_path, "known", markers=[{"glob": "*.none"}])
    with pytest.raises(UnknownDomainError) as error:
        select_domain(root=None, explicit="missing", domains_dir=tmp_path)
    assert "missing" in str(error.value)
    assert "known" in str(error.value)


def test_ambiguous_markers_are_refused(tmp_path: Path):
    _pack(tmp_path, "one", markers=[{"glob": "*.probe"}])
    _pack(tmp_path, "two", markers=[{"glob": "*.probe"}])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "thing.probe").write_text("", encoding="utf-8")
    with pytest.raises(DomainError):
        domains.detect_domain(repo, tmp_path)
