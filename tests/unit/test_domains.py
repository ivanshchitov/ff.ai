"""Домен как данные: загрузка пакета, проверка схемы, выбор активного домена."""

import json
import re
from pathlib import Path

import pytest

from core import domains, memory_layers
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


def test_bundled_domain_declares_the_portal_server():
    """Сервер документации портала — данные пакета, а не код: так его и проверяем."""
    domain = load_domain("aurora-qt5")
    assert len(domain.servers) == 1
    server = domain.servers[0]
    assert server.transport == "http"
    assert server.url.startswith("https://")
    assert server.health_tool, "у сервера должен быть объявлен инструмент проверки"
    assert server.source == "домен"


def test_bundled_domain_points_at_its_tools_file():
    domain = load_domain("aurora-qt5")
    assert domain.tools_path is not None
    assert domain.tools_path.name == "tools.json"
    data = json.loads(domain.tools_path.read_text(encoding="utf-8"))
    assert data["run"]["allowed"], "белый список команд сборки не должен быть пустым"
    assert data["git"]["allowed"], "белый список git-подкоманд не должен быть пустым"


def test_pack_without_servers_is_not_an_error(tmp_path: Path):
    path = _pack(tmp_path, "local-only")
    domain = load_domain("local-only", tmp_path)
    assert domain.servers == ()
    assert domain.tools_path is None
    assert path.is_dir()


def test_broken_servers_section_names_the_file(tmp_path: Path):
    path = _pack(tmp_path, "broken")
    (path / "servers.json").write_text('{"servers": [{"transport": "http"}]}', encoding="utf-8")
    with pytest.raises(DomainSchemaError) as error:
        load_domain("broken", tmp_path)
    assert "servers.json" in str(error.value)


def test_servers_section_must_be_a_list(tmp_path: Path):
    path = _pack(tmp_path, "wrong-shape")
    (path / "servers.json").write_text('{"servers": {}}', encoding="utf-8")
    with pytest.raises(DomainSchemaError) as error:
        load_domain("wrong-shape", tmp_path)
    assert "servers" in str(error.value)


def test_unknown_transport_in_pack_is_refused(tmp_path: Path):
    path = _pack(tmp_path, "bad-transport")
    (path / "servers.json").write_text(
        '{"servers": [{"name": "s", "transport": "smtp"}]}', encoding="utf-8"
    )
    with pytest.raises(DomainSchemaError) as error:
        load_domain("bad-transport", tmp_path)
    assert "транспорт" in str(error.value)
# --- разделы памяти и профиля (P8) ---


def _memory_section(path: Path, rules: list) -> None:
    (path / "memory.json").write_text(
        json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8"
    )


def _profile_section(path: Path, **fields) -> None:
    data = {"name_label": "Имя", "name_question": "Как вас зовут?", "name_prefix": "профиль"}
    data.update(fields)
    (path / "profile.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _rule(**overrides) -> dict:
    rule = {
        "layer": "long_term",
        "category": "решения",
        "key": "решения",
        "pattern": r"\bзапомни\b",
        "description": "просьба запомнить",
    }
    rule.update(overrides)
    return rule


def _section(**overrides) -> dict:
    section = {"id": "style", "label": "Стиль", "question": "Как отвечать?", "default": ""}
    section.update(overrides)
    return section


def test_bundled_domain_declares_memory_rules():
    """Правила маршрутизации памяти — данные пакета: ядро не знает ни одного образца."""
    domain = load_domain("aurora-qt5")
    assert domain.memory is not None
    rules = domain.memory.rules
    assert rules, "у домена должны быть правила памяти"
    for rule in rules:
        assert rule.layer in memory_layers.STORABLE_LAYERS
        assert rule.category and rule.key and rule.description
        assert re.compile(rule.pattern), "образец правила должен компилироваться"


def test_bundled_domain_declares_profile_sections():
    """Разделы профиля — тоже данные: их подписи и вопросы видит опросник."""
    domain = load_domain("aurora-qt5")
    assert domain.profile is not None
    profile = domain.profile
    assert profile.name_question and profile.name_prefix
    assert profile.sections
    assert len({section.id for section in profile.sections}) == len(profile.sections)
    for section in profile.sections:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", section.id)
        assert section.label and section.question


def test_pack_without_memory_and_profile_is_not_an_error(tmp_path: Path):
    """Разделы необязательны: пакет без них — это домен без памяти и персонализации."""
    _pack(tmp_path, "bare")
    domain = load_domain("bare", tmp_path)
    assert domain.memory is None
    assert domain.profile is None


def test_broken_memory_section_names_the_file(tmp_path: Path):
    path = _pack(tmp_path, "bad-memory")
    _memory_section(path, [_rule(pattern="")])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("bad-memory", tmp_path)
    assert "memory.json" in str(error.value)
    assert "pattern" in str(error.value)


def test_empty_memory_rules_are_refused(tmp_path: Path):
    path = _pack(tmp_path, "empty-memory")
    _memory_section(path, [])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("empty-memory", tmp_path)
    assert "rules" in str(error.value)


def test_unknown_memory_layer_is_refused(tmp_path: Path):
    """Слой вне модели памяти — опечатка пакета, а не запись, которую потом никто не найдёт."""
    path = _pack(tmp_path, "bad-layer")
    _memory_section(path, [_rule(layer="short_term")])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("bad-layer", tmp_path)
    assert "слой" in str(error.value)


def test_uncompilable_memory_pattern_is_refused(tmp_path: Path):
    path = _pack(tmp_path, "bad-pattern")
    _memory_section(path, [_rule(pattern="(")])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("bad-pattern", tmp_path)
    assert "компилируется" in str(error.value)


def test_broken_profile_section_names_the_file(tmp_path: Path):
    path = _pack(tmp_path, "bad-profile")
    _profile_section(path, sections=[_section(question="")])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("bad-profile", tmp_path)
    assert "profile.json" in str(error.value)
    assert "question" in str(error.value)


def test_profile_section_id_must_be_a_latin_name(tmp_path: Path):
    """Имя раздела — машинное: по нему хранится значение, и кириллица в нём недопустима."""
    path = _pack(tmp_path, "russian-id")
    _profile_section(path, sections=[_section(id="Стиль")])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("russian-id", tmp_path)
    assert "id" in str(error.value)


def test_double_profile_section_id_is_refused(tmp_path: Path):
    path = _pack(tmp_path, "double-id")
    _profile_section(path, sections=[_section(), _section()])
    with pytest.raises(DomainSchemaError) as error:
        load_domain("double-id", tmp_path)
    assert "дважды" in str(error.value)


def test_profile_without_the_name_question_is_refused(tmp_path: Path):
    path = _pack(tmp_path, "no-name-question")
    _profile_section(path, sections=[_section()])
    data = json.loads((path / "profile.json").read_text(encoding="utf-8"))
    del data["name_question"]
    (path / "profile.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DomainSchemaError) as error:
        load_domain("no-name-question", tmp_path)
    assert "name_question" in str(error.value)
