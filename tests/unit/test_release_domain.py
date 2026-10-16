"""Release policy is optional domain data, validated before any project action."""

import copy
import json
import re
from pathlib import Path

import pytest

from core.domains import DomainSchemaError, load_domain
from tests.unit.test_domains import _pack


POLICY = {
    "schema": 1,
    "metadata": [{"glob": "package/*.meta", "required": [r"(?m)^Version: \S+"],
                  "identity_pattern": r"(?m)^Name: (\S+)"}],
}


def _write(root: Path, data):
    path = _pack(root)
    (path / "release.json").write_text(json.dumps(data), encoding="utf-8")


def test_release_policy_is_optional(tmp_path):
    _pack(tmp_path)
    assert load_domain("probe", tmp_path).release is None


def test_release_policy_loads_defaults_and_trusted_test_argv(tmp_path):
    _write(tmp_path, POLICY)
    release = load_domain("probe", tmp_path).release
    assert release.build_required is True
    assert release.tests_required is False
    assert release.test_command == ()
    assert release.metadata[0].glob == "package/*.meta"
    assert release.metadata[0].required == (r"(?m)^Version: \S+",)
    assert release.metadata[0].identity_pattern == r"(?m)^Name: (\S+)"
    data = dict(POLICY, build_required=False, tests_required=True,
                test_command=["./run-tests", "--verbose"])
    _write(tmp_path, data)
    release = load_domain("probe", tmp_path).release
    assert release.build_required is False
    assert release.tests_required is True
    assert release.test_command == ("./run-tests", "--verbose")


@pytest.mark.parametrize("update", [
    {"schema": True}, {"schema": "1"}, {"schema": 2}, {"schema": None},
    {"unknown": 1}, {"build_required": 1}, {"tests_required": "yes"},
    {"test_command": "pytest"}, {"test_command": [1]}, {"test_command": [""]},
    {"test_command": ["run\x00tests"]}, {"metadata": {}}, {"metadata": []},
    {"metadata": [1]}, {"metadata": [{"glob": "*.meta", "required": [], "extra": 1}]},
    {"metadata": [{"glob": "../*.meta", "required": []}]},
    {"metadata": [{"glob": "/tmp/*.meta", "required": []}]},
    {"metadata": [{"glob": "dir\\*.meta", "required": []}]},
    {"metadata": [{"glob": "", "required": []}]},
    {"metadata": [{"glob": "./*.meta", "required": []}]},
    {"metadata": [{"glob": "C:/*.meta", "required": []}]},
    {"metadata": [{"glob": "*.meta", "required": "Version"}]},
    {"metadata": [{"glob": "*.meta", "required": [1]}]},
    {"metadata": [{"glob": "*.meta", "required": [""]}]},
    {"metadata": [{"glob": "*.meta", "required": ["["]}]},
    {"metadata": [{"glob": "*.meta", "required": [], "identity_pattern": 1}]},
    {"metadata": [{"glob": "*.meta", "required": [], "identity_pattern": "["}]},
    {"metadata": [{"glob": "*.meta", "required": [], "identity_pattern": "Name"}]},
])
def test_malformed_release_policy_names_file(tmp_path, update):
    data = copy.deepcopy(POLICY)
    data.update(update)
    _write(tmp_path, data)
    with pytest.raises(DomainSchemaError, match=r"release\.json"):
        load_domain("probe", tmp_path)


def test_release_schema_is_required(tmp_path):
    data = copy.deepcopy(POLICY)
    del data["schema"]
    _write(tmp_path, data)
    with pytest.raises(DomainSchemaError, match="schema"):
        load_domain("probe", tmp_path)


def test_bundled_release_policy_checks_complete_metadata():
    release = load_domain("aurora-qt5").release
    assert release.build_required and not release.tests_required
    assert release.test_command == ()
    samples = {
        "rpm/org.example.App.spec": "Name: org.example.App\nVersion: 1.0\nRelease: 1\nLicense: MIT\nBuildRequires: pkgconfig(test)\n",
        "app/org.example.App.desktop": "[Desktop Entry]\nType=Application\nName=Example\nExec=/usr/bin/org.example.App\nIcon=org.example.App\n[X-Application]\nPermissions=UserDirs\nOrganizationName=org.example\nApplicationName=App\n",
    }
    captures = []
    for rule in release.metadata:
        matching = [text for path, text in samples.items() if Path(path).match(rule.glob)]
        assert matching, rule.glob
        for text in matching:
            assert all(re.search(pattern, text) for pattern in rule.required)
            captures.append(re.search(rule.identity_pattern, text).group(1))
            for field in ("Version: 1.0", "Exec=/usr/bin/org.example.App"):
                if field in text:
                    assert any(not re.search(pattern, text.replace(field, ""))
                               for pattern in rule.required)
    assert captures == ["org.example.App", "org.example.App"]
