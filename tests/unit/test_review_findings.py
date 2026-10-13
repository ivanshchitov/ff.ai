"""Merge decisions derive from validated findings, never prose heuristics."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


FINDING = {'severity': 'high', 'path': 'transfer.cpp', 'start_line': 1, 'end_line': 1,
           'description': 'Проверка пути удалена.', 'recommendation': 'Вернуть проверку.'}


@pytest.mark.parametrize('bugs,architecture,blocked', [([FINDING], [], True),
                                                       ([], [FINDING], True), ([], [], False)])
def test_findings_generate_review_and_gate(bugs, architecture, blocked):
    from core.pr_review import parse_review_findings
    data = {'potential_bugs': bugs, 'architectural_problems': architecture,
            'recommendations': ['Добавить тест.']}
    findings, text, gate = parse_review_findings(json.dumps(data), {'transfer.cpp'})
    assert findings == data
    assert (gate['status'] == 'blocked') == blocked
    assert all(h in text for h in ('Потенциальные баги', 'Архитектурные проблемы', 'Рекомендации'))
    assert 'Добавить тест.' in text
    if blocked:
        assert 'transfer.cpp:L1-L1' in text and 'Проверка пути удалена.' in text


@pytest.mark.parametrize('content', ['{}', 'not JSON',
    json.dumps({'potential_bugs': 'none', 'architectural_problems': [], 'recommendations': []}),
    json.dumps({'potential_bugs': [dict(FINDING, path='unknown.cpp')],
                'architectural_problems': [], 'recommendations': []}),
    json.dumps({'potential_bugs': [dict(FINDING, start_line=0)],
                'architectural_problems': [], 'recommendations': []})])
def test_malformed_findings_are_not_treated_as_clean_review(content):
    from core.pr_review import ReviewError, parse_review_findings
    with pytest.raises(ReviewError):
        parse_review_findings(content, {'transfer.cpp'})


@pytest.mark.parametrize('bugs,architecture,status,exit_code', [
    ([FINDING], [], 'ok', 1), ([], [FINDING], 'ok', 1),
    ([], [], 'ok', 0), ([], [], 'failed', 1)])
def test_gate_process_exit_code_blocks_findings_and_errors(tmp_path, bugs, architecture, status, exit_code):
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'status': status, 'findings': {
        'potential_bugs': bugs, 'architectural_problems': architecture,
        'recommendations': ['Тест желателен.']}}))
    script = Path(__file__).resolve().parents[2] / 'scripts/check_review_gate.py'
    result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    assert 'Traceback' not in result.stderr
