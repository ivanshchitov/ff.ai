"""Exit nonzero for blocking findings or unavailable review; no model calls."""
import json
from pathlib import Path
import sys


def main():
    try:
        report = json.loads(Path(sys.argv[1]).read_text())
        if report.get('status') != 'ok':
            raise ValueError()
        findings = report['findings']
        bugs, architecture = findings['potential_bugs'], findings['architectural_problems']
        if not isinstance(bugs, list) or not isinstance(architecture, list):
            raise ValueError()
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        print('AI review gate: успешное ревью недоступно.', file=sys.stderr)
        return 1
    if bugs or architecture:
        print(f'AI review gate: merge заблокирован. Баги: {len(bugs)}, архитектура: {len(architecture)}.')
        return 1
    print('AI review gate: блокирующих замечаний нет.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
