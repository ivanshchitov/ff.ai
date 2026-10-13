"""Trusted Actions publisher; the PR supplies only the finished review artifact."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.github_review import CommentError, GitHubComments, publish_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--pr', type=int, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run-url', required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding='utf-8'))
        result = publish_review(GitHubComments(os.environ.get('GITHUB_TOKEN', '')),
                                args.repository, args.pr, args.review.read_text(encoding='utf-8'),
                                report, args.run_url)
    except (CommentError, OSError, ValueError, KeyError, TypeError) as exc:
        message = str(exc) if isinstance(exc, CommentError) else 'Непригодные данные отчёта комментария.'
        print(message, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
