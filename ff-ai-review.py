#!/usr/bin/env python3
"""Noninteractive ff.ai PR review, without chat history or project writes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / '.venv'
VENV_PYTHON = VENV_DIR / 'bin/python'
if VENV_PYTHON.exists() and Path(sys.prefix) != VENV_DIR and os.environ.get('FFAI_NO_REEXEC') != '1':
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='ff.ai: AI-ревью изменений PR с RAG кода и документации')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--domain', default='')
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--output', type=Path, required=True, help='пустой каталог вне проекта')
    parser.add_argument('--model', default='')
    args = parser.parse_args(argv)
    # Review deliberately uses local token retrieval, without managed model processes.
    os.environ['FFAI_LLAMA_AUTOSTART'] = '0'
    from core import config, domains
    from core.api_client import APIClient
    from core.pr_review import ReviewError, run_review

    try:
        domain = domains.select_domain(args.repo, explicit=args.domain).domain
        report = run_review(args.repo, args.base, args.head, domain, args.output,
                            APIClient(config.get_api_key() or ''), args.model or config.DEFAULT_MODEL,
                            on_stage=lambda name: print(f'ff.ai: {name}', flush=True))
    except (ReviewError, domains.DomainError) as exc:
        print(f'ff.ai: {exc}', file=sys.stderr)
        return 1
    print(f'ff.ai: ревью выполнено; токенов: {report["usage"]["total_tokens"]}')
    print(f'Результаты: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
