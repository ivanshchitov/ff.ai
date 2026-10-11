"""Semantic intent selection with strict output and a conservative no-write guard."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from . import config


@dataclass(frozen=True)
class IntentDecision:
    kind: str
    goal: str = ''
    clarification: str = ''


def write_forbidden(text: str) -> bool:
    # This is a veto, never the intent classifier. Unknown formulations are handled by the model.
    return bool(re.search(r"ничего\s+не\s+меня|не\s+(?:меняй|изменяй|редактируй|правь)|только\s+(?:предложи|исследуй|объясни)|\bdo\s+not\s+(?:modify|change|edit)|\bdon['’]t\s+(?:modify|change|edit)", text, re.I))


def classify_request(text, conversation, domain, ask) -> IntentDecision:
    fallback = IntentDecision('clarify', clarification='Уточните ожидаемый результат: объяснить, исследовать или изменить проект?')
    messages = [
        {'role':'system', 'content': (config.ASSETS_DIR / 'request_routing_prompt.md').read_text(encoding='utf-8')},
        {'role':'user', 'content': json.dumps({'request':text, 'conversation':list(conversation)[-8:],
                                             'domain_rules':domain.prompt('task')}, ensure_ascii=False)},
    ]
    try:
        raw = ask(messages)
        if hasattr(raw, 'content'):
            raw = raw.content
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {'kind','goal','clarification'}:
            return fallback
        if not all(isinstance(data[k], str) for k in data):
            return fallback
        kind, goal, clarification = (data[k].strip() for k in ('kind','goal','clarification'))
        if kind not in ('answer','investigate','develop','clarify'):
            return fallback
        if kind == 'develop' and not goal or kind == 'clarify' and not clarification:
            return fallback
        if kind == 'develop' and write_forbidden(text):
            return IntentDecision('investigate')
        return IntentDecision(kind, goal if kind == 'develop' else '', clarification if kind == 'clarify' else '')
    except Exception:
        # A model/provider/parse failure cannot become write authorization.
        return fallback
