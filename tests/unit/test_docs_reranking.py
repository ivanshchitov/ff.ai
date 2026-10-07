"""Вторая ступень отбора: сообщения оценщика, строгий разбор оценок и отбор по порогу."""

import json
from dataclasses import dataclass

import pytest

from core import config, reranking
from core.reranking import CandidateRating, RerankError


@dataclass(frozen=True)
class Candidate:
    path: str
    title: str = "Документ"
    section: str = "docs"
    url: str = ""
    snippet: str = "текст сниппета"
    score: float = 0.0
    reason: str = ""


CANDIDATES = (
    Candidate(path="doc/one", title="Первый"),
    Candidate(path="doc/two", title="Второй"),
    Candidate(path="doc/three", title="Третий"),
)


def _response(*pairs, extra: dict = None) -> str:
    results = [{"id": identifier, "score": score, "reason": f"причина {identifier}"} for identifier, score in pairs]
    payload = {"results": results}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def test_messages_carry_question_candidates_and_instruction():
    messages = reranking.build_messages("как получить координаты?", CANDIDATES)
    assert messages[0]["role"] == "system"
    assert reranking.instruction() in messages[0]["content"]
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert payload["question"] == "как получить координаты?"
    assert [item["id"] for item in payload["candidates"]] == [1, 2, 3]
    assert payload["candidates"][0]["path"] == "doc/one"
    assert "score" not in payload["candidates"][0], "оценщику не показывают прежние оценки"


def test_candidate_payload_trims_the_snippet():
    long_snippet = "с" * (config.DOCS_SNIPPET_CHARS + 500)
    payload = reranking.candidate_payload(1, Candidate(path="doc/one", snippet=long_snippet))
    assert len(payload["snippet"]) == config.DOCS_SNIPPET_CHARS


def test_valid_response_is_parsed():
    ratings = reranking.parse_response(_response((1, 0.9), (2, 0.4), (3, 0.0)), CANDIDATES)
    assert [rating.identifier for rating in ratings] == [1, 2, 3]
    assert ratings[0].score == 0.9
    assert ratings[0].reason == "причина 1"


def test_json_inside_prose_and_fences_is_found():
    text = "Вот результат:\n```json\n" + _response((1, 0.5), (2, 0.5), (3, 0.5)) + "\n```\nГотово."
    assert len(reranking.parse_response(text, CANDIDATES)) == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        (_response((1, 0.5), (2, 0.5)), "не оценил"),
        (_response((1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5)), "неизвестный"),
        (json.dumps({"results": [{"id": 1, "score": 0.5}, {"id": 1, "score": 0.5}, {"id": 2, "score": 0.5}, {"id": 3, "score": 0.5}]}), "повторяется"),
        (_response((1, 1.5), (2, 0.5), (3, 0.5)), "вне диапазона"),
        (_response((1, "много"), (2, 0.5), (3, 0.5)), "не число"),
        (json.dumps({"results": [{"id": "первый", "score": 0.5}]}), "не целое"),
        ("не JSON вовсе", "не разобран"),
        ("", "не разобран"),
        (json.dumps({"items": []}), "нет списка"),
        (json.dumps({"results": [1, 2, 3]}), "не объект"),
    ],
)
def test_invalid_responses_are_refused_with_a_reason(text, expected):
    with pytest.raises(RerankError) as error:
        reranking.parse_response(text, CANDIDATES)
    assert expected in str(error.value)


def test_missing_ids_are_listed():
    with pytest.raises(RerankError) as error:
        reranking.parse_response(_response((1, 0.5), (3, 0.5)), CANDIDATES)
    assert "2" in str(error.value)


def test_selection_keeps_the_threshold_boundary_and_the_order():
    ratings = reranking.parse_response(_response((1, 0.6), (2, 0.59), (3, 0.61)), CANDIDATES)
    selected = reranking.select(CANDIDATES, ratings, 0.6)
    assert [item.path for item in selected.kept] == ["doc/one", "doc/three"]
    assert [item.path for item in selected.dropped] == ["doc/two"]
    assert selected.threshold == 0.6
    assert not selected.empty


def test_selection_can_be_empty():
    ratings = reranking.parse_response(_response((1, 0.2), (2, 0.1), (3, 0.0)), CANDIDATES)
    selected = reranking.select(CANDIDATES, ratings, 0.6)
    assert selected.empty
    assert len(selected.dropped) == 3


def test_mark_puts_scores_on_copies():
    ratings = reranking.parse_response(_response((1, 0.9), (2, 0.4), (3, 0.0)), CANDIDATES)
    marked = reranking.mark(CANDIDATES, ratings)
    assert [item.score for item in marked] == [0.9, 0.4, 0.0]
    assert [item.reason for item in marked] == ["причина 1", "причина 2", "причина 3"]
    assert all(item.score == 0.0 for item in CANDIDATES), "исходные кандидаты не меняются"


def test_rating_lines_render_score_path_and_reason():
    ratings = (
        CandidateRating(identifier=1, score=0.85, reason="подтверждает API"),
        CandidateRating(identifier=2, score=0.1, reason=""),
    )
    lines = reranking.rating_lines(ratings, {1: "doc/one", 2: "doc/two"})
    assert "0.85" in lines[0] and "doc/one" in lines[0] and "подтверждает API" in lines[0]
    assert "без пояснения" in lines[1]


def test_modes_are_declared_as_data():
    assert reranking.MODES == ("enhanced", "baseline")
    assert config.DOCS_RETRIEVAL_MODE in reranking.MODES


def test_instruction_asset_exists_and_states_the_scale():
    text = reranking.instruction()
    assert "0.9–1.0" in text or "0.9-1.0" in text
    assert "Порог" not in text or "порог" in text.lower()
    assert (config.ASSETS_DIR / reranking.RERANK_ASSET).is_file()
