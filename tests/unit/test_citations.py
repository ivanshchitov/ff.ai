"""Проверка ссылок и цитат: подтверждение, нарушения, повтор и замена ответа."""

from dataclasses import dataclass

import pytest

from core import citations, config


@dataclass(frozen=True)
class Fragment:
    """Доставленный фрагмент в том виде, в каком его видит проверка: путь, источник, текст."""

    identifier: str
    source: str
    text: str
    section: str = "docs"
    version: str = "5.2.1"
    truncated: bool = False


DOCUMENT = Fragment(
    identifier="doc/software_development/guides/cpp_api/positioning",
    source="developer.auroraos.ru",
    text=(
        "Получение координат устройства\n\n"
        "Информацию о местоположении предоставляет служба Geoclue. В приложениях её можно "
        "получить с помощью модуля Qt Positioning как в C++, так и в QML-коде."
    ),
)


def test_confirmed_answer_with_source_and_quote():
    answer = (
        "Координаты даёт модуль Qt Positioning.\n"
        "Цитата: «Информацию о местоположении предоставляет служба Geoclue»\n"
        "Источник: doc/software_development/guides/cpp_api/positioning"
    )
    check = citations.check_answer(answer, [DOCUMENT])
    assert check.confirmed
    assert check.violations == ()


def test_quote_is_matched_after_normalization():
    answer = (
        "Смотри doc/software_development/guides/cpp_api/positioning: "
        "«ИНФОРМАЦИЮ О МЕСТОПОЛОЖЕНИИ\nпредоставляет   служба Geoclue»"
    )
    assert citations.check_answer(answer, [DOCUMENT]).confirmed


def test_source_without_quote_is_a_violation():
    answer = "Смотри doc/software_development/guides/cpp_api/positioning — там всё есть."
    check = citations.check_answer(answer, [DOCUMENT])
    assert not check.confirmed
    assert citations.NO_QUOTE_MARKER in check.violations


def test_quote_without_source_is_a_violation():
    answer = "Информацию о местоположении предоставляет служба Geoclue."
    check = citations.check_answer(answer, [DOCUMENT])
    assert not check.confirmed
    assert citations.NO_SOURCE_MARKER in check.violations


def test_short_quote_does_not_confirm():
    source = "doc/software_development/guides/cpp_api/positioning"
    short = DOCUMENT.text[: config.DOCS_CITATION_MIN_CHARS - 5]
    check = citations.check_answer(f"{source}: «{short}»", [DOCUMENT])
    assert not check.confirmed
    assert citations.NO_QUOTE_MARKER in check.violations


def test_quote_from_an_undelivered_fragment_does_not_confirm():
    """Ссылка на чужой документ — самая вероятная форма выдумки, и она не подтверждает ответ."""
    answer = (
        "Цитата: «Информацию о местоположении предоставляет служба Geoclue»\n"
        "Источник: doc/software_development/guides/somewhere/else"
    )
    check = citations.check_answer(answer, [DOCUMENT])
    assert not check.confirmed
    assert citations.NO_SOURCE_MARKER in check.violations


def test_window_must_start_at_a_word_boundary():
    answer = "Источник doc/software_development/guides/cpp_api/positioning: «служба Geoclue»."
    check = citations.check_answer(answer, [DOCUMENT])
    assert not check.confirmed, "обрывок слова не должен считаться цитатой"


def test_empty_fragments_skip_the_check():
    check = citations.check_answer("Любой ответ по памяти.", [])
    assert check.confirmed
    assert check.no_context


def test_empty_answer_is_a_violation_not_a_crash():
    check = citations.check_answer("", [DOCUMENT])
    assert not check.confirmed
    assert set(check.violations) == {citations.NO_SOURCE_MARKER, citations.NO_QUOTE_MARKER}


def test_context_message_names_path_section_and_version():
    message = citations.context_message([DOCUMENT], note="документация портала, версия 5.2.1")
    assert message is not None
    assert "doc/software_development/guides/cpp_api/positioning" in message
    assert "раздел: docs" in message
    assert "версия: 5.2.1" in message
    assert "developer.auroraos.ru" in message
    assert "документация портала, версия 5.2.1" in message
    assert DOCUMENT.text in message


def test_context_message_marks_truncated_fragments():
    truncated = Fragment(
        identifier="doc/sdk/tools/mb2",
        source="developer.auroraos.ru",
        text="Сборка пакетов",
        truncated=True,
    )
    message = citations.context_message([truncated])
    assert "(фрагмент обрезан)" in message


def test_context_message_without_fragments_is_none():
    assert citations.context_message([]) is None


def test_numbering_is_stable_across_fragments():
    second = Fragment(identifier="doc/sdk/tools/mb2", source="developer.auroraos.ru", text="Второй")
    message = citations.context_message([DOCUMENT, second])
    assert "[1]" in message and "[2]" in message
    assert message.index("[1]") < message.index("[2]")


def test_retry_prompt_lists_violations_and_demands_a_quote():
    text = citations.retry_prompt([citations.NO_SOURCE_MARKER, citations.NO_QUOTE_MARKER])
    assert citations.NO_SOURCE_MARKER in text
    assert "дословную цитату" in text


def test_disclaimer_asks_for_a_refinement():
    text = citations.disclaimer_text()
    assert text.startswith("Не знаю")
    assert "уточн" in text.casefold()


def test_instruction_asset_is_present_and_non_empty():
    message = citations.citations_message()
    assert "Цитаты:" in message and "Источники:" in message
    assert citations.CITATIONS_ASSET == "docs_citations_prompt.md"
    assert (config.ASSETS_DIR / citations.CITATIONS_ASSET).is_file()


def test_explicit_opt_out_is_an_allowed_form():
    """Домен разрешает смежные вопросы без документации — проверка обязана это пропускать."""
    answer = (
        "Документация не относится к вопросу: moc — это генератор метаобъектного кода Qt, "
        "он вызывается системой сборки до компиляции."
    )
    check = citations.check_answer(answer, [DOCUMENT])
    assert check.confirmed
    assert check.opted_out
    assert check.violations == ()


def test_opt_out_marker_must_be_at_the_beginning():
    """Отговорка в конце ответа не считается отказом от фрагментов: окно проверки ограничено."""
    preamble = "Сначала подробно про Qt и сборку. " * 12
    answer = preamble + citations.OPT_OUT_MARKER
    assert len(preamble) > citations.OPT_OUT_WINDOW
    check = citations.check_answer(answer, [DOCUMENT])
    assert not check.confirmed


def test_retry_prompt_mentions_the_opt_out():
    text = citations.retry_prompt([citations.NO_QUOTE_MARKER])
    assert citations.OPT_OUT_MARKER in text
