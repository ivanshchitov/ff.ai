"""Профиль пользователя: хранилище профилей, автомат опросника и сообщение модели."""

import json
from pathlib import Path

import pytest

from core import config, domains
from core.domains import DomainProfile, ProfileSection
from core.user_profile import (
    NAME_FIELD,
    PROFILE_VALUE_MAX_CHARS,
    InterviewState,
    ProfileStore,
    UserProfile,
    describe_sections,
    profile_message,
    questions,
)


@pytest.fixture
def domain_profile() -> DomainProfile:
    profile = domains.load_domain("aurora-qt5").profile
    assert profile is not None, "у пакета домена должны быть разделы профиля"
    return profile


@pytest.fixture
def script(domain_profile):
    return questions(domain_profile)


def _store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(tmp_path / "profile.json")


def _profile(name: str = "проба", **sections: str) -> UserProfile:
    profile = UserProfile(name=name)
    for section, value in sections.items():
        profile = profile.with_value(section, value)
    return profile


# --- хранилище ---


def test_sections_land_in_the_file_immediately(tmp_path):
    store = _store(tmp_path)
    store.save(_profile(style="коротко и просто"))

    saved = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert saved["profiles"]["проба"]["style"] == "коротко и просто"
    assert saved["active"] == "проба"


def test_several_profiles_live_in_one_file_with_the_active_one(tmp_path):
    store = _store(tmp_path)
    store.save(_profile("первый", style="коротко"))
    store.save(_profile("второй", style="развёрнуто"))

    assert store.names() == ("первый", "второй")
    assert store.active_name == "второй"
    assert store.active().value("style") == "развёрнуто"


def test_profile_without_a_name_is_not_saved(tmp_path):
    """Профиль без имени не различим в файле: запись молча ничего не меняет."""
    store = _store(tmp_path)
    store.save(UserProfile())

    assert not (tmp_path / "profile.json").exists()
    assert store.names() == ()


def test_using_an_unknown_profile_does_not_create_it(tmp_path):
    store = _store(tmp_path)
    store.save(_profile("первый", style="коротко"))

    assert store.use("нет такого") is None
    assert store.names() == ("первый",)
    assert store.active_name == "первый"


def test_using_a_known_profile_switches_the_active_one(tmp_path):
    store = _store(tmp_path)
    store.save(_profile("первый", style="коротко"))
    store.save(_profile("второй", style="развёрнуто"))

    switched = store.use("первый")

    assert switched is not None and switched.name == "первый"
    assert store.active_name == "первый"
    assert store.active().value("style") == "коротко"


def test_forget_removes_only_the_named_profile(tmp_path):
    store = _store(tmp_path)
    store.save(_profile("первый", style="коротко"))
    store.save(_profile("второй", style="развёрнуто"))

    assert store.forget("второй") is True
    assert store.names() == ("первый",)
    assert store.forget("второй") is False


def test_forgetting_the_active_profile_leaves_no_active_profile(tmp_path):
    store = _store(tmp_path)
    store.save(_profile("первый", style="коротко"))

    store.forget("первый")

    assert store.active_name == ""
    assert store.active().is_empty


def test_missing_and_broken_file_read_as_no_profiles(tmp_path):
    missing = _store(tmp_path)
    assert missing.names() == () and missing.active_name == ""
    assert missing.last_error is None

    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{не json", encoding="utf-8")
    broken = ProfileStore(broken_path)
    assert broken.names() == () and broken.active().is_empty
    assert broken.last_error, "причина читается отчётом, а не выпадает в лог"


def test_active_profile_pointing_nowhere_is_ignored(tmp_path):
    (tmp_path / "profile.json").write_text(
        json.dumps({"active": "нет такого", "profiles": {"первый": {"style": "коротко"}}}),
        encoding="utf-8",
    )

    store = ProfileStore(tmp_path / "profile.json")

    assert store.active_name == ""
    assert store.active().is_empty


def test_next_name_skips_taken_names(tmp_path, domain_profile):
    store = _store(tmp_path)
    store.save(_profile(f"{domain_profile.name_prefix} 1", style="коротко"))

    assert store.next_name(domain_profile.name_prefix) == f"{domain_profile.name_prefix} 2"


def test_write_error_does_not_break_the_session(tmp_path):
    store = ProfileStore(tmp_path)  # каталог вместо файла: запись невозможна

    store.save(_profile("первый", style="коротко"))

    assert store.active().value("style") == "коротко"
    assert store.last_error, "причина видна отчёту, а не только в логе ошибок"


def test_default_path_is_read_at_creation_time():
    """Путь состояния ленивый: значение по умолчанию не «замерзает» на импорте модуля."""
    assert ProfileStore().path == config.PROFILE_FILE


# --- опросник ---


def test_interview_asks_the_name_and_every_section(domain_profile, script):
    assert [question.field for question in script] == [
        NAME_FIELD,
        *[section.id for section in domain_profile.sections],
    ]
    assert len(script) <= 5, "опросник не должен растягиваться: вопрос на раздел и имя"


def test_interview_asks_one_question_at_a_time(script):
    state = InterviewState(script=script)

    assert state.question.field == NAME_FIELD
    assert state.total == len(script)

    state = state.answer("проба")

    assert state.question.field == script[1].field
    assert state.index == 2
    assert state.last_question is not None and state.last_question.field == NAME_FIELD


def test_answers_become_the_sections_of_the_profile(domain_profile, script):
    answers = ["проба"] + [f"значение {index}" for index in range(len(script) - 1)]
    state = InterviewState(script=script)
    for answer in answers:
        state = state.answer(answer)

    profile = state.profile(UserProfile(), "запасное имя")

    assert state.finished is True
    assert profile.name == "проба"
    for section, answer in zip(domain_profile.sections, answers[1:]):
        assert profile.value(section.id) == answer


def test_empty_answer_keeps_the_section_as_it_was(domain_profile, script):
    section = domain_profile.sections[0]
    base = _profile("проба", **{section.id: "прежнее значение"})
    state = InterviewState(script=script)

    for answer in ["", ""] + [""] * (len(script) - 2):
        state = state.answer(answer)
    profile = state.profile(base, "запасное имя")

    assert profile.name == "проба"
    assert profile.value(section.id) == "прежнее значение"


def test_empty_answer_takes_the_domain_default(domain_profile, script):
    """Значение по умолчанию — то, каким домен хочет видеть раздел, если о нём не сказали."""
    section = next(section for section in domain_profile.sections if section.default)
    state = InterviewState(script=script)
    for _ in script:
        state = state.answer("")

    profile = state.profile(UserProfile(), "запасное имя")

    assert profile.value(section.id) == section.default


def test_answer_wins_over_the_domain_default(domain_profile, script):
    section = next(section for section in domain_profile.sections if section.default)
    index = [item.field for item in script].index(section.id)
    state = InterviewState(script=script)
    for position in range(len(script)):
        state = state.answer("своё значение" if position == index else "")

    profile = state.profile(UserProfile(), "запасное имя")

    assert profile.value(section.id) == "своё значение"


def test_empty_name_gives_a_free_default_name(script):
    state = InterviewState(script=script)
    for _ in script:
        state = state.answer("")

    assert state.profile(UserProfile(), "профиль 3").name == "профиль 3"


def test_repeat_setup_replaces_only_the_answered_sections(domain_profile, script):
    first, second = domain_profile.sections[0], domain_profile.sections[1]
    base = _profile("проба", **{first.id: "прежнее", second.id: "прежнее"})
    state = InterviewState(script=script)
    for position in range(len(script)):
        if position == 0:
            state = state.answer("")
            continue
        state = state.answer("новое" if script[position].field == second.id else "")

    profile = state.profile(base, "запасное имя")

    assert profile.name == "проба"
    assert profile.value(second.id) == "новое"
    assert profile.value(first.id) == "прежнее"


def test_long_answer_is_clipped(domain_profile, script):
    section = domain_profile.sections[0]
    state = InterviewState(script=script)
    for position in range(len(script)):
        state = state.answer("о" * (PROFILE_VALUE_MAX_CHARS + 50) if position == 1 else "")

    profile = state.profile(UserProfile(), "запасное имя")

    assert len(profile.value(section.id)) == PROFILE_VALUE_MAX_CHARS
    assert profile.value(section.id).endswith("…")


def test_with_value_replaces_and_adds_sections():
    profile = UserProfile(name="проба").with_value("style", "коротко")

    assert profile.value("style") == "коротко"
    assert profile.value("нет такого") == ""
    assert profile.with_value("style", "развёрнуто").value("style") == "развёрнуто"


def test_profile_without_sections_is_empty():
    assert UserProfile(name="проба").is_empty
    assert not UserProfile(name="проба").with_value("style", "коротко").is_empty


# --- сообщение для модели ---


def test_empty_profile_gives_no_message(domain_profile):
    sections = domain_profile.sections
    assert profile_message(UserProfile(), sections) is None
    assert profile_message(UserProfile(name="проба"), sections) is None


def test_message_lists_the_filled_sections_with_the_domain_labels(domain_profile, script):
    section = domain_profile.sections[0]
    message = profile_message(_profile("проба", **{section.id: "коротко"}), domain_profile.sections)

    assert message is not None
    assert "проба" in message
    assert f"- {section.label}: коротко" in message
    for other in domain_profile.sections[1:]:
        assert f"- {other.label}: " not in message


def test_message_carries_the_instruction_asset(domain_profile):
    section = domain_profile.sections[0]
    message = profile_message(
        _profile("проба", **{section.id: "коротко"}), domain_profile.sections
    )

    assert message is not None
    assert "настройкам приложения" in message
    assert "правилам домена" in message


def test_describe_sections_lists_the_domain_labels(domain_profile):
    assert describe_sections(domain_profile.sections) == tuple(
        section.label for section in domain_profile.sections
    )


def test_questions_follow_the_domain_script(domain_profile):
    script = questions(domain_profile)

    assert script[0].field == NAME_FIELD
    assert script[0].prompt == domain_profile.name_question
    for question, section in zip(script[1:], domain_profile.sections):
        assert question.field == section.id
        assert question.label == section.label
        assert question.prompt == section.question
        assert question.default == section.default


def test_foreign_section_of_a_profile_is_kept_in_the_file(tmp_path):
    """Значение чужого домена не теряется при переключении пакета: оно просто не показывается."""
    store = _store(tmp_path)
    store.save(_profile("проба", **{"чуждый_раздел": "значение"}))

    saved = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert saved["profiles"]["проба"]["чуждый_раздел"] == "значение"
    assert ProfileStore(tmp_path / "profile.json").active().value("чуждый_раздел") == "значение"


def test_section_labels_come_from_the_pack_not_from_the_code():
    """Разделы — данные: чужой пакет даёт чужой опросник без правок ядра."""
    foreign = DomainProfile(
        name_label="Имя",
        name_question="Как вас зовут?",
        name_prefix="профиль",
        sections=(
            ProfileSection(id="tone", label="Тон", question="Каким тоном отвечать?", default=""),
        ),
    )

    script = questions(foreign)
    profile = InterviewState(script=script).answer("проба").answer("ровным").profile(
        UserProfile(), "профиль 1"
    )

    assert [question.field for question in script] == [NAME_FIELD, "tone"]
    assert profile.value("tone") == "ровным"
    assert profile_message(profile, foreign.sections) is not None
