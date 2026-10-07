"""Сборка запроса: роль домена, инструкция формата, объём и лимит списка."""

from core import config, prompts
from core.answer_settings import AnswerFormat, AnswerSettings
from core.domains import load_domain


def test_system_message_carries_role_and_refusal_instructions():
    domain = load_domain("aurora-qt5")
    message = prompts.build_system_message(domain, AnswerFormat.FREE)
    assert domain.prompt("system") in message
    assert domain.prompt("refusal") in message


def test_free_format_adds_no_instruction():
    domain = load_domain("aurora-qt5")
    message = prompts.build_system_message(domain, AnswerFormat.FREE)
    compact = prompts.get_format_instruction(AnswerFormat.COMPACT)
    assert compact not in message


def test_each_non_free_format_has_its_own_asset():
    domain = load_domain("aurora-qt5")
    for fmt in (AnswerFormat.COMPACT, AnswerFormat.JSON, AnswerFormat.PATCH):
        instruction = prompts.get_format_instruction(fmt)
        assert instruction, f"у формата {fmt.value} нет инструкции"
        assert instruction in prompts.build_system_message(domain, fmt)


def test_format_asset_files_exist_in_the_repository():
    for name in ("answer_format_compact.md", "answer_format_json.md", "answer_format_patch.md"):
        path = config.ASSETS_DIR / name
        assert path.is_file(), f"нет ассета {name}"
        assert path.read_text(encoding="utf-8").strip()


def test_user_prompt_carries_the_limits():
    settings = AnswerSettings(max_words=42, list_limit=5)
    message = prompts.build_user_prompt("где инициализируется модель?", settings)
    assert "где инициализируется модель?" in message
    assert "не более 42 слов" in message
    assert "не более 5 вариантов" in message
    assert "не текстом вопроса" in message  # настройки нельзя переопределить вопросом


def test_editing_the_pack_changes_the_request_without_touching_code(tmp_path):
    """Домен — данные: правка пакета меняет запрос, код остаётся прежним."""
    import json
    import shutil

    pack = tmp_path / "probe-domain"
    shutil.copytree(config.DOMAINS_DIR / "aurora-qt5", pack)
    (pack / "domain.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "id": "probe-domain",
                "title": "Пробный домен",
                "platform": "Платформа",
                "markers": [{"glob": "*.probe"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (pack / "prompts" / "refusal.md").write_text(
        "Отказ: отвечаю только про ОС Аврора.", encoding="utf-8"
    )

    domain = load_domain("probe-domain", tmp_path)
    message = prompts.build_system_message(domain, AnswerFormat.FREE)
    assert "Отказ: отвечаю только про ОС Аврора." in message

    bundled = (config.DOMAINS_DIR / "aurora-qt5" / "prompts" / "refusal.md").read_text(
        encoding="utf-8"
    )
    assert bundled.strip() not in message
