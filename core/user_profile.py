"""Профиль пользователя — слой персонализации агента.

Профиль отвечает на вопрос «как отвечать этому человеку», а не «что известно»: разделы задаёт сам
пользователь в опроснике, и профиль уходит модели в каждом запросе. Поэтому он и не долговременная
память: та собирается правилами из реплик (`core/memory_layers`), а профиль — объявленное
предпочтение, которое переживает и `/clear`, и перезапуск.

Модуль держит три вещи: модель профиля с разделами, файл профилей (несколько профилей и активный)
и автомат опросника. Имена и вопросы разделов — данные пакета домена (`DomainProfile`), поэтому
ядро не знает ни одного раздела; автомат тоже не знает терминала: `InterviewState` переводит ответ
пользователя в следующий вопрос, а терминальный слой только печатает вопрос и читает строку.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .domains import DomainProfile, ProfileSection

# Имя профиля — не раздел профиля: оно различает профили в файле, но ответ на первый вопрос
# опросника показывается той же журнальной строкой, что и разделы.
NAME_FIELD = "name"

PROFILE_PROMPT_ASSET = "profile_prompt.md"

# Предел длины раздела: профиль уходит в каждый запрос, а объём ответа задаётся настройками
# приложения, а не размером профиля.
PROFILE_VALUE_MAX_CHARS = 400

_TRUNCATION_MARK = "…"


def clip_value(value: str, limit: int = PROFILE_VALUE_MAX_CHARS) -> str:
    """Обрезает текст раздела до предела длины; многоточие входит в предел."""
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + _TRUNCATION_MARK


@dataclass(frozen=True)
class ProfileValue:
    """Заполненный раздел профиля: машинное имя раздела и его содержимое."""

    section: str
    value: str


@dataclass(frozen=True)
class UserProfile:
    """Профиль пользователя: имя и заполненные разделы.

    Разделы приходят из пакета домена, поэтому они и хранятся парой «имя — значение», а не полями
    класса: иначе каждый новый домен требовал бы правки этого модуля. Пустой раздел — нормальное
    состояние (пользователь мог пропустить вопрос), поэтому профиль считается непустым, только
    когда заполнен хотя бы один раздел: пустой профиль не даёт сообщения модели, и форма запроса
    без персонализации не меняется.
    """

    name: str = ""
    values: Tuple[ProfileValue, ...] = ()

    def value(self, section: str) -> str:
        """Содержимое раздела по машинному имени; пустая строка — раздела нет."""
        for item in self.values:
            if item.section == section:
                return item.value
        return ""

    def with_value(self, section: str, value: str) -> "UserProfile":
        """Копия профиля с заменённым разделом."""
        values = tuple(
            ProfileValue(
                section=item.section,
                value=value if item.section == section else item.value,
            )
            for item in self.values
        )
        if not any(item.section == section for item in self.values):
            values = values + (ProfileValue(section=section, value=value),)
        return replace(self, values=values)

    def filled(self, sections: Sequence[ProfileSection]) -> Tuple[Tuple[str, str], ...]:
        """Заполненные разделы в порядке объявления домена: подпись и содержимое."""
        return tuple(
            (section.label, self.value(section.id))
            for section in sections
            if self.value(section.id)
        )

    @property
    def is_empty(self) -> bool:
        """Профиль без единого заполненного раздела — сообщения модели он не даёт."""
        return not any(item.value for item in self.values)


@dataclass(frozen=True)
class ProfileQuestion:
    """Вопрос опросника: какой раздел он заполняет, как тот называется и что спрашивают."""

    field: str
    label: str
    prompt: str
    default: str = ""


def questions(profile: DomainProfile) -> Tuple[ProfileQuestion, ...]:
    """Скрипт опросника домена: вопрос об имени и по вопросу на каждый раздел.

    Имя спрашивается первым — так у профиля появляется имя до того, как его разделы заполнены,
    и повторная настройка узнаётся по нему, а не по содержимому.
    """
    script = [
        ProfileQuestion(
            field=NAME_FIELD, label=profile.name_label, prompt=profile.name_question
        )
    ]
    script.extend(
        ProfileQuestion(
            field=section.id,
            label=section.label,
            prompt=section.question,
            default=section.default,
        )
        for section in profile.sections
    )
    return tuple(script)


@dataclass(frozen=True)
class InterviewState:
    """Состояние опросника: скрипт домена и ответы по порядку вопросов.

    Автомат не знает ни о терминале, ни о файле профиля: он отдаёт очередной вопрос и собирает
    профиль из ответов поверх прежнего (пустой ответ оставляет раздел как был).
    """

    script: Tuple[ProfileQuestion, ...]
    answers: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def index(self) -> int:
        """Номер текущего вопроса, с единицы."""
        return len(self.answers) + 1

    @property
    def total(self) -> int:
        return len(self.script)

    @property
    def finished(self) -> bool:
        return len(self.answers) >= len(self.script)

    @property
    def question(self) -> ProfileQuestion:
        """Текущий вопрос; на исчерпанном скрипте — последний."""
        return self.script[min(len(self.answers), len(self.script) - 1)]

    @property
    def last_question(self) -> Optional[ProfileQuestion]:
        """Вопрос, на который ответил последний принятый ответ (для журнальной строки)."""
        return self.script[len(self.answers) - 1] if self.answers else None

    def answer(self, text: str) -> "InterviewState":
        """Принимает ответ и переходит к следующему вопросу."""
        return replace(self, answers=self.answers + (text,))

    def profile(self, base: UserProfile, default_name: str) -> UserProfile:
        """Собирает профиль из ответов поверх прежнего профиля.

        Пустой ответ оставляет раздел как был, поэтому пропущенный вопрос ничего не стирает, а
        повторная настройка под тем же именем — редактирование: заменяются только отвеченные
        разделы. Если ответа нет и раздела у профиля ещё нет, берётся значение по умолчанию
        домена — так пакет задаёт, каким он хочет видеть ответ по умолчанию. Пустое имя берёт
        прежнее, а если его нет — свободное имя по порядку.
        """
        profile = base
        for question, answer in zip(self.script, self.answers):
            value = clip_value(answer)
            if question.field == NAME_FIELD:
                if value:
                    profile = replace(profile, name=value)
                continue
            if not value:
                if profile.value(question.field):
                    continue
                value = clip_value(question.default)
                if not value:
                    continue
            profile = profile.with_value(question.field, value)
        if not profile.name:
            profile = replace(profile, name=default_name)
        return profile


class ProfileStore:
    """Файл профилей: несколько профилей и имя активного, чтение при создании, запись после изменения.

    Хранилище отдельное от истории диалога и долговременной памяти: профиль настраивает
    пользователь, и он живёт ровно столько, сколько ему скажет пользователь, — `/clear` его не
    касается. Чтение терпимое: нет файла, битый JSON или чужая форма — профилей нет, без ошибки;
    запись ошибкой сессию не роняет, но причина остаётся в `last_error` для отчёта.
    """

    def __init__(self, path: Optional[Path] = None):
        # Путь берётся в момент создания, а не при импорте модуля: раскладка состояния задаётся
        # переменными окружения, и значение по умолчанию не должно «замерзать» на импорте.
        self.path = Path(path) if path is not None else config.PROFILE_FILE
        self.last_error: Optional[str] = None
        self._profiles: Dict[str, UserProfile] = {}
        self._active: str = ""
        self._load()

    def _load(self) -> None:
        """Читает файл; нет файла, битый JSON или чужая форма — профилей нет, без ошибки."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as error:
            self.last_error = f"{self.path}: {error}"
            return
        if not isinstance(data, dict):
            self.last_error = f"{self.path}: ожидался объект JSON"
            return
        raw_profiles = data.get("profiles")
        if isinstance(raw_profiles, dict):
            for name, sections in raw_profiles.items():
                if not isinstance(sections, dict):
                    continue
                profile = UserProfile(name=str(name))
                for section, value in sections.items():
                    if isinstance(value, str):
                        profile = profile.with_value(str(section), value)
                self._profiles[str(name)] = profile
        active = data.get("active")
        if isinstance(active, str) and active in self._profiles:
            self._active = active

    @property
    def active_name(self) -> str:
        """Имя активного профиля; пусто — активного профиля нет."""
        return self._active

    def active(self) -> UserProfile:
        """Активный профиль; без активного — пустой профиль."""
        return self._profiles.get(self._active, UserProfile())

    def names(self) -> Tuple[str, ...]:
        """Имена профилей файла по порядку появления."""
        return tuple(self._profiles)

    def get(self, name: str) -> Optional[UserProfile]:
        return self._profiles.get(name)

    def save(self, profile: UserProfile, activate: bool = True) -> None:
        """Записывает профиль под его именем и делает его активным; на диск — сразу."""
        if not profile.name:
            return
        self._profiles[profile.name] = profile
        if activate:
            self._active = profile.name
        self._write()

    def use(self, name: str) -> Optional[UserProfile]:
        """Делает профиль активным; None — такого профиля нет (создаёт профили только опросник)."""
        profile = self._profiles.get(name)
        if profile is None:
            return None
        self._active = name
        self._write()
        return profile

    def forget(self, name: str) -> bool:
        """Удаляет профиль; False — такого профиля нет. Удаление активного оставляет файл без активного."""
        if name not in self._profiles:
            return False
        del self._profiles[name]
        if self._active == name:
            self._active = ""
        self._write()
        return True

    def next_name(self, prefix: str) -> str:
        """Свободное имя по порядку — для профиля, названного пустым ответом."""
        index = 1
        while f"{prefix} {index}" in self._profiles:
            index += 1
        return f"{prefix} {index}"

    def _write(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "active": self._active,
                        "profiles": {
                            name: {
                                item.section: item.value for item in profile.values
                            }
                            for name, profile in self._profiles.items()
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.last_error = None
        except OSError as error:
            self.last_error = f"{self.path}: {error}"


@lru_cache(maxsize=None)
def profile_instruction() -> str:
    """Инструкция работы с профилем из ассета assets/profile_prompt.md."""
    return (config.ASSETS_DIR / PROFILE_PROMPT_ASSET).read_text(encoding="utf-8").strip()


def profile_message(profile: UserProfile, sections: Sequence[ProfileSection]) -> Optional[str]:
    """Системное сообщение с профилем для запроса к модели; пустой профиль сообщения не даёт.

    Разделы печатаются с подписями домена, чтобы модель различала подачу, ограничения и интересы,
    а инструкция ассета подчиняет предпочтения пользователя настройкам приложения: формат ответа
    и его объём задаёт не профиль.
    """
    if profile.is_empty:
        return None
    lines: List[str] = [
        f"Профиль пользователя «{profile.name}» (персонализация — учитывай в каждом ответе):"
    ]
    for label, value in profile.filled(sections):
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append(profile_instruction())
    return "\n".join(lines)


def describe_sections(sections: Sequence[ProfileSection]) -> Tuple[str, ...]:
    """Подписи разделов профиля текстом — для отчёта интерфейса, без обращения к модели."""
    return tuple(section.label for section in sections)
