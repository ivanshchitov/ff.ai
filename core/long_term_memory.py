"""Долговременная память агента (сведения о пользователе между сессиями) — отдельный файл.

Слой живёт дольше диалога: `/clear` его не касается, поэтому хранить его в конверте истории
нельзя. Записи — плоская таблица «ключ → значение с меткой категории»; ключи приходят из правил
домена и команд пользователя, поэтому число записей ограничено: файл уходит в каждый запрос, и
расти без предела ему нечего. Вытесняются самые ранние записи — источник записей один и тот же
(реплики пользователя), и давняя запись с тем же смыслом обычно уже заменена новой.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import config, memory_layers

MAX_ENTRIES = 50


def _read_entries(raw: object) -> Dict[str, Tuple[str, str]]:
    """Разбор таблицы записей: ключ → (значение, категория).

    Запись без категории читается как заметка: файл могли править руками, и терять данные из-за
    отсутствующего поля нельзя.
    """
    if not isinstance(raw, dict):
        return {}
    entries: Dict[str, Tuple[str, str]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            text = value.get("value")
            category = value.get("category")
            if isinstance(text, str):
                entries[str(key)] = (
                    text,
                    str(category) if isinstance(category, str) and category else memory_layers.CATEGORY_NOTE,
                )
            continue
        if isinstance(value, str):
            entries[str(key)] = (value, memory_layers.CATEGORY_NOTE)
    return entries


class LongTermMemory:
    """Файл долговременной памяти: чтение при создании, запись сразу после изменения."""

    def __init__(self, path: Optional[Path] = None):
        # Путь берётся в момент создания, а не при импорте модуля: раскладка состояния задаётся
        # переменными окружения, и значение по умолчанию не должно «замерзать» на импорте.
        self.path = Path(path) if path is not None else config.MEMORY_FILE
        self.last_error: Optional[str] = None
        self._entries: Dict[str, Tuple[str, str]] = self._load()

    def _load(self) -> Dict[str, Tuple[str, str]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as error:
            self.last_error = f"{self.path}: {error}"
            return {}
        if not isinstance(data, dict):
            self.last_error = f"{self.path}: ожидался объект JSON"
            return {}
        return _read_entries(data.get("entries"))

    def records(self) -> Tuple[memory_layers.MemoryRecord, ...]:
        """Записи слоя по порядку появления: слой, категория, ключ и значение."""
        return tuple(
            memory_layers.MemoryRecord(
                layer=memory_layers.LONG_TERM, category=category, key=key, value=value
            )
            for key, (value, category) in self._entries.items()
        )

    def get(self, key: str) -> Optional[str]:
        entry = self._entries.get(key)
        return entry[0] if entry is not None else None

    def remember(
        self, key: str, value: str, category: str = memory_layers.CATEGORY_NOTE
    ) -> None:
        """Записывает сведение: повторная запись по ключу заменяет значение, а не добавляет запись."""
        # Повторная запись переезжает в конец: вытесняется самая давняя запись, а не та, к которой
        # пользователь только что вернулся.
        if key in self._entries:
            del self._entries[key]
        self._entries[key] = (memory_layers.clip_value(value), category)
        while len(self._entries) > MAX_ENTRIES:
            del self._entries[next(iter(self._entries))]
        self.save()

    def forget(self, key: str) -> bool:
        """Удаляет запись по ключу; False — такой записи нет."""
        if key not in self._entries:
            return False
        del self._entries[key]
        self.save()
        return True

    def clear(self) -> None:
        self._entries = {}
        self.save()

    def save(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "entries": {
                            key: {"value": value, "category": category}
                            for key, (value, category) in self._entries.items()
                        }
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.last_error = None
        except OSError as error:
            # Ошибка записи не должна ронять сессию: память просто не сохранится, и это видно
            # в отчёте, а не только в отвалившемся запросе.
            self.last_error = f"{self.path}: {error}"
