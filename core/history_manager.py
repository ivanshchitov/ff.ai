"""Сохранение и загрузка истории диалогов (с метриками, памятью стратегии и рабочей памятью).

Файл — конверт: {"summary", "summary_covers", "facts", "working", "dialogues"}. Он держит
краткосрочный слой памяти агента (обмены дословно) и рабочую память задачи (блок working: цель и
ограничения); резюме и факты — производная память активной стратегии. Долговременная память
пользователя живёт в отдельном файле (см. core/long_term_memory.py) и конверта не касается.
Исходные записи хранятся полностью, вытеснения нет. Файл прежнего вида (голый список записей)
читается как конверт без резюме, фактов и рабочей памяти.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .usage import SessionUsage, sum_usage


def _read_block(raw: Any) -> Dict[str, str]:
    """Разбор плоского блока конверта (факты, рабочая память): пары «строка — строка»."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(value, (str, int, float, bool))
    }


class HistoryManager:
    def __init__(self, path: Optional[Path] = None):
        # Путь берётся в момент создания, а не при импорте модуля: раскладка состояния задаётся
        # переменными окружения, и значение по умолчанию не должно «замерзать» на импорте.
        self.path = Path(path) if path is not None else config.HISTORY_FILE
        self.summary: Optional[str] = None
        self.summary_covers: int = 0
        self.facts: Dict[str, str] = {}
        self.working: Dict[str, str] = {}
        self.dialogues: List[Dict[str, Any]] = self._load()
        self.last_error: Optional[str] = None

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            # Старый формат — голый список записей: конверт без резюме и фактов.
            return data
        if isinstance(data, dict):
            summary = data.get("summary")
            self.summary = summary if isinstance(summary, str) else None
            try:
                self.summary_covers = int(data.get("summary_covers", 0))
            except (TypeError, ValueError):
                self.summary_covers = 0
            self.facts = _read_block(data.get("facts"))
            self.working = _read_block(data.get("working"))
            records = data.get("dialogues", [])
            return records if isinstance(records, list) else []
        return []

    def add(
        self,
        question: str,
        answer: str,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Дописывает обмен без вытеснения прежних; usage — метрики запроса."""
        record: Dict[str, Any] = {"question": question, "answer": answer}
        if usage is not None:
            record["usage"] = usage
        self.dialogues.append(record)
        self.save()

    def set_summary(self, summary: str, summary_covers: int) -> None:
        """Обновляет сжатое резюме и число покрытых им ведущих обменов; файл сразу переписывается."""
        self.summary = summary
        self.summary_covers = summary_covers
        self.save()

    def set_facts(self, facts: Dict[str, str]) -> None:
        """Обновляет блок фактов диалога; файл сразу переписывается."""
        self.facts = dict(facts)
        self.save()

    def set_working(self, working: Dict[str, str]) -> None:
        """Обновляет блок рабочей памяти задачи (цель и ограничения); файл сразу переписывается."""
        self.working = dict(working)
        self.save()

    def clear(self) -> None:
        self.summary = None
        self.summary_covers = 0
        self.facts = {}
        self.working = {}
        self.dialogues = []
        self.save()

    def save(self) -> None:
        """Пишет файл истории целиком; каталог состояния создаётся при первой записи.

        Ошибка записи диалог не ломает (сбой хранилища не должен отменять ответ), но и не
        молчит: причина остаётся в `last_error`, и интерфейс сообщает о ней строкой журнала.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": self.summary,
                        "summary_covers": self.summary_covers,
                        "facts": self.facts,
                        "working": self.working,
                        "dialogues": self.dialogues,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.last_error = None
        except OSError as error:
            self.last_error = str(error)

    def count(self) -> int:
        return len(self.dialogues)

    def total_usage(self) -> SessionUsage:
        """Расход всей сохранённой истории: сумма метрик по записям файла.

        Записей в файле теперь неограниченно — итог честный расход всего диалога.
        """
        return sum_usage(self.dialogues)
