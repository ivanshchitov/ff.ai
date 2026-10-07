"""Состояние панели выбора команды `/commands`, отделённое от терминала.

Схема та же, что у `/settings` и `/models`: вся реакция на клавиши живёт здесь — в чистом
редукторе, тестируемом без псевдотерминала; `ui/tui_app.py` остаётся только цикл
read-key/redraw и выполнение выбранной команды тем же диспетчером, что и ручной набор.
"""

from dataclasses import dataclass, replace
from typing import List, Tuple

from . import keyboard

# Единственный источник списка команд: из него выводится и автодополнение в tui_app,
# и содержимое панели. Порядок — от самой частой к самой редкой.
COMMAND_OPTIONS: List[Tuple[str, str]] = [
    ("/exit", "выйти из приложения (история сохраняется)"),
    ("/commands", "показать эту панель команд"),
    ("/settings", "настройки ответа: формат, объём, стратегия контекста"),
    ("/models", "выбрать модель для ответов"),
    ("/clear", "очистить диалог (память домена не трогает)"),
    ("/usage", "расход токенов: последний запрос, сессия, вся история"),
    ("/context", "состояние контекста: стратегия, память стратегии, ветки"),
    ("/branches", "ветки диалога: переключить, чекпоинт, новая ветка"),
    ("/domain", "активный домен: область применимости, версии, маркеры"),
    ("/docs", "документация портала: режим, версия, отчёт о поиске"),
    ("/code", "корпус кода: сборка индекса, состояние, сравнение стратегий"),
    ("/memory", "память: слои, цель, запомнить и забыть"),
    ("/profile", "профиль: настройка, выбор и удаление"),
    ("/invariants", "правила домена и результат последней проверки"),
    ("/schedule", "расписание: задания, прогоны и накопленное"),
    ("/mcp", "MCP-серверы: подключения, инструменты, обновление отчёта"),
    ("/tool", "инструменты MCP: перечень и ручной вызов"),
]


@dataclass(frozen=True)
class CommandsScreenState:
    """Снимок панели команд: курсор на одной из команд и исход подтверждения."""

    selected_index: int = 0
    confirmed: bool = False
    cancelled: bool = False

    @property
    def selected(self) -> Tuple[str, str]:
        return COMMAND_OPTIONS[self.selected_index]


def initial_state() -> CommandsScreenState:
    return CommandsScreenState(selected_index=0)


def apply_key(state: CommandsScreenState, key: str) -> CommandsScreenState:
    if state.confirmed or state.cancelled:
        return state
    if key == keyboard.ENTER:
        return replace(state, confirmed=True)
    if key == keyboard.ESC:
        return replace(state, cancelled=True)
    if key in (keyboard.UP, keyboard.DOWN):
        step = -1 if key == keyboard.UP else 1
        return replace(
            state, selected_index=(state.selected_index + step) % len(COMMAND_OPTIONS)
        )
    return state
