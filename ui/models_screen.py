"""Состояние панели выбора модели /models, отделённое от терминала.

Схема та же, что у `/commands` и `/branches`: вся реакция на клавиши живёт здесь —
в чистом редьюсере, тестируемом без псевдотерминала; `ui/tui_app.py` остаётся только
цикл read-key/redraw и применение выбранной модели к сессии.
"""

from dataclasses import dataclass, replace
from typing import List

from core import config

from . import keyboard


@dataclass(frozen=True)
class ModelSelectionState:
    """Снимок панели выбора модели: курсор, текущая модель и исход подтверждения.

    `current` — модель, активная в сессии на момент открытия панели: она нужна отрисовке
    для пометки строки «текущая», отдельно от курсора `selected_index`.
    """

    selected_index: int = 0
    current: str = ""
    confirmed: bool = False
    cancelled: bool = False

    @property
    def available(self) -> List[str]:
        return config.AVAILABLE_MODELS

    @property
    def selected(self) -> str:
        return config.AVAILABLE_MODELS[self.selected_index]


def initial_state(current_model: str) -> ModelSelectionState:
    """Курсор стартует на текущей модели сессии; неизвестное имя — на дефолтную позицию."""
    index = config.AVAILABLE_MODELS.index(current_model) if current_model in config.AVAILABLE_MODELS else 0
    return ModelSelectionState(selected_index=index, current=current_model)


def apply_key(state: ModelSelectionState, key: str) -> ModelSelectionState:
    if state.confirmed or state.cancelled:
        return state
    if key == keyboard.ENTER:
        return replace(state, confirmed=True)
    if key == keyboard.ESC:
        return replace(state, cancelled=True)
    if key in (keyboard.UP, keyboard.DOWN):
        step = -1 if key == keyboard.UP else 1
        return replace(
            state, selected_index=(state.selected_index + step) % len(config.AVAILABLE_MODELS)
        )
    return state
