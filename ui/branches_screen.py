"""Состояние панели веток диалога /branches, отделённое от терминала.

Схема та же, что у `/settings` и `/models`: вся реакция на клавиши живёт здесь — в чистом
редьюсере, тестируемом без псевдотерминала; `ui/tui_app.py` остаётся только цикл
read-key/redraw и выполнение исхода (переключить ветку, поставить чекпоинт, создать ветку).

Две клавиши панели не являются строками списка: `c` ставит чекпоинт, `n` создаёт ветку от
него. Строками их выражать нечем — это действия над диалогом, а не выбор из списка.
"""

from dataclasses import dataclass, replace
from typing import Tuple

from . import keyboard

# Клавиши-действия панели: чекпоинт в активной ветке и новая ветка от него.
KEY_CHECKPOINT = "c"
KEY_NEW_BRANCH = "n"

Branches = Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class BranchesScreenState:
    """Снимок панели веток: список, курсор, активная ветка и исход нажатия.

    Исходы взаимоисключающие: либо переключение на выбранную ветку, либо запрос чекпоинта,
    либо создание ветки, либо отмена — по ним `ui/tui_app.py` и решает, что сделать.
    """

    branches: Branches = ()
    active: str = ""
    selected_index: int = 0
    switched: bool = False
    checkpoint_requested: bool = False
    new_branch_requested: bool = False
    cancelled: bool = False

    @property
    def finished(self) -> bool:
        return (
            self.switched
            or self.checkpoint_requested
            or self.new_branch_requested
            or self.cancelled
        )

    @property
    def selected_name(self) -> str:
        if not self.branches:
            return ""
        return self.branches[self.selected_index][0]


def initial_state(branches: Branches, active: str) -> BranchesScreenState:
    """Курсор стартует на активной ветке; пустая панель остаётся работоспособной."""
    names = [name for name, _ in branches]
    index = names.index(active) if active in names else 0
    return BranchesScreenState(branches=tuple(branches), active=active, selected_index=index)


def apply_key(state: BranchesScreenState, key: str) -> BranchesScreenState:
    if state.finished:
        return state
    if key == keyboard.ENTER:
        return replace(state, switched=True)
    if key == keyboard.ESC:
        return replace(state, cancelled=True)
    if key == KEY_CHECKPOINT:
        return replace(state, checkpoint_requested=True)
    if key == KEY_NEW_BRANCH:
        return replace(state, new_branch_requested=True)
    if key in (keyboard.UP, keyboard.DOWN) and state.branches:
        step = -1 if key == keyboard.UP else 1
        return replace(
            state, selected_index=(state.selected_index + step) % len(state.branches)
        )
    return state
