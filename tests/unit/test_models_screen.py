"""Редьюсер панели выбора модели /models."""

from core import config
from ui import keyboard
from ui.models_screen import ModelSelectionState, initial_state, apply_key


def test_panel_lists_all_available_models():
    assert len(initial_state(config.DEFAULT_MODEL).available) == len(config.AVAILABLE_MODELS)
    assert initial_state(config.DEFAULT_MODEL).selected == config.AVAILABLE_MODELS[0]


def test_initial_state_puts_cursor_on_current_model():
    state = initial_state("kimi-k3")
    assert state.selected == "kimi-k3"
    assert state.current == "kimi-k3"
    assert not state.confirmed
    assert not state.cancelled


def test_down_and_up_move_selection_with_wraparound():
    state = apply_key(initial_state(config.DEFAULT_MODEL), keyboard.DOWN)
    assert state.selected == config.AVAILABLE_MODELS[1]
    state = apply_key(state, keyboard.UP)
    assert state.selected == config.AVAILABLE_MODELS[0]
    state = apply_key(state, keyboard.UP)  # с нуля вверх — на последнюю
    assert state.selected == config.AVAILABLE_MODELS[-1]
    state = apply_key(state, keyboard.DOWN)  # с последней вниз — на первую
    assert state.selected == config.AVAILABLE_MODELS[0]


def test_enter_confirms_selected_model():
    state = apply_key(initial_state("glm-5.3-flash"), keyboard.DOWN)
    state = apply_key(state, keyboard.ENTER)
    assert state.confirmed
    assert not state.cancelled
    assert state.selected == "mimo-v2.5-free"


def test_esc_cancels_panel():
    state = apply_key(initial_state("glm-5.3-flash"), keyboard.ESC)
    assert state.cancelled
    assert not state.confirmed


def test_keys_after_finish_do_not_change_state():
    confirmed = apply_key(initial_state(config.DEFAULT_MODEL), keyboard.ENTER)
    cancelled = apply_key(initial_state(config.DEFAULT_MODEL), keyboard.ESC)
    for state in (confirmed, cancelled):
        assert apply_key(state, keyboard.DOWN) == state
        assert apply_key(state, keyboard.UP) == state
        assert apply_key(state, keyboard.ENTER) == state
        assert apply_key(state, keyboard.ESC) == state


def test_unrecognized_keys_do_not_change_state():
    state = initial_state(config.DEFAULT_MODEL)
    for key in (keyboard.LEFT, keyboard.RIGHT, keyboard.BACKSPACE, "5", "x"):
        assert apply_key(state, key) == state


def test_state_is_immutable_dataclass():
    state = ModelSelectionState()
    assert state.selected_index == 0
    assert not state.confirmed
    assert not state.cancelled
