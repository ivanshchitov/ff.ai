"""Редьюсер панели веток диалога для /branches."""

from ui import branches_screen, keyboard
from ui.branches_screen import initial_state, apply_key


BRANCHES = (("ветка 1", 3), ("ветка 2", 1))
ACTIVE = "ветка 1"


def test_panel_lists_branches_and_marks_the_active_one():
    state = initial_state(BRANCHES, ACTIVE)
    assert state.branches == BRANCHES
    assert state.active == ACTIVE
    assert state.selected_index == 0


def test_cursor_starts_on_the_active_branch():
    state = initial_state(BRANCHES, "ветка 2")
    assert state.selected_index == 1


def test_down_and_up_move_selection_with_wraparound():
    state = apply_key(initial_state(BRANCHES, ACTIVE), keyboard.DOWN)
    assert state.selected_index == 1
    state = apply_key(state, keyboard.UP)
    assert state.selected_index == 0
    state = apply_key(state, keyboard.UP)  # с нуля вверх — на последнюю
    assert state.selected_index == len(BRANCHES) - 1
    state = apply_key(state, keyboard.DOWN)  # с последней вниз — на первую
    assert state.selected_index == 0


def test_enter_switches_to_the_selected_branch():
    state = apply_key(apply_key(initial_state(BRANCHES, ACTIVE), keyboard.DOWN), keyboard.ENTER)
    assert state.switched
    assert state.selected_name == "ветка 2"


def test_checkpoint_and_new_branch_are_separate_outcomes():
    state = apply_key(initial_state(BRANCHES, ACTIVE), branches_screen.KEY_CHECKPOINT)
    assert state.checkpoint_requested
    assert not state.switched
    assert not state.cancelled

    state = apply_key(initial_state(BRANCHES, ACTIVE), branches_screen.KEY_NEW_BRANCH)
    assert state.new_branch_requested
    assert not state.checkpoint_requested


def test_esc_cancels_the_panel():
    state = apply_key(initial_state(BRANCHES, ACTIVE), keyboard.ESC)
    assert state.cancelled
    assert not state.switched
    assert not state.checkpoint_requested
    assert not state.new_branch_requested


def test_enter_on_the_active_branch_keeps_it():
    state = apply_key(initial_state(BRANCHES, ACTIVE), keyboard.ENTER)
    assert state.switched
    assert state.selected_name == ACTIVE


def test_keys_after_an_outcome_do_not_change_state():
    for outcome_key in (keyboard.ENTER, branches_screen.KEY_CHECKPOINT, branches_screen.KEY_NEW_BRANCH):
        state = apply_key(initial_state(BRANCHES, ACTIVE), outcome_key)
        assert apply_key(state, keyboard.DOWN) == state
        assert apply_key(state, keyboard.ESC) == state

    cancelled = apply_key(initial_state(BRANCHES, ACTIVE), keyboard.ESC)
    assert apply_key(cancelled, branches_screen.KEY_NEW_BRANCH) == cancelled


def test_unrecognized_keys_do_not_change_state():
    state = initial_state(BRANCHES, ACTIVE)
    for key in (keyboard.LEFT, keyboard.RIGHT, keyboard.BACKSPACE, "5", "x"):
        assert apply_key(state, key) == state


def test_panel_of_a_single_branch_works():
    state = initial_state((("ветка 1", 0),), "ветка 1")
    assert state.selected_name == "ветка 1"
    assert apply_key(state, keyboard.DOWN).selected_index == 0
