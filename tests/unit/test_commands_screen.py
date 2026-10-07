"""Редьюсер панели команд: список команд и реакция на клавиши."""

from ui import commands_screen, keyboard


def test_command_list_is_well_formed():
    commands = [command for command, _ in commands_screen.COMMAND_OPTIONS]
    assert commands[0] == "/exit"
    assert len(commands) == len(set(commands))
    for command, description in commands_screen.COMMAND_OPTIONS:
        assert command.startswith("/")
        assert description.strip()


def test_initial_state_starts_at_the_first_command():
    state = commands_screen.initial_state()
    assert state.selected == commands_screen.COMMAND_OPTIONS[0]
    assert not state.confirmed and not state.cancelled


def test_arrows_move_and_wrap():
    count = len(commands_screen.COMMAND_OPTIONS)
    state = commands_screen.initial_state()
    assert commands_screen.apply_key(state, keyboard.DOWN).selected_index == 1
    assert commands_screen.apply_key(state, keyboard.UP).selected_index == count - 1


def test_enter_confirms_and_esc_cancels():
    state = commands_screen.initial_state()
    assert commands_screen.apply_key(state, keyboard.ENTER).confirmed
    assert commands_screen.apply_key(state, keyboard.ESC).cancelled


def test_finished_state_ignores_further_keys():
    state = commands_screen.apply_key(commands_screen.initial_state(), keyboard.ENTER)
    assert commands_screen.apply_key(state, keyboard.DOWN).selected_index == state.selected_index
