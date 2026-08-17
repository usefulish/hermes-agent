"""`pre_verify` reaches turns whose writes went through the shell.

`pre_verify` fires only when `write_file`/`patch` recorded a mutation path, so
a turn that wrote to the tree through `terminal` never reaches the hook at all
and a listener tracking an obligation across the turn is silently skipped. The
opt-in `agent.pre_verify_on_shell_turn` widens the trigger without asserting
that a shell turn edited anything: `changed_paths` stays empty, and the
listener decides.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES,
    SHELL_TOOL_NAMES,
)
from agent.verify_hooks import pre_verify_on_shell_turn
from run_agent import AIAgent


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            session_id="pre-verify-shell-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._turn_failed_file_mutations = {}
    instance._turn_file_mutation_paths = set()
    instance._turn_shell_tool_used = False
    return instance


def _terminal_result(output=""):
    return SimpleNamespace(output=output)


def test_shell_tools_are_not_file_mutating_tools():
    # The whole point: nothing here claims a shell command wrote a file.
    assert not (SHELL_TOOL_NAMES & FILE_MUTATING_TOOL_NAMES)
    assert "terminal" in SHELL_TOOL_NAMES
    assert "execute_code" in SHELL_TOOL_NAMES


def test_successful_shell_call_marks_the_turn_without_inventing_a_path(agent):
    agent._record_file_mutation_result(
        "terminal", {"command": "echo hi > a.txt"}, _terminal_result(), False
    )
    assert agent._turn_shell_tool_used is True
    assert agent._turn_file_mutation_paths == set(), (
        "a shell command's effects are unknown; recording a path would be a lie"
    )


def test_failed_shell_call_does_not_mark_the_turn(agent):
    agent._record_file_mutation_result(
        "terminal", {"command": "false"}, _terminal_result(), True
    )
    assert agent._turn_shell_tool_used is False


def test_non_shell_tools_leave_the_flag_alone(agent):
    agent._record_file_mutation_result("read_file", {"path": "a.txt"}, None, False)
    assert agent._turn_shell_tool_used is False


def test_flag_is_reset_beside_the_other_per_turn_verifier_state():
    """The flag must be cleared per turn or one shell call arms every later turn.

    The reset lives deep inside build_turn_context, which needs far more
    scaffolding than this invariant is worth, so this is a structural check:
    the two per-turn verifier fields must be reset together. If someone moves
    one, this fails and points at the other.
    """
    import inspect

    from agent import turn_context

    source = inspect.getsource(turn_context)
    assert "agent._turn_file_mutation_paths = set()\n    agent._turn_shell_tool_used = False" in source


def test_config_flag_defaults_to_off_and_reads_truthy_values():
    assert pre_verify_on_shell_turn({}) is False
    assert pre_verify_on_shell_turn({"agent": {}}) is False
    assert pre_verify_on_shell_turn({"agent": {"pre_verify_on_shell_turn": False}}) is False
    assert pre_verify_on_shell_turn({"agent": {"pre_verify_on_shell_turn": True}}) is True
    assert pre_verify_on_shell_turn({"agent": {"pre_verify_on_shell_turn": "yes"}}) is True
