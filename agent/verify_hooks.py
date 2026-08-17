"""Verification-loop helpers for the ``pre_verify`` round-end gate.

When the agent has edited code and is about to verify/finish, the loop fires the
``pre_verify`` hook (user directives resolved by
:func:`hermes_cli.plugins.get_pre_verify_continue_message`). A directive keeps
the agent going one more turn — run a check, defer it, tidy the diff — instead of
stopping immediately.

The shipped coding guidance lives on the evidence-based verification-stop nudge
(``agent/verification_stop.py``), not as a second default stop gate. That keeps
the default token cost tied to the existing "missing verification evidence"
decision while preserving ``pre_verify`` for user/plugin policy.
"""

from __future__ import annotations

from typing import Any, Optional

from utils import is_truthy_value

DEFAULT_MAX_VERIFY_NUDGES = 3

# Shipped guidance appended to the verification-stop nudge when code lacks fresh
# verification evidence. Wording mirrors the user-facing "clean your work"
# workflow, but does not create its own extra model turn.
CODING_VERIFY_GUIDANCE = (
    "[Coding] Before you run tests/linters or call this done: if this is "
    "creative UI/visual work, hold off on tests and linters until the user says "
    "they like the result or you're about to commit. And before every commit, "
    "clean your work: keep it KISS/DRY, match the surrounding code style, and be "
    "elitist, shorthand, clever, concise, efficient, and elegant."
)


def max_verify_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_verify`` continue directives per turn (>= 0)."""
    agent_cfg = _agent_cfg(config)
    raw = agent_cfg.get("max_verify_nudges")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_VERIFY_NUDGES


def pre_verify_on_shell_turn(config: Optional[dict[str, Any]] = None) -> bool:
    """Also fire ``pre_verify`` on a turn that ran a shell tool. Default off.

    ``pre_verify`` normally fires only when ``write_file``/``patch`` mutated
    something. That misses a turn whose writes went through the shell: the
    tree changed, but nothing recorded a path, so a listener tracking an
    obligation across the turn is never asked about it.

    Detecting *which* shell commands wrote is not possible without parsing
    them, and parsing is the wrong place for that judgement — the listener
    already holds whatever state it cares about and can decline in a
    microsecond. So this widens the trigger rather than trying to be clever,
    and it is opt-in: enabling it means "ask my pre_verify hooks after shell
    turns too", not "assume a shell turn edited code". Existing listeners that
    equate being called with code having changed are unaffected while it is
    off.
    """
    return is_truthy_value(
        _agent_cfg(config).get("pre_verify_on_shell_turn", False), default=False
    )


def coding_verify_guidance(config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return the optional guidance appended to verification-stop nudges."""
    if not is_truthy_value(_agent_cfg(config).get("verify_guidance", True), default=True):
        return None
    return CODING_VERIFY_GUIDANCE


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


__all__ = [
    "CODING_VERIFY_GUIDANCE",
    "DEFAULT_MAX_VERIFY_NUDGES",
    "coding_verify_guidance",
    "max_verify_nudges",
    "pre_verify_on_shell_turn",
]
