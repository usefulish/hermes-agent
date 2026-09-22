"""Process-level SIGTERM drain regression (task #390, incident de3364c1).

A launchd-supervised gateway was SIGKILLed mid-drain because the plist's
``ExitTimeOut`` (25s) sat far below the configured stop budget (drain 180s
+ cleanup + headroom). The kill landed before any in-process backstop could
fire, so no exit path ran and the ledger recorded an unclean death.

This test spawns a REAL gateway process (temp ``HERMES_HOME``, one multiplexed
profile, no messaging platforms — no credentials needed), waits for readiness,
sends SIGTERM, and pins the contract the supervisor budget depends on: the
whole teardown — every ``_stop_*`` phase through SessionDB close — completes
well inside the ExitTimeOut the plist now derives, and the process exits on
its own (no external kill needed).

The temp home mirrors the incident's shape: a multiplexed gateway (default +
one secondary profile) with cron ticking both.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The test spawns and always reaps its own gateway child (kill in finally), the
# documented pattern for the conftest live-system guard.
pytestmark = [pytest.mark.macos_only, pytest.mark.spawns_gateway_lookalike]

# Boot allowance: the manual reproduction reaches readiness in ~6s; MCP
# discovery and tool-schema warm-up can stretch this on a busy CI runner.
_READINESS_TIMEOUT_S = 90.0
# Drain allowance: healthy teardown observed at 0.02s; the derived launchd
# ExitTimeOut floor is 60s, so anything past this is the incident class.
_DRAIN_WALL_BUDGET_S = 60.0


def _spawn_gateway(home: Path) -> subprocess.Popen:
    (home / "profiles" / "coder").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\ngateway:\n  multiplex_profiles: true\n"
    )
    (home / "profiles" / "coder" / "config.yaml").write_text(
        "model:\n  default: test-model\n"
    )
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    return subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", "gateway", "run"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_ready(gateway_log: Path, deadline: float) -> None:
    """Block until the gateway prints its readiness line (or fail with the log tail)."""
    marker = "Press Ctrl+C to stop"
    while time.monotonic() < deadline:
        if gateway_log.exists() and marker in gateway_log.read_text(encoding="utf-8", errors="replace"):
            return
        if gateway_log.exists() and "Gateway stopped" in gateway_log.read_text(
            encoding="utf-8", errors="replace"
        ):
            pytest.fail("gateway exited before reaching readiness:\n" + _tail(gateway_log))
        time.sleep(0.25)
    pytest.fail(f"gateway did not become ready within {_READINESS_TIMEOUT_S}s:\n" + _tail(gateway_log))


def _tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return f"(no log at {path})"


@pytest.mark.macos_only
def test_sigterm_drain_completes_within_supervisor_budget(tmp_path):
    from hermes_cli.gateway import resolve_launchd_exit_timeout

    home = tmp_path / "hermes-home"
    home.mkdir()
    proc = _spawn_gateway(home)
    try:
        _wait_for_ready(home / "logs" / "gateway.log", time.monotonic() + _READINESS_TIMEOUT_S)

        exit_timeout = resolve_launchd_exit_timeout(0.0)
        assert exit_timeout >= 60  # the budget the plist grants must be the floor

        sent_at = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        # SIGTERM-initiated shutdown without a restart request exits 1 so the
        # service manager revives the gateway (issue #42675).
        returncode = proc.wait(timeout=exit_timeout + 30.0)
        drained_s = time.monotonic() - sent_at

        log = (home / "logs" / "gateway.log").read_text(encoding="utf-8", errors="replace")
        assert returncode == 1, f"unexpected exit {returncode}; log tail:\n{_tail(home / 'logs' / 'gateway.log')}"
        # The teardown completed on its own — no supervisor SIGKILL involved.
        assert "Shutdown phase: notify_active_sessions done" in log
        assert "Shutdown phase: drain done" in log
        assert "Shutdown phase: SessionDB close done" in log
        assert "Gateway stopped (total teardown" in log
        # The incident class: a drain that outlives the supervisor budget.
        assert drained_s < _DRAIN_WALL_BUDGET_S < exit_timeout, (
            f"drain took {drained_s:.1f}s vs supervisor budget {exit_timeout}s"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10.0)
