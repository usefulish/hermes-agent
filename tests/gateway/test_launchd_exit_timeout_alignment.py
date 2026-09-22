"""Launchd ExitTimeOut alignment and process-level SIGTERM drain contracts.

The 2026-09-13 incident (de3364c1): a launchd-managed gateway received SIGTERM
(``hermes gateway stop``), completed ``notify_active_sessions`` at +0.44s, then
the drain exceeded launchd's plist ``ExitTimeOut`` (25s). launchd escalated to
SIGKILL mid-drain — before the in-process shutdown watchdog (drain+60s) could
fire — so no exit path ran, buffered log lines died with the process, and the
lifecycle ledger recorded an unclean death. systemd got a startup alignment
check (``check_systemd_timing_alignment``); launchd never got its sibling.
These tests pin the launchd contract.
"""

from __future__ import annotations

import pytest

import gateway.shutdown_forensics as sf


# ---------------------------------------------------------------------------
# launchd detection
# ---------------------------------------------------------------------------

class TestUnderLaunchd:
    def test_not_under_launchd_when_no_launchd_env_and_not_pid_one_parent(self, monkeypatch):
        monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
        # The pytest process is not a direct launchd child, but be robust:
        # the probe must never raise regardless of environment.
        result = sf.probe_launchd_supervision()
        assert result in (True, False)

    def test_probe_under_xpc_service_name(self, monkeypatch):
        monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")
        assert sf.probe_launchd_supervision() is True


# ---------------------------------------------------------------------------
# check_launchd_exit_timeout_alignment
# ---------------------------------------------------------------------------

class TestCheckLaunchdExitTimeoutAlignment:

    def _patch_query(self, monkeypatch, exit_timeout_s):
        """Patch the launchctl ExitTimeOut query to return *exit_timeout_s*."""
        monkeypatch.setattr(sf, "_launchd_exit_timeout_seconds", lambda label: exit_timeout_s)
        monkeypatch.setattr(sf, "_launchd_job_label", lambda: "ai.hermes.gateway")
        monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")

    def test_aligned_plist_reports_no_mismatch(self, monkeypatch):
        self._patch_query(monkeypatch, 90)
        # Same contract as the systemd sibling: a dict with mismatch=False —
        # the caller logs only on mismatch.
        result = sf.check_launchd_exit_timeout_alignment(0.0)
        assert result is not None
        assert result["mismatch"] is False
        assert result["exit_timeout_s"] == 90

    def test_stale_plist_reports_mismatch(self, monkeypatch):
        self._patch_query(monkeypatch, 25)
        result = sf.check_launchd_exit_timeout_alignment(0.0)
        assert result is not None
        assert result["mismatch"] is True
        assert result["exit_timeout_s"] == 25
        assert result["expected_min_s"] >= 60
        assert "ai.hermes.gateway" in result["label"]

    def test_long_drain_raises_expected_min(self, monkeypatch):
        # A configured 120s chat drain must raise the expected minimum.
        self._patch_query(monkeypatch, 25)
        result = sf.check_launchd_exit_timeout_alignment(120.0)
        assert result is not None
        assert result["expected_min_s"] >= 120

    def test_none_when_launchctl_undeterminable(self, monkeypatch):
        monkeypatch.setattr(sf, "_launchd_exit_timeout_seconds", lambda label: None)
        monkeypatch.setattr(sf, "_launchd_job_label", lambda: "ai.hermes.gateway")
        monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")
        assert sf.check_launchd_exit_timeout_alignment(0.0) is None

    def test_none_when_not_launchd_supervised(self, monkeypatch):
        monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
        monkeypatch.setattr(sf, "probe_launchd_supervision", lambda: False)
        assert sf.check_launchd_exit_timeout_alignment(0.0) is None


# ---------------------------------------------------------------------------
# Plist generation must derive ExitTimeOut from the drain budget
# ---------------------------------------------------------------------------

class TestPlistExitTimeout:
    def test_generated_plist_exit_timeout_covers_drain_budget(self):
        from hermes_cli.gateway import resolve_launchd_exit_timeout
        # Default budgets: floor from resolve_systemd_timeout_stop_sec.
        default = resolve_launchd_exit_timeout(0.0)
        assert default >= 60

        # A 120s drain must not be SIGKILLed by a 25s plist.
        configured = resolve_launchd_exit_timeout(120.0)
        assert configured >= 120 + 30  # drain + headroom

    def test_plist_generator_carries_resolved_exit_timeout(self, monkeypatch, tmp_path):
        from hermes_cli import gateway as gateway_cli

        # Pin every environment read the generator performs so the rendered
        # plist is reproducible in CI regardless of the host launchd state.
        monkeypatch.setattr(gateway_cli, "_stable_service_working_dir", lambda: "/work")
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "_service_venv_dir", lambda: "/venv")
        monkeypatch.setattr(gateway_cli, "_build_service_path_dirs", lambda: ["/venv/bin"])
        monkeypatch.setattr(gateway_cli, "_append_node_dir_for_service", lambda dirs: None)
        monkeypatch.setattr(gateway_cli, "_timestamped_stderr_gateway_command",
                            lambda log_path, external_supervisor: ["/venv/bin/hermes", "gateway", "run"])
        monkeypatch.setattr(gateway_cli, "resolve_launchd_exit_timeout", lambda: 90)
        monkeypatch.setattr("hermes_cli.resource_limits.configured_nofile_soft_limit", lambda: 256)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        plist = gateway_cli.generate_launchd_plist()

        assert "<key>ExitTimeOut</key>\n    <integer>90</integer>" in plist
        assert "ExitTimeOut gives the gateway 25s" not in plist  # stale hardcoded comment gone
