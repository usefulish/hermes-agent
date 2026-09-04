"""Tests: a2a_agents auth token resolution supports env references (fleet #253).

Config may hold secret bearer tokens either literally (legacy) or by env-var
name via ``token_env`` (the config's key_env pattern). A ``token_env`` value
is resolved at call time so a rotation takes effect without a config edit or
process reload. Missing env or a bare token_env with no value must fail safe
(no Authorization header) rather than send an empty bearer.
"""

from __future__ import annotations

from plugins.platforms.a2a.tools import _auth_header


def test_literal_token_backward_compatible():
    """Legacy shape unchanged: a literal bearer token still works."""
    assert _auth_header({"type": "bearer", "token": "sk-literal"}) == {
        "Authorization": "Bearer sk-literal"
    }


def test_token_env_resolves_from_environment(monkeypatch):
    """token_env reads the named env var at call time."""
    monkeypatch.setenv("FLEET_A2A_TOKEN", "env-token-value")
    assert _auth_header({"type": "bearer", "token_env": "FLEET_A2A_TOKEN"}) == {
        "Authorization": "Bearer env-token-value"
    }


def test_token_env_wins_over_literal(monkeypatch):
    """When both are present, token_env takes precedence (secrets out of config)."""
    monkeypatch.setenv("FLEET_A2A_TOKEN", "env-wins")
    assert _auth_header(
        {"type": "bearer", "token": "sk-literal", "token_env": "FLEET_A2A_TOKEN"}
    ) == {"Authorization": "Bearer env-wins"}


def test_token_env_missing_environment_fails_safe(monkeypatch):
    """An unset token_env sends NO auth header — never an empty bearer."""
    monkeypatch.delenv("FLEET_A2A_TOKEN", raising=False)
    assert _auth_header({"type": "bearer", "token_env": "FLEET_A2A_TOKEN"}) == {}


def test_no_auth_returns_empty():
    """Empty or non-bearer auth sends nothing."""
    assert _auth_header({}) == {}
    assert _auth_header(None) == {}
    assert _auth_header({"type": "api_key", "key": "x"}) == {}


def test_token_env_resolves_via_secret_scope(monkeypatch):
    """In a multiplexed gateway the value lives in the profile secret scope,
    not os.environ — _resolve_env_secret must use get_secret (fleet #253)."""
    # Simulate multiplex: value NOT in os.environ, but get_secret returns it
    monkeypatch.delenv("FLEET_A2A_TOKEN", raising=False)
    from plugins.platforms.a2a import tools as a2a_tools
    monkeypatch.setattr(
        a2a_tools, "_resolve_env_secret",
        lambda name: "scope-value" if name == "FLEET_A2A_TOKEN" else None,
    )
    assert a2a_tools._auth_header(
        {"type": "bearer", "token_env": "FLEET_A2A_TOKEN"}
    ) == {"Authorization": "Bearer scope-value"}
