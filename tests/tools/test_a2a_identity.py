"""Regression coverage for outbound A2A identity provenance."""

from plugins.platforms.a2a import adapter, tools


def test_outbound_identity_uses_active_profile_and_short_hostname(monkeypatch):
    monkeypatch.setattr(adapter, "_active_profile_name", lambda: "librarian")
    monkeypatch.setattr("socket.gethostname", lambda: "kimchi.local")

    assert adapter.outbound_identity() == "librarian-kimchi"


def test_identity_header_uses_outbound_identity(monkeypatch):
    monkeypatch.setattr(adapter, "outbound_identity", lambda: "librarian-kimchi")

    assert tools._identity_headers() == {"X-A2A-Identity": "librarian-kimchi"}
