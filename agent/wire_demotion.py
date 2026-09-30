"""Key-preserving wire demotion of old tool results, for providers without prompt caching.

On a route that caches nothing, every request re-sends the whole conversation, and
old tool results dominate the bill. This rewrites the REQUEST copy only (via the
context engine's per-request ``select_context`` hook): tool results older than the
last ``keep_last_messages`` messages, and longer than ``min_result_chars``, are replaced
by a stub that keeps what a later step may need to cite:

  - the tool name and a short form of its arguments, so the model can re-run it;
  - the original size;
  - the result's *evidence keys*: UUIDs, 8+ hex ids (record ids, commit SHAs),
    absolute paths and long numbers. Those are version-addressed handles, so they
    stay valid; the content behind them is re-fetched, never paraphrased.

Persisted history is never touched. On a caching provider this is a bad trade (every
demotion changes the prefix and forfeits the cache), so it is opt-in per profile via
``compression.wire_demotion``. Measured basis: knowfleet task #564 replay, record
edc31437. Over 6 recorded runs this shape cut input ~44% with 1/57 dangling citations
in final writes, against 10/57 for demotion without keys.

Mutable state (task/investigation status, live host state, current file contents) is
exactly what a stub must NOT stand in for; the stub text says to re-run the call.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

EVIDENCE_KEY_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"  # UUID
    r"|\b[0-9a-f]{8,40}\b"                                              # record ids, SHAs
    r"|(?<![\w.])/(?:Users|home|private|opt|etc|var|tmp|srv|mnt)/[\w@.+/-]+"  # absolute paths
    r"|\b\d{5,}\b"                                                      # long numbers
)
MAX_KEYS = 40
ARG_PREVIEW_CHARS = 160
DEMOTED_MARKER = "[wire-demoted]"
# Tool results a stub must never replace: skill bodies are instructions the model is
# still following, and a stub would silently drop them (same concern as #32106).
NEVER_DEMOTE_TOOLS = frozenset({"skill_view"})


def evidence_keys(text: str, limit: int = MAX_KEYS) -> List[str]:
    """Distinct evidence keys in first-seen order, capped at ``limit``."""
    seen: Dict[str, None] = {}
    for match in EVIDENCE_KEY_RE.finditer(text or ""):
        key = match.group(0).rstrip(".,;:)")
        if key and key not in seen:
            seen[key] = None
            if len(seen) >= limit:
                break
    return list(seen)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _call_index(messages: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """tool_call_id -> (tool name, argument preview), unwrapping the tool_call bridge."""
    index: Dict[str, Tuple[str, str]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name, args = fn.get("name") or "?", fn.get("arguments") or ""
            if name == "tool_call":
                try:
                    inner = (json.loads(args) or {}).get("calls") or []
                    if inner and isinstance(inner[0], dict):
                        name = inner[0].get("name") or name
                        args = json.dumps(inner[0].get("arguments") or {}, sort_keys=True)
                except (ValueError, AttributeError):
                    pass
            if call.get("id"):
                index[call["id"]] = (name, args[:ARG_PREVIEW_CHARS])
    return index


def stub_for(name: str, args_preview: str, original: str) -> str:
    keys = evidence_keys(original)
    keys_part = ", ".join(keys) if keys else "none"
    return (f"{DEMOTED_MARKER} Earlier result of {name}({args_preview}) — {len(original)} chars, removed from "
            f"this request to save tokens. Evidence keys it contained: {keys_part}. If you need anything "
            f"from it beyond these keys, call the tool again; do not rely on memory for its content.")


def demote_request(messages: List[Dict[str, Any]], *, keep_last_messages: int = 6,
                   min_result_chars: int = 500) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, int]]:
    """Return (new request list or None if nothing changed, stats). Never mutates ``messages``."""
    stats = {"demoted": 0, "chars_before": 0, "chars_after": 0}
    if not messages:
        return None, stats
    boundary = len(messages) - max(0, int(keep_last_messages))
    calls = _call_index(messages)
    out: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= boundary or msg.get("role") != "tool":
            out.append(msg)
            continue
        text = _content_text(msg.get("content"))
        name, args = calls.get(msg.get("tool_call_id") or "", (msg.get("name") or "?", ""))
        if len(text) < min_result_chars or text.startswith(DEMOTED_MARKER) or name in NEVER_DEMOTE_TOOLS:
            out.append(msg)
            continue
        stub = stub_for(name, args, text)
        new_msg = dict(msg)
        new_msg["content"] = stub
        out.append(new_msg)
        stats["demoted"] += 1
        stats["chars_before"] += len(text)
        stats["chars_after"] += len(stub)
    return (out if stats["demoted"] else None), stats


def ab_arm(session_id: str) -> str:
    """Deterministic 50/50 arm for A/B runs: 'on' or 'off', stable per session id."""
    digest = hashlib.sha256((session_id or "").encode()).digest()
    return "on" if digest[0] % 2 == 0 else "off"
