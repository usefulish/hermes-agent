"""Key-preserving wire demotion of old tool results, for providers without prompt caching.

On a route that caches nothing, every request re-sends the whole conversation, and
old tool results dominate the bill. This rewrites the REQUEST copy only (via the
context engine's per-request ``select_context`` hook): cold tool results older than
the last ``keep_last_messages`` messages, and longer than ``min_result_chars``, are
replaced by a stub that keeps what a later step may need to cite:

  - the tool name and a short form of its arguments, so the model can re-run it;
  - the original size;
  - the result's *evidence keys*: UUIDs, 8+ hex ids (record ids, commit SHAs),
    absolute paths and long numbers. Those are version-addressed handles, so they
    stay valid; the content behind them is re-fetched, never paraphrased.

v2 (after the first live run, knowfleet incident f1778a10): demoting the WORKING SET
makes the model re-fetch it on every turn. With a 6-message tail, the record under
investigation was re-read 27x and the run hit its iteration cap. So:

  - a result stays verbatim while its call is HOT: the call's own argument keys
    (e.g. the record id it read) recur in the last ``hot_window`` assistant messages;
  - identical (tool, args) calls keep only their NEWEST result verbatim; older copies
    become short pointers, not full stubs;
  - the stub wording no longer pushes an immediate re-fetch;
  - the host caps how many requests per session it demotes (a progress brake): if a
    run is still going past the cap, it finishes on full context. A re-fetch counter
    was tried and rejected: this workload re-reads records naturally (2-60 re-reads of
    stubbed calls per healthy run, median ~16), so re-fetch counts and densities did
    not separate healthy runs from the v1 loop.

Persisted history is never touched. On a caching provider this is a bad trade (every
demotion changes the prefix and forfeits the cache), so it is opt-in per profile via
``compression.wire_demotion``. Measured basis: knowfleet task #564 replay (record
edc31437) and live runs (#566).
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


def _unwrap(call: Dict[str, Any]) -> Tuple[str, str]:
    """(tool name, canonical args) for one tool call, unwrapping the tool_call bridge."""
    fn = call.get("function") or {}
    name, args = fn.get("name") or "?", fn.get("arguments") or ""
    try:
        parsed = json.loads(args) if args else {}
    except ValueError:
        parsed = None
    if name == "tool_call" and isinstance(parsed, dict):
        inner = parsed.get("calls") or []
        if inner and isinstance(inner[0], dict):
            name = inner[0].get("name") or name
            parsed = inner[0].get("arguments") or {}
    if isinstance(parsed, (dict, list)):
        args = json.dumps(parsed, sort_keys=True)
    return name, args


def _call_index(messages: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """tool_call_id -> (tool name, canonical args)."""
    index: Dict[str, Tuple[str, str]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if call.get("id"):
                index[call["id"]] = _unwrap(call)
    return index


def _hot_keys(messages: List[Dict[str, Any]], hot_window: int) -> set:
    """Evidence keys in the arguments and text of the last ``hot_window`` assistant messages."""
    hot: set = set()
    seen = 0
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        seen += 1
        text = _content_text(msg.get("content"))
        for call in msg.get("tool_calls") or []:
            text += " " + _unwrap(call)[1]
        hot.update(evidence_keys(text, limit=200))
        if seen >= hot_window:
            break
    return hot


def stub_for(name: str, args_preview: str, original: str) -> str:
    keys = evidence_keys(original)
    keys_part = ", ".join(keys) if keys else "none"
    return (f"{DEMOTED_MARKER} Earlier result of {name}({args_preview}) — {len(original)} chars, trimmed from "
            f"this request to save tokens. Evidence keys it contained: {keys_part}. Use these keys as "
            f"references; call the tool again only if you need specific details beyond them.")


def pointer_for(name: str, args_preview: str, original: str) -> str:
    return (f"{DEMOTED_MARKER} Earlier result of {name}({args_preview}) — {len(original)} chars; the same "
            f"call was made again later in this conversation, see the newer result.")


def demote_request(messages: List[Dict[str, Any]], *, keep_last_messages: int = 10,
                   min_result_chars: int = 500, hot_window: int = 8,
                   ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, int]]:
    """Return (new request list or None if nothing changed, stats). Never mutates ``messages``."""
    stats = {"demoted": 0, "pointers": 0, "pinned_hot": 0, "chars_before": 0, "chars_after": 0}
    if not messages:
        return None, stats
    boundary = len(messages) - max(0, int(keep_last_messages))
    calls = _call_index(messages)
    hot = _hot_keys(messages, hot_window)
    # Newest occurrence of each identical call: older copies become pointers.
    newest: Dict[Tuple[str, str], int] = {}
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool" and msg.get("tool_call_id") in calls:
            newest[calls[msg["tool_call_id"]]] = i
    out: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= boundary or msg.get("role") != "tool":
            out.append(msg)
            continue
        text = _content_text(msg.get("content"))
        key = calls.get(msg.get("tool_call_id") or "", (msg.get("name") or "?", ""))
        name, args = key
        if len(text) < min_result_chars or text.startswith(DEMOTED_MARKER) or name in NEVER_DEMOTE_TOOLS:
            out.append(msg)
            continue
        preview = args[:ARG_PREVIEW_CHARS]
        if key in newest and newest[key] != i:
            replacement = pointer_for(name, preview, text)
            stats["pointers"] += 1
        elif set(evidence_keys(args, limit=50)) & hot:
            out.append(msg)  # working set: the model is still using what this call fetched
            stats["pinned_hot"] += 1
            continue
        else:
            replacement = stub_for(name, preview, text)
        new_msg = dict(msg)
        new_msg["content"] = replacement
        out.append(new_msg)
        stats["demoted"] += 1
        stats["chars_before"] += len(text)
        stats["chars_after"] += len(replacement)
    return (out if stats["demoted"] else None), stats


def ab_arm(session_id: str) -> str:
    """Deterministic 50/50 arm for A/B runs: 'on' or 'off', stable per session id."""
    digest = hashlib.sha256((session_id or "").encode()).digest()
    return "on" if digest[0] % 2 == 0 else "off"
