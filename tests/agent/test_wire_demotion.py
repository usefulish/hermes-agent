"""Key-preserving wire demotion for uncached routes (agent/wire_demotion.py; knowfleet #566).

Old tool results in the REQUEST copy become stubs that keep the result's evidence keys
(UUIDs, hex ids/SHAs, absolute paths, long numbers), so later citations stay grounded
while the content is re-fetched instead of re-sent. Pins: the tail and small results
are kept; skill bodies are never demoted; the tool_call bridge is unwrapped for the stub
label; input is never mutated; a disabled compressor leaves the host hook short-circuited
(no cost, byte-identical requests); the A/B arm is deterministic and 'off' leaves the
request untouched.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from agent.conversation_loop import _apply_context_engine_selection, _engine_overrides_hook
from agent.wire_demotion import DEMOTED_MARKER, ab_arm, demote_request, evidence_keys

UUID = "0c9dba73-5aef-46fa-b4ce-288f9a337fda"
SHA = "6e45453abc1234def"
BIG = f"record {UUID} at commit {SHA} in /Users/guru/Code/x/file.py line 123456 " + ("filler " * 200)


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _bridge(cid, inner, args):
    return _call(cid, "tool_call", {"calls": [{"name": inner, "arguments": args}]})


def _conversation(n_old=3):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(n_old):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            _bridge(f"c{i}", "mcp__knowfleet__knowledge_read", {"id": f"id{i}"})]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": BIG})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [_call("s", "skill_view", {"name": "x"})]})
    msgs.append({"role": "tool", "tool_call_id": "s", "content": "SKILL BODY " * 200})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [_call("t", "terminal", {"command": "ls"})]})
    msgs.append({"role": "tool", "tool_call_id": "t", "content": "tiny"})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [_call("last", "read_file", {"path": "/a"})]})
    msgs.append({"role": "tool", "tool_call_id": "last", "content": BIG})
    return msgs


def _compressor(**wire):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="test/model", quiet_mode=True, wire_demotion=wire or None)


def test_evidence_keys_extracts_version_addressed_handles():
    keys = evidence_keys(BIG)
    assert UUID in keys and SHA in keys and "/Users/guru/Code/x/file.py" in keys and "123456" in keys


def test_demotes_old_large_results_keeps_tail_small_and_skills_without_mutating():
    msgs = _conversation()
    before = json.dumps(msgs)
    out, stats = demote_request(msgs, keep_last_messages=4, min_result_chars=500)
    assert json.dumps(msgs) == before, "input must not be mutated"
    assert stats["demoted"] == 3
    tools = [m for m in out if m["role"] == "tool"]
    for stub in tools[:3]:
        assert stub["content"].startswith(DEMOTED_MARKER)
        assert UUID in stub["content"] and SHA in stub["content"]
        assert "mcp__knowfleet__knowledge_read" in stub["content"], "bridge call is unwrapped for the label"
    assert tools[3]["content"].startswith("SKILL BODY"), "skill bodies are never demoted"
    assert tools[4]["content"] == "tiny", "small results are kept"
    assert tools[5]["content"] == BIG, "the protected tail is kept verbatim"
    assert out[0] == msgs[0] and len(out) == len(msgs)


def test_nothing_to_demote_returns_none_and_already_demoted_is_left_alone():
    out, _ = demote_request([{"role": "user", "content": "hi"}])
    assert out is None
    once, _ = demote_request(_conversation(), keep_last_messages=4)
    twice, stats = demote_request(once, keep_last_messages=4)
    assert twice is None and stats["demoted"] == 0


def test_disabled_compressor_leaves_the_host_hook_short_circuited():
    comp = _compressor()
    assert comp.wire_demotion["enabled"] is False
    assert not _engine_overrides_hook(comp, "select_context"), "disabled => no per-request cost"
    request = _conversation()
    agent = SimpleNamespace(context_compressor=comp, session_id="s1")
    assert _apply_context_engine_selection(agent, request, request, None, logger=MagicMock()) is request


def test_enabled_compressor_demotes_through_the_host_hook():
    comp = _compressor(enabled=True, keep_last_messages=4)
    assert _engine_overrides_hook(comp, "select_context")
    comp.bind_session_state(session_db=None, session_id="sess-x")
    request = _conversation()
    agent = SimpleNamespace(context_compressor=comp, session_id="sess-x")
    out = _apply_context_engine_selection(agent, request, request, None, logger=MagicMock())
    assert out is not request
    assert sum(1 for m in out if m["role"] == "tool" and m["content"].startswith(DEMOTED_MARKER)) == 3
    assert request[3]["content"] == BIG, "persisted/request input list untouched"


def test_ab_split_is_deterministic_and_off_arm_leaves_request_untouched():
    assert ab_arm("same") == ab_arm("same")
    arms = {ab_arm(f"cron_job_2026093{i}") for i in range(10)}
    assert arms == {"on", "off"}
    off_id = next(f"s{i}" for i in range(100) if ab_arm(f"s{i}") == "off")
    on_id = next(f"s{i}" for i in range(100) if ab_arm(f"s{i}") == "on")
    comp = _compressor(enabled=True, ab_split=True, keep_last_messages=4)
    comp.bind_session_state(session_db=None, session_id=off_id)
    assert comp.select_context(_conversation()) is None
    comp.bind_session_state(session_db=None, session_id=on_id)
    assert comp.select_context(_conversation()) is not None


def test_config_normalization_clamps_and_rejects_truthy_strings():
    n = ContextCompressor._normalize_wire_demotion({"enabled": "yes", "keep_last_messages": 0, "min_result_chars": 5})
    assert n["enabled"] is False, "only a real boolean true enables it"
    assert n["keep_last_messages"] == 2 and n["min_result_chars"] == 200


# -- v2 (after the v1 re-fetch loop, knowfleet incident f1778a10) ------------------------------------


def _read(cid, rid):
    return {"role": "assistant", "content": "", "tool_calls": [_bridge(cid, "mcp__knowfleet__knowledge_read", {"id": rid})]}


def _filler(n):
    """n small, unrelated exchanges to push older results out of the protected tail."""
    out = []
    for i in range(n):
        out.append({"role": "assistant", "content": "", "tool_calls": [_call(f"f{i}", "terminal", {"command": f"echo {i}"})]})
        out.append({"role": "tool", "tool_call_id": f"f{i}", "content": "ok"})
    return out


def test_working_set_stays_verbatim_while_its_key_is_in_recent_calls():
    target, other = UUID, "11111111-2222-3333-4444-555555555555"
    msgs = [{"role": "user", "content": "go"},
            _read("a", target), {"role": "tool", "tool_call_id": "a", "content": BIG},
            _read("b", other), {"role": "tool", "tool_call_id": "b", "content": BIG.replace(UUID, other)}]
    msgs += _filler(6)
    # a recent call still about the target: its record is the working set
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        _bridge("c", "mcp__knowfleet__investigation_list", {"record_id": target})]})
    msgs.append({"role": "tool", "tool_call_id": "c", "content": "small"})
    # hot_window=4: the last 4 assistant messages are 3 filler calls + the target call, so
    # only the target is hot; the call that read `other` is older than the window.
    out, stats = demote_request(msgs, keep_last_messages=4, hot_window=4)
    assert out[2]["content"] == BIG, "the record being worked on is pinned"
    assert out[4]["content"].startswith(DEMOTED_MARKER), "a cold record is demoted"
    assert stats["pinned_hot"] == 1


def test_identical_calls_keep_only_the_newest_result_and_older_copies_become_pointers():
    msgs = [{"role": "user", "content": "go"},
            _read("a", UUID), {"role": "tool", "tool_call_id": "a", "content": BIG},
            _read("b", UUID), {"role": "tool", "tool_call_id": "b", "content": BIG}]
    msgs += _filler(3)
    out, stats = demote_request(msgs, keep_last_messages=4, hot_window=1)
    assert "same call was made again later" in out[2]["content"], "older duplicate is a pointer"
    assert stats["pointers"] == 1


def test_stub_wording_does_not_push_an_immediate_refetch():
    out, _ = demote_request(_conversation(), keep_last_messages=4, hot_window=1)
    stub = next(m["content"] for m in out if m["role"] == "tool" and m["content"].startswith(DEMOTED_MARKER))
    assert "only if you need specific details" in stub
    assert "do not rely on memory" not in stub


def test_progress_brake_stops_demoting_after_the_cap():
    comp = _compressor(enabled=True, keep_last_messages=4, hot_window=1, max_demoted_requests=2)
    comp.bind_session_state(session_db=None, session_id="s")
    assert comp.select_context(_conversation()) is not None
    assert comp.select_context(_conversation()) is not None
    assert comp.select_context(_conversation()) is None, "past the cap the run finishes on full context"
    comp.bind_session_state(session_db=None, session_id="s2")
    assert comp.select_context(_conversation()) is not None, "the cap is per session"


def test_v2_defaults():
    n = ContextCompressor._normalize_wire_demotion({"enabled": True})
    assert (n["keep_last_messages"], n["hot_window"], n["max_demoted_requests"]) == (10, 8, 30)
