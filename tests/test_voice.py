"""Regression tests for the voice/canvas backend + the evaluation fixes."""
import threading

import pytest

from agentica_core import net_guard, voice_stream, voice_tools, apiserver


# --- net_guard / SSRF (H3, H4) ---------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "https://127.0.0.1/x", "http://example.com/", "https://localhost/",
    "https://169.254.169.254/latest/meta-data", "https://10.0.0.5/", "ftp://x/",
])
def test_net_guard_blocks_unsafe(bad):
    with pytest.raises(net_guard.UnsafeUrl):
        net_guard.require_safe_public_url(bad)


def test_net_guard_allows_public_https():
    assert net_guard.require_safe_public_url("https://example.com/p") == "https://example.com/p"


def test_agenticlocal_web_client_ssrf_guard():
    import agentic_loop.tools as alt
    with pytest.raises(alt.UnsafeUrlError):
        alt.require_public_http_url("http://127.0.0.1:11434/api/tags")
    assert alt.require_public_http_url("https://api.open-meteo.com/v1/forecast")


# --- voice_stream high-water paging (H7) ------------------------------------ #
class _FakeApp:
    def __init__(self, n):
        self.ev = [{"id": i, "event_type": "tool_result",
                    "payload": {"tool": "x", "result": "y"}} for i in range(1, n + 1)]

    def events(self, session_id=None, after_id=0):
        return [e for e in self.ev if e["id"] > after_id][:100]


def test_max_event_id_pages_past_100():
    # naive max() over a single LIMIT-100 page would return 100, not 250.
    assert voice_stream._max_event_id(_FakeApp(250), "s") == 250


def test_drain_forwards_all_events():
    frames = []
    after = voice_stream._drain(_FakeApp(250), "s", 0, frames.append)
    assert after == 250
    assert sum(1 for f in frames if "step" in f) == 250


def test_translate_event_extracts_artifact():
    ev = {"id": 1, "event_type": "tool_result",
          "payload": {"tool": "get_weather",
                      "result": {"summary": "ok", "_artifact": {"kind": "weather", "data": {}}}}}
    frames = voice_stream.translate_event(ev)
    assert any("step" in f for f in frames)
    art = [f["artifact"] for f in frames if "artifact" in f][0]
    assert art["kind"] == "weather" and art["source_tool"] == "get_weather"


# --- cancel token (H5) ------------------------------------------------------ #
class _CancelApp:
    """Minimal app whose chat() honours a pre-set cancel_event."""
    def create_session(self):
        return "s1"

    def events(self, session_id=None, after_id=0):
        return []

    def chat(self, message, session_id, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            return {"final_answer": "(interrupted)", "steps": [], "session_id": session_id}
        return {"final_answer": "done", "steps": [], "session_id": session_id}


def test_stream_agent_turn_threads_cancel_event():
    ev = threading.Event(); ev.set()
    res = voice_stream.stream_agent_turn(_CancelApp(), "hi", None, lambda f: None, cancel_event=ev)
    assert res["final_answer"] == "(interrupted)"


# --- get_weather offline + location cleaning (canvas tool) ------------------ #
class _FakeWeb:
    def get_json(self, url, timeout_s=10.0):
        if "geocoding" in url:
            return {"results": [{"name": "Boston", "admin1": "MA", "country": "US",
                                 "latitude": 42.36, "longitude": -71.06}]}
        return {"daily": {"time": ["2026-06-07"], "temperature_2m_max": [24.0],
                          "temperature_2m_min": [14.0], "precipitation_probability_max": [10],
                          "weathercode": [1]}}

    def get_text(self, url, timeout_s=10.0):
        return ""


def test_get_weather_artifact_offline():
    from agentic_loop.tools import ToolContext
    from pathlib import Path
    res = voice_tools.get_weather(ToolContext(workspace_root=Path("."), web_client=_FakeWeb()),
                                  {"location": "Boston this week"})  # time-word stripped
    art = res["_artifact"]
    assert art["kind"] == "weather"
    assert art["data"]["days"][0]["desc"] == "Mainly clear"


def test_clean_location_strips_time_words():
    assert voice_tools._clean_location("weather in London this week") == "London"
    assert voice_tools._clean_location("Tokyo tomorrow") == "Tokyo"
    assert voice_tools._clean_location("New York") == "New York"


# --- backstop protect derivation (C3) --------------------------------------- #
def test_derive_protect_finds_test_files(tmp_path):
    (tmp_path / "test_spec.py").write_text("def test_x(): assert True\n")
    got = apiserver._derive_protect("python -m pytest test_spec.py -q", str(tmp_path))
    assert got == ["test_spec.py"]
    assert apiserver._derive_protect("pytest nonexistent.py", str(tmp_path)) == []


# --- conversational routing + answer hygiene (usability overhaul) ----------- #
def test_needs_tools_routes_tool_intents():
    from agentica_core.voice_gateway import _needs_tools
    assert _needs_tools("What's the weather in Boston?")
    assert _needs_tools("open example.com and tell me what's there")
    assert _needs_tools("read the file notes.txt")
    assert _needs_tools("search for the latest news")
    assert not _needs_tools("Say hello in one short sentence.")
    assert not _needs_tools("How are you today?")
    assert not _needs_tools("Tell me a joke about penguins")


def test_ack_matches_intent():
    from agentica_core.voice_gateway import _ack_for
    assert "weather" in _ack_for("what's the weather in Paris").lower()
    assert _ack_for("read my notes file")  # always non-empty


def test_clean_answer_never_speaks_loop_junk():
    from agentica_core.voice_gateway import _clean_answer
    junk = "Tool get_weather already completed."
    assert _clean_answer(junk, "It's 72 and sunny in Boston.") == "It's 72 and sunny in Boston."
    assert _clean_answer(junk, None) == "Done — the details are on your screen."
    assert _clean_answer("", None) == "Sorry, I hit a snag with that one."
    assert _clean_answer("It's sunny today!", None) == "It's sunny today!"
    assert "interrupted" not in _clean_answer("(interrupted)", None)
