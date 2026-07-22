"""Regression tests for the voice/canvas backend + the evaluation fixes."""
import asyncio
import json
import sys
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
    guard = getattr(alt, "require_public_http_url", None)
    error = getattr(alt, "UnsafeUrlError", None)
    if guard is None or error is None:
        pytest.skip("installed AgenticLocal revision does not expose its legacy URL guard")
    with pytest.raises(error):
        guard("http://127.0.0.1:11434/api/tags")
    assert guard("https://api.open-meteo.com/v1/forecast")


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


# --- everything-agentic + answer hygiene ------------------------------------ #
def test_no_plain_fast_path_remains():
    # Requirement: every voice turn runs the agent loop. The old plain-completion
    # "fast path" (and its intent router) must not exist.
    import agentica_core.voice_gateway as vg
    assert not hasattr(vg, "_needs_tools")
    assert not hasattr(vg._Conn, "_run_fast_turn")
    assert not hasattr(vg, "_VOICE_SYSTEM")


def test_greeting_needs_no_ack():
    from agentica_core.voice_gateway import _ack_for
    assert _ack_for("hi there") == ""
    assert _ack_for("thanks!") == ""


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


# --- local voice gateway lifecycle ----------------------------------------- #
class _VoiceWs:
    def __init__(self):
        self.frames = []
        self.closed = False

    async def send(self, raw):
        self.frames.append(json.loads(raw))

    async def close(self):
        self.closed = True


class _VoiceApp:
    def create_session(self):
        return "voice-session"


class _VoiceState:
    def __init__(self):
        self.app = _VoiceApp()
        self.calls = []

    def app_for(self, workspace, **kwargs):
        self.calls.append((workspace, kwargs))
        return self.app


async def _quiet_tts(*_args, **_kwargs):
    return None


def _terminal_count(frames):
    return sum(frame.get("type") == "done" for frame in frames)


def test_gateway_turn_is_always_agentic_and_propagates_runtime(monkeypatch):
    import agentica_core.voice_gateway as vg

    monkeypatch.delenv("AGENTICA_API_TOKEN", raising=False)
    calls = []

    def agent_turn(app, message, session_id, emit, cancel_event=None, **_kwargs):
        calls.append((app, message, session_id, cancel_event))
        return {"final_answer": "agent answer", "steps": [], "session_id": "s2"}

    monkeypatch.setattr(vg, "stream_agent_turn", agent_turn)

    async def scenario():
        ws = _VoiceWs()
        state = _VoiceState()
        conn = vg._Conn(state, ws)
        monkeypatch.setattr(conn, "_synth_send_blocking", lambda *_: None)
        monkeypatch.setattr(conn, "_speak", _quiet_tts)
        await conn._dispatch({
            "type": "start", "target": "gpu-box", "model": "org/model",
            "model_engine": "vllm", "workspace": "/repo",
            "workspace_target": "local",
        })
        ws.frames.clear()
        await conn._run_turn("inspect the repository")
        return ws.frames, state.calls

    frames, state_calls = asyncio.run(scenario())
    assert len(calls) == 1
    assert calls[0][1].startswith("inspect the repository")
    assert state_calls[-1] == ("/repo", {
        "target": "gpu-box", "model": "org/model", "engine": "vllm",
        "workspace_target": "local",
    })
    assert [f["type"] for f in frames] == ["transcript", "status", "answer", "done"]
    assert _terminal_count(frames) == 1


@pytest.mark.parametrize("result,error_text", [
    ("", "no speech recognized"),
    (RuntimeError("decoder broke"), "speech-to-text unavailable"),
])
def test_audio_transcription_failure_always_terminates(monkeypatch, result, error_text):
    import agentica_core.voice_gateway as vg
    import agentica_core.voice_provision as vp

    monkeypatch.delenv("AGENTICA_API_TOKEN", raising=False)

    def transcribe(*_args):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(vp, "transcribe_pcm16", transcribe)

    async def scenario():
        ws = _VoiceWs()
        conn = vg._Conn(_VoiceState(), ws)
        conn.audio.extend(b"\x00\x00")
        await conn._transcribe_and_turn()
        return ws.frames

    frames = asyncio.run(scenario())
    assert error_text in frames[0]["error"]
    assert frames[-1] == {"type": "done"}
    assert _terminal_count(frames) == 1


def test_agent_failure_always_terminates(monkeypatch):
    import agentica_core.voice_gateway as vg

    monkeypatch.delenv("AGENTICA_API_TOKEN", raising=False)
    monkeypatch.setattr(
        vg, "stream_agent_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("agent broke")),
    )

    async def scenario():
        ws = _VoiceWs()
        conn = vg._Conn(_VoiceState(), ws)
        monkeypatch.setattr(conn, "_synth_send_blocking", lambda *_: None)
        monkeypatch.setattr(conn, "_speak", _quiet_tts)
        await conn._run_turn("do the task")
        return ws.frames

    frames = asyncio.run(scenario())
    assert any("agent broke" in f.get("error", "") for f in frames)
    assert frames[-1] == {"type": "done"}
    assert _terminal_count(frames) == 1


def test_agent_timeout_and_barge_in_each_terminate(monkeypatch):
    import agentica_core.voice_gateway as vg

    monkeypatch.delenv("AGENTICA_API_TOKEN", raising=False)

    def waits_for_cancel(_app, _message, _session_id, _emit, cancel_event=None, **_kwargs):
        cancel_event.wait(1)
        return {"final_answer": "(interrupted)", "steps": []}

    monkeypatch.setattr(vg, "stream_agent_turn", waits_for_cancel)
    monkeypatch.setattr(vg, "_TURN_TIMEOUT_S", 0.01)

    async def timed_out():
        ws = _VoiceWs()
        conn = vg._Conn(_VoiceState(), ws)
        monkeypatch.setattr(conn, "_synth_send_blocking", lambda *_: None)
        monkeypatch.setattr(conn, "_speak", _quiet_tts)
        await conn._run_turn("slow task")
        return ws.frames

    timeout_frames = asyncio.run(timed_out())
    assert any("too long" in f.get("error", "") for f in timeout_frames)
    assert _terminal_count(timeout_frames) == 1

    # Give cancellation a long deadline; barge-in, not the watchdog, must end it.
    monkeypatch.setattr(vg, "_TURN_TIMEOUT_S", 10)
    started = threading.Event()

    def cancellable(_app, _message, _session_id, _emit, cancel_event=None, **_kwargs):
        started.set()
        cancel_event.wait(1)
        return {"final_answer": "(interrupted)", "steps": []}

    monkeypatch.setattr(vg, "stream_agent_turn", cancellable)

    async def interrupted():
        ws = _VoiceWs()
        conn = vg._Conn(_VoiceState(), ws)
        monkeypatch.setattr(conn, "_synth_send_blocking", lambda *_: None)
        monkeypatch.setattr(conn, "_speak", _quiet_tts)
        conn.task = asyncio.create_task(conn._run_turn("cancel me"))
        while not started.is_set():
            await asyncio.sleep(0.001)
        await conn._interrupt()
        return ws.frames

    interrupted_frames = asyncio.run(interrupted())
    assert interrupted_frames[-1] == {"type": "done"}
    assert _terminal_count(interrupted_frames) == 1


# --- local-only STT/TTS provisioning --------------------------------------- #
def test_stt_auto_falls_back_when_metal_is_unavailable(monkeypatch):
    import agentica_core.voice_provision as vp

    monkeypatch.setattr(vp, "STT_ENGINE", "auto")
    monkeypatch.setattr(vp, "_mlx_whisper_installed", lambda: True)
    monkeypatch.setattr(vp, "_metal_available", lambda: False)
    monkeypatch.setattr(vp, "_faster_whisper_installed", lambda: True)
    assert vp.stt_engine() == "faster"


def test_voice_status_requires_cached_stt_weights(monkeypatch):
    import agentica_core.voice_provision as vp

    monkeypatch.setattr(vp, "stt_engine", lambda: "faster")
    monkeypatch.setattr(vp, "_stt_model_ready", lambda _engine=None: False)
    monkeypatch.setattr(vp, "_kokoro_ready", lambda: True)
    monkeypatch.setattr(vp, "_piper_binary", lambda: None)
    monkeypatch.setattr(vp, "_piper_voice", lambda: None)
    status = vp.voice_status()
    assert status["whisper_installed"] is True
    assert status["stt_ready"] is False
    assert status["tts_engine"] == "kokoro"
    assert "remote_tts" not in status


def test_local_tts_ignores_remote_url_and_falls_back_to_piper(monkeypatch):
    import agentica_core.voice_provision as vp

    monkeypatch.setenv("AGENTICA_TTS_URL", "http://remote.invalid")
    monkeypatch.setattr(vp, "synthesize_kokoro_pcm", lambda _text: (b"kokoro", 24000))
    assert vp.synthesize_pcm("hello") == (b"kokoro", 24000)
    monkeypatch.setattr(
        vp, "synthesize_kokoro_pcm",
        lambda _text: (_ for _ in ()).throw(vp.VoiceUnavailable("missing")),
    )
    monkeypatch.setattr(vp, "synthesize_piper_pcm", lambda _text: (b"piper", 22050))
    assert vp.synthesize_pcm("hello") == (b"piper", 22050)
    assert not hasattr(vp, "synthesize_remote_pcm")


def test_warmup_downloads_missing_stt_weights(monkeypatch):
    import agentica_core.voice_provision as vp

    calls = []
    ready = {"ok": False}
    monkeypatch.setattr(vp, "stt_engine", lambda: "mlx")
    monkeypatch.setattr(vp, "_stt_model_ready", lambda _engine=None: ready["ok"])
    monkeypatch.setattr(
        vp, "install_voice",
        lambda _progress: calls.append("install") or ready.update(ok=True) or True,
    )
    monkeypatch.setattr(vp, "transcribe_pcm16", lambda *_args: "")
    monkeypatch.setattr(vp, "_load_kokoro", lambda: object())
    out = vp.warmup()
    assert calls == ["install"]
    assert out["downloaded"] is True
    assert out["stt"] is True
    assert out["tts"] is True


def test_warmup_can_skip_download(monkeypatch):
    import agentica_core.voice_provision as vp

    monkeypatch.setattr(vp, "stt_engine", lambda: "mlx")
    monkeypatch.setattr(vp, "_stt_model_ready", lambda _engine=None: False)
    monkeypatch.setattr(
        vp, "transcribe_pcm16",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    monkeypatch.setattr(vp, "_load_kokoro", lambda: object())
    assert vp.warmup(download_missing=False) == {"stt": False, "tts": True, "downloaded": False}


def test_install_voice_fetches_stt_weights(monkeypatch, tmp_path):
    import agentica_core.voice_provision as vp

    calls = []
    monkeypatch.setattr(vp, "VOICE_HOME", tmp_path)
    monkeypatch.setattr(vp, "stt_engine", lambda: "faster")
    monkeypatch.setattr(vp, "transcribe_pcm16", lambda *_args: calls.append("stt") or "")
    # Avoid optional voice dependencies/network in this provisioning unit test.
    monkeypatch.setitem(sys.modules, "kokoro_onnx", object())
    model = tmp_path / "kokoro.onnx"
    model.touch()
    voices = tmp_path / "voices.bin"
    voices.touch()
    monkeypatch.setattr(vp, "KOKORO_MODEL", model)
    monkeypatch.setattr(vp, "KOKORO_VOICES", voices)
    piper = tmp_path / "piper"
    piper.touch()
    voice = tmp_path / "voice.onnx"
    voice.touch()
    monkeypatch.setattr(vp, "_piper_binary", lambda: piper)
    monkeypatch.setattr(vp, "_piper_voice", lambda: voice)
    progress = []
    assert vp.install_voice(progress.append)
    assert calls == ["stt"]
    assert "speech-recognition model ready" in progress
