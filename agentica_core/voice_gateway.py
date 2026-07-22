"""WebSocket voice gateway (:8771) — the duplex transport for voice mode.

The HTTP API (:8770) streams one way (SSE) and can't carry mic audio up while
TTS streams down, nor an out-of-band barge-in. This gateway adds a full-duplex
WebSocket alongside the (untouched) sync HTTP server, started on a daemon
asyncio thread from ``apiserver.serve``.

Voice is LOCAL-ONLY: open-weight Whisper STT + Kokoro TTS on this machine.
Every turn — spoken or typed — runs the same tool-using agent loop as Chat
mode; there is intentionally no plain-completion path and no browser/cloud
speech engine.

Client → server frames (JSON):
  {type:"start", token?, target, model, model_engine?, workspace?, workspace_target?}
  {type:"text", text}                      # typed input (still an agent turn)
  {type:"audio", pcm: <base64 pcm16>, sr}  # mic chunk
  {type:"audio_end"}                       # utterance finished -> STT
  {type:"artifact_action", action}         # clicked a canvas card
  {type:"barge_in"}                        # interrupt the agent

Server → client frames (JSON):
  {type:"session", session_id}
  {type:"transcript", text, final}
  {type:"status", text}
  {type:"step", step}        {type:"artifact", artifact}
  {type:"answer", text}
  {type:"tts_audio", pcm: <base64>, sr}    # local TTS audio
  {type:"done"}    {type:"error", error}

Degrades: no ``websockets`` lib -> the gateway is skipped (HTTP still serves);
no STT/TTS models -> spoken turns error with a download hint, typed turns work.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import re as _re
import threading
from typing import Any

from .voice_stream import stream_agent_turn

_TURN_TIMEOUT_S = 180

# Appended to the user's words before the agent runs (the UI still shows the
# user's actual transcript): keeps spoken replies short + conversational.
_VOICE_STEER = (
    "\n\n[Voice mode: answer in one or two short, friendly spoken sentences. "
    "Still call tools to show weather/charts/pages on the canvas, but do NOT read "
    "tables, lists, or long details aloud — just summarize and point to the screen.]"
)

# ---------------------------------------------------------------------------
# Latency: an agentic turn takes seconds before the first real content — dead
# air that makes voice mode feel broken. Every turn therefore SPEAKS a short,
# intent-matched acknowledgment up front while the loop works. (The old
# "fast path" that skipped the agent loop for chit-chat is gone on purpose:
# every turn is agentic, so tools/memory always work and always persist.)
# ---------------------------------------------------------------------------

# Internal agent-loop text that must never be spoken as the answer.
_JUNK_ANSWER = _re.compile(
    r"(tool .{0,60} already (completed|ran)|"
    r"step limit|max[_ ]steps|\(interrupted\)|no answer was produced)", _re.IGNORECASE)


def _ack_for(text: str) -> str:
    t = (text or "").lower()
    if _re.search(r"weather|forecast|temperature|rain|snow", t):
        return "Let me check the weather."
    if _re.search(r"news|search|look ?up", t):
        return "Let me look that up."
    if _re.search(r"open|show|browse|website|url|link", t):
        return "Sure — opening that now."
    if _re.search(r"\b(hi|hello|hey|thanks|thank you)\b", t):
        return ""  # a greeting needs no "on it" preamble
    return "On it — give me a few seconds."


def _clean_answer(answer: str, tool_summary: str | None) -> str:
    """The text we show AND speak. The 4B model sometimes ends a turn with loop
    bookkeeping ('Tool get_weather already completed.') — prefer the last tool's
    human summary over junk, and never return empty."""
    a = (answer or "").strip()
    if a and not _JUNK_ANSWER.search(a):
        return a
    if tool_summary:
        return tool_summary
    if a:
        return "Done — the details are on your screen."
    return "Sorry, I hit a snag with that one."


def start_voice_gateway(state: Any, host: str = "127.0.0.1", port: int = 8771):
    """Start the gateway on a daemon thread. Returns the thread, or None if the
    ``websockets`` dependency is unavailable (HTTP API keeps working)."""
    try:
        import websockets  # noqa: F401
    except Exception:  # noqa: BLE001
        print("voice gateway: 'websockets' not installed — voice mode disabled. "
              "pip install websockets to enable.")
        return None

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve(state, host, port))

    t = threading.Thread(target=_run, name="agentica-voice-gateway", daemon=True)
    t.start()
    # Load the STT/TTS models off the first turn's critical path: the first
    # utterance otherwise pays multi-second lazy Whisper/Kokoro construction.
    threading.Thread(target=_warm_models, name="agentica-voice-warmup",
                     daemon=True).start()
    print(f"agentica voice gateway (WebSocket) on ws://{host}:{port}")
    return t


def _warm_models():
    try:
        from .voice_provision import warmup

        warmup()
    except Exception:  # noqa: BLE001 - models not installed yet; first turn reports it
        pass


async def _serve(state, host, port):
    import websockets

    async def handler(ws, *_):
        await _Conn(state, ws).run()

    # ping_interval keeps the socket alive + detects a dead client (the mic stays
    # open otherwise); max_size bounds a single audio frame.
    async with websockets.serve(handler, host, port, max_size=8 * 1024 * 1024,
                                ping_interval=20, ping_timeout=20):
        await asyncio.Future()  # run forever


class _Conn:
    def __init__(self, state, ws):
        self.state = state
        self.ws = ws
        self.loop = asyncio.get_event_loop()
        self.target = "local"
        self.model = None
        self.model_engine = None
        self.workspace = None
        self.workspace_target = "local"
        self.session_id = None
        self.turn_id = 0
        self.task: asyncio.Task | None = None
        self.cancel: threading.Event | None = None  # set on barge-in to stop the agent loop
        self.audio = bytearray()
        self.audio_sr = 16000
        self._approval_box: dict[str, Any] | None = None
        # Same shared-secret gate as the HTTP API: when AGENTICA_API_TOKEN is set
        # the client's `start` message must carry it, else any local process /
        # web page could open ws://127.0.0.1:8771 and drive the agent.
        self._required_token = os.environ.get("AGENTICA_API_TOKEN")
        self.authed = not self._required_token

    async def run(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    self.audio.extend(raw)
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                await self._dispatch(msg)
        except Exception:  # noqa: BLE001 - client dropped
            pass
        finally:
            task = self.task
            if self.cancel:
                self.cancel.set()
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _send(self, obj):
        try:
            await self.ws.send(json.dumps(obj))
        except Exception:  # noqa: BLE001
            pass

    async def _dispatch(self, msg):
        t = msg.get("type")
        # Authenticate on the start frame; reject everything until authed.
        if not self.authed:
            if t == "start" and self._required_token and hmac.compare_digest(
                str(msg.get("token") or ""), self._required_token
            ):
                self.authed = True
            else:
                await self._send({"type": "error", "error": "unauthorized"})
                try:
                    await self.ws.close()
                except Exception:  # noqa: BLE001
                    pass
                return
        if t == "start":
            self.target = msg.get("target", "local")
            self.model = msg.get("model")
            self.model_engine = msg.get("model_engine")
            self.workspace = msg.get("workspace")
            self.workspace_target = msg.get("workspace_target", "local")
            try:
                app = self._app()
                self.session_id = app.create_session() if hasattr(app, "create_session") else None
            except Exception as exc:  # noqa: BLE001
                await self._send({"type": "error", "error": f"start: {exc}"})
                return
            await self._send({"type": "session", "session_id": self.session_id})
        elif t == "text":
            text = (msg.get("text") or "").strip()
            if text:
                await self._spawn_turn(text)
            else:
                await self._error_done("text is empty")
        elif t == "audio":
            pcm = msg.get("pcm")
            if pcm:
                self.audio.extend(base64.b64decode(pcm))
                self.audio_sr = int(msg.get("sr", self.audio_sr))
        elif t == "audio_end":
            await self._transcribe_and_turn()
        elif t == "artifact_action":
            phrase = _action_to_phrase(msg.get("action") or {})
            if phrase:
                await self._spawn_turn(phrase)
        elif t == "barge_in":
            await self._interrupt()  # supersede emits/speaking AND stop the agent loop
        elif t in {"approve", "deny"}:
            box = self._approval_box
            if box and box.get("event") is not None:
                box["result"] = t == "approve"
                box["event"].set()

    def _app(self):
        return self.state.app_for(self.workspace, target=self.target, model=self.model,
                                  engine=self.model_engine,
                                  workspace_target=self.workspace_target)

    async def _error_done(self, error: str):
        """Terminate an accepted turn that cannot reach ``_run_turn``."""
        await self._send({"type": "error", "error": error})
        await self._send({"type": "done"})

    async def _interrupt(self):
        # Stop the in-flight turn: drop its emits (turn_id), stop the agent loop at
        # its next step boundary (cancel event -> no more tools/model calls), and
        # cancel the asyncio wrapper. Await it so its terminal ``done`` is ordered
        # before a replacement turn begins.
        self.turn_id += 1
        if self.cancel:
            self.cancel.set()
        task = self.task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.task is task:
            self.task = None

    async def _spawn_turn(self, text: str):
        if not text:
            return
        await self._interrupt()  # a new utterance supersedes any running turn
        self.task = self.loop.create_task(self._run_turn(text))

    async def _transcribe_and_turn(self):
        pcm = bytes(self.audio)
        self.audio = bytearray()
        if not pcm:
            await self._error_done("no speech audio received")
            return
        try:
            from .voice_provision import transcribe_pcm16

            text = await self.loop.run_in_executor(None, transcribe_pcm16, pcm, self.audio_sr)
        except Exception as exc:  # noqa: BLE001
            await self._error_done(f"speech-to-text unavailable: {exc}")
            return
        text = (text or "").strip()
        if not text:
            await self._error_done("no speech recognized")
            return
        # (_run_turn echoes the final transcript — sending here too double-fires it.)
        await self._spawn_turn(text)

    async def _run_turn(self, text: str):
        """Every turn is agentic — the same tool loop as Chat mode."""
        # turn_id was already advanced by _interrupt() in _spawn_turn; just capture it.
        my = self.turn_id
        # A fresh cancel token for THIS turn, captured in the closure so a later turn
        # replacing self.cancel can't make us pass the wrong event into the executor.
        cancel = self.cancel = threading.Event()
        try:
            await self._send({"type": "transcript", "text": text, "final": True})
            try:
                app = self._app()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"turn: {exc}") from exc

            # Speak an acknowledgment BEFORE the loop starts — silent "thinking" is
            # what makes voice mode feel broken.
            ack = _ack_for(text)
            if ack:
                await self.loop.run_in_executor(None, self._synth_send_blocking, ack, my)
            await self._send({"type": "status", "text": "working on it…"})

            last_summary: list[str | None] = [None]
            spoken_final = threading.Event()
            approval_box: dict[str, Any] = {"event": None, "result": None}

            def emit(frame):
                if self.turn_id != my:
                    return
                if "final_answer" in frame:
                    # Start TTS as soon as the engine logs final_answer — don't
                    # wait for stream_agent_turn's worker thread to join.
                    answer = _clean_answer(frame.get("final_answer") or "", last_summary[0])
                    if answer and not spoken_final.is_set():
                        spoken_final.set()
                        asyncio.run_coroutine_threadsafe(
                            self._send({"type": "answer", "text": answer}), self.loop)
                        threading.Thread(
                            target=self._speak_blocking,
                            args=(answer, my),
                            daemon=True,
                            name="agentica-early-tts",
                        ).start()
                    return
                if "approval" in frame:
                    asyncio.run_coroutine_threadsafe(self._send({
                        "type": "approval_required",
                        **(frame.get("approval") or {}),
                    }), self.loop)
                    return
                if "step" in frame:
                    obs = (frame["step"] or {}).get("observation")
                    if isinstance(obs, dict) and isinstance(obs.get("summary"), str):
                        last_summary[0] = obs["summary"]
                    out = {"type": "step", "step": frame["step"]}
                elif "artifact" in frame:
                    out = {"type": "artifact", "artifact": frame["artifact"]}
                else:
                    return
                asyncio.run_coroutine_threadsafe(self._send(out), self.loop)

            def approval_callback(call, decision):
                # Pause the agent loop until the client sends approve/deny.
                approval_box["event"] = threading.Event()
                approval_box["result"] = None
                asyncio.run_coroutine_threadsafe(self._send({
                    "type": "approval_required",
                    "tool": getattr(call, "name", None),
                    "arguments": getattr(call, "arguments", {}) or {},
                    "reason": getattr(decision, "reason", ""),
                    "blocking": True,
                }), self.loop)
                ev = approval_box["event"]
                while not ev.wait(0.25):
                    if self.turn_id != my or cancel.is_set():
                        return False
                return bool(approval_box.get("result"))

            # Steer the model toward a short, conversational spoken reply — details,
            # lists, and tables belong on the canvas, not read aloud.
            agent_text = text + _VOICE_STEER
            self._approval_box = approval_box

            def work():
                return stream_agent_turn(
                    app, agent_text, self.session_id, emit,
                    cancel_event=cancel,
                    max_steps=4,
                    approval_callback=approval_callback,
                )

            # Per-turn wall-clock deadline so a wedged model/tool can't leave the
            # conversation stuck in "thinking" forever. On timeout, tell the agent
            # loop to stop (cancel) and report the error.
            res = await asyncio.wait_for(
                self.loop.run_in_executor(None, work), timeout=_TURN_TIMEOUT_S)
            if self.turn_id != my:
                return
            self.session_id = res.get("session_id", self.session_id)
            # Never speak loop bookkeeping — prefer the last tool's human summary.
            answer = _clean_answer(res.get("final_answer") or "", last_summary[0])
            if not spoken_final.is_set():
                await self._send({"type": "answer", "text": answer})
                await self._speak(answer, my)
            else:
                # Early TTS already started; still ensure the text answer is on screen.
                await self._send({"type": "answer", "text": answer})
        except asyncio.TimeoutError:
            cancel.set()
            if self.turn_id == my:
                await self._send({"type": "error", "error": "the agent took too long and was stopped"})
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception as exc:  # noqa: BLE001
            if self.turn_id == my:
                await self._send({"type": "error", "error": str(exc)})
        finally:
            # Exactly one terminal frame for every accepted turn — success, timeout,
            # cancellation, app/model failure, or TTS failure.
            await self._send({"type": "done"})
            if self.cancel is cancel:
                self.cancel = None
            if self.task is asyncio.current_task():
                self.task = None

    def _synth_send_blocking(self, sentence: str, my: int):
        """Synthesize one sentence and ship it (called from an executor thread)."""
        try:
            from .voice_provision import synthesize_pcm

            pcm, sr = synthesize_pcm(sentence)
        except Exception:  # noqa: BLE001 - no TTS: the text answer still shows
            return
        if self.turn_id == my:
            asyncio.run_coroutine_threadsafe(self._send({
                "type": "tts_audio",
                "pcm": base64.b64encode(pcm).decode("ascii"),
                "sr": sr,
            }), self.loop)

    def _speak_blocking(self, text: str, my: int):
        """Full-answer TTS from a worker thread (early final_answer path)."""
        text = _spoken_text(text)
        if not text:
            return
        try:
            from .voice_provision import synthesize_pcm, voice_status

            if not voice_status().get("tts_ready"):
                return
            for sentence in _split_sentences(text):
                if self.turn_id != my:
                    return
                self._synth_send_blocking(sentence, my)
        except Exception:  # noqa: BLE001
            pass

    async def _speak(self, text: str, my: int):
        text = _spoken_text(text)
        if not text:
            return
        try:
            from .voice_provision import synthesize_pcm, voice_status

            if not voice_status().get("tts_ready"):
                return  # the text answer is already on screen
            for sentence in _split_sentences(text):
                if self.turn_id != my:
                    return
                pcm, sr = await self.loop.run_in_executor(None, synthesize_pcm, sentence)
                await self._send({
                    "type": "tts_audio",
                    "pcm": base64.b64encode(pcm).decode("ascii"),
                    "sr": sr,
                })
        except Exception:  # noqa: BLE001 - TTS failure must not kill the turn
            pass


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def _spoken_text(text: str, cap: int = 700) -> str:
    """Make an answer sound natural when SPOKEN: drop markdown tables/formatting
    (the rich version stays in the canvas) and cap to a conversational length."""
    import re

    kept = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.count("|") >= 2:  # markdown table row
            continue
        if s and set(s) <= set("|-:` "):  # table separator / rule
            continue
        kept.append(ln)
    out = " ".join(kept)
    out = re.sub(r"[*#`>_]", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > cap:
        acc = ""
        for sent in re.split(r"(?<=[.!?])\s+", out):
            if len(acc) + len(sent) > cap:
                break
            acc += sent + " "
        out = acc.strip() or out[:cap]
    return out


def _action_to_phrase(a: dict) -> str | None:
    if a.get("type") == "weather_day":
        place = f" in {a['place']}" if a.get("place") else ""
        when = a.get("weekday") or a.get("date") or "that day"
        return f"Tell me more about the weather{place} on {when}."
    if a.get("type") == "open_link" and a.get("url"):
        return f"Open {a['url']} and tell me what's there."
    return None
