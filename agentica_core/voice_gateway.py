"""WebSocket voice gateway (:8771) — the duplex transport for voice mode.

The HTTP API (:8770) streams one way (SSE) and can't carry mic audio up while
TTS streams down, nor an out-of-band barge-in. This gateway adds a full-duplex
WebSocket alongside the (untouched) sync HTTP server, started on a daemon
asyncio thread from ``apiserver.serve``.

Client → server frames (JSON):
  {type:"start", engine, target, model, workspace}
  {type:"text", text}                      # transcript ready (browser STT) / typed
  {type:"audio", pcm: <base64 pcm16>, sr}  # local engine: mic chunk
  {type:"audio_end"}                       # local engine: utterance finished -> STT
  {type:"artifact_action", action}         # clicked a canvas card
  {type:"barge_in"}                        # interrupt the agent

Server → client frames (JSON):
  {type:"session", session_id}
  {type:"transcript", text, final}
  {type:"status", text}
  {type:"step", step}        {type:"artifact", artifact}
  {type:"answer", text}
  {type:"speak", text}                     # browser TTS (no local Piper)
  {type:"tts_audio", pcm: <base64>, sr}    # local Piper audio
  {type:"done"}    {type:"error", error}

Everything degrades: no ``websockets`` lib -> gateway is skipped (HTTP still
serves); no Whisper/Piper -> the local engine asks the client to type / use
browser TTS.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import threading
from typing import Any

from .voice_stream import stream_agent_turn

# Appended to the user's words before the agent runs (the UI still shows the
# user's actual transcript): keeps spoken replies short + conversational.
_VOICE_STEER = (
    "\n\n[Voice mode: answer in one or two short, friendly spoken sentences. "
    "Still call tools to show weather/charts/pages on the canvas, but do NOT read "
    "tables, lists, or long details aloud — just summarize and point to the screen.]"
)


def start_voice_gateway(state: Any, host: str = "127.0.0.1", port: int = 8771):
    """Start the gateway on a daemon thread. Returns the thread, or None if the
    ``websockets`` dependency is unavailable (HTTP API keeps working)."""
    try:
        import websockets  # noqa: F401
    except Exception:  # noqa: BLE001
        print("voice gateway: 'websockets' not installed — voice 'local' engine disabled "
              "(browser/cloud engines still work). pip install websockets to enable.")
        return None

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve(state, host, port))

    t = threading.Thread(target=_run, name="agentica-voice-gateway", daemon=True)
    t.start()
    print(f"agentica voice gateway (WebSocket) on ws://{host}:{port}")
    return t


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
        self.engine = "browser"
        self.target = "local"
        self.model = None
        self.workspace = None
        self.session_id = None
        self.turn_id = 0
        self.task: asyncio.Task | None = None
        self.cancel: threading.Event | None = None  # set on barge-in to stop the agent loop
        self.audio = bytearray()
        self.audio_sr = 16000
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
            if self.task:
                self.task.cancel()

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
            self.engine = msg.get("engine", "browser")
            self.target = msg.get("target", "local")
            self.model = msg.get("model")
            self.workspace = msg.get("workspace")
            try:
                app = self._app()
                self.session_id = app.create_session() if hasattr(app, "create_session") else None
            except Exception as exc:  # noqa: BLE001
                await self._send({"type": "error", "error": f"start: {exc}"})
                return
            await self._send({"type": "session", "session_id": self.session_id})
        elif t == "text":
            self._spawn_turn((msg.get("text") or "").strip())
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
                self._spawn_turn(phrase)
        elif t == "barge_in":
            self._interrupt()  # supersede emits/speaking AND stop the agent loop

    def _app(self):
        return self.state.app_for(self.workspace, target=self.target, model=self.model, engine=None)

    def _interrupt(self):
        # Stop the in-flight turn: drop its emits (turn_id), stop the agent loop at
        # its next step boundary (cancel event -> no more tools/model calls), and
        # cancel the asyncio wrapper.
        self.turn_id += 1
        if self.cancel:
            self.cancel.set()
        if self.task and not self.task.done():
            self.task.cancel()

    def _spawn_turn(self, text: str):
        if not text:
            return
        self._interrupt()  # a new utterance supersedes any running turn
        self.task = self.loop.create_task(self._run_turn(text))

    async def _transcribe_and_turn(self):
        pcm = bytes(self.audio)
        self.audio = bytearray()
        if not pcm:
            return
        try:
            from .voice_provision import transcribe_pcm16

            text = await self.loop.run_in_executor(None, transcribe_pcm16, pcm, self.audio_sr)
        except Exception as exc:  # noqa: BLE001
            await self._send({"type": "error", "error": f"speech-to-text unavailable: {exc}"})
            return
        await self._send({"type": "transcript", "text": text, "final": True})
        self._spawn_turn(text)

    async def _run_turn(self, text: str):
        # turn_id was already advanced by _interrupt() in _spawn_turn; just capture it.
        my = self.turn_id
        # A fresh cancel token for THIS turn, captured in the closure so a later turn
        # replacing self.cancel can't make us pass the wrong event into the executor.
        cancel = self.cancel = threading.Event()
        await self._send({"type": "transcript", "text": text, "final": True})
        try:
            app = self._app()
        except Exception as exc:  # noqa: BLE001
            await self._send({"type": "error", "error": f"turn: {exc}"})
            return

        def emit(frame):
            if self.turn_id != my:
                return
            if "step" in frame:
                out = {"type": "step", "step": frame["step"]}
            elif "artifact" in frame:
                out = {"type": "artifact", "artifact": frame["artifact"]}
            else:
                return
            asyncio.run_coroutine_threadsafe(self._send(out), self.loop)

        # Steer the model toward a short, conversational spoken reply — details,
        # lists, and tables belong on the canvas, not read aloud.
        agent_text = text + _VOICE_STEER

        def work():
            return stream_agent_turn(app, agent_text, self.session_id, emit, cancel_event=cancel)

        try:
            # Per-turn wall-clock deadline so a wedged model/tool can't leave the
            # conversation stuck in "thinking" forever. On timeout, tell the agent
            # loop to stop (cancel) and report the error.
            res = await asyncio.wait_for(self.loop.run_in_executor(None, work), timeout=180)
        except asyncio.TimeoutError:
            cancel.set()
            if self.turn_id == my:
                await self._send({"type": "error", "error": "the agent took too long and was stopped"})
                await self._send({"type": "done"})
            return
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            await self._send({"type": "error", "error": str(exc)})
            return
        if self.turn_id != my:
            return
        self.session_id = res.get("session_id", self.session_id)
        answer = res.get("final_answer") or ""
        await self._send({"type": "answer", "text": answer})
        await self._speak(answer, my)
        await self._send({"type": "done"})

    async def _speak(self, text: str, my: int):
        text = _spoken_text(text)
        if not text:
            return
        # Local engine + provisioned Piper -> stream PCM; else browser TTS.
        if self.engine == "local":
            try:
                from .voice_provision import synthesize_pcm, voice_status

                if voice_status().get("tts_ready"):
                    for sentence in _split_sentences(text):
                        if self.turn_id != my:
                            return
                        pcm, sr = await self.loop.run_in_executor(None, synthesize_pcm, sentence)
                        await self._send({
                            "type": "tts_audio",
                            "pcm": base64.b64encode(pcm).decode("ascii"),
                            "sr": sr,
                        })
                    return
            except Exception:  # noqa: BLE001 - fall through to browser TTS
                pass
        await self._send({"type": "speak", "text": text})


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
