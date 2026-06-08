"""Standalone TTS HTTP server — runs an EXPRESSIVE voice model on a GPU box.

The local default voice is Kokoro (fast, natural, CPU). For a more expressive,
Grok-like voice we run a heavier model on a remote GPU (e.g. the TITAN Xp boxes)
and stream its audio back through an SSH tunnel — reusing Agentica's existing
remote-target transport. This file is that remote service: a dependency-light
stdlib HTTP server wrapping a pluggable TTS engine.

  GET  /health        -> {ok, engine, sample_rate, device}
  POST /tts {text, exaggeration?, voice?} -> audio/wav (PCM16 WAV)

Engines (``--engine``):
  chatterbox  Resemble Chatterbox (MIT, ~0.5B, expressive, Pascal-friendly)
  dia         Nari Dia-1.6B (most expressive; heavier VRAM)
  xtts        Coqui XTTS-v2 (expressive + voice cloning; non-commercial license)
  kokoro      Kokoro (the local default — handy as a stand-in to test the path)

Run on the GPU box:  python -m agentica_core.tts_server --engine chatterbox --port 8780
"""

from __future__ import annotations

import argparse
import io
import json
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class TTSEngine:
    name = "base"
    sample_rate = 24000
    device = "cpu"

    def synth(self, text: str, **kw) -> bytes:
        """Return PCM16 mono bytes at self.sample_rate."""
        raise NotImplementedError


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"

    def __init__(self, device: str | None = None):
        import torch

        # Chatterbox's optional Perth watermarker is None in some builds and
        # crashes loading; fall back to its bundled no-op DummyWatermarker.
        try:
            import perth
            if getattr(perth, "PerthImplicitWatermarker", None) is None and hasattr(perth, "DummyWatermarker"):
                perth.PerthImplicitWatermarker = perth.DummyWatermarker
        except Exception:  # noqa: BLE001
            pass
        from chatterbox.tts import ChatterboxTTS

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(self.model, "sr", 24000))

    def synth(self, text: str, exaggeration: float = 0.6, cfg_weight: float = 0.5, **kw) -> bytes:
        import numpy as np

        wav = self.model.generate(text, exaggeration=float(exaggeration), cfg_weight=float(cfg_weight))
        arr = wav.detach().cpu().numpy().reshape(-1)
        return (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()


class DiaEngine(TTSEngine):
    name = "dia"

    def __init__(self, device: str | None = None):
        import torch
        from dia.model import Dia

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=self.device)
        self.sample_rate = 44100

    def synth(self, text: str, **kw) -> bytes:
        import numpy as np

        arr = np.asarray(self.model.generate(text)).reshape(-1)
        return (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()


class XttsEngine(TTSEngine):
    name = "xtts"

    def __init__(self, device: str | None = None, speaker_wav: str | None = None):
        import torch
        from TTS.api import TTS

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.sample_rate = 24000
        self.speaker_wav = speaker_wav

    def synth(self, text: str, language: str = "en", **kw) -> bytes:
        import numpy as np

        arr = np.asarray(
            self.model.tts(text=text, language=language, speaker_wav=self.speaker_wav)
        ).reshape(-1)
        return (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()


class KokoroEngine(TTSEngine):
    name = "kokoro"

    def __init__(self, **kw):
        from .voice_provision import synthesize_kokoro_pcm

        self._synth = synthesize_kokoro_pcm
        self.sample_rate = 24000
        self.device = "cpu"

    def synth(self, text: str, voice: str | None = None, **kw) -> bytes:
        pcm, sr = self._synth(text, voice)
        self.sample_rate = sr
        return pcm


ENGINES = {
    "chatterbox": ChatterboxEngine,
    "dia": DiaEngine,
    "xtts": XttsEngine,
    "kokoro": KokoroEngine,
}


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def make_handler(engine: TTSEngine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            return

        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({
                    "ok": True, "engine": engine.name,
                    "sample_rate": engine.sample_rate, "device": engine.device,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/tts":
                return self.send_error(404)
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode() or "{}")
            text = (req.get("text") or "").strip()
            if not text:
                return self.send_error(400, "text required")
            try:
                pcm = engine.synth(text, **{k: v for k, v in req.items() if k != "text"})
                wav = _pcm_to_wav(pcm, engine.sample_rate)
            except Exception as exc:  # noqa: BLE001
                return self.send_error(500, str(exc))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("X-Sample-Rate", str(engine.sample_rate))
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)

    return H


def main():
    ap = argparse.ArgumentParser(description="Agentica expressive TTS server")
    ap.add_argument("--engine", choices=list(ENGINES), default="chatterbox")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print(f"loading TTS engine '{args.engine}'…")
    engine = ENGINES[args.engine](device=args.device)
    print(f"engine ready: {engine.name} @ {engine.sample_rate}Hz on {engine.device}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"Agentica TTS server on http://{args.host}:{args.port}  (POST /tts)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
