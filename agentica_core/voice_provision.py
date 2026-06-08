"""Local speech-to-text (Whisper) and text-to-speech (Piper) for voice mode.

Mirrors the rootless-Ollama pattern in apiserver: heavy assets live OUTSIDE the
~110MB app bundle, under ``~/.local/share/agentica/voice``, and download on first
use. Everything here degrades gracefully: if faster-whisper / Piper aren't
installed or provisioned, the functions raise ``VoiceUnavailable`` and the voice
gateway falls back to the browser / text path instead of crashing.

STT: faster-whisper (CTranslate2) — a pip wheel, models auto-download from HF.
TTS: Piper — a native binary + a per-voice .onnx model, streams raw PCM.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import wave
from pathlib import Path

# Serialize first-use model construction (reached from the gateway's executor):
# without it two concurrent first-uses each build a multi-hundred-MB model.
_load_lock = threading.Lock()

VOICE_HOME = Path(os.environ.get("AGENTICA_VOICE_HOME") or
                  os.path.expanduser("~/.local/share/agentica/voice"))
WHISPER_DIR = VOICE_HOME / "whisper"
PIPER_DIR = VOICE_HOME / "piper"
KOKORO_DIR = VOICE_HOME / "kokoro"

DEFAULT_WHISPER_MODEL = os.environ.get("AGENTICA_WHISPER_MODEL", "base.en")
# Kokoro is the default natural (open-weight, Apache-2.0) voice. af_heart is the
# flagship; override with AGENTICA_KOKORO_VOICE (e.g. am_michael, af_bella).
DEFAULT_KOKORO_VOICE = os.environ.get("AGENTICA_KOKORO_VOICE", "af_heart")
KOKORO_MODEL = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = KOKORO_DIR / "voices-v1.0.bin"
KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)
# Pinned SHA-256 of the model assets — verified before the onnx is loaded so a
# re-uploaded/compromised release asset can't be silently used.
KOKORO_MODEL_SHA256 = "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
KOKORO_VOICES_SHA256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"


class VoiceUnavailable(RuntimeError):
    """STT/TTS asset or dependency is missing (caller should fall back)."""


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def _faster_whisper_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _piper_binary() -> Path | None:
    # A provisioned piper binary, or one on PATH.
    cand = PIPER_DIR / ("piper.exe" if os.name == "nt" else "piper")
    if cand.exists():
        return cand
    found = shutil.which("piper")
    return Path(found) if found else None


def _piper_voice() -> Path | None:
    if not PIPER_DIR.exists():
        return None
    voices = sorted(PIPER_DIR.glob("*.onnx"))
    return voices[0] if voices else None


def _kokoro_ready() -> bool:
    try:
        import kokoro_onnx  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return KOKORO_MODEL.exists() and KOKORO_VOICES.exists()


def voice_status() -> dict:
    piper_bin = _piper_binary()
    piper_voice = _piper_voice()
    whisper_ok = _faster_whisper_installed()
    kokoro_ok = _kokoro_ready()
    remote = remote_tts_status()  # None unless AGENTICA_TTS_URL is set + reachable
    if remote:
        engine = f"remote:{remote.get('engine', '?')}"
    elif kokoro_ok:
        engine = "kokoro"
    elif piper_voice:
        engine = "piper"
    else:
        engine = None
    return {
        "whisper_installed": whisper_ok,
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "kokoro_installed": kokoro_ok,
        "kokoro_voice": DEFAULT_KOKORO_VOICE if kokoro_ok else None,
        "piper_installed": piper_bin is not None,
        "piper_voice": piper_voice.name if piper_voice else None,
        "remote_tts": remote,  # {engine, sample_rate, device} or null
        "stt_ready": whisper_ok,
        # remote expressive > Kokoro (natural) > Piper (fallback).
        "tts_ready": bool(remote) or kokoro_ok or (piper_bin is not None and piper_voice is not None),
        "tts_engine": engine,
        "voice_home": str(VOICE_HOME),
    }


# --------------------------------------------------------------------------- #
# STT — faster-whisper
# --------------------------------------------------------------------------- #
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _load_lock:  # double-checked: don't build two copies on concurrent first use
        if _whisper_model is not None:
            return _whisper_model
        return _build_whisper()


def _build_whisper():
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        raise VoiceUnavailable(
            "faster-whisper is not installed (pip install faster-whisper)"
        ) from exc
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    # int8 on CPU is fast + small; downloads the model into WHISPER_DIR on first use.
    _whisper_model = WhisperModel(
        DEFAULT_WHISPER_MODEL, device="cpu", compute_type="int8",
        download_root=str(WHISPER_DIR),
    )
    return _whisper_model


def transcribe_pcm16(pcm: bytes, sample_rate: int = 16000) -> str:
    """Transcribe little-endian PCM16 mono audio to text."""
    model = _load_whisper()
    import numpy as np  # faster-whisper pulls numpy in

    audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
    if sample_rate != 16000:
        # crude resample to 16k (whisper's native rate)
        ratio = 16000 / float(sample_rate)
        idx = (np.arange(int(len(audio) * ratio)) / ratio).astype("int64")
        idx = idx[idx < len(audio)]
        audio = audio[idx]
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    return " ".join(seg.text for seg in segments).strip()


# --------------------------------------------------------------------------- #
# TTS — Kokoro (natural, default) with a Piper fallback
# --------------------------------------------------------------------------- #
_kokoro = None


def _load_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    with _load_lock:  # double-checked: don't build two copies on concurrent first use
        if _kokoro is not None:
            return _kokoro
        try:
            from kokoro_onnx import Kokoro
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable("kokoro-onnx not installed (pip install kokoro-onnx)") from exc
        if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
            raise VoiceUnavailable("Kokoro model/voices not provisioned")
        _kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        return _kokoro


def synthesize_kokoro_pcm(text: str, voice: str | None = None) -> tuple[bytes, int]:
    """Synthesize ``text`` to (pcm16_bytes, sample_rate) with Kokoro (24kHz)."""
    import numpy as np

    k = _load_kokoro()
    samples, sr = k.create(text, voice=voice or DEFAULT_KOKORO_VOICE, speed=1.0, lang="en-us")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    return pcm, int(sr)


def synthesize_piper_pcm(text: str) -> tuple[bytes, int]:
    piper = _piper_binary()
    voice = _piper_voice()
    if piper is None or voice is None:
        raise VoiceUnavailable("Piper binary or voice model not provisioned")
    proc = subprocess.run(
        [str(piper), "--model", str(voice), "--output_file", "-"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    )
    return _wav_to_pcm(proc.stdout)


# --------------------------------------------------------------------------- #
# Remote expressive TTS (a GPU box running agentica_core.tts_server, reached
# over an SSH tunnel). Set AGENTICA_TTS_URL to route the voice through it.
# --------------------------------------------------------------------------- #
def remote_tts_url() -> str | None:
    return os.environ.get("AGENTICA_TTS_URL")


def remote_tts_status(timeout: float = 1.5) -> dict | None:
    import urllib.request

    url = remote_tts_url()
    if not url:
        return None
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def synthesize_remote_pcm(text: str, **kw) -> tuple[bytes, int]:
    import urllib.request

    url = remote_tts_url()
    if not url:
        raise VoiceUnavailable("AGENTICA_TTS_URL not set")
    payload = json.dumps({"text": text, **kw}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/tts", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        wav = r.read()
    return _wav_to_pcm(wav)


def synthesize_pcm(text: str) -> tuple[bytes, int]:
    """Synthesize to (pcm16_bytes, sample_rate). Order of preference:
    remote expressive (if AGENTICA_TTS_URL reachable) -> Kokoro (natural) -> Piper."""
    if remote_tts_url():
        try:
            return synthesize_remote_pcm(text)
        except Exception:  # noqa: BLE001 - remote down -> fall back to local
            pass
    try:
        return synthesize_kokoro_pcm(text)
    except VoiceUnavailable:
        return synthesize_piper_pcm(text)


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    return frames, sr


# --------------------------------------------------------------------------- #
# Provisioning (best-effort; reports progress, never raises)
# --------------------------------------------------------------------------- #
PIPER_RELEASES = {
    # piper release tarballs (extract FLAT: piper binary + espeak-ng-data)
    "darwin": "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_aarch64.tar.gz",
    "linux": "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz",
}
# A small, good-quality default English voice.
PIPER_VOICE_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
)
PIPER_VOICE_FILES = ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"]


def install_voice(progress=lambda m: None) -> bool:
    """Provision Whisper (pip) + Piper (binary + voice). Best-effort; returns ok."""
    ok = True
    VOICE_HOME.mkdir(parents=True, exist_ok=True)
    # Whisper: ensure the wheel is importable (the model itself lazy-downloads).
    if not _faster_whisper_installed():
        progress("faster-whisper not installed; run: pip install faster-whisper")
        ok = False
    else:
        progress("faster-whisper (STT) available")

    # Kokoro: the natural default voice (model + voices download here).
    try:
        import kokoro_onnx  # noqa: F401
        if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
            _download_kokoro(progress)
        else:
            progress("kokoro voice present")
    except ImportError:
        progress("kokoro-onnx not installed; run: pip install kokoro-onnx")
        ok = False
    except Exception as exc:  # noqa: BLE001
        progress(f"kokoro download failed: {exc}")
        ok = False

    # Piper binary
    if _piper_binary() is None:
        try:
            _download_piper(progress)
        except Exception as exc:  # noqa: BLE001
            progress(f"piper download failed: {exc}")
            ok = False
    else:
        progress("piper binary present")

    # Piper voice
    if _piper_voice() is None:
        try:
            _download_piper_voice(progress)
        except Exception as exc:  # noqa: BLE001
            progress(f"piper voice download failed: {exc}")
            ok = False
    else:
        progress("piper voice present")
    return ok


def _download_verified(url: str, dest: Path, progress, *, sha256: str | None = None, label: str = "") -> None:
    """Download to a temp file, verify sha256 (if pinned), then atomically rename.
    Never leaves a partial/poisoned file at ``dest``; never chmod/loads an
    unverified asset when a hash is pinned."""
    import hashlib
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    progress(f"downloading {label or dest.name}…")
    urllib.request.urlretrieve(url, tmp)
    if sha256:
        h = hashlib.sha256()
        with open(tmp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise VoiceUnavailable(f"checksum mismatch for {dest.name}: expected {sha256[:12]}…, got {got[:12]}…")
    else:
        progress(f"warning: no pinned checksum for {dest.name} (trust-on-first-use)")
    os.replace(tmp, dest)


def _download_piper(progress) -> None:
    import platform
    import tarfile

    sysname = "darwin" if platform.system() == "Darwin" else "linux"
    url = PIPER_RELEASES[sysname]
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    tgz = PIPER_DIR / "piper.tar.gz"
    _download_verified(url, tgz, progress, label=f"piper ({sysname})")
    with tarfile.open(tgz, "r:gz") as tf:
        # filter='data' rejects absolute paths / .. members (Zip-Slip / Python
        # 3.12+ default; required explicitly on the 3.11 runtime).
        try:
            tf.extractall(PIPER_DIR, filter="data")
        except TypeError:
            tf.extractall(PIPER_DIR)  # very old tarfile without the filter kwarg
    # piper tarballs extract into a "piper/" subdir — flatten it.
    nested = PIPER_DIR / "piper" / ("piper" if os.name != "nt" else "piper.exe")
    if nested.exists():
        for item in (PIPER_DIR / "piper").iterdir():
            shutil.move(str(item), str(PIPER_DIR / item.name))
    binp = PIPER_DIR / "piper"
    if binp.exists():
        os.chmod(binp, 0o755)
    tgz.unlink(missing_ok=True)
    progress("piper installed")


def _download_piper_voice(progress) -> None:
    import urllib.request

    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    for fname in PIPER_VOICE_FILES:
        progress(f"downloading voice {fname}…")
        urllib.request.urlretrieve(PIPER_VOICE_BASE + fname, PIPER_DIR / fname)
    progress("voice installed")


def _download_kokoro(progress) -> None:
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    if not KOKORO_MODEL.exists():
        _download_verified(KOKORO_MODEL_URL, KOKORO_MODEL, progress,
                           sha256=KOKORO_MODEL_SHA256, label="Kokoro voice model (~310MB)")
    if not KOKORO_VOICES.exists():
        _download_verified(KOKORO_VOICES_URL, KOKORO_VOICES, progress,
                           sha256=KOKORO_VOICES_SHA256, label="Kokoro voices")
    progress("kokoro (natural TTS) installed")
