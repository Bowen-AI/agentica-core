"""Local speech-to-text (Whisper) and text-to-speech (Kokoro/Piper) for voice mode.

Mirrors the rootless-Ollama pattern in apiserver: heavy assets live OUTSIDE the
~110MB app bundle, under ``~/.local/share/agentica/voice``, and download on first
use. Everything here degrades gracefully: if the STT/TTS deps aren't installed
or provisioned, the functions raise ``VoiceUnavailable`` and the voice gateway
reports it (typed turns keep working) instead of crashing.

STT engines (AGENTICA_STT_ENGINE = auto|mlx|faster, default auto):
  * mlx-whisper   — Whisper on the Apple-Silicon GPU (Metal). Measurably faster
                    than CPU decoding on M-series; preferred when importable.
  * faster-whisper — CTranslate2 int8 on CPU. Portable default everywhere else.
TTS: Kokoro (kokoro-onnx, natural open-weight voice) with a Piper fallback.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import threading
import time
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

# Whisper size is engine-aware unless pinned via AGENTICA_WHISPER_MODEL:
# benchmarked on an M-series Mac (Kokoro-synthesized utterances), mlx small.en
# decodes in ~0.26s/utterance — the same latency the old CPU base.en int8 paid —
# with a full model-size accuracy jump; base.en stays the CPU default.
_WHISPER_MODEL_ENV = os.environ.get("AGENTICA_WHISPER_MODEL")
DEFAULT_WHISPER_MODEL = _WHISPER_MODEL_ENV or "base.en"
MLX_WHISPER_MODEL = _WHISPER_MODEL_ENV or "small.en"
# auto: mlx-whisper on Apple-Silicon GPU when importable, else faster-whisper CPU.
STT_ENGINE = (os.environ.get("AGENTICA_STT_ENGINE") or "auto").strip().lower()
# mlx-whisper loads models from the HF hub by repo id; map the faster-whisper
# style short names onto the community MLX conversions.
_MLX_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
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
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


def _mlx_whisper_installed() -> bool:
    # Do not import MLX just to answer /status. In headless/sandboxed macOS
    # sessions its native Metal initializer can terminate the process rather
    # than raise a catchable Python exception.
    import importlib.util

    return importlib.util.find_spec("mlx_whisper") is not None


@functools.lru_cache(maxsize=1)
def _metal_available() -> bool:
    """Probe Metal without importing MLX.

    Importing MLX with no usable Metal device can terminate the whole process
    from native code, so a normal try/except is insufficient. The framework
    probe safely returns a null device in headless/sandboxed macOS sessions.
    """
    import ctypes
    import platform

    if platform.system() != "Darwin":
        return False
    try:
        metal = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/Metal.framework/Metal")
        default_device = metal.MTLCreateSystemDefaultDevice
        default_device.restype = ctypes.c_void_p
        return bool(default_device())
    except Exception:  # noqa: BLE001
        return False


def stt_engine() -> str | None:
    """The STT engine that transcribe_pcm16 will actually use, or None."""
    if STT_ENGINE == "mlx":
        return "mlx" if _mlx_whisper_installed() and _metal_available() else None
    if STT_ENGINE == "faster":
        return "faster" if _faster_whisper_installed() else None
    if _mlx_whisper_installed() and _metal_available():
        return "mlx"
    if _faster_whisper_installed():
        return "faster"
    return None


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


def _cached_hf_file(repo_id: str, filename: str, *, cache_dir: Path | None = None) -> bool:
    """Return whether a complete Hub snapshot file is already local, without
    touching the network. ``try_to_load_from_cache`` also respects HF_HOME and
    HF_HUB_CACHE, unlike a hand-built ``~/.cache`` path."""
    try:
        from huggingface_hub import try_to_load_from_cache

        kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        cached = try_to_load_from_cache(repo_id, filename, **kwargs)
        return isinstance(cached, str) and Path(cached).is_file()
    except Exception:  # noqa: BLE001 - missing hub helper means "not ready"
        return False


def _stt_model_ready(engine: str | None = None) -> bool:
    """Whether the selected STT weights can be loaded without a first-turn
    download. Dependency availability alone is not readiness."""
    selected = engine or stt_engine()
    if selected == "mlx":
        if Path(MLX_WHISPER_MODEL).is_dir():
            return (Path(MLX_WHISPER_MODEL) / "weights.npz").is_file()
        return _cached_hf_file(_mlx_whisper_repo(), "weights.npz")
    if selected == "faster":
        if _whisper_model is not None:
            return True
        if Path(DEFAULT_WHISPER_MODEL).is_dir():
            return (Path(DEFAULT_WHISPER_MODEL) / "model.bin").is_file()
        repo = (DEFAULT_WHISPER_MODEL if "/" in DEFAULT_WHISPER_MODEL
                else f"Systran/faster-whisper-{DEFAULT_WHISPER_MODEL}")
        return _cached_hf_file(repo, "model.bin", cache_dir=WHISPER_DIR)
    return False


def voice_status() -> dict:
    piper_bin = _piper_binary()
    piper_voice = _piper_voice()
    stt = stt_engine()
    stt_ready = bool(stt and _stt_model_ready(stt))
    kokoro_ok = _kokoro_ready()
    if kokoro_ok:
        engine = "kokoro"
    elif piper_bin is not None and piper_voice is not None:
        engine = "piper"
    else:
        engine = None
    return {
        "whisper_installed": stt is not None,
        "whisper_model": MLX_WHISPER_MODEL if stt == "mlx" else DEFAULT_WHISPER_MODEL,
        "stt_engine": stt,  # mlx (Apple GPU) | faster (CPU) | null
        "kokoro_installed": kokoro_ok,
        "kokoro_voice": DEFAULT_KOKORO_VOICE if kokoro_ok else None,
        "piper_installed": piper_bin is not None,
        "piper_voice": piper_voice.name if piper_voice else None,
        "stt_ready": stt_ready,
        # Local-only: Kokoro (natural) > Piper (small fallback).
        "tts_ready": kokoro_ok or (piper_bin is not None and piper_voice is not None),
        "tts_engine": engine,
        "voice_home": str(VOICE_HOME),
    }


def selftest(phrase: str = "Agentica local voice self test, one two three.") -> dict:
    """Prove the local STT+TTS pipeline works end-to-end, in-process (no renderer).

    Synthesizes ``phrase`` with the active TTS engine, transcribes it back with
    Whisper, and reports the round-trip. Never raises — collects errors instead,
    so it can back a ``/api/voice/selftest`` health check and a CLI smoke test.
    """
    import numpy as np

    out: dict = {
        "ok": False, "stt_ok": False, "tts_ok": False,
        "tts_engine": None, "sample_rate": None, "pcm_bytes": 0,
        "duration_s": 0.0, "roundtrip_text": "", "errors": [],
    }
    out["tts_engine"] = voice_status().get("tts_engine")
    pcm, sr = b"", 0
    try:
        pcm, sr = synthesize_pcm(phrase)
        arr = np.frombuffer(pcm, dtype=np.int16)
        out["sample_rate"] = sr
        out["pcm_bytes"] = len(pcm)
        out["duration_s"] = round(arr.size / float(sr), 2) if sr else 0.0
        # Healthy audio = non-empty and not silence.
        out["tts_ok"] = arr.size > 0 and int(np.abs(arr).max()) > 200
        if not out["tts_ok"]:
            out["errors"].append("TTS produced empty/silent audio")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"TTS failed: {type(exc).__name__}: {exc}")
    if pcm:
        try:
            text = transcribe_pcm16(pcm, sample_rate=sr or 24000)
            out["roundtrip_text"] = text
            out["stt_ok"] = bool(text and text.strip())
            if not out["stt_ok"]:
                out["errors"].append("STT produced no transcript")
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"STT failed: {type(exc).__name__}: {exc}")
    out["ok"] = bool(out["stt_ok"] and out["tts_ok"])
    return out


# --------------------------------------------------------------------------- #
# STT — mlx-whisper (Apple-Silicon GPU) or faster-whisper (CPU)
# --------------------------------------------------------------------------- #
_whisper_model = None
_mlx_repo: str | None = None


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


def _mlx_whisper_repo() -> str:
    global _mlx_repo
    if _mlx_repo is None:
        model = MLX_WHISPER_MODEL
        _mlx_repo = _MLX_WHISPER_REPOS.get(model, model)  # allow a raw HF repo id
    return _mlx_repo


def _to_float32_16k(pcm: bytes, sample_rate: int):
    import numpy as np

    audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
    if sample_rate != 16000:
        # crude resample to 16k (whisper's native rate)
        ratio = 16000 / float(sample_rate)
        idx = (np.arange(int(len(audio) * ratio)) / ratio).astype("int64")
        idx = idx[idx < len(audio)]
        audio = audio[idx]
    return audio


def transcribe_pcm16(pcm: bytes, sample_rate: int = 16000) -> str:
    """Transcribe little-endian PCM16 mono audio to text with the active engine."""
    _touch_voice_use()
    engine = stt_engine()
    if engine is None:
        raise VoiceUnavailable(
            "no STT engine installed (pip install mlx-whisper or faster-whisper)")
    audio = _to_float32_16k(pcm, sample_rate)
    if engine == "mlx":
        import mlx_whisper

        # mlx-whisper caches the loaded model per repo internally; the HF weights
        # download on first use. language hint skips detection (a full extra pass).
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=_mlx_whisper_repo(),
            language="en", fp16=True, verbose=None,
        )
        return str(result.get("text", "")).strip()
    model = _load_whisper()
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    return " ".join(seg.text for seg in segments).strip()


def warmup(*, download_missing: bool = True) -> dict:
    """Pre-build the STT + TTS models so the FIRST voice turn doesn't pay lazy
    model construction (measured multi-second on cold start). Called from a
    daemon thread at gateway start. When weights are missing and
    ``download_missing`` is True, trigger provisioning first.
    """
    out = {"stt": False, "tts": False, "downloaded": False}
    selected = stt_engine()
    need_stt = bool(selected) and not _stt_model_ready(selected)
    need_tts = not _kokoro_ready() and not (_piper_binary() and _piper_voice())
    if download_missing and (need_stt or need_tts or selected is None):
        try:
            out["downloaded"] = bool(install_voice(lambda _m: None))
        except Exception:  # noqa: BLE001
            out["downloaded"] = False
    selected = stt_engine()
    if selected and _stt_model_ready(selected):
        try:
            import numpy as np

            silence = np.zeros(1600, dtype=np.int16).tobytes()  # 0.1s @16k
            transcribe_pcm16(silence, 16000)
            out["stt"] = True
        except Exception:  # noqa: BLE001
            pass
    try:
        _load_kokoro()
        out["tts"] = True
    except Exception:  # noqa: BLE001
        # Piper may still be ready even if Kokoro isn't installed.
        if _piper_binary() is not None and _piper_voice() is not None:
            out["tts"] = True
    return out


# Idle unload: release heavy STT/TTS globals after AGENTICA_VOICE_IDLE_S of
# inactivity (default 30 minutes) so memory isn't pinned forever.
_VOICE_IDLE_S = float(os.environ.get("AGENTICA_VOICE_IDLE_S", str(30 * 60)))
_last_voice_use = 0.0
_idle_timer: threading.Timer | None = None
_idle_lock = threading.Lock()


def _touch_voice_use() -> None:
    global _last_voice_use, _idle_timer
    _last_voice_use = time.time()
    with _idle_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        if _VOICE_IDLE_S <= 0:
            return
        _idle_timer = threading.Timer(_VOICE_IDLE_S, unload_voice_models)
        _idle_timer.daemon = True
        _idle_timer.start()


def unload_voice_models() -> dict:
    """Drop cached Whisper/Kokoro handles so RAM can be reclaimed."""
    global _whisper_model, _kokoro, _idle_timer
    with _load_lock:
        _whisper_model = None
        _kokoro = None
    with _idle_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
    return {"unloaded": True}


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


def synthesize_pcm(text: str) -> tuple[bytes, int]:
    """Synthesize locally to ``(pcm16_bytes, sample_rate)``.

    Kokoro is the natural default; Piper is the small, portable fallback. Voice
    mode deliberately never calls a remote/cloud TTS endpoint.
    """
    _touch_voice_use()
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
    """Provision local Whisper + Kokoro/Piper. Best-effort; returns ok."""
    ok = True
    VOICE_HOME.mkdir(parents=True, exist_ok=True)
    # STT: prefer the Apple-GPU engine; ensure a wheel is importable and pre-pull
    # the model weights so the first spoken turn doesn't pay the download.
    engine = stt_engine()
    if engine is None:
        progress("no STT engine installed; run: pip install mlx-whisper (Apple Silicon) "
                 "or pip install faster-whisper")
        ok = False
    else:
        progress(f"STT engine: {'mlx-whisper (Apple GPU)' if engine == 'mlx' else 'faster-whisper (CPU)'}")
        try:
            progress("fetching the speech-recognition model (first time only)…")
            # 0.1 s of PCM16 silence. Keep provisioning independent of NumPy:
            # the selected STT implementation owns any array conversion it needs.
            transcribe_pcm16(b"\0" * 3200, 16000)
            progress("speech-recognition model ready")
        except Exception as exc:  # noqa: BLE001
            progress(f"speech model fetch failed: {exc}")
            ok = False

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
    unverified asset when a hash is pinned. Reports byte-level progress when
    Content-Length is available so the Voice UI can show a real progress bar.
    """
    import hashlib
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    progress(f"downloading {label or dest.name}…")
    req = urllib.request.Request(url, headers={"User-Agent": "agentica-voice"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -1
        h = hashlib.sha256() if sha256 else None
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                if h is not None:
                    h.update(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(100 * done / total)
                    if pct != last_pct and (pct == 100 or pct - last_pct >= 2):
                        last_pct = pct
                        progress({
                            "status": f"downloading {label or dest.name}",
                            "completed": done,
                            "total": total,
                            "pct": pct,
                        })
    if sha256:
        got = h.hexdigest() if h is not None else ""
        if got != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise VoiceUnavailable(
                f"checksum mismatch for {dest.name}: expected {sha256[:12]}…, got {got[:12]}…"
            )
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
