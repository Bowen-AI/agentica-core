#!/usr/bin/env python3
"""Measure Agentica's local voice path and an optional native STS model.

The default run exercises the production Whisper + Kokoro round-trip. Pass
``--lfm`` to also try the experimental MLX LFM2.5-Audio 1.5B model. The native
model is deliberately benchmarked with an agent/tool request: conversational
latency alone is not enough to replace Agentica's audited tool loop.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path


DEFAULT_PROMPT = (
    "Use your tools to inspect the current git branch, then create latency-check.txt "
    "containing the branch name and report what you changed."
)
LFM_REPO = "mlx-community/LFM2.5-Audio-1.5B-4bit"


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def benchmark_pipeline(prompt: str, output_dir: Path) -> dict:
    from agentica_core.voice_provision import synthesize_pcm, transcribe_pcm16, voice_status

    started = time.perf_counter()
    pcm, sample_rate = synthesize_pcm(prompt)
    tts_seconds = time.perf_counter() - started
    input_path = output_dir / "agentic-request.wav"
    _write_wav(input_path, pcm, sample_rate)

    started = time.perf_counter()
    transcript = transcribe_pcm16(pcm, sample_rate)
    stt_seconds = time.perf_counter() - started
    return {
        "engine": "agentica-pipeline",
        "status": voice_status(),
        "prompt": prompt,
        "transcript": transcript,
        "tts_seconds": round(tts_seconds, 3),
        "stt_seconds": round(stt_seconds, 3),
        "audio_seconds": round(len(pcm) / 2 / sample_rate, 3),
        "input_wav": str(input_path),
        "agentic": True,
    }


def benchmark_lfm(input_path: Path, output_dir: Path) -> dict:
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.sts.models.lfm_audio import (
        ChatState,
        LFM2AudioModel,
        LFM2AudioProcessor,
        LFMModality,
    )

    load_started = time.perf_counter()
    model = LFM2AudioModel.from_pretrained(LFM_REPO)
    processor = LFM2AudioProcessor.from_pretrained(LFM_REPO)
    load_seconds = time.perf_counter() - load_started

    audio, sample_rate = sf.read(str(input_path))
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Respond concisely with interleaved text and audio.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(mx.array(audio.astype(np.float32)), sample_rate=sample_rate)
    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens = []
    audio_tokens = []
    first_text = None
    first_audio = None
    generate_started = time.perf_counter()
    for token, modality in model.generate_interleaved(**dict(chat), max_new_tokens=512):
        mx.eval(token)
        elapsed = time.perf_counter() - generate_started
        if modality == LFMModality.TEXT:
            first_text = elapsed if first_text is None else first_text
            text_tokens.append(token)
        else:
            first_audio = elapsed if first_audio is None else first_audio
            audio_tokens.append(token)
    generate_seconds = time.perf_counter() - generate_started

    text = "".join(processor.decode_text(token[None]) for token in text_tokens).strip()
    output_path = output_dir / "lfm-response.wav"
    if len(audio_tokens) > 1:
        codes = mx.stack(audio_tokens[:-1], axis=1)[None, :]
        waveform = processor.decode_with_detokenizer(codes)
        sf.write(str(output_path), waveform[0].tolist(), 24000)

    return {
        "engine": "lfm2.5-audio-1.5b-mlx-4bit",
        "model": LFM_REPO,
        "load_seconds": round(load_seconds, 3),
        "first_text_seconds": round(first_text, 3) if first_text is not None else None,
        "first_audio_token_seconds": round(first_audio, 3) if first_audio is not None else None,
        "generation_seconds": round(generate_seconds, 3),
        "text": text,
        "response_wav": str(output_path) if output_path.exists() else None,
        "agentic": False,
        "agentic_note": "Native MLX STS exposes no Agentica tool-call/result loop; it cannot verify or edit the repository.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfm", action="store_true", help="also benchmark the experimental native MLX STS model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="/tmp/agentica-voice-bench")
    parser.add_argument("--repeats", type=int, default=2, help="pipeline runs in one process (first cold, the rest warm)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(max(1, args.repeats)):
        result = benchmark_pipeline(args.prompt, output_dir)
        result["run"] = "cold" if index == 0 else f"warm-{index}"
        results.append(result)
    if args.lfm:
        try:
            results.append(benchmark_lfm(Path(results[0]["input_wav"]), output_dir))
        except Exception as exc:  # noqa: BLE001 - record optional model/download failures
            results.append({
                "engine": "lfm2.5-audio-1.5b-mlx-4bit",
                "model": LFM_REPO,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "agentic": False,
                "agentic_note": (
                    "Experimental native speech-to-speech is never selected as the "
                    "product engine: it has no Agentica tool-call/result loop."
                ),
            })

    payload = {"results": results}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
