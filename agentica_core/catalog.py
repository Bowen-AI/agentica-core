"""Model + GPU catalog, VRAM fit math, and a validated library of working configs.

This is the "GPU config vs model" engine the framework relies on:

* :data:`GPUS`    -- per-GPU VRAM + architecture facts (FP8 support, NVLink).
* :data:`MODELS`  -- open-weight models with param counts + KV-cache shape.
* :func:`preflight_fit` -- given (model, quant, gpus) decide fit / refuse / how-many.
* :data:`PRESETS` -- a curated *library* of known-good (model, gpu, quant) configs;
  every entry is checked by :func:`validate_library` (and the test-suite) so the
  library only ever advertises configs that actually fit.

Numbers follow the plan's sizing tables. INT4 weight sizes are param-rule
estimates unless a measured AWQ size is pinned; all KV/throughput verdicts are
estimates and must be measured on the real nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

# Popular default model (2026 family). Sizing for it is an ESTIMATE; the preset
# library and `recommend()` fall back to a smaller popular model on small GPUs.
DEFAULT_MODEL = "qwen3.6"


# --------------------------------------------------------------------------- #
# GPUs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GpuSpec:
    key: str
    name: str
    vram_gb: float
    arch: str           # "ada" | "ampere" | "hopper"
    fp8: bool           # native FP8 tensor cores (Ada / Hopper only)
    nvlink: bool        # NVLink available on this card/form factor

    @property
    def usable_gb(self) -> float:
        return self.vram_gb


GPUS: dict[str, GpuSpec] = {
    "l40s": GpuSpec("l40s", "NVIDIA L40S", 48, "ada", True, False),
    "a40": GpuSpec("a40", "NVIDIA A40", 48, "ampere", False, True),  # 2-way NVLink bridge if installed
    "a100-40": GpuSpec("a100-40", "NVIDIA A100 40GB", 40, "ampere", False, True),
    "a100-80": GpuSpec("a100-80", "NVIDIA A100 80GB", 80, "ampere", False, True),
    "h100": GpuSpec("h100", "NVIDIA H100 80GB", 80, "hopper", True, True),
    "rtx4090": GpuSpec("rtx4090", "NVIDIA RTX 4090", 24, "ada", True, False),
    "titan-xp": GpuSpec("titan-xp", "NVIDIA TITAN Xp", 12, "pascal", False, False),  # local benchmark box
    "v100": GpuSpec("v100", "NVIDIA V100 32GB", 32, "volta", False, True),
    "p100": GpuSpec("p100", "NVIDIA P100 16GB", 16, "pascal", False, False),  # old/slow: raise model timeout
}

# Aliases users might write in cluster.yaml (gpu_type).
_GPU_ALIASES = {
    "a100": "a100-80",
    "a100_80": "a100-80",
    "a100_40": "a100-40",
    "l40": "l40s",
    "h100-80": "h100",
}


def get_gpu(key: str) -> GpuSpec:
    k = key.strip().lower()
    k = _GPU_ALIASES.get(k, k)
    if k not in GPUS:
        raise KeyError(f"unknown gpu: {key!r} (known: {', '.join(sorted(GPUS))})")
    return GPUS[k]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    total_b: float                 # total parameters in billions (size by THIS, even for MoE)
    active_b: float                # active params/token (compute only; not VRAM)
    n_layers: int
    n_heads: int                   # attention heads (TP must divide this)
    n_kv_heads: int                # GQA KV heads (drives KV cache)
    head_dim: int
    context_len: int
    kind: str = "llm"              # "llm" | "diffusion"
    family: str = ""
    estimate: bool = False         # True -> param counts/sizes are best-effort (e.g. 2026 models)
    measured_int4_gb: float | None = None  # pinned measured AWQ/GPTQ size, if known
    checkpoints: tuple[str, ...] = ()      # known quant checkpoints / tags
    note: str = ""

    def kv_gb_per_token(self, kv_dtype_bytes: float = 2.0) -> float:
        # KV cache = 2 (K+V) * layers * kv_heads * head_dim * bytes, per token.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * kv_dtype_bytes / 1e9


MODELS: dict[str, ModelSpec] = {
    # ---- small/medium agentic, fit 1 GPU ----
    "gemma3-27b": ModelSpec(
        "gemma3-27b", "Gemma 3 27B (dense, multimodal)", 27, 27, 62, 32, 16, 128, 128_000,
        family="gemma", checkpoints=("ollama: gemma3:27b", "google/gemma-3-27b-it"),
        note="dense, vision-language; strong general agentic model",
    ),
    "qwen3-32b": ModelSpec(
        "qwen3-32b", "Qwen3 32B (dense)", 32, 32, 64, 64, 8, 128, 131_072,
        family="qwen3", checkpoints=("ollama: qwen3:32b", "Qwen/Qwen3-32B"),
    ),
    "qwen3-30b-a3b": ModelSpec(
        "qwen3-30b-a3b", "Qwen3 30B-A3B (MoE)", 30, 3, 48, 32, 4, 128, 131_072,
        family="qwen3", checkpoints=("ollama: qwen3:30b-a3b", "Qwen/Qwen3-30B-A3B"),
        note="MoE: cheap compute, but all experts resident -> size by 30B",
    ),
    "qwen3-8b": ModelSpec(
        "qwen3-8b", "Qwen3 8B (dense)", 8, 8, 36, 32, 8, 128, 131_072,
        family="qwen3", checkpoints=("ollama: qwen3:8b", "Qwen/Qwen3-8B"),
    ),
    "llama3.2-3b": ModelSpec(
        "llama3.2-3b", "Llama 3.2 3B (dense)", 3, 3, 28, 24, 8, 128, 131_072,
        family="llama", checkpoints=("ollama: llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct"),
        note="small, good tool-use; staged on Discovery for the real-cluster demo",
    ),
    "gpt-oss-20b": ModelSpec(
        "gpt-oss-20b", "gpt-oss 20B (MoE, MXFP4)", 21, 3.6, 24, 32, 8, 64, 131_072,
        family="gpt-oss", checkpoints=("ollama: gpt-oss:20b", "openai/gpt-oss-20b"),
        note="ships MXFP4; runs ~16GB on Hopper, more elsewhere (upcast)",
    ),
    # ---- locally-runnable models (benchmarked on the TITAN Xp 12GB) ----
    "qwen3.5-9b": ModelSpec(
        "qwen3.5-9b", "Qwen3.5 9B (local)", 9, 9, 40, 32, 8, 128, 131_072,
        family="qwen3.5", estimate=True, checkpoints=("ollama: qwen3.5:9b",),
        note="local 2026 model; q4 on-disk ~6.6GB",
    ),
    "gemma4-e4b": ModelSpec(
        "gemma4-e4b", "Gemma 4 E4B (local)", 8, 4, 34, 16, 8, 128, 131_072,
        family="gemma4", estimate=True, checkpoints=("ollama: gemma4:e4b",),
        note="local 2026 efficient variant; q4 on-disk ~9.6GB",
    ),
    "gemma4-e2b": ModelSpec(
        "gemma4-e2b", "Gemma 4 E2B (local)", 4, 2, 26, 16, 8, 128, 131_072,
        family="gemma4", estimate=True, checkpoints=("ollama: gemma4:e2b",),
        note="local 2026 efficient variant; q4 on-disk ~7.2GB",
    ),
    # ---- big MoE, 1-2 GPU only with 80GB ----
    "glm-4.5-air": ModelSpec(
        "glm-4.5-air", "GLM-4.5-Air (MoE)", 106, 12, 46, 96, 8, 128, 131_072,
        family="glm", estimate=True, checkpoints=("zai-org/GLM-4.5-Air", "AWQ community"),
        note="best replicable big MoE; INT4 ~60-66GB -> 1xA100-80",
    ),
    "gpt-oss-120b": ModelSpec(
        "gpt-oss-120b", "gpt-oss 120B (MoE, MXFP4)", 117, 5.1, 36, 64, 8, 64, 131_072,
        family="gpt-oss", checkpoints=("ollama: gpt-oss:120b", "openai/gpt-oss-120b"),
        note="MXFP4 ~63GB is Hopper-specific; on Ada/Ampere vLLM upcasts -> bigger",
    ),
    "qwen3.6": ModelSpec(
        "qwen3.6", "Qwen3.6 (2026, ~120B MoE)", 120, 12, 80, 64, 8, 128, 262_144,
        family="qwen3.6", estimate=True,
        checkpoints=("ollama: qwen3.6 (verify tag)", "community FP8/AWQ — verify"),
        note="2026 model; official card often BF16 only; sizing is an ESTIMATE",
    ),
    # ---- heavyweight, multi-GPU / multi-node ----
    "qwen3-235b-a22b": ModelSpec(
        "qwen3-235b-a22b", "Qwen3 235B-A22B (MoE)", 235, 22, 94, 64, 4, 128, 262_144,
        family="qwen3", checkpoints=("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",),
        note="heavyweight; INT4 ~118-145GB (estimate) -> 4xA100-80 TP4",
    ),
    "glm-4.6": ModelSpec(
        "glm-4.6", "GLM-4.6 (357B-A32B MoE)", 357, 32, 92, 96, 8, 128, 200_000,
        family="glm", measured_int4_gb=184,
        checkpoints=("QuantTrio/GLM-4.6-AWQ (184GB, TP8+EP)", "zai-org/GLM-4.5-FP8"),
        note="multi-node only; measured AWQ 184GB",
    ),
    "deepseek-v3": ModelSpec(
        "deepseek-v3", "DeepSeek-V3 (671B-A37B MoE)", 685, 37, 61, 128, 128, 128, 163_840,
        family="deepseek", measured_int4_gb=352,
        checkpoints=("QuixiAI/DeepSeek-V3-AWQ (352GB)", "deepseek-ai/DeepSeek-V3 (FP8 native)"),
        note="not feasible on L40S/A40 PCIe fleet; ~8xA100-80 even at INT4",
    ),
    # ---- diffusion: video + image (served via ComfyUI/diffusers, not ollama/vllm) ----
    "wan2.2-ti2v-5b": ModelSpec(
        "wan2.2-ti2v-5b", "Wan2.2 TI2V-5B (video+image)", 5, 5, 0, 0, 0, 0, 0,
        kind="diffusion", family="wan", checkpoints=("Wan-AI/Wan2.2-TI2V-5B",),
        note="T2V+I2V(+still) 720p; fits 48GB even BF16; ComfyUI",
    ),
    "ltx-video-13b": ModelSpec(
        "ltx-video-13b", "LTX-Video 13B (video)", 13, 13, 0, 0, 0, 0, 0,
        kind="diffusion", family="ltx", checkpoints=("Lightricks/LTX-Video",),
        note="T2V+I2V near-real-time; ~14-18GB FP8; ComfyUI",
    ),
    "flux.1-schnell": ModelSpec(
        "flux.1-schnell", "FLUX.1-schnell 12B (image)", 12, 12, 0, 0, 0, 0, 0,
        kind="diffusion", family="flux", checkpoints=("black-forest-labs/FLUX.1-schnell",),
        note="text-to-image, Apache-2.0; ~12-16GB FP8; ComfyUI",
    ),
}

_MODEL_ALIASES = {
    "qwen3:32b": "qwen3-32b",
    "qwen3:30b-a3b": "qwen3-30b-a3b",
    "qwen3:8b": "qwen3-8b",
    "gemma3:27b": "gemma3-27b",
    "gpt-oss:20b": "gpt-oss-20b",
    "gpt-oss:120b": "gpt-oss-120b",
    "glm-4.5": "glm-4.6",
    "qwen3.5:9b": "qwen3.5-9b",
    "gemma4:e4b": "gemma4-e4b",
    "gemma4:e2b": "gemma4-e2b",
    "llama3.2:3b": "llama3.2-3b",
}


def get_model(key: str) -> ModelSpec | None:
    k = key.strip().lower()
    k = _MODEL_ALIASES.get(k, k)
    return MODELS.get(k)


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #
# Effective bytes-per-parameter for weights (incl. scales/zeros overhead).
_QUANT_BYTES = {
    "fp16": 2.0, "bf16": 2.0, "f16": 2.0, "none": 2.0,
    "fp8": 1.0, "int8": 1.0, "w8a16": 1.0, "q8_0": 1.06,
    "int4": 0.55, "awq": 0.55, "gptq": 0.55, "q4": 0.55, "q4_k_m": 0.58, "w4a16": 0.55,
    "q5_k_m": 0.70, "q6_k": 0.85,
    "mxfp4": 0.55,  # native size; upcast handled in preflight_fit
}


def quant_bytes(quant: str) -> float:
    return _QUANT_BYTES.get(quant.strip().lower(), 2.0)


def choose_quant(model: ModelSpec, gpu: GpuSpec) -> str:
    """Pick a sane default quant for an (model, gpu) pair.

    Ada/Hopper can use FP8 natively. Ampere has no native FP8 -> bf16 for models
    that fit, else INT4 (AWQ/GPTQ). Big models default to INT4 everywhere.
    """
    big = model.total_b >= 40
    if gpu.fp8:  # ada / hopper
        return "fp8" if big else "bf16"
    # ampere
    return "int4" if big else "bf16"


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
OVERHEAD_FLAT_GB = 2.0
OVERHEAD_PER_GPU_GB = 0.6


@dataclass
class Fit:
    ok: bool
    verdict: str                 # "good" | "tight" | "headroom" | "won't fit"
    model: str
    gpu: str
    gpu_count: int
    quant: str
    weights_gb: float
    kv_gb: float
    overhead_gb: float
    total_gb: float
    available_gb: float
    gpus_needed: int
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    estimate: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def estimate_weights_gb(model: ModelSpec, quant: str, gpu: GpuSpec) -> tuple[float, list[str]]:
    warnings: list[str] = []
    q = quant.strip().lower()
    # Measured INT4 sizes win when pinned.
    if q in {"int4", "awq", "gptq", "w4a16", "q4"} and model.measured_int4_gb:
        return model.measured_int4_gb, warnings
    bytes_pp = quant_bytes(q)
    # FP8 caveat: native only on Ada/Hopper. On Ampere it's weight-only (memory
    # savings, no speedup) -- allowed but flag it.
    if q == "fp8" and not gpu.fp8:
        warnings.append(
            f"{gpu.name} ({gpu.arch}) has no native FP8 — FP8 is weight-only here "
            "(memory savings, no speedup). Prefer bf16 or int4."
        )
    # MXFP4 upcast: ~native size only on Hopper; elsewhere vLLM upcasts to bf16.
    if q == "mxfp4" and gpu.arch != "hopper":
        bytes_pp = 2.0
        warnings.append(
            "MXFP4 weights upcast to bf16 off-Hopper — VRAM inflates to ~bf16 size."
        )
    return model.total_b * bytes_pp, warnings


def preflight_fit(
    model: str | ModelSpec,
    gpu: str | GpuSpec,
    gpu_count: int = 1,
    quant: str | None = None,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    batch: int = 1,
    tensor_parallel_size: int | None = None,
    kv_dtype_bytes: float = 2.0,
) -> Fit:
    """Decide whether ``model`` fits on ``gpu_count`` x ``gpu`` and how many are needed."""
    m = model if isinstance(model, ModelSpec) else get_model(model)
    try:
        g = gpu if isinstance(gpu, GpuSpec) else get_gpu(gpu)
    except KeyError:
        # Unknown/unspecified GPU (e.g. a bare ssh alias before `discover`): skip the
        # fit gate rather than crash; the user sets slurm.gpu_type after discovery.
        name = gpu if isinstance(gpu, str) else "?"
        return Fit(
            ok=True, verdict="unknown-gpu", model=str(model), gpu=str(name), gpu_count=gpu_count,
            quant=quant or "?", weights_gb=0, kv_gb=0, overhead_gb=0, total_gb=0,
            available_gb=0, gpus_needed=0,
            warnings=[f"gpu {name!r} not in catalog — fit check skipped; run `slurm-agentic discover` "
                      "and set slurm.gpu_type to size it"],
            message=f"GPU {name!r} unknown; skipping fit check.", estimate=True,
        )
    if m is None:
        # Unknown model -> can't size; surface clearly.
        name = model if isinstance(model, str) else "?"
        return Fit(
            ok=False, verdict="unknown", model=str(name), gpu=g.key, gpu_count=gpu_count,
            quant=quant or "?", weights_gb=0, kv_gb=0, overhead_gb=0, total_gb=0,
            available_gb=g.usable_gb * gpu_count * gpu_memory_utilization, gpus_needed=0,
            warnings=[f"model {name!r} not in catalog — add it or measure on node"],
            message=f"Unknown model {name!r}; cannot estimate fit. Add it to the catalog.",
            estimate=True,
        )

    q = (quant or choose_quant(m, g)).strip().lower()
    warnings: list[str] = []

    if m.kind == "diffusion":
        # Diffusion VRAM is activation-dominated, not param*bytes. Use a coarse rule:
        # fits a 48GB card if total_b <= ~14 at fp8/offload; flag to benchmark.
        weights_gb = m.total_b * (1.0 if q in {"fp8", "int8"} else 2.0)
        total_gb = weights_gb + 6.0  # rough activation headroom
        warnings.append("diffusion model: VRAM is activation-dominated — measure on node (ComfyUI/diffusers)")
        available = g.usable_gb * gpu_memory_utilization  # diffusion: single GPU
        ok = total_gb <= available
        return Fit(
            ok=ok, verdict="good" if ok else "won't fit", model=m.key, gpu=g.key, gpu_count=1,
            quant=q, weights_gb=round(weights_gb, 1), kv_gb=0.0, overhead_gb=6.0,
            total_gb=round(total_gb, 1), available_gb=round(available, 1),
            gpus_needed=1 if ok else 2, warnings=warnings,
            message=_fit_message(ok, m, g, 1, round(total_gb, 1), 1 if ok else 2, q),
            estimate=True,
        )

    weights_gb, w_warn = estimate_weights_gb(m, q, g)
    warnings.extend(w_warn)
    kv_gb = m.kv_gb_per_token(kv_dtype_bytes) * max_model_len * max(1, batch)
    overhead = OVERHEAD_FLAT_GB + OVERHEAD_PER_GPU_GB * gpu_count
    total = weights_gb + kv_gb + overhead
    available = g.usable_gb * gpu_count * gpu_memory_utilization
    per_gpu_avail = g.usable_gb * gpu_memory_utilization
    gpus_needed = max(1, math.ceil((weights_gb + kv_gb + OVERHEAD_FLAT_GB) / per_gpu_avail))
    ok = total <= available

    # Tensor-parallel sanity: TP must divide attention heads; warn on no-NVLink TP.
    tp = tensor_parallel_size or gpu_count
    if tp > 1:
        if m.n_heads and m.n_heads % tp != 0:
            warnings.append(
                f"tensor_parallel_size={tp} does not divide {m.n_heads} attention heads — pick a divisor"
            )
        if not g.nvlink:
            warnings.append(
                f"{g.name} has no NVLink — tensor-parallel is bandwidth-bound over PCIe; prefer pipeline-parallel"
            )
    if m.total_b >= 300 and gpu_count >= 4:
        warnings.append("very large MoE — enable expert-parallel (vLLM --enable-expert-parallel) and multi-node Ray")

    # Verdict.
    if not ok:
        verdict = "won't fit"
    else:
        ratio = total / available
        if ratio > 0.92:
            verdict = "tight"
        elif ratio < 0.45 and gpu_count > 1:
            verdict = "headroom"  # likely overkill — fewer GPUs would do
        else:
            verdict = "good"

    return Fit(
        ok=ok, verdict=verdict, model=m.key, gpu=g.key, gpu_count=gpu_count, quant=q,
        weights_gb=round(weights_gb, 1), kv_gb=round(kv_gb, 1), overhead_gb=round(overhead, 1),
        total_gb=round(total, 1), available_gb=round(available, 1), gpus_needed=gpus_needed,
        warnings=warnings,
        message=_fit_message(ok, m, g, gpu_count, round(total, 1), gpus_needed, q),
        estimate=m.estimate or (q in {"int4", "awq", "gptq"} and m.measured_int4_gb is None),
    )


def _fit_message(ok, m, g, gpu_count, total_gb, gpus_needed, quant) -> str:
    if ok:
        return (
            f"{m.display} @ {quant} fits {gpu_count}x {g.name} "
            f"(~{total_gb}GB used). "
        )
    return (
        f"{m.display} @ {quant} needs ~{total_gb}GB and does NOT fit {gpu_count}x {g.name}. "
        f"Need ~{gpus_needed}x {g.name} (or smaller quant / lower max_model_len / smaller model)."
    )


# --------------------------------------------------------------------------- #
# Library of working configs ("presets")
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Preset:
    name: str
    model: str
    gpu: str
    gpu_count: int
    quant: str
    engine: str = "ollama"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    note: str = ""


# Curated, known-good configs. validate_library() (and the test suite) asserts
# each one actually fits via preflight_fit so the library never lies.
PRESETS: tuple[Preset, ...] = (
    # --- default popular agentic models, 1 GPU ---
    Preset("gemma3-27b@1xl40s", "gemma3-27b", "l40s", 1, "int4", "ollama",
           note="default 1-GPU agentic; INT4 on one L40S"),
    Preset("gemma3-27b@1xa40", "gemma3-27b", "a40", 1, "int4", "ollama"),
    Preset("qwen3-32b@1xl40s", "qwen3-32b", "l40s", 1, "int4", "ollama"),
    Preset("qwen3-32b@1xa40", "qwen3-32b", "a40", 1, "int4", "ollama"),
    Preset("qwen3-32b@1xa100-80", "qwen3-32b", "a100-80", 1, "bf16", "ollama"),
    Preset("qwen3-30b-a3b@1xl40s", "qwen3-30b-a3b", "l40s", 1, "int4", "ollama"),
    Preset("qwen3-8b@1xrtx4090", "qwen3-8b", "rtx4090", 1, "int4", "ollama",
           note="light default for a 24GB card"),
    # --- locally-runnable + benchmarked configs (TITAN Xp 12GB) ---
    Preset("qwen3.5-9b@1xtitan-xp", "qwen3.5-9b", "titan-xp", 1, "int4", "ollama",
           note="local benchmark config"),
    Preset("gemma4-e4b@1xtitan-xp", "gemma4-e4b", "titan-xp", 1, "int4", "ollama",
           note="local benchmark config"),
    Preset("gemma4-e2b@1xtitan-xp", "gemma4-e2b", "titan-xp", 1, "int4", "ollama",
           note="local benchmark config"),
    # --- big-but-feasible ---
    Preset("glm-4.5-air@1xa100-80", "glm-4.5-air", "a100-80", 1, "int4", "vllm",
           note="best replicable big MoE"),
    Preset("gpt-oss-120b@1xh100", "gpt-oss-120b", "h100", 1, "mxfp4", "vllm",
           note="MXFP4 native on Hopper -> ~63GB on one 80GB card"),
    Preset("gpt-oss-120b@1xa100-80", "gpt-oss-120b", "a100-80", 1, "int4", "vllm",
           note="Ampere has no MXFP4 -> use INT4 AWQ community quant"),
    Preset("qwen3.6@2xa100-80", "qwen3.6", "a100-80", 2, "int4", "vllm",
           tensor_parallel_size=2, note="2026 default on big GPUs; verify community quant"),
    # --- heavyweight (multi-GPU, single node) ---
    Preset("qwen3-235b@4xa100-80", "qwen3-235b-a22b", "a100-80", 4, "int4", "vllm",
           tensor_parallel_size=4, note="heavyweight TP4"),
    # --- diffusion: video + image on one 48GB card ---
    Preset("flux-schnell@1xl40s", "flux.1-schnell", "l40s", 1, "fp8", "comfyui",
           note="image gen on one 48GB card"),
    Preset("wan2.2-ti2v-5b@1xl40s", "wan2.2-ti2v-5b", "l40s", 1, "bf16", "comfyui",
           note="video+image gen on one 48GB card"),
    Preset("ltx-video-13b@1xa40", "ltx-video-13b", "a40", 1, "fp8", "comfyui",
           note="fast video gen"),
)


def library() -> list[dict]:
    """Return the preset library with each entry's computed fit verdict."""
    rows = []
    for p in PRESETS:
        fit = preflight_fit(
            p.model, p.gpu, p.gpu_count, quant=p.quant,
            tensor_parallel_size=p.tensor_parallel_size,
        )
        rows.append({
            "name": p.name, "model": p.model, "gpu": p.gpu, "gpus": p.gpu_count,
            "quant": p.quant, "engine": p.engine, "tp": p.tensor_parallel_size,
            "verdict": fit.verdict, "fits": fit.ok, "vram_gb": fit.total_gb,
            "note": p.note,
        })
    return rows


def validate_library() -> list[str]:
    """Return a list of preset names that FAIL their own fit check (should be empty)."""
    bad = []
    for p in PRESETS:
        fit = preflight_fit(
            p.model, p.gpu, p.gpu_count, quant=p.quant,
            tensor_parallel_size=p.tensor_parallel_size,
        )
        if not fit.ok:
            bad.append(p.name)
    return bad


def presets_for_gpu(gpu: str, gpu_count: int | None = None) -> list[Preset]:
    g = get_gpu(gpu).key
    out = []
    for p in PRESETS:
        if get_gpu(p.gpu).key != g:
            continue
        if gpu_count is not None and p.gpu_count > gpu_count:
            continue
        out.append(p)
    return out


def recommend(gpu: str, gpu_count: int = 1, prefer: Iterable[str] = ("qwen3.6", "qwen3", "gemma")) -> Preset | None:
    """Pick the best working preset for the hardware, preferring popular families.

    Prefers larger models that still fit, biased toward the ``prefer`` families
    (default leans to qwen3.6 / qwen3 / gemma per the user's "default to popular").
    """
    candidates = [p for p in presets_for_gpu(gpu, gpu_count)
                  if MODELS.get(p.model) and MODELS[p.model].kind == "llm"]
    if not candidates:
        return None
    prefer = list(prefer)

    def rank(p: Preset) -> tuple:
        fam = MODELS[p.model].family
        fam_rank = next((i for i, f in enumerate(prefer) if fam.startswith(f)), len(prefer))
        return (fam_rank, -MODELS[p.model].total_b)  # preferred family first, then bigger model

    candidates.sort(key=rank)
    return candidates[0]
