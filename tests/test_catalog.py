"""GPU-vs-model fit + validated preset library."""

from agentica_core import catalog as c


def test_library_validates():
    # Every curated preset must actually fit -- the library never lies.
    assert c.validate_library() == []
    assert len(c.PRESETS) >= 12


def test_recommend_defaults_to_popular():
    # Big GPUs -> qwen3.6 (the popular default); small -> a fitting qwen3.
    assert c.recommend("a100-80", 2).model.startswith("qwen3.6")
    assert c.recommend("l40s", 1).model.startswith("qwen3")
    assert c.recommend("a40", 1) is not None


def test_refuses_oversized_model():
    fit = c.preflight_fit("glm-4.6", "a100-80", 2, quant="int4")
    assert not fit.ok
    assert fit.verdict == "won't fit"
    assert fit.gpus_needed > 2
    assert "does NOT fit" in fit.message


def test_fits_small_model_one_gpu():
    fit = c.preflight_fit("qwen3-32b", "l40s", 1, quant="int4")
    assert fit.ok
    assert fit.verdict in {"good", "tight"}


def test_fp8_on_ampere_warns():
    fit = c.preflight_fit("qwen3-32b", "a40", 1, quant="fp8")
    assert any("no native FP8" in w or "FP8" in w for w in fit.warnings)


def test_moe_sized_by_total_params():
    # 30B MoE with 3B active must be sized by 30B (~16-18GB int4), not 3B.
    fit = c.preflight_fit("qwen3-30b-a3b", "l40s", 1, quant="int4")
    assert fit.weights_gb > 12


def test_unknown_model_is_explicit():
    fit = c.preflight_fit("totally-made-up-9000", "l40s", 1)
    assert not fit.ok
    assert fit.verdict == "unknown"
    assert fit.estimate


def test_no_native_fp8_on_a100():
    assert c.get_gpu("a100-80").fp8 is False
    assert c.get_gpu("l40s").fp8 is True
    assert c.get_gpu("h100").fp8 is True


def test_tensor_parallel_head_divisibility_warning():
    # TP=3 does not divide 64 heads -> warn.
    fit = c.preflight_fit("qwen3-235b-a22b", "a100-80", 4, quant="int4", tensor_parallel_size=3)
    assert any("does not divide" in w for w in fit.warnings)
