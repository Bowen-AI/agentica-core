"""Tests for the ollama setup / install / model detection helpers.

Emphasis on the user's requirement: install must NEVER raise an error -- failures
are reported gracefully (return False + an ERROR progress message)."""

from agentica_core import apiserver


def _state(model="gemma4:e4b"):
    return apiserver.State(ollama_host="http://127.0.0.1:11434", model=model,
                           workspace="sample_workspace", db_path="/tmp/agentica-setup-test.db")


def test_model_present():
    assert apiserver.model_present("llama3.2:3b", ["llama3.2:3b", "x:y"])
    assert apiserver.model_present("qwen2.5", ["qwen2.5:latest"])         # bare name matches :latest
    assert apiserver.model_present("gemma4", ["gemma4:e4b"])              # bare name matches a tag
    assert not apiserver.model_present("llama3.2:3b", ["gemma4:e4b"])
    assert not apiserver.model_present("x", [])


def test_setup_status(monkeypatch):
    monkeypatch.setattr(apiserver, "ollama_reachable", lambda h, **k: True)
    monkeypatch.setattr(apiserver, "ollama_models", lambda h, **k: ["gemma4:e4b", "qwen3.5:9b"])
    monkeypatch.setattr(apiserver, "find_ollama_bin", lambda: "/usr/bin/ollama")
    s = apiserver.setup_status(_state("gemma4:e4b"))
    assert s["ollama_running"] and s["model_present"] and s["ready"]
    assert s["models"] == ["gemma4:e4b", "qwen3.5:9b"]

    s2 = apiserver.setup_status(_state("not-pulled:7b"))
    assert s2["ollama_running"] and not s2["model_present"] and not s2["ready"]


def test_resolve_model_keeps_configured_when_present(monkeypatch):
    monkeypatch.setattr(apiserver, "ollama_models", lambda h, **k: ["qwen3.5:4b-mlx", "gemma4:e4b"])
    assert _state("qwen3.5:4b-mlx").resolve_model() == "qwen3.5:4b-mlx"


def test_resolve_model_falls_back_to_installed_when_absent(monkeypatch):
    # configured model missing -> prefer same family, then qwen*, then first installed
    monkeypatch.setattr(apiserver, "ollama_models", lambda h, **k: ["gemma4:e4b", "qwen3.5:9b"])
    assert _state("qwen3.5:4b-mlx").resolve_model() == "qwen3.5:9b"   # same qwen family
    monkeypatch.setattr(apiserver, "ollama_models", lambda h, **k: ["gemma4:e4b"])
    assert _state("llama3.2:3b").resolve_model() == "gemma4:e4b"       # first installed


def test_resolve_model_keeps_configured_when_ollama_empty(monkeypatch):
    # Ollama down/empty: don't invent a model; Setup flow guides the user to pull it.
    monkeypatch.setattr(apiserver, "ollama_models", lambda h, **k: [])
    assert _state("qwen3.5:4b-mlx").resolve_model() == "qwen3.5:4b-mlx"


def test_ollama_models_handles_null_models(monkeypatch):
    # ollama returns {"models": null} when none are pulled -- this used to crash
    # ollama_models with TypeError and surface as /api/setup -> 500.
    class FakeResp:
        status = 200
        def read(self):
            return b'{"models": null}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(apiserver.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    assert apiserver.ollama_models("http://x") == []


def test_setup_status_never_500s(monkeypatch):
    # Even if a helper raises unexpectedly, setup_status returns a valid degraded status.
    monkeypatch.setattr(apiserver, "ollama_reachable", lambda h, **k: True)
    monkeypatch.setattr(apiserver, "ollama_models",
                        lambda h, **k: (_ for _ in ()).throw(TypeError("boom")))
    s = apiserver.setup_status(_state("qwen3.5:4b-mlx"))
    assert s["ready"] is False and s["model"] == "qwen3.5:4b-mlx"


def test_setup_status_when_ollama_down(monkeypatch):
    monkeypatch.setattr(apiserver, "ollama_reachable", lambda h, **k: False)
    monkeypatch.setattr(apiserver, "find_ollama_bin", lambda: None)
    s = apiserver.setup_status(_state())
    assert not s["ollama_running"] and not s["ready"] and not s["ollama_installed"]
    assert s["models"] == []


def test_install_returns_true_when_already_installed(monkeypatch):
    monkeypatch.setattr(apiserver, "find_ollama_bin", lambda: "/usr/bin/ollama")
    msgs = []
    assert apiserver.install_ollama_rootless(msgs.append) is True
    assert any("already" in m.lower() for m in msgs)


def test_install_is_graceful_without_zstd(monkeypatch):
    # ollama missing, download works, but no decompressor available -> graceful False, no raise.
    monkeypatch.setattr(apiserver, "find_ollama_bin", lambda: None)
    monkeypatch.setattr(apiserver.urllib.request, "urlretrieve", lambda *a, **k: ("x", None))
    monkeypatch.setattr(apiserver, "_zstd_decompress", lambda src, dst: False)
    msgs = []
    ok = apiserver.install_ollama_rootless(msgs.append)  # must not raise
    assert ok is False
    assert any("ERROR" in m and "zstd" in m for m in msgs)


def test_install_is_graceful_on_download_failure(monkeypatch):
    monkeypatch.setattr(apiserver, "find_ollama_bin", lambda: None)

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(apiserver.urllib.request, "urlretrieve", boom)
    msgs = []
    ok = apiserver.install_ollama_rootless(msgs.append)  # must not raise
    assert ok is False
    assert any("ERROR" in m for m in msgs)


def test_start_ollama_when_already_running(monkeypatch):
    monkeypatch.setattr(apiserver, "ollama_reachable", lambda h, **k: True)
    ok, msg = apiserver.start_ollama("http://127.0.0.1:11434")
    assert ok and "running" in msg


def test_start_ollama_when_not_installed(monkeypatch):
    monkeypatch.setattr(apiserver, "ollama_reachable", lambda h, **k: False)
    monkeypatch.setattr(apiserver, "find_ollama_bin", lambda: None)
    ok, msg = apiserver.start_ollama("http://127.0.0.1:11434")
    assert not ok and "not installed" in msg
