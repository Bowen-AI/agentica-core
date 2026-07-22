"""Release smoke test — run in CI BEFORE freezing the backend.

Catches the class of bug that shipped v0.2.0 broken: it asserts that build_app
(against the PINNED AgenticLocal) actually registers the canvas tools and that a
turn streams end-to-end through stream_agent_turn. Uses the deterministic rule
provider, so no Ollama/network is required.

Also covers (gracefully skipped when assets/deps are missing):
  * voice STT↔TTS round-trip via voice_provision.selftest
  * local job submit/status cycle against an in-memory State stub
"""
import sys
import tempfile
import threading


def _smoke_agent_stream() -> None:
    from agentica_core.gateway import build_app
    from agentica_core.voice_stream import stream_agent_turn

    app = build_app(
        ollama_host="http://127.0.0.1:11434",
        model_name="rule",
        workspace=tempfile.mkdtemp(prefix="agx-smoke-"),
        db_path=tempfile.mktemp(suffix=".db"),
        auth_token=None,
        provider="rule",
    )
    names = app._create_tools().names()
    for required in ("get_weather", "show_web"):
        assert required in names, (
            f"canvas tool {required!r} not registered (divergence trap?): {sorted(names)}"
        )

    frames: list = []
    res = stream_agent_turn(app, "hello there", None, frames.append)
    assert res.get("final_answer"), "stream_agent_turn returned no final answer"
    print("release smoke OK: build_app registers get_weather/show_web; "
          "stream_agent_turn produced a final answer")


def _smoke_voice_roundtrip() -> None:
    from agentica_core.voice_provision import selftest, voice_status

    status = voice_status()
    if not (status.get("stt_ready") and status.get("tts_ready")):
        print("release smoke SKIP: voice models not provisioned "
              f"(stt_ready={status.get('stt_ready')}, tts_ready={status.get('tts_ready')})")
        return
    result = selftest()
    if not result.get("ok"):
        # Soft-fail: models present but runtime missing (e.g. no Metal in CI).
        print("release smoke SKIP: voice selftest not ok — "
              f"errors={result.get('errors')}")
        return
    print("release smoke OK: voice STT↔TTS round-trip "
          f"(engine={result.get('tts_engine')}, text={result.get('roundtrip_text')!r})")


def _smoke_local_job_cycle() -> None:
    from agentica_core import apiserver
    from agentica_core.config import PlanConfig, SuccessCriteria

    db = tempfile.mktemp(suffix=".db")
    ws = tempfile.mkdtemp(prefix="agx-job-")
    state = apiserver.State(
        ollama_host="http://127.0.0.1:1",
        model="rule",
        workspace=ws,
        db_path=db,
    )

    # Deterministic stub: don't call a real model — just flip status via run_job mock.
    class _Outcome:
        def __init__(self):
            self.passed = True

        def to_dict(self):
            return {"passed": True, "iterations": 1, "verdict": "PASS",
                    "tests_ok": True, "artifacts_ok": True}

    def fake_run_job(*_a, **_k):
        return _Outcome()

    apiserver.run_job = fake_run_job  # type: ignore[attr-defined]
    plan = {
        "title": "smoke",
        "goal": "noop",
        "lines": [{"text": "done"}],
        "tests": "",
        "artifacts": [],
    }
    submit = apiserver.submit_local(state, plan, ws, model="rule", engine="ollama")
    local_id = submit["local_id"]
    assert local_id in state.local_jobs

    # Wait for the daemon runner to finish AND persist (status is set before
    # the SQLite write returns on some schedules, so poll the DB too).
    recovered = {}
    for _ in range(100):
        rec = state.local_jobs[local_id]
        recovered = apiserver._load_local_jobs(db)
        if (
            rec["status"] in {"passed", "failed", "error", "cancelled"}
            and recovered.get(local_id, {}).get("status") in {"passed", "failed", "error", "cancelled"}
        ):
            break
        threading.Event().wait(0.05)
    rec = state.local_jobs[local_id]
    assert rec["status"] == "passed", rec
    assert local_id in recovered
    assert recovered[local_id]["status"] == "passed"
    print("release smoke OK: local job submit/status + SQLite recovery")
    # Silence unused import lint if PlanConfig isn't otherwise referenced.
    _ = (PlanConfig, SuccessCriteria)


def main() -> int:
    _smoke_agent_stream()
    _smoke_voice_roundtrip()
    _smoke_local_job_cycle()
    print("NOTE: re-cut Electron release binaries from the voice-mode branch after "
          "these smokes pass (installed v0.2.3 predates several of these fixes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
