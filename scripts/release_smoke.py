"""Release smoke test — run in CI BEFORE freezing the backend.

Catches the class of bug that shipped v0.2.0 broken: it asserts that build_app
(against the PINNED AgenticLocal) actually registers the canvas tools and that a
turn streams end-to-end through stream_agent_turn. Uses the deterministic rule
provider, so no Ollama/network is required.
"""
import sys
import tempfile


def main() -> int:
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
        assert required in names, f"canvas tool {required!r} not registered (divergence trap?): {sorted(names)}"

    frames: list = []
    res = stream_agent_turn(app, "hello there", None, frames.append)
    assert res.get("final_answer"), "stream_agent_turn returned no final answer"

    print("release smoke OK: build_app registers get_weather/show_web; stream_agent_turn produced a final answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
