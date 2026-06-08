"""Stream an agentic turn's per-step events as they happen.

agentica-core's ``/api/chat/stream`` agentic branch (``apiserver._stream_chat``)
historically called ``AgentServerApp.chat()``, which runs the whole
Planner/tool loop monolithically and only returns at the very end -- so the UI
saw no tool steps (and no visual artifacts) until the turn was over.

``stream_agent_turn`` closes that gap WITHOUT modifying AgenticLocal: it runs
``app.chat()`` in a worker thread and concurrently TAILS the SQLite event log
that ``AgentController`` already writes during the run
(``app.events(session_id, after_id)`` -> ``storage.events_after``). Each new
event is translated into a UI frame (``{"step": ...}`` / ``{"artifact": ...}``)
and handed to ``emit`` as the tool fires.

We deliberately tail SQLite rather than adding a callback into
``AgentController``: the packaged release ``pip install``s AgenticLocal from git
``main``, so we depend only on surfaces already there
(``create_session``, ``events`` -> ``events_after``). If those surfaces are
absent (a much older engine), we degrade gracefully to a single final frame.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

# Reserved key a tool may put in its (dict) result to ship a canvas artifact
# alongside the human/agent-readable summary. See agentica_core/voice_tools.py.
ARTIFACT_KEY = "_artifact"

# Per-tool terminal events -> one UI step each (so the incremental step list and
# the authoritative final `steps` list have matching cardinality, no dedup).
_TERMINAL_TOOL_EVENTS = {"tool_result", "tool_error", "tool_repeated", "approval_required"}


def _short(value: Any, limit: int = 400) -> Any:
    """Trim a tool observation for the lightweight live step (the authoritative,
    untrimmed value still arrives in the terminal ``steps`` payload)."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def artifact_from_result(tool: str | None, result: Any) -> dict[str, Any] | None:
    """Extract a canvas artifact from a tool result, if the tool emitted one.

    Convention: an artifact-producing tool returns a dict containing
    ``{"_artifact": {kind, title, data, interactive?}}``. We stamp provenance
    (``source_tool``) and leave ``id`` to the caller/UI if unset.
    """
    if not isinstance(result, dict):
        return None
    art = result.get(ARTIFACT_KEY)
    if not isinstance(art, dict) or not art.get("kind"):
        return None
    enriched = dict(art)
    enriched.setdefault("source_tool", tool)
    enriched.setdefault("interactive", False)
    return enriched


def translate_event(ev: dict[str, Any]) -> list[dict[str, Any]]:
    """Map one AgenticLocal SQLite event row to zero or more UI frames.

    The ``step`` shape mirrors the UI's ``Step`` interface (``tool_name`` /
    ``action`` / ``error`` / ``observation``) so ChatView renders a streamed
    step exactly like it renders the final ``steps`` list.
    """
    event_type = ev.get("event_type")
    payload = ev.get("payload") or {}
    frames: list[dict[str, Any]] = []

    if event_type == "tool_result":
        result = payload.get("result")
        frames.append({"step": {
            "action": "tool_call",
            "tool_name": payload.get("tool"),
            "observation": _short(result),
        }})
        art = artifact_from_result(payload.get("tool"), result)
        if art is not None:
            frames.append({"artifact": art})
    elif event_type == "tool_repeated":
        frames.append({"step": {
            "action": "tool_call",
            "tool_name": payload.get("tool"),
        }})
    elif event_type == "tool_error":
        frames.append({"step": {
            "action": "tool_error",
            "tool_name": payload.get("tool"),
            "error": payload.get("error"),
        }})
    elif event_type == "approval_required":
        frames.append({"step": {
            "action": "approval_required",
            "tool_name": payload.get("tool"),
            "error": payload.get("reason"),
        }})
    return frames


def _max_event_id(app: Any, session_id: str) -> int:
    try:
        events = app.events(session_id=session_id)
    except Exception:  # noqa: BLE001 - never let event tailing break the turn
        return 0
    return max((ev["id"] for ev in events), default=0)


def _drain(app: Any, session_id: str, after_id: int, emit: Callable[[dict], None]) -> int:
    """Forward every event newer than ``after_id``; return the new high-water id."""
    try:
        events = app.events(session_id=session_id, after_id=after_id)
    except Exception:  # noqa: BLE001 - transient sqlite lock: retry next poll
        return after_id
    for ev in events:
        after_id = ev["id"]
        for frame in translate_event(ev):
            emit(frame)
    return after_id


def stream_agent_turn(
    app: Any,
    message: str,
    session_id: str | None,
    emit: Callable[[dict], None],
    *,
    poll_interval: float = 0.1,
) -> dict[str, Any]:
    """Run an agentic turn, streaming per-step ``{step}``/``{artifact}`` frames.

    ``emit`` is called (from the calling thread) for each incremental frame.
    Returns the final ``app.chat`` payload (``final_answer``/``steps``/
    ``session_id``) so the caller can send the authoritative terminal frame.
    """
    # Degrade gracefully on an engine without the tail surfaces.
    if not (hasattr(app, "events") and hasattr(app, "create_session")):
        result = app.chat(message, session_id)
        result.setdefault("session_id", session_id)
        return result

    if not session_id:
        session_id = app.create_session()
    after_id = _max_event_id(app, session_id)

    box: dict[str, Any] = {}

    def _worker():
        try:
            box["result"] = app.chat(message, session_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            box["error"] = exc

    worker = threading.Thread(target=_worker, name="agentica-chat-turn", daemon=True)
    worker.start()
    while worker.is_alive():
        after_id = _drain(app, session_id, after_id, emit)
        time.sleep(poll_interval)
    # Final drain to catch the tail (the final_answer event + last tool result).
    _drain(app, session_id, after_id, emit)

    if "error" in box:
        raise box["error"]
    result = box.get("result", {}) or {}
    result.setdefault("session_id", session_id)
    return result
