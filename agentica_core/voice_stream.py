"""Stream an agentic turn's per-step events as they happen.

Prefer an in-process ``on_event`` callback into ``AgentController.run`` (via
``app.chat(..., on_event=...)``) so voice/chat UI frames arrive without a
SQLite poll. Falls back to tailing ``app.events()`` when the engine does not
accept ``on_event`` (older AgenticLocal pins).
"""

from __future__ import annotations

import inspect
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
        frames.append({"approval": {
            "tool": payload.get("tool"),
            "arguments": payload.get("arguments") or {},
            "reason": payload.get("reason"),
            "blocking": bool(payload.get("blocking")),
        }})
    elif event_type == "tools_unsupported":
        frames.append({"step": {
            "action": "tools_unsupported",
            "error": payload.get("content"),
        }})
    elif event_type == "final_answer":
        frames.append({"final_answer": payload.get("content") or ""})
    return frames


# app.events() -> storage.events_after() pages with a default LIMIT (100). For a
# session with >100 prior events, a single call returns the OLDEST 100, so the
# naive max() lands on the 100th id (not the latest) and the turn replays stale
# rows. Page through to the true high-water mark and drain fully.
_PAGE = 100


def _max_event_id(app: Any, session_id: str) -> int:
    hi = 0
    try:
        while True:
            batch = app.events(session_id=session_id, after_id=hi)
            if not batch:
                break
            hi = max(hi, max(ev["id"] for ev in batch))
            if len(batch) < _PAGE:
                break
    except Exception:  # noqa: BLE001 - never let event tailing break the turn
        return hi
    return hi


def _drain(app: Any, session_id: str, after_id: int, emit: Callable[[dict], None]) -> int:
    """Forward every event newer than ``after_id`` (paging fully); return the new
    high-water id. A single turn can emit >100 rows, so loop until drained."""
    while True:
        try:
            events = app.events(session_id=session_id, after_id=after_id)
        except Exception:  # noqa: BLE001 - transient sqlite lock: retry next poll
            return after_id
        if not events:
            return after_id
        for ev in events:
            after_id = ev["id"]
            for frame in translate_event(ev):
                emit(frame)
        if len(events) < _PAGE:
            return after_id


def _chat_supports(app, *names: str) -> bool:
    chat = getattr(app, "chat", None)
    if chat is None:
        return False
    try:
        params = inspect.signature(chat).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in names)


def _chat(
    app,
    message,
    session_id,
    cancel_event,
    *,
    on_event=None,
    max_steps=None,
    approval_callback=None,
):
    kwargs: dict[str, Any] = {}
    if cancel_event is not None and _chat_supports(app, "cancel_event"):
        kwargs["cancel_event"] = cancel_event
    if on_event is not None and _chat_supports(app, "on_event"):
        kwargs["on_event"] = on_event
    if max_steps is not None and _chat_supports(app, "max_steps"):
        kwargs["max_steps"] = max_steps
    if approval_callback is not None and _chat_supports(app, "approval_callback"):
        kwargs["approval_callback"] = approval_callback
    try:
        return app.chat(message, session_id, **kwargs)
    except TypeError:
        # Older engines that reject unexpected kwargs.
        if cancel_event is not None:
            try:
                return app.chat(message, session_id, cancel_event=cancel_event)
            except TypeError:
                pass
        return app.chat(message, session_id)


def stream_agent_turn(
    app: Any,
    message: str,
    session_id: str | None,
    emit: Callable[[dict], None],
    *,
    poll_interval: float = 0.1,
    cancel_event=None,
    max_steps: int | None = None,
    approval_callback=None,
) -> dict[str, Any]:
    """Run an agentic turn, streaming per-step ``{step}``/``{artifact}`` frames.

    Prefers ``on_event`` (native callback) over SQLite polling when the engine
    supports it. ``emit`` is called for each incremental frame. Returns the
    final ``app.chat`` payload so the caller can send the terminal frame.
    """
    # Degrade gracefully on an engine without the tail surfaces.
    if not (hasattr(app, "events") and hasattr(app, "create_session")):
        result = _chat(
            app, message, session_id, cancel_event,
            max_steps=max_steps, approval_callback=approval_callback,
        )
        result.setdefault("session_id", session_id)
        return result

    if not session_id:
        session_id = app.create_session()

    use_callback = _chat_supports(app, "on_event")
    after_id = 0 if use_callback else _max_event_id(app, session_id)
    box: dict[str, Any] = {}
    seen_lock = threading.Lock()

    def _on_event(ev: dict[str, Any]) -> None:
        with seen_lock:
            for frame in translate_event(ev):
                emit(frame)

    def _worker():
        try:
            box["result"] = _chat(
                app,
                message,
                session_id,
                cancel_event,
                on_event=_on_event if use_callback else None,
                max_steps=max_steps,
                approval_callback=approval_callback,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            box["error"] = exc

    worker = threading.Thread(target=_worker, name="agentica-chat-turn", daemon=True)
    worker.start()
    while worker.is_alive():
        if not use_callback:
            after_id = _drain(app, session_id, after_id, emit)
        time.sleep(poll_interval if not use_callback else min(poll_interval, 0.05))
    if not use_callback:
        # Final drain to catch the tail (the final_answer event + last tool result).
        _drain(app, session_id, after_id, emit)

    if "error" in box:
        raise box["error"]
    result = box.get("result", {}) or {}
    result.setdefault("session_id", session_id)
    return result
