"""Tier 3 — the agentic browser.

The agent drives a headless Chromium (Playwright); each action returns a
``browser_view`` artifact carrying a screenshot, so the canvas shows the live
page and the user's clicks on it become browser actions ("interact as we talk").

Playwright's sync API must be used from the thread that created the browser, so
a single ``BrowserSession`` owns a dedicated worker thread and the agent tools
post commands to it. Everything degrades: no Playwright/Chromium -> the tools
return a friendly "browser unavailable" message instead of raising.

OFF BY DEFAULT (high-risk surface): registered only when AGENTICA_BROWSER_TOOLS
is set. Production should additionally gate these behind policy approval and a
URL allowlist.
"""

from __future__ import annotations

import base64
import os
import queue
import threading
from typing import Any

from agentic_loop.tools import Tool, ToolContext, ToolRegistry

VIEWPORT = {"width": 1280, "height": 800}


class BrowserUnavailable(RuntimeError):
    pass


class BrowserSession:
    """A headless Chromium owned by one worker thread; commands via a queue."""

    def __init__(self):
        self._cmd: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._start_err: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._url = "about:blank"

    def _ensure(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                if not self._ready.wait(timeout=30):
                    raise BrowserUnavailable(self._start_err or "browser did not start")
                return
            self._ready.clear()
            self._start_err = None
            self._thread = threading.Thread(target=self._run, name="agentica-browser", daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout=60):
            raise BrowserUnavailable(self._start_err or "browser did not start")
        if self._start_err:
            raise BrowserUnavailable(self._start_err)

    def _run(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            self._start_err = f"playwright not installed: {exc}"
            self._ready.set()
            return
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport=VIEWPORT)
                self._ready.set()
                while True:
                    op, args, reply = self._cmd.get()
                    if op == "__stop__":
                        break
                    try:
                        reply.put(("ok", self._exec(page, op, args)))
                    except Exception as exc:  # noqa: BLE001
                        reply.put(("err", str(exc)))
                browser.close()
        except Exception as exc:  # noqa: BLE001
            self._start_err = str(exc)
            self._ready.set()

    def _exec(self, page, op: str, args: dict) -> dict:
        if op == "goto":
            page.goto(args["url"], wait_until="domcontentloaded", timeout=20000)
        elif op == "click":
            page.mouse.click(float(args["x"]), float(args["y"]))
            page.wait_for_timeout(600)
        elif op == "type":
            page.keyboard.type(str(args.get("text", "")))
            if args.get("enter"):
                page.keyboard.press("Enter")
            page.wait_for_timeout(600)
        elif op == "snapshot":
            pass
        self._url = page.url
        png = page.screenshot(type="png")
        title = page.title()
        return {
            "url": self._url,
            "title": title,
            "frame": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "dims": {"w": VIEWPORT["width"], "h": VIEWPORT["height"]},
        }

    def call(self, op: str, **args) -> dict:
        self._ensure()
        reply: queue.Queue = queue.Queue()
        self._cmd.put((op, args, reply))
        status, payload = reply.get(timeout=40)
        if status == "err":
            raise BrowserUnavailable(payload)
        return payload


_SESSION = BrowserSession()


def _view_artifact(result: dict, summary: str) -> dict:
    return {
        "summary": summary,
        "_artifact": {
            "kind": "browser_view",
            "title": result.get("title") or result.get("url"),
            "data": {"frame": result["frame"], "url": result["url"], "dims": result["dims"]},
            "interactive": True,
        },
    }


def _guard(fn):
    def wrapped(context: ToolContext, arguments: dict[str, Any]) -> Any:
        try:
            return fn(context, arguments)
        except BrowserUnavailable as exc:
            return {"summary": f"The browser is unavailable: {exc}"}
    return wrapped


@_guard
def open_browser(context, arguments):
    url = str(arguments.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    res = _SESSION.call("goto", url=url)
    return _view_artifact(res, f"Opened {res.get('title') or url}.")


@_guard
def browser_click(context, arguments):
    res = _SESSION.call("click", x=arguments.get("x", 0), y=arguments.get("y", 0))
    return _view_artifact(res, f"Clicked. Now on {res.get('title') or res.get('url')}.")


@_guard
def browser_type(context, arguments):
    res = _SESSION.call("type", text=arguments.get("text", ""), enter=arguments.get("enter", False))
    return _view_artifact(res, "Typed into the page.")


@_guard
def browser_snapshot(context, arguments):
    res = _SESSION.call("snapshot")
    return _view_artifact(res, f"Showing {res.get('title') or res.get('url')}.")


BROWSER_TOOLS = [
    Tool(
        name="open_browser",
        description="Open a web page in a real browser and SHOW it in the canvas. The user can click it; you can also click/type to drive it. Use for browsing, searching, or interacting with any site.",
        parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        handler=open_browser, source="browser", risk_level="medium", ui_component_hint="browser_view",
    ),
    Tool(
        name="browser_click",
        description="Click at pixel coordinates (x, y) in the currently open browser page.",
        parameters={"type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"]},
        handler=browser_click, source="browser", risk_level="medium", ui_component_hint="browser_view",
    ),
    Tool(
        name="browser_type",
        description="Type text into the focused element of the open browser page (set enter=true to submit).",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}, "enter": {"type": "boolean"}},
                    "required": ["text"]},
        handler=browser_type, source="browser", risk_level="medium", ui_component_hint="browser_view",
    ),
    Tool(
        name="browser_snapshot",
        description="Take a fresh screenshot of the currently open browser page.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=browser_snapshot, source="browser", risk_level="low", ui_component_hint="browser_view",
    ),
]


def browser_tools_enabled() -> bool:
    return bool(os.environ.get("AGENTICA_BROWSER_TOOLS"))


def register_browser_tools(registry: ToolRegistry) -> ToolRegistry:
    for tool in BROWSER_TOOLS:
        if tool.name not in registry.names():
            registry.register(tool)
    return registry
