"""Canvas-producing agent tools for Agentica voice/visual mode.

These tools run inside the same AgenticLocal agent loop as the built-ins, but
they return a reserved ``_artifact`` payload (see ``voice_stream.ARTIFACT_KEY``)
that the backend forwards to the UI's canvas panel as a ``{"artifact": ...}``
frame. The agent / TTS still get a plain ``summary`` string.

Everything lives in agentica-core (bundled in the release) rather than
AgenticLocal so we never depend on engine code that may not be on git ``main``.
``register_voice_tools`` is the extension point: later tiers add ``show_web``
(embedded widget) and ``open_browser``/``browser_*`` (agentic browser) here.
"""

from __future__ import annotations

from typing import Any

from agentic_loop.tools import Tool, ToolContext, ToolRegistry, UrllibWebClient

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> short human label + a coarse icon key the
# UI maps to an emoji/glyph. https://open-meteo.com/en/docs (WMO Weather codes).
_WMO: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "sun"),
    1: ("Mainly clear", "sun"),
    2: ("Partly cloudy", "cloud-sun"),
    3: ("Overcast", "cloud"),
    45: ("Fog", "fog"),
    48: ("Rime fog", "fog"),
    51: ("Light drizzle", "drizzle"),
    53: ("Drizzle", "drizzle"),
    55: ("Dense drizzle", "drizzle"),
    56: ("Freezing drizzle", "sleet"),
    57: ("Freezing drizzle", "sleet"),
    61: ("Light rain", "rain"),
    63: ("Rain", "rain"),
    65: ("Heavy rain", "rain"),
    66: ("Freezing rain", "sleet"),
    67: ("Freezing rain", "sleet"),
    71: ("Light snow", "snow"),
    73: ("Snow", "snow"),
    75: ("Heavy snow", "snow"),
    77: ("Snow grains", "snow"),
    80: ("Rain showers", "rain"),
    81: ("Rain showers", "rain"),
    82: ("Violent rain showers", "rain"),
    85: ("Snow showers", "snow"),
    86: ("Snow showers", "snow"),
    95: ("Thunderstorm", "storm"),
    96: ("Thunderstorm w/ hail", "storm"),
    99: ("Thunderstorm w/ hail", "storm"),
}

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _web(context: ToolContext):
    return context.web_client or UrllibWebClient()


def _clean_location(text: str) -> str:
    """Strip time-words the model often leaves in the location (e.g. spoken
    'weather in London this week' -> location 'London this week'), which break
    geocoding. Keep only the place part."""
    import re

    s = text.strip().strip("?.!,")
    # drop leading filler
    s = re.sub(r"^(the\s+)?(weather|forecast)\s+(in|for|at)\s+", "", s, flags=re.I)
    # drop trailing time expressions
    s = re.sub(
        r"\b(this|next|the)?\s*(week|weekend|month|morning|afternoon|evening|"
        r"tonight|today|tomorrow|now|right now|currently|these days)\b.*$",
        "", s, flags=re.I,
    )
    return s.strip().strip(",").strip()


def _weekday(iso_date: str) -> str:
    # iso_date like "2026-06-07"; avoid importing heavy date libs for a label.
    from datetime import date

    try:
        y, m, d = (int(p) for p in iso_date.split("-"))
        return _WEEKDAYS[date(y, m, d).weekday()]
    except Exception:  # noqa: BLE001
        return ""


def get_weather(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up a multi-day forecast (Open-Meteo, no API key) for a place name.

    Returns a spoken ``summary`` plus a ``weather`` canvas artifact whose
    ``data.days`` drives the forecast chart in the UI.
    """
    location = _clean_location(str(arguments.get("location") or ""))
    if not location:
        return {"summary": "I need a place name to look up the weather."}
    try:
        days_n = int(arguments.get("days", 5))
    except (TypeError, ValueError):
        days_n = 5
    days_n = max(1, min(days_n, 16))
    units = "fahrenheit" if str(arguments.get("units", "")).lower().startswith("f") else "celsius"
    temp_unit = "°F" if units == "fahrenheit" else "°C"

    web = _web(context)
    # 1) Geocode the place name -> lat/lon.
    try:
        geo = web.get_json(
            f"{GEOCODE_URL}?name={_q(location)}&count=1&language=en&format=json"
        )
    except Exception as exc:  # noqa: BLE001
        return {"summary": f"Couldn't reach the weather service ({exc})."}
    results = (geo or {}).get("results") or []
    if not results:
        return {"summary": f"I couldn't find a place called \"{location}\"."}
    place = results[0]
    label = ", ".join(
        p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p
    )
    lat, lon = place.get("latitude"), place.get("longitude")

    # 2) Daily forecast.
    try:
        fc = web.get_json(
            f"{FORECAST_URL}?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
            f"&forecast_days={days_n}&timezone=auto&temperature_unit={units}"
        )
    except Exception as exc:  # noqa: BLE001
        return {"summary": f"Couldn't load the forecast for {label} ({exc})."}

    daily = (fc or {}).get("daily") or {}
    times = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    precip = daily.get("precipitation_probability_max") or []
    codes = daily.get("weathercode") or []

    days: list[dict[str, Any]] = []
    for i, iso in enumerate(times):
        code = codes[i] if i < len(codes) else None
        desc, icon = _WMO.get(int(code), ("", "cloud")) if code is not None else ("", "cloud")
        days.append({
            "date": iso,
            "weekday": _weekday(iso),
            "tmax": tmax[i] if i < len(tmax) else None,
            "tmin": tmin[i] if i < len(tmin) else None,
            "precip": precip[i] if i < len(precip) else None,
            "code": code,
            "desc": desc,
            "icon": icon,
        })

    summary = _spoken_summary(label, temp_unit, days)
    return {
        "summary": summary,
        "_artifact": {
            "kind": "weather",
            "title": f"{label} — {len(days)}-day forecast",
            "data": {
                "place": label,
                "latitude": lat,
                "longitude": lon,
                "temp_unit": temp_unit,
                "days": days,
            },
            "interactive": True,
        },
    }


def _spoken_summary(label: str, unit: str, days: list[dict[str, Any]]) -> str:
    if not days:
        return f"I couldn't get any forecast data for {label}."
    today = days[0]
    parts = [
        f"In {label} today: {today.get('desc') or 'mixed conditions'}, "
        f"high {_fmt(today.get('tmax'))}{unit}, low {_fmt(today.get('tmin'))}{unit}."
    ]
    if len(days) > 1:
        highs = [d.get("tmax") for d in days if d.get("tmax") is not None]
        if highs:
            parts.append(
                f"Over the next {len(days)} days highs range "
                f"{_fmt(min(highs))} to {_fmt(max(highs))}{unit}."
            )
        wet = [d for d in days if (d.get("precip") or 0) >= 50]
        if wet:
            names = ", ".join(d.get("weekday") or d.get("date") for d in wet)
            parts.append(f"Rain is likely on {names}.")
    return " ".join(parts)


def _fmt(v: Any) -> str:
    try:
        return f"{round(float(v))}"
    except (TypeError, ValueError):
        return "?"


def _q(text: str) -> str:
    from urllib.parse import quote

    return quote(text)


def show_web(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Embed a live, interactive web page in the canvas (Tier 2)."""
    url = str(arguments.get("url") or "").strip()
    title = str(arguments.get("title") or "").strip()
    if not url.lower().startswith("https://"):
        return {"summary": "I can only embed secure (https) pages."}
    return {
        "summary": f"Showing {title or url} in the canvas.",
        "_artifact": {
            "kind": "web_embed",
            "title": title or url,
            "data": {"url": url},
            "interactive": True,
        },
    }


SHOW_WEB_TOOL = Tool(
    name="show_web",
    description=(
        "Embed a live, interactive web page in the canvas so the user can SEE and "
        "CLICK it (maps, dashboards, live sites). Pass a full https URL. Use this "
        "when the user wants to look at or interact with a specific website."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full https:// URL to embed."},
            "title": {"type": "string", "description": "Optional short label."},
        },
        "required": ["url"],
    },
    handler=show_web,
    source="network",
    risk_level="low",
    ui_component_hint="web_embed",
)


WEATHER_TOOL = Tool(
    name="get_weather",
    description=(
        "Get a multi-day weather forecast for a place and SHOW it to the user as a "
        "forecast chart. Use this whenever the user asks about weather, temperature, "
        "rain, or what to wear."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or place name, e.g. 'Boston' or 'Paris, France'.",
            },
            "days": {"type": "integer", "default": 5, "minimum": 1, "maximum": 16},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        },
        "required": ["location"],
    },
    handler=get_weather,
    source="network",
    risk_level="low",
    ui_component_hint="weather_forecast",
)


def register_voice_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register the canvas/visual tools onto an existing registry, in place.

    Tolerant of double-registration (ToolRegistry.register raises on a dup name)
    so it's safe to call on a registry that already has some of them.
    """
    for tool in (WEATHER_TOOL, SHOW_WEB_TOOL):
        if tool.name not in registry.names():
            registry.register(tool)
    # Tier 3 agentic browser — opt-in (AGENTICA_BROWSER_TOOLS), high-risk surface.
    try:
        from .browser_tools import browser_tools_enabled, register_browser_tools

        if browser_tools_enabled():
            register_browser_tools(registry)
    except Exception:  # noqa: BLE001 - never let an optional tool pack break tools
        pass
    return registry
