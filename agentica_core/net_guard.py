"""SSRF / unsafe-URL guard for agent network tools.

Agent tools (fetch_url, show_web, open_browser) take model- or user-supplied URLs
and are reachable over the local API. Without a guard they are an SSRF pivot to
loopback (the bundled Ollama on 127.0.0.1:11434, the agent's own :8770/:8771),
link-local cloud metadata (169.254.169.254), and private LAN services. This
module validates the scheme and rejects any URL that resolves to a non-public IP.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeUrl(ValueError):
    """A URL was rejected (bad scheme, or resolves to a non-public address)."""


def _addr_is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def host_is_public(host: str) -> bool:
    """Resolve ``host`` and return True only if EVERY resolved address is public.
    Conservative: an unresolvable host returns False."""
    if not host:
        return False
    # A bare IP literal.
    try:
        ipaddress.ip_address(host)
        return _addr_is_public(host)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:  # noqa: BLE001
        return False
    if not infos:
        return False
    return all(_addr_is_public(info[4][0]) for info in infos)


def require_safe_public_url(url: str, *, allow_http: bool = False) -> str:
    """Return the URL if its scheme is allowed and its host resolves to public
    address(es); else raise UnsafeUrl. (Note: TOCTOU — re-validate at connect
    time for full DNS-rebinding safety; this is the first line of defence.)"""
    url = (url or "").strip()
    parsed = urlparse(url)
    allowed = {"https"} | ({"http"} if allow_http else set())
    if parsed.scheme.lower() not in allowed:
        raise UnsafeUrl(f"only {'/'.join(sorted(allowed))} URLs are allowed")
    host = parsed.hostname or ""
    if not host_is_public(host):
        raise UnsafeUrl(f"refusing a non-public or unresolvable host: {host!r}")
    return url
