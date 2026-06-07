"""Read the user's ~/.ssh/config so the project can target ANY host defined there.

The user picks a server by its ssh alias; OpenSSH already knows the HostName, User,
Port, IdentityFile and ProxyJump (hops) for it, so agentica-core just runs
``ssh <alias>`` and inherits all of that. ``list_hosts`` powers `slurm-agentic hosts`;
``resolve`` uses ``ssh -G`` for the fully-resolved effective config of one alias.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def ssh_config_path() -> Path:
    return Path(os.path.expanduser("~/.ssh/config"))


def list_hosts() -> list[dict]:
    """List literal Host aliases (skipping wildcard patterns) with their configured
    HostName/User/ProxyJump if present in the file."""
    path = ssh_config_path()
    if not path.exists():
        return []
    hosts: list[dict] = []
    current: list[str] = []
    block: dict = {}

    def flush():
        for alias in current:
            if any(c in alias for c in "*?!"):
                continue
            hosts.append({"alias": alias,
                          "hostname": block.get("hostname", alias),
                          "user": block.get("user", ""),
                          "proxy_jump": block.get("proxyjump", "")})

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        key = key.strip().lower()
        value = value.strip()
        if key == "host":
            flush()
            current = value.split()
            block = {}
        elif current:
            if key in {"hostname", "user", "proxyjump", "port"}:
                block[key] = value
    flush()
    # de-dupe by alias, keep first
    seen, out = set(), []
    for h in hosts:
        if h["alias"] not in seen:
            seen.add(h["alias"])
            out.append(h)
    return out


def resolve(alias: str) -> dict:
    """Effective config for an alias via ``ssh -G`` (does not connect)."""
    try:
        out = subprocess.run(["ssh", "-G", alias], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {"alias": alias, "hostname": alias}
    cfg: dict = {"alias": alias}
    for line in out.splitlines():
        k, _, v = line.strip().partition(" ")
        k = k.lower()
        if k in {"hostname", "user", "port", "proxyjump", "identityfile"} and k not in cfg:
            cfg[k] = v.strip()
    return cfg


def known_alias(alias: str) -> bool:
    return any(h["alias"] == alias for h in list_hosts())
