"""Probe a cluster's resources into a draft cluster.yaml.

MVP: run sinfo / nvidia-smi over ssh and summarize partitions + GPU types into a
draft ``resources:`` manifest the user can paste into cluster.yaml. (The fuller
"simple LLM explores the environment" path -- feeding raw output to the agent to
author the YAML -- is a Slice-6 extension noted in the plan.)
"""

from __future__ import annotations

from . import serving
from .config import ClusterConfig
from .transport import Transport


def discover(cluster_path: str, _print=print) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    _print(f"[discover] probing {cluster.ssh.host} ...")

    who = transport.exec("echo $(hostname) $(whoami)", timeout=30)
    if not who.ok:
        _print(f"[discover] could not connect: {who.err.strip() or who.out.strip()}")
        _print("  (check ~/.ssh/config / agent / VPN for this host)")
        return 1
    _print(f"[discover] connected: {who.out.strip()}")
    scheduler = serving.detect_scheduler(transport, cluster)
    _print(f"[discover] scheduler: {scheduler}")

    partitions: set[str] = set()
    gpu_types: set[str] = set()
    sinfo = None
    if scheduler == "slurm":
        sinfo = transport.exec(["sinfo", "-h", "-o", "%P|%G|%D|%T"], timeout=30)
        for line in sinfo.out.splitlines():
            parts = line.split("|")
            if not parts or not parts[0]:
                continue
            partitions.add(parts[0].strip().rstrip("*"))
            if len(parts) > 1 and "gpu" in parts[1].lower():
                for tok in parts[1].split(","):  # gres like gpu:a100:4
                    bits = tok.split(":")
                    if len(bits) >= 2 and bits[0].strip().lower() == "gpu":
                        gpu_types.add(bits[1].strip())

    smi = transport.exec(["bash", "-lc", "nvidia-smi -L 2>/dev/null | head -8 || true"], timeout=30)

    _print("\n# --- draft cluster.yaml (review + edit) ---")
    _print(f"name: {cluster.ssh.host}")
    _print("ssh:")
    _print(f"  host: {cluster.ssh.host}        # uses your ~/.ssh/config")
    _print(f"scheduler: {scheduler}")
    if scheduler == "slurm":
        _print("resources:")
        _print(f"  partitions: [{', '.join(sorted(partitions)) or '# none found'}]")
        _print(f"  gpu_types:  [{', '.join(sorted(gpu_types)) or '# none found'}]")
    if smi.out.strip():
        _print("# nvidia-smi -L:")
        for line in smi.out.strip().splitlines():
            _print(f"#   {line}")
    elif scheduler == "ssh":
        _print("# (no GPU detected on the login host; on SLURM clusters GPUs live on compute nodes)")
    if sinfo is not None and not sinfo.ok:
        _print(f"# NOTE: sinfo failed: {sinfo.err.strip()}")
    return 0
