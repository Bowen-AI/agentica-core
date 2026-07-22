"""YAML config + validation for agentica-core.

Two schemas:

* ``cluster.yaml`` -> :class:`ClusterConfig`  (where/how to connect + serving knobs
  + the visible/allowed-resources manifest).
* ``plan.yaml``    -> :class:`PlanConfig`     (what agentic job to run).

PyYAML is the only hard dependency here. Config maps cleanly onto AgenticLocal's
``ModelSelection`` / ``create_controller`` kwargs and onto SLURM resource specs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "qwen3.5:9b"  # real, benchmark-validated default; see catalog.py preset library
DEFAULT_ENGINE = "ollama"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_VLLM_PORT = 8000


class ConfigError(ValueError):
    """Raised when a cluster.yaml / plan.yaml is malformed or missing fields."""


# --------------------------------------------------------------------------- #
# cluster.yaml
# --------------------------------------------------------------------------- #
@dataclass
class SSHConfig:
    """SSH target. ``host`` may be an alias from ~/.ssh/config; user/port/key/proxy
    are all OPTIONAL and only override the config when set, so a bare
    ``{host: discovery.usc.edu}`` uses your existing ~/.ssh/config (User, HostName,
    IdentityFile, ProxyJump hops, ForwardAgent, ...) verbatim."""

    host: str
    user: str | None = None
    port: int | None = None
    key_path: str | None = None
    proxy_jump: str | None = None        # ssh -J hop1[,hop2]; or leave to ssh config's ProxyJump
    options: list[str] = field(default_factory=list)  # extra "-o Name=Value" tokens

    @property
    def target_str(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def expanded_key(self) -> str | None:
        return os.path.expanduser(self.key_path) if self.key_path else None


@dataclass
class SlurmConfig:
    partition: str = "gpu"
    gpu_type: str = "any"
    gpu_count: int = 1
    cpus: int = 8
    mem_mb: int = 32000
    time_minutes: int = 120
    job_name_prefix: str = "agentic"
    account: str | None = None
    extra_sbatch: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    engine: str = DEFAULT_ENGINE  # "ollama" | "vllm" | "mlx" | "mlx-lm"
    name: str = DEFAULT_MODEL
    quantization: str | None = None  # None -> chosen by GPU arch in preflight_fit
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    serve_port: int = DEFAULT_OLLAMA_PORT
    timeout_s: float = 120.0  # per-request HTTP timeout (raise for slow GPUs like P100)

    def __post_init__(self) -> None:
        if self.engine not in {"ollama", "vllm", "mlx", "mlx-lm"}:
            raise ConfigError(
                f"model.engine must be 'ollama', 'vllm', or 'mlx', got {self.engine!r}"
            )
        if self.engine == "vllm" and self.serve_port == DEFAULT_OLLAMA_PORT:
            self.serve_port = DEFAULT_VLLM_PORT


@dataclass
class ResourcesManifest:
    """The 'visible / allowed' resources a user may use (hand-written or filled by discover)."""

    partitions: list[str] = field(default_factory=list)
    gpu_types: list[str] = field(default_factory=list)
    max_gpus: int | None = None
    max_time_minutes: int | None = None


@dataclass
class GatewayConfig:
    bind_host: str = "127.0.0.1"
    port: int = 8765
    tunnel_port: int = 9434


@dataclass
class ClusterConfig:
    name: str
    ssh: SSHConfig
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resources: ResourcesManifest = field(default_factory=ResourcesManifest)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    auth_token: str | None = None
    # remote working dir where jobs/artifacts live (per-job subdirs are created under it)
    remote_workdir: str = "~/.slurm-agentic/jobs"
    # "auto" -> detect sbatch on the host; "slurm" -> use sbatch; "ssh" -> plain GPU box
    scheduler: str = "auto"
    # shell lines run on the node before serving (e.g. `module load`, or put a
    # user-space ollama on PATH). Injected at the top of every sbatch/serve body.
    setup: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterConfig":
        if not isinstance(data, dict):
            raise ConfigError("cluster config root must be a mapping")
        ssh_raw = data.get("ssh")
        if not isinstance(ssh_raw, dict) or not ssh_raw.get("host"):
            raise ConfigError("cluster.ssh requires at least 'host' (an ~/.ssh/config alias is fine)")
        proxy = ssh_raw.get("proxy_jump") or ssh_raw.get("jump")
        if isinstance(proxy, (list, tuple)):
            proxy = ",".join(str(p) for p in proxy)
        ssh = SSHConfig(
            host=str(ssh_raw["host"]),
            user=str(ssh_raw["user"]) if ssh_raw.get("user") else None,  # else from ~/.ssh/config
            port=int(ssh_raw["port"]) if ssh_raw.get("port") else None,
            key_path=ssh_raw.get("key_path"),
            proxy_jump=str(proxy) if proxy else None,
            options=[str(o) for o in (ssh_raw.get("options") or [])],
        )
        slurm = SlurmConfig(**_subset(data.get("slurm") or {}, SlurmConfig))
        model = ModelConfig(**_subset(data.get("model") or {}, ModelConfig))
        resources = ResourcesManifest(**_subset(data.get("resources") or {}, ResourcesManifest))
        gateway = GatewayConfig(**_subset(data.get("gateway") or {}, GatewayConfig))
        auth = data.get("auth") or {}
        return cls(
            name=str(data.get("name", "cluster")),
            ssh=ssh,
            slurm=slurm,
            model=model,
            resources=resources,
            gateway=gateway,
            auth_token=(auth.get("token") if isinstance(auth, dict) else None),
            remote_workdir=str(data.get("remote_workdir", "~/.slurm-agentic/jobs")),
            scheduler=str(data.get("scheduler", "auto")),
            setup=[str(s) for s in (data.get("setup") or [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClusterConfig":
        return cls.from_dict(_read_yaml(path))

    @classmethod
    def from_ssh(cls, alias: str, **overrides: Any) -> "ClusterConfig":
        """Build a default cluster targeting an ~/.ssh/config alias (no yaml needed).

        User/HostName/Port/ProxyJump all come from ~/.ssh/config at connect time, so
        only the alias is required. Refine with `slurm-agentic discover`.
        """
        cfg = cls(name=overrides.pop("name", alias), ssh=SSHConfig(host=alias))
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def resolve(cls, target: str | Path) -> "ClusterConfig":
        """Accept EITHER a cluster.yaml path OR a bare ~/.ssh/config host alias."""
        p = Path(str(target))
        if p.exists() and p.is_file():
            return cls.load(p)
        return cls.from_ssh(str(target))


# --------------------------------------------------------------------------- #
# plan.yaml
# --------------------------------------------------------------------------- #
@dataclass
class JobResources:
    gpus: int | None = None
    cpus: int | None = None
    mem_mb: int | None = None
    time_minutes: int | None = None


@dataclass
class SuccessCriteria:
    tests: str | None = None  # shell command; exit code 0 required
    artifacts: list[str] = field(default_factory=list)
    # --- hardening knobs (all optional, backward compatible) ---
    # If set, the tests command must ALSO print this token to stdout to pass -- closes
    # the "reduce the test to `print('OK')` / `sys.exit(0)`" exit-code-only gaming gap.
    tests_success_token: str | None = None
    # Seeded grader/fixture files (e.g. a spec-test) the agent must NOT alter. They are
    # snapshotted in memory at job start and restored from that pristine copy before each
    # backstop run, so the gate always executes against the original (tamper-evident).
    protect: list[str] = field(default_factory=list)
    # {path: [symbol, ...]} -> the artifact must DEFINE these top-level names (AST-checked),
    # not merely exist -- closes the "empty/stub file satisfies the artifact gate" gap.
    artifact_symbols: dict = field(default_factory=dict)
    # False when the tests command was DRAFTED by the model (not a vetted gate). The
    # backstop treats a passing non-authoritative test as PROVISIONAL, not a clean PASS.
    tests_authoritative: bool = True


@dataclass
class PlanConfig:
    title: str
    goal: str
    workspace: str = "."
    checklist: list[str] = field(default_factory=list)
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    model: ModelConfig | None = None  # defaults to cluster.model
    resources: JobResources = field(default_factory=JobResources)
    max_iterations: int = 4
    max_steps_per_iteration: int = 12
    kind: str = "code"  # "code" | "video" | generic

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanConfig":
        if not isinstance(data, dict):
            raise ConfigError("plan config root must be a mapping")
        if not data.get("goal"):
            raise ConfigError("plan requires a 'goal'")
        model = None
        if isinstance(data.get("model"), dict):
            model = ModelConfig(**_subset(data["model"], ModelConfig))
        sc_raw = data.get("success_criteria") or {}
        success = SuccessCriteria(
            tests=sc_raw.get("tests"),
            artifacts=list(sc_raw.get("artifacts") or []),
            tests_success_token=sc_raw.get("tests_success_token"),
            protect=[str(p) for p in (sc_raw.get("protect") or [])],
            artifact_symbols={
                str(k): [str(s) for s in (v or [])]
                for k, v in (sc_raw.get("artifact_symbols") or {}).items()
            },
        )
        return cls(
            title=str(data.get("title", data["goal"][:60])),
            goal=str(data["goal"]),
            workspace=str(data.get("workspace", ".")),
            checklist=[str(c) for c in (data.get("checklist") or [])],
            success_criteria=success,
            model=model,
            resources=JobResources(**_subset(data.get("resources") or {}, JobResources)),
            max_iterations=int(data.get("max_iterations", 4)),
            max_steps_per_iteration=int(data.get("max_steps_per_iteration", 12)),
            kind=str(data.get("kind", "code")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PlanConfig":
        return cls.from_dict(_read_yaml(path))

    def effective_model(self, cluster: ClusterConfig) -> ModelConfig:
        return self.model or replace(cluster.model)

    def effective_resources(self, cluster: ClusterConfig) -> SlurmConfig:
        r = self.resources
        return replace(
            cluster.slurm,
            gpu_count=r.gpus if r.gpus is not None else cluster.slurm.gpu_count,
            cpus=r.cpus if r.cpus is not None else cluster.slurm.cpus,
            mem_mb=r.mem_mb if r.mem_mb is not None else cluster.slurm.mem_mb,
            time_minutes=r.time_minutes if r.time_minutes is not None else cluster.slurm.time_minutes,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if data is None:
        raise ConfigError(f"empty config file: {p}")
    return data


def _subset(data: dict[str, Any], dc_type: type) -> dict[str, Any]:
    """Keep only keys that are fields of ``dc_type`` (ignore unknown keys gracefully)."""
    valid = {f.name for f in dc_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in valid}
