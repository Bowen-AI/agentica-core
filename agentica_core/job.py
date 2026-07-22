"""Mode 2 -- submit an agentic JOB to SLURM and track it.

``submit`` rsyncs the workspace + plan (and the agentica_core/agentic_loop source,
so the cluster needs no pre-install) to a per-job dir, renders an sbatch that
brings up ollama on the node and runs the Planner->Executor->Auditor loop, then
submits it. ``status`` / ``logs`` / ``cancel`` track it.
"""

from __future__ import annotations

import json
import shlex
import uuid
from pathlib import Path

import agentic_loop

from . import serving
from .config import ClusterConfig, PlanConfig
from .transport import Transport, TransportError

import agentica_core


def _package_root(module) -> Path:
    return Path(module.__file__).resolve().parent  # the package dir itself


def _shell_path(path: str) -> str:
    """Quote a local/remote path for interpolation into generated shell scripts."""
    return shlex.quote(path)


def _workspace_spec(raw: str, source: str | None) -> tuple[str, str, bool]:
    """Return (source, path, explicit) for a plan workspace.

    workspace_source is the programmatic API. The prefixes make the same
    distinction available to callers which can currently supply only a plan:

    * local:/path  -- stage that local directory into the per-job workspace
    * remote:/path -- run against that existing path on the selected target

    Unprefixed plans remain local-source for backwards compatibility. A missing
    legacy local path keeps the historical empty-workspace behavior; explicitly
    selected paths fail loudly instead of silently running in the wrong directory.
    """
    workspace = str(raw or ".")
    explicit = source is not None
    for prefix in ("local:", "remote:"):
        if workspace.startswith(prefix):
            prefixed_source = prefix[:-1]
            if source is not None and source != prefixed_source:
                raise ValueError(
                    f"workspace_source {source!r} conflicts with {prefixed_source!r} prefix"
                )
            source = prefixed_source
            workspace = workspace[len(prefix):]
            explicit = True
            break
    source = source or "local"
    if source not in {"local", "remote"}:
        raise ValueError("workspace_source must be 'local' or 'remote'")
    if not workspace:
        raise ValueError("workspace path must not be empty")
    return source, workspace, explicit


def _prepare_execution_workspace(
    transport: Transport,
    plan: PlanConfig,
    remote_jobdir: str,
    workspace_source: str | None,
    _print,
) -> str:
    """Stage a local source or validate and return a real remote source path."""
    source, workspace, explicit = _workspace_spec(plan.workspace, workspace_source)
    if source == "remote":
        remote_workspace = transport.expand_home(workspace)
        transport.exec(f"test -d {_shell_path(remote_workspace)}").check("remote workspace")
        _print(f"[workspace] using remote workspace {remote_workspace}")
        return remote_workspace

    local_workspace = Path(workspace).expanduser()
    staged_workspace = f"{remote_jobdir}/workspace"
    if local_workspace.is_dir():
        _print(f"[stage] rsync workspace {local_workspace} -> {staged_workspace}")
        transport.push_dir(str(local_workspace), staged_workspace).check("rsync workspace")
    elif explicit:
        raise TransportError(
            f"local workspace does not exist or is not a directory: {local_workspace}"
        )
    else:
        # Preserve the pre-existing behavior for old plan files which used a
        # non-existent path to request an empty per-job workspace.
        _print(f"[stage] local workspace {local_workspace} not found; using empty workspace")
    return staged_workspace


def render_job_sbatch(
    cluster: ClusterConfig,
    plan: PlanConfig,
    remote_jobdir: str,
    execution_workspace: str | None = None,
) -> str:
    model = plan.effective_model(cluster)
    res = plan.effective_resources(cluster)
    gres = serving.gres_spec(ClusterConfig(name=cluster.name, ssh=cluster.ssh,
                                           slurm=res, model=model,
                                           resources=cluster.resources, gateway=cluster.gateway))
    head = [
        "#!/bin/bash -l",  # login shell so `module` works on the compute node
        f"#SBATCH --job-name={res.job_name_prefix}-job",
        f"#SBATCH --partition={res.partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={res.cpus}",
        f"#SBATCH --mem={res.mem_mb}M",
        f"#SBATCH --time={res.time_minutes}",
        f"#SBATCH --output={remote_jobdir}/job-%j.out",
    ]
    if res.account:
        head.append(f"#SBATCH --account={res.account}")
    head.extend(f"#SBATCH {opt}" for opt in res.extra_sbatch)

    port = model.serve_port
    setup = ("\n".join(cluster.setup) + "\n") if cluster.setup else ""
    if model.engine == "ollama":
        serve = f"""export OLLAMA_HOST=0.0.0.0:{port}
export OLLAMA_KEEP_ALIVE=${{OLLAMA_KEEP_ALIVE:-30m}}
ollama serve &
SERVE_PID=$!
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:{port}/api/tags" >/dev/null 2>&1 && break; sleep 1; done
ollama pull "{model.name}" || echo "WARN: ollama pull failed (tag may exist already)"
RUNNER_PROVIDER_ARGS=(--provider ollama --ollama-host "http://127.0.0.1:{port}")
"""
    else:
        quant_flag = f"--quantization {model.quantization} " if model.quantization in {"awq", "gptq", "fp8"} else ""
        serve = f"""python -m vllm.entrypoints.openai.api_server \\
  --model "{model.name}" \\
  --port {port} --host 0.0.0.0 \\
  --tensor-parallel-size {model.tensor_parallel_size} \\
  --pipeline-parallel-size {model.pipeline_parallel_size} \\
  --gpu-memory-utilization {model.gpu_memory_utilization} \\
  --max-model-len {model.max_model_len} {quant_flag}&
SERVE_PID=$!
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:{port}/v1/models" >/dev/null 2>&1 && break; sleep 2; done
RUNNER_PROVIDER_ARGS=(--provider openai-compatible --api-base "http://127.0.0.1:{port}/v1")
"""
    execution_workspace = execution_workspace or f"{remote_jobdir}/workspace"
    body = setup + f"""set -uo pipefail
echo "JOB_NODE=$(hostname)"
export PYTHONPATH="{remote_jobdir}/code:${{PYTHONPATH:-}}"
{serve}
python -m agentica_core.on_node_runner \\
  --plan "{remote_jobdir}/plan.yaml" \\
  --workspace {_shell_path(execution_workspace)} \\
  --db "{remote_jobdir}/job.db" \\
  --model "{model.name}" \\
  "${{RUNNER_PROVIDER_ARGS[@]}}" \\
  --model-timeout {model.timeout_s} \\
  --checkpoint-dir "{remote_jobdir}/checkpoints" \\
  --result "{remote_jobdir}/result.json"
RC=$?
kill $SERVE_PID 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit $RC
"""
    return "\n".join(head) + "\n\n" + body


def submit(
    cluster_path: str,
    plan_path: str,
    *,
    sync_code: bool = True,
    workspace_source: str | None = None,
    _print=print,
) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    plan = PlanConfig.load(plan_path)

    fit = serving.preflight(cluster, plan.effective_model(cluster))
    _print(f"[preflight] {fit.message}")
    for w in fit.warnings:
        _print(f"[preflight] ! {w}")
    if fit.verdict == "won't fit":
        _print("[preflight] Refusing to submit: model does not fit. Adjust plan/cluster resources.")
        return 2

    transport = Transport.from_cluster(cluster)
    scheduler = serving.detect_scheduler(transport, cluster)
    remote_jobdir = transport.expand_home(f"{cluster.remote_workdir}/job-{uuid.uuid4().hex[:8]}")
    transport.exec(f"mkdir -p {remote_jobdir}/workspace {remote_jobdir}/code").check("mkdir remote jobdir")

    # A local source is copied into the isolated per-job workspace. A remote
    # source is used in-place on the target (for SLURM this assumes shared storage
    # visible from both the login node and compute node).
    execution_workspace = _prepare_execution_workspace(
        transport, plan, remote_jobdir, workspace_source, _print,
    )
    transport.exec(serving._write_remote_file(  # noqa: SLF001 - reuse heredoc helper
        f"{remote_jobdir}/plan.yaml", Path(plan_path).read_text(encoding="utf-8"))).check("write plan.yaml")

    # Stage code so the cluster needs no pre-install.
    if sync_code:
        for mod in (agentica_core, agentic_loop):
            root = _package_root(mod)
            _print(f"[stage] rsync {root.name} -> {remote_jobdir}/code/{root.name}")
            transport.exec(f"mkdir -p {remote_jobdir}/code/{root.name}").check("mkdir code dir")
            transport.push_dir(str(root), f"{remote_jobdir}/code/{root.name}").check(f"rsync {root.name}")

    if scheduler == "ssh":
        return _submit_ssh(
            transport,
            cluster,
            plan,
            remote_jobdir,
            cluster_path,
            _print,
            execution_workspace=execution_workspace,
        )

    # SLURM: render + sbatch.
    script = render_job_sbatch(cluster, plan, remote_jobdir, execution_workspace)
    transport.exec(serving._write_remote_file(  # noqa: SLF001
        f"{remote_jobdir}/job.sbatch", script)).check("write job.sbatch")
    job_id = transport.sbatch(f"{remote_jobdir}/job.sbatch")
    transport.exec(serving._write_remote_file(  # noqa: SLF001 - record jobdir for status/logs
        f"{remote_jobdir}/jobid.txt", job_id)).check("write jobid")

    _print("")
    _print(f"[submitted] job_id={job_id}  jobdir={remote_jobdir}")
    _print(f"  status: agentica job status {cluster_path} --job {job_id} --jobdir {remote_jobdir}")
    _print(f"  logs:   agentica job logs   {cluster_path} --job {job_id} --jobdir {remote_jobdir}")
    _print(f"  cancel: agentica job cancel {cluster_path} --job {job_id}")
    return 0


def _ssh_runner_script(
    cluster: ClusterConfig,
    plan: PlanConfig,
    remote_jobdir: str,
    execution_workspace: str | None = None,
) -> str:
    """Run the Planner->Executor->Auditor loop on a plain GPU box (no SLURM)."""
    model = plan.effective_model(cluster)
    port = model.serve_port
    setup = ("\n".join(cluster.setup) + "\n") if cluster.setup else ""
    if model.engine == "ollama":
        serve = f"""export OLLAMA_HOST=0.0.0.0:{port}
export OLLAMA_KEEP_ALIVE=${{OLLAMA_KEEP_ALIVE:-30m}}
(curl -sf "http://127.0.0.1:{port}/api/tags" >/dev/null 2>&1 || (nohup ollama serve >{remote_jobdir}/ollama.log 2>&1 &))
for i in $(seq 1 90); do curl -sf "http://127.0.0.1:{port}/api/tags" >/dev/null 2>&1 && break; sleep 1; done
ollama pull "{model.name}" || echo "WARN: ollama pull failed (tag may exist)"
RUNNER_PROVIDER_ARGS=(--provider ollama --ollama-host "http://127.0.0.1:{port}")
"""
    else:
        quant_flag = f"--quantization {model.quantization} " if model.quantization in {"awq", "gptq", "fp8"} else ""
        serve = f"""(curl -sf "http://127.0.0.1:{port}/v1/models" >/dev/null 2>&1 || (nohup python -m vllm.entrypoints.openai.api_server \\
  --model "{model.name}" --port {port} --host 0.0.0.0 \\
  --tensor-parallel-size {model.tensor_parallel_size} \\
  --pipeline-parallel-size {model.pipeline_parallel_size} \\
  --gpu-memory-utilization {model.gpu_memory_utilization} \\
  --max-model-len {model.max_model_len} {quant_flag}>{remote_jobdir}/vllm.log 2>&1 &))
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:{port}/v1/models" >/dev/null 2>&1 && break; sleep 2; done
RUNNER_PROVIDER_ARGS=(--provider openai-compatible --api-base "http://127.0.0.1:{port}/v1")
"""
    execution_workspace = execution_workspace or f"{remote_jobdir}/workspace"
    return f"""#!/bin/bash
set -uo pipefail
cd {_shell_path(remote_jobdir)}
{setup}export PYTHONPATH="{remote_jobdir}/code:${{PYTHONPATH:-}}"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
{serve}
"$PY" -m agentica_core.on_node_runner \\
  --plan "{remote_jobdir}/plan.yaml" --workspace {_shell_path(execution_workspace)} \\
  --db "{remote_jobdir}/job.db" --model "{model.name}" \\
  "${{RUNNER_PROVIDER_ARGS[@]}}" --model-timeout {model.timeout_s} \\
  --result "{remote_jobdir}/result.json"
echo "JOB_DONE rc=$?" > {remote_jobdir}/done.marker
"""


def _submit_ssh(transport: Transport, cluster: ClusterConfig, plan: PlanConfig,
                remote_jobdir: str, cluster_path: str, _print,
                execution_workspace: str | None = None) -> int:
    transport.exec(serving._write_remote_file(  # noqa: SLF001
        f"{remote_jobdir}/runner.sh",
        _ssh_runner_script(cluster, plan, remote_jobdir, execution_workspace),
    )).check("write runner.sh")
    # Launch in its own process group where setsid exists and persist the PID.
    # Status/cancel can now distinguish a live plain-SSH job from a dead runner
    # instead of falling through to unavailable SLURM commands forever.
    launch = (
        f"cd {_shell_path(remote_jobdir)} && ("
        "if command -v setsid >/dev/null 2>&1; then "
        "nohup setsid bash runner.sh </dev/null >runner.log 2>&1 & "
        "else nohup bash runner.sh </dev/null >runner.log 2>&1 & fi; "
        "pid=$!; printf '%s\\n' \"$pid\" > runner.pid; "
        "printf 'launched pid=%s\\n' \"$pid\")"
    )
    transport.exec(launch).check("launch ssh job")
    job_id = Path(remote_jobdir).name
    _print("")
    _print(f"[submitted] job_id={job_id}  jobdir={remote_jobdir}   (ssh: {cluster.ssh.host})")
    _print(f"  status: agentica job status {cluster_path} --job {job_id} --jobdir {remote_jobdir}")
    _print(f"  logs:   agentica job logs   {cluster_path} --job {job_id} --jobdir {remote_jobdir}")
    return 0


def status(cluster_path: str, job_id: str, jobdir: str | None = None, _print=print) -> int:
    structured = status_struct(cluster_path, job_id, jobdir=jobdir)
    for line in structured.get("lines") or []:
        _print(line)
    return 0


# SLURM states that mean the job ended without our result.json (i.e. it failed before
# writing one). Used to give the UI a terminal "error" instead of polling forever.
_SLURM_TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                        "BOOT_FAIL", "DEADLINE", "PREEMPTED", "COMPLETED"}


def _recent_job_log(transport: Transport, jobdir: str | None, tail: int = 25) -> list[str]:
    if not jobdir:
        return []
    n = max(1, min(int(tail), 500))
    # SLURM writes job-<id>.out; plain SSH writes runner.log. Keep the command
    # scheduler-neutral so status can expose output even during auto-detection.
    res = transport.exec(
        f"tail -n {n} {_shell_path(jobdir)}/job-*.out 2>/dev/null || "
        f"tail -n {n} {_shell_path(f'{jobdir}/runner.log')} 2>/dev/null || true"
    )
    return [line for line in res.out.rstrip().splitlines() if line]


_PROBE_SEP = "__AGENTICA_SECTION__"


def _probe_jobdir(transport: Transport, jobdir: str, tail: int = 25) -> dict:
    """Read result.json / cancel.marker / done.marker / PID state / log tail in
    ONE remote exec. Status is polled every few seconds per active job; issuing
    4-5 ssh round-trips per poll got connections dropped by rate-limiting
    bastions (observed on the USC jump host)."""
    n = max(1, min(int(tail), 500))
    jd = _shell_path(jobdir)
    pid_file = _shell_path(f"{jobdir}/runner.pid")
    cmd = (
        f"cat {_shell_path(f'{jobdir}/result.json')} 2>/dev/null || true; "
        f"printf '\\n{_PROBE_SEP}\\n'; "
        f"cat {_shell_path(f'{jobdir}/cancel.marker')} 2>/dev/null || true; "
        f"printf '\\n{_PROBE_SEP}\\n'; "
        f"cat {_shell_path(f'{jobdir}/done.marker')} 2>/dev/null || true; "
        f"printf '\\n{_PROBE_SEP}\\n'; "
        f"pid=$(cat {pid_file} 2>/dev/null || true); "
        "case \"$pid\" in ''|*[!0-9]*) echo UNKNOWN;; "
        "*) if kill -0 \"$pid\" 2>/dev/null && "
        "ps -p \"$pid\" -o command= 2>/dev/null | grep -Fq runner.sh; then echo RUNNING; "
        "else echo STOPPED; fi;; esac; "
        f"printf '\\n{_PROBE_SEP}\\n'; "
        f"tail -n {n} {jd}/job-*.out 2>/dev/null || "
        f"tail -n {n} {_shell_path(f'{jobdir}/runner.log')} 2>/dev/null || true"
    )
    parts = transport.exec(cmd).out.split(f"\n{_PROBE_SEP}\n")
    parts += [""] * (5 - len(parts))
    pid_raw = parts[3].strip().splitlines()[-1].upper() if parts[3].strip() else "UNKNOWN"
    return {
        "result": parts[0].strip(),
        "cancel": parts[1].strip(),
        "done": parts[2].strip(),
        "pid_state": pid_raw.lower() if pid_raw in {"RUNNING", "STOPPED", "UNKNOWN"} else "unknown",
        "log": [ln for ln in parts[4].rstrip().splitlines() if ln],
    }


def _append_log_lines(lines: list[str], recent: list[str]) -> None:
    if recent:
        lines.append("--- recent output ---")
        lines.extend(recent)


def status_struct(cluster_path: str, job_id: str, jobdir: str | None = None) -> dict:
    """Structured status for the JSON API / UI (mirrors the local-job shape):
    {status: passed|failed|error|running, outcome: dict|None, lines: [str], target, job}.
    result.json is authoritative; scheduler/PID markers decide the live state."""
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    out: dict = {"target": cluster_path, "job": job_id, "status": "running",
                 "outcome": None, "lines": []}
    lines: list[str] = []
    probe: dict | None = None
    if jobdir:
        probe = _probe_jobdir(transport, jobdir)
        if probe["result"]:
            try:
                data = json.loads(probe["result"])
                out["outcome"] = data
                out["status"] = "passed" if data.get("passed") else "failed"
                lines.append(f"job {job_id}: state=COMPLETED")
                lines.append(f"  result: passed={data.get('passed')} iterations={data.get('iterations')} "
                             f"verdict={data.get('verdict')} tests_ok={data.get('tests_ok')}")
                lines += [str(x) for x in (data.get("log") or [])]
                out["lines"] = lines
                return out
            except json.JSONDecodeError:
                pass
        if probe["cancel"]:
            out["status"] = "cancelled"
            lines.append(f"job {job_id}: {probe['cancel']}")
            _append_log_lines(lines, probe["log"])
            out["lines"] = lines
            return out
        if probe["done"]:
            out["status"] = "error"  # finished but never wrote result.json
            lines.append(
                f"job {job_id}: {probe['done']} "
                "(finished; no result.json — check logs)"
            )
            _append_log_lines(lines, probe["log"])
            out["lines"] = lines
            return out

    scheduler = serving.detect_scheduler(transport, cluster)
    if scheduler == "ssh":
        if not jobdir or probe is None:
            lines.append(f"job {job_id}: state=UNKNOWN (pass jobdir to inspect the SSH runner)")
        else:
            pid_state = probe["pid_state"]
            if pid_state == "running":
                lines.append(f"job {job_id}: state=RUNNING (ssh runner PID is alive)")
            elif pid_state == "stopped":
                out["status"] = "error"
                lines.append(
                    f"job {job_id}: state=STOPPED but no result.json or done.marker — check logs"
                )
            else:
                # Compatibility with jobs launched before runner.pid was added.
                lines.append(
                    f"job {job_id}: state=UNKNOWN (runner PID unavailable; no result yet)"
                )
            _append_log_lines(lines, probe["log"])
        out["lines"] = lines
        return out

    info = transport.squeue_job(job_id)
    if info:  # still in the SLURM queue -> queued (PENDING) or running
        st = (info.get("state") or "").upper()
        out["status"] = "queued" if st in {"PENDING", "PD", "CONFIGURING", "CF", "REQUEUED"} else "running"
        reason = info.get("nodelist") or info.get("reason") or "-"
        lines.append(f"job {job_id}: state={info.get('state')} node/reason={reason}")
    else:
        sacct = transport.sacct_state(job_id)
        base_state = sacct.split("+", 1)[0].upper()
        if base_state == "CANCELLED":
            out["status"] = "cancelled"
            lines.append(f"job {job_id}: state={sacct}")
        elif base_state in _SLURM_TERMINAL_FAIL:  # ended but no result.json above
            out["status"] = "error"
            lines.append(f"job {job_id}: state={sacct} but no result.json — check logs")
        else:
            out["status"] = "running"
            lines.append(f"job {job_id}: state={'RUNNING' if sacct == 'UNKNOWN' else sacct} (no result.json yet)")
    if probe is not None:
        _append_log_lines(lines, probe["log"])
    out["lines"] = lines
    return out


def logs(cluster_path: str, job_id: str, jobdir: str | None = None, tail: int = 80, _print=print) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    if not jobdir:
        _print("Pass --jobdir to read logs (printed at submit time).")
        return 1
    recent = _recent_job_log(transport, jobdir, tail=tail)
    _print("\n".join(recent) if recent else "(no log yet)")
    return 0


def cancel(
    cluster_path: str,
    job_id: str,
    jobdir: str | None = None,
    _print=print,
) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    scheduler = serving.detect_scheduler(transport, cluster)
    if scheduler == "slurm":
        res = transport.scancel(job_id)
        _print(f"scancel {job_id}: rc={res.rc} {res.err.strip()}")
        return 0 if res.ok else 1

    # Plain-SSH jobs use the per-job runner PID, not a scheduler ID. Existing API
    # callers do not pass jobdir, but the generated job ID is the directory name,
    # so reconstruct it under the configured remote work root.
    if not jobdir:
        if (not job_id or job_id in {".", ".."} or Path(job_id).name != job_id
                or not job_id.startswith("job-")):
            _print(f"kill {job_id}: invalid SSH job id")
            return 1
        root = transport.expand_home(cluster.remote_workdir.rstrip("/"))
        jobdir = f"{root}/{job_id}"
    pid_file = _shell_path(f"{jobdir}/runner.pid")
    cancel_file = _shell_path(f"{jobdir}/cancel.marker")
    command = (
        f"pid=$(cat {pid_file} 2>/dev/null || true); "
        "case \"$pid\" in ''|*[!0-9]*) echo 'runner PID unavailable' >&2; exit 2;; esac; "
        "if ! kill -0 \"$pid\" 2>/dev/null; then echo 'runner is not active' >&2; exit 1; fi; "
        "if ! ps -p \"$pid\" -o command= 2>/dev/null | grep -Fq runner.sh; "
        "then echo 'runner PID no longer belongs to this job' >&2; exit 1; fi; "
        "kill -TERM -- -\"$pid\" 2>/dev/null || kill -TERM \"$pid\"; "
        f"printf 'CANCELLED by user\\n' > {cancel_file}; "
        "printf 'terminated pid=%s\\n' \"$pid\""
    )
    res = transport.exec(command)
    detail = res.out.strip() or res.err.strip()
    _print(f"kill {job_id}: rc={res.rc} {detail}")
    return 0 if res.ok else 1


def fetch_artifacts(cluster_path: str, jobdir: str, local_dir: str, _print=print) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    try:
        transport.pull_dir(f"{jobdir}/workspace", local_dir).check("rsync artifacts back")
    except TransportError as exc:
        _print(f"fetch failed: {exc}")
        return 1
    _print(f"artifacts synced to {local_dir}")
    return 0
