"""Mode 2 -- submit an agentic JOB to SLURM and track it.

``submit`` rsyncs the workspace + plan (and the agentica_core/agentic_loop source,
so the cluster needs no pre-install) to a per-job dir, renders an sbatch that
brings up ollama on the node and runs the Planner->Executor->Auditor loop, then
submits it. ``status`` / ``logs`` / ``cancel`` track it.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import agentic_loop

from . import serving
from .config import ClusterConfig, PlanConfig
from .transport import Transport, TransportError

import agentica_core


def _package_root(module) -> Path:
    return Path(module.__file__).resolve().parent  # the package dir itself


def render_job_sbatch(cluster: ClusterConfig, plan: PlanConfig, remote_jobdir: str) -> str:
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
    body = setup + f"""set -uo pipefail
echo "JOB_NODE=$(hostname)"
export PYTHONPATH="{remote_jobdir}/code:${{PYTHONPATH:-}}"
export OLLAMA_HOST=0.0.0.0:{port}
export OLLAMA_KEEP_ALIVE=24h
ollama serve &
SERVE_PID=$!
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:{port}/api/tags" >/dev/null 2>&1 && break; sleep 1; done
ollama pull "{model.name}" || echo "WARN: ollama pull failed (tag may exist already)"
python -m agentica_core.on_node_runner \\
  --plan "{remote_jobdir}/plan.yaml" \\
  --workspace "{remote_jobdir}/workspace" \\
  --db "{remote_jobdir}/job.db" \\
  --provider ollama --model "{model.name}" \\
  --ollama-host "http://127.0.0.1:{port}" \\
  --model-timeout {model.timeout_s} \\
  --checkpoint-dir "{remote_jobdir}/checkpoints" \\
  --result "{remote_jobdir}/result.json"
RC=$?
kill $SERVE_PID 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit $RC
"""
    return "\n".join(head) + "\n\n" + body


def submit(cluster_path: str, plan_path: str, *, sync_code: bool = True, _print=print) -> int:
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

    # Stage workspace + plan.
    ws = Path(plan.workspace)
    if ws.exists():
        _print(f"[stage] rsync workspace {ws} -> {remote_jobdir}/workspace")
        transport.push_dir(str(ws), f"{remote_jobdir}/workspace").check("rsync workspace")
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
        return _submit_ssh(transport, cluster, plan, remote_jobdir, cluster_path, _print)

    # SLURM: render + sbatch.
    script = render_job_sbatch(cluster, plan, remote_jobdir)
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


def _ssh_runner_script(cluster: ClusterConfig, plan: PlanConfig, remote_jobdir: str) -> str:
    """Run the Planner->Executor->Auditor loop on a plain GPU box (no SLURM)."""
    model = plan.effective_model(cluster)
    port = model.serve_port
    setup = ("\n".join(cluster.setup) + "\n") if cluster.setup else ""
    return f"""#!/bin/bash
set -uo pipefail
cd {remote_jobdir}
{setup}export PYTHONPATH="{remote_jobdir}/code:${{PYTHONPATH:-}}"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
export OLLAMA_HOST=0.0.0.0:{port}
export OLLAMA_KEEP_ALIVE=24h
(curl -sf "http://127.0.0.1:{port}/api/tags" >/dev/null 2>&1 || (nohup ollama serve >{remote_jobdir}/ollama.log 2>&1 &))
for i in $(seq 1 90); do curl -sf "http://127.0.0.1:{port}/api/tags" >/dev/null 2>&1 && break; sleep 1; done
ollama pull "{model.name}" || echo "WARN: ollama pull failed (tag may exist)"
"$PY" -m agentica_core.on_node_runner \\
  --plan "{remote_jobdir}/plan.yaml" --workspace "{remote_jobdir}/workspace" \\
  --db "{remote_jobdir}/job.db" --provider ollama --model "{model.name}" \\
  --ollama-host "http://127.0.0.1:{port}" --model-timeout {model.timeout_s} \\
  --result "{remote_jobdir}/result.json"
echo "JOB_DONE rc=$?" > {remote_jobdir}/done.marker
"""


def _submit_ssh(transport: Transport, cluster: ClusterConfig, plan: PlanConfig,
                remote_jobdir: str, cluster_path: str, _print) -> int:
    transport.exec(serving._write_remote_file(  # noqa: SLF001
        f"{remote_jobdir}/runner.sh", _ssh_runner_script(cluster, plan, remote_jobdir))).check("write runner.sh")
    # Launch detached so it survives the ssh disconnect.
    transport.exec(f"cd {remote_jobdir} && nohup bash runner.sh >runner.log 2>&1 & echo launched").check("launch ssh job")
    job_id = Path(remote_jobdir).name
    _print("")
    _print(f"[submitted] job_id={job_id}  jobdir={remote_jobdir}   (ssh: {cluster.ssh.host})")
    _print(f"  status: agentica job status {cluster_path} --job {job_id} --jobdir {remote_jobdir}")
    _print(f"  logs:   agentica job logs   {cluster_path} --job {job_id} --jobdir {remote_jobdir}")
    return 0


def status(cluster_path: str, job_id: str, jobdir: str | None = None, _print=print) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    # Result file is authoritative for both ssh and finished SLURM jobs.
    if jobdir:
        res = transport.exec(f"cat {jobdir}/result.json 2>/dev/null || true")
        if res.out.strip():
            try:
                data = json.loads(res.out)
                _print(f"job {job_id}: state=COMPLETED")
                _print(f"  result: passed={data.get('passed')} iterations={data.get('iterations')} "
                       f"verdict={data.get('verdict')} tests_ok={data.get('tests_ok')}")
                return 0
            except json.JSONDecodeError:
                pass
        marker = transport.exec(f"cat {jobdir}/done.marker 2>/dev/null || true")
        if marker.out.strip():
            _print(f"job {job_id}: {marker.out.strip()} (finished; no result.json — check logs)")
            return 0
    info = transport.squeue_job(job_id)
    if info:
        _print(f"job {job_id}: state={info.get('state')} node={info.get('nodelist', '-')}")
    else:
        # no scheduler info (ssh box) and no result yet -> still running, or unknown
        sacct = transport.sacct_state(job_id)
        state = "RUNNING" if sacct == "UNKNOWN" else sacct
        _print(f"job {job_id}: state={state} (no result.json yet)")
    return 0


# SLURM states that mean the job ended without our result.json (i.e. it failed before
# writing one). Used to give the UI a terminal "error" instead of polling forever.
_SLURM_TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                        "BOOT_FAIL", "DEADLINE", "PREEMPTED", "COMPLETED"}


def status_struct(cluster_path: str, job_id: str, jobdir: str | None = None) -> dict:
    """Structured status for the JSON API / UI (mirrors the local-job shape):
    {status: passed|failed|error|running, outcome: dict|None, lines: [str], target, job}.
    result.json is authoritative; absent it, squeue/sacct decide running vs failed."""
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    out: dict = {"target": cluster_path, "job": job_id, "status": "running",
                 "outcome": None, "lines": []}
    lines: list[str] = []
    if jobdir:
        res = transport.exec(f"cat {jobdir}/result.json 2>/dev/null || true")
        if res.out.strip():
            try:
                data = json.loads(res.out)
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
        marker = transport.exec(f"cat {jobdir}/done.marker 2>/dev/null || true")
        if marker.out.strip():
            out["status"] = "error"  # finished but never wrote result.json
            out["lines"] = [f"job {job_id}: {marker.out.strip()} (finished; no result.json — check logs)"]
            return out
    info = transport.squeue_job(job_id)
    if info:  # still in the SLURM queue -> queued (PENDING) or running
        st = (info.get("state") or "").upper()
        out["status"] = "queued" if st in {"PENDING", "PD", "CONFIGURING", "CF", "REQUEUED"} else "running"
        reason = info.get("nodelist") or info.get("reason") or "-"
        lines.append(f"job {job_id}: state={info.get('state')} node/reason={reason}")
    else:
        sacct = transport.sacct_state(job_id)
        if sacct in _SLURM_TERMINAL_FAIL:  # ended (incl. COMPLETED) but no result.json above
            out["status"] = "error"
            lines.append(f"job {job_id}: state={sacct} but no result.json — check logs")
        else:
            out["status"] = "running"
            lines.append(f"job {job_id}: state={'RUNNING' if sacct == 'UNKNOWN' else sacct} (no result.json yet)")
    out["lines"] = lines
    return out


def logs(cluster_path: str, job_id: str, jobdir: str | None = None, tail: int = 80, _print=print) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    if not jobdir:
        _print("Pass --jobdir to read logs (printed at submit time).")
        return 1
    res = transport.exec(
        f"tail -n {tail} {jobdir}/job-*.out 2>/dev/null || "      # slurm
        f"tail -n {tail} {jobdir}/runner.log 2>/dev/null || true")  # ssh
    _print(res.out or "(no log yet)")
    return 0


def cancel(cluster_path: str, job_id: str, _print=print) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    res = transport.scancel(job_id)
    _print(f"scancel {job_id}: rc={res.rc} {res.err.strip()}")
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
