"""Remote model serving: preflight VRAM check + bring up ollama/vLLM on a node.

Mode 1 (gateway) and Mode 2 (job) both bring a model server up inside a SLURM
allocation, then either tunnel to it (gateway) or hit it on localhost (job).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import catalog
from .config import ClusterConfig, ModelConfig
from .transport import Transport, TransportError


@dataclass
class ServeHandle:
    job_id: str
    node: str
    port: int
    engine: str
    model: str
    remote_url: str  # http://<node>:<port> as seen from the login node


def preflight(cluster: ClusterConfig, model: ModelConfig | None = None) -> catalog.Fit:
    """Map cluster config onto the catalog fit check (the 'GPU config vs model' gate)."""
    m = model or cluster.model
    return catalog.preflight_fit(
        m.name,
        cluster.slurm.gpu_type,
        gpu_count=cluster.slurm.gpu_count,
        quant=m.quantization,
        max_model_len=m.max_model_len,
        gpu_memory_utilization=m.gpu_memory_utilization,
        tensor_parallel_size=m.tensor_parallel_size,
    )


def gres_spec(cluster: ClusterConfig) -> str:
    gt = cluster.slurm.gpu_type.strip().lower()
    n = cluster.slurm.gpu_count
    if gt in {"", "any"}:
        return f"gpu:{n}"
    return f"gpu:{gt}:{n}"


def render_serve_sbatch(cluster: ClusterConfig, remote_jobdir: str, model: ModelConfig | None = None) -> str:
    """Render an sbatch script that starts the model server on a GPU node and pulls the model."""
    m = model or cluster.model
    s = cluster.slurm
    sbatch_lines = [
        "#!/bin/bash -l",  # login shell so `module` works on the compute node
        f"#SBATCH --job-name={s.job_name_prefix}-serve",
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --gres={gres_spec(cluster)}",
        f"#SBATCH --cpus-per-task={s.cpus}",
        f"#SBATCH --mem={s.mem_mb}M",
        f"#SBATCH --time={s.time_minutes}",
        f"#SBATCH --output={remote_jobdir}/serve-%j.out",
    ]
    if s.account:
        sbatch_lines.append(f"#SBATCH --account={s.account}")
    sbatch_lines.extend(f"#SBATCH {opt}" for opt in s.extra_sbatch)

    if m.engine == "ollama":
        body = _ollama_serve_body(m)
    else:
        body = _vllm_serve_body(cluster, m)

    setup = ("\n".join(cluster.setup) + "\n") if cluster.setup else ""
    return "\n".join(sbatch_lines) + "\n\n" + setup + body + "\n"


def _ollama_serve_body(m: ModelConfig) -> str:
    return f"""set -uo pipefail
echo "SERVE_NODE=$(hostname)"
export OLLAMA_HOST=0.0.0.0:{m.serve_port}
export OLLAMA_KEEP_ALIVE=24h
ollama serve &
SERVE_PID=$!
echo "waiting for ollama on :{m.serve_port}..."
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:{m.serve_port}/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
echo "pulling model: {m.name}"
ollama pull "{m.name}" || echo "WARN: ollama pull failed; model may already exist or tag is wrong"
echo "SERVE_READY engine=ollama model={m.name} port={m.serve_port}"
wait $SERVE_PID
"""


def _vllm_serve_body(cluster: ClusterConfig, m: ModelConfig) -> str:
    quant = m.quantization or catalog.choose_quant(
        catalog.get_model(m.name) or catalog.MODELS["qwen3-32b"],
        catalog.get_gpu(cluster.slurm.gpu_type),
    )
    quant_flag = f"--quantization {quant} " if quant in {"awq", "gptq", "fp8"} else ""
    return f"""set -uo pipefail
echo "SERVE_NODE=$(hostname)"
python -m vllm.entrypoints.openai.api_server \\
  --model "{m.name}" \\
  --port {m.serve_port} --host 0.0.0.0 \\
  --tensor-parallel-size {m.tensor_parallel_size} \\
  --pipeline-parallel-size {m.pipeline_parallel_size} \\
  --gpu-memory-utilization {m.gpu_memory_utilization} \\
  --max-model-len {m.max_model_len} {quant_flag}&
SERVE_PID=$!
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:{m.serve_port}/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
echo "SERVE_READY engine=vllm model={m.name} port={m.serve_port}"
wait $SERVE_PID
"""


def readiness_path(engine: str) -> str:
    return "/api/tags" if engine == "ollama" else "/v1/models"


def detect_scheduler(transport: Transport, cluster: ClusterConfig) -> str:
    """Return 'slurm' or 'ssh' for the target host ('auto' -> probe for sbatch)."""
    if cluster.scheduler in {"slurm", "ssh"}:
        return cluster.scheduler
    res = transport.exec("command -v sbatch >/dev/null 2>&1 && echo slurm || echo ssh", timeout=30)
    return "slurm" if "slurm" in res.out else "ssh"


def bring_up_ssh(transport: Transport, cluster: ClusterConfig,
                 model: ModelConfig | None = None) -> ServeHandle:
    """Plain GPU box (no SLURM): start the model server directly over ssh."""
    m = model or cluster.model
    p = m.serve_port
    if m.engine == "ollama":
        cmd = (
            f"export OLLAMA_HOST=0.0.0.0:{p}; mkdir -p ~/.slurm-agentic; "
            f"(curl -sf http://127.0.0.1:{p}/api/tags >/dev/null 2>&1 || "
            f"(nohup ollama serve >~/.slurm-agentic/serve.log 2>&1 &)); "
            f"for i in $(seq 1 60); do curl -sf http://127.0.0.1:{p}/api/tags >/dev/null 2>&1 && break; sleep 1; done; "
            f'ollama pull "{m.name}"'
        )
    else:
        setup = "; ".join(cluster.setup) + "; " if cluster.setup else ""
        try:
            quant = m.quantization or catalog.choose_quant(
                catalog.get_model(m.name) or catalog.MODELS["qwen3-32b"],
                catalog.get_gpu(cluster.slurm.gpu_type),
            )
        except Exception:  # noqa: BLE001 - bare ssh target may not have a discovered GPU type yet
            quant = m.quantization
        quant_flag = f"--quantization {quant} " if quant in {"awq", "gptq", "fp8"} else ""
        cmd = (
            f"mkdir -p ~/.slurm-agentic; {setup}"
            f"(curl -sf http://127.0.0.1:{p}/v1/models >/dev/null 2>&1 || "
            f"(nohup python -m vllm.entrypoints.openai.api_server "
            f'--model "{m.name}" --port {p} --host 0.0.0.0 '
            f"--tensor-parallel-size {m.tensor_parallel_size} "
            f"--pipeline-parallel-size {m.pipeline_parallel_size} "
            f"--gpu-memory-utilization {m.gpu_memory_utilization} "
            f"--max-model-len {m.max_model_len} {quant_flag}"
            f">~/.slurm-agentic/vllm-{p}.log 2>&1 &)); "
            f"for i in $(seq 1 180); do curl -sf http://127.0.0.1:{p}/v1/models >/dev/null 2>&1 && break; sleep 2; done"
        )
    transport.exec(cmd, timeout=1800).check("start model server on plain server")
    return ServeHandle(job_id="(ssh-direct)", node="127.0.0.1", port=p, engine=m.engine,
                       model=m.name, remote_url=f"http://127.0.0.1:{p}")


def bring_up(
    transport: Transport,
    cluster: ClusterConfig,
    remote_jobdir: str,
    model: ModelConfig | None = None,
    wait_timeout_s: float = 600.0,
) -> ServeHandle:
    """Bring up the model server and return a handle.

    SLURM: sbatch a serve job and wait for the node. Plain server: start ollama
    over ssh. Readiness (ollama up + pulled) is confirmed by the gateway's tunnel
    HTTP poll.
    """
    if detect_scheduler(transport, cluster) == "ssh":
        return bring_up_ssh(transport, cluster, model)
    m = model or cluster.model
    script = render_serve_sbatch(cluster, remote_jobdir, m)
    remote_script = f"{remote_jobdir}/serve.sbatch"
    transport.exec(f"mkdir -p {remote_jobdir}").check("mkdir remote jobdir")
    # Write the script on the remote side via a heredoc (avoids a temp-file round trip).
    transport.exec(_write_remote_file(remote_script, script)).check("write serve.sbatch")
    job_id = transport.sbatch(remote_script)
    node = transport.wait_for_node(job_id, timeout_s=wait_timeout_s)
    return ServeHandle(
        job_id=job_id,
        node=node,
        port=m.serve_port,
        engine=m.engine,
        model=m.name,
        remote_url=f"http://{node}:{m.serve_port}",
    )


def _write_remote_file(path: str, content: str) -> str:
    # Use a quoted heredoc so nothing in the script is expanded by the writing shell.
    marker = "SLURM_AGENTIC_EOF"
    return f"cat > {path} <<'{marker}'\n{content}\n{marker}"
