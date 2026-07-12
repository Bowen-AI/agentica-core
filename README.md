# agentica-core

Backend and runtime for the [Agentica desktop app](../Agentica): open-weight,
tool-using agents on local, SSH, and SLURM machines. The Python package is
`agentica_core`; the CLI is `agentica` (`slurm-agentic` remains as a compatibility
alias). It is built on [AgenticLocal](../AgenticLocal)'s model + state + tools +
policy loop.

## Install and run locally

Requirements: Python 3.11+, Ollama, and a sibling AgenticLocal checkout.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ../AgenticLocal
pip install -e ".[voice]"       # use `-e .` for typed-only operation

agentica serve-api --clusters-dir ~/.config/agentica/clusters
# HTTP/SSE API: http://127.0.0.1:8770
# local voice WS: ws://127.0.0.1:8771
```

The desktop API is agentic-only: every Chat and Voice request runs the tool loop.

## One request, two machines

Model inference and workspace tools are independent:

- `target` is the **model machine**. It loads the selected Ollama or vLLM model.
- `workspace_target` is the **workspace machine**. File and shell tools execute
  inside `workspace` on this machine and default to `local`.

This lets a large model run on an SSH GPU server while it edits a repository on
your laptop. Set both targets to the same SSH alias to work directly in a remote
checkout.

```json
POST /api/chat
{
  "message": "Inspect the failing API test, fix it, and run that test.",
  "target": "gpu-box",
  "model": "qwen3.5:9b",
  "workspace": "/Users/me/code/project",
  "workspace_target": "local"
}
```

Hosts come from `~/.ssh/config`; full cluster definitions come from
`~/.config/agentica/clusters/*.yaml` (or `--clusters-dir`). ProxyJump, ports,
identity files, accounts, partitions, setup commands, and GPU requests stay in
those existing configurations.

## Desktop API

```text
GET  /api/hosts
GET  /api/models?target=local
POST /api/chat                 agentic turn; no plain-chat mode
POST /api/chat/stream          SSE tool steps, artifacts, and final answer
GET  /api/history?session_id=…
GET  /api/sessions
POST /api/sessions/delete

POST /api/plan/draft           model/workspace targets are independent
POST /api/plan/refine
POST /api/job/submit           local thread, SSH process, or SLURM job
GET  /api/job/status
GET  /api/job/logs
POST /api/job/cancel
POST /api/job/fetch            sync a staged workspace back

GET  /api/voice/status
POST /api/voice/install
```

Set `AGENTICA_AUTH_TOKEN` to require the same local bearer token used by the
packaged Electron app.

## Local voice, optimized for response time

Voice never needs a browser speech API or a cloud provider:

```text
microphone → Silero VAD → Whisper → agent loop + tools → Kokoro → speakers
```

- Apple Silicon automatically prefers `mlx-whisper` `small.en` on Metal.
- Other platforms use `faster-whisper` `base.en` with CPU int8.
- Kokoro is the default open-weight TTS; Piper is a small fallback when present.
- The gateway warms cached models in a background thread so the first live turn
  does not also pay model construction time.
- Weights are downloaded on first use under
  `~/.local/share/agentica/voice`. Override that with `AGENTICA_VOICE_HOME`.

Useful controls:

```sh
agentica voice-status
agentica voice-selftest

AGENTICA_STT_ENGINE=mlx agentica serve-api       # Apple Silicon + Metal
AGENTICA_STT_ENGINE=faster agentica serve-api    # portable CPU path
AGENTICA_WHISPER_MODEL=tiny.en agentica serve-api
python scripts/bench_voice.py                    # cold/warm STT + TTS timings
```

The app stores Voice history as user/assistant text only. Raw microphone audio is
processed in memory and is not added to history.

## Parallel tracked jobs

Each UI submission is an independent run. Submit the same plan to multiple
workers and the app monitors them concurrently, records logs and output, supports
cancellation, and keeps polling when the user changes views.

A remote run has two workspace modes:

- `workspace_source: "local"` stages a snapshot in the job directory. The API
  returns `sync_to`, and `/api/job/fetch` copies completed output back.
- `workspace_source: "remote"` operates in an existing path on the worker. For
  SLURM, that path must be visible from the compute node.

The plan-file CLI exposes the same choice with a workspace prefix:

```yaml
# Snapshot a folder from the submission machine:
workspace: local:./examples/plans/ws_retry

# Or modify an existing checkout on the selected worker:
workspace: remote:~/work/my-project
```

```sh
agentica job submit gpu-box examples/plans/retry_backoff.yaml
agentica job status gpu-box --job <id> --jobdir <dir>
agentica job logs gpu-box --job <id> --jobdir <dir>
agentica job cancel gpu-box --job <id>
agentica job fetch gpu-box --jobdir <dir> --out job-artifacts
```

For a local worker, the job runs in a background thread. For a plain SSH worker,
it runs as a recorded remote process group. For a SLURM target, it submits with
`sbatch`; status and cancellation use the scheduler.

## Target any SSH host or SLURM cluster

```sh
agentica hosts
agentica discover discovery.usc.edu
agentica up examples/cluster.yaml --v1-mode agentic
agentica fit qwen3-235b-a22b a100-80 --count 2 --quant int4
agentica library --gpu l40s
```

`scheduler: auto` probes for `sbatch`, so a bare SSH GPU box and a cluster alias
use the same high-level workflow. `setup:` lines in cluster YAML run before model
startup; see [examples/discovery.yaml](examples/discovery.yaml).

The repository has also been exercised end to end on USC CARC Discovery with a
rootless Ollama install and deterministic test/artifact backstops. Cluster queues,
model availability, and hardware change, so treat the example configs as starting
points rather than current capacity claims.

## Model deployment / GPU sizing

The framework checks **model-vs-GPU fit** before serving and ships a validated
**library** of working `(model, gpu, quant)` configs (default leans to popular
models like `qwen3.6`):

```
slurm-agentic fit qwen3-235b-a22b a100-80 --count 2 --quant int4   # does it fit?
slurm-agentic library --gpu l40s                                   # working configs + a default
```

Key facts baked into `catalog.py`:

- **Size MoE by TOTAL params** (all experts stay resident).
- **Native FP8 is Ada (L40S) / Hopper only** — on A40/A100 use **BF16 or INT4 (AWQ/GPTQ)**.
- `preflight_fit()` computes `weights + KV + overhead` vs `gpus × VRAM × util` and
  refuses/advises (more GPUs, smaller quant, lower `max_model_len`, or smaller model).
- For very large models: **tensor/pipeline/expert-parallel** (one model sharded across
  GPUs/nodes) vs **replicas** (independent copies for throughput).

| Local default (1 GPU) | Big-but-feasible | Video + image (1×48GB) |
|---|---|---|
| Gemma 3 27B / Qwen3-32B INT4 | GLM-4.5-Air INT4 @1×A100-80; gpt-oss-120b @2×A100-80 | FLUX.1-schnell + Wan2.2-TI2V-5B (ComfyUI) |

## Layout

```
agentica_core/
  config.py        cluster.yaml / plan.yaml schemas
  catalog.py       GPU+model catalog, preflight_fit(), validated preset library
  transport.py     sync ssh/scp/rsync + sbatch/squeue/scancel + tunnel ctx mgr
  serving.py       bring up ollama/vLLM on a node, readiness, preflight
  apiserver.py     agentic desktop API, history, planning, jobs, model routing
  gateway.py       model/workspace split, tool registry, optional /v1 gateway
  voice_gateway.py local agentic voice WebSocket
  voice_provision.py  MLX/faster Whisper + Kokoro/Piper provisioning
  slurm_tools.py   run_shell/run_tests/check_artifact/submit_for_audit/generate_*
  workflows.py     Planner/Executor/Auditor workflows (seeded into the registry)
  on_node_runner.py  Mode 2 auditor outer loop (runs inside the SLURM job)
  job.py           local/SSH/SLURM submit, status, logs, cancel, fetch
  cli.py           `agentica` entrypoint
```

## Testing without a cluster

`tests/` uses local transports and fake SSH/SLURM shims for most coverage. Run:

```sh
python -m pytest tests -q
```

Tests that intentionally resolve public network names may be skipped in a
network-restricted environment.

<!-- BENCH:START -->
## Benchmark (online e2e)

Every **locally-runnable** model run through the **10 example agentic-job tasks** (`scripts/e2e.py` J1–J10), **5× each** on a single **NVIDIA TITAN Xp (12GB)**. Each task is graded by its deterministic backstop (test exit code + artifact), so PASS means the Planner→Executor→Auditor pipeline actually produced working output.

> The library's cluster presets (A100/L40S/H100 — `slurm-agentic library`) are validated by the `preflight_fit` GPU-vs-model check but require the cluster to execute; only the three models installed locally are benchmarked here.

### Ranking

| rank | model | pass rate | avg s/task | runs |
|---|---|---|---|---|
| 1 | `qwen3.5:9b` | **100.0%** | 42.9s | 50 |
| 2 | `gemma4:e2b` | **86.0%** | 10.5s | 50 |
| 3 | `gemma4:e4b` | **82.0%** | 14.0s | 50 |

![pass rate](docs/bench_passrate.png)
![avg latency](docs/bench_time.png)
![per-task heatmap](docs/bench_heatmap.png)

### Per-task pass rate (%)

| task | `qwen3.5:9b` | `gemma4:e2b` | `gemma4:e4b` |
|---|---|---|---|
| J1-hello | 100 | 100 | 100 |
| J2-add | 100 | 100 | 100 |
| J3-greet | 100 | 100 | 100 |
| J4-json | 100 | 100 | 100 |
| J5-reverse | 100 | 100 | 100 |
| J6-fixbug | 100 | 0 | 0 |
| J7-readme | 100 | 100 | 100 |
| J8-count | 100 | 60 | 20 |
| J9-fizzbuzz | 100 | 100 | 100 |
| J10-video | 100 | 100 | 100 |

_Regenerate: `python scripts/bench.py --repeats 5`._
<!-- BENCH:END -->
