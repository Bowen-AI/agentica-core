"""Benchmark every locally-runnable model over the 10 example tasks, N times each.

Produces: bench_results.json (raw + aggregate), docs/bench_passrate.png,
docs/bench_time.png, docs/bench_heatmap.png, and a "## Benchmark" section injected
into README.md between BENCH markers.

    python scripts/bench.py --repeats 5
    python scripts/bench.py --render-only        # rebuild charts/README from JSON

Only models actually installed in Ollama are benchmarked. The library's cluster
presets (A100/L40S/...) are validated by the fit check but not executed here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import agentica_core  # noqa: F401
from e2e import JOBS, run_single_job  # noqa: E402

RESULTS = ROOT / "bench_results.json"
DOCS = ROOT / "docs"
DEFAULT_MODELS = ["qwen3.5:9b", "gemma4:e4b", "gemma4:e2b"]


def installed_models() -> set[str]:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return set()
    names = set()
    for line in out.splitlines()[1:]:
        if line.strip():
            names.add(line.split()[0])
    return names


def run(models: list[str], host: str, repeats: int, timeout: float) -> dict:
    raw: list[dict] = []
    state = {"models": models, "repeats": repeats, "tasks": [j["id"] for j in JOBS],
             "host": host, "raw": raw}
    for model in models:
        for rep in range(1, repeats + 1):
            for spec in JOBS:
                r = run_single_job(spec, model, host, timeout=timeout)
                rec = {"model": model, "rep": rep, "task": spec["id"],
                       "passed": r["passed"], "secs": round(r["secs"], 2), "detail": r["detail"]}
                raw.append(rec)
                print(f"  {model:12s} rep{rep} {spec['id']:12s} "
                      f"{'PASS' if r['passed'] else 'FAIL'} {r['secs']:5.1f}s")
                RESULTS.write_text(json.dumps(state, indent=2))  # incremental save
    return state


def aggregate(state: dict) -> dict:
    raw = state["raw"]
    models = state["models"]
    tasks = state["tasks"]
    per_model = {}
    for m in models:
        runs = [r for r in raw if r["model"] == m]
        if not runs:
            continue
        npass = sum(1 for r in runs if r["passed"])
        per_model[m] = {
            "runs": len(runs),
            "pass": npass,
            "pass_rate": round(100 * npass / len(runs), 1),
            "avg_secs": round(sum(r["secs"] for r in runs) / len(runs), 1),
            "per_task": {
                t: round(100 * sum(1 for r in runs if r["task"] == t and r["passed"])
                         / max(1, sum(1 for r in runs if r["task"] == t)), 0)
                for t in tasks
            },
        }
    ranking = sorted(per_model.items(), key=lambda kv: (-kv[1]["pass_rate"], kv[1]["avg_secs"]))
    return {"per_model": per_model, "ranking": [m for m, _ in ranking], "tasks": tasks}


def charts(agg: dict, repeats: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    DOCS.mkdir(exist_ok=True)
    models = agg["ranking"]
    pm = agg["per_model"]
    tasks = agg["tasks"]
    colors = ["#2da44e", "#0969da", "#bf8700", "#cf222e", "#8250df"]

    # 1) pass-rate bar
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rates = [pm[m]["pass_rate"] for m in models]
    bars = ax.barh(models[::-1], rates[::-1], color=[colors[i % len(colors)] for i in range(len(models))][::-1])
    ax.set_xlabel("pass rate (%)")
    ax.set_xlim(0, 100)
    ax.set_title(f"Mode-2 agentic-job pass rate  (10 tasks x {repeats} repeats)")
    for b, r in zip(bars, rates[::-1]):
        ax.text(min(r + 1, 95), b.get_y() + b.get_height() / 2, f"{r}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "bench_passrate.png", dpi=120)
    plt.close(fig)

    # 2) avg-time bar
    fig, ax = plt.subplots(figsize=(7, 3.2))
    times = [pm[m]["avg_secs"] for m in models]
    ax.barh(models[::-1], times[::-1], color="#57606a")
    ax.set_xlabel("avg seconds / task (lower = faster)")
    ax.set_title("Mode-2 agentic-job avg latency per task")
    for i, t in enumerate(times[::-1]):
        ax.text(t, i, f" {t}s", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "bench_time.png", dpi=120)
    plt.close(fig)

    # 3) per-task heatmap
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(models) + 1.5))
    mat = np.array([[pm[m]["per_task"][t] for t in tasks] for m in models])
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_title("Per-task pass rate (%)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(DOCS / "bench_heatmap.png", dpi=120)
    plt.close(fig)


def readme_section(agg: dict, repeats: int, host_gpu: str) -> str:
    pm = agg["per_model"]
    models = agg["ranking"]
    lines = [
        "<!-- BENCH:START -->",
        "## Benchmark (online e2e)",
        "",
        f"Every **locally-runnable** model run through the **10 example agentic-job tasks** "
        f"(`scripts/e2e.py` J1–J10), **{repeats}× each** on a single **{host_gpu}**. Each task is "
        "graded by its deterministic backstop (test exit code + artifact), so PASS means the "
        "Planner→Executor→Auditor pipeline actually produced working output.",
        "",
        "> The library's cluster presets (A100/L40S/H100 — `slurm-agentic library`) are validated "
        "by the `preflight_fit` GPU-vs-model check but require the cluster to execute; only the "
        "three models installed locally are benchmarked here.",
        "",
        "### Ranking",
        "",
        "| rank | model | pass rate | avg s/task | runs |",
        "|---|---|---|---|---|",
    ]
    for i, m in enumerate(models, 1):
        d = pm[m]
        lines.append(f"| {i} | `{m}` | **{d['pass_rate']}%** | {d['avg_secs']}s | {d['runs']} |")
    lines += [
        "",
        "![pass rate](docs/bench_passrate.png)",
        "![avg latency](docs/bench_time.png)",
        "![per-task heatmap](docs/bench_heatmap.png)",
        "",
        "### Per-task pass rate (%)",
        "",
        "| task | " + " | ".join(f"`{m}`" for m in models) + " |",
        "|---|" + "|".join(["---"] * len(models)) + "|",
    ]
    for t in agg["tasks"]:
        row = " | ".join(f"{int(pm[m]['per_task'][t])}" for m in models)
        lines.append(f"| {t} | {row} |")
    lines += ["", "_Regenerate: `python scripts/bench.py --repeats 5`._", "<!-- BENCH:END -->"]
    return "\n".join(lines)


def update_readme(section: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
    if start in text and end in text:
        pre = text[: text.index(start)]
        post = text[text.index(end) + len(end):]
        text = pre + section + post
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    readme.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated; default = installed subset of DEFAULT_MODELS")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--gpu", default="NVIDIA TITAN Xp (12GB)")
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    if args.render_only:
        state = json.loads(RESULTS.read_text())
    else:
        want = args.models.split(",") if args.models else DEFAULT_MODELS
        have = installed_models()
        models = [m for m in want if m in have] or want
        print(f"benchmarking {models} x{args.repeats} on {args.gpu}\n")
        state = run(models, args.ollama_host, args.repeats, args.timeout)

    agg = aggregate(state)
    state["aggregate"] = agg
    RESULTS.write_text(json.dumps(state, indent=2))
    charts(agg, state["repeats"])
    update_readme(readme_section(agg, state["repeats"], args.gpu))

    print("\n=== RANKING ===")
    for i, m in enumerate(agg["ranking"], 1):
        d = agg["per_model"][m]
        print(f"  {i}. {m:12s} pass={d['pass_rate']}%  avg={d['avg_secs']}s  ({d['pass']}/{d['runs']})")
    print(f"\nwrote {RESULTS.name}, docs/bench_*.png, README Benchmark section")
    return 0


if __name__ == "__main__":
    sys.exit(main())
