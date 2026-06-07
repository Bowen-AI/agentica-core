"""`agentica` command-line entrypoint (agentica-core backend).

Subcommands:
  serve-api Start the JSON API the Agentica UI talks to (chat, plan, jobs, hosts)
  up        Bring up the interactive gateway (web chat + OpenAI /v1) for a cluster.yaml
  down      Cancel a running serve job
  job       submit | status | logs | cancel | fetch  -- agentic batch jobs
  fit       Check whether a model fits given GPUs (the GPU-vs-model gate)
  library   Print the validated library of working (model, gpu) configs
  hosts     List servers from your ~/.ssh/config
  discover  probe a host's scheduler/GPUs into a draft cluster.yaml
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="agentica")
    sub = parser.add_subparsers(dest="command")

    p_api = sub.add_parser("serve-api", help="Start the JSON API for the Agentica UI.")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8770)
    # default None -> serve() resolves to a writable dir ($AGENTICA_DATA_DIR or ~/.local/...)
    p_api.add_argument("--workspace", default=None)
    p_api.add_argument("--db", default=None)
    p_api.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p_api.add_argument("--model", default="qwen3.5:4b-mlx")
    p_api.add_argument("--clusters-dir", default=None,
                       help="Folder of cluster.yaml files exposed as job targets (SLURM "
                            "account/partition/setup). Default: $AGENTICA_CLUSTERS_DIR or "
                            "~/.config/agentica/clusters.")

    p_up = sub.add_parser("up", help="Bring up the interactive gateway.")
    p_up.add_argument("cluster", help="Path to cluster.yaml")
    p_up.add_argument("--ollama-host", default=None,
                      help="Skip SLURM; point the gateway at this ollama host (local or manual tunnel).")
    p_up.add_argument("--model", default=None, help="Override the served model tag (else cluster.yaml model.name).")
    p_up.add_argument("--workspace", default="sample_workspace")
    p_up.add_argument("--db", default=".agentic/agentic.db")
    p_up.add_argument("--v1-mode", choices=["passthrough", "agentic"], default="passthrough")
    p_up.add_argument("--skip-preflight", action="store_true")

    p_down = sub.add_parser("down", help="Cancel a running serve job.")
    p_down.add_argument("cluster")
    p_down.add_argument("--job", default=None)

    p_job = sub.add_parser("job", help="Agentic batch jobs.")
    job_sub = p_job.add_subparsers(dest="job_command")
    j_submit = job_sub.add_parser("submit")
    j_submit.add_argument("cluster")
    j_submit.add_argument("plan")
    j_submit.add_argument("--no-sync-code", action="store_true",
                          help="Do not rsync the package source (assume installed on the cluster).")
    for name in ("status", "logs", "cancel"):
        jp = job_sub.add_parser(name)
        jp.add_argument("cluster")
        jp.add_argument("--job", required=True)
        jp.add_argument("--jobdir", default=None)
    j_fetch = job_sub.add_parser("fetch")
    j_fetch.add_argument("cluster")
    j_fetch.add_argument("--jobdir", required=True)
    j_fetch.add_argument("--out", default="job-artifacts")

    p_fit = sub.add_parser("fit", help="Check model-vs-GPU fit.")
    p_fit.add_argument("model")
    p_fit.add_argument("gpu")
    p_fit.add_argument("--count", type=int, default=1)
    p_fit.add_argument("--quant", default=None)
    p_fit.add_argument("--max-model-len", type=int, default=8192)
    p_fit.add_argument("--tp", type=int, default=None)

    p_lib = sub.add_parser("library", help="List validated working (model, gpu) configs.")
    p_lib.add_argument("--gpu", default=None, help="Filter to a GPU and recommend a default.")
    p_lib.add_argument("--count", type=int, default=1)

    p_disc = sub.add_parser("discover", help="probe a host's scheduler/GPUs into a draft cluster.yaml.")
    p_disc.add_argument("cluster", help="cluster.yaml path OR an ~/.ssh/config host alias")

    sub.add_parser("hosts", help="List servers from your ~/.ssh/config (pick any as a target).")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "serve-api":
        from . import apiserver
        return apiserver.serve(host=args.host, port=args.port, workspace=args.workspace,
                               db_path=args.db, ollama_host=args.ollama_host, model=args.model,
                               clusters_dir=args.clusters_dir)
    if args.command == "up":
        from . import gateway
        return gateway.up(args.cluster, ollama_host=args.ollama_host, model_override=args.model,
                          workspace=args.workspace, db_path=args.db, v1_mode=args.v1_mode,
                          skip_preflight=args.skip_preflight)
    if args.command == "down":
        from . import gateway
        return gateway.down(args.cluster, args.job)
    if args.command == "job":
        return _job(args)
    if args.command == "fit":
        return _fit(args)
    if args.command == "library":
        return _library(args)
    if args.command == "discover":
        from . import discover
        return discover.discover(args.cluster)
    if args.command == "hosts":
        return _hosts()
    parser.print_help()
    return 1


def _hosts() -> int:
    from . import sshconfig
    hosts = sshconfig.list_hosts()
    if not hosts:
        print("No ~/.ssh/config hosts found.")
        return 0
    print(f"{'alias':28s} {'hostname':28s} {'user':12s} proxy_jump")
    for h in hosts:
        print(f"{h['alias']:28s} {h['hostname']:28s} {h['user']:12s} {h['proxy_jump']}")
    print("\nUse any alias directly, e.g.:  slurm-agentic discover <alias>   |   slurm-agentic up <alias>")
    return 0


def _job(args) -> int:
    from . import job
    if args.job_command == "submit":
        return job.submit(args.cluster, args.plan, sync_code=not args.no_sync_code)
    if args.job_command == "status":
        return job.status(args.cluster, args.job, jobdir=args.jobdir)
    if args.job_command == "logs":
        return job.logs(args.cluster, args.job, jobdir=args.jobdir)
    if args.job_command == "cancel":
        return job.cancel(args.cluster, args.job)
    if args.job_command == "fetch":
        return job.fetch_artifacts(args.cluster, args.jobdir, args.out)
    print("usage: slurm-agentic job {submit|status|logs|cancel|fetch} ...")
    return 1


def _fit(args) -> int:
    from . import catalog
    fit = catalog.preflight_fit(args.model, args.gpu, args.count, quant=args.quant,
                                max_model_len=args.max_model_len, tensor_parallel_size=args.tp)
    print(f"verdict: {fit.verdict}  (fits={fit.ok})")
    print(f"  {fit.message}")
    print(f"  weights={fit.weights_gb}GB kv={fit.kv_gb}GB overhead={fit.overhead_gb}GB "
          f"total={fit.total_gb}GB available={fit.available_gb}GB gpus_needed={fit.gpus_needed}")
    if fit.estimate:
        print("  (estimate -- measure on the real node)")
    for w in fit.warnings:
        print(f"  ! {w}")
    return 0 if fit.ok else 2


def _library(args) -> int:
    from . import catalog
    rows = catalog.library()
    if args.gpu:
        gpu_key = catalog.get_gpu(args.gpu).key
        rows = [r for r in rows if catalog.get_gpu(r["gpu"]).key == gpu_key]
    print(f"{'preset':28s} {'model':18s} {'gpus':6s} {'quant':6s} {'verdict':9s} {'vram':>7s}  note")
    for r in rows:
        print(f"{r['name']:28s} {r['model']:18s} {str(r['gpus'])+'x'+r['gpu']:6s} "
              f"{r['quant']:6s} {r['verdict']:9s} {r['vram_gb']:>6}G  {r['note']}")
    if args.gpu:
        rec = catalog.recommend(args.gpu, args.count)
        if rec:
            print(f"\nrecommended default for {args.count}x {args.gpu}: {rec.name} "
                  f"({rec.model} @ {rec.quant}, engine={rec.engine})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
