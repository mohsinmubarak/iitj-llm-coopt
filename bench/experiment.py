"""Run one experiment config end to end: start the server, run every workload x rate x repeat,
stop the server. Results land in results/<profile>/<name>/<workload>/<rate>_rep<k>.json.
Runs whose JSON already exists are skipped, so an interrupted queue simply resumes.

    python -m bench.experiment bench/configs/A2.yaml --profile rtx4090
    python -m bench.experiment bench/configs/A2.yaml --profile rtx4090 --limit 20 --repeats 1 --tag smoke
"""
import argparse
import asyncio
import json
import math
import os
import re
import shlex
import socket
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

from bench import tracking
from bench.run import benchmark, load_workload

ROOT = Path(__file__).resolve().parent.parent
PORT = 8000
URL = f"http://localhost:{PORT}"


def wait_healthy(proc, log_path, timeout_s=1200):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup, see {log_path}:\n"
                               + Path(log_path).read_text(errors="ignore")[-3000:])
        try:
            if urllib.request.urlopen(f"{URL}/health", timeout=2).status == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server did not become healthy")


def start_server(cfg, extra_args, log_path):
    srv = cfg["server"]
    cmd = srv.get("cmd", ["vllm", "serve", srv["model"]])
    out_dir = str(Path(log_path).parent)
    args = [str(a).replace("{out_dir}", out_dir) for a in srv.get("args", [])]
    cmd = cmd + ["--port", str(PORT)] + args + shlex.split(extra_args)
    log = open(log_path, "w")
    log.write("CMD: " + " ".join(cmd) + "\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=os.environ | srv.get("env", {}))
    return proc, cmd


def parse_server_log(log_path):
    """KV cache capacity and model memory as reported by vLLM at startup (empty for other servers)."""
    info = {}
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        if m := re.search(r"GPU KV cache size: ([\d,]+) tokens", line):
            info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        elif m := re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", line):
            info["max_concurrency_at_max_len"] = float(m.group(2))
        elif m := re.search(r"Model loading took ([\d.]+) GiB", line):
            info["model_mem_gib"] = float(m.group(1))
        elif m := re.search(r"Available KV cache memory: ([\d.]+) GiB", line):
            info["kv_cache_mem_gib"] = float(m.group(1))
    return info


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, help="cap requests per workload (smoke tests)")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--only", action="append", help="run only these workloads")
    ap.add_argument("--extra-args", default="", help="appended to the server command line")
    ap.add_argument("--tag", default="", help="suffix for the results folder, e.g. 'smoke'")
    ap.add_argument("--profile", default="a100", help="hardware profile in bench/profiles/")
    ap.add_argument("--force", action="store_true", help="re-run results that already exist")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    prof = yaml.safe_load((ROOT / "bench" / "profiles" / f"{args.profile}.yaml").read_text())
    os.environ.setdefault("WANDB_PROJECT", prof["wandb_project"])
    max_len = prof["max_model_len"]
    name = cfg["name"] + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "results" / args.profile / name
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats = args.repeats or cfg.get("repeats", prof.get("repeats", 3))
    env_file = ROOT / "results" / f"env_{socket.gethostname()}.json"
    env = json.loads(env_file.read_text()) if env_file.exists() else {}

    # Expand runs and drop the ones already on disk, so a crashed queue resumes where it stopped.
    todo = []
    for run in cfg["runs"]:
        if args.only and run["workload"] not in args.only:
            continue
        for rate in run.get("request_rates", [run.get("request_rate", "inf")]):
            for k in range(repeats):
                tag = "file" if run.get("arrivals") == "file" else "inf" if math.isinf(float(rate)) else f"r{rate}"
                path = out_dir / run.get("label", run["workload"]) / f"{tag}_rep{k}.json"
                if args.force or not path.exists():
                    todo.append((run, rate, k, path))
    if not todo:
        print(f"[{name}] all results present, skipping")
        return

    extra = " ".join(map(str, prof.get("server_args", []))) + f" --max-model-len {max_len} " + args.extra_args
    log_path = out_dir / "server.log"
    proc, cmd = start_server(cfg, extra, log_path)
    exp_run = tracking.ExperimentRun(name, args.profile, cfg, cmd, out_dir, env)
    error = None
    try:
        startup_s = wait_healthy(proc, log_path)
        server_info = {"startup_s": round(startup_s, 1), **parse_server_log(log_path)}
        exp_run.server_ready(server_info)
        print(f"[{name}] server healthy after {startup_s:.0f}s", flush=True)
        for run, rate, k, path in todo:
            wl = run["workload"]
            if proc.poll() is not None:
                raise RuntimeError(f"server died (exit {proc.returncode}); see {log_path}")
            # Drop requests that cannot fit this GPU's context limit (e.g. 31K-token prompts on a 4090)
            all_reqs = load_workload(wl, args.limit)
            reqs = [r for r in all_reqs if r["prompt_len"] + r["max_tokens"] <= max_len]
            if not reqs:
                print(f"[{name}] {wl}: no request fits max_model_len={max_len}, skipping")
                continue
            res = asyncio.run(benchmark(
                URL, cfg["server"]["model"], reqs, request_rate=float(rate),
                arrivals=run.get("arrivals", "rate"),
                time_scale=run.get("time_scale", 1.0) * prof.get("arrival_time_scale", 1.0),
                max_concurrency=run.get("max_concurrency"), seed=k,
                temperature=run.get("temperature", 0.0)))
            res["meta"] = {"experiment": name, "profile": args.profile, "config": cfg, "workload": wl,
                           "label": run.get("label", wl), "rate": rate, "repeat": k, "server_cmd": cmd, "git": git_commit(), "env": env,
                           "limit": args.limit, "requests_dropped_by_max_len": len(all_reqs) - len(reqs),
                           "server_alive_after": proc.poll() is None, "server_info": server_info}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, default=str))
            tracking.safe_track(path, res)
            exp_run.bench_done(run.get("label", wl), rate, res["summary"], res)
            s = res["summary"]
            print(f"[{name}] {wl} rate={rate} rep={k}: {s['completed']}/{s['num_requests']} ok | "
                  f"out {s['output_throughput_tok_s']:.1f} tok/s | TTFT p99 {s['ttft_p99']:.2f}s | "
                  f"TPOT p99 {s['tpot_p99'] * 1e3:.1f}ms | goodput {s['goodput_req_s']:.2f} req/s",
                  flush=True)
    except BaseException as e:
        error = e
        raise
    finally:
        stop_server(proc)
        exp_run.finish(error)  # after stop: the server log and trace are complete
        print(f"[{name}] index rebuilt: {tracking.rebuild_index()} runs", flush=True)


if __name__ == "__main__":
    main()
