"""Pod-level W&B run: live system metrics and log files for a whole experiment queue.

W&B's system monitor samples GPU (util, memory, power, temperature), CPU, RAM, disk and network
for as long as this run is open, so every server start-up, benchmark and idle gap of the queue is
on one timeline. Log files are uploaded live (re-synced whenever they change).

Started by bench/queue.sh; stops by itself when the queue log says QUEUE DONE.

    python -m bench.podlog --profile rtx4090 --until-done
"""
import argparse
import os
import socket
import subprocess
import time
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<{e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--until-done", action="store_true", help="exit when queue.log says QUEUE DONE")
    ap.add_argument("--project", default=None)
    args = ap.parse_args()

    import yaml
    prof = yaml.safe_load((ROOT / "bench" / "profiles" / f"{args.profile}.yaml").read_text())
    res = ROOT / "results" / args.profile
    res.mkdir(parents=True, exist_ok=True)
    queue_log = res / "queue.log"
    host = socket.gethostname()

    # pod facts snapshotted as files (shown under the run's Files tab)
    facts = res / f"pod_{host}"
    facts.mkdir(exist_ok=True)
    (facts / "nvidia-smi.txt").write_text(sh("nvidia-smi"))
    (facts / "pip-freeze.txt").write_text(sh("/root/venv/bin/pip freeze 2>/dev/null || pip freeze"))
    (facts / "system.txt").write_text("\n".join([sh("uname -a"), sh("nproc"), sh("free -g"), sh("df -h / /workspace")]))

    run = wandb.init(project=args.project or prof["wandb_project"], job_type="pod", group="pod",
                     name=f"pod|{host}|{time.strftime('%m%d-%H%M')}", tags=["pod", args.profile],
                     config={"profile": args.profile, "host": host, **prof},
                     dir=str(ROOT), settings=wandb.Settings(x_stats_sampling_interval=5.0))
    for pattern in (str(queue_log), str(facts / "*"), str(ROOT / "results" / "env_*.json"),
                    "/root/setup*.log", "/workspace/setup*.log"):
        try:
            run.save(pattern, base_path=str(Path(pattern).parent), policy="live")
        except Exception as e:
            print(f"[podlog] cannot watch {pattern}: {e}", flush=True)

    # Mirror queue events as a W&B table/log stream so progress is visible without opening files.
    seen = 0
    while True:
        text = queue_log.read_text() if queue_log.exists() else ""
        lines = text.splitlines()
        for line in lines[seen:]:
            print(line, flush=True)  # captured in the run's Logs tab
        seen = len(lines)
        run.log({"queue/lines": seen,
                 "queue/experiments_done": sum("END " in l for l in lines),
                 "queue/benchmarks_done": sum("rep=" in l for l in lines)})
        if args.until_done and "QUEUE DONE" in text:
            break
        time.sleep(15)
    run.finish()


if __name__ == "__main__":
    main()
