"""Experiment tracking.

Every result JSON (written by bench.experiment) is the source of truth. From it we derive:
  1. results/index.jsonl: one flat row per run (pandas: pd.read_json(path, lines=True))
  2. a Weights & Biases run: config, summary metrics, time-series plots, per-request table,
     and the raw JSON attached, so the dashboard can be shared with a link.

W&B is optional: with no WANDB_API_KEY it logs offline into ./wandb, and
`wandb sync wandb/offline-run-*` uploads later. A tracking failure never stops an experiment.

    python -m bench.tracking --index               # rebuild results/index.jsonl
    python -m bench.tracking --wandb results/A2    # (re)log existing results to W&B
"""
import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGGED = RESULTS / ".wandb_logged"


def result_files(base=RESULTS):
    for p in sorted(Path(base).resolve().rglob("*.json")):
        if p.name.startswith("env_"):
            continue
        yield p


def derived(res):
    """Metrics computed from server counters: spec-decoding acceptance, prefix-cache hit rate."""
    c = res.get("server_counters", {})
    out = {}
    drafted = c.get("vllm:spec_decode_num_draft_tokens_total", 0)
    if drafted:
        accepted = c.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        out["spec_acceptance_rate"] = accepted / drafted
        drafts = c.get("vllm:spec_decode_num_drafts_total", 0)
        if drafts:
            out["spec_mean_accepted_per_draft"] = accepted / drafts
    q = c.get("vllm:prefix_cache_queries_total", 0)
    if q:
        out["prefix_cache_hit_rate"] = c.get("vllm:prefix_cache_hits_total", 0) / q
    out["preemptions"] = c.get("vllm:num_preemptions_total", 0)
    ts = res.get("timeseries", [])
    if ts:
        mem = [r["gpu_mem_used_gb"] for r in ts if "gpu_mem_used_gb" in r]
        kv = [v for r in ts for k, v in r.items() if "kv_cache_usage" in k]
        if mem:
            out["gpu_mem_peak_gb"] = max(mem)
        if kv:
            out["kv_cache_usage_peak"] = max(kv)
    return out


def index_row(path, res):
    m = res.get("meta", {})
    row = {"file": str(Path(path).resolve().relative_to(ROOT)), "experiment": m.get("experiment"),
           "tier": (m.get("experiment") or "?")[0], "profile": m.get("profile"), "workload": m.get("workload"), "rate": str(m.get("rate")),
           "repeat": m.get("repeat"), "limit": m.get("limit"), "git": m.get("git"),
           "gpu": m.get("env", {}).get("gpu")}
    row.update({f"server_{k}": v for k, v in m.get("server_info", {}).items()})
    row.update({k: v for k, v in res["summary"].items() if not isinstance(v, (dict, list))})
    row.update(derived(res))
    return row


def rebuild_index():
    rows = []
    for p in result_files():
        try:
            res = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "summary" in res and "meta" in res:
            rows.append(index_row(p, res))
    with open(RESULTS / "index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return len(rows)


def _clean(v):
    return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v


def log_wandb(path, res):
    import wandb

    if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "offline"
    m = res["meta"]
    exp, wl, rate, rep = m["experiment"], m.get("label", m["workload"]), m["rate"], m["repeat"]
    srv = m.get("config", {}).get("server", {})
    config = {"experiment": exp, "tier": exp[0], "profile": m.get("profile"), "workload": wl, "rate": str(rate), "repeat": rep,
              "limit": m.get("limit"), "description": m.get("config", {}).get("description"),
              "model": srv.get("model"), "server_args": " ".join(map(str, m.get("server_cmd", [])[3:])),
              "git": m.get("git"), **{f"env/{k}": v for k, v in m.get("env", {}).items()},
              **{f"server/{k}": v for k, v in m.get("server_info", {}).items()}}

    with wandb.init(project=os.environ.get("WANDB_PROJECT", "iitj-llm-coopt"), name=f"{exp}|{wl}|{rate}|rep{rep}", group=exp, job_type=wl,
                    tags=[f"tier-{exp[0]}", exp, wl, m.get("profile", "na")], config=config, dir=str(ROOT),
                    reinit="create_new") as run:
        # time series against seconds-since-start
        run.define_metric("t")
        run.define_metric("ts/*", step_metric="t")
        prev = None
        for row in res.get("timeseries", []):
            out = {"t": row["t"]}
            for k, v in row.items():
                if k == "t":
                    continue
                key = k.replace("vllm:", "vllm_")
                if key.endswith("_total"):  # cumulative counter -> per-second rate
                    if prev and row["t"] > prev["t"] and k in prev:
                        out[f"ts/rate/{key[:-6]}_per_s"] = (v - prev[k]) / (row["t"] - prev["t"])
                else:
                    out[f"ts/{key}"] = v
            run.log(out)
            prev = row

        run.summary.update({k: _clean(v) for k, v in res["summary"].items() if not isinstance(v, dict)})
        run.summary.update(derived(res))
        run.summary.update({f"server/{k.replace('vllm:', '')}": v for k, v in res.get("server_counters", {}).items()})

        cols = ["id", "category", "prompt_len", "max_tokens", "output_tokens", "scheduled_s", "start_s",
                "ttft", "tpot", "e2e", "error"]
        data = []
        for r in res["requests"]:
            tpot = (r["e2e"] - r["ttft"]) / (r["output_tokens"] - 1) if r["output_tokens"] > 1 else None
            data.append([r["id"], r["category"], r["prompt_len"], r["max_tokens"], r["output_tokens"],
                         r["scheduled_s"], r["start_s"], _clean(r["ttft"]), _clean(tpot), _clean(r["e2e"]),
                         r["error"]])
        table = wandb.Table(columns=cols, data=data)
        run.log({"requests": table,
                 "ttft_hist": wandb.plot.histogram(table, "ttft", title="TTFT distribution (s)"),
                 "ttft_vs_prompt_len": wandb.plot.scatter(table, "prompt_len", "ttft",
                                                          title="TTFT vs prompt length")})
        run.save(str(path), base_path=str(ROOT), policy="now")
    with open(LOGGED, "a") as f:
        f.write(str(Path(path).resolve().relative_to(ROOT)) + "\n")


class ExperimentRun:
    """One W&B run per experiment (one server lifetime): server log and controller trace uploaded
    live, one row per benchmark, and the trace replayed as time-series charts at the end.
    Every method swallows errors: tracking must never stop an experiment."""

    def __init__(self, name, profile, cfg, cmd, out_dir, env):
        self.out_dir, self.rows, self.run = Path(out_dir), [], None
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "iitj-llm-coopt"), name=f"{name}|experiment",
                group=name, job_type="experiment", tags=[f"tier-{name[0]}", name, profile, "experiment"],
                config={"experiment": name, "tier": name[0], "profile": profile,
                        "description": cfg.get("description"), "server_cmd": " ".join(map(str, cmd)),
                        **{f"env/{k}": v for k, v in env.items()}},
                dir=str(ROOT), reinit="create_new")
            for f in ("server.log", "trace.jsonl"):
                self.run.save(str(self.out_dir / f), base_path=str(self.out_dir), policy="live")
        except Exception as e:
            print(f"[tracking] experiment run disabled: {e}", flush=True)

    def server_ready(self, info):
        if self.run:
            try:
                self.run.summary.update({f"server/{k}": v for k, v in info.items()})
            except Exception:
                pass

    def bench_done(self, label, rate, summary, res):
        if not self.run:
            return
        try:
            row = {"label": label, "rate": str(rate), **{k: _clean(v) for k, v in summary.items()
                                                       if isinstance(v, (int, float, bool))}, **derived(res)}
            self.rows.append(row)
            self.run.log({f"bench/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
        except Exception as e:
            print(f"[tracking] bench row failed: {e}", flush=True)

    def finish(self, error=None):
        if not self.run:
            return
        try:
            wandb = self.wandb
            if self.rows:
                cols = sorted({k for r in self.rows for k in r}, key=lambda k: (k not in ("label", "rate"), k))
                self.run.log({"benchmarks": wandb.Table(columns=cols, data=[[r.get(c) for c in cols]
                                                                           for r in self.rows])})
            trace = self.out_dir / "trace.jsonl"
            if trace.exists():  # controller decisions over time (experiment M2)
                self.run.define_metric("trace/t_s")
                self.run.define_metric("trace/*", step_metric="trace/t_s")
                rows = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
                t0 = rows[0]["t"] if rows else 0
                for r in rows:
                    self.run.log({"trace/t_s": r["t"] - t0, **{f"trace/{k}": v for k, v in r.items()
                                                               if isinstance(v, (int, float)) and k != "t"}})
            if error:
                self.run.summary["error"] = str(error)[:2000]
            self.run.finish(exit_code=1 if error else 0)
        except Exception as e:
            print(f"[tracking] finishing experiment run failed: {e}", flush=True)


def safe_track(path, res):
    """Called by bench.experiment after every run; never raises."""
    try:
        log_wandb(path, res)
    except Exception as e:
        print(f"[tracking] W&B logging failed for {path}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true", help="rebuild results/index.jsonl")
    ap.add_argument("--wandb", metavar="DIR", help="log every result JSON under DIR to W&B")
    ap.add_argument("--force", action="store_true", help="re-log runs already sent to W&B")
    args = ap.parse_args()
    if args.wandb:
        done = set(LOGGED.read_text().split()) if LOGGED.exists() and not args.force else set()
        for p in result_files(args.wandb):
            rel = str(p.relative_to(ROOT))
            res = json.loads(p.read_text())
            if rel in done or "meta" not in res:
                continue
            print("logging", rel)
            log_wandb(p, res)
    n = rebuild_index()
    print(f"index: {n} runs -> {RESULTS / 'index.jsonl'}")


if __name__ == "__main__":
    main()
