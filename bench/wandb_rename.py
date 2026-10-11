"""Rename W&B runs from experiment codes to descriptive names (bench/names.py). Idempotent.

Run name   ours-cooptimizer | phase-shift | scheduled | rep0   (benchmark runs)
           ours-cooptimizer | experiment                         (experiment runs)
Group      ours-cooptimizer
Config     experiment_code (old code), experiment_name, workload_name, rate_name added
Tags       code tags replaced by the new names; tier-*, superseded, engine_v* and profile kept

    python -m bench.wandb_rename --project iitj-coopt-rtx4090 [--dry-run]
"""
import argparse

import wandb

from bench import names

KEEP_TAG_PREFIXES = ("tier-", "superseded", "engine_v", "pod", "experiment", "validation", "quality",
                     "rtx4090", "a100", "M4", "q1", "q2")


def new_identity(run):
    cfg, old_name = run.config, run.name or ""
    jt = run.job_type or ""
    if jt == "pod":
        return None
    if jt == "validation":
        if old_name.startswith("M4|"):
            return f"engine-correctness | {old_name.split('|', 1)[1]}", "engine-correctness", {}
        if old_name.startswith("Q|"):
            what = {"q1": "kv-quantization", "q2": "spec-vs-no-spec"}.get(old_name.split("|")[1], old_name)
            return f"quality | {what}", "quality", {}
        return None
    code = cfg.get("experiment_code") or cfg.get("experiment") or run.group
    if not code:
        return None
    exp = names.experiment(code)
    extra = {"experiment_code": code, "experiment_name": exp}
    if jt == "experiment" or old_name.endswith("|experiment") or old_name.endswith("| experiment"):
        return f"{exp} | experiment", exp, extra
    parts = [p.strip() for p in old_name.split("|")]
    if len(parts) == 4 and parts[0] in (code, exp):
        wl_code, r, rep = parts[1], parts[2], parts[3]
        wl_code = cfg.get("workload_code", wl_code)
        base = wl_code.split("_c")[0]
        scheduled = base in ("W3", "W4") and "_c" not in wl_code and r in ("inf", "scheduled")
        wl = names.workload(wl_code)
        rt = names.rate(r, scheduled) if r not in ("1rps", "4rps", "16rps", "saturation", "scheduled") else r
        extra.update(workload_code=wl_code, workload_name=wl, rate_name=rt)
        return f"{exp} | {wl} | {rt} | {rep}", exp, extra
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="iitj-coopt-rtx4090")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = wandb.Api()
    runs = api.runs(f"{api.default_entity}/{args.project}")
    done = skipped = 0
    for run in runs:
        if run.state == "running":
            skipped += 1
            continue
        ident = new_identity(run)
        if ident is None:
            continue
        name, group, extra = ident
        code_tags = set(names.EXPERIMENTS) | set(names.WORKLOADS) | {extra.get("experiment_code"), extra.get("workload_code")}
        tags = [t for t in run.tags if t not in code_tags or t.startswith(KEEP_TAG_PREFIXES)]
        for t in (group, extra.get("workload_name")):
            if t and t not in tags:
                tags.append(t)
        if name == run.name and group == run.group and all(run.config.get(k) == v for k, v in extra.items()):
            continue
        if args.dry_run:
            print(f"{run.name!r:60s} -> {name!r}  [group {group}]")
            done += 1
            continue
        run.name, run.group, run.tags = name, group, tags
        for k, v in extra.items():
            run.config[k] = v
        run.update()
        done += 1
    print(f"{'would rename' if args.dry_run else 'renamed'} {done} runs; skipped {skipped} still running")


if __name__ == "__main__":
    main()
