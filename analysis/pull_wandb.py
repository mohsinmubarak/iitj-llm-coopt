"""Pull result JSONs from W&B that are missing locally (when the pod is gone and the sync did not run).

Benchmark runs save their raw JSON under its repo path (results/<profile>/<experiment>/...), so
missing files land exactly where bench/ and analysis/ expect them. Existing files are never overwritten.

    wandb login                                   # once
    python -m analysis.pull_wandb                 # list what W&B has that we don't
    python -m analysis.pull_wandb --download      # fetch it
"""
import argparse
import collections
import os

import wandb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", action="append", help="default: iitj-coopt-a100")
    ap.add_argument("--entity", default=None, help="default: your W&B default entity")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    api = wandb.Api(timeout=60)
    entity = args.entity or api.default_entity
    missing = collections.defaultdict(list)
    for project in args.project or ["iitj-coopt-a100"]:
        runs = api.runs(f"{entity}/{project}")
        print(f"{entity}/{project}: {len(runs)} runs")
        for run in runs:
            for f in run.files():
                if f.name.startswith("results/") and not os.path.exists(os.path.join(ROOT, f.name)):
                    exp = "/".join(f.name.split("/")[1:3])
                    missing[exp].append((run, f))
    if not missing:
        print("nothing missing: local results/ already has every result file on W&B")
        return
    print("\non W&B, missing locally:")
    for exp, items in sorted(missing.items()):
        print(f"  {exp:55s} {len(items)} files  (runs: {', '.join(sorted({r.state for r, _ in items}))})")
    if args.download:
        for exp, items in sorted(missing.items()):
            for _, f in items:
                f.download(root=ROOT, replace=False)
            print(f"downloaded {exp}")
    else:
        print("\nre-run with --download to fetch them")


if __name__ == "__main__":
    main()
