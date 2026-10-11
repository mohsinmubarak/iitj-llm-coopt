"""Rename result folders/files, workload data files and experiment configs from codes to the
descriptive names in bench/names.py. Idempotent; run on the pod when no experiment is running.

    results/<profile>/C4_el/W4/file_rep0.json -> results/<profile>/ours-cooptimizer/phase-shift/scheduled_rep0.json
    data/W4.jsonl                              -> data/phase-shift.jsonl
    bench/configs/C4_el.yaml                   -> bench/configs/ours-cooptimizer.yaml (name + workloads)

Result JSONs keep their content; `meta` gains experiment_code / workload_code and its experiment,
workload and label fields switch to the new names.

    python -m bench.rename [--dry-run]
"""
import argparse
import json
import re
from pathlib import Path

from bench import names

ROOT = Path(__file__).resolve().parent.parent
FILE_RATE = re.compile(r"^(r(\d+(?:\.\d+)?)|inf|file)_rep(\d+)\.json$")


def plan_moves():
    moves = []
    # data files
    for code, new in names.WORKLOADS.items():
        src = ROOT / "data" / f"{code}.jsonl"
        if src.exists():
            moves.append((src, ROOT / "data" / f"{new}.jsonl"))
    # configs
    for code, new in names.EXPERIMENTS.items():
        src = ROOT / "bench" / "configs" / f"{code}.yaml"
        if src.exists():
            moves.append((src, ROOT / "bench" / "configs" / f"{new}.yaml"))
    return moves


def rename_results(dry):
    n = 0
    for prof in [ROOT / "results" / p for p in ("rtx4090", "a100") if (ROOT / "results" / p).is_dir()]:
        for parent in [prof, prof / "_superseded", prof / "_failed"] + \
                [p for p in (prof / "_superseded").glob("engine_v*") if p.is_dir()]:
            if not parent.exists():
                continue
            for exp_dir in [d for d in parent.iterdir() if d.is_dir() and not d.name.startswith("_")]:
                if exp_dir.name.startswith(("engine_v", "pod_")):
                    continue
                code = exp_dir.name
                # superseded folders carry a reason suffix: C1_v1_costmodel -> ours-...-only_v1_costmodel
                base = next((c for c in sorted(names.EXPERIMENTS, key=len, reverse=True)
                             if code == c or code.startswith(c + "_")), None)
                new_exp = names.experiment(code) if base is None or code in names.EXPERIMENTS or \
                    code[len(base) + 1:] in names.TAGS else f"{names.EXPERIMENTS[base]}{code[len(base):]}"
                for wl_dir in [d for d in exp_dir.iterdir() if d.is_dir()]:
                    wl_code = wl_dir.name
                    for f in wl_dir.glob("*.json"):
                        m = FILE_RATE.match(f.name)
                        if not m:
                            continue
                        tag, num, rep = m.group(1), m.group(2), m.group(3)
                        rate_name = names.rate("file", scheduled=True) if tag == "file" else \
                            names.rate("inf") if tag == "inf" else names.rate(num)
                        new_name = f"{rate_name}_rep{rep}.json"
                        if not dry:
                            res = json.loads(f.read_text())
                            meta = res.get("meta", {})
                            if meta and "experiment_code" not in meta:
                                meta["experiment_code"] = meta.get("experiment")
                                meta["workload_code"] = meta.get("workload")
                                meta["label_code"] = meta.get("label")
                                meta["experiment"] = new_exp
                                meta["workload"] = names.workload(meta.get("workload") or wl_code)
                                meta["label"] = names.workload(meta.get("label") or wl_code)
                                meta["rate_name"] = rate_name
                                f.write_text(json.dumps(res, default=str))
                            if new_name != f.name:
                                f.rename(f.with_name(new_name))
                        n += 1
                    new_wl = names.workload(wl_code)
                    if new_wl != wl_code and not dry:
                        wl_dir.rename(wl_dir.with_name(new_wl))
                if new_exp != code:
                    print(f"{exp_dir.relative_to(ROOT)} -> {new_exp}")
                    if not dry:
                        exp_dir.rename(exp_dir.with_name(new_exp))
    return n


def rename_configs_content(dry):
    for p in (ROOT / "bench" / "configs").glob("*.yaml"):
        text = p.read_text()
        new = re.sub(r"^name: (\S+)$", lambda m: f"name: {names.experiment(m.group(1))}", text, flags=re.M)
        new = re.sub(r"(workload: )(\S+)", lambda m: m.group(1) + names.workload(m.group(2)), new)
        new = re.sub(r"(label: )(\S+)", lambda m: m.group(1) + names.workload(m.group(2)), new)
        if new != text and not dry:
            p.write_text(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    n = rename_results(args.dry_run)
    print(f"{n} result files {'would be ' if args.dry_run else ''}updated")
    rename_configs_content(args.dry_run)
    for src, dst in plan_moves():
        print(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        if not args.dry_run:
            src.rename(dst)


if __name__ == "__main__":
    main()
