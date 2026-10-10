"""Side-by-side summary of experiments from their result JSONs.

    python -m bench.compare rtx4090/A2 rtx4090/B1D_g4 rtx4090/C0 --workload W1
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
COLS = [("output_throughput_tok_s", "tok/s", 1, "{:7.1f}"), ("goodput_req_s", "goodput", 1, "{:6.2f}"),
        ("ttft_p50", "TTFT50", 1, "{:6.2f}"), ("ttft_p99", "TTFT99", 1, "{:6.2f}"),
        ("tpot_p50", "TPOT50ms", 1e3, "{:7.1f}"), ("tpot_p99", "TPOT99ms", 1e3, "{:7.1f}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", help="paths under results/, e.g. rtx4090/A2")
    ap.add_argument("--workload", default=None)
    args = ap.parse_args()
    print(f"{'experiment':22s} {'run':16s} {'ok':>7s} " + " ".join(f"{c[1]:>8s}" for c in COLS))
    for exp in args.experiments:
        for p in sorted((ROOT / exp).rglob("*_rep*.json")):
            label = p.parent.name
            if args.workload and not label.startswith(args.workload):
                continue
            s = json.loads(p.read_text())["summary"]
            vals = " ".join(f"{c[3].format(s[c[0]] * c[2]):>8s}" for c in COLS)
            print(f"{exp:22s} {label + '/' + p.stem:16s} {s['completed']:>3d}/{s['num_requests']:<3d} {vals}")


if __name__ == "__main__":
    main()
