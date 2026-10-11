"""Regret of static configurations across regimes (thesis Claim 1).

For each regime (W1 at each load, W2, W3, each W4 phase) and metric, find the best configuration
and report every configuration's shortfall ("regret") relative to it. A configuration that is
best somewhere but far from best elsewhere is regime-specific; the claim is that every static
configuration has a large regret in at least one regime.

    python -m analysis.regret --profile rtx4090 --configs A2 A2_nopc B1 B1D_g1 B1D_g2 B1D_g4 B2
    python -m analysis.regret --profile rtx4090 --configs A2 B1D_g4 C0_nospec C0 C4_el --md out.md
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (regime label, result file under <config>/, summary key, higher_is_better, unit)
REGIMES = [
    ("W1 1 req/s · TPOT p50", "W1/r1_rep0.json", "tpot_p50", False, "ms"),
    ("W1 4 req/s · TPOT p50", "W1/r4_rep0.json", "tpot_p50", False, "ms"),
    ("W1 16 req/s · throughput", "W1/r16_rep0.json", "output_throughput_tok_s", True, "tok/s"),
    ("W1 16 req/s · goodput", "W1/r16_rep0.json", "goodput_req_s", True, "req/s"),
    ("W1 saturation · throughput", "W1/inf_rep0.json", "output_throughput_tok_s", True, "tok/s"),
    ("W1 saturation · goodput", "W1/inf_rep0.json", "goodput_req_s", True, "req/s"),
    ("W2 long · throughput", "W2/inf_rep0.json", "output_throughput_tok_s", True, "tok/s"),
    ("W3 bursty · goodput", "W3/file_rep0.json", "goodput_req_s", True, "req/s"),
    ("W3 bursty · TTFT p99", "W3/file_rep0.json", "ttft_p99", False, "s"),
    ("W4 P1 chat · TPOT p50", "W4/file_rep0.json", "P1_chat_tpot_p50", False, "ms"),
    ("W4 P2 long · SLO", "W4/file_rep0.json", "P2_long_slo_attainment", True, ""),
    ("W4 P3 burst · SLO", "W4/file_rep0.json", "P3_burst_slo_attainment", True, ""),
    ("W4 P4 recovery · TPOT p50", "W4/file_rep0.json", "P4_recovery_tpot_p50", False, "ms"),
]


def load(profile, config, rel):
    p = ROOT / "results" / profile / config / rel
    return json.loads(p.read_text())["summary"] if p.exists() else None


def regret(value, best, higher):
    """Shortfall vs the best, as a fraction: 0 = best; 0.5 = half as good (or 1.5x as slow)."""
    if value is None or best is None:
        return None
    if higher:
        return (best - value) / best if best else 0.0
    return (value - best) / best if best else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="rtx4090")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--md", help="also write a markdown table here")
    args = ap.parse_args()

    rows, worst = [], {c: (0.0, None) for c in args.configs}
    for label, rel, key, higher, unit in REGIMES:
        vals = {}
        for c in args.configs:
            s = load(args.profile, c, rel)
            if s is not None and key in s and s[key] == s[key]:  # skip NaN
                v = s[key] * (1e3 if unit == "ms" else 1.0)
                vals[c] = v
        if len(vals) < 2:
            continue
        best_c = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
        best = vals[best_c]
        cells = {}
        for c in args.configs:
            r = regret(vals.get(c), best, higher)
            cells[c] = r
            if r is not None and r > worst[c][0]:
                worst[c] = (r, label)
        rows.append((label, unit, best_c, best, vals, cells))

    w = max(len(c) for c in args.configs) + 2
    head = f"{'regime':30s} {'best':>12s} {'value':>9s} | " + " ".join(f"{c:>{w}s}" for c in args.configs)
    print(head)
    print("-" * len(head))
    for label, unit, best_c, best, vals, cells in rows:
        cs = " ".join(f"{'—':>{w}s}" if cells[c] is None else f"{'best' if cells[c] == 0 else f'+{100 * cells[c]:.0f}%':>{w}s}"
                      for c in args.configs)
        print(f"{label:30s} {best_c:>12s} {best:9.3g} | {cs}")
    print("\nworst regret per configuration (where it falls furthest from the best):")
    for c, (r, label) in worst.items():
        print(f"  {c:12s} {100 * r:5.0f}%  in {label}")

    if args.md:
        lines = ["| Regime | Best | " + " | ".join(args.configs) + " |",
                 "|---|---|" + "---|" * len(args.configs)]
        for label, unit, best_c, best, vals, cells in rows:
            cs = " | ".join("—" if cells[c] is None else ("**best**" if cells[c] == 0 else f"+{100 * cells[c]:.0f}%")
                            for c in args.configs)
            lines.append(f"| {label} | {best_c} ({best:.3g} {unit}) | {cs} |")
        lines.append("| **Worst regret** | | " + " | ".join(f"**{100 * worst[c][0]:.0f}%**" for c in args.configs) + " |")
        Path(args.md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
