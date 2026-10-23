"""Figures for the supervisor meeting, read straight from results/a100-22gb.

    python -m analysis.slide_figures            # writes figures/*.png and prints the tables
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "a100-22gb")
OUT = os.path.join(ROOT, "figures")
SLO = {"short": 1.0, "long": 10.0, "tpot": 0.1}


def runs(exp, workload, load):
    files = sorted(glob.glob(os.path.join(RES, exp, workload, f"{load}_rep*.json")))
    return [json.load(open(f)) for f in files]


def metric(exp, workload, load, key):
    vals = [r["summary"][key] for r in runs(exp, workload, load)]
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote figures/" + name)


def bars(ax, labels, vals, colors, fmt="{:.2f}", horizontal=False):
    """Bars with value labels. Error bars only where there are repeats; single runs are hatched."""
    means = [m for m, _, _ in vals]
    for i, ((m, s, n), c) in enumerate(zip(vals, colors)):
        kw = dict(color=c, edgecolor="0.2", linewidth=0.6, hatch="//" if n == 1 else None)
        err = s if n > 1 else None
        if horizontal:
            ax.barh(i, m, xerr=err, capsize=4, **kw)
        else:
            ax.bar(i, m, yerr=err, capsize=4, **kw)
    top = max(m + (s if n > 1 else 0) for m, s, n in vals)
    for i, (m, s, n) in enumerate(vals):
        pad = (s if n > 1 else 0) + top * 0.02
        if horizontal:
            ax.text(m + pad, i, fmt.format(m), va="center", fontsize=10)
        else:
            ax.text(i, m + pad, fmt.format(m), ha="center", va="bottom", fontsize=10)
    if horizontal:
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlim(0, top * 1.15)
    else:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, top * 1.18)
    ax.spines[["top", "right"]].set_visible(False)


def fig_static_spec():
    exps = ["vllm-default", "vllm-draft-spec-gamma1", "vllm-draft-spec-gamma2", "vllm-draft-spec-gamma4",
            "vllm-draft-spec-gamma6", "vllm-draft-spec-gamma8"]
    labels = ["no spec", "γ=1", "γ=2", "γ=4", "γ=6", "γ=8"]
    sat = [metric(e, "chat", "saturation", "goodput_req_s") for e in exps]
    light = [metric(e, "chat", "1rps", "tpot_p50") for e in exps]
    light = [(m * 1000, s * 1000, n) for m, s, n in light]
    colors = ["0.35"] + ["0.7"] * 5
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(a, labels, sat, colors)
    a.set_title("Chat, all 200 requests at once")
    a.set_ylabel("goodput (req/s within SLO)")
    bars(b, labels, light, colors, fmt="{:.1f}")
    b.set_title("Chat, 1 req/s")
    b.set_ylabel("time per output token, p50 (ms)")
    fig.suptitle("Stock vLLM, A100: small γ is faster at light load, every γ collapses under load")
    fig.text(0.99, 0.01, "hatched = single run", ha="right", fontsize=9, color="0.4")
    save(fig, "1_static_spec_regimes.png")


def fig_coordination():
    cases = [("chat", "4rps", "chat 4 rps"), ("chat", "16rps", "chat 16 rps"), ("chat", "saturation", "chat saturation"),
             ("phase-shift", "scheduled", "phase-shift"), ("mixed-bursty", "scheduled", "mixed-bursty")]
    fig, axes = plt.subplots(1, len(cases), figsize=(16, 4.4))
    print("\nCoordination (goodput, mean ± std over repeats)")
    for ax, (w, l, title) in zip(axes, cases):
        rows = [("indep.", "ours-independent-controllers"), ("co-opt.", "ours-cooptimizer"),
                ("indep.\n+ fix", "ours-independent-controllers-preempt-backoff"),
                ("co-opt.\n+ fix", "ours-cooptimizer-preempt-backoff")]
        vals = [metric(e, w, l, "goodput_req_s") for _, e in rows]
        bars(ax, [n for n, _ in rows], vals, ["0.8", "0.3", "0.8", "0.3"])
        ax.axvline(1.5, color="0.6", lw=0.8, ls=":")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(title)
        print(f"  {title:16s} " + "  ".join(f"{n.replace(chr(10), ' ')} {m:.2f}±{s:.2f}" for (n, _), (m, s, _) in zip(rows, vals))
              + f"  | gain {vals[1][0] / vals[0][0] - 1:+.0%}, with fix {vals[3][0] / vals[2][0] - 1:+.0%}")
    axes[0].set_ylabel("goodput (req/s)")
    fig.suptitle("Same three controllers, with vs without the shared view; left pair original, right pair with the "
                 "burst backoff fix (A100, mean ± std of 3)", fontsize=11)
    save(fig, "2_coordination.png")


def fig_ablation():
    exps = [("γ only", "ours-gamma-controller-only"), ("KV only, no parking", "ours-kv-controller-only-no-elastic"),
            ("sched only", "ours-scheduler-controller-only"), ("γ + KV", "ours-gamma-and-kv"),
            ("γ + sched", "ours-gamma-and-scheduler"), ("KV + sched", "ours-kv-and-scheduler"),
            ("all three, no parking", "ours-cooptimizer-no-elastic"), ("KV only + parking", "ours-kv-controller-only"),
            ("independent (all three)", "ours-independent-controllers"), ("co-optimizer", "ours-cooptimizer")]
    rows = sorted(((n, metric(e, "chat", "saturation", "goodput_req_s")) for n, e in exps), key=lambda r: r[1][0])
    colors = ["0.3" if n == "co-optimizer" else "0.75" for n, _ in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars(ax, [n for n, _ in rows], [v for _, v in rows], colors, horizontal=True)
    ax.set_xlabel("goodput at chat saturation (req/s)")
    ax.set_title("Ablations, A100 (hatched = single run; others mean ± std of 3)")
    save(fig, "3_ablation.png")


def fig_baselines():
    cases = [("chat", "16rps", "chat 16 rps"), ("chat", "saturation", "chat saturation"),
             ("phase-shift", "scheduled", "phase-shift"), ("mixed-bursty", "scheduled", "mixed-bursty")]
    exps = [("vLLM\nadaptive\nspec", "vllm-adaptive-spec-schedule"),
            ("Nightjar\n-like", "ours-nightjar-like"),
            ("co-opt.", "ours-cooptimizer"), ("co-opt.\n+ fix", "ours-cooptimizer-preempt-backoff")]
    fig, axes = plt.subplots(1, len(cases), figsize=(16, 4.6))
    print("\nBaselines (goodput; n = repeats)")
    for ax, (w, l, title) in zip(axes, cases):
        vals = [metric(e, w, l, "goodput_req_s") for _, e in exps]
        bars(ax, [n for n, _ in exps], vals, ["0.85", "0.6", "0.3", "0.15"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title)
        print("  " + title.ljust(16) + "  ".join(f"{n.replace(chr(10), ' ')} {m:.2f} (n={k})"
                                                for (n, _), (m, _, k) in zip(exps, vals)))
    axes[0].set_ylabel("goodput (req/s)")
    fig.suptitle("Against baselines, A100 (mean ± std of 3 repeats)")
    save(fig, "4_baselines.png")


def phase_slo(run):
    """SLO attainment per phase, computed from the per-request records."""
    out = {}
    for r in run["requests"]:
        ok = not r["error"] and r["ttft"] is not None
        if ok:
            ttft_slo = SLO["long"] if r["category"] == "long" else SLO["short"]
            n = max(r["output_tokens"] - 1, 1)
            ok = r["ttft"] <= ttft_slo and (r["e2e"] - r["ttft"]) / n <= SLO["tpot"]
        hit, tot = out.get(r["phase"], (0, 0))
        out[r["phase"]] = (hit + ok, tot + 1)
    return out


def fig_phase_shift():
    run = runs("ours-cooptimizer-preempt-backoff", "phase-shift", "scheduled")[0]
    ts = run["timeseries"]
    t = [p["t"] for p in ts]
    bounds = {}
    for r in run["requests"]:
        lo, hi = bounds.get(r["phase"], (1e9, 0))
        bounds[r["phase"]] = (min(lo, r["scheduled_s"]), max(hi, r["scheduled_s"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    series = [("coopt:gamma_mean", "mean γ used"), ("coopt:drafter_parked", "drafter parked"),
              ("coopt:num_requests_waiting", "requests waiting")]
    for ax, (k, name) in zip(axes, series):
        ax.plot(t, [p.get(k, 0) for p in ts], color="0.2", lw=1.2)
        ax.set_ylabel(name)
        for ph, (lo, _) in sorted(bounds.items(), key=lambda x: x[1][0]):
            ax.axvline(lo, color="0.6", ls="--", lw=0.8)
    for ph, (lo, hi) in bounds.items():
        axes[0].text((lo + hi) / 2, axes[0].get_ylim()[1] * 0.9, ph.split("_", 1)[-1], ha="center", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Co-optimizer (with burst fix) on the phase-shift workload (A100, rep0)")
    save(fig, "5_phase_shift_trace.png")

    print("\nPhase-shift SLO attainment per phase (mean over repeats)")
    for exp in ["vllm-default", "ours-static-no-spec", "vllm-adaptive-spec-schedule", "ours-nightjar-like",
                "ours-independent-controllers", "ours-cooptimizer", "ours-independent-controllers-preempt-backoff", "ours-cooptimizer-preempt-backoff"]:
        per = [phase_slo(r) for r in runs(exp, "phase-shift", "scheduled")]
        phases = sorted(per[0], key=lambda p: bounds[p][0])
        cells = [f"{p.split('_', 1)[-1]} {st.mean(d[p][0] / d[p][1] for d in per):.2f}" for p in phases]
        print(f"  {exp:30s} " + "  ".join(cells) + f"  (n={len(per)})")


def table_light_load():
    print("\nLight load, chat (TPOT p50 / p99 ms, mean over repeats)")
    for exp in ["vllm-default", "ours-static-no-spec", "vllm-draft-spec-gamma4", "vllm-adaptive-spec-schedule",
                "ours-independent-controllers", "ours-cooptimizer", "ours-independent-controllers-preempt-backoff", "ours-cooptimizer-preempt-backoff"]:
        cells = []
        for load in ["1rps", "4rps"]:
            p50, _, n = metric(exp, "chat", load, "tpot_p50")
            p99, _, _ = metric(exp, "chat", load, "tpot_p99")
            cells.append(f"{load} {p50 * 1000:5.1f} / {p99 * 1000:5.1f}")
        print(f"  {exp:30s} " + "   ".join(cells) + f"  (n={n})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_static_spec()
    fig_coordination()
    fig_ablation()
    fig_baselines()
    fig_phase_shift()
    table_light_load()
