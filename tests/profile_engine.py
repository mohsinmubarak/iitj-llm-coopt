"""Where does a saturated engine step go? (no drafter, W1 prompts all at once)

Times each phase of Engine.step with wall clock and the GPU part with CUDA events, for
decode-only steps at the batch sizes that occur.
    python -m tests.profile_engine --n 200
"""
import argparse
import collections
import time

import numpy as np
import torch

from bench.run import load_workload
from engine.engine import Engine, Sequence
from engine.policy import StaticPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--new", type=int, default=256)
    args = ap.parse_args()
    e = Engine("unsloth/Meta-Llama-3.1-8B-Instruct", None, policy=StaticPolicy(gamma=0, max_batched_tokens=2048),
               max_model_len=16384, gpu_memory_utilization=0.92)
    for i, r in enumerate(load_workload("W1")[: args.n]):
        e.scheduler.waiting.append(Sequence(seq_id=i, tokens=list(r["prompt_token_ids"]), max_tokens=args.new,
                                            temperature=0.0, ignore_eos=True))

    # instrument: wrap the phases of a step
    phases = collections.defaultdict(list)
    orig_schedule, orig_run = e.scheduler.schedule, e._run
    gpu = []

    def schedule():
        t = time.perf_counter()
        p = orig_schedule()
        phases["schedule"].append(time.perf_counter() - t)
        phases["_decodes"].append(len(p.decodes))
        phases["_chunks"].append(len(p.chunks))
        return p

    def run(model, runner, bm, items):
        t = time.perf_counter()
        s, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_run(model, runner, bm, items)
        en.record()
        phases["run_launch"].append(time.perf_counter() - t)
        gpu.append((s, en))
        return out

    e.scheduler.schedule, e._run = schedule, run
    steps = []
    while e.scheduler.running or e.scheduler.waiting:
        t = time.perf_counter()
        e.step()
        steps.append(time.perf_counter() - t)
    torch.cuda.synchronize()
    g = [s.elapsed_time(en) / 1e3 for s, en in gpu]
    dec = np.array(phases["_decodes"])
    chunks = np.array(phases["_chunks"])
    steps, g = np.array(steps), np.array(g)
    sched, launch = np.array(phases["schedule"]), np.array(phases["run_launch"])
    for lo, hi in ((1, 64), (64, 128), (128, 192), (192, 257)):
        m = (dec >= lo) & (dec < hi) & (chunks == 0)
        if m.sum() < 3:
            continue
        print(f"decode-only steps, batch {lo:3d}-{hi - 1:3d} (n={m.sum():4d}): step {1e3 * steps[m].mean():6.2f} ms | "
              f"GPU {1e3 * g[m].mean():6.2f} ms | schedule {1e3 * sched[m].mean():5.2f} | "
              f"prep+launch {1e3 * launch[m].mean():5.2f} | rest {1e3 * (steps[m] - g[m] - sched[m]).mean():5.2f} ms")
    m = chunks > 0
    print(f"steps with prefill chunks: {m.sum()} (mean {1e3 * steps[m].mean():.1f} ms, GPU {1e3 * g[m].mean():.1f} ms)")
    gen = e.counters["generation_tokens_total"]
    print(f"total: {len(steps)} steps, {steps.sum():.1f} s, {gen} tokens -> {gen / steps.sum():.0f} tok/s; "
          f"GPU busy {g.sum() / steps.sum():.0%}")


if __name__ == "__main__":
    main()
