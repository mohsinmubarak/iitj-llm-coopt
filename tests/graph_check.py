"""CUDA-graph path vs eager path: same logits for decode / verify-shaped batches, and the speed-up.

    python -m tests.graph_check --model unsloth/Llama-3.2-1B-Instruct
"""
import argparse
import random
import time

import torch

from engine.batch import WorkItem, build_batch
from engine.block_manager import BlockManager
from engine.cuda_graphs import GraphRunner
from engine.model import Llama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    args = ap.parse_args()
    rng = random.Random(0)
    m = Llama.from_pretrained(args.model)
    nb = m.kv_blocks_for_bytes(2 * 2**30)
    bm = BlockManager(nb)
    m.allocate_kv_cache(nb + 1, bm.block_size)
    runner = GraphRunner(m, 4096, [1, 2, 5], [1, 2, 4, 8, 16, 32], scratch_block=nb)
    t = time.perf_counter()
    runner.capture()
    print(f"captured {len(runner.graphs)} graphs in {time.perf_counter() - t:.1f}s")

    for B in (1, 3, 8, 32):
        # prefill B sequences of different lengths eagerly
        lens = [rng.randint(20, 300) for _ in range(B)]
        for i, L in enumerate(lens):
            batch, _ = build_batch([WorkItem(i, [rng.randint(100, 30000) for _ in range(L)], 0)], bm)
            m.forward(batch)
        for q in (1, 2, 5):  # decode, draft catch-up, verify with γ=4
            toks = [[rng.randint(100, 30000) for _ in range(rng.randint(1, q))] for _ in range(B)]
            items = [WorkItem(i, toks[i], lens[i], all_logits=True) for i in range(B)]
            batch, spans = build_batch(items, bm)
            eager = m.forward(batch)
            g_logits, S = runner.run([(i, toks[i], lens[i]) for i in range(B)], bm)
            max_diff, agree, n = 0.0, 0, 0
            for i, (row, cnt) in enumerate(spans):
                e = eager[row:row + cnt]
                g = g_logits[i * S: i * S + cnt].float()
                max_diff = max(max_diff, (e - g).abs().max().item())
                agree += (e.argmax(-1) == g.argmax(-1)).sum().item()
                n += cnt
            # timing: eager build+forward vs graph run (both include host->device input copies)
            def eager_step():
                b, _ = build_batch(items, bm)
                m.forward(b).argmax(-1).tolist()

            def graph_step():
                runner.run([(i, toks[i], lens[i]) for i in range(B)], bm)[0].argmax(-1).tolist()

            ts = {}
            for name, fn in (("eager", eager_step), ("graph", graph_step)):
                fn()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(20):
                    fn()
                ts[name] = (time.perf_counter() - t) / 20 * 1e3
            print(f"B={B:2d} q<={q}: argmax agree {agree}/{n}, max|dlogit| {max_diff:.3f} | "
                  f"eager {ts['eager']:.2f} ms -> graph {ts['graph']:.2f} ms ({ts['eager'] / ts['graph']:.1f}x)")
        for i in range(B):
            bm.free(i)


if __name__ == "__main__":
    main()
