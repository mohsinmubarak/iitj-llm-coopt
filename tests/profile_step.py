"""Where does a decode step's time go? Batch building vs forward pass, draft vs target.

    python -m tests.profile_step --model unsloth/Llama-3.2-1B-Instruct --batch 1 8 32
"""
import argparse
import time

import torch

from engine.batch import WorkItem, build_batch
from engine.block_manager import BlockManager
from engine.model import Llama


def timed(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()
    m = Llama.from_pretrained(args.model)
    bm = BlockManager(m.kv_blocks_for_bytes(2 * 2**30))
    m.allocate_kv_cache(bm.num_blocks, bm.block_size)
    for B in args.batch:
        for s in range(B):
            bm.ensure(s, args.ctx + 1)
        items = [WorkItem(s, [1], args.ctx) for s in range(B)]
        batch, _ = build_batch(items, bm)
        t_build = timed(lambda: build_batch(items, bm))
        t_fwd = timed(lambda: m.forward(batch))
        t_fwd_argmax = timed(lambda: m.forward(batch).argmax(-1).tolist())
        print(f"{args.model.split('/')[-1]} B={B:3d} ctx={args.ctx}: build_batch {t_build:6.2f} ms | "
              f"forward {t_fwd:6.2f} ms | forward+argmax+sync {t_fwd_argmax:6.2f} ms")
        for s in range(B):
            bm.free(s)


if __name__ == "__main__":
    main()
