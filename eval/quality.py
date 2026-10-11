"""Quality experiments (need a free GPU).

Q1  KV quantization cost. A swap to the CPU tier quantizes a sequence's KV and restores it. Here a
    document's context KV is prefilled, quantized and dequantized in place (bits = 16 / 8 / 4), and
    the next `--cont` tokens are scored teacher-forced. Reports perplexity on the continuation, its
    change vs. bits=16, and how often greedy next-token predictions still match bits=16.
Q2  End-to-end losslessness of speculative decoding in our engine: greedy outputs with the 1B
    drafter at fixed γ vs. the same engine without a drafter, one request at a time.

    python -m eval.quality q1 --workload prompt-4k --n 32
    python -m eval.quality q2 --n 32 --new 128
Results: results/<profile>/quality/{q1,q2}.json (+ a W&B validation run).
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch

from bench.run import load_workload
from engine.batch import WorkItem, build_batch
from engine.block_manager import BlockManager
from engine.kv_tier_manager import dequantize, quantize
from engine.model import Llama

ROOT = Path(__file__).resolve().parent.parent
TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"
DRAFT = "unsloth/Llama-3.2-1B-Instruct"


def q1(args):
    model = Llama.from_pretrained(TARGET)
    docs = [r["prompt_token_ids"] for r in load_workload(args.workload)[: args.n]]
    max_len = max(len(d) for d in docs)
    bm = BlockManager(-(-max_len // 16) + 2)
    model.allocate_kv_cache(bm.num_blocks, 16)
    out = {"workload": args.workload, "docs": len(docs), "continuation_tokens": args.cont, "bits": {}}
    base_pred = {}
    for bits in (16, 8, 4):
        nll, n, agree = 0.0, 0, 0
        for i, doc in enumerate(docs):
            ctx, cont = doc[: -args.cont], doc[-args.cont:]
            batch, _ = build_batch([WorkItem(0, ctx, 0)], bm)
            model.forward(batch)
            if bits < 16:  # simulate swap-out + restore of the whole context KV
                blocks = torch.tensor(bm.tables[0][: bm.blocks_for(len(ctx))], device="cuda")
                for cache in model.k_cache + model.v_cache:
                    q, s = quantize(cache[blocks], bits)
                    cache[blocks] = dequantize(q, s, bits, cache.dtype)
            # teacher-forced: last context token + continuation[:-1] predict continuation
            feed = [ctx[-1]] + cont[:-1]
            bm.truncate(0, len(ctx) - 1)
            batch, _ = build_batch([WorkItem(0, feed, len(ctx) - 1, all_logits=True)], bm)
            logits = model.forward(batch)
            target = torch.tensor(cont, device="cuda")
            nll += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
            n += len(cont)
            pred = logits.argmax(-1).cpu()
            if bits == 16:
                base_pred[i] = pred
            agree += (pred == base_pred[i]).sum().item()
            bm.free(0)
        ppl = math.exp(nll / n)
        out["bits"][bits] = {"ppl": ppl, "top1_agreement_vs_bf16": agree / n}
        print(f"Q1 bits={bits:2d}: perplexity {ppl:.4f} | greedy agreement vs bf16 {agree / n:.4f}", flush=True)
    base = out["bits"][16]["ppl"]
    for bits in (8, 4):
        out["bits"][bits]["ppl_change_pct"] = 100 * (out["bits"][bits]["ppl"] - base) / base
    return out


def greedy_outputs(engine, prompts, new):
    from engine.engine import Sequence
    outs = []
    for i, p in enumerate(prompts):
        s = Sequence(seq_id=i, tokens=list(p), max_tokens=new, temperature=0.0, ignore_eos=True)
        engine.scheduler.waiting.append(s)
        while not s.finished:
            engine.step()
        outs.append(s.tokens[len(p):])
    return outs


def q2(args):
    from engine.engine import Engine
    from engine.policy import StaticPolicy
    prompts = [r["prompt_token_ids"] for r in load_workload("chat")[: args.n]]
    runs = {}
    for name, draft, gamma in (("spec_gamma4", DRAFT, 4), ("no_spec", None, 0)):
        eng = Engine(TARGET, draft, policy=StaticPolicy(gamma=gamma), max_model_len=4096,
                     gpu_memory_utilization=0.9)
        runs[name] = greedy_outputs(eng, prompts, args.new)
        acc = eng.counters["spec_decode_num_accepted_tokens_total"]
        drafted = eng.counters["spec_decode_num_draft_tokens_total"]
        if drafted:
            print(f"Q2 {name}: acceptance {acc / drafted:.3f}", flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()

    def prefix(a, b):
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        return k
    a, b = runs["spec_gamma4"], runs["no_spec"]
    res = {"prompts": len(prompts), "new_tokens": args.new,
           "exact_match_rate": sum(x == y for x, y in zip(a, b)) / len(a),
           "mean_matched_prefix": sum(prefix(x, y) for x, y in zip(a, b)) / len(a),
           "token_agreement": sum(sum(p == q for p, q in zip(x, y)) for x, y in zip(a, b)) / (len(a) * args.new)}
    print(f"Q2: exact match {res['exact_match_rate']:.3f} | mean matched prefix "
          f"{res['mean_matched_prefix']:.1f}/{args.new}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["q1", "q2"])
    ap.add_argument("--workload", default="prompt-4k")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--profile", default="rtx4090")
    args = ap.parse_args()
    res = q1(args) if args.which == "q1" else q2(args)
    out = ROOT / "results" / args.profile / "quality" / f"{args.which}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    try:
        import wandb
        import yaml
        prof = yaml.safe_load((ROOT / "bench" / "profiles" / f"{args.profile}.yaml").read_text())
        with wandb.init(project=os.environ.get("WANDB_PROJECT", prof["wandb_project"]), job_type="validation",
                        group="quality", name=f"quality | {dict(q1='kv-quantization', q2='spec-vs-no-spec')[args.which]}",
                        tags=["quality", args.which, args.profile],
                        config=vars(args), dir=str(ROOT)) as run:
            run.summary.update({k: v for k, v in res.items() if not isinstance(v, dict)})
            for bits, v in res.get("bits", {}).items():
                run.summary.update({f"bits{bits}/{k}": x for k, x in v.items()})
            run.save(str(out), base_path=str(out.parent), policy="now")
    except Exception as e:
        print(f"[Q] W&B logging failed: {e}")


if __name__ == "__main__":
    main()
