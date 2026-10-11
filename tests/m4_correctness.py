"""M4: engine correctness against HF transformers (needs a free GPU).

Three checks, each written to results/<profile>/M4/<model>.json:
  1. teacher-forced: HF's greedy continuation is fed through our model in one prefill; the share
     of positions where our argmax equals HF's next token (robust: errors cannot compound).
  2. last-position logits: max |Δlogit| and top-1 agreement after prefill.
  3. free-running greedy, single request and batched: exact-match prefix length vs HF generate().
bf16 near-ties can legitimately flip a token between two attention kernels, so check 1 is the
pass/fail criterion (>= 99%) and check 3 is reported for transparency.

    python -m tests.m4_correctness --model unsloth/Llama-3.2-1B-Instruct --n 16 --new 64
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from bench.run import load_workload
from engine.batch import WorkItem, build_batch
from engine.block_manager import BlockManager
from engine.model import Llama

ROOT = Path(__file__).resolve().parent.parent


def hf_reference(model_name, prompts, new_tokens):
    """HF greedy continuations plus, per continuation position, HF's argmax and top1-top2 margin.
    Also the noise floor: HF with eager attention vs HF with SDPA on the same teacher-forced input."""
    hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").cuda().eval()
    outs, last_logits, margins = [], [], []
    with torch.inference_mode():
        for p in prompts:
            ids = torch.tensor([p], device="cuda")
            last_logits.append(hf(ids).logits[0, -1].float().cpu())
            g = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=new_tokens,
                            do_sample=False, temperature=None, top_p=None, top_k=None, max_length=None)
            outs.append(g[0, len(p):].tolist())
            full = hf(g).logits[0, len(p) - 1: g.shape[1] - 1].float()
            top2 = full.topk(2, dim=-1).values
            margins.append((top2[:, 0] - top2[:, 1]).cpu())
        floor_agree = floor_total = 0
        try:  # noise floor: same weights, different attention kernel
            hf.set_attn_implementation("eager")
            for p, cont in zip(prompts, outs):
                g = torch.tensor([p + cont], device="cuda")
                pred = hf(g).logits[0, len(p) - 1: g.shape[1] - 1].argmax(-1).cpu()
                floor_agree += (pred == torch.tensor(cont)).sum().item()
                floor_total += len(cont)
        except Exception as e:
            print(f"[M4] noise floor unavailable: {e}")
    del hf
    torch.cuda.empty_cache()
    floor = floor_agree / floor_total if floor_total else None
    return outs, last_logits, margins, floor


def greedy(model, bm, prompts, new_tokens, seq_base):
    """Batched greedy decoding through our engine primitives; returns generated ids per prompt."""
    seqs = {seq_base + i: list(p) for i, p in enumerate(prompts)}
    gen = {s: [] for s in seqs}
    items = [WorkItem(s, toks, 0) for s, toks in seqs.items()]
    for _ in range(new_tokens):
        batch, spans = build_batch(items, bm)
        nxt = model.forward(batch).argmax(-1).tolist()
        items = []
        for (s, toks), (row, _) in zip(seqs.items(), spans):
            gen[s].append(nxt[row])
            items.append(WorkItem(s, [nxt[row]], len(toks)))
            toks.append(nxt[row])
    for s in seqs:
        bm.free(s)
    return [gen[s] for s in seqs]


def matched_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=16, help="prompts")
    ap.add_argument("--new", type=int, default=64, help="generated tokens per prompt")
    ap.add_argument("--kv-gb", type=float, default=2.0)
    ap.add_argument("--profile", default="rtx4090")
    args = ap.parse_args()

    reqs = load_workload("W1")[: args.n - 2] + load_workload("fixed_1024")[:2]
    prompts = [r["prompt_token_ids"] for r in reqs]
    ref, ref_logits, ref_margins, noise_floor = hf_reference(args.model, prompts, args.new)

    model = Llama.from_pretrained(args.model)
    bm = BlockManager(model.kv_blocks_for_bytes(args.kv_gb * 2**30))
    model.allocate_kv_cache(bm.num_blocks, bm.block_size)

    # 1 + 2: teacher-forced prefill over prompt + HF continuation
    agree = total = 0
    max_dlogit, top1_last, last_margins = 0.0, 0, []
    disagree_margins = []  # HF's top1-top2 margin wherever we pick a different token
    for i, (p, cont) in enumerate(zip(prompts, ref)):
        full = p + cont
        batch, _ = build_batch([WorkItem(i, full, 0, all_logits=True)], bm)
        logits = model.forward(batch)
        pred = logits[len(p) - 1: len(full) - 1].argmax(-1).cpu()
        hit = pred == torch.tensor(cont)
        agree += hit.sum().item()
        total += len(cont)
        disagree_margins += ref_margins[i][~hit].tolist()
        last = logits[len(p) - 1].cpu()
        max_dlogit = max(max_dlogit, (last - ref_logits[i]).abs().max().item())
        hit_last = int(last.argmax() == ref_logits[i].argmax())
        top1_last += hit_last
        if not hit_last:  # HF top1-top2 margin where the last-prefill-token argmax differs
            t2 = ref_logits[i].topk(2).values
            last_margins.append((t2[0] - t2[1]).item())
        bm.free(i)

    # 3: free-running greedy, one request at a time and all together
    single = [greedy(model, bm, [p], args.new, 1000 + i)[0] for i, p in enumerate(prompts)]
    batched = greedy(model, bm, prompts, args.new, 2000)
    res = {
        "model": args.model, "prompts": len(prompts), "new_tokens": args.new,
        "teacher_forced_agreement": agree / total,
        "last_logit_max_abs_diff": max_dlogit,
        "last_logit_top1_agreement": top1_last / len(prompts),
        "single_exact_match_rate": sum(a == b for a, b in zip(single, ref)) / len(ref),
        "single_mean_matched_prefix": sum(matched_prefix(a, b) for a, b in zip(single, ref)) / len(ref),
        "batched_exact_match_rate": sum(a == b for a, b in zip(batched, ref)) / len(ref),
        "batched_mean_matched_prefix": sum(matched_prefix(a, b) for a, b in zip(batched, ref)) / len(ref),
        "batched_equals_single": sum(a == b for a, b in zip(batched, single)) / len(ref),
    }
    res["hf_eager_vs_sdpa_agreement"] = noise_floor  # how often HF disagrees with itself
    res["disagreements"] = len(disagree_margins)
    res["last_token_disagreement_hf_margins"] = last_margins
    res["disagreement_hf_margin_max"] = max(disagree_margins, default=0.0)
    res["disagreement_hf_margin_median"] = float(torch.tensor(disagree_margins).median()) if disagree_margins else 0.0
    # Pass: we agree with HF at least as often as HF's two attention kernels agree with each other
    # (minus 0.5 pp), or >= 99%; and every disagreement is a near-tie (HF margin < 1 logit), which
    # a real bug would not produce.
    floor_ok = noise_floor is not None and res["teacher_forced_agreement"] >= noise_floor - 0.005
    res["pass"] = ((res["teacher_forced_agreement"] >= 0.99 or floor_ok)
                   and max(last_margins, default=0.0) < 1.0
                   and res["disagreement_hf_margin_max"] < 1.0)
    out = ROOT / "results" / args.profile / "M4" / f"{args.model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    try:
        import os

        import wandb
        import yaml
        prof = yaml.safe_load((ROOT / "bench" / "profiles" / f"{args.profile}.yaml").read_text())
        with wandb.init(project=os.environ.get("WANDB_PROJECT", prof["wandb_project"]), job_type="validation",
                        group="M4", name=f"M4|{args.model.split('/')[-1]}", tags=["M4", "validation", args.profile],
                        config=vars(args), dir=str(ROOT)) as run:
            run.summary.update(res)
            run.save(str(out), base_path=str(out.parent), policy="now")
    except Exception as e:
        print(f"[M4] W&B logging failed: {e}")


if __name__ == "__main__":
    main()
