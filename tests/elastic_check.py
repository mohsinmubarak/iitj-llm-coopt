"""GPU integration check for memory-elastic speculation (engine/elastic.py).

Reference: greedy outputs with the drafter parked (plain decoding). Test: the same requests start
with speculation, the drafter is parked mid-run so sequences spill into the drafter's former
memory, then unparked mid-run (live blocks migrate out of the tail, weights are restored) and the
requests finish with speculation. Speculation is lossless and migration must not corrupt KV, so the
outputs must match the reference up to bf16 near-ties. Exit code 1 on failure.

    python -m tests.elastic_check
"""
import sys

import torch

from bench.run import load_workload
from engine.engine import Engine, Sequence
from engine.policy import StaticPolicy

TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"
DRAFT = "unsloth/Llama-3.2-1B-Instruct"
NEW = 96


def submit(engine, prompts, base_id):
    seqs = [Sequence(seq_id=base_id + i, tokens=list(p), max_tokens=NEW, temperature=0.0, ignore_eos=True)
            for i, p in enumerate(prompts)]
    engine.scheduler.waiting.extend(seqs)
    return seqs


def run(engine, seqs, hook=None):
    step = 0
    while not all(s.finished for s in seqs):
        engine.step()
        step += 1
        if hook:
            hook(step)
    return [s.tokens[s.prompt_len:] for s in seqs]


def main():
    reqs = load_workload("prompt-1k")[:24] + load_workload("chat")[:40]
    prompts = [r["prompt_token_ids"] for r in reqs]
    e = Engine(TARGET, DRAFT, policy=StaticPolicy(gamma=4, max_num_seqs=128), max_model_len=4096,
               gpu_memory_utilization=0.92, elastic=True)
    el = e.elastic
    base_tokens = el.base * el.block_size
    need = sum(len(p) + NEW for p in prompts)
    print(f"base pool {base_tokens} tokens, parked pool {(el.base + el.tail) * el.block_size}, "
          f"workload needs {need}", flush=True)

    # reference: drafter parked for the whole run
    el.park()
    ref = run(e, submit(e, prompts, 0))

    # test: start with speculation, park mid-run, unpark mid-run
    el.unpark()
    assert not el.parked
    events = {}

    def hook(step):
        bm = e.bm
        if step == 8 and not el.parked:
            el.park()
            events["parked_at"] = step
        if "parked_at" in events and "unparked_at" not in events and step >= 40:
            in_tail = sum(b in el.tail_ids for t in bm.tables.values() for b in t)
            events.setdefault("tail_blocks_in_use_before_unpark", in_tail)
            if el.unpark():
                events["unparked_at"] = step
                events["migrated"] = e.counters["drafter_migrated_blocks_total"]

    out = run(e, submit(e, prompts, 10_000), hook)
    total = sum(len(r) for r in ref)
    agree = sum(sum(a == b for a, b in zip(x, y)) for x, y in zip(out, ref))
    acc = e.counters["spec_decode_num_accepted_tokens_total"]
    drafted = e.counters["spec_decode_num_draft_tokens_total"]
    print(f"events: {events}", flush=True)
    print(f"speculation: acceptance {acc / max(drafted, 1):.3f} over {drafted} draft tokens")
    print(f"token agreement with the parked reference {agree / total:.4f} (informative only: bf16 near-ties diverge)")

    # Validity: re-score every generated token with a fresh teacher-forced pass in clean KV.
    # A greedy token is "confidently wrong" if it is not the argmax and loses by > 0.5 logits;
    # corrupted KV or weights produce many of those, bf16 noise essentially none.
    from engine.batch import WorkItem, build_batch
    bad = {"reference": 0, "elastic": 0}
    for name, outs in (("reference", ref), ("elastic", out)):
        for i, (p, o) in enumerate(zip(prompts, outs)):
            sid = 900_000 + i
            batch, _ = build_batch([WorkItem(sid, list(p), 0)], e.bm)
            e.target.forward(batch)
            batch, _ = build_batch([WorkItem(sid, [p[-1]] + o[:-1], len(p) - 1, all_logits=True)], e.bm)
            logits = e.target.forward(batch)
            chosen = logits.gather(1, torch.tensor(o, device="cuda").unsqueeze(1)).squeeze(1)
            bad[name] += int(((logits.max(-1).values - chosen) > 0.5).sum())
            e.bm.free(sid)
    print(f"confidently-wrong tokens: reference {bad['reference']}/{total}, elastic {bad['elastic']}/{total}")
    ok = ("unparked_at" in events and events.get("tail_blocks_in_use_before_unpark", 0) > 0
          and bad["elastic"] <= max(2, 0.002 * total) and drafted > 0 and acc / drafted > 0.3)
    print("ELASTIC CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
