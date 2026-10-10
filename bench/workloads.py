"""Seeded workload builders.

Every workload is a JSONL file of requests holding *exact prompt token ids*, so vLLM and our
engine receive byte-identical inputs. Build once, commit the files' hashes, never regenerate.

    python -m bench.workloads --all            # W1, W2, W3 + fixed-length sets for A3
    python -m bench.workloads --only W1 --n 50 # quick smoke set

Request schema (one JSON object per line):
    id, category ("short" | "long"), prompt_token_ids, prompt_len, max_tokens, arrival_s (W3 only)
"""
import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 2026

LONGBENCH_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
                   "musique", "gov_report", "qmsum", "multi_news"]
FIXED_LENGTHS = [1024, 4096, 8192, 16384, 31744]  # A3; 31744 + 128 output + chat overhead < 32768


def load_sharegpt():
    """First (human, gpt) turn of every ShareGPT conversation."""
    path = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
                           "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    pairs = []
    for c in convs:
        turns = c.get("conversations", [])
        if len(turns) >= 2 and turns[0]["from"] == "human" and turns[1]["from"] == "gpt":
            pairs.append((turns[0]["value"], turns[1]["value"]))
    return pairs


def load_longbench():
    """(context, question) pairs from the English LongBench v1 tasks."""
    path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    items = []
    with zipfile.ZipFile(path) as z:
        for task in LONGBENCH_TASKS:
            with z.open(f"data/{task}.jsonl") as f:
                for line in f:
                    row = json.loads(line)
                    items.append((task, row["context"], row["input"]))
    return items


def chat_ids(tok, text):
    out = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):  # transformers >= 5 returns a BatchEncoding
        out = out["input_ids"]
    return list(out)


def build_short(tok, n, rng):
    """ShareGPT prompts 4–1024 tokens; output length = reference answer length clamped to [32, 512]."""
    pairs = load_sharegpt()
    rng.shuffle(pairs)
    out = []
    for prompt, answer in pairs:
        ids = chat_ids(tok, prompt)
        if not 4 <= len(ids) <= 1024:
            continue
        out_len = len(tok(answer, add_special_tokens=False).input_ids)
        out.append({"category": "short", "prompt_token_ids": ids, "prompt_len": len(ids),
                    "max_tokens": int(np.clip(out_len, 32, 512))})
        if len(out) == n:
            break
    return out


def build_long(tok, n, rng, lo, hi, max_tokens=256):
    """LongBench prompts whose chat-formatted length falls in [lo, hi] tokens."""
    items = load_longbench()
    rng.shuffle(items)
    out = []
    for task, context, question in items:
        text = f"{context}\n\nAnswer the question based on the text above.\nQuestion: {question}\nAnswer:"
        ids = chat_ids(tok, text)
        if lo <= len(ids) <= hi:
            out.append({"category": "long", "task": task, "prompt_token_ids": ids,
                        "prompt_len": len(ids), "max_tokens": max_tokens})
            if len(out) == n:
                break
    return out


def bursty_arrivals(n, rng, base_rate=1.5, burst_rate=6.0, period_s=60.0, burst_s=20.0):
    """Non-homogeneous Poisson arrivals (thinning): base_rate, with a burst_rate window every period."""
    t, times = 0.0, []
    while len(times) < n:
        t += rng.exponential(1.0 / burst_rate)
        rate = burst_rate if (t % period_s) >= period_s - burst_s else base_rate
        if rng.random() < rate / burst_rate:
            times.append(round(t, 4))
    return times


def build_fixed(tok, length, n, output_len=128):
    """Fixed-length prompts for the context-length sweep (A3): LongBench text cut to exactly `length` ids."""
    items = load_longbench()
    corpus = tok("\n\n".join(c for _, c, _ in items[:400]), add_special_tokens=False).input_ids
    rng = random.Random(SEED + length)
    out = []
    for _ in range(n):
        start = rng.randrange(0, len(corpus) - length)
        ids = [tok.bos_token_id] + corpus[start:start + length - 1]
        out.append({"category": "long" if length >= 4096 else "short", "prompt_token_ids": ids,
                    "prompt_len": len(ids), "max_tokens": output_len})
    return out


def write(name, reqs):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(reqs):
            f.write(json.dumps({"id": f"{name}-{i:04d}", **r}) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    lens = [r["prompt_len"] for r in reqs]
    print(f"{name}: {len(reqs)} reqs | prompt len min/median/max = "
          f"{min(lens)}/{int(np.median(lens))}/{max(lens)} | sha256 {digest}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["W1", "W2", "W3", "fixed"], action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, help="override request count (smoke tests)")
    args = ap.parse_args()
    todo = ["W1", "W2", "W3", "fixed"] if args.all or not args.only else args.only

    tok = AutoTokenizer.from_pretrained(TARGET)
    rng = np.random.default_rng(SEED)
    pyrng = random.Random(SEED)

    if "W1" in todo:
        write("W1", build_short(tok, args.n or 200, pyrng))
    if "W2" in todo:
        write("W2", build_long(tok, args.n or 60, pyrng, 8192, 31744))
    if "W3" in todo:
        n = args.n or 300
        n_long = int(round(0.3 * n))
        reqs = build_short(tok, n - n_long, random.Random(SEED + 3)) + \
            build_long(tok, n_long, random.Random(SEED + 4), 4096, 16384)
        random.Random(SEED + 5).shuffle(reqs)
        for r, t in zip(reqs, bursty_arrivals(n, rng)):
            r["arrival_s"] = t
        write("W3", reqs)
    if "fixed" in todo:
        for length in FIXED_LENGTHS:
            write(f"fixed_{length}", build_fixed(tok, length, args.n or 32))


if __name__ == "__main__":
    main()
