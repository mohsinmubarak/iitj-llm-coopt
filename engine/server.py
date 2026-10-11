"""OpenAI-compatible HTTP server for our engine, a drop-in for `vllm serve` in the benchmark.

Endpoints: POST /v1/completions (streaming SSE, token-id prompts, ignore_eos, usage chunk),
GET /health, GET /metrics (Prometheus text). Shared counters use vLLM's metric names so
bench/run.py, bench/tracking.py and the W&B dashboard treat both systems identically;
engine-only signals (γ, α, KV tiers, policy knobs) are exported under `coopt:`.

    python -m engine.server --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --draft unsloth/Llama-3.2-1B-Instruct --gamma 4 --port 8000
"""
import argparse
import asyncio
import itertools
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from transformers import AutoTokenizer

from engine.co_optimizer import CoOptimizer
from engine.engine import Engine, Sequence
from engine.kv_tier_manager import KVTierManager
from engine.metrics_engine import MetricsEngine
from engine.policy import StaticPolicy


def build_app(engine: Engine, tokenizer, model_name: str) -> FastAPI:
    app = FastAPI()
    ids = itertools.count()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.get("/metrics")
    async def metrics():
        lines = [f"vllm:{k} {v}" for k, v in engine.counters.items()]
        lines += [f"vllm:{k} {v}" for k, v in engine.gauges.items()
                  if k in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")]
        lines += [f"coopt:{k} {v}" for k, v in engine.gauges.items() if isinstance(v, (int, float))]
        lines += [f"coopt:{k} {v}" for k, v in getattr(engine, "extra_metrics", {}).items()]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt = body["prompt"]
        if isinstance(prompt, str):
            prompt = tokenizer(prompt).input_ids
        max_tokens = int(body.get("max_tokens", 16))
        if len(prompt) + max_tokens > engine.max_model_len:
            return JSONResponse({"error": f"prompt ({len(prompt)}) + max_tokens ({max_tokens}) exceeds "
                                          f"max_model_len {engine.max_model_len}"}, status_code=400)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def on_output(seq, new_tokens, finished):
            loop.call_soon_threadsafe(q.put_nowait, (new_tokens, finished))

        seq = Sequence(seq_id=next(ids), tokens=list(prompt), max_tokens=max_tokens,
                       temperature=float(body.get("temperature", 1.0)),
                       ignore_eos=bool(body.get("ignore_eos", False)), on_output=on_output)
        engine.add_request(seq)
        rid, created = f"cmpl-{seq.seq_id}", int(time.time())

        def chunk(text, finish=None, usage=None):
            c = {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                 "choices": [] if usage else [{"index": 0, "text": text, "finish_reason": finish}]}
            if usage:
                c["usage"] = usage
            return f"data: {json.dumps(c)}\n\n"

        async def stream():
            while True:
                toks, done = await q.get()
                yield chunk(tokenizer.decode(toks), seq.finish_reason if done else None)
                if done:
                    break
            if (body.get("stream_options") or {}).get("include_usage"):
                n = seq.num_generated
                yield chunk("", usage={"prompt_tokens": seq.prompt_len, "completion_tokens": n,
                                       "total_tokens": seq.prompt_len + n})
            yield "data: [DONE]\n\n"

        if body.get("stream"):
            return StreamingResponse(stream(), media_type="text/event-stream")
        out = []
        while True:
            toks, done = await q.get()
            out += toks
            if done:
                break
        return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": tokenizer.decode(out), "finish_reason": seq.finish_reason}],
                "usage": {"prompt_tokens": seq.prompt_len, "completion_tokens": len(out),
                          "total_tokens": seq.prompt_len + len(out)}}

    return app


def make_policy(args):
    common = dict(gamma=args.gamma, max_num_seqs=args.max_num_seqs,
                  max_batched_tokens=args.max_num_batched_tokens, tick_every=args.tick_every)
    if args.policy == "static":
        return StaticPolicy(**common, kv_swap=args.kv_cpu_gb > 0)
    if args.policy == "coopt":
        on = set(args.controllers.split(","))
        return CoOptimizer(**common, enable_gamma="gamma" in on, enable_kv="kv" in on,
                           enable_sched="sched" in on, coupled=not args.independent,
                           gamma_max=args.gamma_max)
    raise ValueError(f"unknown policy {args.policy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--policy", default="static")
    ap.add_argument("--gamma", type=int, default=4, help="fixed γ (static policy, or γ when the γ controller is off)")
    ap.add_argument("--gamma-max", type=int, default=8)
    ap.add_argument("--controllers", default="gamma,kv,sched", help="coopt: which controllers are on")
    ap.add_argument("--independent", action="store_true", help="coopt: no shared snapshot (C6)")
    ap.add_argument("--tick-every", type=int, default=8)
    ap.add_argument("--kv-cpu-gb", type=float, default=0.0, help="CPU KV tier size; 0 disables tiering")
    ap.add_argument("--kv-bits", type=int, default=8, choices=[4, 8, 16])
    ap.add_argument("--trace", default=None, help="JSONL file for per-tick snapshots")
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048,
                    help="prefill chunk / tokens per step; 2048 = vLLM 0.31 default on these GPUs")
    ap.add_argument("--kv-gb", type=float, default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", help="accepted for CLI parity; bf16 only")
    ap.add_argument("--no-cuda-graphs", action="store_true", help="eager decode (debugging)")
    ap.add_argument("--elastic", action="store_true",
                    help="memory-elastic speculation: drafter in the KV buffer tail, parkable under pressure")
    args = ap.parse_args()

    engine = Engine(args.model, args.draft, policy=make_policy(args), kv_gb=args.kv_gb,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len, seed=args.seed, cuda_graphs=not args.no_cuda_graphs,
                    elastic=args.elastic)
    if args.kv_cpu_gb > 0:
        KVTierManager(engine, cpu_gb=args.kv_cpu_gb, bits=args.kv_bits)
    MetricsEngine(engine, trace_path=args.trace)
    engine.policy.engine = engine
    print(f"[engine] hardware {engine.metrics.profile}", flush=True)
    print(f"[engine] KV capacity {engine.kv_capacity_tokens} tokens "
          f"({engine.bm.num_blocks} blocks), policy={engine.policy.name}", flush=True)
    engine.start()
    tok = AutoTokenizer.from_pretrained(args.model)
    uvicorn.run(build_app(engine, tok, args.model), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
