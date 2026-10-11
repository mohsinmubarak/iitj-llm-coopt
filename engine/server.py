"""OpenAI-compatible HTTP server for our engine, a drop-in for `vllm serve` in the benchmark.

Endpoints: POST /v1/completions (streaming SSE, token-id prompts, ignore_eos, usage chunk),
GET /health, GET /metrics (Prometheus text). Shared counters use vLLM's metric names so
bench/run.py, bench/tracking.py and the W&B dashboard treat both systems identically;
engine-only signals (γ, α, KV tiers, policy knobs) are exported under `coopt:`.

The GPU engine runs in its own process (engine/worker.py); this process only handles HTTP,
detokenisation and streaming, so neither competes with the other for the GIL.

    python -m engine.server --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --draft unsloth/Llama-3.2-1B-Instruct --gamma 4 --port 8000
"""
import argparse
import asyncio
import itertools
import json
import multiprocessing as mp
import os
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from transformers import AutoTokenizer

from engine.co_optimizer import CoOptimizer
from engine.policy import StaticPolicy


class EngineClient:
    """Server-side handle on the engine process: submit requests, route batched outputs."""

    def __init__(self, args):
        ctx = mp.get_context("spawn")
        self.req_q, self.out_q = ctx.Queue(), ctx.Queue()
        from engine.worker import engine_worker
        self.proc = ctx.Process(target=engine_worker, args=(vars(args), self.req_q, self.out_q), daemon=True)
        self.proc.start()
        kind, info = self.out_q.get()
        if kind != "ready":
            raise RuntimeError(f"engine failed to start:\n{info}")
        self.max_model_len = info["max_model_len"]
        self.metrics = {"counters": {}, "gauges": {}, "extra": {}}
        self.streams = {}          # request id -> (loop, asyncio.Queue)
        self.loop = None
        threading.Thread(target=self._receive, daemon=True).start()

    def _receive(self):
        import queue as _queue
        while True:
            try:
                kind, payload = self.out_q.get(timeout=1.0)
            except _queue.Empty:
                if not self.proc.is_alive():  # hard crash without a "fatal" message
                    os._exit(1)
                continue
            if kind == "out":
                self.loop.call_soon_threadsafe(self._dispatch, payload)
            elif kind == "metrics":
                self.metrics = payload
            elif kind == "fatal":
                # the engine died (e.g. OOM): exit so clients see dropped connections, as with vLLM
                os._exit(1)

    def _dispatch(self, items):
        for rid, toks, done, reason, prompt_len, n_gen in items:
            q = self.streams.get(rid)
            if q is not None:
                q.put_nowait((toks, done, reason, prompt_len, n_gen))

    def submit(self, rid, prompt, max_tokens, temperature, ignore_eos):
        q = asyncio.Queue()
        self.streams[rid] = q
        self.req_q.put(("req", rid, prompt, max_tokens, temperature, ignore_eos))
        return q


def build_app(client: EngineClient, tokenizer, model_name: str) -> FastAPI:
    app = FastAPI()
    ids = itertools.count()

    @app.on_event("startup")
    async def bind_loop():
        client.loop = asyncio.get_running_loop()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok" if client.proc.is_alive() else "engine dead",
                                 status_code=200 if client.proc.is_alive() else 503)

    @app.get("/metrics")
    async def metrics():
        m = client.metrics
        lines = [f"vllm:{k} {v}" for k, v in m["counters"].items()]
        lines += [f"vllm:{k} {v}" for k, v in m["gauges"].items()
                  if k in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")]
        lines += [f"coopt:{k} {v}" for k, v in m["gauges"].items() if isinstance(v, (int, float))]
        lines += [f"coopt:{k} {v}" for k, v in m["extra"].items()]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt = body["prompt"]
        if isinstance(prompt, str):
            prompt = tokenizer(prompt).input_ids
        max_tokens = int(body.get("max_tokens", 16))
        if len(prompt) + max_tokens > client.max_model_len:
            return JSONResponse({"error": f"prompt ({len(prompt)}) + max_tokens ({max_tokens}) exceeds "
                                          f"max_model_len {client.max_model_len}"}, status_code=400)
        rid = next(ids)
        q = client.submit(rid, list(prompt), max_tokens, float(body.get("temperature", 1.0)),
                          bool(body.get("ignore_eos", False)))
        cid, created = f"cmpl-{rid}", int(time.time())
        final = {}

        def chunk(text, finish=None, usage=None):
            c = {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                 "choices": [] if usage else [{"index": 0, "text": text, "finish_reason": finish}]}
            if usage:
                c["usage"] = usage
            return f"data: {json.dumps(c)}\n\n"

        async def stream():
            try:
                while True:
                    toks, done, reason, prompt_len, n_gen = await q.get()
                    yield chunk(tokenizer.decode(toks), reason if done else None)
                    if done:
                        final.update(prompt_len=prompt_len, n_gen=n_gen)
                        break
                if (body.get("stream_options") or {}).get("include_usage"):
                    yield chunk("", usage={"prompt_tokens": final["prompt_len"], "completion_tokens": final["n_gen"],
                                           "total_tokens": final["prompt_len"] + final["n_gen"]})
                yield "data: [DONE]\n\n"
            finally:
                client.streams.pop(rid, None)

        if body.get("stream"):
            return StreamingResponse(stream(), media_type="text/event-stream")
        out = []
        try:
            while True:
                toks, done, reason, prompt_len, n_gen = await q.get()
                out += toks
                if done:
                    break
        finally:
            client.streams.pop(rid, None)
        return {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": tokenizer.decode(out), "finish_reason": reason}],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(out),
                          "total_tokens": prompt_len + len(out)}}

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
    ap.add_argument("--max-num-seqs", type=int, default=256, help="running batch cap (vLLM 0.31 default: 256)")
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

    client = EngineClient(args)
    tok = AutoTokenizer.from_pretrained(args.model)
    uvicorn.run(build_app(client, tok, args.model), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
