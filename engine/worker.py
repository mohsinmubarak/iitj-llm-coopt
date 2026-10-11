"""Engine process: owns the GPU and runs the step loop; talks to the HTTP server through queues.

With the server and engine in one process, the server's per-token work (detokenising, JSON, SSE
for every sequence every step) competed with the step loop for Python's GIL; at 100+ concurrent
sequences that showed up as ~30 % CPU overhead per step on the 4090 rehearsal. vLLM separates the
two for the same reason. Here the server only pickles requests in and unpickles one batched
message per engine step out.

Messages
  in  (req_q):  ("req", rid, prompt_ids, max_tokens, temperature, ignore_eos)
  out (out_q):  ("ready", info) once; ("out", [(rid, new_ids, done, finish_reason, prompt_len,
                n_generated), ...]) per step; ("metrics", {...}) every 0.5 s; ("fatal", text)
"""
import os
import queue
import sys
import time
import traceback


def engine_worker(args: dict, req_q, out_q):
    try:
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        from engine.engine import Engine, Sequence
        from engine.kv_tier_manager import KVTierManager
        from engine.metrics_engine import MetricsEngine
        from engine.server import make_policy

        class A:  # attribute access for make_policy
            pass
        a = A()
        a.__dict__.update(args)
        engine = Engine(a.model, a.draft, policy=make_policy(a), kv_gb=a.kv_gb,
                        gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
                        seed=a.seed, cuda_graphs=not a.no_cuda_graphs, elastic=a.elastic)
        if a.kv_cpu_gb > 0:
            KVTierManager(engine, cpu_gb=a.kv_cpu_gb, bits=a.kv_bits)
        MetricsEngine(engine, trace_path=a.trace)
        engine.policy.engine = engine
        print(f"[engine] hardware {engine.metrics.profile}", flush=True)
        print(f"[engine] KV capacity {engine.kv_capacity_tokens} tokens ({engine.bm.num_blocks} blocks), "
              f"policy={engine.policy.name}", flush=True)
        out_q.put(("ready", {"max_model_len": engine.max_model_len, "kv_capacity_tokens": engine.kv_capacity_tokens}))

        pending = []

        def collect(seq, new_tokens, done):
            pending.append((seq.seq_id, new_tokens, done, seq.finish_reason, seq.prompt_len, seq.num_generated))

        last_metrics = 0.0
        while True:
            busy = bool(engine.scheduler.running or engine.scheduler.waiting)
            try:
                msg = req_q.get(timeout=0.05) if not busy else req_q.get_nowait()
                while True:
                    _, rid, prompt, max_tokens, temperature, ignore_eos = msg
                    engine.scheduler.waiting.append(Sequence(
                        seq_id=rid, tokens=list(prompt), max_tokens=max_tokens, temperature=temperature,
                        ignore_eos=ignore_eos, on_output=collect))
                    msg = req_q.get_nowait()
            except queue.Empty:
                pass
            if engine.scheduler.running or engine.scheduler.waiting:
                engine.step()
            if pending:
                out_q.put(("out", pending))
                pending = []
            now = time.perf_counter()
            if now - last_metrics >= 0.5:
                out_q.put(("metrics", {"counters": dict(engine.counters), "gauges": dict(engine.gauges),
                                       "extra": dict(engine.extra_metrics)}))
                last_metrics = now
    except KeyboardInterrupt:  # normal shutdown: the launcher stops the server's process group
        os._exit(0)
    except BaseException:
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        try:
            out_q.put(("fatal", text))
        finally:
            os._exit(1)
