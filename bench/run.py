"""Benchmark runner shared by every tier (A, B, C).

Replays a workload file against an OpenAI-compatible server (vLLM, or our engine's server),
measures per-request TTFT / TPOT / E2E client-side, samples GPU memory (NVML) and the server's
Prometheus metrics, and writes one JSON result.

    python -m bench.run --workload W1 --request-rate 4 --out results/A2/W1/r4_rep0.json
"""
import argparse
import asyncio
import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
import psutil

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_workload(name, limit=None):
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        reqs = [json.loads(line) for line in f]
    return reqs[:limit] if limit else reqs


def schedule(reqs, request_rate, arrivals, time_scale, seed):
    """Arrival offset (seconds from start) for every request."""
    if arrivals == "file":
        return [r["arrival_s"] * time_scale for r in reqs]
    if math.isinf(request_rate):
        return [0.0] * len(reqs)
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in reqs:
        out.append(t)
        t += rng.expovariate(request_rate)
    return out


@dataclass
class Result:
    id: str
    category: str
    prompt_len: int
    max_tokens: int
    scheduled_s: float
    start_s: float = 0.0
    ttft: float = float("nan")
    e2e: float = float("nan")
    output_tokens: int = 0
    chunk_gaps: list = field(default_factory=list)  # seconds between streamed chunks
    error: str = ""
    phase: str = ""                                 # W4 phase label, "" otherwise


async def send(session, url, model, req, res, temperature):
    payload = {"model": model, "prompt": req["prompt_token_ids"], "max_tokens": req["max_tokens"],
               "temperature": temperature, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    last = None
    try:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    res.output_tokens = chunk["usage"]["completion_tokens"]
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    now = time.perf_counter()
                    if last is None:
                        res.ttft = now - t0
                    else:
                        res.chunk_gaps.append(round(now - last, 5))
                    last = now
        res.e2e = time.perf_counter() - t0
    except Exception as e:  # connection reset = server died (usually OOM); recorded, not raised
        res.error = f"{type(e).__name__}: {e}"[:300]


class MetricsSampler:
    """Background sampling of NVML GPU memory/util and the server's /metrics endpoint."""

    # Server metrics kept as time series. Cumulative counters are included so the dashboard can
    # plot rates (tokens/s, accepted draft tokens/s, preemptions/s) over the run.
    GAUGE_KEYS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage", "gpu_cache_usage",
                  "cpu_cache_usage", "num_preemptions", "generation_tokens_total", "prompt_tokens_total",
                  "spec_decode", "prefix_cache_hits_total", "prefix_cache_queries_total", "coopt:")

    def __init__(self, url, interval=1.0):
        self.url, self.interval = url, interval
        self.series, self.counters_start, self.counters_end = [], {}, {}
        self._stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _scrape(self):
        import urllib.request
        try:
            text = urllib.request.urlopen(f"{self.url}/metrics", timeout=2).read().decode()
        except Exception:
            return {}
        vals = {}
        for line in text.splitlines():
            if line.startswith("#") or "_bucket" in line or "_created" in line:
                continue
            name, _, value = line.rpartition(" ")
            name = name.split("{")[0]
            try:
                vals[name] = vals.get(name, 0.0) + float(value)
            except ValueError:
                pass
        return vals

    def _hw(self):
        row = {"cpu_util": psutil.cpu_percent(), "ram_used_gb": round(psutil.virtual_memory().used / 2**30, 3)}
        if self._nvml:
            nv, h = self._nvml, self._h
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            row["gpu_mem_used_gb"] = round(mem.used / 2**30, 3)
            row["gpu_util"] = nv.nvmlDeviceGetUtilizationRates(h).gpu
            row["gpu_power_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1)
            row["gpu_temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            # PCIe traffic (KB/s): the cost signal for KV offload between GPU and CPU tiers
            row["pcie_tx_mb_s"] = round(nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_TX_BYTES) / 1024, 1)
            row["pcie_rx_mb_s"] = round(nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_RX_BYTES) / 1024, 1)
        return row

    def _loop(self, t0):
        while not self._stop.is_set():
            row = {"t": round(time.perf_counter() - t0, 2)}
            try:
                row.update(self._hw())
            except Exception:
                pass
            m = self._scrape()
            row.update({k: v for k, v in m.items() if any(g in k for g in self.GAUGE_KEYS)})
            self.series.append(row)
            self._stop.wait(self.interval)

    def start(self, t0):
        self.counters_start = self._scrape()
        self._thread = threading.Thread(target=self._loop, args=(t0,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.counters_end = self._scrape()

    def counter_deltas(self):
        return {k: v - self.counters_start.get(k, 0.0) for k, v in self.counters_end.items()
                if k.endswith(("_total", "_sum", "_count"))}


def pct(xs, q):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.percentile(xs, q)) if xs else float("nan")


def summarize(results, duration, slo):
    ok = [r for r in results if not r.error]
    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_len for r in ok)
    ttft = [r.ttft for r in ok]
    tpot = [(r.e2e - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    gaps = [g for r in ok for g in r.chunk_gaps]
    e2e = [r.e2e for r in ok]

    def meets_slo(r):
        ttft_slo = slo["ttft_long"] if r.category == "long" else slo["ttft_short"]
        tp = (r.e2e - r.ttft) / max(r.output_tokens - 1, 1)
        return r.ttft <= ttft_slo and tp <= slo["tpot"]

    s = {
        "num_requests": len(results), "completed": len(ok), "failed": len(results) - len(ok),
        "duration_s": duration,
        "output_throughput_tok_s": out_tok / duration,
        "total_throughput_tok_s": (in_tok + out_tok) / duration,
        "request_throughput": len(ok) / duration,
        "goodput_req_s": sum(meets_slo(r) for r in ok) / duration,
        "slo": slo,
        "oom_suspected": any("out of memory" in r.error.lower() or "Disconnected" in r.error
                             or "ClientConnector" in r.error for r in results),
    }
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("itl_chunk", gaps), ("e2e", e2e)):
        s[f"{name}_mean"] = float(np.mean(xs)) if xs else float("nan")
        for q in (50, 90, 99):
            s[f"{name}_p{q}"] = pct(xs, q)
    for cat in ("short", "long"):
        sub = [r.ttft for r in ok if r.category == cat]
        if sub:
            s[f"ttft_p99_{cat}"] = pct(sub, 99)
    # Phase-shift workloads (W4): per-phase SLO attainment and latency. Failed requests count
    # as SLO misses, so a phase cannot look good by dropping its hardest requests.
    for phase in sorted({r.phase for r in results if r.phase}):
        sub = [r for r in results if r.phase == phase]
        okp = [r for r in sub if not r.error]
        tp = [(r.e2e - r.ttft) / (r.output_tokens - 1) for r in okp if r.output_tokens > 1]
        s[f"{phase}_requests"] = len(sub)
        s[f"{phase}_slo_attainment"] = sum(meets_slo(r) for r in okp) / len(sub)
        s[f"{phase}_ttft_p50"] = pct([r.ttft for r in okp], 50)
        s[f"{phase}_ttft_p99"] = pct([r.ttft for r in okp], 99)
        s[f"{phase}_tpot_p50"] = pct(tp, 50)
        s[f"{phase}_tpot_p99"] = pct(tp, 99)
    return s


async def benchmark(url, model, reqs, *, request_rate=math.inf, arrivals="rate", time_scale=1.0,
                    max_concurrency=None, seed=0, temperature=0.0, slo=None):
    slo = slo or {"ttft_short": 1.0, "ttft_long": 10.0, "tpot": 0.10}
    offsets = schedule(reqs, request_rate, arrivals, time_scale, seed)
    results = [Result(r["id"], r["category"], r["prompt_len"], r["max_tokens"], o, phase=r.get("phase", ""))
               for r, o in zip(reqs, offsets)]
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    sampler = MetricsSampler(url)
    timeout = aiohttp.ClientTimeout(total=6 * 3600)
    conn = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        t0 = time.perf_counter()
        sampler.start(t0)

        async def one(req, res):
            await asyncio.sleep(max(0.0, res.scheduled_s - (time.perf_counter() - t0)))
            if sem:
                async with sem:
                    res.start_s = time.perf_counter() - t0
                    await send(session, url, model, req, res, temperature)
            else:
                res.start_s = time.perf_counter() - t0
                await send(session, url, model, req, res, temperature)

        await asyncio.gather(*(one(q, r) for q, r in zip(reqs, results)))
        duration = time.perf_counter() - t0
        sampler.stop()

    return {
        "summary": summarize(results, duration, slo),
        "server_counters": sampler.counter_deltas(),
        "timeseries": sampler.series,
        "requests": [asdict(r) for r in results],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--request-rate", type=float, default=math.inf)
    ap.add_argument("--arrivals", choices=["rate", "file"], default="rate")
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--max-concurrency", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reqs = load_workload(args.workload, args.limit)
    res = asyncio.run(benchmark(args.url, args.model, reqs, request_rate=args.request_rate,
                                arrivals=args.arrivals, time_scale=args.time_scale,
                                max_concurrency=args.max_concurrency, seed=args.seed))
    res["args"] = vars(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    s = res["summary"]
    print(f"{args.workload}: {s['completed']}/{s['num_requests']} ok | "
          f"out {s['output_throughput_tok_s']:.1f} tok/s | TTFT p99 {s['ttft_p99']:.3f}s | "
          f"TPOT p99 {s['tpot_p99'] * 1e3:.1f}ms | goodput {s['goodput_req_s']:.2f} req/s")


if __name__ == "__main__":
    main()
