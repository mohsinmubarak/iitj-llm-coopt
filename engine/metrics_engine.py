"""Hardware profiler + live metrics: turns engine state into one MetricsSnapshot per tick.

Every controller reads the *same* snapshot, which is the point of the co-optimizer: γ, KV
tiering and scheduling decisions are made from one coherent view of the system instead of
three separately sampled, possibly stale ones (experiment C6 removes exactly this).

Signals are deltas/EMAs over the last tick window; per-step cost models (target and draft
forward time as a linear function of batch tokens) are fitted online so the γ controller can
price speculation on the hardware it is actually running on.
"""
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import psutil
import torch


@dataclass
class HardwareProfile:
    gpu_name: str
    vram_gb: float
    hbm_bw_gbs: float      # measured device-to-device copy bandwidth
    pcie_h2d_gbs: float    # measured pinned host -> device bandwidth (KV swap-in cost)
    pcie_d2h_gbs: float    # measured device -> pinned host bandwidth (KV swap-out cost)
    cpu_cores: int
    ram_gb: float

    @classmethod
    def measure(cls, mb=256):
        n = mb * 2**20
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        dev2 = torch.empty_like(dev)
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)

        def bw(fn, reps=5):
            fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            return n * reps / (time.perf_counter() - t) / 1e9

        p = torch.cuda.get_device_properties(0)
        prof = cls(gpu_name=p.name, vram_gb=p.total_memory / 2**30,
                   hbm_bw_gbs=2 * bw(lambda: dev2.copy_(dev)),  # read + write
                   pcie_h2d_gbs=bw(lambda: dev.copy_(host, non_blocking=True)),
                   pcie_d2h_gbs=bw(lambda: host.copy_(dev, non_blocking=True)),
                   cpu_cores=psutil.cpu_count(), ram_gb=psutil.virtual_memory().total / 2**30)
        del dev, dev2, host
        torch.cuda.empty_cache()
        return prof


@dataclass
class MetricsSnapshot:
    step: int
    t: float
    # speculative decoding
    alpha_ema: float           # draft-token acceptance rate (EMA)
    gamma_mean: float          # mean γ actually used in the window
    tokens_per_step: float     # generated tokens per engine step (window)
    # memory
    p_gpu: float               # GPU KV-pool occupancy [0, 1]
    p_cpu: float               # CPU KV-tier occupancy [0, 1]
    evictions_per_step: float  # sequences swapped out per step (window)
    kv_hit_rate: float         # resumed sequences restored from the CPU tier instead of recomputed
    # scheduling / load
    queue_depth: int
    running: int
    pending_prefill_tokens: int
    mean_context_len: float
    long_ctx_fraction: float   # running sequences with context > 4096
    preemption_rate: float     # preemptions per step (window)
    ttft_p99_recent: float     # seconds, last 128 first tokens
    tpot_recent: float         # mean step time of decode-bearing steps (s)
    throughput_tps: float      # generated tokens / s (window)
    # cost model (fitted online): step_time ≈ a + b * batch_tokens
    target_a: float
    target_b: float
    draft_step_s: float        # mean draft forward time per draft step


class MetricsEngine:
    def __init__(self, engine, ema=0.8, trace_path=None):
        self.e, self.ema = engine, ema
        self.profile = HardwareProfile.measure()
        self.alpha = 0.7  # prior until the first speculative steps arrive
        self.prev_counters = dict(engine.counters)
        self.prev_t, self.prev_step = time.perf_counter(), 0
        self.step_samples = deque(maxlen=512)   # (batch_tokens, target_time_s)
        self.draft_samples = deque(maxlen=512)  # draft forward times
        self.gamma_samples = deque(maxlen=512)
        self.latest: MetricsSnapshot | None = None
        self.trace = open(trace_path, "a") if trace_path else None
        engine.metrics = self

    # Engine calls these every step (cheap appends only).
    def record_step(self, batch_tokens, target_s, draft_s_list, gammas):
        self.step_samples.append((batch_tokens, target_s))
        self.draft_samples.extend(draft_s_list)
        self.gamma_samples.extend(gammas)

    def _fit_cost(self):
        if len(self.step_samples) < 8:
            return 0.02, 1e-6
        x, y = np.array(self.step_samples).T
        if np.ptp(x) < 1:
            return float(y.mean()), 0.0
        b, a = np.polyfit(x, y, 1)
        return float(max(a, 1e-4)), float(max(b, 0.0))

    def snapshot(self) -> MetricsSnapshot:
        e, c = self.e, self.e.counters
        now, steps = time.perf_counter(), max(1, e.step_count - self.prev_step)
        d = {k: c[k] - self.prev_counters.get(k, 0) for k in c}
        dt = max(now - self.prev_t, 1e-6)
        if d["spec_decode_num_draft_tokens_total"]:
            a = d["spec_decode_num_accepted_tokens_total"] / d["spec_decode_num_draft_tokens_total"]
            self.alpha = self.ema * self.alpha + (1 - self.ema) * a
        run, wait = e.scheduler.running, e.scheduler.waiting
        ctx = [len(s.tokens) for s in run]
        ttft = list(e.recent_ttft)
        tier = getattr(e, "kv_tiers", None)
        a_fit, b_fit = self._fit_cost()
        snap = MetricsSnapshot(
            step=e.step_count, t=now,
            alpha_ema=self.alpha,
            gamma_mean=float(np.mean(self.gamma_samples)) if self.gamma_samples else 0.0,
            tokens_per_step=d["generation_tokens_total"] / steps,
            p_gpu=e.bm.usage,
            p_cpu=tier.cpu_usage if tier else 0.0,
            evictions_per_step=(d.get("kv_evictions_total", 0)) / steps,
            kv_hit_rate=tier.hit_rate if tier else 0.0,
            queue_depth=len(wait), running=len(run),
            pending_prefill_tokens=sum(s.remaining for s in wait) + sum(s.remaining - 1 for s in run),
            mean_context_len=float(np.mean(ctx)) if ctx else 0.0,
            long_ctx_fraction=float(np.mean([x > 4096 for x in ctx])) if ctx else 0.0,
            preemption_rate=d["num_preemptions_total"] / steps,
            ttft_p99_recent=float(np.percentile(ttft, 99)) if ttft else 0.0,
            tpot_recent=float(np.mean([t for _, t in self.step_samples])) if self.step_samples else 0.0,
            throughput_tps=d["generation_tokens_total"] / dt,
            target_a=a_fit, target_b=b_fit,
            draft_step_s=float(np.mean(self.draft_samples)) if self.draft_samples else 0.0,
        )
        self.prev_counters, self.prev_t, self.prev_step = dict(c), now, e.step_count
        self.gamma_samples.clear()
        self.latest = snap
        e.extra_metrics = {k: v for k, v in asdict(snap).items() if k not in ("step", "t")}
        if self.trace:
            self.trace.write(json.dumps({**asdict(snap), **e.policy.state()}) + "\n")
            self.trace.flush()
        return snap
