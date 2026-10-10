"""The co-optimizer: three controllers driven by one MetricsSnapshot per tick.

  DraftLengthController  γ per sequence from a cost model fitted online:
                         maximise E[tokens/step](α, γ) / step_time(γ, batch), per-sequence α
                         with shrinkage to the global EMA, capped by system pressure.
  KVController           swap-vs-recompute per preemption (PCIe round trip vs prefill cost),
                         INT8/INT4 level from CPU-tier pressure.
  SchedulerController    AIMD batch cap, adaptive prefill chunk, pressure-aware admission
                         (defer long prompts under memory pressure), aged shortest-first
                         ordering, victim = most KV freed per unit of sunk work.

Coupling (the thesis claim) is the shared `pressure` level and cross-terms: high pressure
lowers the γ cap (each draft token reserves KV), defers long admissions and makes swapping
more attractive than recomputing on a busy GPU; low pressure restores swapped sequences first.

`coupled=False` (experiment C6) keeps the same three controllers but removes the shared view:
each reads only its own signals and refreshes on its own clock (γ every tick, KV every 2,
scheduler every 4), as three independently built systems would.
"""
from dataclasses import dataclass, field

import numpy as np

from engine.policy import StaticPolicy

ALPHA_BINS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def expected_tokens(alpha: float, g: int) -> float:
    """E[tokens per verification] with γ drafts and i.i.d. acceptance α (Leviathan et al.)."""
    if g == 0:
        return 1.0
    if alpha >= 0.999:
        return g + 1.0
    return (1 - alpha ** (g + 1)) / (1 - alpha)


@dataclass
class DraftLengthController:
    gamma_max: int = 8
    prior_weight: float = 8.0      # pseudo draft tokens of the global α in each sequence's estimate
    hysteresis: float = 0.03       # a larger γ must be ≥3% faster to be chosen
    table: dict = field(default_factory=dict)
    cap: int = 8
    base: int = 4

    def _cost(self, snap, g, n):
        return snap.target_a + snap.target_b * n * (g + 1) + g * snap.draft_step_s

    def best_gamma(self, alpha, snap, n, cap):
        best, best_rate = 0, expected_tokens(alpha, 0) / self._cost(snap, 0, n)
        for g in range(1, cap + 1):
            rate = expected_tokens(alpha, g) / self._cost(snap, g, n)
            if rate > best_rate * (1 + self.hysteresis):
                best, best_rate = g, rate
        return best

    def update(self, snap, pressure):
        cap = self.gamma_max
        if pressure is not None:  # coupled: memory/queue pressure caps speculation
            cap = {0: cap, 1: min(cap, 6), 2: min(cap, 3), 3: min(cap, 1)}[pressure]
        self.cap = cap
        n = max(1, snap.running)
        self.table = {a: self.best_gamma(a, snap, n, cap) for a in ALPHA_BINS}
        self.base = self.table[ALPHA_BINS[np.abs(ALPHA_BINS - snap.alpha_ema).argmin()]]

    def gamma_for(self, seq, alpha_global):
        if not self.table:
            return self.base
        a = (seq.accepted_total + self.prior_weight * alpha_global) / (seq.drafted_total + self.prior_weight)
        return self.table[ALPHA_BINS[np.abs(ALPHA_BINS - a).argmin()]]


@dataclass
class KVController:
    cpu_pressure_int4: float = 0.75   # switch new evictions to INT4 above this CPU-tier occupancy
    swap_fixed_s: float = 0.003       # per-swap launch/sync overhead (2 x layers of copy + quantize)

    def update(self, snap, pressure, engine):
        if engine.kv_tiers:
            engine.kv_tiers.bits = 4 if snap.p_cpu > self.cpu_pressure_int4 else 8

    def swap_on_preempt(self, seq, engine, snap, pressure) -> bool:
        tiers, prof = engine.kv_tiers, engine.metrics.profile
        if not tiers or snap is None:
            return False
        nbytes = tiers.estimate_bytes(seq.num_computed, tiers.bits)
        swap_s = self.swap_fixed_s + nbytes / (prof.pcie_d2h_gbs * 1e9) + nbytes / (prof.pcie_h2d_gbs * 1e9)
        # a recompute rides along in a later batched prefill: its marginal cost is per-token only
        recompute_s = snap.target_b * seq.num_computed
        if pressure is not None and snap.running:
            # coupled: on a busy GPU recompute also delays every co-scheduled request
            recompute_s *= 1 + snap.queue_depth / snap.running
        return swap_s < recompute_s


@dataclass
class SchedulerController:
    ttft_slo: float = 2.0
    tpot_slo: float = 0.10
    min_seqs: int = 8
    max_seqs: int = 256
    chunk_small: int = 2048
    chunk_default: int = 8192
    chunk_large: int = 16384
    long_ctx: int = 4096
    aging_s: float = 5.0             # a request waiting this long is admitted regardless of size

    def update(self, snap, pressure, policy):
        if snap.preemption_rate > 0.05 or pressure == 3:
            policy.max_num_seqs = max(self.min_seqs, int(policy.max_num_seqs * 0.75))
        elif snap.queue_depth > 0 and (pressure is None or pressure <= 1):
            policy.max_num_seqs = min(self.max_seqs, policy.max_num_seqs + 8)
        if snap.ttft_p99_recent > self.ttft_slo and snap.queue_depth > 0:
            policy.max_batched_tokens = self.chunk_large
        elif snap.tpot_recent > self.tpot_slo:
            policy.max_batched_tokens = self.chunk_small
        else:
            policy.max_batched_tokens = self.chunk_default

    def admit(self, seq, engine, pressure) -> bool:
        if pressure is None or pressure < 2 or not engine.scheduler.running:
            return True
        waited = engine.now() - seq.arrival
        return len(seq.tokens) <= self.long_ctx or waited > self.aging_s

    def pick_next(self, waiting, engine, pressure) -> int:
        now, tiers = engine.now(), engine.kv_tiers
        best, best_score = 0, -1.0
        for i, s in enumerate(list(waiting)[:32]):
            score = (now - s.arrival + 0.1) / (1 + s.remaining / 2048)
            if pressure is not None and pressure <= 1 and tiers and tiers.has(s.seq_id):
                score *= 4  # restoring a swapped sequence is cheaper than a fresh prefill
            if score > best_score:
                best, best_score = i, score
        return best

    def preempt_victim(self, running):
        def freed_per_sunk_work(s):
            progress = s.num_generated / max(1, s.max_tokens)
            return len(s.tokens) * (1.0 - progress)
        return max(running, key=freed_per_sunk_work)


@dataclass
class CoOptimizer(StaticPolicy):
    enable_gamma: bool = True
    enable_kv: bool = True
    enable_sched: bool = True
    coupled: bool = True
    gamma_max: int = 8
    name = "coopt"

    def __post_init__(self):
        self.dl = DraftLengthController(gamma_max=self.gamma_max)
        self.kv = KVController()
        # The engine sizes its activation headroom for the configured max_batched_tokens, so the
        # controller may shrink prefill chunks under decode pressure but never grow them past it.
        self.sc = SchedulerController(chunk_default=self.max_batched_tokens, chunk_large=self.max_batched_tokens,
                                      chunk_small=max(256, self.max_batched_tokens // 4))
        self.kv_swap = self.enable_kv
        self.engine = None
        self.snap = None
        self.pressure = 0
        self.ticks = 0
        self.stale = {}  # independent mode: each controller's own last snapshot

    @staticmethod
    def pressure_level(snap) -> int:
        if snap.p_gpu >= 0.97 or snap.preemption_rate > 0.1:
            return 3
        if snap.p_gpu >= 0.90 or (snap.queue_depth > 0 and snap.p_gpu >= 0.85):
            return 2
        if snap.p_gpu >= 0.75:
            return 1
        return 0

    def tick(self, snap):
        self.ticks += 1
        self.snap = snap
        if self.coupled:
            self.pressure = self.pressure_level(snap)
            p = self.pressure
            if self.enable_gamma:
                self.dl.update(snap, p)
            if self.enable_kv:
                self.kv.update(snap, p, self.engine)
            if self.enable_sched:
                self.sc.update(snap, p, self)
            return
        # independent controllers: no shared pressure, staggered refresh, own snapshot copies
        self.pressure = None
        if self.enable_gamma:
            self.stale["gamma"] = snap
            self.dl.update(snap, None)
        if self.enable_kv and self.ticks % 2 == 0:
            self.stale["kv"] = snap
            self.kv.update(snap, None, self.engine)
        if self.enable_sched and self.ticks % 4 == 0:
            self.stale["sched"] = snap
            self.sc.update(snap, None, self)

    # ---- hooks called by the engine ------------------------------------------------------
    def gamma_for(self, seq) -> int:
        if not self.enable_gamma or self.snap is None:
            return self.gamma
        return self.dl.gamma_for(seq, self.snap.alpha_ema)

    def swap_on_preempt(self, seq, engine) -> bool:
        if not self.enable_kv:
            return False
        snap = self.snap if self.coupled else self.stale.get("kv", self.snap)
        return self.kv.swap_on_preempt(seq, engine, snap, self.pressure)

    def admit(self, seq, engine) -> bool:
        return self.sc.admit(seq, engine, self.pressure) if self.enable_sched else True

    def pick_next(self, waiting, engine) -> int:
        return self.sc.pick_next(waiting, engine, self.pressure) if self.enable_sched else 0

    def preempt_victim(self, running):
        return self.sc.preempt_victim(running) if self.enable_sched else running[-1]

    def state(self) -> dict:
        s = super().state()
        s.update({"gamma_base": self.dl.base if self.enable_gamma else self.gamma,
                  "gamma_cap": self.dl.cap if self.enable_gamma else self.gamma,
                  "pressure": -1 if self.pressure is None else self.pressure,
                  "kv_bits": self.engine.kv_tiers.bits if self.engine and self.engine.kv_tiers else 16})
        return s
