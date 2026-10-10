"""GPU -> CPU KV tiering with quantize-on-evict.

When the scheduler preempts a sequence, the default (vLLM V1, our C0) is to drop its KV and
recompute prompt + output later. With tiering, the target model's KV blocks are instead
quantized (INT8, or INT4 under pressure; per-token-per-head absmax scales) and copied to pinned
CPU memory, then dequantized back into fresh GPU blocks when the sequence is re-admitted.
That trades PCIe traffic for prefill recompute; the co-optimizer decides per preemption which
is cheaper, and how much CPU tier to keep.

The draft model's KV is not tiered: it is small and recomputed lazily (draft_computed = 0),
which keeps rollback logic in one place.

Quantization is lossy, so restored sequences can differ from an uninterrupted run; experiment
Q1/S5 measures exactly this (perplexity with KV quantization none / INT8 / INT4).
"""
from dataclasses import dataclass

import torch


def quantize(x: torch.Tensor, bits: int):
    """x [..., D] -> (q, scale [...]) with symmetric absmax per last-dim vector."""
    if bits == 16:
        return x.clone(), None
    qmax = 127 if bits == 8 else 7
    scale = x.abs().amax(-1, keepdim=True).float().clamp_min(1e-8) / qmax
    q = torch.round(x.float() / scale).clamp(-qmax, qmax).to(torch.int8)
    if bits == 4:  # pack two signed nibbles per byte
        q = q + 8
        q = (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)
    return q, scale.squeeze(-1).to(torch.float16)


def dequantize(q: torch.Tensor, scale, bits: int, dtype=torch.bfloat16):
    if bits == 16:
        return q
    if bits == 4:
        lo = (q & 0x0F).to(torch.int8) - 8
        hi = (q >> 4).to(torch.int8) - 8
        q = torch.stack((lo, hi), dim=-1).flatten(-2)
    return (q.float() * scale.float().unsqueeze(-1)).to(dtype)


@dataclass
class SwappedKV:
    k: list              # per layer: quantized [nb, bs, Hkv, D(/2)] on pinned CPU
    v: list
    k_scale: list
    v_scale: list
    num_tokens: int
    bits: int
    nbytes: int


class KVTierManager:
    def __init__(self, engine, cpu_gb: float = 32.0, bits: int = 8):
        self.e = engine
        self.capacity = int(cpu_gb * 2**30)
        self.bits = bits                 # current quantization level, set by the co-optimizer
        self.store: dict[int, SwappedKV] = {}
        self.used = 0
        self.stream = torch.cuda.Stream()
        self.restores = self.recomputes = 0
        self.bytes_out = self.bytes_in = 0
        engine.kv_tiers = self

    @property
    def cpu_usage(self):
        return self.used / self.capacity

    @property
    def hit_rate(self):
        total = self.restores + self.recomputes
        return self.restores / total if total else 0.0

    def estimate_bytes(self, num_tokens: int, bits: int) -> int:
        cfg = self.e.target.cfg
        per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * bits / 8
        return int(per_token * num_tokens * (1 + (2 / cfg.head_dim if bits < 16 else 0)))

    def has(self, seq_id):
        return seq_id in self.store

    def swap_out(self, seq) -> bool:
        """Copy seq's target KV (num_computed tokens) to the CPU tier. False if it does not fit."""
        n = seq.num_computed
        if n == 0:
            return False
        need = self.estimate_bytes(n, self.bits)
        if self.used + need > self.capacity:
            return False
        e, bm = self.e, self.e.bm
        blocks = torch.tensor(bm.tables[seq.seq_id][: bm.blocks_for(n)], device="cuda")
        out = SwappedKV([], [], [], [], n, self.bits, need)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for kc, vc in zip(e.target.k_cache, e.target.v_cache):
                for cache, qs, ss in ((kc, out.k, out.k_scale), (vc, out.v, out.v_scale)):
                    q, s = quantize(cache[blocks], self.bits)
                    qs.append(_to_pinned(q))
                    ss.append(_to_pinned(s) if s is not None else None)
        self.stream.synchronize()
        self.store[seq.seq_id] = out
        self.used += need
        self.bytes_out += need
        e.counters["kv_evictions_total"] = e.counters.get("kv_evictions_total", 0) + 1
        return True

    def swap_in(self, seq) -> None:
        """Restore seq's KV into newly allocated GPU blocks (caller has checked capacity)."""
        sw = self.store.pop(seq.seq_id)
        e, bm = self.e, self.e.bm
        bm.ensure(seq.seq_id, sw.num_tokens)
        blocks = torch.tensor(bm.tables[seq.seq_id][: bm.blocks_for(sw.num_tokens)], device="cuda")
        for li, (kc, vc) in enumerate(zip(e.target.k_cache, e.target.v_cache)):
            for cache, qs, ss in ((kc, sw.k, sw.k_scale), (vc, sw.v, sw.v_scale)):
                q = qs[li].to("cuda", non_blocking=True)
                s = ss[li].to("cuda", non_blocking=True) if ss[li] is not None else None
                cache[blocks] = dequantize(q, s, sw.bits, cache.dtype)
        self.used -= sw.nbytes
        self.bytes_in += sw.nbytes
        self.restores += 1
        e.counters["kv_restores_total"] = e.counters.get("kv_restores_total", 0) + 1

    def drop(self, seq_id):
        sw = self.store.pop(seq_id, None)
        if sw:
            self.used -= sw.nbytes

    def note_recompute(self):
        self.recomputes += 1


def _to_pinned(t: torch.Tensor) -> torch.Tensor:
    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    host.copy_(t, non_blocking=True)
    return host
