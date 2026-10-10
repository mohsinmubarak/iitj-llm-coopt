"""CPU unit tests: block manager, rejection sampling, controller decisions.

    pytest -q tests/test_units.py
"""
from dataclasses import dataclass

import pytest
import torch

from engine.block_manager import BlockManager, OutOfBlocks
from engine.co_optimizer import CoOptimizer, DraftLengthController, KVController, expected_tokens
from engine.metrics_engine import MetricsSnapshot
from engine.spec_decode import verify


# ---- block manager ---------------------------------------------------------------------------
def test_alloc_slots_truncate_free():
    bm = BlockManager(num_blocks=8, block_size=16)
    bm.ensure(1, 20)                        # 2 blocks
    assert len(bm.tables[1]) == 2 and bm.num_free == 6
    b0, b1 = bm.tables[1]
    slots = bm.slot_mapping(1, 14, 18).tolist()
    assert slots == [b0 * 16 + 14, b0 * 16 + 15, b1 * 16 + 0, b1 * 16 + 1]
    bm.ensure(1, 33)                        # grows to 3 blocks
    bm.truncate(1, 16)                      # rollback to exactly one block
    assert len(bm.tables[1]) == 1 and bm.num_free == 7
    assert bm.free(1) == 1 and bm.num_free == 8


def test_out_of_blocks():
    bm = BlockManager(num_blocks=2, block_size=16)
    bm.ensure(1, 32)
    assert not bm.can_fit(2, 1)
    with pytest.raises(OutOfBlocks):
        bm.ensure(2, 1)


# ---- speculative decoding ----------------------------------------------------------------------
def onehot_logits(tokens, vocab=10):
    x = torch.full((len(tokens), vocab), -1e4)
    x[torch.arange(len(tokens)), torch.tensor(tokens)] = 0
    return x


def test_greedy_verify_accepts_matching_prefix():
    # target argmax at each position: 3, 5, 7, 9 ; drafts 3, 5, 1
    out = verify(onehot_logits([3, 5, 7, 9]), [3, 5, 1], None, temperature=0)
    assert out == [3, 5, 7]                 # two accepted + target's correction
    out = verify(onehot_logits([3, 5, 7, 9]), [3, 5, 7], None, temperature=0)
    assert out == [3, 5, 7, 9]              # all accepted + bonus


def test_rejection_sampling_is_lossless():
    """First emitted token must follow the target distribution whatever the draft proposes."""
    torch.manual_seed(0)
    V, N = 5, 40000
    p_logits = torch.log(torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05]))
    q = torch.tensor([0.05, 0.05, 0.1, 0.4, 0.4])             # a deliberately bad draft
    counts = torch.zeros(V)
    for _ in range(N):
        d = torch.multinomial(q, 1).item()
        out = verify(torch.stack([p_logits, p_logits]), [d], q.unsqueeze(0), temperature=1.0)
        counts[out[0]] += 1
    emp = counts / N
    assert torch.allclose(emp, torch.softmax(p_logits, -1), atol=0.01), emp


# ---- controllers -------------------------------------------------------------------------------
def snap(**kw):
    base = dict(step=0, t=0.0, alpha_ema=0.8, gamma_mean=4, tokens_per_step=3, p_gpu=0.5, p_cpu=0.0,
                evictions_per_step=0, kv_hit_rate=0, queue_depth=0, running=8, pending_prefill_tokens=0,
                mean_context_len=1000, long_ctx_fraction=0, preemption_rate=0, ttft_p99_recent=0.1,
                tpot_recent=0.02, throughput_tps=500, target_a=0.015, target_b=2e-6, draft_step_s=0.003)
    base.update(kw)
    return MetricsSnapshot(**base)


def test_expected_tokens():
    assert expected_tokens(0.0, 4) == 1.0
    assert expected_tokens(1.0, 4) == 5.0
    assert expected_tokens(0.8, 1) == pytest.approx(1.8)


def test_gamma_grows_with_alpha_and_shrinks_with_pressure():
    dl = DraftLengthController(gamma_max=8)
    s = snap()
    gs = [dl.best_gamma(a, s, 8, 8) for a in (0.2, 0.5, 0.8, 0.95)]
    assert gs == sorted(gs) and gs[-1] > gs[0]
    dl.update(snap(alpha_ema=0.9), pressure=0)
    free_cap = dl.cap
    dl.update(snap(alpha_ema=0.9), pressure=3)
    assert dl.cap < free_cap and dl.base <= 1


def test_gamma_shrinks_when_verification_is_expensive():
    """Large batches make each extra verified token costly -> smaller γ (compute-bound regime)."""
    dl = DraftLengthController(gamma_max=8)
    cheap = dl.best_gamma(0.8, snap(target_b=1e-7), 8, 8)
    costly = dl.best_gamma(0.8, snap(target_b=1e-3), 128, 8)
    assert costly < cheap


@dataclass
class FakeSeq:
    num_computed: int


class FakeTiers:
    bits = 8

    def estimate_bytes(self, n, bits):
        return n * 64 * 1024  # ~64 KB/token after INT8


class FakeProfile:
    pcie_d2h_gbs = pcie_h2d_gbs = 20.0


class FakeEngine:
    kv_tiers = FakeTiers()

    class metrics:
        profile = FakeProfile()


def test_swap_long_sequences_recompute_short_ones():
    kv, s = KVController(), snap(target_a=0.015, target_b=1e-4)  # ~0.1 ms per prefill token
    assert kv.swap_on_preempt(FakeSeq(8000), FakeEngine, s, pressure=0)
    assert not kv.swap_on_preempt(FakeSeq(16), FakeEngine, s, pressure=0)


def test_independent_mode_has_no_shared_pressure():
    co = CoOptimizer(coupled=False)
    co.engine = FakeEngine
    co.tick(snap(p_gpu=0.99, preemption_rate=0.5))
    assert co.pressure is None and co.dl.cap == co.gamma_max     # γ controller never sees memory
    coupled = CoOptimizer()
    coupled.engine = FakeEngine
    coupled.tick(snap(p_gpu=0.99, preemption_rate=0.5))
    assert coupled.pressure == 3 and coupled.dl.cap <= 1
