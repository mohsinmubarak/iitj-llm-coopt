"""CPU unit tests: block manager, rejection sampling, controller decisions.

    pytest -q tests/test_units.py
"""
import random
from collections import Counter
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


def simulate(dl, n, rate_of, steps=600):
    """Drive the bandit with a synthetic GPU whose tokens/s for each γ is rate_of(γ) (+5 % noise)."""
    noise = random.Random(1)
    for _ in range(steps):
        g = dl.choose(n)
        dl.observe(g, n, tokens=rate_of(g) * 0.02 * (1 + 0.05 * noise.uniform(-1, 1)), step_s=0.02)
    return Counter(dl.choose(n) for _ in range(200))


def test_bandit_learns_speculation_at_low_load_and_none_at_saturation():
    """Shapes taken from the 4090 rehearsal: γ=4 best at low load, γ=0 best at saturation."""
    low = {0: 58, 1: 86, 2: 100, 3: 104, 4: 106, 6: 98, 8: 90}          # tokens/s, 1 sequence
    sat = {0: 3400, 1: 2000, 2: 2140, 3: 2100, 4: 2130, 6: 1800, 8: 1500}  # tokens/s, 128 sequences
    dl = DraftLengthController(gamma_max=8)
    picks_low = simulate(dl, 1, low.get)
    picks_sat = simulate(dl, 128, sat.get)
    assert picks_low.most_common(1)[0][0] == 4
    assert picks_sat.most_common(1)[0][0] == 0
    assert dl.best_by_band()[0] == 4 and dl.best_by_band()[7] == 0   # bands learned independently


def test_pressure_caps_the_arms():
    dl = DraftLengthController(gamma_max=8)
    dl.update(snap(), pressure=0)
    assert max(dl.arms()) == 8
    dl.update(snap(), pressure=2)
    assert max(dl.arms()) == 3
    dl.update(snap(), pressure=3)
    assert dl.arms() == [0, 1]
    assert all(dl.choose(4) <= 1 for _ in range(50))


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


# ---- memory-elastic speculation ---------------------------------------------------------------
def test_block_manager_grows_and_shrinks():
    bm = BlockManager(num_blocks=4, block_size=16)
    bm.add_blocks([4, 5, 6])
    assert bm.num_blocks == 7 and bm.num_free == 7
    bm.ensure(1, 16 * 7)                    # uses every block, including the added ones
    with pytest.raises(OutOfBlocks):
        bm.withdraw_free([4, 5, 6])          # still in use
    bm.free(1)
    bm.withdraw_free([4, 5, 6])
    assert bm.num_blocks == 4 and set(bm.free_blocks) == {0, 1, 2, 3}


class FakeElastic:
    def __init__(self):
        self.parked, self.last_change, self.usage = False, 0.0, 0.3

    def usage_if_unparked(self):
        return self.usage


class FakeElasticEngine:
    def __init__(self):
        self.elastic = FakeElastic()


def test_drafter_parks_under_sustained_pressure_and_returns_when_calm():
    kv, eng = KVController(), FakeElasticEngine()
    kv.update_drafter(snap(queue_depth=5), 2, eng)
    assert kv.want_drafter                   # one hot tick is not enough
    kv.update_drafter(snap(queue_depth=5), 2, eng)
    assert not kv.want_drafter               # two consecutive pressured ticks: park
    eng.elastic.parked = True
    for _ in range(3):
        kv.update_drafter(snap(queue_depth=0), 0, eng)
    assert not kv.want_drafter               # needs four calm ticks
    kv.update_drafter(snap(queue_depth=0), 0, eng)
    assert kv.want_drafter


def test_coupled_drafter_decision_uses_the_gamma_verdict():
    dl = DraftLengthController(gamma_max=8)
    simulate(dl, 8, {0: 3000, 1: 2000, 2: 2100, 3: 2000, 4: 1900, 6: 1700, 8: 1500}.get)
    assert dl.best_for(8) == 0
    kv, eng = KVController(), FakeElasticEngine()
    for _ in range(2):                        # no memory pressure at all, but speculation does not pay
        kv.update_drafter(snap(queue_depth=3, running=8), 0, eng, gamma_ctl=dl)
    assert not kv.want_drafter
    eng.elastic.parked = True
    for _ in range(6):                        # calm, yet the γ verdict keeps it parked
        kv.update_drafter(snap(queue_depth=0, running=8), 0, eng, gamma_ctl=dl)
    assert not kv.want_drafter


# ---- prefix caching -------------------------------------------------------------------------
def test_prefix_cache_reuses_full_blocks_and_survives_release():
    bm = BlockManager(num_blocks=16, block_size=4, prefix_caching=True)
    a = list(range(100, 111))                       # 11 tokens: 2 full blocks + 3
    bm.ensure(1, len(a))
    bm.register(1, a, num_computed=len(a))
    assert len(bm.hash_of) == 2                     # only full blocks are hashed
    b = a[:8] + [7, 7, 7, 7, 7]                     # same first 8 tokens, then different
    assert bm.match_prefix(2, b) == 8               # both cached blocks claimed
    assert bm.tables[2] == bm.tables[1][:2] and bm.ref[bm.tables[1][0]] == 2
    bm.free(1)                                      # shared blocks stay with seq 2
    assert bm.ref[bm.tables[2][0]] == 1
    bm.free(2)                                      # now cached-evictable, still counted free
    assert bm.num_free == 16 and len(bm.evictable) == 2
    assert bm.match_prefix(3, a) == 8               # a "preempted" request finds its blocks again
    assert len(bm.evictable) == 0


def test_prefix_cache_leaves_one_token_and_evicts_lru():
    bm = BlockManager(num_blocks=3, block_size=4, prefix_caching=True)
    a = list(range(8))                              # exactly 2 full blocks
    bm.ensure(1, 8)
    bm.register(1, a, 8)
    bm.free(1)
    assert bm.match_prefix(2, a) == 4               # last token must still be computed
    bm.free(2)
    bm.ensure(3, 12)                                # needs all 3 blocks: cached ones are evicted
    assert len(bm.hash_of) == 0 and bm.match_prefix(4, a) == 0


def test_migration_keeps_shared_blocks_shared():
    bm = BlockManager(num_blocks=4, block_size=4, prefix_caching=True)
    bm.add_blocks([4, 5])                           # "parked drafter" tail
    for b in list(bm.free_blocks):                  # force the next allocations into the tail
        if b < 4:
            bm.free_blocks.remove(b)
    toks = list(range(9))
    bm.ensure(1, 8)
    bm.register(1, toks, 8)
    assert bm.match_prefix(2, toks) == 8            # seq 2 shares both tail blocks
    bm.free_blocks.extend([0, 1, 2, 3])
    copies = []
    assert bm.migrate_out_of({4, 5}, lambda s, d: copies.append((s, d))) == 2
    assert len(copies) == 2                         # each shared block copied once
    assert bm.tables[1] == bm.tables[2] and not set(bm.tables[1]) & {4, 5}
    bm.withdraw_free({4, 5})
    assert bm.num_blocks == 4 and bm.match_prefix(3, toks) == 8   # hashes follow the move


def test_burst_parks_drafter_at_once():
    kv, eng = KVController(), FakeElasticEngine()
    kv.update_drafter(snap(queue_depth=40, p_gpu=0.5), 0, eng)
    assert not kv.want_drafter                      # a long queue parks without waiting two ticks
