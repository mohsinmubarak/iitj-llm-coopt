"""Memory-elastic speculation: the drafter lives inside the target's KV memory and can be parked.

Measured on the RTX 4090 rehearsal: the 1B drafter's weights (2.5 GB) plus its KV pool take about
half the memory the 8B target could use for KV. γ = 0 stops drafting but does not return that
memory, so under bursts the drafted engine queued far more than the same engine without a drafter
(TTFT p50 19 s vs 5 s), while speculation was clearly better at light load (TPOT -38 %).

Layout: all target KV is one block-major buffer  [blocks, layers, K/V, block_size, kv_heads, dim],
so block i of every layer is one contiguous chunk and each layer's cache is a strided view
(FlashAttention accepts it; tests/strided_kv_check.py). The buffer is
    [0, base)              target KV blocks, always usable
    [base, base + tail)    drafter: weights, then its own KV pool (contiguous per layer)
    [base + tail]          scratch block for CUDA-graph padding writes
Parking the drafter hands the tail blocks to the target block manager (capacity grows by about
`tail` blocks). Unparking migrates any target block still in the tail to a free base block (one
contiguous copy per block covers all layers), withdraws the tail from the pool and copies the
drafter's weights back from a pinned CPU copy (2.5 GB over PCIe, about 0.1 s). The draft KV is
then empty and every sequence's draft catches up through the bounded catch-up path. Draft CUDA
graphs stay valid: weights return to the same addresses.

The decision when to park belongs to the co-optimizer; the engine only executes it.
"""
import time

import torch

ALIGN = 256


def _align(n):
    return -(-n // ALIGN) * ALIGN


class ElasticKV:
    def __init__(self, engine, buffer_bytes: int, block_size: int):
        t, d = engine.target, engine.draft
        tc, dc = t.cfg, d.cfg
        self.e, self.block_size = engine, block_size
        dt = t.dtype
        self.block_bytes = tc.num_hidden_layers * 2 * block_size * tc.num_key_value_heads * tc.head_dim * dt.itemsize
        d_block_bytes = dc.num_hidden_layers * 2 * block_size * dc.num_key_value_heads * dc.head_dim * dt.itemsize

        # 1. Drafter weights to pinned CPU (kept for unparking), freeing their GPU memory now.
        self.slots = d.weight_slots()
        self.cpu = []
        w_bytes = 0
        for cont, key in self.slots:
            w = cont[key]
            host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
            host.copy_(w)
            self.cpu.append(host)
            w_bytes += _align(w.numel() * w.element_size())
            cont[key] = None
        d.lm_head = d.embed = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 2. Size the pools: total = base + tail + 1 blocks, with the draft KV pool holding as many
        #    tokens as the base target pool (as in the non-elastic engine).
        total = int(buffer_bytes // self.block_bytes)
        per_layer_d = lambda nd: _align((nd + 1) * block_size * dc.num_key_value_heads * dc.head_dim * dt.itemsize)
        base = int((total - 2 - w_bytes / self.block_bytes) / (1 + d_block_bytes / self.block_bytes))
        while True:
            tail_bytes = w_bytes + 2 * dc.num_hidden_layers * per_layer_d(base)
            tail = -(-tail_bytes // self.block_bytes)
            if base + tail + 1 <= total:
                break
            base -= 1
        assert base > 0, "no room for an elastic KV buffer"
        self.base, self.tail, self.scratch = base, tail, base + tail
        self.draft_blocks = base

        # 3. One block-major buffer; the target uses strided per-layer views of it.
        shape = (base + tail + 1, tc.num_hidden_layers, 2, block_size, tc.num_key_value_heads, tc.head_dim)
        self.buf = torch.empty(shape, dtype=dt, device="cuda")
        t.use_kv_views([self.buf[:, i, 0] for i in range(tc.num_hidden_layers)],
                       [self.buf[:, i, 1] for i in range(tc.num_hidden_layers)], block_size)

        # 4. Drafter weights and drafter KV inside the tail region.
        region = self.buf[base:base + tail].reshape(-1).view(torch.uint8)
        off = 0
        self.weight_views = []
        for (cont, key), host in zip(self.slots, self.cpu):
            n = host.numel() * host.element_size()
            view = region[off:off + n].view(host.dtype).view(host.shape)
            view.copy_(host, non_blocking=True)
            cont[key] = view
            self.weight_views.append(view)
            off += _align(n)
        d.retie()
        kshape = (base + 1, block_size, dc.num_key_value_heads, dc.head_dim)
        nper = per_layer_d(base)
        ks, vs = [], []
        for _ in range(dc.num_hidden_layers):
            for lst in (ks, vs):
                n = (base + 1) * block_size * dc.num_key_value_heads * dc.head_dim * dt.itemsize
                lst.append(region[off:off + n].view(dt).view(kshape))
                off += nper
        assert off <= region.numel(), "drafter does not fit its tail region"
        d.use_kv_views(ks, vs, block_size)
        torch.cuda.synchronize()

        self.tail_ids = set(range(base, base + tail))
        self.parked = False
        self.last_change = time.perf_counter()
        print(f"[elastic] target blocks base {base} (+{tail} when the drafter is parked) | "
              f"drafter weights {w_bytes / 2**30:.2f} GiB + draft KV {2 * dc.num_hidden_layers * nper / 2**30:.2f} GiB "
              f"in the tail | block {self.block_bytes / 2**20:.1f} MiB", flush=True)

    # ---- park / unpark ------------------------------------------------------------------------
    def park(self):
        """Stop speculation and give the drafter's memory to the target KV pool."""
        if self.parked:
            return True
        e = self.e
        for s in list(e.scheduler.running) + list(e.scheduler.waiting):
            e.dbm.free(s.seq_id)
            s.draft_computed = 0
        e.bm.add_blocks(sorted(self.tail_ids))
        self.parked = True
        self.last_change = time.perf_counter()
        e.counters["drafter_parks_total"] += 1
        return True

    def unpark(self):
        """Take the tail back (migrating target blocks out of it) and restore the drafter."""
        if not self.parked:
            return True
        e, bm = self.e, self.e.bm
        used_tail = [(sid, i, b) for sid, table in bm.tables.items() for i, b in enumerate(table)
                     if b in self.tail_ids]
        free_outside = sum(1 for b in bm.free_blocks if b not in self.tail_ids)
        if free_outside < len(used_tail):
            return False  # not enough room outside the tail yet: stay parked
        for sid, i, src in used_tail:
            dst = bm.take_free_excluding(self.tail_ids)
            self.buf[dst].copy_(self.buf[src])  # one contiguous copy moves the block in every layer
            bm.tables[sid][i] = dst
            bm.free_blocks.append(src)
        bm.withdraw_free(self.tail_ids)
        for view, host in zip(self.weight_views, self.cpu):
            view.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        self.parked = False
        self.last_change = time.perf_counter()
        e.counters["drafter_unparks_total"] += 1
        e.counters["drafter_migrated_blocks_total"] += len(used_tail)
        return True

    def usage_if_unparked(self) -> float:
        """Fraction of the base pool the current sequences would occupy after unparking."""
        bm = self.e.bm
        used = bm.num_blocks - bm.num_free
        return used / self.base
