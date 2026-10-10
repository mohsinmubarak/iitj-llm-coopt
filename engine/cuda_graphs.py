"""CUDA graphs for decode-shaped steps.

Eager PyTorch launches ~20 small kernels per layer from Python; for a decode step that CPU work
(~10 ms for the 1B draft, regardless of batch size) dwarfs the GPU work. A captured graph replays
the whole forward with one launch.

A graph needs fixed shapes, so decode-shaped batches use a padded layout: sequence i owns token
rows [i*S, (i+1)*S); its n_i real tokens come first and the rest is padding. Every sequence gets
q_len = S and seq_len = cached + S. With bottom-right causal alignment, real query j still sees
exactly keys [0, cached + j], so padding never affects real outputs. Padding tokens write their
K/V into a scratch block that the block manager never hands out, and read garbage beyond a
sequence's allocated blocks only through padding queries, whose outputs are discarded.
Batches are padded to the next captured size B with dummy sequences (seq_len S, scratch slots).

Families by stride S: target decode (S=1) and verification (S = γ+1 buckets), draft
catch-up (S=2) and draft decode (S=1). Steps that contain prefill chunks stay eager (as in vLLM).
"""
import numpy as np
import torch

from engine.model import Batch, Llama


class GraphRunner:
    def __init__(self, model: Llama, max_model_len: int, strides, batch_sizes, scratch_block: int):
        self.m = model
        self.bs_list = sorted(batch_sizes)
        self.strides = sorted(strides)
        self.block_size = model.block_size
        self.scratch_slot = scratch_block * self.block_size
        self.max_len = max_model_len
        self.graphs = {}   # (S, B) -> (graph, inputs dict, logits)
        self.pool = torch.cuda.graph_pool_handle()

    def capture(self, max_tokens_per_graph=4096):
        width = -(-(self.max_len + max(self.strides)) // self.block_size)
        dev = "cuda"
        stream = torch.cuda.Stream()
        max_rows = max(B * S for S in self.strides for B in self.bs_list if B * S <= max_tokens_per_graph)
        self.logits_buf = torch.empty((max_rows, self.m.cfg.vocab_size), dtype=self.m.dtype, device=dev)
        for S in reversed(self.strides):
            for B in reversed(self.bs_list):
                T = B * S
                if T > max_tokens_per_graph:
                    continue
                inp = {
                    "input_ids": torch.zeros(T, dtype=torch.int64, device=dev),
                    "positions": torch.arange(T, dtype=torch.int64, device=dev) % S,
                    "cu_seqlens_q": torch.arange(B + 1, dtype=torch.int32, device=dev) * S,
                    "seq_lens": torch.full((B,), S, dtype=torch.int32, device=dev),
                    "block_table": torch.zeros((B, width), dtype=torch.int32, device=dev),
                    "slot_mapping": torch.full((T,), self.scratch_slot, dtype=torch.int64, device=dev),
                }
                # every tensor the graph reads must outlive it, so all inputs live in `inp`
                inp["logits_indices"] = torch.arange(T, device=dev)
                batch = Batch(**inp, max_seqlen_q=S, max_seqlen_k=self.max_len + S)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):  # warm up kernels/allocator outside the graph
                        self.m.forward(batch, out=self.logits_buf)
                torch.cuda.current_stream().wait_stream(stream)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    logits = self.m.forward(batch, out=self.logits_buf)
                self.graphs[(S, B)] = (g, inp, logits)
        torch.cuda.synchronize()

    def pick(self, S_needed, B_needed):
        for S in self.strides:
            if S >= S_needed:
                for B in self.bs_list:
                    if B >= B_needed and (S, B) in self.graphs:
                        return S, B
        return None

    def run(self, seqs, bm):
        """seqs: list of (seq_id, tokens, start). Returns (logits [B*S, V] bf16, S) or None if no graph fits."""
        need_S = max(len(t) for _, t, _ in seqs)
        key = self.pick(need_S, len(seqs))
        if key is None:
            return None
        S, B = key
        g, inp, logits = self.graphs[key]
        T, bs = B * S, self.block_size
        ids = np.zeros(T, dtype=np.int64)
        pos = np.tile(np.arange(S, dtype=np.int64), B)
        slots = np.full(T, self.scratch_slot, dtype=np.int64)
        lens = np.full(B, S, dtype=np.int32)
        table = np.zeros((B, inp["block_table"].shape[1]), dtype=np.int32)
        for i, (sid, toks, start) in enumerate(seqs):
            n, base = len(toks), i * S
            bm.ensure(sid, start + n)
            t = bm.tables[sid]
            table[i, :len(t)] = t
            ids[base:base + n] = toks
            pos[base:base + S] += start
            p = np.arange(start, start + n)
            slots[base:base + n] = table[i, p // bs].astype(np.int64) * bs + p % bs
            lens[i] = start + S
        for name, src in (("input_ids", ids), ("positions", pos), ("slot_mapping", slots),
                          ("seq_lens", lens), ("block_table", table)):
            inp[name].copy_(torch.from_numpy(src), non_blocking=True)
        g.replay()
        return logits, S
