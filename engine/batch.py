"""Builds a model Batch from per-sequence work items.

A work item says: sequence `seq_id` appends `tokens` at positions [start, start + len(tokens)).
That one shape covers prefill (start=0, whole prompt), chunked prefill, decode (one token) and
speculative verification (last accepted token + γ drafts, logits for every position).
"""
from dataclasses import dataclass

import torch

from engine.block_manager import BlockManager
from engine.model import Batch


@dataclass
class WorkItem:
    seq_id: int
    tokens: list[int]
    start: int
    all_logits: bool = False  # verification needs logits at every new position, decode only the last


def build_batch(items: list[WorkItem], bm: BlockManager, device="cuda"):
    """Allocates blocks for the new tokens and returns (Batch, [(row, n_rows)] per item)."""
    ids, pos, slots, logit_rows, spans = [], [], [], [], []
    cu, lens = [0], []
    for it in items:
        n = len(it.tokens)
        end = it.start + n
        bm.ensure(it.seq_id, end)
        ids.extend(it.tokens)
        pos.append(torch.arange(it.start, end, dtype=torch.int64))
        slots.append(bm.slot_mapping(it.seq_id, it.start, end))
        offset = cu[-1]
        if it.all_logits:
            spans.append((len(logit_rows), n))
            logit_rows.extend(range(offset, offset + n))
        else:
            spans.append((len(logit_rows), 1))
            logit_rows.append(offset + n - 1)
        cu.append(offset + n)
        lens.append(end)
    to = dict(device=device, non_blocking=True)
    batch = Batch(
        input_ids=torch.tensor(ids, dtype=torch.int64).pin_memory().to(**to),
        positions=torch.cat(pos).pin_memory().to(**to),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32).to(**to),
        seq_lens=torch.tensor(lens, dtype=torch.int32).to(**to),
        block_table=bm.block_table_tensor([it.seq_id for it in items]).to(**to),
        slot_mapping=torch.cat(slots).pin_memory().to(**to),
        logits_indices=torch.tensor(logit_rows, dtype=torch.int64).to(**to),
        max_seqlen_q=max(len(it.tokens) for it in items),
        max_seqlen_k=max(lens),
    )
    return batch, spans
