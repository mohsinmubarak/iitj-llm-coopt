"""Paged KV-cache bookkeeping for the GPU tier.

The cache is a pool of fixed-size blocks (block_size tokens each). Every sequence owns an ordered
list of block ids (its block table); token at position p lives in slot
    table[p // block_size] * block_size + p % block_size.
Blocks are allocated lazily as a sequence grows and returned when it finishes, is preempted,
or rolls back rejected speculative tokens. The KV tier manager (CPU tier, eviction) builds on this.
"""
from collections import deque

import torch


class OutOfBlocks(Exception):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.free_blocks = deque(range(num_blocks))
        self.tables: dict[int, list[int]] = {}

    # ---- capacity -------------------------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    @property
    def usage(self) -> float:
        return 1.0 - self.num_free / self.num_blocks

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def extra_blocks_needed(self, seq_id: int, total_tokens: int) -> int:
        return max(0, self.blocks_for(total_tokens) - len(self.tables.get(seq_id, ())))

    def can_fit(self, seq_id: int, total_tokens: int) -> bool:
        return self.extra_blocks_needed(seq_id, total_tokens) <= self.num_free

    # ---- allocation ----------------------------------------------------------------------
    def ensure(self, seq_id: int, total_tokens: int) -> None:
        """Grow seq_id's table so it can hold total_tokens tokens."""
        need = self.extra_blocks_needed(seq_id, total_tokens)
        if need > self.num_free:
            raise OutOfBlocks(f"seq {seq_id} needs {need} blocks, {self.num_free} free")
        table = self.tables.setdefault(seq_id, [])
        for _ in range(need):
            table.append(self.free_blocks.popleft())

    def truncate(self, seq_id: int, total_tokens: int) -> None:
        """Release trailing blocks beyond total_tokens (speculative rollback)."""
        table = self.tables.get(seq_id)
        if not table:
            return
        keep = self.blocks_for(total_tokens)
        while len(table) > keep:
            self.free_blocks.append(table.pop())

    def free(self, seq_id: int) -> int:
        table = self.tables.pop(seq_id, [])
        self.free_blocks.extend(table)
        return len(table)

    # ---- elastic capacity (memory-elastic speculation) ----------------------------------------
    def add_blocks(self, ids) -> None:
        """Grow the pool with blocks whose memory was just handed over (the parked drafter's)."""
        ids = list(ids)
        self.free_blocks.extend(ids)
        self.num_blocks += len(ids)

    def withdraw_free(self, ids) -> None:
        """Shrink the pool by removing blocks that are currently free (caller migrated the rest)."""
        ids = set(ids)
        remaining = deque(b for b in self.free_blocks if b not in ids)
        if len(self.free_blocks) - len(remaining) != len(ids):
            raise OutOfBlocks("withdraw_free: some blocks are still in use")
        self.free_blocks = remaining
        self.num_blocks -= len(ids)

    def take_free_excluding(self, excluded) -> int:
        """Pop one free block that is not in `excluded` (migration target)."""
        for i, b in enumerate(self.free_blocks):
            if b not in excluded:
                del self.free_blocks[i]
                return b
        raise OutOfBlocks("no free block outside the excluded range")

    # ---- tensors for the forward pass -----------------------------------------------------
    def slot_mapping(self, seq_id: int, start: int, end: int) -> torch.Tensor:
        """Cache slots for positions [start, end) of seq_id (CPU int64 tensor)."""
        table = torch.tensor(self.tables[seq_id], dtype=torch.int64)
        pos = torch.arange(start, end, dtype=torch.int64)
        return table[pos // self.block_size] * self.block_size + pos % self.block_size

    def block_table_tensor(self, seq_ids: list[int]) -> torch.Tensor:
        """[B, max_blocks] int32, rows padded with 0 (padding is never read: seq_lens bounds it)."""
        width = max(len(self.tables[s]) for s in seq_ids)
        out = torch.zeros((len(seq_ids), width), dtype=torch.int32)
        for i, s in enumerate(seq_ids):
            t = self.tables[s]
            out[i, :len(t)] = torch.tensor(t, dtype=torch.int32)
        return out
