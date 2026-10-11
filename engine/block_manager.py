"""Paged KV-cache bookkeeping for the GPU tier, with prefix caching.

The cache is a pool of fixed-size blocks (block_size tokens each). Every sequence owns an ordered
list of block ids (its block table); token at position p lives in slot
    table[p // block_size] * block_size + p % block_size.
Blocks are allocated lazily as a sequence grows and returned when it finishes, is preempted,
or rolls back rejected speculative tokens.

Prefix caching (as in vLLM, on by default there): every *full* block of computed KV gets a chain
hash (its tokens + the previous block's hash), so equal prefixes have equal hashes. A new or
preempted sequence first claims the cached blocks matching its prefix and skips computing them.
Blocks are reference-counted; a released block that carries a hash is not discarded but parked in
an LRU "evictable" set: it counts as free and is only overwritten when memory is needed. That makes
re-admitting a preempted request almost free. Measured on the 4090 rehearsal (W4): vLLM without
prefix caching drops from 0.75 to 0.56 burst SLO and halves its saturation goodput.
"""
from collections import OrderedDict, deque

import torch


class OutOfBlocks(Exception):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16, prefix_caching: bool = False):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.prefix_caching = prefix_caching
        self.free_blocks = deque(range(num_blocks))     # never used, or content not reusable
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref 0 but cached (LRU order)
        self.tables: dict[int, list[int]] = {}
        self.ref: dict[int, int] = {}                   # block -> number of tables holding it
        self.hash_of: dict[int, int] = {}               # block -> chain hash of its full contents
        self.by_hash: dict[int, int] = {}               # chain hash -> block
        self.seq_hashes: dict[int, list[int]] = {}      # seq -> hashes of its leading full blocks
        self.hit_tokens = self.query_tokens = 0

    # ---- capacity -------------------------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free_blocks) + len(self.evictable)

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
    def _forget(self, b):
        h = self.hash_of.pop(b, None)
        if h is not None and self.by_hash.get(h) == b:
            del self.by_hash[h]

    def _take(self) -> int:
        if self.free_blocks:
            b = self.free_blocks.popleft()
        elif self.evictable:
            b, _ = self.evictable.popitem(last=False)  # least recently released cached block
            self._forget(b)
        else:
            raise OutOfBlocks("no free block")
        self.ref[b] = 1
        return b

    def _release(self, b):
        n = self.ref.get(b, 1) - 1
        if n > 0:
            self.ref[b] = n
            return
        self.ref.pop(b, None)
        if b in self.hash_of:
            self.evictable[b] = None
            self.evictable.move_to_end(b)
        else:
            self.free_blocks.append(b)

    def ensure(self, seq_id: int, total_tokens: int) -> None:
        """Grow seq_id's table so it can hold total_tokens tokens."""
        need = self.extra_blocks_needed(seq_id, total_tokens)
        if need > self.num_free:
            raise OutOfBlocks(f"seq {seq_id} needs {need} blocks, {self.num_free} free")
        table = self.tables.setdefault(seq_id, [])
        for _ in range(need):
            table.append(self._take())

    def truncate(self, seq_id: int, total_tokens: int) -> None:
        """Release trailing blocks beyond total_tokens (speculative rollback). Hashed blocks are
        full and below num_computed, so a rollback never reaches them."""
        table = self.tables.get(seq_id)
        if not table:
            return
        keep = self.blocks_for(total_tokens)
        while len(table) > keep:
            self._release(table.pop())

    def free(self, seq_id: int) -> int:
        table = self.tables.pop(seq_id, [])
        for b in table:
            self._release(b)
        self.seq_hashes.pop(seq_id, None)
        return len(table)

    # ---- prefix caching ------------------------------------------------------------------------
    def _chain(self, prev, tokens):
        return hash((prev, tuple(tokens)))

    def match_prefix(self, seq_id: int, tokens: list[int]) -> int:
        """Claim cached blocks for the longest cached prefix of `tokens` (leaving at least one
        token to compute, whose logits produce the next token). Returns tokens covered."""
        if not self.prefix_caching or self.tables.get(seq_id):
            return 0
        bs, prev, hashes, table = self.block_size, None, [], []
        for i in range((len(tokens) - 1) // bs):
            h = self._chain(prev, tokens[i * bs:(i + 1) * bs])
            b = self.by_hash.get(h)
            if b is None:
                break
            if b in self.evictable:
                del self.evictable[b]
            self.ref[b] = self.ref.get(b, 0) + 1
            table.append(b)
            hashes.append(h)
            prev = h
        self.query_tokens += len(tokens)
        if table:
            self.tables[seq_id] = table
            self.seq_hashes[seq_id] = hashes
        self.hit_tokens += len(table) * bs
        return len(table) * bs

    def register(self, seq_id: int, tokens: list[int], num_computed: int) -> None:
        """Hash newly completed full blocks of seq_id so later sequences can reuse them."""
        if not self.prefix_caching:
            return
        table = self.tables.get(seq_id)
        if not table:
            return
        hashes = self.seq_hashes.setdefault(seq_id, [])
        bs, full = self.block_size, min(num_computed // self.block_size, len(table))
        prev = hashes[-1] if hashes else None
        for i in range(len(hashes), full):
            h = self._chain(prev, tokens[i * bs:(i + 1) * bs])
            b = table[i]
            if h not in self.by_hash and b not in self.hash_of:
                self.by_hash[h], self.hash_of[b] = b, h
            hashes.append(h)
            prev = h

    # ---- elastic capacity (memory-elastic speculation) ----------------------------------------
    def add_blocks(self, ids) -> None:
        """Grow the pool with blocks whose memory was just handed over (the parked drafter's)."""
        ids = list(ids)
        self.free_blocks.extend(ids)
        self.num_blocks += len(ids)

    def migrate_out_of(self, ids, copy_fn) -> int:
        """Move every in-use block in `ids` to a free block outside `ids` (copy_fn(src, dst) copies
        the KV), keeping shared blocks shared and hashes valid. Returns blocks moved, or -1 if
        there is not enough room outside `ids`."""
        ids = set(ids)
        used = sorted({b for t in self.tables.values() for b in t if b in ids})
        outside = sum(1 for b in self.free_blocks if b not in ids) + sum(1 for b in self.evictable if b not in ids)
        if outside < len(used):
            return -1
        for src in used:
            dst = self.take_free_excluding(ids)
            copy_fn(src, dst)
            self.ref[dst] = self.ref.pop(src)
            h = self.hash_of.pop(src, None)
            if h is not None:
                self.hash_of[dst] = h
                self.by_hash[h] = dst
            for t in self.tables.values():
                for i, b in enumerate(t):
                    if b == src:
                        t[i] = dst
            self.free_blocks.append(src)
        return len(used)

    def withdraw_free(self, ids) -> None:
        """Shrink the pool by removing blocks that are currently free or merely cached."""
        ids = set(ids)
        for b in [b for b in self.evictable if b in ids]:
            del self.evictable[b]
            self._forget(b)
            self.free_blocks.append(b)
        remaining = deque(b for b in self.free_blocks if b not in ids)
        if len(self.free_blocks) - len(remaining) != len(ids):
            raise OutOfBlocks("withdraw_free: some blocks are still in use")
        self.free_blocks = remaining
        self.num_blocks -= len(ids)

    def take_free_excluding(self, excluded) -> int:
        """Take one free (or cached-evictable) block that is not in `excluded` (migration target)."""
        for i, b in enumerate(self.free_blocks):
            if b not in excluded:
                del self.free_blocks[i]
                return b
        for b in self.evictable:
            if b not in excluded:
                del self.evictable[b]
                self._forget(b)
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
