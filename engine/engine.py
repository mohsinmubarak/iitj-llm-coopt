"""Continuous-batching inference engine with optional speculative decoding.

Each step: the scheduler picks work (decodes first, then prefill chunks, then new admissions),
the draft model proposes γ tokens for decoding sequences, the target model runs one mixed
forward pass (prefill chunks + decodes + (γ+1)-token verifications), and rejection sampling
accepts a prefix. All decisions come from the policy (engine/policy.py).

Token bookkeeping per sequence: `tokens` = prompt + generated; the target KV cache holds the
first `num_computed` of them. A step feeds tokens[num_computed : ...]; when that reaches the end
of `tokens`, the step samples new tokens. So decoding is just "one remaining token", and a
preempted sequence (num_computed reset to 0) recomputes prompt + output as one prefill.
"""
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import torch

from engine.batch import WorkItem, build_batch
from engine.block_manager import BlockManager
from engine.cuda_graphs import GraphRunner
from engine.model import Llama
from engine.policy import StaticPolicy
from engine.spec_decode import sample, verify


@dataclass
class Sequence:
    seq_id: int
    tokens: list[int]
    max_tokens: int
    temperature: float = 0.0
    ignore_eos: bool = False
    arrival: float = field(default_factory=time.perf_counter)
    on_output: object = None        # callback(seq, new_tokens, finished)
    prompt_len: int = 0
    num_computed: int = 0           # tokens with target KV in cache
    draft_computed: int = 0         # tokens with draft KV in cache
    first_token_time: float | None = None
    finished: bool = False
    finish_reason: str | None = None
    num_preemptions: int = 0
    gamma: int = 0                  # γ chosen for the current step
    accepted_total: int = 0
    drafted_total: int = 0

    def __post_init__(self):
        self.prompt_len = self.prompt_len or len(self.tokens)

    @property
    def num_generated(self):
        return len(self.tokens) - self.prompt_len

    @property
    def remaining(self):
        return len(self.tokens) - self.num_computed


@dataclass
class Plan:
    chunks: list = field(default_factory=list)   # (seq, end) prefill chunks: feed tokens[num_computed:end]
    decodes: list = field(default_factory=list)  # seqs with exactly one remaining token
    gamma: int = 0                               # γ the policy chose for this step's decodes


class Scheduler:
    def __init__(self, engine):
        self.e = engine
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def _preempt(self, seq):
        """Swap the victim's KV to the CPU tier if the policy says so, otherwise drop it (recompute)."""
        self.running.remove(seq)
        tiers = self.e.kv_tiers
        swapped = bool(tiers) and self.e.policy.swap_on_preempt(seq, self.e) and tiers.swap_out(seq)
        self.e.release(seq)
        if not swapped:
            seq.num_computed = 0
        seq.draft_computed = 0  # draft KV is always recomputed lazily
        seq.num_preemptions += 1
        self.e.counters["num_preemptions_total"] += 1
        self.waiting.appendleft(seq)

    def schedule(self) -> Plan:
        e, pol, bm = self.e, self.e.policy, self.e.bm
        plan, budget = Plan(), pol.max_batched_tokens

        # 1. Decodes: reserve KV for 1 + γ tokens each, preempting the policy's victim if short.
        while True:
            decodes = [s for s in self.running if s.remaining == 1]
            g_step = pol.gamma_for_step(len(decodes), e) if e.draft and decodes else 0
            plan.gamma = g_step
            for s in decodes:
                left = min(s.max_tokens - s.num_generated, e.max_model_len - len(s.tokens)) - 1
                s.gamma = max(0, min(g_step, left))
            need = sum(bm.extra_blocks_needed(s.seq_id, len(s.tokens) + s.gamma) for s in decodes)
            if need <= bm.num_free or len(self.running) <= 1:
                break
            self._preempt(pol.preempt_victim(self.running))
        for s in decodes:
            bm.ensure(s.seq_id, len(s.tokens) + s.gamma)  # reserve before admissions can take it
            plan.decodes.append(s)
            budget -= 1 + s.gamma

        # 2. Continue running prefills, oldest first.
        for s in self.running:
            if s.remaining > 1 and budget > 0:
                end = s.num_computed + min(s.remaining, budget)
                if not bm.can_fit(s.seq_id, end):
                    continue
                bm.ensure(s.seq_id, end)  # reserve now so later admissions see the true free count
                plan.chunks.append((s, end))
                budget -= end - s.num_computed

        # 3. Admit waiting requests while batch, token budget and KV watermark allow.
        reserve = int(pol.watermark * bm.num_blocks)
        tiers = e.kv_tiers
        while self.waiting and budget > 0 and len(self.running) < pol.max_num_seqs:
            i = pol.pick_next(self.waiting, e)
            s = self.waiting[i]
            end = s.num_computed + min(s.remaining, budget)
            if bm.extra_blocks_needed(s.seq_id, end) > bm.num_free - reserve or not pol.admit(s, e):
                break
            del self.waiting[i]
            self.running.append(s)
            if tiers and tiers.has(s.seq_id):
                tiers.swap_in(s)  # restore quantized KV instead of recomputing
            elif s.num_preemptions and tiers:
                tiers.note_recompute()
            bm.ensure(s.seq_id, end)
            if s.remaining == 1:  # resumed sequence that only needs its pending token
                plan.decodes.append(s)
                s.gamma = 0
                budget -= 1
            else:
                plan.chunks.append((s, end))
                budget -= end - s.num_computed
        return plan


class Engine:
    def __init__(self, target: str, draft: str | None = None, policy=None, kv_gb: float | None = None,
                 gpu_memory_utilization=0.9, max_model_len=32768, block_size=16, seed=0, eos_ids=(),
                 cuda_graphs=True, graph_batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128)):
        torch.manual_seed(seed)
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.policy = policy or StaticPolicy()
        self.max_model_len = max_model_len
        self.eos_ids = set(eos_ids)
        self.target = Llama.from_pretrained(target)
        self.draft = Llama.from_pretrained(draft) if draft else None
        if not self.eos_ids:
            eos = self.target.cfg.eos_token_id
            self.eos_ids = set(eos if isinstance(eos, list) else [eos])

        # KV budget: explicit, or what is left under the utilisation cap after the *measured* peak of
        # the largest eager prefill chunk, the CUDA-graph logits buffers and a 256 MiB margin.
        # Target and draft pools get the same number of blocks.
        if kv_gb is None:
            total = torch.cuda.get_device_properties(0).total_memory
            act = self._profile_activation_bytes(self.policy.max_batched_tokens, block_size)
            graphs = 0
            if cuda_graphs:
                gmax = getattr(self.policy, "gamma_max", self.policy.gamma) if self.draft else 0
                rows = max(graph_batch_sizes) * (gmax + 1)
                graphs = rows * self.target.cfg.vocab_size * 2 + 384 * 2**20
                if self.draft:
                    graphs += max(graph_batch_sizes) * 2 * self.draft.cfg.vocab_size * 2 + 128 * 2**20
            kv_bytes = total * gpu_memory_utilization - torch.cuda.memory_allocated() - act - graphs - 256 * 2**20
            print(f"[engine] memory: weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
                  f"prefill activations {act / 2**30:.2f} GiB, graphs {graphs / 2**30:.2f} GiB "
                  f"-> KV {kv_bytes / 2**30:.2f} GiB", flush=True)
        else:
            kv_bytes = kv_gb * 2**30
        per_token = self.target.cfg.kv_bytes_per_token() + (self.draft.cfg.kv_bytes_per_token() if self.draft else 0)
        num_blocks = int(kv_bytes // (per_token * block_size))
        assert num_blocks > 0, "no memory left for the KV cache"
        # One extra block per cache is the CUDA-graph scratch block (padding writes land there);
        # the block managers never hand it out.
        self.bm = BlockManager(num_blocks, block_size)
        self.target.allocate_kv_cache(num_blocks + 1, block_size)
        if self.draft:
            self.dbm = BlockManager(num_blocks, block_size)
            self.draft.allocate_kv_cache(num_blocks + 1, block_size)
        self.kv_capacity_tokens = num_blocks * block_size
        if self.kv_capacity_tokens < self.max_model_len:
            print(f"[engine] KV pool holds {self.kv_capacity_tokens} tokens < max_model_len "
                  f"{self.max_model_len}; capping max_model_len", flush=True)
            self.max_model_len = self.kv_capacity_tokens

        self.target_graphs = self.draft_graphs = None
        if cuda_graphs:
            t = time.perf_counter()
            gmax = getattr(self.policy, "gamma_max", self.policy.gamma) if self.draft else 0
            strides = sorted({1} | {g + 1 for g in (1, 2, 3, 4, 6, 8) if g <= gmax} | ({gmax + 1} if gmax else set()))
            self.target_graphs = GraphRunner(self.target, self.max_model_len, strides, graph_batch_sizes, num_blocks)
            self.target_graphs.capture()
            if self.draft:
                self.draft_graphs = GraphRunner(self.draft, self.max_model_len, [1, 2], graph_batch_sizes, num_blocks)
                self.draft_graphs.capture()
            print(f"[engine] captured {len(self.target_graphs.graphs)} target + "
                  f"{len(self.draft_graphs.graphs) if self.draft_graphs else 0} draft CUDA graphs "
                  f"(target strides {strides}) in {time.perf_counter() - t:.0f}s", flush=True)

        self.scheduler = Scheduler(self)
        self.inbox: queue.Queue = queue.Queue()
        self.counters = {k: 0 for k in (
            "prompt_tokens_total", "generation_tokens_total", "num_preemptions_total",
            "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total", "request_success_total", "steps_total")}
        self.gauges = {}
        self.step_count = 0
        self.recent_ttft = deque(maxlen=128)  # seconds, newest first tokens
        self.metrics = None                   # MetricsEngine attaches itself here
        self.extra_metrics = {}
        self.kv_tiers = None                  # KVTierManager attaches itself here
        self._draft_times = []
        self._stop = threading.Event()

    def _profile_activation_bytes(self, chunk, block_size):
        """Peak extra memory of one `chunk`-token prefill (target, then draft), measured with a
        throwaway KV cache just large enough for it."""
        peak = 0
        for model in [m for m in (self.target, self.draft) if m]:
            bm = BlockManager(-(-chunk // block_size), block_size)
            model.allocate_kv_cache(bm.num_blocks, block_size)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            batch, _ = build_batch([WorkItem(0, [1] * chunk, 0)], bm)
            model.forward(batch)
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated() - base)
            model.k_cache, model.v_cache = [], []
            torch.cuda.empty_cache()
        return peak

    # ---- request lifecycle ---------------------------------------------------------------
    @staticmethod
    def now():
        return time.perf_counter()

    def add_request(self, seq: Sequence):
        self.inbox.put(seq)

    def release(self, seq):
        self.bm.free(seq.seq_id)
        if self.draft:
            self.dbm.free(seq.seq_id)

    def _finish(self, seq, reason):
        seq.finished, seq.finish_reason = True, reason
        self.scheduler.running.remove(seq)
        self.release(seq)
        self.counters["request_success_total"] += 1

    def _append(self, seq, new_tokens) -> list[int]:
        """Append sampled tokens, honouring max_tokens / EOS / max_model_len. Returns those kept."""
        kept = []
        for t in new_tokens:
            seq.tokens.append(t)
            kept.append(t)
            if seq.num_generated >= seq.max_tokens:
                seq.finish_reason = "length"
                break
            if not seq.ignore_eos and t in self.eos_ids:
                seq.finish_reason = "stop"
                break
            if len(seq.tokens) >= self.max_model_len:
                seq.finish_reason = "length"
                break
        return kept

    # ---- one engine step -----------------------------------------------------------------
    @staticmethod
    def _run(model, runner, bm, items):
        """One forward: replay a CUDA graph if the batch is decode-shaped, else run eagerly.
        Returns (logits, spans) with spans[i] = (first logits row, rows) for items[i]."""
        if runner is not None and max(len(it.tokens) for it in items) <= runner.strides[-1]:
            out = runner.run([(it.seq_id, it.tokens, it.start) for it in items], bm)
            if out is not None:
                logits, S = out
                spans = [(i * S, len(it.tokens)) if it.all_logits else (i * S + len(it.tokens) - 1, 1)
                         for i, it in enumerate(items)]
                return logits, spans
        batch, spans = build_batch(items, bm)
        return model.forward(batch), spans

    def _draft_phase(self, plan):
        """Draft model: catch up on new tokens (prefill chunks too) and propose γ tokens."""
        drafts = {s.seq_id: [] for s in plan.decodes}
        dprobs = {s.seq_id: [] for s in plan.decodes}
        items, owners = [], []
        for s, end in plan.chunks:
            if s.draft_computed < end:
                items.append(WorkItem(s.seq_id, s.tokens[s.draft_computed:end], s.draft_computed))
                owners.append((s, end, False))
        # A sequence whose draft KV lags (restored from the CPU tier, or γ was 0 for a while)
        # catches up in chunks bounded by the step's token budget and does not speculate until
        # it has caught up: an unbounded catch-up of a long context would blow the activation
        # budget the KV pool was sized for.
        catchup_budget = self.policy.max_batched_tokens - sum(len(it.tokens) for it in items)
        spec = []
        for s in [s for s in plan.decodes if s.gamma > 0]:
            lag = len(s.tokens) - s.draft_computed
            if lag <= 2:
                spec.append(s)
                items.append(WorkItem(s.seq_id, s.tokens[s.draft_computed:], s.draft_computed))
                owners.append((s, len(s.tokens), True))
                continue
            s.gamma = 0
            n = min(lag - 1, catchup_budget)
            if n > 0:
                catchup_budget -= n
                items.append(WorkItem(s.seq_id, s.tokens[s.draft_computed:s.draft_computed + n], s.draft_computed))
                owners.append((s, s.draft_computed + n, False))
        if not items:
            return drafts, dprobs
        td = time.perf_counter()
        logits, spans = self._run(self.draft, self.draft_graphs, self.dbm, items)
        props = []
        for (s, end, proposes), (row, _) in zip(owners, spans):
            s.draft_computed = end
            if proposes:
                props.append((s, row))
        for step in range(max((s.gamma for s in spec), default=0)):
            if step > 0:
                active = [s for s in spec if s.gamma > step]
                if not active:
                    break
                td = time.perf_counter()
                items = [WorkItem(s.seq_id, [drafts[s.seq_id][-1]], s.draft_computed) for s in active]
                logits, spans = self._run(self.draft, self.draft_graphs, self.dbm, items)
                props = [(s, row) for s, (row, _) in zip(active, spans)]
                for s in active:
                    s.draft_computed += 1
            rows = torch.tensor([r for _, r in props], device=logits.device)
            temp = props[0][0].temperature  # one temperature per batch (the benchmark uses 0)
            toks, probs = sample(logits[rows].float(), temp, self.gen)
            toks = toks.tolist()  # syncs: closes the timing of this draft forward
            self._draft_times.append(time.perf_counter() - td)
            td = time.perf_counter()
            for i, (s, _) in enumerate(props):
                drafts[s.seq_id].append(toks[i])
                if probs is not None:
                    dprobs[s.seq_id].append(probs[i])
        return drafts, dprobs

    def step(self) -> bool:
        plan = self.scheduler.schedule()
        if not plan.chunks and not plan.decodes:
            return False
        t0 = time.perf_counter()
        drafts, dprobs = self._draft_phase(plan) if self.draft else ({}, {})

        items = [WorkItem(s.seq_id, s.tokens[s.num_computed:end], s.num_computed) for s, end in plan.chunks]
        for s in plan.decodes:
            d = drafts.get(s.seq_id, [])
            items.append(WorkItem(s.seq_id, [s.tokens[-1]] + d, s.num_computed, all_logits=bool(d)))
        tt = time.perf_counter()
        n_tokens = sum(len(it.tokens) for it in items)
        logits, spans = self._run(self.target, self.target_graphs, self.bm, items)
        # Greedy requests need only the argmax: one batched argmax + one sync for the whole step.
        greedy = logits.argmax(-1).tolist() if all(s.temperature <= 0 for s in plan.decodes) and \
            all(s.temperature <= 0 for s, _ in plan.chunks) else None
        target_s = time.perf_counter() - tt

        outputs, decode_tokens = [], 0
        for (s, end), (row, _) in zip(plan.chunks, spans):
            self.counters["prompt_tokens_total"] += end - s.num_computed
            s.num_computed = end
            if end == len(s.tokens):  # prefill finished: first token
                if greedy is not None:
                    tok = [greedy[row]]
                else:
                    tok = sample(logits[row:row + 1].float(), s.temperature, self.gen)[0].tolist()
                outputs.append((s, self._append(s, tok)))
        for s, (row, n) in zip(plan.decodes, spans[len(plan.chunks):]):
            d = drafts.get(s.seq_id, [])
            if d and greedy is not None:
                tgt, k = greedy[row:row + n], 0
                while k < len(d) and tgt[k] == d[k]:
                    k += 1
                new = d[:k] + [tgt[k]]
            elif d:
                q = torch.stack(dprobs[s.seq_id]) if dprobs[s.seq_id] else None
                new = verify(logits[row:row + n].float(), d, q, s.temperature, self.gen)
            if d:
                acc = len(new) - 1
                s.accepted_total += acc
                s.drafted_total += len(d)
                self.counters["spec_decode_num_drafts_total"] += 1
                self.counters["spec_decode_num_draft_tokens_total"] += len(d)
                self.counters["spec_decode_num_accepted_tokens_total"] += acc
            elif greedy is not None:
                new = [greedy[row]]
            else:
                new = sample(logits[row:row + 1].float(), s.temperature, self.gen)[0].tolist()
            kept = self._append(s, new)
            decode_tokens += len(kept)
            outputs.append((s, kept))

        now = time.perf_counter()
        for s, kept in outputs:
            # KV is valid for every token except the newly sampled last one.
            s.num_computed = len(s.tokens) - 1
            self.bm.truncate(s.seq_id, s.num_computed)
            if self.draft:
                s.draft_computed = min(s.draft_computed, len(s.tokens) - 1)
                self.dbm.truncate(s.seq_id, s.draft_computed)
            self.counters["generation_tokens_total"] += len(kept)
            if s.first_token_time is None:
                s.first_token_time = now
                self.recent_ttft.append(now - s.arrival)
            done = s.finish_reason is not None
            if done:
                self._finish(s, s.finish_reason)
            if s.on_output:
                s.on_output(s, kept, done)

        self.step_count += 1
        self.counters["steps_total"] += 1
        self.gauges = {
            "num_requests_running": len(self.scheduler.running),
            "num_requests_waiting": len(self.scheduler.waiting),
            "kv_cache_usage_perc": self.bm.usage,
            "step_time_s": now - t0,
            "batch_tokens": n_tokens,
            **self.policy.state(),
        }
        self.policy.observe_step(plan.gamma, len(plan.decodes), decode_tokens, now - t0)
        if self.metrics:
            self.metrics.record_step(n_tokens, target_s, self._draft_times,
                                     [s.gamma for s in plan.decodes])
            if self.step_count % self.policy.tick_every == 0:
                self.policy.tick(self.metrics.snapshot())
        self._draft_times = []
        return True

    # ---- background loop -----------------------------------------------------------------
    def _drain_inbox(self, block):
        try:
            seq = self.inbox.get(timeout=0.05) if block else self.inbox.get_nowait()
            while True:
                self.scheduler.waiting.append(seq)
                seq = self.inbox.get_nowait()
        except queue.Empty:
            pass

    def run_forever(self):
        try:
            while not self._stop.is_set():
                busy = self.scheduler.running or self.scheduler.waiting
                self._drain_inbox(block=not busy)
                if self.scheduler.running or self.scheduler.waiting:
                    self.step()
        except BaseException:
            # A dead engine thread behind a live HTTP server would hang every client forever.
            # Exit the process instead: clients see dropped connections and the benchmark
            # records the failure (same as a vLLM engine crash or OOM).
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)

    def start(self):
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
