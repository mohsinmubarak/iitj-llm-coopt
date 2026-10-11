# Related work and future-work references

Literature survey of work close to this project (joint control of speculative-decoding γ, KV-cache
tiering and request scheduling). Compiled 11 Oct 2026.

Nightjar and SLO-Tuner were read in full. The other entries are summarised from abstracts and reviews.
All numbers are as reported by their authors and were not reproduced here.

**Bottom line:** no published system we found controls γ, KV swap/recompute and admission/batch cap
jointly from one shared state, and none isolates the value of coordination itself (our C4 vs. C6).
Several papers cover one or two axes and must be cited and positioned against.

---

## 1. Closest prior work: Nightjar

**Nightjar: Dynamic Adaptive Speculative Decoding for LLM Serving.** R. Li, Z. Zhang et al. (NUDT).
arXiv [2512.22420](https://arxiv.org/abs/2512.22420) (v3, Feb 2026). Built on vLLM 0.8.2.

- **What it controls:** γ per batch size (including γ = 0, i.e. speculation off); whether the draft model
  stays on GPU (offloaded to CPU under pressure, so that memory goes to the KV pool).
- **Method:** a bandit over γ with separate state per batch size (ADA-BINGREEDY style, blocks and bins).
  The loss is per-token latency plus a cost for switching from γ = 0 back to γ > 0.
- **Signals:** batch size, measured per-token latency, free KV blocks, waiting-queue length, and an
  offline-profiled switching-cost table.
- **Setups:** 7B target + 0.5B draft on an **RTX 4090** (very close to our rehearsal), Vicuna-13B on
  A100-40GB, Qwen2.5-32B on 2× L20.
- **Results:** up to about 27 % more throughput and 13 % lower latency. The abstract and the body give
  inconsistent baselines for these numbers. Baselines were no SD, fixed γ = 3, DSD, BanditSpec and TETRIS.
- **Not covered:** KV swap vs. recompute, CPU tiering or quantization, admission or batch-cap control,
  any coordination ablation. Assumes stationary traffic.
- **Relation to us:** our γ controller is "Nightjar-like plus pressure coupling to KV and scheduling".
  Its finding that speculation should switch off at high load bears on our open question about the
  bandit not learning γ = 0 at large batches (see C0_nospec).

## 2. Adaptive speculation length (γ only)

| Work | Idea | Relevance to us |
|---|---|---|
| **TurboSpec / SmartSpec.** X. Liu et al., arXiv [2406.14066](https://arxiv.org/abs/2406.14066) (v1 titled *Optimizing Speculative Decoding for Serving LLMs Using Goodput*) | Closed-loop γ control in vLLM: offline profiling plus online goodput feedback | Already cited. Same goodput framing as ours |
| **BanditSpec.** Hou et al., ICML 2025, arXiv [2505.15141](https://arxiv.org/abs/2505.15141) | UCBSpec / EXP3Spec for choosing γ, with regret bounds | Theory citation for our bandit γ controller |
| **DSDE.** arXiv [2509.01083](https://arxiv.org/abs/2509.01083) | γ per sequence and per step from KL-divergence variance; batch-wide γ cap; in vLLM | Per-sequence γ, like ours; no memory or scheduling coupling |
| **TETRIS.** Wu et al., ACL 2025, arXiv [2502.15197](https://arxiv.org/abs/2502.15197) | Chooses which draft tokens to verify across the batch, by acceptance probability | Alternative to a per-sequence γ. Possible future work: budget allocation by acceptance probability under our shared draft-token budget |
| **TreeSpark.** Zhou et al., arXiv [2609.22098](https://arxiv.org/abs/2609.22098) (Aug 2026) | Calibrated draft trees that shrink back to a chain as load rises | Load-adaptive speculation; trees are out of scope for us |
| **The Synergy of Speculative Decoding and Batching.** arXiv [2310.18813](https://arxiv.org/abs/2310.18813) | The best γ depends on batch size; offline lookup table | Supports our finding that the best fixed γ depends on load |
| **Speculative Speculative Decoding.** T. Kumar, T. Dao, A. May, arXiv [2603.03251](https://arxiv.org/abs/2603.03251) | Drafting runs in parallel with verification; serving stack preempts when lookahead reservations fail | Speculation that affects preemption |
| **vLLM `num_speculative_tokens_per_batch_size`** ([docs](https://docs.vllm.ai/en/v0.26.0/features/speculative_decoding/dynamic_speculative_decoding/)) | Static table from concurrency range to γ (e.g. γ = 0 above 128 concurrent requests). Tested with Eagle, Eagle-3 and DFlash only | **Action:** check whether vLLM 0.31 supports it with a draft model. If it does, it is the fair "strongest stock vLLM" baseline |

## 3. Speculation together with scheduling or SLOs

| Work | Idea | Relevance to us |
|---|---|---|
| **AdaServe.** Z. Li, Z. Jia et al., arXiv [2501.12162](https://arxiv.org/abs/2501.12162) | Speculation trees sized to each request's latency target; speculate-select-verify pipeline; up to 4.3× fewer SLO violations | Already cited. Covers scheduling and speculation, not KV |
| **StreamServe.** arXiv [2604.09562](https://arxiv.org/abs/2604.09562) | Disaggregated prefill/decode; speculation reacts to system load | Multi-GPU setting; we are single-GPU |
| **SLO-Tuner.** Atinafu, Lin, Cohen (Waterloo), arXiv [2603.11340](https://arxiv.org/abs/2603.11340) (Mar 2026) | Black-box hill-climb over concurrency, `max_num_seqs` and speculation width for p99-SLO goodput | Only other work that tunes speculation and batching together. Weak: restarts vLLM at every step, TinyLlama-scale models, no KV control. Its best point disabled speculation |
| **Semi-Clairvoyant Scheduling of Speculative Decoding Requests.** arXiv [2505.17074](https://arxiv.org/abs/2505.17074) | Request ordering that accounts for speculation, to cut latency | Speculation-aware ordering; compare with our aged shortest-first |
| **NebulaSD** arXiv [2609.29364](https://arxiv.org/abs/2609.29364); **DSpark** arXiv [2607.05147](https://arxiv.org/abs/2607.05147) | NebulaSD: scheduling re-runs only when state changes. DSpark: admits draft tokens by survival probability up to a batch-size target | Recent; read before submitting |

## 4. KV preemption: swap vs. recompute, tiering, compression

| Work | Idea | Relevance to us |
|---|---|---|
| **CacheOPT.** arXiv [2503.13773](https://arxiv.org/abs/2503.13773) | For each preemption, picks whichever of swap or recompute is faster | **Same rule as our KV controller; must cite.** Our additions: shared pressure signal and INT8/INT4 swap |
| **ConServe.** arXiv [2410.01228](https://arxiv.org/abs/2410.01228) | Per-token KV placement in GPU memory, host memory or recompute; incremental checkpointing; overlapped swap-in | Finer than our per-sequence decision. Future work |
| **Semantic Scheduling for LLM Inference.** arXiv [2506.12204](https://arxiv.org/abs/2506.12204) | Analytical break-even: prefill cost grows quadratically, reload cost linearly | Supports our measured PCIe vs. prefill-cost decision |
| **Saving GPU Hours… DBMS-inspired cache replacement.** arXiv [2411.07447](https://arxiv.org/abs/2411.07447) | Recompute usually beats swap under PCIe limits | Counter-evidence; our quantized swap has to beat it |
| **EVICPRESS.** arXiv [2512.14946](https://arxiv.org/abs/2512.14946) | Joint KV compression and eviction | Joint control within KV only |
| **CacheWise.** arXiv [2606.16824](https://arxiv.org/abs/2606.16824) | KV management for coding-agent workloads; CPU offload | Agent workloads: future work |
| **vLLM V1 preemption.** [forum thread](https://discuss.vllm.ai/t/request-preemption-option/1672) | Recompute is the default; swap is kept mainly for beam search | The default our KV controller is measured against |
| **QuantSpec** (ICML 2025, [2502.10424](https://arxiv.org/abs/2502.10424)), **MagicDec** ([2408.11049](https://arxiv.org/abs/2408.11049)), **SpeCache** ([2503.16163](https://arxiv.org/abs/2503.16163)) | Speculation and KV combined at the kernel or algorithm level (quantized or sparse KV for drafting) | Already cited (QuantSpec, MagicDec). Not runtime control loops |

## 5. Multi-knob tuning and autoscaling

| Work | Idea | Relevance to us |
|---|---|---|
| **SCOOT** | Bayesian optimisation over engine parameters, random-forest feasibility; excludes speculation | Offline tuning, a contrast to our online control |
| **Chiron.** arXiv [2501.08090](https://arxiv.org/abs/2501.08090) | Hierarchical autoscaler; local controller sets each instance's batch cap from backpressure | Close to our AIMD batch cap |
| **UNISON** [2609.09643](https://arxiv.org/abs/2609.09643); **Unified AI Gateway** [2609.06940](https://arxiv.org/abs/2609.06940) | UNISON: KV residency scheduling for agents. Gateway: joint model routing and KV | Joint control in other areas |

---

## Action items

1. **Comparison table:** add Nightjar, BanditSpec, CacheOPT, SLO-Tuner and Chiron to the README
   comparison table (§3) and references (§16).
2. **Nightjar positioning:** state explicitly what Nightjar does not do (KV tiering, swap vs. recompute,
   admission, coordination ablation).
3. **Coordination ablation:** make C4 vs. C6 the centrepiece. No paper above isolates coordination.
4. **vLLM baseline:** check `num_speculative_tokens_per_batch_size` in vLLM 0.31 as a Tier-B baseline
   (a static per-regime table, which fits the "matches the best static config in every regime" framing).

## Future-work ideas drawn from this survey

- **Draft model offload:** offload or resize the draft model under KV pressure (Nightjar), as a fourth
  knob under the same pressure level.
- **Token-level KV placement:** per-token placement and incremental checkpointing (ConServe) instead of
  per-sequence swap or recompute.
- **Draft-token budget allocation:** allocate the shared draft-token budget across sequences by
  acceptance probability (TETRIS) rather than per-sequence γ caps.
- **Adversarial-bandit γ:** use an adversarial bandit (EXP3Spec from BanditSpec) for γ under
  non-stationary traffic such as the W4 phase-shift workload.
- **Tree speculation:** tree-shaped speculation that adapts to load (TreeSpark, AdaServe).
- **Per-request SLO targets:** per-request SLO targets driving γ and admission (AdaServe), extending
  goodput to multi-SLO goodput.
