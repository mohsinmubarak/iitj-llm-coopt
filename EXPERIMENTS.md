# Experiment Catalogue — Unified LLM Inference Co-Optimization Framework

Every experiment exists to support (or test) a specific thesis claim.
If an experiment doesn't map to a claim, it gets cut.

**Priority:** **P0** = thesis fails without it · **P1** = examiners will ask for it · **P2** = only if time allows
**Workloads:** W1 short/chat (ShareGPT) · W2 long-context (LongBench, 8K–32K) · W3 mixed bursty (Poisson, 70/30 short/long, 60+ concurrent)
**Hardware:** all reported numbers come from 1× A100 80GB, 3 repeats, mean ± std.
**Rehearsal:** every experiment runs first on 1× RTX 4090 (profile `rtx4090`: 16K context, 1 repeat, W&B project `iitj-coopt-rtx4090`), then again on the A100 (profile `a100`, project `iitj-coopt-a100`). The 4090 set doubles as the second-GPU result (G2).
**Models:** target `unsloth/Meta-Llama-3.1-8B-Instruct` (BF16) · draft `unsloth/Llama-3.2-1B-Instruct` (BF16): ungated mirrors of the official Meta weights
**Clean slate:** every number below is measured fresh in this project. No Phase-1 results are reused.

---

## The claims being tested

| ID | Hypothesis | Proven by |
|---|---|---|
| **H1** | No single static configuration (γ, KV policy, batch size) is optimal across workloads and load levels. That is the gap. | A3, B1-S, B2-S |
| **H2** | Each controller alone improves performance in the regime it targets | C1, C2, C3 vs C0 |
| **H3** | **Joint control through one shared snapshot beats the same controllers acting independently.** This is the core novelty | C4 vs C5 vs C6 |
| **H4** | The gains cost no quality, no tail latency and no stability | Q1–Q4, L4, TTFT p99 in all runs |
| **H5** | The mechanism behaves as designed and its overhead is negligible | M1–M4, sensitivity sweeps |
| **H6** | The system beats the strongest vLLM configuration on the same hardware | C4 vs B3/B4 (main table) |

---

## Tier A — Floors (no optimisations)

| ID | Experiment | Config | Workloads | Pri | Significance |
|---|---|---|---|---|---|
| **A1** | Naive inference | HF `transformers` `generate()`, static batching, no paged KV | W1, W2 | P1 | The "naive" end of the motivation claim ("2–4× headroom between optimised and naive inference on identical hardware"). Measures that headroom yourself instead of citing it |
| **A2** | Stock vLLM, defaults | vLLM out of the box: BF16, CUDA graphs, max_len 32K, no spec, no offload, default `max_num_seqs` | W1–W3 | P0 | The production floor. Separates what vLLM already gives (paged attention, continuous batching, graphs) from every later optimisation |
| **A3** | Context-length collapse | A2, prompt length ∈ {1K, 4K, 8K, 16K, 32K} at fixed concurrency | synthetic | P0 | Measures the "throughput collapse at long context" claim **on your own hardware**. Turns a cited number into a measured one, and pinpoints the regime where co-optimization should matter most (H1) |

## Tier B — Strongest vLLM baselines

| ID | Experiment | Config | Workloads | Pri | Significance |
|---|---|---|---|---|---|
| **B1** | Fixed-γ spec decoding | A2 + n-gram drafter, γ=4 | W1–W3 | P0 | The baseline for the γ controller: what spec decoding gives *without* adaptation |
| **B1-S** | Fixed-γ sweep (configs `B1D_g{1,2,4,6,8}`) | 1B draft model, γ ∈ {1, 2, 4, 6, 8} × request rate ∈ {1, 4, 16, ∞}. The draft model rather than n-gram, because that is the drafter our γ controller adapts | W1, W2 | **P0** | **Key motivation result for H1.** Expected: the best γ shifts with load and workload (high γ helps at low load and hurts at high load/memory pressure). If one γ wins everywhere, dynamic γ has no reason to exist, so you need to know early |
| **B1-D** | Draft-model spec decoding | Same Llama-3.2-1B drafter as our engine, fixed γ=4 | W1, W2 | **P0** | **The like-for-like baseline for the γ controller**: same target, same drafter, fixed γ. Any C1 gain over this comes from adapting γ alone. n-gram acceptance is often low on chat, so B1 alone could be called a weak baseline |
| **B2** | KV offload | A2 + vLLM CPU offload / LMCache connector | W2, W3 | P0 | The baseline for the KV tier manager: memory offload with a static policy |
| **B2-S** | Memory-pressure sweep | `gpu_memory_utilization` ∈ {0.5, 0.7, 0.9} × concurrency ∈ {16, 32, 64, 128} | W3 | P1 | Maps the preemption/OOM cliff of static vLLM, showing *where* blind eviction and an uninformed scheduler break (H1, the "coherence gap") |
| **B3** | Best tuned vLLM | marlin + graphs + best spec config + offload + chunked prefill + tuned `max_num_seqs` | W1–W3 | P0 | **The bar for H6.** Every headline speedup is measured against this. Document the tuning grid so nobody can call it a straw man |
| **B4** | B3 + FP8 KV cache | B3 with `kv_cache_dtype=fp8` | W2, W3 | P1 | Fair comparison for your INT8/INT4 KV compression: vLLM already offers KV quantization, so the examiner will ask how yours compares |

## Tier C — Your engine (ablation)

| ID | Experiment | γ ctrl | KV mgr | Sched | Shared snapshot | Pri | Significance |
|---|---|---|---|---|---|---|---|
| **C0** | Static engine | fixed γ=4 | static LRU | fixed batch | – | **P0** | Your engine with every controller frozen. **Isolates control-policy gains from engine/implementation differences.** Without it, nobody can attribute B→C gains to co-optimization |
| **C1** | γ only | ✓ | static | fixed | – | P0 | H2 for speculative decoding: dynamic γ vs fixed γ under identical conditions |
| **C2** | KV only | fixed | ✓ | fixed | – | P0 | H2 for memory: priority-weighted tiered eviction + quantization vs LRU. Watch KV hit rate and OOMs |
| **C3** | Scheduler only | fixed | static | ✓ | – | P0 | H2 for scheduling: adaptive batching + TTFT preemption. Watch TTFT p99 and preemption % |
| **C4** | Full co-optimizer | ✓ | ✓ | ✓ | ✓ | **P0** | **The thesis result (H6).** All three axes driven by one `CoOptimizer.tick()` |
| **C5** | Pairwise ablations | {γ+KV, γ+Sched, KV+Sched} | | | ✓ | P1 | Shows which axis interactions produce the gain. Lets you report *synergy*: gain(C4) − [gain(C1) + gain(C2) + gain(C3)]. A positive value is direct evidence for "joint > sum of parts" |
| **C6** | **Independent controllers** | ✓ | ✓ | ✓ | **✗** (each reads its own signals, on its own clock) | **P0** | **The most important experiment for novelty (H3).** C4 and C6 run the *same* three controllers, so any difference comes purely from coordination through the shared feedback loop. This answers "isn't this just three known techniques stacked?", which is the first question a reviewer will ask |

> Run C0–C4 + C6 on W1–W3, 3 repeats. Run C5 on W3 only.

## Sensitivity sweeps (robustness of the controller design)

| ID | Parameter swept | Values | Setup | Pri | Significance |
|---|---|---|---|---|---|
| **S1** | γ_max cap | {4, 5, 6, 8} | C4, W1+W2 | P1 | Shows the controller isn't relying on one tuned cap. Expected: gains saturate, and too-high caps waste compute on long context |
| **S2** | KV eviction threshold | P_gpu ∈ {0.7, 0.8, 0.9} | C4, W3 | P1 | Trade-off between early offloading (PCIe traffic) and late offloading (OOM risk). Justifies your default |
| **S3** | Tick interval N | {1, 4, 16, 64} steps | C4, W3 | P1 | Responsiveness vs overhead. Too frequent costs compute; too rare means stale decisions. Justifies "every N steps" |
| **S4** | α EMA smoothing | β ∈ {0.5, 0.8, 0.95} | C1, W1 | P2 | Too reactive leads to γ oscillation; too smooth adapts slowly. Supports the cooldown-gate design |
| **S5** | KV quant level | none / INT8 / INT4 | C2, W2 | P0 | The memory-vs-quality trade-off curve. Paired with Q1, it is the *real* quality-loss evidence |

## Load & robustness

| ID | Experiment | Setup | Pri | Significance |
|---|---|---|---|---|
| **L1** | Throughput–latency curve | B3 vs C4, request rate ∈ {1, 2, 4, 8, 16, 32, ∞}, W3 | **P0** | Standard serving-paper figure. Lets you report **goodput** (requests/s meeting a TTFT SLO), often the most convincing single number |
| **L2** | Concurrency sweep | 1 → 128 concurrent, B3 vs C4 | P0 | Directly checks the "60+ concurrent requests" claim and where each system saturates |
| **L3** | Phase-shift workload | 5 min short-chat → 5 min long-context → 5 min burst, single run | **P0** | **Best showcase of adaptation.** A static config tuned for one phase is wrong for the next. Plotting γ, P_gpu, batch size and TTFT over time against the phase boundaries shows the controller reacting, which is a figure static systems can't produce |
| **L4** | OOM stress test | max concurrency + 32K contexts, 30 min, B3 vs C4 | P0 | Hard pass/fail for "Zero OOM". Also log preemptions and recomputations |
| **L5** | Long-context scaling | prompt length ∈ {4K, 8K, 16K, 32K}, B3 vs C4 | P1 | Tests the stretch target (≥3× on long context). Mirror of A3: A3 shows the problem, L5 shows the fix |

## Quality (H4)

| ID | Experiment | Setup | Pri | Significance |
|---|---|---|---|---|
| **Q1** | Perplexity | WikiText-2 + ShareGPT sample. KV quant none / INT8 / INT4 | P0 | The measured basis for the "< 1% quality loss" criterion. Isolates the only lossy component (KV quantization) |
| **Q2** | Greedy equivalence | Same prompts, temperature 0: C4 output vs no-spec output, exact-match % | P0 | Empirically shows the spec-decoding pipeline is lossless (exact match apart from numerical noise). Also catches verifier bugs, which a perplexity check alone would miss |
| **Q3** | MMLU subset | 500–1000 questions, B3 vs C4 | P1 | Task-level accuracy, the metric examiners trust most |
| **Q4** | LongBench scores | 3–4 tasks, B3 vs C4 (INT8 / INT4 KV) | P1 | Checks that KV compression on evicted blocks doesn't hurt long-range retrieval, which is where quantized KV most often fails |

## Mechanism & overhead (H5)

| ID | Experiment | Setup | Pri | Significance |
|---|---|---|---|---|
| **M1** | Controller overhead | Time per `tick()` and % of decode-step time, C4 at max load | P0 | Shows the control plane is cheap (target < 2% of step time). Without it, someone can argue the gains get eaten at scale |
| **M2** | Live signal traces | Log all secondary metrics per tick during C4 W3 and L3 | P0 | Supports the secondary criteria: γ range ≥ 3, α ≥ 0.65, KV hit ≥ 60%, evictions < 5/step, preemption < 5% |
| **M3** | Effective tokens/step vs γ×α | From C1/C4 logs | P1 | Checks the spec pipeline is net positive and that verification batching delivers the expected tokens per step |
| **M4** | Engine correctness | C0 with γ=0: greedy outputs and logits vs HF `generate()` on 100 prompts (single request and batched) | **P0** | A from-scratch engine must first prove it computes the same model. Run it on Day 1 and re-run it after every engine change. Every performance number depends on this passing |

## Generalization (stretch)

| ID | Experiment | Pri | Significance |
|---|---|---|---|
| **G1** | Second model size (Llama-3.1-8B target with a different draft, or a Llama-3.x 3B target), B3 vs C0 vs C4 on W3 | P2 | Shows the controller isn't overfit to one target/draft pair. Stays in the Llama family because the engine only implements the Llama architecture |
| **G2** | Second GPU (L40S or A6000), B3 vs C4 on W3 | P2 | Tests the hardware-profiler claim: thresholds auto-adapt to different VRAM/bandwidth |

---

## Metrics recorded in *every* run (ExperimentTracker JSON)
- **Primary:** output tok/s · request throughput · TTFT p50/p99 · TPOT/ITL p50/p99 · E2E latency p99 · OOM count
- **Secondary:** γ (mean/min/max/trace) · α (EMA trace) · P_gpu/P_cpu trace · KV hit rate · evictions/step · preemption % · eff. tok/step
- **Metadata:** config hash · git commit · vLLM/torch versions · GPU · seed · workload file hash

---

## Execution order & GPU budget (≈ 40 GPU-hours)

| Day | Experiments | Est. GPU-h |
|---|---|---|
| 1 | A1, A2, A3 · M4 (engine correctness, single request) | 4 |
| 2 | B1, **B1-S**, **B1-D**, B2, B2-S, B3, B4 (in the background while the engine is written) | 10 |
| 3 | M4 (batched + spec) · C0, C1 (debug on a cheap GPU first) | 6 |
| 4 | C2, C3, C4, **C6**, C5, M1, M2 | 9 |
| 5 | 3× repeats of A2/B3/B1-D/C0–C4/C6 · L1–L5 · S1–S3, S5 · Q1–Q4 | 14 |

**If time runs short, cut in this order:** G → S4 → A1 → C5 → B2-S → S1–S3 → Q3/Q4 → B4.
**Never cut:** M4, A2, B1-S, B1-D, B3, C0, C1–C4, **C6**, L1, L3, L4, Q1, Q2, M1, M2.

## How results map to thesis chapters
- **Motivation / problem chapter:** A1 vs A2 (software headroom), A3, B1-S, B2-S (the gap exists, measured)
- **Engine validation:** M4, Q2 (the custom engine is correct before any speed claim)
- **Main results table:** A2 · B1 · B1-D · B2 · B3 · C0 · C4, across W1–W3 (H6)
- **Ablation section:** C0–C6, plus the synergy number (H2, H3)
- **Adaptation figure:** L3 time series (the signature figure)
- **Serving-performance figures:** L1 goodput curve, L2, L5
- **Quality section:** Q1–Q4, S5
- **Design justification / sensitivity:** S1–S4, M1–M3
- **Threats to validity:** single GPU, single model (or G1/G2 if done), n-gram vs model drafter
