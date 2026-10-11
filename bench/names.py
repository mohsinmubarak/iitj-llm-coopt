"""Descriptive names for experiments, workloads and load levels (single source of truth).

Early in the project everything had short codes (A2, B1D_g4, C4_el, W3, r16, ...). These maps turn
codes into readable names; bench/rename.py applies them to result folders, configs and W&B runs.
"""

EXPERIMENTS = {
    # vLLM baselines
    "A2": "vllm-default",
    "A2_nopc": "vllm-default-no-prefix-cache",
    "A3": "vllm-context-length-sweep",
    "B1": "vllm-ngram-spec-gamma4",
    "B1D_g1": "vllm-draft-spec-gamma1",
    "B1D_g2": "vllm-draft-spec-gamma2",
    "B1D_g4": "vllm-draft-spec-gamma4",
    "B1D_g6": "vllm-draft-spec-gamma6",
    "B1D_g8": "vllm-draft-spec-gamma8",
    "B2": "vllm-cpu-kv-offload",
    "B2S_kv1g": "vllm-kv-budget-1gb",
    "B2S_kv2g": "vllm-kv-budget-2gb",
    "B2S_kv2p5g": "vllm-kv-budget-2.5gb",
    "B2S_kv3g": "vllm-kv-budget-3gb",
    "B2S_kv4g": "vllm-kv-budget-4gb",
    "B5": "vllm-adaptive-spec-schedule",
    # our engine
    "C0_nospec": "ours-static-no-spec",
    "C0": "ours-static-spec-gamma4",
    "C1": "ours-gamma-controller-only",
    "C2": "ours-kv-controller-only-no-elastic",
    "C2_el": "ours-kv-controller-only",
    "C3": "ours-scheduler-controller-only",
    "C4": "ours-cooptimizer-no-elastic",
    "C4_el": "ours-cooptimizer",
    "C5_gk": "ours-gamma-and-kv",
    "C5_gs": "ours-gamma-and-scheduler",
    "C5_ks": "ours-kv-and-scheduler",
    "C6": "ours-independent-controllers-no-elastic",
    "C6_el": "ours-independent-controllers",
    "N1": "ours-nightjar-like",
    # validation
    "M4": "engine-correctness",
    "Q": "quality",
}

# suffixes added by --tag for one-off runs
TAGS = {"smoke": "smoke-test", "pc": "prefix-cache-test", "w2check": "long-context-check",
        "fix": "engine-fix-test"}

WORKLOADS = {
    "W1": "chat", "W2": "long-context", "W3": "mixed-bursty", "W4": "phase-shift",
    "fixed_1024": "prompt-1k", "fixed_4096": "prompt-4k", "fixed_8192": "prompt-8k",
    "fixed_16384": "prompt-16k", "fixed_31744": "prompt-32k",
}

RATES = {"1": "1rps", "4": "4rps", "16": "16rps", "inf": "saturation", "file": "scheduled",
         "r1": "1rps", "r4": "4rps", "r16": "16rps"}


def experiment(code: str) -> str:
    """Experiment code (possibly with a --tag suffix) -> descriptive name. Unknown names pass through."""
    if code in EXPERIMENTS:
        return EXPERIMENTS[code]
    for base in sorted(EXPERIMENTS, key=len, reverse=True):
        if code.startswith(base + "_"):
            tag = code[len(base) + 1:]
            return f"{EXPERIMENTS[base]}_{TAGS.get(tag, tag)}"
    return code


def workload(label: str) -> str:
    """Workload label -> name; concurrency-swept labels like W3_c16 -> mixed-bursty_concurrency16."""
    if label in WORKLOADS:
        return WORKLOADS[label]
    base, _, rest = label.partition("_c")
    if base in WORKLOADS and rest.isdigit():
        return f"{WORKLOADS[base]}_concurrency{rest}"
    return label


def rate(r, scheduled: bool = False) -> str:
    """Load level -> name: 'inf' -> saturation, 4 -> 4rps, file-scheduled arrivals -> scheduled."""
    if scheduled:
        return "scheduled"
    r = str(r)
    if r in RATES:
        return RATES[r]
    try:
        return f"{float(r):g}rps"
    except ValueError:
        return r
