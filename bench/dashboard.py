"""Create / refresh the saved W&B workspace view used to present results.

    python -m bench.dashboard
"""
import os

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

PROJECT = os.environ.get("WANDB_PROJECT", "iitj-llm-coopt")


def bars(title, metric):
    return wr.BarPlot(title=title, metrics=[metric], orientation="v")


def lines(title, *ys):
    return wr.LinePlot(title=title, x="t", y=list(ys))


def main():
    entity = wandb.Api().default_entity
    view = ws.Workspace(
        entity=entity, project=PROJECT, name="Thesis dashboard",
        sections=[
            ws.Section(name="1. Headline metrics (per run)", is_open=True, panels=[
                bars("Output throughput (tok/s)", "output_throughput_tok_s"),
                bars("Goodput (req/s meeting SLO)", "goodput_req_s"),
                bars("TTFT p99 (s)", "ttft_p99"),
                bars("TPOT p99 (s)", "tpot_p99"),
                bars("E2E latency p99 (s)", "e2e_p99"),
                bars("Failed requests (OOM / errors)", "failed"),
            ]),
            ws.Section(name="2. Mechanism (secondary metrics)", is_open=True, panels=[
                bars("Spec-decoding acceptance rate α", "spec_acceptance_rate"),
                bars("Accepted tokens per draft", "spec_mean_accepted_per_draft"),
                bars("Preemptions", "preemptions"),
                bars("Peak KV-cache usage", "kv_cache_usage_peak"),
                bars("Peak GPU memory (GB)", "gpu_mem_peak_gb"),
                bars("Prefix-cache hit rate", "prefix_cache_hit_rate"),
            ]),
            ws.Section(name="3. Live behaviour over time", is_open=True, panels=[
                lines("Requests running / waiting", "ts/vllm_num_requests_running", "ts/vllm_num_requests_waiting"),
                lines("KV-cache usage", "ts/vllm_kv_cache_usage_perc"),
                lines("Generation throughput (tok/s)", "ts/rate/vllm_generation_tokens_per_s"),
                lines("Accepted draft tokens / s", "ts/rate/vllm_spec_decode_num_accepted_tokens_per_s"),
                lines("GPU memory (GB)", "ts/gpu_mem_used_gb"),
                lines("GPU utilisation (%)", "ts/gpu_util"),
                lines("PCIe traffic (MB/s)", "ts/pcie_tx_mb_s", "ts/pcie_rx_mb_s"),
                lines("GPU power (W)", "ts/gpu_power_w"),
                lines("CPU util (%) / RAM (GB)", "ts/cpu_util", "ts/ram_used_gb"),
            ]),
        ],
    )
    view.save()
    print("workspace:", view.url)


if __name__ == "__main__":
    main()
