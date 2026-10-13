#!/usr/bin/env bash
# A100 validation before any benchmark: unit tests, engine correctness vs HF (8B), CUDA graphs, elastic drafter.
#   setsid nohup bash runpod/validate_a100.sh > /dev/null 2>&1 < /dev/null &
cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true OMP_NUM_THREADS=8
mkdir -p results/a100
LOG=results/a100/validation.log
run() { echo "=== $(date -u +%T) $*" >> $LOG; "$@" >> $LOG 2>&1; echo "=== exit $?" >> $LOG; }
run python -m pytest -q tests/test_units.py
run python -m tests.m4_correctness --model unsloth/Meta-Llama-3.1-8B-Instruct --n 16 --new 64 --kv-gb 8 --profile a100
run python -m tests.graph_check --model unsloth/Meta-Llama-3.1-8B-Instruct
run python -m tests.elastic_check
echo "=== VALIDATION DONE" >> $LOG
