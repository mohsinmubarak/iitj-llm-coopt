#!/usr/bin/env bash
# A100 arrival-time calibration: stock vLLM on the bursty schedule at scale 1.0 (tag 'calibration').
# Target, as on the 4090: the mean load sits near capacity and only the bursts overload.
#   setsid nohup bash runpod/calibrate_a100.sh > /dev/null 2>&1 < /dev/null &
cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true OMP_NUM_THREADS=8
python -m bench.experiment bench/configs/vllm-default.yaml --profile a100 --only mixed-bursty --repeats 1 --tag calibration \
  >> results/a100/calibration.log 2>&1
echo "=== CALIBRATION DONE" >> results/a100/calibration.log
