#!/usr/bin/env bash
# Final A100 runs after the 13 Oct queues (a100_plan.sh, a100_fix.sh), all under the 4090-equal
# memory budget (a100-22gb):
#  (5a) our engine without speculation, 3 repeats: separates engine efficiency from control
#       (stock vLLM wins goodput under load on the A100; this says whether the engine or the policy is why)
#  (5b) repeats 2 and 3 of the prior-work baselines, so every comparison in the main table is mean +- std.
#       vllm-adaptive-spec-schedule keeps a100_fix.sh's 0.285 memory override, as its rep0 did.
# Resumable: finished results are skipped.
#   setsid nohup bash runpod/a100_final.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ \
       WANDB_SILENT=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
LOG=results/a100/queue.log
sed -i '/QUEUE DONE/d' $LOG
python -m bench.podlog --profile a100 --until-done > results/a100/podlog.out 2>&1 &
run() {
  local step=$1 prof=$2 cfg=$3; shift 3
  echo "=== $(date -u +%FT%TZ) START [$step] $cfg on $prof $*" >> $LOG
  python -m bench.experiment bench/configs/$cfg.yaml --profile $prof "$@" 2>&1 \
    | grep --line-buffered -E "^\[|Error|error:|Traceback" >> $LOG
  echo "=== $(date -u +%FT%TZ) END [$step] $cfg (exit ${PIPESTATUS[0]})" >> $LOG
}
run 5a a100-22gb ours-static-no-spec --repeats 3
run 5b a100-22gb ours-nightjar-like --repeats 3
run 5b a100-22gb vllm-adaptive-spec-schedule --repeats 3 --extra-args "--gpu-memory-utilization 0.285"
python -m bench.tracking --index >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) QUEUE DONE" >> $LOG
wait
