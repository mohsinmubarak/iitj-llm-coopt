#!/usr/bin/env bash
# Follow-up to a100_plan.sh, fixing two problems in its results:
#  (1) vllm-adaptive-spec-schedule and vllm-draft-spec-gamma6/8 carry override_args
#      --gpu-memory-utilization 0.95 (a 4090 start-up exception), which beat the a100-22gb profile's
#      0.28 and gave them ~76 GB. Re-run at 0.285: the same +3% exception, applied to the 22 GiB budget.
#  (2) the ablation configs defined only some workloads, so the chat/phase-shift filter skipped them.
#   setsid nohup bash runpod/a100_fix.sh > /dev/null 2>&1 < /dev/null &
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
OLD=results/a100-22gb/_superseded/memory_override_76gb
mkdir -p $OLD
for c in vllm-adaptive-spec-schedule vllm-draft-spec-gamma6 vllm-draft-spec-gamma8; do
  [ -d results/a100-22gb/$c ] && mv results/a100-22gb/$c $OLD/
done
EQ="--gpu-memory-utilization 0.285"
run 4a a100-22gb vllm-adaptive-spec-schedule --repeats 1 --extra-args "$EQ"
for c in vllm-draft-spec-gamma6 vllm-draft-spec-gamma8; do run 4a a100-22gb $c --repeats 1 --only chat --extra-args "$EQ"; done
for c in ours-gamma-and-kv ours-gamma-and-scheduler ours-kv-and-scheduler \
         ours-gamma-controller-only ours-kv-controller-only-no-elastic ours-scheduler-controller-only; do
  run 4b a100-22gb $c --repeats 1 --only chat --only phase-shift
done
python -m bench.tracking --index >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) QUEUE DONE" >> $LOG
wait
