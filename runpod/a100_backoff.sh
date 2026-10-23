#!/usr/bin/env bash
# Scheduler backoff fix (--sched-backoff preempt) on the 4090-equal memory budget, 3 repeats each.
# Both the co-optimizer and the independent controllers get the fix, so the coordination ablation
# stays like-for-like. Waits for a100_final.sh to finish first (one GPU).
#   setsid nohup bash runpod/a100_backoff.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/IITJ
while pgrep -f "runpod/a100_final.sh" > /dev/null; do sleep 60; done
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
run 6a a100-22gb ours-cooptimizer-preempt-backoff --repeats 3
run 6b a100-22gb ours-independent-controllers-preempt-backoff --repeats 3
python -m bench.tracking --index >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) QUEUE DONE" >> $LOG
wait
