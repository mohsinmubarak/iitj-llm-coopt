#!/usr/bin/env bash
# Runs the quality experiments once the current experiment queue has finished.
#   setsid nohup bash runpod/after_queue_quality.sh rtx4090 > /dev/null 2>&1 < /dev/null &
PROFILE=${1:-rtx4090}
cd /workspace/IITJ
until tail -1 results/$PROFILE/queue.log | grep -q "QUEUE DONE"; do sleep 30; done
sleep 45   # let the pod logger and the last server exit
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true PYTHONUNBUFFERED=1
LOG=results/$PROFILE/quality/quality.log
mkdir -p results/$PROFILE/quality
echo "=== $(date -u +%FT%TZ) Q1 start" >> $LOG
python -m eval.quality q1 --workload prompt-4k --n 32 --profile $PROFILE >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) Q2 start" >> $LOG
python -m eval.quality q2 --n 32 --new 128 --profile $PROFILE >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) QUALITY DONE" >> $LOG
