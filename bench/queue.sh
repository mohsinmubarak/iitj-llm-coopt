#!/usr/bin/env bash
# Run experiment configs back to back on the pod. A failing config is logged and skipped;
# finished runs are skipped on re-launch, so the same command resumes an interrupted queue.
# A pod-level W&B run (bench/podlog.py) records system metrics and uploads the logs live.
#
#   setsid nohup bash bench/queue.sh rtx4090 A2 A3 B1 > /dev/null 2>&1 < /dev/null &
#   tail -f results/rtx4090/queue.log
set -uo pipefail
PROFILE=$1; shift
cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ \
       WANDB_SILENT=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p results/$PROFILE
LOG=results/$PROFILE/queue.log
sed -i '/QUEUE DONE/d' $LOG 2>/dev/null || true   # a resumed queue must not stop the new pod logger

python -m bench.podlog --profile $PROFILE --until-done > results/$PROFILE/podlog.out 2>&1 &

for NAME in "$@"; do
  echo "=== $(date -u +%FT%TZ) START $NAME" >> $LOG
  python -m bench.experiment bench/configs/$NAME.yaml --profile $PROFILE 2>&1 \
    | grep --line-buffered -E "^\[|Error|error:|Traceback" >> $LOG
  echo "=== $(date -u +%FT%TZ) END $NAME (exit ${PIPESTATUS[0]})" >> $LOG
done
echo "=== $(date -u +%FT%TZ) QUEUE DONE" >> $LOG
wait
