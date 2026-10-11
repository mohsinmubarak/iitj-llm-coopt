#!/usr/bin/env bash
# After the quality runs: validate memory-elastic speculation on the GPU, and only if it passes run
# the elastic ablation and the fixed gamma-sweep configs.
#   setsid nohup bash runpod/after_quality_elastic.sh rtx4090 > /dev/null 2>&1 < /dev/null &
PROFILE=${1:-rtx4090}
cd /workspace/IITJ
until grep -q "QUALITY DONE" results/$PROFILE/Q/quality.log 2>/dev/null; do sleep 30; done
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ \
       WANDB_SILENT=true PYTHONUNBUFFERED=1
LOG=results/$PROFILE/elastic_check.log
python -m tests.elastic_check > $LOG 2>&1
if grep -q "ELASTIC CHECK PASS" $LOG; then
  bash bench/queue.sh $PROFILE C4_el C6_el C2_el B1D_g6 B1D_g8
else
  echo "=== $(date -u +%FT%TZ) elastic check FAILED: elastic ablation not started (see $LOG)" >> results/$PROFILE/queue.log
fi
