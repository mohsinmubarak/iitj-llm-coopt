#!/usr/bin/env bash
# Re-run the speculative core configs after the draft-pool fix (gated on the first one), then the
# final 4090 block (smoke tests + overload stress).
#   setsid nohup bash runpod/core_v3b.sh > /dev/null 2>&1 < /dev/null &
cd /workspace/IITJ
LOG=results/rtx4090/core_v3b.log
bash bench/queue.sh rtx4090 ours-static-spec-gamma4
if grep -A12 "START ours-static-spec-gamma4" results/rtx4090/queue.log | grep -q "server died\|Traceback"; then
  echo "$(date -u +%T) ours-static-spec-gamma4 crashed again: stopping" >> $LOG
  exit 1
fi
echo "$(date -u +%T) gate passed: running the rest" >> $LOG
bash bench/queue.sh rtx4090 ours-cooptimizer ours-independent-controllers ours-nightjar-like vllm-adaptive-spec-schedule
bash runpod/final_4090.sh
echo "$(date -u +%T) CORE V3B AND FINAL BLOCK DONE" >> $LOG
