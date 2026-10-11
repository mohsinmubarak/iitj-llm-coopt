#!/usr/bin/env bash
# One-shot: once the core comparison (C6_el) has finished, stop the running queue and restart it
# with the vLLM no-prefix-caching baseline first. Finished runs are skipped on restart.
#   setsid nohup bash runpod/reorder_after_core.sh > /dev/null 2>&1 < /dev/null &
cd /workspace/IITJ
LOG=results/rtx4090/queue.log
until grep -q "END C6_el" $LOG; do sleep 20; done
sleep 5
Q=$(pgrep -f "^bash bench/queue.sh rtx4090 C0_nospec")
X=$(pgrep -f "^python -m bench.experiment")
S=$(pgrep -f "^python -m engine.server|vllm serve" | head -1)
[ -n "$Q" ] && kill -TERM $Q
[ -n "$X" ] && kill -TERM $X
if [ -n "$S" ]; then
  kill -INT -$S 2>/dev/null || kill -INT $S
  for i in $(seq 1 24); do kill -0 $S 2>/dev/null || break; sleep 5; done
  kill -0 $S 2>/dev/null && kill -KILL -$S 2>/dev/null
fi
pkill -TERM -f "^python -m bench.podlog" 2>/dev/null
echo "=== $(date -u +%FT%TZ) reordered: A2_nopc first (prefix-caching test)" >> $LOG
sleep 10
bash bench/queue.sh rtx4090 A2_nopc B2 C4 C6 C2_el C1 C2 C3 C5_gk C5_gs C5_ks B1D_g6 B1D_g8
