#!/usr/bin/env bash
# One-shot: once config $1 has finished, stop the running queue cleanly (queue, experiment,
# server, pod logger) and mark the queue done.
#   setsid nohup bash runpod/stop_after.sh C6 > /dev/null 2>&1 < /dev/null &
NAME=$1
cd /workspace/IITJ
LOG=results/rtx4090/queue.log
until grep -q "END $NAME " $LOG; do sleep 15; done
Q=$(pgrep -f "^bash bench/queue.sh")
X=$(pgrep -f "^python -m bench.experiment")
S=$(pgrep -f "^python -m engine.server|vllm serve" | head -1)
[ -n "$Q" ] && kill -TERM $Q
[ -n "$X" ] && kill -TERM $X
if [ -n "$S" ]; then
  kill -INT -$S 2>/dev/null || kill -INT $S
  for i in $(seq 1 24); do kill -0 $S 2>/dev/null || break; sleep 5; done
  kill -0 $S 2>/dev/null && kill -KILL -$S 2>/dev/null
fi
echo "=== $(date -u +%FT%TZ) stopped after $NAME (engine fixes next)" >> $LOG
echo "=== $(date -u +%FT%TZ) QUEUE DONE" >> $LOG
