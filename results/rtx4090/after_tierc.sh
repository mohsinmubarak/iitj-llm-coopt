#!/usr/bin/env bash
# After the current queue: no-spec baseline, and re-runs of every config that uses the bandit
# gamma controller and/or KV tiering with the bounded draft catch-up fix.
cd /workspace/IITJ
until tail -1 results/rtx4090/queue.log | grep -q "QUEUE DONE"; do sleep 30; done
sleep 30
bash bench/queue.sh rtx4090 C0_nospec C1 C2 C4 C6
