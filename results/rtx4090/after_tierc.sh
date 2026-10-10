#!/usr/bin/env bash
# Runs after the current queue: re-run C1 (bandit gamma controller) and C2 (draft catch-up fix),
# and the new no-speculation engine baseline.
cd /workspace/IITJ
until tail -1 results/rtx4090/queue.log | grep -q "QUEUE DONE"; do sleep 30; done
sleep 30
bash bench/queue.sh rtx4090 C0_nospec C1 C2
