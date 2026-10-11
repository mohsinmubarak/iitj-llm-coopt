#!/usr/bin/env bash
# Run on the laptop. Waits for the pod's queue to finish, then switches everything from experiment
# codes to descriptive names (bench/names.py): code -> pod, files/data/configs renamed on the pod,
# W&B runs renamed, index rebuilt, and the laptop's results/data/configs re-pulled under the new names.
set -euo pipefail
cd "$(dirname "$0")/.."
until ssh -o ConnectTimeout=20 runpod 'tail -1 /workspace/IITJ/results/rtx4090/queue.log | grep -q "QUEUE DONE"'; do sleep 120; done
sleep 60
# 1. final pull of anything written under the old names
bash runpod/sync_from_pod.sh > /dev/null 2>&1 || true
# 2. push the renamed code (everything except configs and data, which the pod renames itself)
for f in bench/*.py engine/*.py tests/*.py eval/*.py analysis/*.py runpod/*.sh; do
  scp -q "$f" "runpod:/workspace/IITJ/$f"
done
# 3. rename on the pod + W&B + index
ssh runpod 'set -a; . /workspace/.env; set +a; cd /workspace/IITJ && export PYTHONPATH=. && \
  /root/venv/bin/python -m bench.rename && \
  /root/venv/bin/python -m bench.wandb_rename --project iitj-coopt-rtx4090 && \
  /root/venv/bin/python -m bench.tracking --index' 2>&1 | grep -v -i "pynvml\|deprecat\|Loaded cred" | tail -5
# 4. laptop: replace results, data and configs with the pod's renamed copies
rm -rf results/rtx4090 data/*.jsonl bench/configs/*.yaml results/index.jsonl
ssh runpod "cd /workspace/IITJ && tar cf - --exclude='*/latest-run' results/rtx4090 results/index.jsonl data bench/configs" | tar xf - -C .
rm -f .last_pod_sync
bash runpod/sync_from_pod.sh --verify
echo "FINALIZE_RENAME_DONE"
