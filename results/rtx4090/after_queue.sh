#!/usr/bin/env bash
# One-shot chain: wait for the current queue, validate the engine (M4), then run the follow-up queue.
cd /workspace/IITJ
until grep -q "QUEUE DONE" results/rtx4090/queue.log; do sleep 30; done
sleep 20  # let the pod logger finish
mv bench/queue.next.sh bench/queue.sh
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true PYTHONUNBUFFERED=1 WANDB_PROJECT=iitj-coopt-rtx4090
mkdir -p results/rtx4090/M4
echo "$(date -u +%FT%TZ) M4 1B start" >> results/rtx4090/after_queue.log
python -m tests.m4_correctness --model unsloth/Llama-3.2-1B-Instruct --n 16 --new 64 > results/rtx4090/M4/m4_1b.log 2>&1
echo "$(date -u +%FT%TZ) M4 1B exit $?" >> results/rtx4090/after_queue.log
python -m tests.m4_correctness --model unsloth/Meta-Llama-3.1-8B-Instruct --n 16 --new 64 > results/rtx4090/M4/m4_8b.log 2>&1
echo "$(date -u +%FT%TZ) M4 8B exit $?" >> results/rtx4090/after_queue.log
bash bench/queue.sh rtx4090 B1D_g6 B1D_g8 B2S_kv4g B2S_kv2p5g B2S_kv3g A2
echo "$(date -u +%FT%TZ) follow-up queue done" >> results/rtx4090/after_queue.log
