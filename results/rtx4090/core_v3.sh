cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true PYTHONUNBUFFERED=1 WANDB_PROJECT=iitj-coopt-rtx4090
LOG=results/rtx4090/core_v3.log
python -m tests.m4_correctness --model unsloth/Meta-Llama-3.1-8B-Instruct --n 16 --new 64 --profile rtx4090 > results/rtx4090/M4/m4_8b_v3.log 2>&1
grep -E "teacher_forced_agreement|\"pass\"" results/rtx4090/M4/m4_8b_v3.log >> $LOG
python -m tests.profile_engine --n 200 2>&1 | grep -E "decode-only|total:" >> $LOG
if grep -q "\"pass\": true" results/rtx4090/M4/m4_8b_v3.log; then
  echo "M4 PASS: starting core v3" >> $LOG
  bash bench/queue.sh rtx4090 C0_nospec C0 C4_el C6_el N1 B5
else
  echo "M4 FAILED: core v3 not started" >> $LOG
fi
