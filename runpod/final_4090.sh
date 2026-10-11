#!/usr/bin/env bash
# Last 4090 block before the A100: (1) smoke-test every config not yet run on the final engine,
# (2) overload stress test (2x arrival rate) for the zero-OOM claim. Uses the descriptive names.
#   setsid nohup bash runpod/final_4090.sh > /dev/null 2>&1 < /dev/null &
cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ \
       WANDB_SILENT=true PYTHONUNBUFFERED=1
LOG=results/rtx4090/final_4090.log
echo "=== $(date -u +%FT%TZ) smoke tests (20 requests per workload)" >> $LOG
for c in ours-gamma-controller-only ours-kv-controller-only ours-kv-controller-only-no-elastic \
         ours-scheduler-controller-only ours-cooptimizer-no-elastic ours-independent-controllers-no-elastic \
         ours-gamma-and-kv ours-gamma-and-scheduler ours-kv-and-scheduler \
         vllm-draft-spec-gamma6 vllm-draft-spec-gamma8; do
  out=$(timeout 1800 python -m bench.experiment bench/configs/$c.yaml --profile rtx4090 --limit 20 --repeats 1 \
        --tag smoke-test --force 2>&1)
  ok=$(echo "$out" | grep -c "rep=0: 20/20 ok\|rep=0: [0-9]*/[0-9]* ok")
  bad=$(echo "$out" | grep -c "Traceback\|server died\|exited during startup")
  echo "$(date -u +%T) $c: $ok runs ok, $bad errors" >> $LOG
  [ "$bad" != "0" ] && echo "$out" | grep -A3 "Error\|error" | tail -6 >> $LOG
done

echo "=== $(date -u +%FT%TZ) overload stress (2x arrival rate)" >> $LOG
for c in vllm-default vllm-cpu-kv-offload ours-cooptimizer; do
  f=bench/configs/$c-overload.yaml
  sed -e "s/^name: $c$/name: $c-overload/" bench/configs/$c.yaml | awk '
        /^runs:/ {print; print "  - workload: mixed-bursty\n    label: mixed-bursty_2x\n    arrivals: file\n    time_scale: 0.5\n  - workload: phase-shift\n    label: phase-shift_2x\n    arrivals: file\n    time_scale: 0.5"; skip=1; next}
        skip && /^  - workload:/ {next} skip && /^    / {next} {skip=0; print}' > $f
  out=$(timeout 3600 python -m bench.experiment $f --profile rtx4090 --repeats 1 2>&1)
  echo "$out" | grep "rep=0" | sed "s/^/  /" >> $LOG
  echo "$(date -u +%T) $c-overload: errors $(echo "$out" | grep -c "Traceback\|server died")" >> $LOG
done
echo "=== $(date -u +%FT%TZ) FINAL 4090 DONE" >> $LOG
