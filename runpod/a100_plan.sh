#!/usr/bin/env bash
# The A100 experiment plan, most important first, so stopping early still leaves the core results.
# Resumable: finished results are skipped, so re-launching continues where it stopped.
#   setsid nohup bash runpod/a100_plan.sh > /dev/null 2>&1 < /dev/null &
#   tail -f results/a100/queue.log
set -uo pipefail
cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ \
       WANDB_SILENT=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
mkdir -p results/a100 results/a100-22gb
LOG=results/a100/queue.log
sed -i '/QUEUE DONE/d' $LOG 2>/dev/null || true
python -m bench.podlog --profile a100 --until-done > results/a100/podlog.out 2>&1 &

# run <step> <profile> <config> [bench.experiment args...]
run() {
  local step=$1 prof=$2 cfg=$3; shift 3
  echo "=== $(date -u +%FT%TZ) START [$step] $cfg on $prof $*" >> $LOG
  python -m bench.experiment bench/configs/$cfg.yaml --profile $prof "$@" 2>&1 \
    | grep --line-buffered -E "^\[|Error|error:|Traceback" >> $LOG
  echo "=== $(date -u +%FT%TZ) END [$step] $cfg (exit ${PIPESTATUS[0]})" >> $LOG
}
CORE="vllm-default ours-cooptimizer ours-independent-controllers"
PRESSURE="--only chat --only phase-shift"   # the workloads where the controllers separated on the 4090

# Step 1a: core comparison under 4090-equal memory, first repeat of each (the thesis claim, early)
for c in $CORE; do run 1a a100-22gb $c --repeats 1; done
# Step 1b: repeats 2 and 3 of the same (mean +- std)
for c in $CORE; do run 1b a100-22gb $c --repeats 3; done
# Step 1c: prior-work baselines under equal memory, 1 repeat
for c in ours-nightjar-like vllm-adaptive-spec-schedule; do run 1c a100-22gb $c --repeats 1; done
# Step 1d: natural A100 memory (80 GB), 1 repeat: does coordination still help, or at least not hurt?
for c in $CORE ours-nightjar-like vllm-adaptive-spec-schedule; do run 1d a100 $c --repeats 1; done

# Step 2a: controller ablations under equal memory (which coupling produces the gain)
for c in ours-gamma-controller-only ours-kv-controller-only ours-kv-controller-only-no-elastic \
         ours-scheduler-controller-only ours-gamma-and-kv ours-gamma-and-scheduler ours-kv-and-scheduler \
         ours-cooptimizer-no-elastic ours-independent-controllers-no-elastic; do
  run 2a a100-22gb $c --repeats 1 $PRESSURE
done
# Step 2b: a middle memory point (0.40 x 80 GB = 32 GiB, ~14 GiB KV): gain vs memory pressure curve
for c in ours-cooptimizer ours-independent-controllers vllm-default; do
  run 2b a100 $c --repeats 1 $PRESSURE --tag gpu-memory-32gb --extra-args "--gpu-memory-utilization 0.40"
done
# Step 2c: context-length scaling of stock vLLM up to 32K (the 4090 stopped at 16K)
run 2c a100 vllm-context-length-sweep --repeats 1

# Step 3a: fixed draft lengths in vLLM (no single gamma is best at every load), chat only
for c in vllm-draft-spec-gamma1 vllm-draft-spec-gamma2 vllm-draft-spec-gamma4 vllm-draft-spec-gamma6 vllm-draft-spec-gamma8; do
  run 3a a100-22gb $c --repeats 1 --only chat
done
# Step 3b: 2x overload stress under equal memory
for c in vllm-default ours-cooptimizer; do
  f=bench/configs/$c-overload.yaml
  sed -e "s/^name: $c$/name: $c-overload/" bench/configs/$c.yaml | awk '
        /^runs:/ {print; print "  - workload: mixed-bursty\n    label: mixed-bursty_2x\n    arrivals: file\n    time_scale: 0.5\n  - workload: phase-shift\n    label: phase-shift_2x\n    arrivals: file\n    time_scale: 0.5"; skip=1; next}
        skip && /^  - workload:/ {next} skip && /^    / {next} {skip=0; print}' > $f
  run 3b a100-22gb $c-overload --repeats 1
done
# Step 3c: vLLM CPU KV offload (reported in its own section, not in the headline tables)
run 3c a100-22gb vllm-cpu-kv-offload --repeats 1

python -m bench.tracking --index >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) QUEUE DONE" >> $LOG
wait
