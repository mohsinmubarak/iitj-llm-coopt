cd /workspace/IITJ
until grep -q "CORE V3B AND FINAL BLOCK DONE" results/rtx4090/core_v3b.log 2>/dev/null; do sleep 30; done
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true
LOG=results/rtx4090/final_4090.log
for c in vllm-draft-spec-gamma6 vllm-draft-spec-gamma8; do
  out=$(timeout 1800 python -m bench.experiment bench/configs/$c.yaml --profile rtx4090 --limit 20 --repeats 1 --tag smoke-test --force 2>&1)
  echo "$(date -u +%T) retest $c: $(echo "$out" | grep -c " ok |") runs ok, $(echo "$out" | grep -c "Traceback\|server died\|exited during startup") errors" >> $LOG
done
echo "=== GAMMA68 RETEST DONE" >> $LOG
