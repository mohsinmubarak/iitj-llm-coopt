cd /workspace/IITJ
set -a; . /workspace/.env; set +a
. /root/venv/bin/activate
export HF_HOME=/workspace/hf_cache PYTHONPATH=/workspace/IITJ WANDB_SILENT=true PYTHONUNBUFFERED=1
python -m tests.elastic_check > results/rtx4090/elastic_check_pc.log 2>&1
rm -rf results/rtx4090/C0_nospec_pc
python -m bench.experiment bench/configs/C0_nospec.yaml --profile rtx4090 --repeats 1 --only W4 --tag pc > results/rtx4090/validate_pc.out 2>&1
echo VALIDATE_DONE >> results/rtx4090/validate_pc.out
