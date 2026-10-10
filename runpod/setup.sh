#!/usr/bin/env bash
# RunPod pod bootstrap (~1 min). Run once per new pod:  bash /workspace/IITJ/runpod/setup.sh
#
# Layout:
#   /root/venv          Python env on the container disk (fast; rebuilt per pod, uv makes it ~45 s)
#   /workspace/hf_cache models + datasets on the network volume (persist across pods)
#   /workspace/IITJ     code + results on the network volume
# Installing the venv onto the network volume itself is ~30x slower (many small files), so don't.
# Container disk must be >= 40 GB (vLLM + torch ~ 10 GB, plus uv cache during install).
set -euo pipefail

WS=/workspace
export HF_HOME=$WS/hf_cache
export VLLM_CACHE_ROOT=$WS/vllm_cache
mkdir -p "$HF_HOME" $WS/IITJ/results

grep -q "IITJ env" ~/.bashrc || cat >> ~/.bashrc <<EOF
# --- IITJ env ---
export HF_HOME=$HF_HOME
export VLLM_CACHE_ROOT=$WS/vllm_cache
export PATH=/root/.local/bin:\$PATH
export PYTHONPATH=$WS/IITJ:\${PYTHONPATH:-}
[ -f $WS/.env ] && set -a && . $WS/.env && set +a
[ -f /root/venv/bin/activate ] && . /root/venv/bin/activate
cd $WS/IITJ 2>/dev/null || true
EOF
[ -f $WS/.env ] && set -a && . $WS/.env && set +a

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
[ -d /root/venv ] || uv venv -q --python 3.12 /root/venv
. /root/venv/bin/activate
# Paged attention for our engine comes from vLLM's bundled kernel (vllm.vllm_flash_attn), so
# no separate flash-attn build is needed.
uv pip install -q "vllm==0.31.0" transformers accelerate nvidia-ml-py psutil pandas matplotlib \
    datasets pyyaml tqdm pytest aiohttp wandb wandb-workspaces
uv cache clean -q

python - <<'PY' > $WS/IITJ/results/env_$(hostname).json
import json, platform, subprocess, torch, vllm, transformers
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "vllm": vllm.__version__,
                  "transformers": transformers.__version__, "gpu": gpu}, indent=2))
PY
cat $WS/IITJ/results/env_$(hostname).json

# Models (ungated unsloth mirrors of the official Meta weights); no-op if already cached
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
PY
echo "Setup done. Open a new terminal (or 'source ~/.bashrc')."
