#!/usr/bin/env bash
set -euo pipefail
rl_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qz_root="$(cd -- "$rl_dir/../.." && pwd)"
source "$qz_root/verl/setup/activate.sh"
cd "$rl_dir"
export PYTHONPATH="$rl_dir:$qz_root/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0 VLLM_NO_USAGE_STATS=1
export WANDB_MODE=disabled VERL_DISABLE_FLASH_ATTN_CE=1 RAY_ADDRESS=local
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python -B "$rl_dir/train_rl.py" "$@"
