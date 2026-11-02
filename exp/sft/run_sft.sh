#!/usr/bin/env bash
set -euo pipefail

sft_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qz_root="$(cd -- "$sft_dir/../.." && pwd)"
source "$qz_root/verl/setup/activate.sh"
cd "$sft_dir"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONDONTWRITEBYTECODE=1
exec python -B "$sft_dir/train_sft.py" "$@"
