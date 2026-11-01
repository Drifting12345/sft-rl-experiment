#!/usr/bin/env bash
set -euo pipefail
algorithm="$1"
shift
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
zfy_dir="$(dirname -- "$repo_dir")"
cd "$repo_dir"
model_path="${MODEL_PATH:-$zfy_dir/models/Qwen3-0.6B}"
data_dir="${DATA_DIR:-$zfy_dir/datasets/igsm-rl}"
case "$algorithm" in
  grpo) algorithm_args=(algorithm.adv_estimator=grpo critic.enable=false actor_rollout_ref.rollout.n=4) ;;
  ppo) algorithm_args=(algorithm.adv_estimator=gae critic.enable=true actor_rollout_ref.rollout.n=1) ;;
  *) echo "Select grpo or ppo" >&2; exit 2 ;;
esac
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
# Extra arguments are Hydra overrides; --cfg job prints config without training.
exec python -m verl.trainer.main_ppo \
  --config-path "$repo_dir/examples/igsm/config" --config-name igsm \
  "actor_rollout_ref.model.path=$model_path" "critic.model.path=$model_path" \
  "data.train_files=$data_dir/train.parquet" "data.val_files=$data_dir/val.parquet" \
  "trainer.experiment_name=igsm-$algorithm" \
  "trainer.default_local_dir=$zfy_dir/checkpoints/igsm-$algorithm" \
  "${algorithm_args[@]}" "$@"
