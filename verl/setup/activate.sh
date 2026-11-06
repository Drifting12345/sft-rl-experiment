# Source this file to enter the independent Conda training environment.
source /opt/miniconda3/etc/profile.d/conda.sh || return $?
conda activate /root/sft-rl-experiment/.conda/envs/verl-train || return $?
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export PIP_CACHE_DIR=/root/sft-rl-experiment/.cache/pip-verl
export HF_HOME=/root/sft-rl-experiment/.cache/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/sft-rl-experiment/verl
cd /root/sft-rl-experiment/verl || return $?
