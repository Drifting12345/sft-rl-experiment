# Source this file: source /root/sft-rl-experiment/iGSM/setup/activate.sh
source /opt/miniconda3/etc/profile.d/conda.sh || return $?
conda activate /root/sft-rl-experiment/.conda/envs/igsm-gpu || return $?
# The container constrains its system torch; do not inherit that constraint here.
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HOME=/root/sft-rl-experiment/.cache/huggingface
export PIP_CACHE_DIR=/root/sft-rl-experiment/.cache/pip
export MPLCONFIGDIR=/root/sft-rl-experiment/.cache/matplotlib
cd /root/sft-rl-experiment/iGSM || return $?
