# Source this file: source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM/setup/activate.sh
source /inspire/hdd/project/deepanalysis/guitao-25013/miniconda3/etc/profile.d/conda.sh || return $?
conda activate /inspire/hdd/project/deepanalysis/guitao-25013/zfy/.conda/envs/igsm-gpu || return $?
# The container constrains its system torch; do not inherit that constraint here.
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export HF_HOME=/inspire/hdd/project/deepanalysis/guitao-25013/zfy/.cache/huggingface
export PIP_CACHE_DIR=/inspire/hdd/project/deepanalysis/guitao-25013/zfy/.cache/pip
export MPLCONFIGDIR=/inspire/hdd/project/deepanalysis/guitao-25013/zfy/.cache/matplotlib
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM || return $?
