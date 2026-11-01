# Source this file to enter the independent Conda training environment.
source /inspire/hdd/project/deepanalysis/guitao-25013/miniconda3/etc/profile.d/conda.sh || return $?
conda activate /inspire/hdd/project/deepanalysis/guitao-25013/zfy/.conda/envs/verl-train || return $?
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export PIP_CACHE_DIR=/inspire/hdd/project/deepanalysis/guitao-25013/zfy/.cache/pip-verl
export HF_HOME=/inspire/hdd/project/deepanalysis/guitao-25013/zfy/.cache/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl || return $?
