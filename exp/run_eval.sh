#!/usr/bin/env bash
# Run in the foreground; use nohup when launching a persistent experiment.
qz_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
qz_name="${1:-}"
if [[ ! "$qz_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Usage: bash run_eval.sh RUN_NAME [evaluation options]\n' >&2
    exit 2
fi
shift
mkdir -p "$qz_root/exp/logs" || exit 1
qz_status_file="$qz_root/exp/logs/$qz_name.status"
trap 'qz_exit=$?; printf "%s\n" "$qz_exit" > "$qz_status_file"' EXIT
source "$qz_root/verl/setup/activate.sh" || exit $?
cd "$qz_root/exp" || exit $?
printf 'RUN_NAME=%s\nSTART_UTC=%s\n' "$qz_name" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
python -u -B "$qz_root/exp/eval_qwen3_igsm.py" \
    --output-dir "$qz_root/exp/runs/$qz_name" "$@"
exit $?
