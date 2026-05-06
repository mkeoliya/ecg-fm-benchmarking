#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/camel_logs}"

mkdir -p "$LOG_DIR"

TASKS=(mimic)
GPUS=(9)

i=0
for task in "${TASKS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}

  CUDA_VISIBLE_DEVICES="$gpu" bash "$RUN_SCRIPT" --dataset "$task" \
    > "${LOG_DIR}/${task}.log" 2>&1 &

  i=$((i + 1))

  if (( i % ${#GPUS[@]} == 0 )); then
    wait
  fi
done

wait
