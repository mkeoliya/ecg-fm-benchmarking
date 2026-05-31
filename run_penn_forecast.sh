#!/bin/bash
#SBATCH --job-name=penn_forecast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"

DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/processed}"
PENN_DATA_DIR="${PENN_DATA_DIR:-${DATASET_DIR}/penn_forecast}"
PENN_SPLIT_DIR="${PENN_SPLIT_DIR:-/srv/shared_home/common-data/arpa-h/ca/melp_split/penn_forecast}"

PENN_INTERVAL_CSVS="${PENN_INTERVAL_CSVS:-${PENN_SPLIT_DIR}/penn_forecast_train.csv,${PENN_SPLIT_DIR}/penn_forecast_val.csv,${PENN_SPLIT_DIR}/penn_forecast_test.csv}"

MODEL="${MODEL:-cpc}"
EVAL_MODE="${EVAL_MODE:-finetuning_linear}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
EPOCHS="${EPOCHS:-100}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-0}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-auto}"

LOOKBACK_SEC="${LOOKBACK_SEC:-1800}"
INPUT_SIZE="${INPUT_SIZE:-10}"
STRIDE_FRACTION="${STRIDE_FRACTION:-1}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${BASE_DIR}/models}"
CPC_CONFIG="${CPC_CONFIG:-${CHECKPOINTS_DIR}/ECG-CPC Checkpoint/config_last_11597276_ckpt.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/output/penn_forecast/outputs}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${BASE_DIR}/output/penn_forecast/predictions}"
LOGS_DIR="${LOGS_DIR:-${BASE_DIR}/output/penn_forecast/logs}"

usage() {
    cat <<USAGE
Usage: $0 [options] [-- extra main_lite.py args]

Options:
  --model MODEL                 Model architecture (default: cpc)
  --eval-mode MODE              finetuning_linear, finetuning_nonlinear, frozen, linear (default: finetuning_linear)
  --epochs N                    Training epochs (default: 100)
  --batch-size N                Batch size (default: 16)
  --num-workers N               DataLoader workers (default: SLURM_CPUS_PER_TASK or 8)
  --lr FLOAT                    Learning rate (default: 0.0005)
  --lookback-sec SEC            Window before abnormal_start (default: 1800)
  --input-size SEC              Example length in seconds (default: 10)
  --stride-fraction FLOAT       Example stride as fraction of input size (default: 1)
  --penn-data-dir PATH          Processed Penn memmap directory
  --penn-split-dir PATH         Directory containing penn_forecast_{train,val,test}.csv
  --interval-csvs CSV[,CSV...]  Explicit interval CSV list
  --cpc-config PATH             CPC config/checkpoint YAML
  --checkpoint-monitor NAME     Metric for best checkpoint, e.g. macro_auprc_agg_val0
  --checkpoint-mode MODE        auto, min, or max (default: auto)

Environment variables with the same uppercase names may also be used.
USAGE
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --eval-mode) EVAL_MODE="$2"; shift 2 ;;
        --eval-mode=*) EVAL_MODE="${1#*=}"; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --epochs=*) EPOCHS="${1#*=}"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --num-workers=*) NUM_WORKERS="${1#*=}"; shift ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --lr=*) LEARNING_RATE="${1#*=}"; shift ;;
        --lookback-sec) LOOKBACK_SEC="$2"; shift 2 ;;
        --lookback-sec=*) LOOKBACK_SEC="${1#*=}"; shift ;;
        --input-size) INPUT_SIZE="$2"; shift 2 ;;
        --input-size=*) INPUT_SIZE="${1#*=}"; shift ;;
        --stride-fraction) STRIDE_FRACTION="$2"; shift 2 ;;
        --stride-fraction=*) STRIDE_FRACTION="${1#*=}"; shift ;;
        --penn-data-dir) PENN_DATA_DIR="$2"; shift 2 ;;
        --penn-data-dir=*) PENN_DATA_DIR="${1#*=}"; shift ;;
        --penn-split-dir)
            PENN_SPLIT_DIR="$2"
            PENN_INTERVAL_CSVS="${PENN_SPLIT_DIR}/penn_forecast_train.csv,${PENN_SPLIT_DIR}/penn_forecast_val.csv,${PENN_SPLIT_DIR}/penn_forecast_test.csv"
            shift 2
            ;;
        --penn-split-dir=*)
            PENN_SPLIT_DIR="${1#*=}"
            PENN_INTERVAL_CSVS="${PENN_SPLIT_DIR}/penn_forecast_train.csv,${PENN_SPLIT_DIR}/penn_forecast_val.csv,${PENN_SPLIT_DIR}/penn_forecast_test.csv"
            shift
            ;;
        --interval-csvs) PENN_INTERVAL_CSVS="$2"; shift 2 ;;
        --interval-csvs=*) PENN_INTERVAL_CSVS="${1#*=}"; shift ;;
        --cpc-config) CPC_CONFIG="$2"; shift 2 ;;
        --cpc-config=*) CPC_CONFIG="${1#*=}"; shift ;;
        --checkpoint-monitor) CHECKPOINT_MONITOR="$2"; shift 2 ;;
        --checkpoint-monitor=*) CHECKPOINT_MONITOR="${1#*=}"; shift ;;
        --checkpoint-mode) CHECKPOINT_MODE="$2"; shift 2 ;;
        --checkpoint-mode=*) CHECKPOINT_MODE="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$MODEL" != "cpc" ]]; then
    echo "Error: run_penn_forecast.sh currently configures CPC defaults only. Got MODEL=$MODEL."
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load hpc-env/13.1 CUDA/12.4.0 Anaconda3 git || true
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lightning3" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate lightning3
    else
        echo "Warning: conda command not found; continuing with current Python environment."
    fi
fi

mkdir -p "${OUTPUT_DIR}/${MODEL}" "${PREDICTIONS_DIR}/${MODEL}" "${LOGS_DIR}/${MODEL}"

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="${LOGS_DIR}/${MODEL}/penn_forecast_${JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Penn Forecast run"
echo "  data: ${PENN_DATA_DIR}"
echo "  intervals: ${PENN_INTERVAL_CSVS}"
echo "  lookback_sec: ${LOOKBACK_SEC}"
echo "  input_size: ${INPUT_SIZE}"
echo "  stride_fraction: ${STRIDE_FRACTION}"
echo "  eval_mode: ${EVAL_MODE}"
echo "  num_workers: ${NUM_WORKERS}"
echo "  cpc_config: ${CPC_CONFIG}"
echo "  checkpoint_monitor: ${CHECKPOINT_MONITOR:-default}"
echo "  checkpoint_mode: ${CHECKPOINT_MODE}"

CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINT_MONITOR" ]]; then
    CHECKPOINT_ARGS+=(--checkpoint-monitor "$CHECKPOINT_MONITOR")
fi
CHECKPOINT_ARGS+=(--checkpoint-mode "$CHECKPOINT_MODE")

python "${BASE_DIR}/code/main_lite.py" \
    --data "${PENN_DATA_DIR}" \
    --fs-data 250 \
    --finetune-dataset penn_forecast \
    --data-backend npz \
    --architecture cpc \
    --input-size "${INPUT_SIZE}" \
    --fs-model 240 \
    --input-channels 12 \
    --precision 32 \
    --pretrained "${CPC_CONFIG}" \
    --inference-interval-csv "${PENN_INTERVAL_CSVS}" \
    --inference-interval-splits all \
    --inference-interval-lookback-sec "${LOOKBACK_SEC}" \
    --chunkify-train \
    --chunk-length-train 1 \
    --stride-fraction-train "${STRIDE_FRACTION}" \
    --stride-fraction-valtest "${STRIDE_FRACTION}" \
    --epochs "${EPOCHS}" \
    --modality ecg \
    --lr "${LEARNING_RATE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --finetune \
    --eval-mode "${EVAL_MODE}" \
    --output-path "${OUTPUT_DIR}/${MODEL}" \
    --prediction-path "${PREDICTIONS_DIR}/${MODEL}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
