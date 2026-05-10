#!/bin/bash
#SBATCH --job-name=cpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --nodelist=
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null

BASE_DIR="${BASE_DIR:-}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
DATASET_DIR="${DATASET_DIR:-}"
CAMEL_PRETRAINED="${CAMEL_PRETRAINED:-/srv/shared_home/common-data/arpa-h/ca/models/m4b/curriculum/stage4/v7/lead_drop/llava_proj_epoch0009.pt}"

EVAL_MODE="linear"      # finetuning_linear, frozen, linear
MODEL="camel"           # ecg_founder, ecg_jepa_multiblock, st_mem, merl_resnet, camel, ecgfm_ked, s4, net1d, cpc, hubert_ecg_base
DATASET="ptb"           # mimic, ptb, ptbxl_all, ptbxl_sub, ptbxl_super, chapman, ningbo, sph, cpsc2018, cpsc_extra, echonext, georgia, code15_diag, zzu_pecg
LEARNING_RATE=0.0005
BATCH_SIZE=16
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/processed}"

usage() {
    echo "Usage: $0 [--dataset DATASET] [--bootstrap-iterations N]"
    echo "Example: $0 --dataset mimic --bootstrap-iterations 1000"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --dataset requires a value."
                usage
                exit 1
            fi
            DATASET="$2"
            shift 2
            ;;
        --dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        --bootstrap-iterations)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --bootstrap-iterations requires a value."
                usage
                exit 1
            fi
            BOOTSTRAP_ITERATIONS="$2"
            shift 2
            ;;
        --bootstrap-iterations=*)
            BOOTSTRAP_ITERATIONS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'."
            usage
            exit 1
            ;;
    esac
done


if [ "$EVAL_MODE" == "finetuning_linear" ]; then
    OUTPUT_DIR="${BASE_DIR}/new_finetuning_linear/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/new_finetuning_linear/predictions"
    LOGS_DIR="${BASE_DIR}/new_finetuning_linear/logs"
elif [ "$EVAL_MODE" == "frozen" ]; then
    OUTPUT_DIR="${BASE_DIR}/frozen/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/frozen/predictions"
    LOGS_DIR="${BASE_DIR}/frozen/logs"
elif [ "$EVAL_MODE" == "linear" ]; then
    OUTPUT_DIR="${BASE_DIR}/linear/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/linear/predictions"
    LOGS_DIR="${BASE_DIR}/linear/logs"
else
    echo "Error: Unknown mode '$EVAL_MODE'. Choose from finetuning_linear, finetuning_nonlinear frozen, or linear."
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load hpc-env/13.1 CUDA/12.4.0 Anaconda3 git
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lightning3" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate lightning3
    else
        echo "Warning: conda command not found; continuing with current Python environment."
    fi
fi

mkdir -p "${LOGS_DIR}/${MODEL}"
mkdir -p "${OUTPUT_DIR}/${MODEL}_${DATASET}"
mkdir -p "${PREDICTIONS_DIR}/${MODEL}"

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="${LOGS_DIR}/${MODEL}/${DATASET}_${JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Special handling per dataset
ARGS_DATASET=()
case $DATASET in
  "ptb")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb"
        "--fs-data 1000"
        "--finetune-dataset ptb"
    )
    ;;
  "ningbo")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ningbo"
        "--fs-data 500"
        "--finetune-dataset ningbo"
    )
    ;;
  "cpsc2018")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/cpsc2018"
        "--fs-data 500"
        "--finetune-dataset cpsc2018"
    )
    ;;
  "cpsc_extra")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/cpsc_extra"
        "--fs-data 500"
        "--finetune-dataset cpsc_extra"
    )
    ;;
  "georgia")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/georgia"
        "--fs-data 500"
        "--finetune-dataset georgia"
    )
    ;;
  "chapman")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/chapman"
        "--fs-data 500"
        "--finetune-dataset chapman"
    )
    ;;
  "sph")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/sph"
        "--fs-data 500"
        "--finetune-dataset sph"
    )
    ;;
  "code15"|"code15_diag")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/code15"
        "--fs-data 400"
        "--finetune-dataset code15_diag"
    )
    ;;
  "ptbxl_super")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb-xl/records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_super"
    )
    ;;
  "ptbxl_sub")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb-xl/records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_sub"
    )
    ;;
  "ptbxl_all")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb-xl/records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_all"
    )
    ;;
  "echonext")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/echonext"
        "--fs-data 250"
        "--finetune-dataset echonext"
    )
    ;;
  "mimic")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/mimic"
        "--fs-data 500"
        "--finetune-dataset mimic"
    )
    ;;
  "zzu_pecg")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/zzu_pecg"
        "--fs-data 500"
        "--finetune-dataset zzu_pecg"
    )
    ;;
  *)
    echo "Error: Unknown dataset '$DATASET'."
    exit 1
    ;;
esac

# Special handling per model
ARGS_MODEL=()
case $MODEL in
  "ecg_founder")
    ARGS_MODEL+=(
        "--architecture ecg_founder"
        "--input-size 2.5"
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_founder/12_lead_ECGFounder.pth"
    )
    ;;
  "ecg_jepa_multiblock")
    ARGS_MODEL+=(
        "--architecture ecg_jepa"
        "--input-size 10" 
        "--fs-model 250"
        "--input-channels 8"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/multiblock_epoch100.pth"
    )
    ;;
  "ecg_jepa_random")
    ARGS_MODEL+=(
        "--architecture ecg_jepa"
        "--input-size 10" 
        "--fs-model 250"
        "--input-channels 8"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/random_epoch100.pth"
    )
    ;;
  "st_mem")
    ARGS_MODEL+=(
        "--architecture st_mem"
        "--input-size 2.4"
        "--fs-model 250"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/st_mem/st_mem_vit_base_full.pth"
    )
    ;;
  "merl_resnet")
    ARGS_MODEL+=(
        "--architecture merl"
        "--merl-backbone resnet"
        "--input-size 2.5" 
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/merl/res18_best_encoder.pth"
    )
    ;;
  "camel")
    ARGS_MODEL+=(
        "--architecture camel"
        "--input-size 10" 
        "--fs-model 256"
        "--input-channels 12"
        "--pretrained ${CAMEL_PRETRAINED}"
    )
    ;;
  "ecgfm_ked")
    ARGS_MODEL+=(
        "--architecture ecgfm_ked"
        "--input-size 10" 
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
    )
    ;;
  "s4")
    ARGS_MODEL+=(
        "--architecture s4"
        "--input-size 2.5"
        "--fs-model 100"
        "--input-channels 12"
        "--s4-n 8"
        "--s4-h 512"
        "--s4-layers 4"
        "--precision 32"
    )
    ;;
  "net1d")
    ARGS_MODEL+=(
        "--architecture net1d"
        "--input-size 2.5"
        "--fs-model 500"
        "--input-channels 12"
    )
    ;;
  "cpc")
    ARGS_MODEL+=(
        "--architecture cpc"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/cpc/config_last_11597276_ckpt.yaml"
    )
    ;;
  "hubert_ecg_base")
    ARGS_MODEL+=(
        "--architecture hubert_ecg"
        "--input-size 5" 
        "--fs-model 100"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/hubert_ecg/hubert_ecg_base.safetensors"
    )
    ;;
  "ecg_fm")
    ARGS_MODEL+=(
        "--architecture ecg_fm"
        "--input-size 5" 
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
    )
    ;;              
esac

# Run the experiment
python ${BASE_DIR}/code/main_lite.py \
  ${ARGS_DATASET[@]} \
  ${ARGS_MODEL[@]} \
  --epochs 100 \
  --modality ecg \
  --lr ${LEARNING_RATE} \
  --batch-size ${BATCH_SIZE} \
  --finetune \
  --eval-mode ${EVAL_MODE} \
  --output-path "${OUTPUT_DIR}/${MODEL}_${DATASET}" \
  --prediction-path "${PREDICTIONS_DIR}/${MODEL}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
  ${ARGS_EXTRA[@]}
