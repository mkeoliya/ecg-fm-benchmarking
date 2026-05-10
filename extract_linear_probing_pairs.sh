#!/bin/bash
# Export the ECG ID / label pairs used by the linear probing dataset.

set -euo pipefail

BASE_DIR="${BASE_DIR:-}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET="ptb"
OUTPUT_PATH=""
PAIRS_ONLY=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/processed}"

usage() {
    echo "Usage: $0 [--dataset DATASET] [--output-dir DIR] [--pairs-only]"
    echo "Example: $0"
    echo "Example: $0 --dataset ptbxl_super"
    echo "Example: $0 --dataset georgia --output-dir /tmp/probing_pairs"
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
        --output)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --output requires a value."
                usage
                exit 1
            fi
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --output=*)
            OUTPUT_PATH="${1#*=}"
            shift
            ;;
        --output-dir)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --output-dir requires a value."
                usage
                exit 1
            fi
            OUTPUT_PATH="$2/.keep.csv"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_PATH="${1#*=}/.keep.csv"
            shift
            ;;
        --pairs-only)
            PAIRS_ONLY=1
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

case "$DATASET" in
    ptb|ningbo|cpsc2018|cpsc_extra|georgia|chapman|sph|echonext|zzu_pecg)
        DATA_PATH="${DATASET_DIR}/${DATASET}"
        ;;
    code15|code15_diag)
        DATA_PATH="${DATASET_DIR}/code15"
        ;;
    ptbxl_super|ptbxl_sub|ptbxl_all)
        DATA_PATH="${DATASET_DIR}/ptb-xl/records500"
        ;;
    mimic)
        DATA_PATH="${DATASET_DIR}/mimic"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'."
        usage
        exit 1
        ;;
esac

ARGS=(
    "--dataset" "$DATASET"
    "--data" "$DATA_PATH"
)

if [[ -n "$OUTPUT_PATH" ]]; then
    ARGS+=("--output" "$OUTPUT_PATH")
else
    ARGS+=("--output-dir" "${BASE_DIR}/linear/probing_pairs")
fi

if [[ "$PAIRS_ONLY" == "1" ]]; then
    ARGS+=("--pairs-only")
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lightning3" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate lightning3
    else
        echo "Warning: conda command not found; continuing with current Python environment."
    fi
fi

python "${BASE_DIR}/extract_linear_probing_pairs.py" "${ARGS[@]}"
