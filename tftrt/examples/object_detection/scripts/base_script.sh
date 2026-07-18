#!/bin/bash
# Secure launcher for object detection TF-TRT benchmarks.
# Uses argv arrays + env assignments instead of `eval` to avoid shell injection.

set -euo pipefail

nvidia-smi

# Runtime Parameters
MODEL_NAME=""
DATA_DIR=""
MODEL_DIR=""

# Default Argument Values
NVIDIA_TF32_OVERRIDE_VALUE=""
TF_XLA_FLAGS_VALUE=""

BATCH_SIZE=8
MAX_WORKSPACE_SIZE=$((2 ** (32 + 1)))  # + 1 necessary compared to python
INPUT_SIZE=640

BYPASS_ARGUMENTS=()

# Loop through arguments and process them
for arg in "$@"; do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        ;;
        --no_tf32)
        NVIDIA_TF32_OVERRIDE_VALUE="0"
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        ;;
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        ;;
        --use_xla_auto_jit)
        TF_XLA_FLAGS_VALUE="--tf_xla_auto_jit=2"
        ;;
        *)
        BYPASS_ARGUMENTS+=("${arg}")
        ;;
    esac
done

# ============== Set model specific parameters ============= #

case ${MODEL_NAME} in
  "faster_rcnn_resnet50_coco" | "ssd_mobilenet_v1_fpn_coco")
    MAX_WORKSPACE_SIZE=$((2 ** (24 + 1)))  # + 1 necessary compared to python
    ;;
esac

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
echo "[*] NVIDIA_TF32_OVERRIDE: ${NVIDIA_TF32_OVERRIDE_VALUE}"
echo ""
# Custom Object Detection Task Flags
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] INPUT_SIZE: ${INPUT_SIZE}"
echo "[*] MAX_WORKSPACE_SIZE: ${MAX_WORKSPACE_SIZE}"
echo ""
echo "[*] TF_XLA_FLAGS: ${TF_XLA_FLAGS_VALUE}"
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS[*]:-}"
echo -e "********************************************************************\n"

# ======================= ARGUMENT VALIDATION ======================= #

# Dataset Directory

if [[ -z "${DATA_DIR}" ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

VAL_DATA_DIR="${DATA_DIR}/val2017"
ANNOTATIONS_DATA_FILE="${DATA_DIR}/annotations/instances_val2017.json"

if [[ ! -d "${VAL_DATA_DIR}" ]]; then
    echo "ERROR: the directory \`${VAL_DATA_DIR}\` does not exist."
    exit 1
fi

if [[ ! -f "${ANNOTATIONS_DATA_FILE}" ]]; then
    echo "ERROR: the file \`${ANNOTATIONS_DATA_FILE}\` does not exist."
    exit 1
fi

# ----------------------  Model Directory --------------

if [[ -z "${MODEL_DIR}" ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

if [[ -z "${MODEL_NAME}" ]]; then
    echo "ERROR: \`--model_name=...\` is missing."
    exit 1
fi

# Reject path separators / traversal in model name before joining paths.
if [[ "${MODEL_NAME}" == *"/"* || "${MODEL_NAME}" == *".."* ]]; then
    echo "ERROR: \`--model_name\` contains illegal path characters. [Received: \`${MODEL_NAME}\`]"
    exit 1
fi

# Validate BATCH_SIZE before it is embedded in the model directory path.
if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -eq 0 ]]; then
    echo "ERROR: \`--batch_size\` must be a positive integer. [Received: \`${BATCH_SIZE}\`]"
    exit 1
fi

INPUT_SAVED_MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}_640_bs${BATCH_SIZE}"

if [[ ! -d "${INPUT_SAVED_MODEL_DIR}" ]]; then
    echo "ERROR: the directory \`${INPUT_SAVED_MODEL_DIR}\` does not exist."
    exit 1
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "${BENCH_DIR}"

# Step 1: Installing dependencies if needed.
# Under `set -e`, a failing probe must be in a conditional or the script exits.
if ! python -c "from pycocotools.coco import COCO" > /dev/null 2>&1; then
    bash install_dependencies.sh
fi

# Step 2: Execute the example without shell evaluation of user-controlled strings.
OUTPUT_SAVED_MODEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tftrt_od_XXXXXX")"

CMD=(
    python object_detection.py
    --data_dir "${VAL_DATA_DIR}"
    --calib_data_dir "${VAL_DATA_DIR}"
    --annotation_path "${ANNOTATIONS_DATA_FILE}"
    --input_saved_model_dir "${INPUT_SAVED_MODEL_DIR}"
    --output_saved_model_dir "${OUTPUT_SAVED_MODEL_DIR}"
    --batch_size "${BATCH_SIZE}"
    --input_size "${INPUT_SIZE}"
    --max_workspace_size "${MAX_WORKSPACE_SIZE}"
)
if ((${#BYPASS_ARGUMENTS[@]})); then
    CMD+=("${BYPASS_ARGUMENTS[@]}")
fi

echo -e "**Executing:**\n"
printf '%q ' "${CMD[@]}"
echo -e "\n"
sleep 5

if [[ -n "${NVIDIA_TF32_OVERRIDE_VALUE}" ]]; then
    export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE_VALUE}"
fi
if [[ -n "${TF_XLA_FLAGS_VALUE}" ]]; then
    export TF_XLA_FLAGS="${TF_XLA_FLAGS_VALUE}"
fi

"${CMD[@]}"
