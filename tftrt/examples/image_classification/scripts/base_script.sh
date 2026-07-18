#!/bin/bash
# Secure launcher for image classification TF-TRT benchmarks.
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
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        ;;
        --output_tensor_names=*)
        ;;
        --output_tensor_indices=*)
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

INPUT_SIZE=224
PREPROCESS_METHOD="vgg"
NUM_CLASSES=1001
OUTPUT_TENSOR_NAME_FLAG=()
OUTPUT_TENSOR_IDX_FLAG=()

case ${MODEL_NAME} in
  "inception_v3" | "inception_v4")
    INPUT_SIZE=299
    PREPROCESS_METHOD="inception"
    ;;

  "mobilenet_v1" | "mobilenet_v2")
    PREPROCESS_METHOD="inception"
    ;;

  "nasnet_large")
    INPUT_SIZE=331
    PREPROCESS_METHOD="inception"
    ;;

  "nasnet_mobile")
    PREPROCESS_METHOD="inception"
    ;;

  "resnet_v1.5_50_tfv2" | "vgg_16" | "vgg_19" )
    NUM_CLASSES=1000
    ;;

  "resnet50-v1.5_tf1_ngc" )
    NUM_CLASSES=1000
    OUTPUT_TENSOR_IDX_FLAG=(--output_tensor_indices=0)
    OUTPUT_TENSOR_NAME_FLAG=(--output_tensor_names=classes)
    PREPROCESS_METHOD="resnet50_v1_5_tf1_ngc_preprocess"
    ;;

  "resnet50v2_backbone" | "resnet50v2_sparse_backbone" )
    INPUT_SIZE=256
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
# Custom Image Classification Task Flags
echo "[*] INPUT_SIZE: ${INPUT_SIZE}"
echo "[*] PREPROCESS_METHOD: ${PREPROCESS_METHOD}"
echo "[*] NUM_CLASSES: ${NUM_CLASSES}"
echo "[*] OUTPUT_TENSOR_IDX_FLAG: ${OUTPUT_TENSOR_IDX_FLAG[*]:-}"
echo "[*] OUTPUT_TENSOR_NAME_FLAG: ${OUTPUT_TENSOR_NAME_FLAG[*]:-}"
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

INPUT_SAVED_MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}"

if [[ ! -d "${INPUT_SAVED_MODEL_DIR}" ]]; then
    echo "ERROR: the directory \`${INPUT_SAVED_MODEL_DIR}\` does not exist."
    exit 1
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "${BENCH_DIR}"

# Private, collision-resistant output directory (avoids predictable /tmp/$RANDOM)
OUTPUT_SAVED_MODEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tftrt_ic_XXXXXX")"

# Execute the example without shell evaluation of user-controlled strings.
CMD=(
    python image_classification.py
    --data_dir "${DATA_DIR}"
    --calib_data_dir "${DATA_DIR}"
    --input_saved_model_dir "${INPUT_SAVED_MODEL_DIR}"
    --output_saved_model_dir "${OUTPUT_SAVED_MODEL_DIR}"
    --input_size "${INPUT_SIZE}"
    --preprocess_method "${PREPROCESS_METHOD}"
    --num_classes "${NUM_CLASSES}"
)
if ((${#OUTPUT_TENSOR_IDX_FLAG[@]})); then
    CMD+=("${OUTPUT_TENSOR_IDX_FLAG[@]}")
fi
if ((${#OUTPUT_TENSOR_NAME_FLAG[@]})); then
    CMD+=("${OUTPUT_TENSOR_NAME_FLAG[@]}")
fi
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
