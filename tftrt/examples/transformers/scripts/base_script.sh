#!/bin/bash
# Secure launcher for transformers TF-TRT benchmarks.
# Uses argv arrays + env assignments instead of `eval` to avoid shell injection.

set -euo pipefail

nvidia-smi

# Runtime Parameters
MODEL_NAME=""
MODEL_DIR=""

# Default Argument Values
NVIDIA_TF32_OVERRIDE_VALUE=""
TF_XLA_FLAGS_VALUE=""

# TODO: remove when real dataloader is implemented
DATA_DIR="/tmp"

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
        ;;
        --vocab_size=*)
        ;;
        --minimum_segment_size=*)
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

MIN_SEGMENT_SIZE=5
VOCAB_SIZE=-1

case ${MODEL_NAME} in
  "bert_base_uncased" | "bert_large_uncased")
    VOCAB_SIZE=30522
    ;;

  "bert_base_cased" | "bert_large_cased")
    VOCAB_SIZE=28996
    ;;

  "bart_base" | "bart_large")
    VOCAB_SIZE=50265
    MIN_SEGMENT_SIZE=90
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
# Custom Transormers Task Flags
echo "[*] MIN_SEGMENT_SIZE: ${MIN_SEGMENT_SIZE}"
echo "[*] VOCAB_SIZE: ${VOCAB_SIZE}"
echo ""
echo "[*] TF_XLA_FLAGS: ${TF_XLA_FLAGS_VALUE}"
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS[*]:-}"

echo -e "********************************************************************\n"

# ======================= ARGUMENT VALIDATION ======================= #

# Dataset Directory

if [[ -z ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

# ----------------------  Model Directory --------------

if [[ -z ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

INPUT_SAVED_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}/pb_model

if [[ ! -d ${INPUT_SAVED_MODEL_DIR} ]]; then
    echo "ERROR: the directory \`${INPUT_SAVED_MODEL_DIR}\` does not exist."
    exit 1
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "${BENCH_DIR}"

# Execute the example without shell evaluation of user-controlled strings.
CMD=(
    python transformers.py
    --input_saved_model_dir "${INPUT_SAVED_MODEL_DIR}"
    --data_dir "${DATA_DIR}"
    --vocab_size "${VOCAB_SIZE}"
    --minimum_segment_size "${MIN_SEGMENT_SIZE}"
)
if ((${#BYPASS_ARGUMENTS[@]})); then
    CMD+=("${BYPASS_ARGUMENTS[@]}")
fi

echo -e "**Executing:**\n"
printf '  %q' "${CMD[@]}"
echo -e "\n"
sleep 5

if [[ -n "${NVIDIA_TF32_OVERRIDE_VALUE}" ]]; then
    export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE_VALUE}"
fi
if [[ -n "${TF_XLA_FLAGS_VALUE}" ]]; then
    export TF_XLA_FLAGS="${TF_XLA_FLAGS_VALUE}"
fi

"${CMD[@]}"
