#!/usr/bin/env bash
# Batch reconstruction with a trained UniLoRA (reconstruct_lora_multi.py).
set -euo pipefail
cd "$(dirname "$0")"

INIT_LORA_PATH="${INIT_LORA_PATH:?Set INIT_LORA_PATH to your lora_step_*.safetensors}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-./descriptions/example_descriptions.txt}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./recon_out}"
CACHE_DIR="${CACHE_DIR:-../pretrained_models}"

python reconstruct_lora_multi.py \
    --init_lora_path "${INIT_LORA_PATH}" \
    --description_file="${DESCRIPTION_FILE}" \
    --data_root="${DATA_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    --cache_dir="${CACHE_DIR}" \
    --theta_d_length "${THETA_D_LENGTH:-65536}" \
    --proj_seed "${PROJ_SEED:-42}" \
    --last_k "${LAST_K:-60}" \
    --width "${WIDTH:-512}" \
    --height "${HEIGHT:-512}" \
    --rank "${RANK:-4}" \
    --lora_preset "${LORA_PRESET:-qkvo}" \
    --cfg "${CFG:-1.0}" \
    --seed "${SEED:-42}"

echo "Reconstruction complete. Images under: ${OUTPUT_DIR}"
