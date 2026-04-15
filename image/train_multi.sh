#!/usr/bin/env bash
# Multi-GPU UniLoRA training for Qwen-Image (torchrun + train_lora_multi.py).
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${NUM_GPUS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LR="${LR:-0.002}"
RANK="${RANK:-4}"
LAST_K="${LAST_K:-60}"
CAPTION_START="${CAPTION_START:-1}"
CAPTION_END="${CAPTION_END:-1}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
OUT_DIR="${OUT_DIR:-./checkpoints/run_$(date +%Y%m%d_%H%M%S)}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-./descriptions/example_descriptions.txt}"
DATA_ROOT="${DATA_ROOT:-./data}"
CACHE_DIR="${CACHE_DIR:-../pretrained_models}"

MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
echo "Using master port: ${MASTER_PORT}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train_lora_multi.py \
    --description_file="${DESCRIPTION_FILE}" \
    --data_root="${DATA_ROOT}" \
    --out_dir="${OUT_DIR}" \
    --train_steps="${TRAIN_STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    --lr="${LR}" \
    --rank="${RANK}" \
    --last_k="${LAST_K}" \
    --caption_start="${CAPTION_START}" \
    --caption_end="${CAPTION_END}" \
    --width="${WIDTH}" \
    --height="${HEIGHT}" \
    --save_every="${SAVE_EVERY:-50}" \
    --test_steps="${TEST_STEPS:-50}" \
    --test_cfg="${TEST_CFG:-1.0}" \
    --cache_dir="${CACHE_DIR}"

echo "Training complete. LoRA weights under: ${OUT_DIR}"
