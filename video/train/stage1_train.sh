#!/usr/bin/env bash
set -euo pipefail

# Stage 1: run training from Visual-Memory-0122.
# This script keeps behavior close to the original training launcher while
# exposing commonly changed parameters as env vars.

ROOT_DIR="/data1/Compression_clean/video/train"
SCRIPT_DIR="${ROOT_DIR}/stage1_vm"
PY_SCRIPT="train_lora_single.py"

NAME="${NAME:-HoneyBee}"
RANK="${RANK:-1}"
V_LENGTH="${V_LENGTH:-131072}"
NUM_GPUS="${NUM_GPUS:-2}"
ACCUM_STEP="${ACCUM_STEP:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-0.002}"
LAST_K="${LAST_K:-30}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

FRAMES_DIR="${FRAMES_DIR:-/output/datasets/benchmark_832x480/${NAME}_832x480}"
CAPTION_FILE="${CAPTION_FILE:-./descriptions/descriptions_${NAME}.txt}"
OUT_DIR="${OUT_DIR:-/data1/Compression_clean/video/train/checkpoints/stage1}"
CACHE_DIR="${CACHE_DIR:-/data1/Visual-Memory-0122/Visual-Memory/pretrained_models}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-25}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

mkdir -p "${OUT_DIR}"

echo "[Stage1] Using master port: ${MASTER_PORT}"
echo "[Stage1] Output dir: ${OUT_DIR}"

cd "${SCRIPT_DIR}"
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${PY_SCRIPT}" \
  --frames_dir "${FRAMES_DIR}" \
  --caption_file "${CAPTION_FILE}" \
  --out_dir "${OUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --wan_model_id "${WAN_MODEL_ID}" \
  --train_steps "${TRAIN_STEPS}" \
  --lr "${LR}" \
  --grad_accum_steps "${ACCUM_STEP}" \
  --batch_size "${BATCH_SIZE}" \
  --rank "${RANK}" \
  --last_k "${LAST_K}" \
  --theta_d_length="${V_LENGTH}" \
  --lora_preset qkvo \
  --num_frames 81 \
  --width 832 \
  --height 480 \
  --save_every "${SAVE_EVERY}" \
  --test_steps 50 \
  --test_cfg 1.0 \
  --generate_samples \
  --seed "${SEED}" \
  "$@"
