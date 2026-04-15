#!/usr/bin/env bash
set -euo pipefail

# Stage 2: run training from VISUAL_SUBMIT.
# You can initialize from Stage 1 by setting INIT_LORA_PATH.

ROOT_DIR="/data1/Compression_clean/video/train"
SCRIPT_DIR="${ROOT_DIR}/stage2_vs"
PY_SCRIPT="train_lora_single.py"

NUM_GPUS="${NUM_GPUS:-2}"
ACCUM_STEP="${ACCUM_STEP:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-0.0015}"
RANK="${RANK:-1}"
LAST_K="${LAST_K:-30}"
LAMBDA_RATE="${LAMBDA_RATE:-0.003}"
LAMBDA_RATE_WARMUP="${LAMBDA_RATE_WARMUP:-500}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

FRAMES_DIR="${FRAMES_DIR:-/data1/dataset/data-codec/benchmark_832x480/UVG/HoneyBee_832x480/}"
CAPTION_FILE="${CAPTION_FILE:-./descriptions/descriptions_HoneyBee.txt}"
OUT_DIR="${OUT_DIR:-/data1/Compression_clean/video/train/checkpoints/stage2}"
CACHE_DIR="${CACHE_DIR:-/data1/VISUAL_SUBMIT/pretrained_models}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
SAVE_EVERY="${SAVE_EVERY:-25}"
SEED="${SEED:-42}"
THETA_D_LENGTH="${THETA_D_LENGTH:-131072}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

# Optional previous LoRA checkpoint for initialization.
INIT_LORA_PATH="${INIT_LORA_PATH:-}"

mkdir -p "${OUT_DIR}"

echo "[Stage2] Using master port: ${MASTER_PORT}"
echo "[Stage2] Output dir: ${OUT_DIR}"
if [[ -n "${INIT_LORA_PATH}" ]]; then
  echo "[Stage2] Init LoRA: ${INIT_LORA_PATH}"
fi

cd "${SCRIPT_DIR}"

CMD=(
  torchrun
  --nproc_per_node="${NUM_GPUS}"
  --master_port="${MASTER_PORT}"
  "${PY_SCRIPT}"
  --frames_dir "${FRAMES_DIR}"
  --caption_file "${CAPTION_FILE}"
  --out_dir "${OUT_DIR}"
  --cache_dir "${CACHE_DIR}"
  --wan_model_id "${WAN_MODEL_ID}"
  --train_steps "${TRAIN_STEPS}"
  --lr "${LR}"
  --grad_accum_steps "${ACCUM_STEP}"
  --batch_size "${BATCH_SIZE}"
  --rank "${RANK}"
  --last_k "${LAST_K}"
  --lambda_rate "${LAMBDA_RATE}"
  --lambda_rate_warmup "${LAMBDA_RATE_WARMUP}"
  --theta_d_length="${THETA_D_LENGTH}"
  --lora_preset qkvo
  --num_frames 81
  --width 832
  --height 480
  --save_every "${SAVE_EVERY}"
  --test_steps 50
  --test_cfg 1.0
  --generate_samples
  --seed "${SEED}"
)

if [[ -n "${INIT_LORA_PATH}" ]]; then
  CMD+=(--init_lora_path "${INIT_LORA_PATH}")
fi

"${CMD[@]}" "$@"
