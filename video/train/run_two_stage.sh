#!/usr/bin/env bash
set -euo pipefail

# Two-stage training orchestrator.
# Stage 1 uses Visual-Memory training.
# Stage 2 uses VISUAL_SUBMIT training and initializes from Stage 1 checkpoint.

ROOT_DIR="/data1/Compression_clean/video/train"
STAGE1_SH="${ROOT_DIR}/stage1_train.sh"
STAGE2_SH="${ROOT_DIR}/stage2_train.sh"

STAGE1_OUT_DIR="${STAGE1_OUT_DIR:-${ROOT_DIR}/checkpoints/stage1}"
STAGE2_OUT_DIR="${STAGE2_OUT_DIR:-${ROOT_DIR}/checkpoints/stage2}"

echo "[Pipeline] Running stage 1..."
OUT_DIR="${STAGE1_OUT_DIR}" bash "${STAGE1_SH}"

if [[ -f "${STAGE1_OUT_DIR}/lora_weights.safetensors" ]]; then
  INIT_CKPT="${STAGE1_OUT_DIR}/lora_weights.safetensors"
else
  INIT_CKPT="$(ls -1 "${STAGE1_OUT_DIR}"/lora_step_*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "${INIT_CKPT:-}" ]]; then
  echo "[Pipeline] Error: could not find stage-1 checkpoint in ${STAGE1_OUT_DIR}"
  exit 1
fi

echo "[Pipeline] Found stage-1 checkpoint: ${INIT_CKPT}"
echo "[Pipeline] Running stage 2..."
OUT_DIR="${STAGE2_OUT_DIR}" INIT_LORA_PATH="${INIT_CKPT}" bash "${STAGE2_SH}"

echo "[Pipeline] Done."
