#!/bin/bash
# Edit a video by changing its caption while keeping the same LoRA weights.
# Usage: bash edit_caption.sh
set -euo pipefail
cd "$(dirname "$0")"

NAME="${NAME:-Beauty}"
RANK="${RANK:-1}"
LAST_K="${LAST_K:-60}"
V_LENGTH="${V_LENGTH:-131072}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"

WEIGHTS="${WEIGHTS:-/path/to/lora_weights.safetensors}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
CACHE_DIR="${CACHE_DIR:-../pretrained_models}"

CAPTION="${CAPTION:-A new caption for the edited video.}"

OUTPUT_VIDEO="${OUTPUT_VIDEO:-./edited_caption_${NAME}.mp4}"
ORIGINAL_FRAMES_DIR="${ORIGINAL_FRAMES_DIR:-}"

CMD=(
  python edit_caption.py
    --weights "$WEIGHTS"
    --wan_model_id "$WAN_MODEL_ID"
    --cache_dir "$CACHE_DIR"
    --dtype bf16
    --width 832
    --height 480
    --num_frames 81
    --steps "$STEPS"
    --cfg 1.0
    --video_fps 30
    --rank "$RANK"
    --last_k "$LAST_K"
    --theta_d_length "$V_LENGTH"
    --lora_preset qkvo
    --seed "$SEED"
    --output_video "$OUTPUT_VIDEO"
    --caption "$CAPTION"
)

if [[ -n "${ORIGINAL_FRAMES_DIR}" ]]; then
  CMD+=(--compute_psnr --original_frames_dir "$ORIGINAL_FRAMES_DIR")
fi

"${CMD[@]}" "$@"
