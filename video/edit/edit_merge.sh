#!/bin/bash
# Edit a video by merging two LoRA checkpoints and using a new caption.
# Usage: bash edit_merge.sh
set -euo pipefail
cd "$(dirname "$0")"

NAME1="${NAME1:-Beauty}"
NAME2="${NAME2:-ShakeNDry}"
RANK="${RANK:-1}"
LAST_K="${LAST_K:-60}"
V_LENGTH="${V_LENGTH:-131072}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"

WEIGHTS1="${WEIGHTS1:-/path/to/lora_weights_1.safetensors}"
WEIGHTS2="${WEIGHTS2:-/path/to/lora_weights_2.safetensors}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
CACHE_DIR="${CACHE_DIR:-../pretrained_models}"

CAPTION="${CAPTION:-A new caption for the merged video.}"

OUTPUT_VIDEO="${OUTPUT_VIDEO:-./edited_merge_${NAME1}_${NAME2}.mp4}"
ORIGINAL_FRAMES_DIR="${ORIGINAL_FRAMES_DIR:-}"

CMD=(
  python edit_merge.py
    --weights1 "$WEIGHTS1"
    --weights2 "$WEIGHTS2"
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
