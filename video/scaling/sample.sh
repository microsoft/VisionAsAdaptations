#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

NAME="$1"
STEPS="${2:-50}"

WEIGHTS_DIR="${WEIGHTS_DIR:-/output/CompressIVF/0109_unilora_factorized6e-3_${NAME}_832_1_3B_131072}"

WEIGHTS=$(ls "$WEIGHTS_DIR"/lora_step_*.safetensors \
    | sed 's/.*lora_step_//' \
    | sed 's/.safetensors//' \
    | sort -n \
    | tail -n 1)
echo "WEIGHTS: $WEIGHTS"

WEIGHTS="${WEIGHTS_DIR}/lora_step_${WEIGHTS}.safetensors"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
OUTPUT_VIDEO="./reconstructed_factorized_${NAME}.mp4"
ORIGINAL_FRAMES_DIR="${ORIGINAL_FRAMES_DIR:-/output/datasets/benchmark_832x480/${NAME}_832x480}"
CACHE_DIR="${CACHE_DIR:-../pretrained_models}"

RANK="${RANK:-1}"
LAST_K="${LAST_K:-30}"
SCALING="${SCALING:-true}"

CMD=(
  python reconstruct_lora_single_scaling_encode.py
    --weights "$WEIGHTS"
    --caption_file "./descriptions/descriptions_${NAME}.txt"
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
    --theta_d_length 131072
    --lora_preset qkvo
    --seed 42
    --output_video "$OUTPUT_VIDEO"
    --compute_psnr
    --original_frames_dir "$ORIGINAL_FRAMES_DIR"
    --name "$NAME"
)

if [[ "${SCALING}" == "true" ]]; then
  CMD+=(--scaling)
fi

"${CMD[@]}" "$@" > "$NAME.log"
