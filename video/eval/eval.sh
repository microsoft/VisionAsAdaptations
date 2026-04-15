#!/bin/bash
# Full evaluation pipeline: PSNR/DISTS/LPIPS + FVD.
# Full documentation: README.md in this directory.
#
# Usage:
#   bash eval.sh
#
# Required env vars:
#   REAL_DIR   — directory of original frames (PNG)
#   FAKE_DIR   — directory of reconstructed frames (PNG)
#
# Optional env vars:
#   NAME           — sequence name (default: inferred from REAL_DIR)
#   NUM_FRAMES     — number of frames to evaluate (default: all)
#   WORK_DIR       — clip output directory (default: ./fvd_clips)
#   OUT_DIR        — where to write result JSONs (default: ./results)
#   SKIP_METRICS   — set to "true" to skip PSNR/DISTS/LPIPS
#   SKIP_FVD       — set to "true" to skip FVD
#   DEVICE         — cuda or cpu (default: cuda)
set -euo pipefail
cd "$(dirname "$0")"

REAL_DIR="${REAL_DIR:?Set REAL_DIR to the original frames directory}"
FAKE_DIR="${FAKE_DIR:?Set FAKE_DIR to the reconstructed frames directory}"

NAME="${NAME:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
WORK_DIR="${WORK_DIR:-./fvd_clips}"
OUT_DIR="${OUT_DIR:-./results}"
SKIP_METRICS="${SKIP_METRICS:-false}"
SKIP_FVD="${SKIP_FVD:-false}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"

# Infer sequence name from REAL_DIR if not set
if [[ -z "$NAME" ]]; then
  NAME="$(basename "$REAL_DIR")"
  NAME="${NAME%_832x480}"
fi

echo "============================================"
echo "Evaluating: $NAME"
echo "  Real:  $REAL_DIR"
echo "  Fake:  $FAKE_DIR"
echo "============================================"

# --- Step 1: Image quality metrics ---
if [[ "$SKIP_METRICS" != "true" ]]; then
  echo ""
  echo "[Step 1/3] Computing PSNR / DISTS / LPIPS ..."
  METRICS_CMD=(python eval_metrics.py
    --real_dir "$REAL_DIR"
    --fake_dir "$FAKE_DIR"
    --device "$DEVICE"
    --out_json "${OUT_DIR}/metrics_${NAME}.json"
  )
  if [[ -n "$NUM_FRAMES" ]]; then
    METRICS_CMD+=(--num_frames "$NUM_FRAMES")
  fi
  "${METRICS_CMD[@]}"
else
  echo "[Step 1/3] Skipping image quality metrics."
fi

# --- Step 2: Prepare FVD clips ---
if [[ "$SKIP_FVD" != "true" ]]; then
  echo ""
  echo "[Step 2/3] Preparing FVD clips ..."
  PREP_CMD=(python prepare_fvd_clips.py
    --real_dir "$REAL_DIR"
    --fake_dir "$FAKE_DIR"
    --work_dir "$WORK_DIR"
    --sequence_name "$NAME"
    --overwrite
  )
  if [[ -n "$NUM_FRAMES" ]]; then
    PREP_CMD+=(--num_frames "$NUM_FRAMES")
  fi
  "${PREP_CMD[@]}"

  # --- Step 3: Compute FVD ---
  echo ""
  echo "[Step 3/3] Computing FVD ..."
  python compute_fvd.py \
    --clips_root "$WORK_DIR" \
    --device "$DEVICE" \
    --out_json "${OUT_DIR}/fvd_${NAME}.json"
else
  echo "[Step 2-3/3] Skipping FVD."
fi

echo ""
echo "Done. Results saved to ${OUT_DIR}/"
