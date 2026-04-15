# Video evaluation pipeline

End-to-end evaluation for reconstructed video frames against originals: **PSNR**, **DISTS**, **LPIPS (VGG + Alex)**, and **FVD** (Fréchet Video Distance via I3D features).

## Prerequisites

- Python 3 with `torch`, `numpy`, `opencv-python`, `Pillow`, `scipy`
- `pip install lpips dists-pytorch`
- CUDA recommended for metrics and required for practical FVD throughput

## Whole pipeline (recommended)

Run everything from this directory:

```bash
cd /data1/Compression_clean/video/eval

REAL_DIR=/path/to/original_frames \
FAKE_DIR=/path/to/reconstructed_frames \
bash eval.sh
```

### What `eval.sh` does

1. **Image quality** — `eval_metrics.py` compares aligned PNG pairs; writes `results/metrics_<NAME>.json`.
2. **FVD clip prep** — `prepare_fvd_clips.py` builds 16-frame, 224×224 clips (4×2 spatial crops × 5 non-overlapping temporal segments) under `WORK_DIR/<NAME>/{real,fake}/clip_*/`.
3. **FVD** — `compute_fvd.py` loads a TorchScript **I3D** model (downloaded once to `fvd_cache/i3d_torchscript.pt`), extracts features from real and fake clips, and computes the Fréchet distance; writes `results/fvd_<NAME>.json`.

### Examples

```bash
# Metrics only (no I3D download / FVD)
REAL_DIR=... FAKE_DIR=... SKIP_FVD=true bash eval.sh

# FVD only (skip PSNR/DISTS/LPIPS)
REAL_DIR=... FAKE_DIR=... SKIP_METRICS=true bash eval.sh

# Custom paths and frame cap
REAL_DIR=... FAKE_DIR=... NAME=HoneyBee NUM_FRAMES=81 \
  WORK_DIR=./clips OUT_DIR=./out bash eval.sh
```

## Step-by-step (manual)

If you prefer to run stages yourself:

```bash
# 1) Per-frame metrics
python eval_metrics.py \
  --real_dir "$REAL_DIR" --fake_dir "$FAKE_DIR" \
  --device cuda --out_json results/metrics.json

# 2) Build clips for FVD
python prepare_fvd_clips.py \
  --real_dir "$REAL_DIR" --fake_dir "$FAKE_DIR" \
  --work_dir ./fvd_clips --sequence_name MySeq --overwrite

# 3) FVD on prepared clips
python compute_fvd.py \
  --clips_root ./fvd_clips --device cuda \
  --out_json results/fvd.json --out_csv results/fvd.csv
```
