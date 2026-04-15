#!/usr/bin/env python3
"""Compute per-frame image quality metrics between original and reconstructed frames.

Metrics: PSNR, DISTS, LPIPS-VGG, LPIPS-Alex.

Usage:
    python eval_metrics.py \
        --real_dir /path/to/original_frames \
        --fake_dir /path/to/reconstructed_frames \
        --num_frames 81
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from glob import glob
from typing import List

import cv2
import numpy as np
import torch

try:
    import lpips
except ImportError as exc:
    raise ImportError("pip install lpips") from exc

try:
    from DISTS_pytorch import DISTS
except ImportError as exc:
    raise ImportError("pip install dists-pytorch") from exc


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PSNR / DISTS / LPIPS between two frame folders")
    p.add_argument("--real_dir", type=str, required=True, help="Directory of original (ground-truth) frames")
    p.add_argument("--fake_dir", type=str, required=True, help="Directory of reconstructed frames")
    p.add_argument("--num_frames", type=int, default=None, help="Max frames to evaluate (default: all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default=None, help="Optional path to save results as JSON")
    return p.parse_args()


def sorted_pngs(folder: str) -> List[str]:
    paths = sorted(glob(os.path.join(folder, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames in {folder}")
    return paths


def frame_to_tensors(frame_bgr: np.ndarray, device: torch.device):
    """BGR uint8 frame → (dists_tensor [0,1], lpips_tensor [-1,1]), both [1,C,H,W]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device), (t * 2.0 - 1.0).to(device)


def main():
    args = parse_args()
    device = torch.device(args.device)

    real_paths = sorted_pngs(args.real_dir)
    fake_paths = sorted_pngs(args.fake_dir)
    n = min(len(real_paths), len(fake_paths))
    if args.num_frames is not None:
        n = min(n, args.num_frames)
    if n == 0:
        print("No frames to compare.", file=sys.stderr)
        return

    lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_alex = lpips.LPIPS(net="alex").to(device).eval()
    dists_model = DISTS().to(device).eval()

    mse_vals: List[float] = []
    dists_vals: List[float] = []
    lpips_vgg_vals: List[float] = []
    lpips_alex_vals: List[float] = []

    with torch.no_grad():
        for i in range(n):
            orig = cv2.imread(real_paths[i], cv2.IMREAD_COLOR)
            recon = cv2.imread(fake_paths[i], cv2.IMREAD_COLOR)
            if orig is None:
                raise ValueError(f"Cannot read {real_paths[i]}")
            if recon is None:
                raise ValueError(f"Cannot read {fake_paths[i]}")

            if orig.shape[:2] != recon.shape[:2]:
                recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

            mse_vals.append(float(np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2)))

            orig_d, orig_l = frame_to_tensors(orig, device)
            rec_d, rec_l = frame_to_tensors(recon, device)

            dists_vals.append(float(dists_model(orig_d, rec_d).item()))
            lpips_vgg_vals.append(float(lpips_vgg(orig_l, rec_l).item()))
            lpips_alex_vals.append(float(lpips_alex(orig_l, rec_l).item()))

    avg_mse = float(np.mean(mse_vals))
    psnr = float("inf") if avg_mse <= 0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(avg_mse)

    results = {
        "psnr": round(psnr, 4),
        "mse": round(avg_mse, 6),
        "dists": round(float(np.mean(dists_vals)), 6),
        "lpips_vgg": round(float(np.mean(lpips_vgg_vals)), 6),
        "lpips_alex": round(float(np.mean(lpips_alex_vals)), 6),
        "num_frames": n,
    }

    print("=" * 60)
    print(f"  PSNR:        {results['psnr']:.4f} dB")
    print(f"  MSE:         {results['mse']:.6f}")
    print(f"  DISTS:       {results['dists']:.6f}")
    print(f"  LPIPS-VGG:   {results['lpips_vgg']:.6f}")
    print(f"  LPIPS-Alex:  {results['lpips_alex']:.6f}")
    print(f"  Frames:      {results['num_frames']}")
    print("=" * 60)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
