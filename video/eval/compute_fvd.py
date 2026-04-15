#!/usr/bin/env python3
"""Compute FVD (Fréchet Video Distance) from prepared clip folders.

Expected clip structure (output of prepare_fvd_clips.py):
    clips_root/<sequence>/{real,fake}/clip_XXXXX/YYYYY.png

Uses a pre-trained I3D model (TorchScript) to extract features, then computes
the Fréchet distance between real and fake feature distributions.

Usage:
    python compute_fvd.py --clips_root ./fvd_clips
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from glob import glob
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch

I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "fvd_cache", "i3d_torchscript.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Compute FVD from prepared clip folders")
    p.add_argument("--clips_root", type=str, required=True,
                    help="Root folder containing <sequence>/{real,fake}/clip_*/")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4,
                    help="Number of clips per I3D forward pass")
    p.add_argument("--out_json", type=str, default=None, help="Save results as JSON")
    p.add_argument("--out_csv", type=str, default=None, help="Save results as CSV")
    p.add_argument("--i3d_url", type=str, default=I3D_URL)
    p.add_argument("--i3d_path", type=str, default=I3D_DEFAULT_PATH,
                    help="Local cache path for I3D TorchScript model")
    p.add_argument("--rescale", action="store_true",
                    help="Rescale input to [-1,1] inside detector")
    p.add_argument("--resize", action="store_true",
                    help="Resize input to 224 inside detector")
    return p.parse_args()


def list_dirs(path: str) -> List[str]:
    return [d for d in sorted(glob(os.path.join(path, "*"))) if os.path.isdir(d)]


def list_clip_pairs(seq_dir: str) -> List[Tuple[str, str]]:
    real_root = os.path.join(seq_dir, "real")
    fake_root = os.path.join(seq_dir, "fake")
    if not os.path.isdir(real_root) or not os.path.isdir(fake_root):
        raise FileNotFoundError(f"Missing real/fake under {seq_dir}")
    real_clips = sorted(d for d in glob(os.path.join(real_root, "clip_*")) if os.path.isdir(d))
    fake_clips = sorted(d for d in glob(os.path.join(fake_root, "clip_*")) if os.path.isdir(d))
    if len(real_clips) != len(fake_clips):
        raise ValueError(f"Clip count mismatch: {len(real_clips)} real vs {len(fake_clips)} fake")
    return list(zip(real_clips, fake_clips))


def load_clip_numpy(clip_dir: str) -> np.ndarray:
    frame_paths = sorted(glob(os.path.join(clip_dir, "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames in {clip_dir}")
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths]
    return np.stack(frames, axis=0).astype(np.float32) / 255.0


def batch_iter(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_model(url: str, path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading I3D model to {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


def compute_stats(feats: np.ndarray):
    if feats.ndim != 2:
        raise ValueError("Expected [N, D]")
    mu = feats.mean(axis=0)
    if feats.shape[0] < 2:
        sigma = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    import scipy.linalg

    mu_g, sig_g = compute_stats(feats_fake)
    mu_r, sig_r = compute_stats(feats_real)
    m = float(np.square(mu_g - mu_r).sum())
    s, _ = scipy.linalg.sqrtm(np.dot(sig_g, sig_r), disp=False)
    if np.iscomplexobj(s):
        s = np.real(s)
    return float(np.real(m + np.trace(sig_g + sig_r - 2 * s)))


@torch.no_grad()
def extract_features(detector, clips: List[np.ndarray], device: str,
                     rescale: bool, resize: bool) -> np.ndarray:
    videos = np.stack(clips, axis=0)
    tensor = torch.from_numpy(videos).permute(0, 4, 1, 2, 3).to(device)
    feats = detector(tensor, rescale=rescale, resize=resize, return_features=True)
    return feats.cpu().numpy()


def compute_fvd_for_pairs(clip_pairs, device, batch_size, detector, rescale, resize) -> float:
    feats_real_all: List[np.ndarray] = []
    feats_fake_all: List[np.ndarray] = []
    for batch in batch_iter(clip_pairs, batch_size):
        real_clips = [load_clip_numpy(r) for r, _ in batch]
        fake_clips = [load_clip_numpy(f) for _, f in batch]
        feats_real_all.append(extract_features(detector, real_clips, device, rescale, resize))
        feats_fake_all.append(extract_features(detector, fake_clips, device, rescale, resize))
    return frechet_distance(
        np.concatenate(feats_fake_all, axis=0),
        np.concatenate(feats_real_all, axis=0),
    )


def main():
    args = parse_args()
    clips_root = os.path.abspath(args.clips_root)
    if not os.path.isdir(clips_root):
        raise FileNotFoundError(f"clips_root not found: {clips_root}")

    i3d_path = ensure_model(args.i3d_url, args.i3d_path)
    detector = torch.jit.load(i3d_path).eval().to(args.device)

    seq_dirs = list_dirs(clips_root)
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence folders in {clips_root}")

    results: Dict[str, float] = {}
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        pairs = list_clip_pairs(seq_dir)
        fvd = compute_fvd_for_pairs(
            pairs, args.device, args.batch_size, detector, args.rescale, args.resize,
        )
        results[seq_name] = round(fvd, 6)
        print(f"  {seq_name}: FVD = {fvd:.6f}  ({len(pairs)} clip pairs)")

    if len(results) > 1:
        all_pairs = []
        for seq_dir in seq_dirs:
            all_pairs.extend(list_clip_pairs(seq_dir))
        agg_fvd = compute_fvd_for_pairs(
            all_pairs, args.device, args.batch_size, detector, args.rescale, args.resize,
        )
        results["__aggregate__"] = round(agg_fvd, 6)
        print(f"  AGGREGATE: FVD = {agg_fvd:.6f}  ({len(all_pairs)} clip pairs)")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {args.out_json}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "fvd"])
            for k, v in results.items():
                w.writerow([k, f"{v:.6f}"])
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
