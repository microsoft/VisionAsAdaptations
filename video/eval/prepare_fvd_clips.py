#!/usr/bin/env python3
"""Prepare spatiotemporal clip folders for FVD computation.

Crops 16-frame, 224x224 patches from real and fake frame directories using a
4x2 spatial grid and multiple non-overlapping temporal segments.

Output structure:
    work_dir/<sequence_name>/{real,fake}/clip_XXXXX/YYYYY.png

Usage:
    python prepare_fvd_clips.py \
        --real_dir /path/to/original_frames \
        --fake_dir /path/to/reconstructed_frames \
        --work_dir ./fvd_clips
"""

import argparse
import os
import shutil
from glob import glob
from typing import List, Sequence

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Prepare spatiotemporal clips for FVD")
    p.add_argument("--sequence_name", type=str, default=None,
                    help="Sequence name (auto-inferred from real_dir if omitted)")
    p.add_argument("--real_dir", type=str, required=True, help="Folder with original frames")
    p.add_argument("--fake_dir", type=str, required=True, help="Folder with reconstructed frames")
    p.add_argument("--work_dir", type=str, required=True, help="Root folder to store clip folders")
    p.add_argument("--sequence_length", type=int, default=16, help="Frames per clip")
    p.add_argument("--crop_size", type=int, default=224, help="Spatial crop size")
    p.add_argument("--x_patches", type=int, default=4, help="Number of crops along width")
    p.add_argument("--y_patches", type=int, default=2, help="Number of crops along height")
    p.add_argument("--num_frames", type=int, default=None,
                    help="Total frames to use (default: min(#real, #fake))")
    p.add_argument("--num_temporal_clips", type=int, default=5,
                    help="Number of non-overlapping temporal clips")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing clips")
    return p.parse_args()


def infer_sequence_name(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    if base.endswith("_832x480"):
        return base.rsplit("_832x480", 1)[0]
    return base


def sorted_frames(folder: str) -> List[str]:
    paths = sorted(glob(os.path.join(folder, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No .png frames found in {folder}")
    return paths


def load_frames(paths: Sequence[str], max_frames: int | None) -> np.ndarray:
    if max_frames is not None:
        paths = paths[:max_frames]
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    return np.stack(frames, axis=0)


def start_positions(total: int, crop: int, num: int) -> List[int]:
    if total < crop:
        raise ValueError(f"Crop size {crop} exceeds dimension {total}")
    if num <= 1:
        return [max((total - crop) // 2, 0)]
    max_start = total - crop
    return [int(round(i * max_start / (num - 1))) for i in range(num)]


def temporal_starts(total_frames: int, clip_len: int, num_clips: int) -> List[int]:
    starts = list(range(0, total_frames - clip_len + 1, clip_len))
    if not starts:
        raise ValueError("Not enough frames for even one clip")
    return starts[:num_clips]


def build_clips(
    frames: np.ndarray,
    clip_len: int,
    crop: int,
    xs: Sequence[int],
    ys: Sequence[int],
    ts: Sequence[int],
) -> List[np.ndarray]:
    clips: List[np.ndarray] = []
    for t0 in ts:
        clip_frames = frames[t0 : t0 + clip_len]
        for y in ys:
            for x in xs:
                clips.append(clip_frames[:, y : y + crop, x : x + crop, :])
    return clips


def write_clips(clips: Sequence[np.ndarray], out_root: str, prefix: str):
    os.makedirs(out_root, exist_ok=True)
    for idx, clip in enumerate(clips):
        clip_dir = os.path.join(out_root, f"{prefix}_{idx:05d}")
        os.makedirs(clip_dir, exist_ok=True)
        for t, frame in enumerate(clip):
            Image.fromarray(frame).save(os.path.join(clip_dir, f"{t:05d}.png"))


def main():
    args = parse_args()
    if not os.path.isdir(args.real_dir):
        raise FileNotFoundError(f"--real_dir not found: {args.real_dir}")
    if not os.path.isdir(args.fake_dir):
        raise FileNotFoundError(f"--fake_dir not found: {args.fake_dir}")

    if not args.sequence_name:
        args.sequence_name = infer_sequence_name(args.real_dir)

    seq_root = os.path.join(args.work_dir, args.sequence_name)
    if os.path.exists(seq_root):
        if args.overwrite:
            shutil.rmtree(seq_root)
        else:
            raise FileExistsError(f"Already exists: {seq_root}  (use --overwrite)")

    real_paths = sorted_frames(args.real_dir)
    fake_paths = sorted_frames(args.fake_dir)
    total = args.num_frames or min(len(real_paths), len(fake_paths))

    real_frames = load_frames(real_paths, total)
    fake_frames = load_frames(fake_paths, total)

    h, w = real_frames.shape[1:3]
    if fake_frames.shape[1:3] != (h, w):
        raise ValueError(
            f"Spatial size mismatch: real {real_frames.shape[1:3]} vs fake {fake_frames.shape[1:3]}"
        )

    xs = start_positions(w, args.crop_size, args.x_patches)
    ys = start_positions(h, args.crop_size, args.y_patches)
    ts = temporal_starts(total, args.sequence_length, args.num_temporal_clips)

    real_clips = build_clips(real_frames, args.sequence_length, args.crop_size, xs, ys, ts)
    fake_clips = build_clips(fake_frames, args.sequence_length, args.crop_size, xs, ys, ts)

    write_clips(real_clips, os.path.join(seq_root, "real"), "clip")
    write_clips(fake_clips, os.path.join(seq_root, "fake"), "clip")

    print(f"Wrote {len(real_clips)} real + {len(fake_clips)} fake clips to {seq_root}")


if __name__ == "__main__":
    main()
