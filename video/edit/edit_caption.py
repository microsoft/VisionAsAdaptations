#!/usr/bin/env python3
"""Video editing via caption change: load a trained UniLoRA checkpoint and
generate a new video using a different text prompt (caption editing)."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import load_file
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

try:
    import lpips
except ImportError as exc:  # pragma: no cover - package import guard
    raise ImportError("The 'lpips' package is required. Install it via 'pip install lpips'.") from exc

try:
    from DISTS_pytorch import DISTS
except ImportError as exc:  # pragma: no cover - package import guard
    raise ImportError("The 'dists-pytorch' package is required. Install it via 'pip install dists-pytorch'.") from exc

from pipeline_wan import WanPipeline
from unilora_utils import (
    inject_unilora,
    build_lora_regex,
    load_unilora_state_dict,
    # compute_unilora_rate,
)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_caption_text(args: argparse.Namespace) -> str:
    if args.caption_file is not None:
        if not args.caption_file.is_file():
            raise FileNotFoundError(f"Caption file not found: {args.caption_file}")
        return args.caption_file.read_text(encoding="utf-8").strip()
    if args.caption:
        return args.caption
    raise ValueError("Provide --caption or --caption_file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a Wan video using UniLoRA weights")
    parser.add_argument("--weights", type=Path, required=True, help="Path to UniLoRA .safetensors file")
    parser.add_argument("--caption", type=str, default=None, help="Positive prompt / caption")
    parser.add_argument(
        "--caption_file",
        type=Path,
        default=None,
        help="Path to a text file containing the caption; overrides --caption when set",
    )
    parser.add_argument("--negative_prompt", type=str, default="", help="Optional negative prompt")
    parser.add_argument(
        "--wan_model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Base Wan model identifier or local path",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("../pretrained_models"),
        help="Cache directory for base models",
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale during inference")
    parser.add_argument("--flow_shift", type=float, default=3.0, help="UniPC flow shift (3.0 for 480p, 5.0 for 720p)")
    parser.add_argument("--video_fps", type=int, default=24)
    parser.add_argument(
        "--output_video",
        type=Path,
        default=Path("./reconstructed_video.mp4"),
        help="Where to save the output video",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Computation dtype for Wan pipeline",
    )

    # UniLoRA metadata
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--theta_d_length", type=int, default=65536)
    parser.add_argument("--proj_seed", type=int, default=42)
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument(
        "--lora_preset",
        type=str,
        default="qkvo",
        choices=["qkv", "qkvo", "qkvo_mlp"],
    )
    parser.add_argument("--last_k", type=int, default=20)
    parser.add_argument(
        "--active_lora_file",
        type=Path,
        default=None,
        help="Optional path to save the activated LoRA layer summary",
    )
    parser.add_argument(
        "--original_frames_dir",
        type=Path,
        default=None,
        help="Directory containing original frames for PSNR computation",
    )
    parser.add_argument(
        "--compute_psnr",
        action="store_true",
        help="Compute PSNR, DISTS, and LPIPS metrics between original and reconstructed frames",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width % 16 != 0 or args.height % 16 != 0:
        raise ValueError("Width and height must be divisible by 16 for Wan VAE")
    if args.num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if args.dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {args.dtype}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.last_k <= 0:
        args.last_k = None
    if args.theta_d_length <= 0:
        raise ValueError("theta_d_length must be positive")
    if args.init_theta_d_bound <= 0:
        raise ValueError("init_theta_d_bound must be positive")
    # Caption presence/file existence is enforced in load_caption_text()


def load_pipeline(args: argparse.Namespace, device: torch.device) -> WanPipeline:
    dtype = DTYPE_MAP[args.dtype]
    vae = AutoencoderKLWan.from_pretrained(
        args.wan_model_id,
        subfolder="vae",
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    pipe = WanPipeline.from_pretrained(
        args.wan_model_id,
        vae=vae,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    pipe.to(device=device, dtype=dtype)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if pipe.transformer is None:
        raise RuntimeError("WanPipeline missing transformer component")
    pipe.transformer.requires_grad_(False)
    pipe.transformer2 = copy.deepcopy(pipe.transformer)

    return pipe


def prepare_transformer2(pipe: WanPipeline, primary: torch.device, secondary: torch.device) -> None:
    if secondary == primary:
        print("[Transformer2] Secondary device equals primary; skipping clone")
        return

    if pipe.transformer is None:
        raise RuntimeError("WanPipeline missing transformer component")

    print("[Transformer2] Cloning transformer to second GPU for negative prompt guidance")
    print("  Step 1: Moving transformer to CPU …")
    pipe.transformer.to("cpu")
    print("  Step 2: Deep copying transformer …")
    transformer_clone = copy.deepcopy(pipe.transformer)
    print("  Step 3: Moving transformer back to primary device …")
    pipe.transformer.to(primary)
    print("  Step 4: Moving transformer2 to secondary device …")
    transformer_clone.to(secondary)
    transformer_clone.requires_grad_(False)
    transformer_clone.eval()
    pipe.transformer_2 = transformer_clone
    print(f"[Transformer2] transformer on {primary}, transformer_2 on {secondary}")


def prepare_lora(
    pipe: WanPipeline,
    args: argparse.Namespace,
    export_path: Path | None = None,
) -> None:
    total_blocks = len(pipe.transformer.blocks)
    target_regex = build_lora_regex(
        total_blocks=total_blocks,
        last_k=args.last_k,
        preset=args.lora_preset,
        model="wan",
    )
    inject_unilora(
        pipe.transformer,
        target_regex=target_regex,
        r=args.rank,
        theta_d_length=args.theta_d_length,
        proj_seed=args.proj_seed,
        dropout=0.0,
        init_theta_d_bound=args.init_theta_d_bound,
    )

    print(f"[UniLoRA] Injected adapters with rank={args.rank}, theta_d_length={args.theta_d_length}, preset={args.lora_preset}")

    pipe.transformer.eval()
    state_dict = load_file(args.weights)
    load_unilora_state_dict(pipe.transformer, state_dict)
    print(f"Loaded UniLoRA weights from {args.weights}")

    pipe.transformer.eval()
    # rate_info = compute_unilora_rate(pipe.transformer, use_cached=False)
    # total_pixels = max(1, args.num_frames * args.height * args.width)
    # bpp = (rate_info["total_bits"] / float(total_pixels)).item()
    # print(
    #     f"[UniLoRA] Rate estimate: {rate_info['bits_per_param'].item():.6f} bits/param, {bpp:.6f} bpp"
    # )

    if export_path:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_payload = {
            "weights": str(args.weights),
            "rank": args.rank,
            "last_k": args.last_k,
            "lora_preset": args.lora_preset,
            "theta_d_length": args.theta_d_length,
            "proj_seed": args.proj_seed,
            "init_theta_d_bound": args.init_theta_d_bound,
        }
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"[UniLoRA] Exported active layer summary to {export_path}")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute PSNR between two numpy arrays (uint8, 0-255)."""
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 0:
        return float("inf")
    max_i = 255.0
    return 20.0 * math.log10(max_i) - 10.0 * math.log10(mse)


def _frame_to_tensors(frame: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert BGR frame to RGB tensors for DISTS and LPIPS.
    
    Args:
        frame: BGR frame as numpy array (uint8, 0-255)
        device: Device to place tensors on
    
    Returns:
        Tuple of (dists_tensor, lpips_tensor) where:
        - dists_tensor: [1, C, H, W] in [0, 1] range
        - lpips_tensor: [1, C, H, W] in [-1, 1] range
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0
    lpips_tensor = tensor * 2.0 - 1.0
    return tensor.to(device), lpips_tensor.to(device)


def load_original_frames(frames_dir: Path, num_frames: int) -> list[np.ndarray]:
    """Load original frames from directory, sorted by filename."""
    if not frames_dir.exists():
        raise FileNotFoundError(f"Original frames directory not found: {frames_dir}")
    
    # Collect all PNG files and sort them
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    
    # Load up to num_frames
    frames = []
    for frame_path in frame_files[:num_frames]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        frames.append(frame)
    
    return frames


def load_reference_video_tensor(
    frames_dir: Path,
    num_frames: int,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    """Load frames and convert to video tensor in [B, C, T, H, W] format, [0, 1] range.
    
    Args:
        frames_dir: Directory containing frame images
        num_frames: Number of frames to load
        width: Target width
        height: Target height
        device: Device to place tensor on
    
    Returns:
        Video tensor with shape [1, 3, T, H, W] in [0, 1] range
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Collect all PNG files and sort them
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    
    # Limit to num_frames
    frame_files = frame_files[:num_frames]
    
    # Transform to resize and convert to tensor
    transform = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ])
    
    # Load and transform frames
    pixels = []
    for frame_path in frame_files:
        with Image.open(frame_path) as img:
            # Convert BGR (from cv2) to RGB (PIL expects RGB)
            img_rgb = img.convert("RGB")
            pixels.append(transform(img_rgb))
    
    # Stack frames: [T, C, H, W] -> [C, T, H, W] -> [1, C, T, H, W]
    video = torch.stack(pixels, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return video.to(device)


def compute_psnr_metrics(
    original_frames: list[np.ndarray],
    reconstructed_frames: np.ndarray,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute PSNR, DISTS, and LPIPS metrics between original and reconstructed frames.
    
    Computes MSE for each frame, averages the MSE values, then calculates PSNR.
    Also computes DISTS, LPIPS-VGG, and LPIPS-Alex per frame and averages them.
    
    Args:
        original_frames: List of original frames (BGR format from cv2)
        reconstructed_frames: Reconstructed frames as numpy array (RGB, 0-255)
        device: Device to use for DISTS and LPIPS models (defaults to CUDA if available)
    
    Returns:
        Dictionary with PSNR, MSE, DISTS, LPIPS-VGG, LPIPS-Alex metrics
    """
    num_frames = min(len(original_frames), len(reconstructed_frames))
    if num_frames == 0:
        raise ValueError("No frames to compare")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models for DISTS and LPIPS
    lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_alex = lpips.LPIPS(net="alex").to(device).eval()
    dists_model = DISTS().to(device).eval()
    
    mse_values = []
    dists_values = []
    lpips_vgg_values = []
    lpips_alex_values = []
    
    with torch.no_grad():
        for idx in range(num_frames):
            # Original frame is in BGR format from cv2
            orig = original_frames[idx].astype(np.float32)
            # Reconstructed frame is in RGB format, convert to BGR for comparison
            recon = reconstructed_frames[idx]
            if len(recon.shape) == 3 and recon.shape[2] == 3:
                # Convert RGB to BGR
                recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
            else:
                recon_bgr = recon
            
            # Resize if needed
            if orig.shape[:2] != recon_bgr.shape[:2]:
                recon_bgr = cv2.resize(recon_bgr, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            recon_bgr = recon_bgr.astype(np.float32)
            
            # Compute MSE for this frame
            mse_frame = np.mean((orig - recon_bgr) ** 2)
            mse_values.append(mse_frame)
            
            # Convert to tensors for DISTS and LPIPS
            # Both orig and recon_bgr are in BGR format (uint8, 0-255)
            orig_uint8 = orig.astype(np.uint8)
            recon_uint8 = recon_bgr.astype(np.uint8)
            
            orig_dists, orig_lpips = _frame_to_tensors(orig_uint8, device)
            recon_dists, recon_lpips = _frame_to_tensors(recon_uint8, device)
            
            # Compute DISTS and LPIPS
            dists_val = float(dists_model(orig_dists, recon_dists).item())
            lpips_vgg_val = float(lpips_vgg(orig_lpips, recon_lpips).item())
            lpips_alex_val = float(lpips_alex(orig_lpips, recon_lpips).item())
            
            dists_values.append(dists_val)
            lpips_vgg_values.append(lpips_vgg_val)
            lpips_alex_values.append(lpips_alex_val)
    
    # Average MSE across all frames
    avg_mse = float(np.mean(mse_values))
    
    # Compute PSNR from the averaged MSE
    if avg_mse <= 0:
        psnr = float("inf")
    else:
        max_i = 255.0
        psnr = 20.0 * math.log10(max_i) - 10.0 * math.log10(avg_mse)
    
    # Average DISTS and LPIPS values
    avg_dists = float(np.mean(dists_values))
    avg_lpips_vgg = float(np.mean(lpips_vgg_values))
    avg_lpips_alex = float(np.mean(lpips_alex_values))
    
    return {
        "psnr": float(psnr),
        "mse": avg_mse,
        "dists": avg_dists,
        "lpips_vgg": avg_lpips_vgg,
        "lpips_alex": avg_lpips_alex,
        "num_frames": num_frames,
    }


def convert_video_to_frames_array(video: np.ndarray) -> np.ndarray:
    """Convert video from pipeline format to frames array for metrics computation.
    
    Args:
        video: Video in format [B, C, T, H, W] or [B, T, H, W, C] or similar
    
    Returns:
        Frames array in format [T, H, W, C] with RGB, uint8, 0-255 range
    """
    # Handle different input formats
    if video.ndim == 5:
        # [B, C, T, H, W] or [B, T, C, H, W] or [B, T, H, W, C]
        if video.shape[1] == 3 or video.shape[1] == 1:
            # [B, C, T, H, W] -> take first batch and transpose to [T, H, W, C]
            video = video[0]  # [C, T, H, W]
            video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, C]
        elif video.shape[2] == 3 or video.shape[2] == 1:
            # [B, T, C, H, W] -> take first batch and transpose to [T, H, W, C]
            video = video[0]  # [T, C, H, W]
            video = np.transpose(video, (0, 2, 3, 1))  # [T, H, W, C]
        elif video.shape[4] == 3 or video.shape[4] == 1:
            # [B, T, H, W, C] -> take first batch
            video = video[0]  # [T, H, W, C]
        else:
            # Try to infer: assume [B, C, T, H, W]
            video = video[0]  # [C, T, H, W]
            video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, C]
    elif video.ndim == 4:
        # [T, C, H, W] or [T, H, W, C] or [C, T, H, W]
        if video.shape[1] == 3 or video.shape[1] == 1:
            # [T, C, H, W] -> [T, H, W, C]
            video = np.transpose(video, (0, 2, 3, 1))
        elif video.shape[0] == 3 or video.shape[0] == 1:
            # [C, T, H, W] -> [T, H, W, C]
            video = np.transpose(video, (1, 2, 3, 0))
        # else assume already [T, H, W, C]
    elif video.ndim == 3:
        # Single frame [H, W, C] or [C, H, W]
        if video.shape[0] == 3 or video.shape[0] == 1:
            # [C, H, W] -> [H, W, C]
            video = np.transpose(video, (1, 2, 0))
        # Add time dimension
        video = video[np.newaxis, ...]  # [1, H, W, C]
    
    # Normalize to 0-255 if in 0-1 range
    if video.dtype != np.uint8:
        if video.max() <= 1.0:
            video = (video * 255.0).clip(0, 255)
        video = video.astype(np.uint8)
    
    return video


def run_inference(
    pipe: WanPipeline,
    args: argparse.Namespace,
    device: torch.device,
    caption_text: str,
) -> tuple[str, np.ndarray | None]:
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        print(f"Using seed {args.seed}")

    with torch.no_grad():
        result = pipe(
            prompt=caption_text,
            negative_prompt=args.negative_prompt or None,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
        )
    frames = result.frames[0]
    output_path = args.output_video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=args.video_fps)
    
    # Return frames for PSNR computation if needed
    frames_array = None
    if args.compute_psnr:
        # Convert frames to numpy array (0-255 range, RGB format, uint8)
        if isinstance(frames, torch.Tensor):
            frames_array = frames.cpu().numpy()
            # If tensor is in CHW format, convert to HWC
            if frames_array.ndim == 4 and frames_array.shape[1] == 3:
                frames_array = np.transpose(frames_array, (0, 2, 3, 1))
            # Normalize to 0-255 if in 0-1 range
            if frames_array.dtype != np.uint8:
                if frames_array.max() <= 1.0:
                    frames_array = (frames_array * 255.0).clip(0, 255)
                frames_array = frames_array.astype(np.uint8)
        elif isinstance(frames, np.ndarray):
            frames_array = frames.copy()
            # Normalize to 0-255 if in 0-1 range
            if frames_array.dtype != np.uint8:
                if frames_array.max() <= 1.0:
                    frames_array = (frames_array * 255.0).clip(0, 255)
                frames_array = frames_array.astype(np.uint8)
        else:
            # Convert list of PIL Images or numpy arrays to numpy
            frames_list = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    # PIL Image is already in RGB format, 0-255
                    arr = np.array(frame)
                    frames_list.append(arr)
                elif isinstance(frame, torch.Tensor):
                    arr = frame.cpu().numpy()
                    # Handle CHW -> HWC conversion if needed
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    # Normalize to 0-255 if in 0-1 range
                    if arr.dtype != np.uint8:
                        if arr.max() <= 1.0:
                            arr = (arr * 255.0).clip(0, 255)
                        arr = arr.astype(np.uint8)
                    frames_list.append(arr)
                elif isinstance(frame, np.ndarray):
                    arr = frame.copy()
                    # Normalize to 0-255 if in 0-1 range
                    if arr.dtype != np.uint8:
                        if arr.max() <= 1.0:
                            arr = (arr * 255.0).clip(0, 255)
                        arr = arr.astype(np.uint8)
                    frames_list.append(arr)
                else:
                    arr = np.array(frame)
                    if arr.dtype != np.uint8:
                        if arr.max() <= 1.0:
                            arr = (arr * 255.0).clip(0, 255)
                        arr = arr.astype(np.uint8)
                    frames_list.append(arr)
            frames_array = np.stack(frames_list)
    
    return str(output_path), frames_array


def main() -> None:
    args = parse_args()
    validate_args(args)
    caption_text = load_caption_text(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Running reconstruction on {device} ({args.dtype})")

    pipe = load_pipeline(args, device)
    prepare_lora(pipe, args, args.active_lora_file)
    
    # Load reference video tensor if original_frames_dir is provided
    reference_video = None
    if args.original_frames_dir is not None:
        print(f"\nLoading reference video from {args.original_frames_dir}...")
        reference_video = load_reference_video_tensor(
            args.original_frames_dir,
            args.num_frames,
            args.width,
            args.height,
            device,
        )
        print(f"Reference video tensor shape: {reference_video.shape}")
    
    output_path, reconstructed_frames = run_inference(
        pipe, args, device, caption_text,
    )

    print(f"Saved reconstructed video to {output_path}")
    
    # Compute PSNR if requested
    if args.compute_psnr:
        if args.original_frames_dir is None:
            raise ValueError("--original_frames_dir must be provided when --compute_psnr is set")
        
        print(f"\nLoading original frames from {args.original_frames_dir}...")
        original_frames = load_original_frames(args.original_frames_dir, args.num_frames)
        
        if reconstructed_frames is None:
            raise RuntimeError("Failed to extract reconstructed frames for PSNR computation")
        
        print("Computing metrics for final reconstruction...")
        metrics = compute_psnr_metrics(original_frames, reconstructed_frames, device=device)
        
        # Output PSNR directly
        print(f"PSNR: {metrics['psnr']:.4f}")


if __name__ == "__main__":
    main()
