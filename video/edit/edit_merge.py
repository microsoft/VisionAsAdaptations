#!/usr/bin/env python3
"""Video editing via LoRA merge + caption change: load two UniLoRA checkpoints,
merge them into one model, and generate a new video with a different text prompt.
"""

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
)
from unilora.layer import Linear as UniLoRALinear


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_two_lora_state_dicts(weights1: Path, weights2: Path) -> tuple[dict, dict]:
    """Load two UniLoRA adaptation files (safetensors) into state dicts."""
    if not weights1.is_file():
        raise FileNotFoundError(f"Weights not found: {weights1}")
    if not weights2.is_file():
        raise FileNotFoundError(f"Weights not found: {weights2}")
    state_dict1 = load_file(str(weights1))
    state_dict2 = load_file(str(weights2))
    return state_dict1, state_dict2


def map_theta_d_to_lora_matrices(
    model: torch.nn.Module,
    theta_d: torch.Tensor,
    adapter_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map a single theta_d vector back to full LoRA (A, B) for each adapted layer.

    Uses each layer's stored indices and scales; theta_d is the shared vector
    from a UniLoRA checkpoint. Returns dict: layer_name -> (A, B).
    """
    theta_d = theta_d.to(device=device, dtype=dtype)
    out = {}
    for name, module in model.named_modules():
        if not isinstance(module, UniLoRALinear):
            continue
        if adapter_name not in getattr(module, "unilora_indices_A", {}):
            continue
        idx_a = module.unilora_indices_A[adapter_name].to(device)
        idx_b = module.unilora_indices_B[adapter_name].to(device)
        scale_a = module.unilora_scales_A[adapter_name].to(device=device, dtype=dtype)
        scale_b = module.unilora_scales_B[adapter_name].to(device=device, dtype=dtype)
        A = theta_d[idx_a.long()] * scale_a
        B = theta_d[idx_b.long()] * scale_b
        out[name] = (A, B)
    return out


def combine_two_lora_updates(
    A1: torch.Tensor,
    B1: torch.Tensor,
    A2: torch.Tensor,
    B2: torch.Tensor,
) -> torch.Tensor:
    """Combine the two LoRA updates (A@B) into a single delta matrix per layer.

    A1, B1 from first adaptation; A2, B2 from second. LoRA forward is x @ A @ B
    with A (r, in_features), B (out_features, r), so delta W = B @ A with shape
    (out_features, in_features). Returns combined_delta of that shape.
    """
    # delta = B @ A: (out_features, r) @ (r, in_features) -> (out_features, in_features)
    B_new = torch.cat([B1, B2], dim=1)
    A_new = torch.cat([A1, A2], dim=0)
    combined_delta = B_new @ A_new
    return combined_delta


# ---------------------------------------------------------------------------
# (Legacy: theta-level merge; dual-LoRA flow now uses map + combine_two_lora_updates.)
# ---------------------------------------------------------------------------
# TODO: Concatenate or combine the two LoRAs before injecting into the model.
#
# Options to implement:
# 1. Theta concatenation: Concatenate theta_d from both checkpoints into a
#    single vector (length = theta_d_length_1 + theta_d_length_2), then
#    inject UniLoRA with the combined length and re-run index generation
#    so that indices span both halves. Requires consistent theta_d_length
#    and proj_seed across both checkpoints, or a merge projection.
# 2. Weight averaging: Use merged_theta_d = (theta_d_1 + theta_d_2) / 2
#    (current placeholder below). Simple but may dilute both adapters.
# 3. Two adapters: Extend inject_unilora to support multiple adapter names
#    (e.g. "lora1" and "lora2"), load each state dict into its adapter,
#    and in the forward pass combine outputs (e.g. sum or learned weights).
# 4. Sequential / gated: Use one LoRA for part of the network and the other
#    for the rest, or gate them by layer or timestep.
#
# Until one of the above is implemented, we use a simple average so the
# script runs with both checkpoints "loaded".
# ---------------------------------------------------------------------------
def merge_two_lora_state_dicts(
    state_dict1: dict,
    state_dict2: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Merge two UniLoRA state dicts into one for loading into the model.

    Implement your concatenation/combination logic at the "TODO: WRITE YOUR CODE HERE"
    block inside this function (search for that string in this file).
    """
    if "unilora_theta_d" not in state_dict1 or "unilora_theta_d" not in state_dict2:
        raise KeyError("Both state dicts must contain 'unilora_theta_d'.")

    theta1 = state_dict1["unilora_theta_d"].to(device=device, dtype=dtype)
    theta2 = state_dict2["unilora_theta_d"].to(device=device, dtype=dtype)

    if theta1.shape != theta2.shape:
        raise ValueError(
            f"theta_d shape mismatch: {theta1.shape} vs {theta2.shape}. "
            "For concatenation, implement TODO: use combined length and index mapping."
        )

    # =========================================================================
    # TODO: WRITE YOUR CODE HERE — concatenate or combine theta1 and theta2.
    # Replace the line below with your logic. Examples:
    #   - Concatenate: merged_theta = torch.cat([theta1, theta2], dim=0)
    #     (then you must inject UniLoRA with 2 * theta_d_length and adjust
    #      indices in prepare_lora_dual / unilora_utils.)
    #   - Average (current): merged_theta = (theta1 + theta2) / 2.0
    #   - Weighted blend: merged_theta = alpha * theta1 + (1 - alpha) * theta2
    # =========================================================================
    merged_theta = (theta1 + theta2) / 2.0  # <-- REPLACE THIS with your combination

    return {"unilora_theta_d": merged_theta}


def load_caption_text(args: argparse.Namespace) -> str:
    """Use --caption if provided (non-empty), otherwise use --caption_file."""
    if args.caption is not None and args.caption.strip():
        return args.caption.strip()
    if args.caption_file is not None:
        if not args.caption_file.is_file():
            raise FileNotFoundError(f"Caption file not found: {args.caption_file}")
        return args.caption_file.read_text(encoding="utf-8").strip()
    raise ValueError("Provide --caption or --caption_file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a Wan video using two UniLoRA weight files"
    )
    parser.add_argument(
        "--weights1",
        type=Path,
        required=True,
        help="Path to first UniLoRA .safetensors file",
    )
    parser.add_argument(
        "--weights2",
        type=Path,
        required=True,
        help="Path to second UniLoRA .safetensors file",
    )
    parser.add_argument("--caption", type=str, default=None, help="Positive prompt / caption (used if provided)")
    parser.add_argument(
        "--caption_file",
        type=Path,
        default=None,
        help="Path to a text file containing the caption; used when --caption is not provided",
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
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="UniPC flow shift (3.0 for 480p, 5.0 for 720p)",
    )
    parser.add_argument("--video_fps", type=int, default=24)
    parser.add_argument(
        "--output_video",
        type=Path,
        default=Path("./reconstructed_video_dual.mp4"),
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

    # UniLoRA metadata (must match both checkpoints for simple merge)
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
    if not args.weights1.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights1}")
    if not args.weights2.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights2}")
    if args.last_k <= 0:
        args.last_k = None
    if args.theta_d_length <= 0:
        raise ValueError("theta_d_length must be positive")
    if args.init_theta_d_bound <= 0:
        raise ValueError("init_theta_d_bound must be positive")


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


def prepare_transformer2(
    pipe: WanPipeline, primary: torch.device, secondary: torch.device
) -> None:
    if secondary == primary:
        print("[Transformer2] Secondary device equals primary; skipping clone")
        return

    if pipe.transformer is None:
        raise RuntimeError("WanPipeline missing transformer component")

    print("[Transformer2] Cloning transformer to second GPU for negative prompt guidance")
    pipe.transformer.to("cpu")
    transformer_clone = copy.deepcopy(pipe.transformer)
    pipe.transformer.to(primary)
    transformer_clone.to(secondary)
    transformer_clone.requires_grad_(False)
    transformer_clone.eval()
    pipe.transformer_2 = transformer_clone
    print(f"[Transformer2] transformer on {primary}, transformer_2 on {secondary}")


def _make_dual_lora_forward(layer: UniLoRALinear, lora2_AB: tuple[torch.Tensor, torch.Tensor], adapter_name: str):
    """Return a forward that uses combine_two_lora_updates for the A@B step."""
    A2, B2 = lora2_AB
    dropout_fn = layer.unilora_dropout[adapter_name]

    def dual_forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        import torch.nn.functional as F
        previous_dtype = x.dtype
        if layer.disable_adapters or layer.merged:
            return layer.base_layer(x, *args, **kwargs)
        result = layer.base_layer(x, *args, **kwargs)
        # (1) Map first LoRA from layer's theta_d to full (A1, B1)
        A1, B1 = layer._get_lora_matrices(adapter_name)
        x_ = dropout_fn(x.to(A1.device))
        # (2) Combine both LoRAs at the A@B step (TODO is inside combine_two_lora_updates)
        combined_delta = combine_two_lora_updates(A1, B1, A2, B2)
        result = result + F.linear(x_, combined_delta)
        return result.to(previous_dtype)

    return dual_forward


def prepare_lora_dual(
    pipe: WanPipeline,
    args: argparse.Namespace,
    export_path: Path | None = None,
) -> None:
    """Inject UniLoRA once, load both LoRA files, map both to full (A,B), apply combined A@B (see TODO in combine_two_lora_updates)."""
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

    print(
        f"[UniLoRA] Injected adapters with rank={args.rank}, theta_d_length={args.theta_d_length}, preset={args.lora_preset}"
    )

    pipe.transformer.eval()

    # Load both adaptation files
    state_dict1, state_dict2 = load_two_lora_state_dicts(args.weights1, args.weights2)
    print(f"Loaded first UniLoRA weights from {args.weights1}")
    print(f"Loaded second UniLoRA weights from {args.weights2}")

    adapter_name = "default"
    dtype = DTYPE_MAP[args.dtype]
    device = next(pipe.transformer.parameters()).device

    # (1) Load first LoRA into the model (theta_d in place)
    load_unilora_state_dict(pipe.transformer, state_dict1, adapter_name=adapter_name)
    print("Loaded first LoRA into transformer.")

    # (2) Map second theta_d back to full LoRA (A2, B2) per layer
    theta_d2 = state_dict2["unilora_theta_d"]
    lora_mats_2 = map_theta_d_to_lora_matrices(
        pipe.transformer, theta_d2, adapter_name, device, dtype
    )
    print(f"Mapped second LoRA to full (A,B) for {len(lora_mats_2)} layers.")

    # (3) Patch each layer so the A@B step uses combine_two_lora_updates(A1,B1,A2,B2)
    for name, module in pipe.transformer.named_modules():
        if not isinstance(module, UniLoRALinear):
            continue
        if name not in lora_mats_2:
            continue
        module.forward = _make_dual_lora_forward(module, lora_mats_2[name], adapter_name)
    print("Patched transformer to use combined A@B (see TODO in combine_two_lora_updates).")

    pipe.transformer.eval()

    if export_path:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_payload = {
            "weights1": str(args.weights1),
            "weights2": str(args.weights2),
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
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _frame_to_tensors(
    frame: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0
    )
    lpips_tensor = tensor * 2.0 - 1.0
    return tensor.to(device), lpips_tensor.to(device)


def load_original_frames(frames_dir: Path, num_frames: int) -> list[np.ndarray]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Original frames directory not found: {frames_dir}")
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
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
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    frame_files = frame_files[:num_frames]
    transform = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ])
    pixels = []
    for frame_path in frame_files:
        with Image.open(frame_path) as img:
            img_rgb = img.convert("RGB")
            pixels.append(transform(img_rgb))
    video = torch.stack(pixels, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return video.to(device)


def compute_psnr_metrics(
    original_frames: list[np.ndarray],
    reconstructed_frames: np.ndarray,
    device: torch.device | None = None,
) -> dict[str, float]:
    num_frames = min(len(original_frames), len(reconstructed_frames))
    if num_frames == 0:
        raise ValueError("No frames to compare")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_alex = lpips.LPIPS(net="alex").to(device).eval()
    dists_model = DISTS().to(device).eval()
    mse_values = []
    dists_values = []
    lpips_vgg_values = []
    lpips_alex_values = []
    with torch.no_grad():
        for idx in range(num_frames):
            orig = original_frames[idx].astype(np.float32)
            recon = reconstructed_frames[idx]
            if len(recon.shape) == 3 and recon.shape[2] == 3:
                recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
            else:
                recon_bgr = recon
            if orig.shape[:2] != recon_bgr.shape[:2]:
                recon_bgr = cv2.resize(
                    recon_bgr, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            recon_bgr = recon_bgr.astype(np.float32)
            mse_frame = np.mean((orig - recon_bgr) ** 2)
            mse_values.append(mse_frame)
            orig_uint8 = orig.astype(np.uint8)
            recon_uint8 = recon_bgr.astype(np.uint8)
            orig_dists, orig_lpips = _frame_to_tensors(orig_uint8, device)
            recon_dists, recon_lpips = _frame_to_tensors(recon_uint8, device)
            dists_val = float(dists_model(orig_dists, recon_dists).item())
            lpips_vgg_val = float(lpips_vgg(orig_lpips, recon_lpips).item())
            lpips_alex_val = float(lpips_alex(orig_lpips, recon_lpips).item())
            dists_values.append(dists_val)
            lpips_vgg_values.append(lpips_vgg_val)
            lpips_alex_values.append(lpips_alex_val)
    avg_mse = float(np.mean(mse_values))
    psnr = (
        float("inf")
        if avg_mse <= 0
        else 20.0 * math.log10(255.0) - 10.0 * math.log10(avg_mse)
    )
    return {
        "psnr": float(psnr),
        "mse": avg_mse,
        "dists": float(np.mean(dists_values)),
        "lpips_vgg": float(np.mean(lpips_vgg_values)),
        "lpips_alex": float(np.mean(lpips_alex_values)),
        "num_frames": num_frames,
    }


def convert_video_to_frames_array(video: np.ndarray) -> np.ndarray:
    if video.ndim == 5:
        if video.shape[1] == 3 or video.shape[1] == 1:
            video = video[0]
            video = np.transpose(video, (1, 2, 3, 0))
        elif video.shape[2] == 3 or video.shape[2] == 1:
            video = video[0]
            video = np.transpose(video, (0, 2, 3, 1))
        elif video.shape[4] == 3 or video.shape[4] == 1:
            video = video[0]
        else:
            video = video[0]
            video = np.transpose(video, (1, 2, 3, 0))
    elif video.ndim == 4:
        if video.shape[1] == 3 or video.shape[1] == 1:
            video = np.transpose(video, (0, 2, 3, 1))
        elif video.shape[0] == 3 or video.shape[0] == 1:
            video = np.transpose(video, (1, 2, 3, 0))
    elif video.ndim == 3:
        if video.shape[0] == 3 or video.shape[0] == 1:
            video = np.transpose(video, (1, 2, 0))
        video = video[np.newaxis, ...]
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

    frames_array = None
    if args.compute_psnr:
        if isinstance(frames, torch.Tensor):
            frames_array = frames.cpu().numpy()
            if frames_array.ndim == 4 and frames_array.shape[1] == 3:
                frames_array = np.transpose(frames_array, (0, 2, 3, 1))
            if frames_array.dtype != np.uint8:
                if frames_array.max() <= 1.0:
                    frames_array = (frames_array * 255.0).clip(0, 255)
                frames_array = frames_array.astype(np.uint8)
        elif isinstance(frames, np.ndarray):
            frames_array = frames.copy()
            if frames_array.dtype != np.uint8:
                if frames_array.max() <= 1.0:
                    frames_array = (frames_array * 255.0).clip(0, 255)
                frames_array = frames_array.astype(np.uint8)
        else:
            frames_list = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frames_list.append(np.array(frame))
                elif isinstance(frame, torch.Tensor):
                    arr = frame.cpu().numpy()
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.dtype != np.uint8:
                        if arr.max() <= 1.0:
                            arr = (arr * 255.0).clip(0, 255)
                        arr = arr.astype(np.uint8)
                    frames_list.append(arr)
                elif isinstance(frame, np.ndarray):
                    arr = frame.copy()
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
    print(f"Running dual-LoRA reconstruction on {device} ({args.dtype})")

    pipe = load_pipeline(args, device)
    prepare_lora_dual(pipe, args, args.active_lora_file)

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
        pipe, args, device, caption_text
    )

    print(f"Saved reconstructed video to {output_path}")

    if args.compute_psnr:
        if args.original_frames_dir is None:
            raise ValueError(
                "--original_frames_dir must be provided when --compute_psnr is set"
            )
        print(f"\nLoading original frames from {args.original_frames_dir}...")
        original_frames = load_original_frames(args.original_frames_dir, args.num_frames)
        if reconstructed_frames is None:
            raise RuntimeError(
                "Failed to extract reconstructed frames for PSNR computation"
            )
        print("Computing metrics for final reconstruction...")
        metrics = compute_psnr_metrics(
            original_frames, reconstructed_frames, device=device
        )
        print(f"PSNR: {metrics['psnr']:.4f}")


if __name__ == "__main__":
    main()
