"""DDP UniLoRA overfitting script for Wan2.1 on a single video.

Clean copy for Compression_clean stage-2 training.
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms as T
from safetensors.torch import save_file, load_file

from diffusers import AutoencoderKLWan
from pipeline_wan import WanPipeline
from diffusers.utils import export_to_video

from unilora_utils import (
    inject_unilora,
    build_lora_regex,
    collect_unilora_state_dict,
    load_unilora_state_dict,
    count_trainable_parameters,
    compute_unilora_rate,
)
from config_train import parse_args


def _load_caption_text(args) -> str:
    """Return caption text from --caption or --caption_file (file takes precedence)."""
    if getattr(args, "caption_file", None):
        path = Path(args.caption_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Caption file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    if args.caption:
        return args.caption
    raise ValueError("Provide --caption or --caption_file")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def compute_psnr_between_frames(
    original_paths: Sequence[Path],
    reconstructed_frames,
    height: int,
    width: int,
) -> float:
    """Compute average PSNR between original frame files and reconstructed output."""
    import math

    mse_values = []
    num = min(len(original_paths), len(reconstructed_frames))
    for i in range(num):
        orig = cv2.imread(str(original_paths[i]), cv2.IMREAD_COLOR)
        if orig is None:
            continue
        orig = cv2.resize(orig, (width, height), interpolation=cv2.INTER_LINEAR)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32)

        rec = reconstructed_frames[i]
        if isinstance(rec, Image.Image):
            rec = np.array(rec).astype(np.float32)
        elif isinstance(rec, torch.Tensor):
            rec = rec.detach().cpu().float().numpy()
            if rec.ndim == 3 and rec.shape[0] in (1, 3) and rec.shape[-1] not in (1, 3):
                rec = np.moveaxis(rec, 0, -1)
            if rec.max() <= 1.0:
                rec = rec * 255.0
            rec = np.clip(rec, 0, 255).astype(np.float32)
        else:
            rec = np.asarray(rec).astype(np.float32)
            if rec.ndim == 3 and rec.shape[0] in (1, 3) and rec.shape[-1] not in (1, 3):
                rec = np.moveaxis(rec, 0, -1)
            if rec.max() <= 1.0:
                rec = rec * 255.0
            rec = np.clip(rec, 0, 255).astype(np.float32)

        if orig_rgb.shape != rec.shape:
            rec = cv2.resize(rec, (orig_rgb.shape[1], orig_rgb.shape[0]))

        mse_values.append(np.mean((orig_rgb - rec) ** 2))

    if not mse_values:
        return 0.0
    avg_mse = float(np.mean(mse_values))
    if avg_mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(avg_mse)


def setup_ddp():
    """Initialize distributed training."""

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, device, world_size, rank


def sorted_frame_paths(frames_dir: Path) -> List[Path]:
    frames: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        frames.extend(frames_dir.glob(f"*{ext}"))
        frames.extend(frames_dir.glob(f"*{ext.upper()}"))
    frames = sorted({p.resolve() for p in frames})
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    return frames


def load_video_tensor(
    frames_dir: Path,
    num_frames: Optional[int],
    frame_stride: int,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, Sequence[Path]]:
    """Load frames, resize, and stack into (1, 3, T, H, W)."""

    transform = T.Compose(
        [
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ]
    )

    frame_paths = sorted_frame_paths(frames_dir)[::frame_stride]
    if num_frames is not None:
        frame_paths = frame_paths[:num_frames]
    if not frame_paths:
        raise ValueError("No frames selected after applying stride/num_frames constraints")

    pixels: List[torch.Tensor] = []
    for idx, path in enumerate(frame_paths):
        with Image.open(path) as img:
            if idx == 0:
                orig_w, orig_h = img.size
                print(
                f"Loading frames from: {frames_dir}, resize to {width}x{height}, "
                f"original frame resolution is {orig_w}x{orig_h}"
                )
            pixels.append(transform(img.convert("RGB")))
    video = torch.stack(pixels, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return video, frame_paths


def encode_video_latents(
    pipe: WanPipeline,
    video_tensor: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Encode video frames into Wan latent space and duplicate per GPU batch."""

    video = video_tensor.to(device=device, dtype=pipe.vae.dtype)
    video = video * 2.0 - 1.0

    with torch.no_grad():
        encoded = pipe.vae.encode(video).latent_dist.sample()

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(encoded.device, encoded.dtype)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(encoded.device, encoded.dtype)
    )
    encoded = (encoded - latents_mean) * (1.0 / latents_std)
    encoded = encoded.to(device=device, dtype=dtype)
    encoded = encoded.expand(batch_size, -1, -1, -1, -1).contiguous()
    return encoded


def main():
    args = parse_args()
    caption_text = _load_caption_text(args)
    args.caption = caption_text  # ensure downstream uses resolved caption

    local_rank, device, world_size, rank = setup_ddp()
    torch.manual_seed(args.seed + rank)

    if rank == 0:
        print(args)
        os.makedirs(args.out_dir, exist_ok=True)

    dist.barrier()

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_id = args.wan_model_id
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    pipe.to(device)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if pipe.transformer is None:
        raise RuntimeError("WanPipeline missing transformer component")
    pipe.transformer.requires_grad_(False)

    latents: Optional[torch.Tensor] = None
    latents_shape: Optional[Tuple[int, ...]] = None
    selected_frames: Sequence[Path] = []

    if rank == 0:
        frames_dir = Path(args.frames_dir).expanduser()
        video_tensor, selected_frames = load_video_tensor(
            frames_dir,
            args.num_frames,
            args.frame_stride,
            args.width,
            args.height,
        )
        latents = encode_video_latents(pipe, video_tensor, args.batch_size, dtype, device)
        latents_shape = latents.shape
        print(
            f"[Rank 0] Encoded {len(selected_frames)} frame(s) from {frames_dir.name} -> latents {latents_shape}"
        )

    shape_list = [latents_shape]
    dist.broadcast_object_list(shape_list, src=0)
    latents_shape = shape_list[0]
    if latents_shape is None:
        raise RuntimeError("Latent shape broadcast failed")

    if rank != 0:
        latents = torch.empty(latents_shape, device=device, dtype=dtype)

    dist.broadcast(latents, src=0)
    dist.barrier()

    if rank == 0:
        print("[Rank 0] Broadcasted latents to all GPUs")

    if rank == 0:
        print("[Rank 0] Injecting UniLoRA adapter...")
    torch.manual_seed(args.seed)

    total_blocks = len(pipe.transformer.blocks)
    target_regex = build_lora_regex(
        total_blocks=total_blocks,
        last_k=args.last_k,
        preset=args.lora_preset,
        model="wan",
    )

    replaced = inject_unilora(
        pipe.transformer,
        target_regex=target_regex,
        r=args.rank,
        theta_d_length=args.theta_d_length,
        proj_seed=args.proj_seed,
        dropout=args.dropout,
        init_theta_d_bound=args.init_theta_d_bound,
        enable_rate=args.lambda_rate > 0,
    )
    
    if rank == 0:
        print(f"[UniLoRA] Replaced {len(replaced)} Linear modules:")
        for name in replaced[:5]:
            print(f"  - {name}")
        if len(replaced) > 5:
            print(f"  ... and {len(replaced) - 5} more")
        print(
            f"[UniLoRA] theta_d length={args.theta_d_length}, proj_seed={args.proj_seed}"
        )

    if args.init_lora_path:
        if not os.path.isfile(args.init_lora_path):
            raise FileNotFoundError(f"init_lora_path not found: {args.init_lora_path}")
        if rank == 0:
            print(f"Loading initial UniLoRA weights from {args.init_lora_path}")
        init_state = load_file(args.init_lora_path)
        load_unilora_state_dict(pipe.transformer, init_state)
        if rank == 0:
            print("Loaded initial UniLoRA weights")
    dist.barrier()

    # Verify UniLoRA initialization consistency across all GPUs
    theta_d_sum = pipe.transformer.unilora_theta_d["default"].sum().item()
    if rank == 0:
        print(f"[Rank 0] theta_d sum after init: {theta_d_sum:.6f}")
    dist.barrier()

    # Enable grads only for LoRA params
    trainable, total = count_trainable_parameters(pipe.transformer)
    if trainable == 0:
        raise RuntimeError("No LoRA params found. Check target_modules names.")
    
    if rank == 0:
        effective_batch = args.batch_size * world_size * args.grad_accum_steps
        print(f"Trainable LoRA params: {trainable:8d} / {total/1e6:.2f}M total")
        print(
            "World size: {ws}, Batch size per GPU: {bs}, Gradient accumulation: {ga}, Effective batch size: {eff}".format(
                ws=world_size,
                bs=args.batch_size,
                ga=args.grad_accum_steps,
                eff=effective_batch,
            )
        )
        if args.lambda_rate > 0:
            print(
                f"Rate constraint enabled: lambda_rate={args.lambda_rate}, warmup={args.lambda_rate_warmup}"
            )

    try:
        pipe.transformer.enable_gradient_checkpointing()
        if rank == 0:
            print("Enabled gradient checkpointing.")
    except AttributeError:
        if rank == 0:
            print("Gradient checkpointing not supported for this model.")

    # Wrap transformer with DDP (this will broadcast model state automatically)
    ddp_transformer = DDP(
        pipe.transformer, 
        device_ids=[local_rank], 
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=True  # Ensure buffers are synced
    )
    
    if rank == 0:
        print("[Rank 0] DDP wrapper created, model state broadcasted to all GPUs")
    dist.barrier()
    
    base_params = [p for p in ddp_transformer.parameters() if p.requires_grad]

    if not base_params:
        raise RuntimeError("No UniLoRA parameters found for optimizer setup")

    optimizer = AdamW([
        {"params": base_params, "lr": args.lr}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    if rank == 0:
        print(
            "Optimizer params: {base:,}".format(
                base=sum(p.numel() for p in base_params),
            )
        )

    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=args.caption,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=args.batch_size,
            device=device,
            dtype=dtype,
            max_sequence_length=512,
        )
    
    dist.barrier()
    if rank == 0:
        print("=" * 60)
        print("Starting training...")
        print("Note: Loss values may differ slightly across GPUs (different noise)")
        print("=" * 60)

    ddp_transformer.train()

    torch.manual_seed(args.seed + rank)
    if rank == 0:
        print("[Training] Sampling unique noise per rank for diversity")

    dist.barrier()

    grad_accum_steps = max(1, args.grad_accum_steps)
    if rank == 0 and grad_accum_steps > 1:
        print(f"[Training] Using gradient accumulation steps: {grad_accum_steps}")

    optimizer.zero_grad(set_to_none=True)

    # Total pixels for rate logging (frames x H x W)
    frames_count = args.num_frames if args.num_frames and args.num_frames > 0 else 0
    total_pixels = max(0, frames_count) * args.height * args.width

    for step in range(1, args.train_steps + 1):
        accum_total_loss = 0.0
        accum_model_loss = 0.0
        accum_rate_loss = 0.0
        accum_bpp = 0.0

        for accum_idx in range(grad_accum_steps):
            noise = torch.randn_like(latents)
            timestep = torch.rand((latents.size(0),), device=device, dtype=latents.dtype)
            mix = timestep.view(-1, 1, 1, 1, 1)
            noisy_latents = (1 - mix) * latents + mix * noise

            pred_v = ddp_transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep * 1000,
                return_dict=False,
            )[0]

            model_loss = F.mse_loss(pred_v, noise - latents, reduction="mean")

            rate_loss = torch.zeros((), device=device, dtype=torch.float32)
            bits_per_pixel = torch.zeros((), device=device, dtype=torch.float32)
            if args.lambda_rate > 0:
                rate_info = compute_unilora_rate(ddp_transformer.module, use_cached=False)
                rate_loss = rate_info["bits_per_param"]
                if total_pixels > 0:
                    bits_per_pixel = (rate_info["total_bits"] / float(total_pixels)).detach()

            rate_factor = 1.0
            if args.lambda_rate_warmup > 0:
                rate_factor = min(1.0, max(0.0, (step - 1) / float(max(1, args.lambda_rate_warmup))))

            if latents_shape is not None and len(latents_shape) >= 3 and latents_shape[2] > 0:
                rate_factor = rate_factor * args.lambda_rate

            total_loss = model_loss + rate_factor * rate_loss
            (total_loss / grad_accum_steps).backward()

            accum_total_loss += total_loss.item()
            accum_model_loss += model_loss.item()
            accum_rate_loss += rate_loss.item()
            accum_bpp += bits_per_pixel.item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        metrics = torch.tensor(
            [
                accum_total_loss / grad_accum_steps,
                accum_model_loss / grad_accum_steps,
                accum_rate_loss / grad_accum_steps,
                accum_bpp / grad_accum_steps,
            ],
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        (
            avg_loss,
            avg_model_loss,
            avg_rate_loss,
            avg_bpp,
        ) = metrics.tolist()
        
        # Only rank 0 prints and saves
        if rank == 0:
            if step % 1 == 0 or step == 1:
                print(
                    (
                        f"[{step}/{args.train_steps}] loss={avg_loss:.4f} model={avg_model_loss:.4f} "
                        f"rate={avg_rate_loss:.4f} bpp={avg_bpp:.6f} rate_factor={rate_factor:.4f} "
                        f"(averaged over {world_size} GPUs)"
                    )
                )

            if step % args.save_every == 0:
                lora_sd = collect_unilora_state_dict(ddp_transformer.module)
                save_file(lora_sd, os.path.join(args.out_dir, f"lora_step_{step}.safetensors"))
                print(f"Saved checkpoint at step {step}")
        
        dist.barrier()  # Sync all GPUs

    # ---------------- Save LoRA ----------------
    if rank == 0:
        lora_sd = collect_unilora_state_dict(ddp_transformer.module)
        save_file(lora_sd, os.path.join(args.out_dir, "lora_weights.safetensors"))
        print(f"Saved UniLoRA to: {os.path.join(args.out_dir, 'lora_weights.safetensors')}")

    dist.barrier()  # Wait for rank 0 to save

    if rank == 0:
        ddp_transformer.module.eval()
        with torch.no_grad():
            lora_sd = load_file(os.path.join(args.out_dir, "lora_weights.safetensors"))
            load_unilora_state_dict(ddp_transformer.module, lora_sd)
            pipe.transformer = ddp_transformer.module
            # Ensure pipeline modules and inputs use the training dtype (e.g., bf16 on CUDA)
            pipe.to(device=device, dtype=dtype)

            inference_frames = len(selected_frames) if selected_frames else (args.num_frames or 81)
            output = pipe(
                prompt=args.caption,
                negative_prompt=args.negative_prompt or None,
                height=args.height,
                width=args.width,
                num_frames=inference_frames,
                num_inference_steps=args.test_steps,
                guidance_scale=args.test_cfg,
            ).frames[0]
            out_path = os.path.join(args.out_dir, "sample_after_training.mp4")
            export_to_video(output, out_path, fps=args.video_fps)
            print("Saved sample video:", out_path)
            # Save per-frame PNGs
            frames_dir = os.path.join(args.out_dir, "sample_frames_after_training")
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame in enumerate(output):
                # Robustly convert to PIL and save
                if isinstance(frame, Image.Image):
                    img = frame
                else:
                    arr = torch.tensor(frame).detach().cpu().float().numpy() if torch.is_tensor(frame) else np.asarray(frame)
                    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                        arr = np.moveaxis(arr, 0, -1)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0.0, 1.0)
                        arr = (arr * 255.0 + 0.5).astype(np.uint8)
                    img = Image.fromarray(arr)
                img.save(os.path.join(frames_dir, f"frame_{idx:05d}.png"))
            print("Saved per-frame PNGs to:", frames_dir)

            psnr = compute_psnr_between_frames(
                selected_frames, output, args.height, args.width,
            )
            print(f"[PSNR] Reconstruction PSNR: {psnr:.4f} dB")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
