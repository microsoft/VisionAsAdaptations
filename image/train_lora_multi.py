# train_lora_multi.py
# Multi-image LoRA training with random batch sampling
# pip install -U diffusers transformers safetensors torch torchvision pillow

import os, math, random, logging
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.checkpoint as torch_checkpoint
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torchvision import transforms as T
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file, load_file

from pipelines.pipeline_qwenimage import QwenImagePipeline
from unilora_utils import (
    build_lora_regex,
    count_trainable_parameters,
    inject_unilora,
    collect_unilora_state_dict,
    load_unilora_state_dict,
)
from config_train import parse_args
from data_utils import (
    load_image_descriptions,
    encode_images_to_latents,
    encode_prompts,
    broadcast_data_to_all_ranks
)


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


def setup_logger(out_dir: str, rank: int) -> logging.LoggerAdapter:
    """Configure logger to write to stdout and rank-specific log file."""
    logger = logging.getLogger(f"train_lora_multi.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s][Rank %(rank)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"training_rank{rank}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logging.LoggerAdapter(logger, {"rank": rank})


def main():
    args = parse_args()
    print(args)
    
    # Setup DDP
    local_rank, device, world_size, rank = setup_ddp()
    
    # Set seed for reproducibility (rank-dependent for different noise per GPU)
    torch.manual_seed(args.seed + rank)
    
    logger = setup_logger(args.out_dir, rank)
    if rank == 0:
        logger.info("Arguments: %s", args)
    
    dist.barrier()  # Wait for all ranks to have logger/log file ready
    
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load pipeline on each GPU
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image", torch_dtype=dtype, cache_dir=args.cache_dir
    ).to(device)

    # Freeze everything
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    # ---------------- Load and encode data ----------------
    # Load image descriptions
    image_data_list = load_image_descriptions(
        description_file=args.description_file,
        data_root=args.data_root,
        image_path=args.image_path,
        caption=args.caption,
        rank=rank,
        caption_start=args.caption_start,
        caption_end=args.caption_end,
    )
    
    # Encode images to latents
    latents_list, latents_shape, img_shapes = encode_images_to_latents(
        image_data_list=image_data_list,
        pipe=pipe,
        height=args.height,
        width=args.width,
        device=device,
        dtype=dtype,
        rank=rank
    )
    
    # Get num_images on rank 0
    num_images = len(image_data_list) if rank == 0 else 0
    
    # Broadcast data to all ranks
    image_data_list, latents_list, img_shapes = broadcast_data_to_all_ranks(
        image_data_list=image_data_list,
        latents_list=latents_list,
        latents_shape=latents_shape,
        num_images=num_images,
        height=args.height,
        width=args.width,
        vae_scale_factor=pipe.vae_scale_factor,
        device=device,
        dtype=dtype,
        rank=rank
    )
    
    # Update num_images after broadcast (now valid on all ranks)
    num_images = len(image_data_list)
    
    if rank == 0:
        logger.info("Total images for training: %s", num_images)
    
    dist.barrier()

    # Inject UniLoRA on ALL ranks with the SAME seed for consistent initialization
    if rank == 0:
        logger.info("Injecting UniLoRA adapters...")

    torch.manual_seed(args.seed)  # Same seed on all ranks!

    total_blocks = len(pipe.transformer.transformer_blocks)
    target_regex = build_lora_regex(
        total_blocks=total_blocks,
        last_k=args.last_k,
        preset=args.lora_preset,
    )

    replaced = inject_unilora(
        pipe.transformer,
        target_regex=target_regex,
        r=args.rank,
        theta_d_length=args.theta_d_length,
        proj_seed=args.proj_seed,
        dropout=args.dropout,
        init_theta_d_bound=args.init_theta_d_bound,
    )

    if rank == 0:
        logger.info("[UniLoRA] Replaced %d Linear modules:", len(replaced))
        for name in replaced[:5]:
            logger.info("  - %s", name)
        if len(replaced) > 5:
            logger.info("  ... and %d more", len(replaced) - 5)
        logger.info(
            "[UniLoRA] theta_d length=%d, proj_seed=%d (alpha flag unused)",
            args.theta_d_length,
            args.proj_seed,
        )

    if args.init_lora_path:
        if not os.path.isfile(args.init_lora_path):
            raise FileNotFoundError(f"init_lora_path not found: {args.init_lora_path}")
        if rank == 0:
            logger.info("Loading initial UniLoRA weights from %s", args.init_lora_path)
        state = load_file(args.init_lora_path)
        load_unilora_state_dict(pipe.transformer, state)
        if rank == 0:
            logger.info("Loaded initial UniLoRA weights")
    dist.barrier()

    # Verify UniLoRA initialization consistency across all GPUs
    theta_d_sum = pipe.transformer.unilora_theta_d["default"].sum().item()
    logger.info("theta_d sum: %.6f", theta_d_sum)
    dist.barrier()

    # Enable grads only for LoRA params
    trainable, total = count_trainable_parameters(pipe.transformer)
    if trainable == 0:
        raise RuntimeError("No LoRA params found. Check target_modules names.")
    
    if rank == 0:
        logger.info(
            "Trainable LoRA params: %.2fM / %.2fM total",
            trainable / 1e6,
            total / 1e6,
        )
        logger.info(
            "World size: %d, Batch size per GPU: %d, Grad accum steps: %d, Effective batch size: %d",
            world_size,
            args.batch_size,
            args.grad_accum_steps,
            args.batch_size * args.grad_accum_steps * world_size,
        )

    try:
        pipe.transformer.enable_gradient_checkpointing()
        if rank == 0:
            logger.info("Enabled gradient checkpointing.")
    except AttributeError:
        if rank == 0:
            logger.warning("Gradient checkpointing not supported for this model.")

    # Wrap transformer with DDP (this will broadcast model state automatically)
    ddp_transformer = DDP(
        pipe.transformer, 
        device_ids=[local_rank], 
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=True  # Ensure buffers are synced
    )
    
    if rank == 0:
        logger.info("DDP wrapper created, model state broadcasted to all GPUs")
    dist.barrier()
    
    base_params = [p for p in ddp_transformer.parameters() if p.requires_grad]

    if not base_params:
        raise RuntimeError("No LoRA parameters found for optimizer setup")

    optimizer = AdamW([
        {"params": base_params, "lr": args.lr}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    if rank == 0:
        logger.info(
            "Optimizer params: %d (no gates)",
            sum(p.numel() for p in base_params),
        )

    # Encode all prompts using utility function
    prompt_embeds_list, prompt_embeds_mask_list, padded_seq_len = encode_prompts(
        image_data_list=image_data_list,
        pipe=pipe,
        device=device,
        rank=rank,
        max_batch_size=64,
    )
    
    dist.barrier()
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("Number of images: %s", num_images)
        logger.info("Batch size: %d", args.batch_size)
        logger.info("Note: Each batch randomly samples from all images")
        logger.info("=" * 60)

    ddp_transformer.train()

    # ---------------- Training loop ----------------
    # Each GPU uses different seed for noise (diversity in batch)
    torch.manual_seed(args.seed + rank)
    dist.barrier()
    grad_accum_steps = max(1, args.grad_accum_steps)

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_stats = torch.zeros(3, device=device, dtype=torch.float32)

        for accum_step in range(grad_accum_steps):
            # Randomly sample image indices for this micro-batch
            batch_indices = [random.randint(0, num_images - 1) for _ in range(args.batch_size)]

            # Build batch latents by concatenating selected samples
            batch_latents_list = [latents_list[idx] for idx in batch_indices]
            latents = torch.cat(batch_latents_list, dim=0)

            # Build batch prompt embeddings (now all have same length after padding)
            batch_prompt_embeds = torch.cat([prompt_embeds_list[idx] for idx in batch_indices], dim=0)
            batch_prompt_embeds_mask = torch.cat([prompt_embeds_mask_list[idx] for idx in batch_indices], dim=0)

            # Since all prompts are padded to the same length, txt_seq_lens is just [padded_seq_len] * batch_size
            batch_txt_seq_lens = [padded_seq_len] * args.batch_size

            # Build img_shapes for the batch
            batch_img_shapes = img_shapes * args.batch_size

            # Sample noise & timestep (different per GPU for batch diversity)
            noise = torch.randn_like(latents)
            t = torch.rand((latents.size(0), 1, 1), device=device, dtype=latents.dtype)
            noisy_latents = (1 - t) * latents + t * noise

            # Forward denoiser (MMDiT) - use DDP wrapper
            pred_v = ddp_transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=batch_prompt_embeds,
                encoder_hidden_states_mask=batch_prompt_embeds_mask,
                timestep=t[:, 0, 0],
                img_shapes=batch_img_shapes,
                txt_seq_lens=batch_txt_seq_lens,
                return_dict=False,
            )[0]

            model_loss = F.mse_loss(pred_v, noise - latents, reduction="mean")

            loss = model_loss

            # Scale loss so each optimizer step sees the averaged gradient
            (loss / grad_accum_steps).backward()

            loss_stats += torch.stack(
                (
                    loss.detach().to(torch.float32),
                    model_loss.detach().to(torch.float32),
                    torch.zeros((), device=device, dtype=torch.float32),
                )
            )

        optimizer.step()

        # Gather loss from all ranks and print averaged loss at rank 0
        metrics = torch.zeros(3, device=device, dtype=torch.float32)
        metrics[:3] = loss_stats / grad_accum_steps
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        (
            avg_loss,
            avg_model_loss,
            _,
        ) = metrics.tolist()

        # Only rank 0 prints and saves
        if rank == 0:
            if step % 1 == 0 or step == 1:
                logger.info(
                    "[%d/%d] loss=%.4f model_loss=%.4f (averaged over %d GPUs)",
                    step,
                    args.train_steps,
                    avg_loss,
                    avg_model_loss,
                    world_size,
                )

            if step % args.save_every == 0:
                # Collect from the underlying module (not DDP wrapper)
                lora_sd = collect_unilora_state_dict(ddp_transformer.module)
                save_file(lora_sd, os.path.join(args.out_dir, f"lora_step_{step}.safetensors"))
                logger.info("Saved checkpoint at step %d", step)

        dist.barrier()  # Sync all GPUs

    # ---------------- Save LoRA ----------------
    if rank == 0:
        lora_sd = collect_unilora_state_dict(ddp_transformer.module)
        save_file(lora_sd, os.path.join(args.out_dir, "lora_weights.safetensors"))
        logger.info("Saved UniLoRA to: %s", os.path.join(args.out_dir, "lora_weights.safetensors"))

    dist.barrier()  # Wait for rank 0 to save

    # ---------------- Quick smoke test (only on rank 0) ----------------
    if rank == 0:
        ddp_transformer.module.eval()
        with torch.no_grad():
            # Load the saved LoRA weights back
            lora_sd = load_file(os.path.join(args.out_dir, "lora_weights.safetensors"))
            load_unilora_state_dict(ddp_transformer.module, lora_sd)
            
            # Generate reconstruction for each training image
            for idx, (img_path, caption) in enumerate(image_data_list):
                logger.info("Generating reconstruction for image %d/%d...", idx + 1, num_images)
                img = pipe(
                    prompt=caption,
                    negative_prompt="",
                    width=args.width, 
                    height=args.height,
                    num_inference_steps=args.test_steps,
                    true_cfg_scale=args.test_cfg
                ).images[0]
                
                # Save with descriptive filename
                out_img = os.path.join(args.out_dir, f"sample_{idx}_after_training.png")
                img.save(out_img)
                logger.info("Saved reconstruction: %s", out_img)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
