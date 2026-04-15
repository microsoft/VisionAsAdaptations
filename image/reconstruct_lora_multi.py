"""Generate reconstructions for all samples listed in a descriptions file using a trained UniLoRA."""
import argparse
import copy
import os
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import load_file

from pipelines.pipeline_qwenimage import QwenImagePipeline
from unilora_utils import inject_unilora, build_lora_regex, load_unilora_state_dict
from data_utils import load_image_descriptions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct multiple images using a trained LoRA adapter"
    )

    parser.add_argument(
        "--init_lora_path",
        type=str,
        default=None,
        help="Optional path to an existing LoRA checkpoint to initialize from",
    )
    parser.add_argument(
        "--init_update_path",
        type=Path,
        nargs="+",
        default=None,
        help="One or more LoRA .safetensors files; all will be loaded in order and accumulated",
    )
    parser.add_argument(
        "--description_file",
        type=Path,
        default=Path("../data/descriptions.txt"),
        help="Path to descriptions file used during training (image_path|caption per line)",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("../data"),
        help="Root directory prepended to relative paths inside the descriptions file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where reconstructed images are written",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt applied to every reconstruction",
    )
    parser.add_argument(
        "--caption_start",
        type=int,
        default=None,
        help="If provided, use a substring of caption starting at this index (inclusive)",
    )
    parser.add_argument(
        "--caption_end",
        type=int,
        default=None,
        help="If provided, use a substring of caption ending at this index (exclusive)",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale for Qwen-Image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit on how many descriptions to reconstruct (process order as listed)",
    )

    # UniLoRA configuration (must match training)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--theta_d_length", type=int, default=65536)
    parser.add_argument("--proj_seed", type=int, default=42)
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lora_preset", choices=["qkv", "qkvo", "qkvo_mlp"], default="qkvo")
    parser.add_argument("--last_k", type=int, default=8, help="Number of final transformer blocks adapted")

    parser.add_argument(
        "--filename_prefix",
        type=str,
        default="recon",
        help="Prefix used for output image filenames",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("../pretrained_models"),
        help="Hugging Face cache directory for Qwen/Qwen-Image weights",
    )

    args = parser.parse_args()

    if args.width % 64 != 0 or args.height % 64 != 0:
        raise ValueError("Width/height must be divisible by 64.")

    if args.last_k is not None and args.last_k <= 0:
        args.last_k = None

    return args


def sanitize_sample_name(image_path: str, idx: int) -> str:
    path = Path(image_path)
    parent = path.parent.name
    stem = path.stem
    parts: List[str] = []
    if parent:
        parts.append(parent)
    if stem:
        parts.append(stem)
    if not parts:
        parts = [f"sample_{idx:04d}"]
    safe = "_".join(parts)
    safe = safe.replace(" ", "_").lower()
    return safe


def load_descriptions(description_file: Path, data_root: Path, caption_start=None, caption_end=None) -> List[Tuple[str, str]]:
    if not description_file.exists():
        raise FileNotFoundError(f"Description file not found: {description_file}")

    image_data_list = load_image_descriptions(
        description_file=str(description_file),
        data_root=str(data_root),
        image_path=None,
        caption=None,
        rank=0,
        caption_start=caption_start,
        caption_end=caption_end,
    )

    if not image_data_list:
        raise RuntimeError("No entries loaded from descriptions file.")

    return image_data_list


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device2 = "cuda:1" if torch.cuda.device_count() > 1 else device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"Using device: {device} for transformer (with LoRA)")

    # Load base pipeline - VAE, text encoder on GPU 0, transformer on GPU 0
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=dtype,
        cache_dir=str(args.cache_dir),
    ).to(device)
    
    if args.cfg > 1.0:
        print(f"Using device: {device2} for transformer2 (negative prompt)")
        # Create transformer2 on GPU 1 to avoid OOM
        # Move to CPU first, then deepcopy, then move to GPU1 to avoid OOM during copy
        print("Creating transformer2 on separate GPU...")
        print("  Step 1: Moving transformer to CPU temporarily...")
        pipe.transformer.to('cpu')
        print("  Step 2: Deep copying transformer...")
        pipe.transformer2 = copy.deepcopy(pipe.transformer)
        print("  Step 3: Moving transformer back to GPU 0...")
        pipe.transformer.to(device)
        print("  Step 4: Moving transformer2 to GPU 1...")
        pipe.transformer2.to(device2)
        pipe.transformer2.requires_grad_(False)
        pipe.transformer2.eval()
        print(f"transformer on {device}, transformer2 on {device2}")

    # Inject UniLoRA structure based on training config
    total_blocks = len(pipe.transformer.transformer_blocks)
    target_regex = build_lora_regex(
        total_blocks=total_blocks,
        last_k=args.last_k,
        preset=args.lora_preset
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

    print(f"[UniLoRA] Injected UniLoRA into {len(replaced)} modules")

    def _load_unilora_weights(module):
        if args.init_update_path:
            print(f"Loading UniLoRA weights (count={len(args.init_update_path)})")
            for wpath in args.init_update_path:
                state = load_file(str(wpath))
                load_unilora_state_dict(module, state)
        if args.init_lora_path:
            if not os.path.isfile(args.init_lora_path):
                raise FileNotFoundError(f"init_lora_path not found: {args.init_lora_path}")
            state = load_file(args.init_lora_path)
            load_unilora_state_dict(module, state)
            print(f"UniLoRA weights loaded successfully from {args.init_lora_path}")

    _load_unilora_weights(pipe.transformer)

    
    # Load descriptions
    image_data_list = load_descriptions(
        args.description_file,
        args.data_root,
        caption_start=args.caption_start,
        caption_end=args.caption_end,
    )
    if args.max_images is not None:
        image_data_list = image_data_list[: args.max_images]
        print(f"Limiting reconstruction to first {len(image_data_list)} entries")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe.transformer.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Starting reconstruction for {len(image_data_list)} samples...")
    with torch.no_grad():
        for idx, (img_path, caption) in enumerate(image_data_list, start=1):
            sample_name = sanitize_sample_name(img_path, idx)
            out_path = args.output_dir / f"{args.filename_prefix}_{sample_name}.png"

            print(f"[{idx}/{len(image_data_list)}] Prompt from {img_path} -> {out_path}")
            print("  Caption:", caption)
            image = pipe(
                prompt=caption,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                true_cfg_scale=args.cfg,
            ).images[0]

            image.save(out_path)
            print(f"Saved image to {out_path}")

    print("All reconstructions completed.")


if __name__ == "__main__":
    main()
