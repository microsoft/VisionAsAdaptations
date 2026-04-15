"""
Configuration parser for train_lora_single.py and train_lora_multi.py
Handles command-line arguments for LoRA training.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter on Qwen-Image model"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the input image (for single-image training)"
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Text caption/description of the image (for single-image training)"
    )
    parser.add_argument(
        "--description_file",
        type=str,
        default="../data/descriptions.txt",
        help="Path to descriptions file with image_path|caption per line (for multi-image training)"
    )
    parser.add_argument(
        "--caption_start",
        type=int,
        default=None,
        help="1-based start index of captions to use from the descriptions file (inclusive)",
    )
    parser.add_argument(
        "--caption_end",
        type=int,
        default=None,
        help="1-based end index of captions to use from the descriptions file (inclusive)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data",
        help="Root directory for image paths in descriptions file"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./qwenimage_lora_multi_768x512",
        help="Output directory for saving LoRA weights and samples"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../pretrained_models",
        help="Cache directory for pretrained models"
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
    
    # Training hyperparameters
    parser.add_argument(
        "--train_steps",
        type=int,
        default=300,
        help="Number of training steps (300-1500 typical for single-image LoRA)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (typically 1 for single-image training)"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps per optimizer update"
    )
    
    # UniLoRA configuration
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (8/16/32). Higher = more capacity & overfitting risk"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (typically same as rank)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--theta_d_length",
        type=int,
        default=65536,
        help="Length of the shared UniLoRA parameter vector (theta_d)"
    )
    parser.add_argument(
        "--proj_seed",
        type=int,
        default=42,
        help="Seed used to generate the fixed UniLoRA projection indices"
    )
    parser.add_argument(
        "--init_theta_d_bound",
        type=float,
        default=0.02,
        help="Uniform init bound for theta_d (sampled from [-bound, bound])"
    )
    parser.add_argument(
        "--lora_preset",
        type=str,
        default="qkvo",
        choices=["qkv", "qkvo", "qkvo_mlp"],
        help="Which projections to adapt: qkv (Q,K,V only), qkvo (Q,K,V + output), qkvo_mlp (all including FFN)"
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=8,
        help="Adapt only the last K transformer blocks (set to 0 or negative for all blocks)"
    )
    
    # Image dimensions
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width (must be multiple of 64)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (must be multiple of 64)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save checkpoint every N steps"
    )
    
    # Inference settings (for final test)
    parser.add_argument(
        "--test_steps",
        type=int,
        default=50,
        help="Number of inference steps for final test generation"
    )
    parser.add_argument(
        "--test_cfg",
        type=float,
        default=4.0,
        help="CFG scale for final test generation"
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate sample images after training (only on rank 0)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.width % 64 != 0:
        raise ValueError(f"Width must be multiple of 64, got {args.width}")
    if args.height % 64 != 0:
        raise ValueError(f"Height must be multiple of 64, got {args.height}")
    
    # Convert last_k to None if it's <= 0 (adapt all blocks)
    if args.last_k <= 0:
        args.last_k = None

    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    
    return args


if __name__ == "__main__":
    # Test the argument parser
    args = parse_args()
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
