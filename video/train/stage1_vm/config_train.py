"""
Configuration parser for train_lora_single.py
Handles command-line arguments for UniLoRA training on Wan videos.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter on Qwen-Image or Wan2.1 video model"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the input image (for single-image training)"
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="/data1/dataset/data-codec/benchmark/test_video/HEVC_B/png_original/ParkScene_1920x1080_24_resized_832x480",
        help="Directory containing ordered video frames for Wan2.1 overfitting"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Optional negative prompt used during sample generation"
    )
    parser.add_argument(
        "--description_file",
        type=str,
        default="../data/descriptions.txt",
        help="Path to descriptions file with image_path|caption per line (for multi-image training)"
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
        "--wan_model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model id or local path for Wan2.1 checkpoint"
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
        help="Number of gradient accumulation steps before each optimizer update"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames to sample from frames_dir (<=0 uses all frames)"
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Stride when sampling frames from frames_dir"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=81,
        help="Number of consecutive frames per chunk when using chunked video training",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="How many chunks to load (set >1 for multi-chunk training)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Text caption/description of the clip (used by single-video or fallback flows)"
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default=None,
        help="Path to a text file containing the caption; overrides --caption when set",
    )
    parser.add_argument(
        "--chunk_caption_dir",
        type=str,
        default=None,
        help="Directory containing chunk caption text files (chunk_01.txt, ...) for multi-chunk training",
    )
    
    # LoRA configuration
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (8/16/32). Higher = more capacity & overfitting risk"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="UniLoRA dropout rate"
    )
    parser.add_argument(
        "--lora_preset",
        type=str,
        default="qkvo",
        choices=["qkv", "qkvo", "qkvo_mlp"],
        help="Which projections to adapt: qkv (Q,K,V only), qkvo (Q,K,V + output), qkvo_mlp (all including FFN)"
    )
    parser.add_argument(
        "--theta_d_length",
        type=int,
        default=65536,
        help="Length of shared theta_d vector for UniLoRA"
    )
    parser.add_argument(
        "--proj_seed",
        type=int,
        default=42,
        help="Seed for deterministic UniLoRA index generation"
    )
    parser.add_argument(
        "--init_theta_d_bound",
        type=float,
        default=0.02,
        help="Uniform init bound for theta_d"
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=8,
        help="Adapt only the last K transformer blocks (set to 0 or negative for all blocks)"
    )
    parser.add_argument(
        "--init_lora_path",
        type=str,
        default=None,
        help="Optional path to initialize UniLoRA weights before training",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Resume training from this step (used with --init_lora_path for checkpoint resume)",
    )
    
    # Image dimensions
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image/video width (must be multiple of 64)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image/video height (must be multiple of 64)"
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=24,
        help="FPS used when exporting preview videos"
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
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Flow shift parameter for UniPC scheduler (3.0 for 480p, 5.0 for 720p)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.width % 16 != 0:
        raise ValueError(f"Width must be multiple of 64, got {args.width}")
    if args.height % 16 != 0:
        raise ValueError(f"Height must be multiple of 64, got {args.height}")
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if args.num_frames <= 0:
        args.num_frames = None
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if args.theta_d_length <= 0:
        raise ValueError("theta_d_length must be positive")
    if args.init_theta_d_bound <= 0:
        raise ValueError("init_theta_d_bound must be positive")
    if not args.caption and not args.caption_file:
        raise ValueError("Provide either --caption or --caption_file")
    if args.caption_file and not os.path.isfile(args.caption_file):
        raise ValueError(f"Caption file not found: {args.caption_file}")
    
    # Convert last_k to None if it's <= 0 (adapt all blocks)
    if args.last_k <= 0:
        args.last_k = None
    
    return args


if __name__ == "__main__":
    # Test the argument parser
    args = parse_args()
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
