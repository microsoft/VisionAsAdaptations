"""
Data loading and preprocessing utilities for LoRA training.
Handles image loading, VAE encoding, and prompt encoding.
"""

import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T


def load_image_descriptions(
    description_file,
    data_root,
    image_path=None,
    caption=None,
    rank=0,
    caption_start=None,
    caption_end=None,
):
    """
    Load image paths and captions from descriptions file or command-line args.
    
    Args:
        description_file: Path to descriptions.txt file
        data_root: Root directory for relative image paths
        image_path: Single image path (for single-image mode)
        caption: Single caption (for single-image mode)
        rank: DDP rank (only rank 0 does the loading)
    
    Returns:
        List of (image_path, caption) tuples
    """
    image_data_list = []
    
    if rank == 0:
        if description_file is not None and os.path.exists(description_file):
            # Multi-image mode: read from descriptions.txt
            print(f"[Rank 0] Reading descriptions from: {description_file}")
            with open(description_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) >= 2:
                        rel_path = parts[0].strip()
                        caption_text = parts[1].strip()
                        full_path = os.path.join(data_root, rel_path)
                        if os.path.exists(full_path):
                            image_data_list.append((full_path, caption_text))
                            # print(f"  - Added: {rel_path}")
                        else:
                            print(f"  - Warning: Image not found: {full_path}")
            total_loaded = len(image_data_list)
            # Apply optional slicing (1-based inclusive indices)
            if caption_start is not None or caption_end is not None:
                start_idx = 1 if caption_start is None else caption_start
                end_idx = total_loaded if caption_end is None else caption_end
                if start_idx < 1 or end_idx < start_idx:
                    raise ValueError("Invalid caption range: start must be >=1 and <= end")
                # Convert to 0-based slicing
                image_data_list = image_data_list[start_idx - 1 : end_idx]
                print(
                    f"[Rank 0] Sliced captions/images to indices {start_idx}-{end_idx} (kept {len(image_data_list)} of {total_loaded})"
                )
            print(f"[Rank 0] Loaded {len(image_data_list)} images from descriptions file")
        elif image_path is not None and caption is not None:
            # Single-image mode: use command-line args
            print(f"[Rank 0] Single image mode: {image_path}")
            image_data_list = [(image_path, caption)]
        else:
            raise ValueError("Must provide either --description_file or both --image_path and --caption")
    
    return image_data_list


def encode_images_to_latents(image_data_list, pipe, height, width, device, dtype, rank=0):
    """
    Encode images to VAE latents.
    
    Args:
        image_data_list: List of (image_path, caption) tuples
        pipe: QwenImagePipeline instance
        height: Target image height
        width: Target image width
        device: torch device
        dtype: torch dtype
        rank: DDP rank (only rank 0 does the encoding)
    
    Returns:
        Tuple of (latents_list, latents_shape, img_shapes)
    """
    latents_list = []
    latents_shape = None
    img_shapes = None
    
    if rank == 0:
        print("[Rank 0] Encoding images with VAE...")
        aug = T.Compose([
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ])
        
        with torch.no_grad():
            for idx, (img_path, caption) in enumerate(image_data_list):
                print(f"[Rank 0] Encoding image {idx+1}/{len(image_data_list)}: {img_path}")
                raw = Image.open(img_path).convert("RGB")
                pixel_values = aug(raw).unsqueeze(0).to(device=device, dtype=dtype)
                pixel_values_5d = pixel_values.unsqueeze(2) * 2 - 1.0
                
                latent = pipe.vae.encode(pixel_values_5d).latent_dist.sample()
                
                # Normalize latents
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latent.device, latent.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                    1, pipe.vae.config.z_dim, 1, 1, 1
                ).to(latent.device, latent.dtype)
                latent = (latent - latents_mean) * latents_std
                
                # Pack single latent
                channel, h, w = latent.shape[1], latent.shape[3], latent.shape[4]
                packed_latent = pipe._pack_latents(latent, 1, channel, h, w)
                latents_list.append(packed_latent)
                
                if idx == 0:
                    latents_shape = packed_latent.shape
                    print(f"[Rank 0] Single latent shape: {latents_shape}")
            
            # img_shapes should be based on original image dimensions
            img_shapes = [[(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]]
            print(f"[Rank 0] img_shapes: {img_shapes[0]}")
    
    return latents_list, latents_shape, img_shapes


def encode_prompts(image_data_list, pipe, device, rank=0, max_batch_size=64):
    """
    Encode all prompts/captions for the images.
    
    Args:
        image_data_list: List of (image_path, caption) tuples
        pipe: QwenImagePipeline instance
        device: torch device
        rank: DDP rank
        max_batch_size: Number of captions per encode call (avoid OOM)
    
    Returns:
        Tuple of (prompt_embeds_list, prompt_embeds_mask_list, padded_seq_len)
    """
    if not image_data_list:
        return [], [], 0

    prompt_embeds_list = []
    prompt_embeds_mask_list = []
    padded_seq_len = 0

    all_captions = [caption for _, caption in image_data_list]
    total = len(all_captions)
    effective_batch = max(1, max_batch_size)
    stored_embeds = []
    stored_masks = []

    if rank == 0:
        print(f"[Rank 0] Encoding {total} captions in batches of {effective_batch}")

    with torch.no_grad():
        for start in range(0, total, effective_batch):
            end = min(start + effective_batch, total)
            chunk_captions = all_captions[start:end]
            if rank == 0:
                print(f"[Rank 0]  - Captions {start+1}-{end} / {total}")

            chunk_embeds, chunk_masks = pipe.encode_prompt(
                prompt=chunk_captions,
                device=device,
                max_sequence_length=512,
            )

            padded_seq_len = max(padded_seq_len, chunk_embeds.shape[1])

            for idx in range(chunk_embeds.shape[0]):
                stored_embeds.append(chunk_embeds[idx:idx+1].clone())
                stored_masks.append(chunk_masks[idx:idx+1].clone())

    for idx, (embeds, mask) in enumerate(zip(stored_embeds, stored_masks)):
        seq_len = embeds.shape[1]
        if seq_len < padded_seq_len:
            pad_len = padded_seq_len - seq_len
            embeds = F.pad(embeds, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))

        prompt_embeds_list.append(embeds)
        prompt_embeds_mask_list.append(mask)

        if rank == 0:
            actual_tokens = mask.sum(dim=1).item()
            caption_preview = all_captions[idx]
            print(
                f"[Rank 0] Caption {idx+1}: '{caption_preview[:50]}...' "
                f"(actual tokens: {actual_tokens}, padded to: {padded_seq_len})"
            )

    if rank == 0:
        print(f"[Rank 0] Prompt embeddings padded sequence length: {padded_seq_len}")

    return prompt_embeds_list, prompt_embeds_mask_list, padded_seq_len


def broadcast_data_to_all_ranks(image_data_list, latents_list, latents_shape, num_images, 
                                 height, width, vae_scale_factor, device, dtype, rank):
    """
    Broadcast image data and latents from rank 0 to all other ranks.
    
    Args:
        image_data_list: List of (image_path, caption) tuples (populated on rank 0)
        latents_list: List of latent tensors (populated on rank 0)
        latents_shape: Shape of single latent tensor
        num_images: Number of images
        height: Image height
        width: Image width
        vae_scale_factor: VAE downsampling factor
        device: torch device
        dtype: torch dtype
        rank: DDP rank
    
    Returns:
        Tuple of (image_data_list, latents_list, img_shapes)
    """
    import torch.distributed as dist
    
    # Broadcast image_data_list to all ranks
    image_data_broadcast = [image_data_list]
    dist.broadcast_object_list(image_data_broadcast, src=0)
    image_data_list = image_data_broadcast[0]
    
    # Broadcast latents shape and count to all ranks
    shape_and_count = [latents_shape, num_images]
    dist.broadcast_object_list(shape_and_count, src=0)
    latents_shape, num_images = shape_and_count
    
    # Create empty tensors on other ranks
    if rank != 0:
        latents_list = [torch.empty(latents_shape, device=device, dtype=dtype) for _ in range(num_images)]
    
    # Broadcast all latents from rank 0 to all ranks
    for i in range(num_images):
        dist.broadcast(latents_list[i], src=0)
    
    # Create img_shapes on all ranks
    img_shapes = [[(1, height // vae_scale_factor // 2, width // vae_scale_factor // 2)]]
    
    if rank == 0:
        print(f"[Rank 0] Broadcasted {num_images} latents to all GPUs")
    
    return image_data_list, latents_list, img_shapes
