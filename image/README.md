# Image pipeline

## Environment

Python 3 with CUDA recommended. Install dependencies from this directory:

```bash
cd /data1/Compression_clean/image
pip install -r requirements.txt
```

`requirements.txt` pins the core stack: `torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `pillow`, `numpy`, `huggingface-hub`, `sentencepiece`.

Set **`CACHE_DIR`** (in the shell scripts) to a folder with pretrained Qwen-Image weights compatible with the pipeline, or adjust the default `../pretrained_models`.

## How to run

From `/data1/Compression_clean/image`:

**Multi-GPU training** (uses `torchrun`):

```bash
# Override paths as needed
DATA_ROOT=/path/to/images \
CACHE_DIR=/path/to/pretrained_models \
bash train_multi.sh
```

**Reconstruction** with a trained LoRA checkpoint:

```bash
INIT_LORA_PATH=/path/to/lora_step_*.safetensors \
DATA_ROOT=/path/to/images \
CACHE_DIR=/path/to/pretrained_models \
bash reconstruct_multi.sh
```

Key env vars are documented in `train_multi.sh` and `reconstruct_multi.sh` (`NUM_GPUS`, `OUT_DIR`, `DESCRIPTION_FILE`, `WIDTH`, `HEIGHT`, etc.).

### Subfolders

- [`descriptions/`](descriptions/README.md) — caption / description text files
- [`pipelines/`](pipelines/README.md) — Diffusers-style pipeline code
- [`unilora/`](unilora/README.md) — UniLoRA layer and model helpers
