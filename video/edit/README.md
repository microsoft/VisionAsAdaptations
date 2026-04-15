# Video editing (caption or merge LoRAs)

## Environment

```bash
cd /data1/Compression_clean/video
pip install -r requirements.txt
```

GPU recommended. `CACHE_DIR` in the shell scripts defaults to `../pretrained_models` relative to this folder.

## How to run

From `/data1/Compression_clean/video/edit`:

**Same LoRA, new caption:**

```bash
WEIGHTS=/path/to/lora.safetensors \
CAPTION="Your new prompt." \
bash edit_caption.sh
```

**Merge two LoRAs + new caption:**

```bash
WEIGHTS1=/path/to/first.safetensors \
WEIGHTS2=/path/to/second.safetensors \
CAPTION="Merged prompt." \
bash edit_merge.sh
```

Set `WAN_MODEL_ID`, `OUTPUT_VIDEO`, `ORIGINAL_FRAMES_DIR` (optional, for PSNR), and other env vars as documented in each `.sh` file.

### Subfolders

- [`descriptions/`](descriptions/README.md) — optional reference captions (not always required by the edit scripts)
- [`unilora/`](unilora/README.md) — LoRA helpers for the edit pipelines
