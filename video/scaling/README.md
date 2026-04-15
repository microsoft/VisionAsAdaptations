# Video scaling reconstruction

Reconstruct video from a trained LoRA using the Wan pipeline with optional **scaling** in the encoder (`pipeline_wan_scaling_encode.py`, `reconstruct_lora_single_scaling_encode.py`).

## Environment

Same as the rest of the video tree:

```bash
cd /data1/Compression_clean/video
pip install -r requirements.txt
```

GPU recommended.

## How to run

From this directory:

```bash
cd /data1/Compression_clean/video/scaling
# Usage: bash sample.sh <SEQUENCE_NAME> [diffusion_steps]
# Set WEIGHTS_DIR, CACHE_DIR, ORIGINAL_FRAMES_DIR, SCALING=true/false as needed
bash sample.sh HoneyBee 50
```

The script picks the latest `lora_step_*.safetensors` under `WEIGHTS_DIR` and writes a log plus `reconstructed_factorized_<NAME>.mp4` by default. Override env vars at the top of `sample.sh` for your paths.

### Subfolders

- [`descriptions/`](descriptions/README.md) — `descriptions_<NAME>.txt` for captions
- [`unilora/`](unilora/README.md) — local LoRA modules for this pipeline
