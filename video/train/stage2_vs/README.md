# stage2_vs (stage 2 training code)

## Environment

Install from **`/data1/Compression_clean/video/requirements.txt`**, or use the stage-2 pin file:

```bash
pip install -r /data1/Compression_clean/video/train/stage2_vs/requirements.txt
```

## How to run

Use the parent launcher (sets `torchrun` and paths correctly):

```bash
cd /data1/Compression_clean/video/train
# Optional: INIT_LORA_PATH=/path/to/stage1/lora_*.safetensors
bash stage2_train.sh
```

### Subfolders

- [`descriptions/`](descriptions/README.md) — caption files for stage 2
- [`unilora/`](unilora/README.md) — LoRA modules for stage 2
