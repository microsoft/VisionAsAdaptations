# Video training (two-stage UniLoRA)

## Environment

Use the **`video`** tree requirements (parent of this folder):

```bash
cd /data1/Compression_clean/video
pip install -r requirements.txt
```

For stricter pins matching upstream codebases:

- Stage 1: `stage1_vm/requirements.txt` (Visual-Memory–aligned baseline)
- Stage 2: `stage2_vs/requirements.txt` (`diffusers==0.35.1`, `numpy<2`, etc.)

You need **`torchrun`** on your `PATH` and one or more **GPUs**.

## How to run

All scripts live under `/data1/Compression_clean/video/train`.

**Stage 1** (Visual-Memory–style):

```bash
cd /data1/Compression_clean/video/train
# Set FRAMES_DIR, CAPTION_FILE, OUT_DIR, CACHE_DIR, WAN_MODEL_ID, NUM_GPUS, ...
bash stage1_train.sh
```

**Stage 2** (VISUAL_SUBMIT–style + rate term; optional init from stage 1):

```bash
INIT_LORA_PATH=/path/to/stage1/lora_step_*.safetensors \
bash stage2_train.sh
```

**Both stages in sequence** (stage 2 initializes from the latest stage-1 checkpoint):

```bash
bash run_two_stage.sh
```

See comments in each `.sh` for environment variables (`NAME`, `RANK`, `TRAIN_STEPS`, etc.).

### Per-stage code directories

- [`stage1_vm/`](stage1_vm/README.md) — stage 1 Python package + `train_lora_single.py`
- [`stage2_vs/`](stage2_vs/README.md) — stage 2 Python package + `train_lora_single.py`
