# video pipeline

**Per-folder docs:** [`train/README.md`](train/README.md), [`scaling/README.md`](scaling/README.md), [`edit/README.md`](edit/README.md), [`eval/README.md`](eval/README.md).

This tree holds **self-contained** scripts for Wan2.1 UniLoRA–style workflows: **training** (two stages), **scaling** reconstruction, **editing**, and **evaluation**. Paths below are relative to `/data1/Compression_clean/video`.

---

## Python environment

Install from **`requirements.txt`** in this directory: it includes the same packages as **`/data1/Visual-Memory-0122/Visual-Memory/video/requirements.txt`**, plus extras used by scaling, edit, and eval (`dists-pytorch`, `scipy`, `torchvision`, `regex`).

**Quick install:**

```bash
cd /data1/Compression_clean/video
pip install -r requirements.txt
```

The baseline block (without extras) is:

```text
accelerate>=0.33.0
bitsandbytes>=0.43.1
opencv-python-headless
numpy
pillow
safetensors
peft>=0.11.1
transformers>=4.44.0,<5
huggingface-hub
sentencepiece
ftfy
diffusers>=0.30.0
torch>=2.2.0
pytorch-msssim
lpips
matplotlib
```

**Stage-specific pins (optional):** Training copies also ship local requirement files if you want to mirror known-good pins:

- `train/stage1_vm/requirements.txt` — same baseline as Visual-Memory.
- `train/stage2_vs/requirements.txt` — pins `diffusers==0.35.1` and `numpy<2` for compatibility with the stage-2 stack.

The appended lines in **`requirements.txt`** (`dists-pytorch`, `scipy`, `torchvision`, `regex`) cover scaling / edit metrics paths, eval FVD, and training transforms.

**Runtime notes**

- **Training** needs `torchrun` on your `PATH` (same environment as `torch`).
- **GPU** is expected for training and for practical eval / FVD throughput.

---

## Folder map

| Directory | Purpose |
|-----------|---------|
| `train/` | Two-stage UniLoRA training: stage 1 (Visual-Memory style), stage 2 (VISUAL_SUBMIT style + rate term). |
| `scaling/` | Load a checkpoint and reconstruct video with optional **scaling** behavior in the Wan encode pipeline. |
| `edit/` | **Caption-only** edit (one LoRA, new prompt) or **merge** two LoRAs + new prompt. |
| `eval/` | **PSNR**, **DISTS**, **LPIPS** (VGG + Alex), **FVD** (I3D features + Fréchet). See `eval/README.md`. |

---

## Training (`train/`)

- **`stage1_train.sh`** — launches `stage1_vm/train_lora_single.py` via `torchrun`.
- **`stage2_train.sh`** — launches `stage2_vs/train_lora_single.py`.
- **`run_two_stage.sh`** — runs stage 1, picks the latest checkpoint, then stage 2 with `INIT_LORA_PATH`.

Set **`FRAMES_DIR`**, **`CAPTION_FILE`**, **`OUT_DIR`**, **`CACHE_DIR`**, **`WAN_MODEL_ID`**, **`NUM_GPUS`**, etc. (see comments in each `.sh`). Defaults may point at paths on a specific cluster; override for your setup.

---

## Scaling reconstruction (`scaling/`)

- **`sample.sh`** — example launcher for `reconstruct_lora_single_scaling_encode.py`.
- Uses **`pipeline_wan_scaling_encode.py`** and local **`unilora_utils.py`** / **`entropy_models.py`**.

Override **`WEIGHTS`**, **`CACHE_DIR`**, frame directories, and **`SCALING`** (whether to pass `--scaling`) as needed.

---

## Editing (`edit/`)

- **`edit_caption.sh`** + **`edit_caption.py`** — one LoRA weights file + **`CAPTION`** string.
- **`edit_merge.sh`** + **`edit_merge.py`** — two weight files + **`CAPTION`** string.

Set **`WEIGHTS`** or **`WEIGHTS1`** / **`WEIGHTS2`** to real `.safetensors` paths (defaults are placeholders).

---

## Evaluation (`eval/`)

- **`eval.sh`** — full pipeline: image metrics → FVD clip prep → FVD.
- Details: **`eval/README.md`**.

Install **`requirements.txt`** in this directory first. First FVD run downloads the I3D TorchScript weights into `eval/fvd_cache/`.

---

## Provenance

Training and editing pipelines are derived from **Visual-Memory** and **VISUAL_SUBMIT** video codebases. **`requirements.txt`** here extends **Visual-Memory** `video/requirements.txt` with packages needed for scaling, edit, and eval.
