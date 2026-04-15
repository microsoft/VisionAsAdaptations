# stage1_vm (stage 1 training code)

## Environment

Prefer installing from **`/data1/Compression_clean/video/requirements.txt`**. For pins aligned with Visual-Memory:

```bash
pip install -r /data1/Compression_clean/video/train/stage1_vm/requirements.txt
```

## How to run

Do not `cd` here alone unless you know the CLI. The supported entrypoint is the launcher in the parent folder:

```bash
cd /data1/Compression_clean/video/train
bash stage1_train.sh
```

That runs `torchrun ... stage1_vm/train_lora_single.py` with the flags set in `stage1_train.sh`.

### Subfolders

- [`descriptions/`](descriptions/README.md) — per-sequence caption files (`descriptions_<NAME>.txt`)
- [`unilora/`](unilora/README.md) — LoRA implementation used by the trainer
