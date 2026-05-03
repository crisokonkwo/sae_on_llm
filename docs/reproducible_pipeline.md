# Reproducible SAE training pipeline

This repo's reproducible entry point is:

```bash
python scripts/train_sae.py --config configs/gemma2b_mid_topk.yaml
```

The config records the model, target layer/hook point, activation-shard root,
SAE shape/sparsity mode, optimizer settings, validation settings, and output
path. CLI flags override config values for quick sweeps.

## 1. Harvest disjoint activation splits

Recommended on a 12GB RTX 4070 Ti:

```bash
python scripts/harvest_splits.py \
  --model google/gemma-2-2b \
  --layer -1 \
  --root activations/gemma2b_mid \
  --train-size 50000 \
  --val-size 2000 \
  --test-size 2000 \
  --seq-len 1024 \
  --tokens-per-shard 500000 \
  --dtype bfloat16 \
  --dataset monology/pile-uncopyrighted \
  --dataset-config default
```

This creates:

```text
activations/gemma2b_mid/
  train/shard_*.pt + meta.json
  val/shard_*.pt   + meta.json
  test/shard_*.pt  + meta.json
```

The split is document-level and deterministic: train uses the first N documents,
val skips train, test skips train+val. The slice is recorded in each `meta.json`.

## 2. Train from config

Edit `configs/gemma2b_mid_topk.yaml` if needed, then run:

```bash
python scripts/train_sae.py --config configs/gemma2b_mid_topk.yaml
```

Important config fields:

```yaml
output_dir: runs/gemma2b_mid_topk/run_1
model:
  name: google/gemma-2-2b
  layer_idx: -1
  hook_point: resid_post

data:
  shard_dir: activations/gemma2b_mid   # root is fine; train_sae auto-uses train/
  batch_size: 4096
  buffer_shards: 4

sae:
  expansion_factor: 8                  # n_features = 8 * d_model
  sparsity_mode: topk
  topk:
    k: 64

train:
  max_steps: 12000
  lr: 0.0003
  compute_dtype: float32

validation:
  validate_after: true
  max_batches: 200
  plot_after: true
```

Artifacts written under `output_dir`:

```text
ckpt_final.pt
ckpt_step*.pt
metrics.jsonl
config.json
run_manifest.json
validation_report.json     # reconstruction metrics on val/ if present
training_curves.png
```

## 3. Smoke test

Before long runs, verify the training pipeline itself:

```bash
python scripts/smoke_sae_pipeline.py
```

The smoke test creates synthetic activation shards, trains a tiny Top-K SAE via
`train_sae.py --config`, and checks:

- reconstruction loss decreases,
- L0 is approximately `k`,
- checkpointing works,
- validation report and training curve plot are written.

## 4. Evaluate with LM CE-delta

The training pipeline's auto-validation is reconstruction-only. To compute
CE-delta, use the full eval script:

```bash
python scripts/eval_sae.py \
  --ckpt runs/gemma2b_mid_topk/run_1/ckpt_final.pt \
  --shard-dir activations/gemma2b_mid/val \
  --output runs/gemma2b_mid_topk/run_1/eval_val \
  --model google/gemma-2-2b \
  --compute-dtype bfloat16 \
  --batch-size 2048 \
  --max-batches 100 \
  --ce-max-docs 64 \
  --ce-dataset monology/pile-uncopyrighted \
  --ce-dataset-config default \
  --ce-skip-docs 50000
```

Use `--ce-skip-docs N_train` for val and `N_train + N_val` for test so CE text
is held out from training.

## 5. Plot train/val/test reports

```bash
python scripts/plot_metrics.py \
  --train run1=runs/gemma2b_mid_topk/run_1 \
  --eval train=runs/gemma2b_mid_topk/run_1/eval_train \
         val=runs/gemma2b_mid_topk/run_1/eval_val \
         test=runs/gemma2b_mid_topk/run_1/eval_test \
  --output runs/gemma2b_mid_topk/run_1/plots
```
