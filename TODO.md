# TODO

Roadmap for building a modular Sparse Autoencoder (SAE) stack on Gemma-2B, with a
TOFU-finetuned variant as the foundation for unlearning experiments.

## Milestones

### M1 — SAE on Gemma-2B (base model)
Goal: a modular SAE trained on a single residual-stream layer of Gemma-2B,
plus a demo of concept suppression via latent clamping.

- [ ] **Activation harvesting**
  - [ ] Load Gemma-2B (HF / transformer_lens) and pick one layer's residual stream
        (start mid-network, e.g. layer ~13).
  - [ ] Stream a text corpus (e.g. OpenWebText / The Pile slice / FineWeb-Edu sample)
        through the model and cache `(N, d_model)` activations.
  - [ ] Shuffled, sharded activation buffer for training (disk + RAM buffer).

- [ ] **SAE core (modular)**
  - [ ] Base `SAE` module: encoder `W_enc`, decoder `W_dec` (unit-norm cols), bias `b_dec` pre-subtracted.
  - [ ] Config-driven sparsity backend: `mode ∈ {l1, topk, jumprelu, gated}`.
  - [ ] **Implement Top-k first** (`k = 8·d` or `16·d`, where `d = d_model = 2304`).
  - [ ] Stubs / interfaces for `l1`, `jumprelu`, `gated` so they can be added without refactor.
  - [ ] Decoder weight tying option, decoder-norm constraint, dead-latent re-init.

- [ ] **Training pipeline**
  - [ ] Loss = reconstruction MSE (+ sparsity term per mode).
  - [ ] Auxiliary `aux_k` loss for Top-k (revives dead latents).
  - [ ] Optimizer: Adam, learning-rate warmup, decoder column re-norm step.
  - [ ] Logging: recon loss, explained variance, L0, fraction dead latents,
        feature density histogram (wandb or simple TB/CSV).
  - [ ] Checkpointing + resumable training.

- [ ] **Evaluation / validation**
  - [ ] Held-out reconstruction MSE and explained variance.
  - [ ] Cross-entropy delta when SAE-reconstructed activations are spliced back in.
  - [ ] L0, dead-feature count, feature-density histogram sanity checks.
  - [ ] Quick interpretability pass: top-activating tokens for a handful of features.

- [ ] **Concept suppression demo**
  - [ ] Pick a target concept (e.g. a named entity, a topic, or a style).
  - [ ] Identify SAE features firing on that concept (top-activation search).
  - [ ] Implement intervention hook: clamp chosen feature(s) to 0 (or negative)
        during the forward pass.
  - [ ] Notebook showing pre/post generations and a small quantitative check
        (e.g. probability mass on concept tokens).

### M2 — Reproducible SAE training pipeline
Goal: turn M1 into a config-driven pipeline we can re-run on any checkpoint of Gemma-2B.

- [ ] Single entry point `train_sae.py --config configs/gemma2b_layerX_topk.yaml`.
- [ ] Config fields: model name/path, layer index, hook point, `d`, `k`, sparsity mode,
      dataset, tokens-to-train, optimizer, output dir.
- [ ] Deterministic seeding; pinned deps (`requirements.txt` / `pyproject.toml`).
- [ ] Smoke test: short run that verifies loss decreases and L0 ≈ k.
- [ ] Validation report auto-generated per run (metrics + plots).
- [ ] Documented procedure: "given a Gemma-2B checkpoint, fit an SAE in N steps."

### M3 — TOFU finetuning of Gemma-2B
Goal: a Gemma-2B checkpoint finetuned on TOFU, used as the unlearning starting point.

- [ ] Pull TOFU dataset (`locuslab/TOFU`) and inspect splits (`full`, `forget*`, `retain*`).
- [ ] Finetuning script (LoRA or full FT — start with LoRA for speed, then optional full FT).
- [ ] Train on `full` split; eval on TOFU evaluation harness
      (forget quality / model utility metrics).
- [ ] Save checkpoint as `gemma2b-tofu-ft`.
- [ ] Sanity-check: model answers TOFU author questions correctly post-FT.

### M4 — SAEs on the TOFU-finetuned model (bridge to unlearning)
Goal: re-fit SAEs on `gemma2b-tofu-ft` using the M2 pipeline; this is the substrate
for future unlearning-via-SAE-intervention work.

- [ ] Re-run activation harvesting on the FT checkpoint, same layer as M1.
- [ ] Train SAE with identical config; compare features to base-model SAE.
- [ ] Locate features associated with TOFU "forget set" entities/facts.
- [ ] Pilot suppression experiment: clamp those features and measure forget-quality
      vs. retain-utility trade-off.

## Initial scale / hyperparameters

- Model: `google/gemma-2-2b` (base), residual-stream hook on a single mid layer.
- `d = d_model` (Gemma-2B residual width, ~2304).
- SAE width: start with `k_dict = 8·d` to `16·d` features.
- Top-k active features per token: small (e.g. 32–128) — separate from dictionary size;
  tune with L0 vs. recon trade-off.
- Tokens of activations: start ~100M–500M tokens for the first real run; less for smoke tests.

## Repo layout (proposed)

```
sae/
  __init__.py
  base.py            # SAE module, shared encode/decode
  sparsity/
    topk.py
    l1.py            # stub
    jumprelu.py      # stub
    gated.py         # stub
  train.py           # training loop
  data.py            # activation buffer / streaming
  hooks.py           # model hook utilities
  eval.py            # recon, CE-delta, density, etc.
  intervene.py       # clamping / steering utilities
configs/
  gemma2b_layerX_topk.yaml
scripts/
  harvest_activations.py
  train_sae.py
  eval_sae.py
  suppress_concept_demo.py
finetune/
  tofu_finetune.py
  configs/
```

## Open questions / decisions to revisit

- Which layer to target first? (Mid-network is the usual default.)
- LoRA vs. full finetune for TOFU?
- Dataset for activation harvesting on the FT model — include TOFU text or keep generic?
- When to bring up `JumpReLU` / `GatedSAE` — likely after Top-k baseline is solid.

## Configuration for haversting, training, evaluating, and ploting

python scripts/harvest_splits.py --model google/gemma-2-2b --layer -1 --root activations/gemma2b_mid_pile --train-size 50000 --val-size 2000 --test-size 2000 --seq-len 1024 --tokens-per-shard 500000 --dtype bfloat16 --dataset monology/pile-uncopyrighted --dataset-config default

python scripts/train_sae.py --shard-dir activations/gemma2b_mid_pile --output-dir runs/gemma2b_mid_pile_topk_8d/run_1 --sparsity-mode topk --k 64 --n-features 18432 --batch-size 4096 --buffer-shards 4 --lr 3e-4 --warmup-steps 500 --max-steps 12000 --log-every 50 --ckpt-every 4000 --compute-dtype float32 --device cuda

python scripts/eval_sae.py --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt --shard-dir activations/gemma2b_mid_pile/train --output runs/gemma2b_mid_pile_topk_8d/run_1/eval_train --model google/gemma-2-2b --compute-dtype bfloat16 --batch-size 2048 --max-batches 100 --ce-max-docs 64 --ce-dataset monology/pile-uncopyrighted --ce-dataset-config default --ce-skip-docs 0

python scripts/eval_sae.py --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt --shard-dir activations/gemma2b_mid_pile/val --output runs/gemma2b_mid_pile_topk_8d/run_1/eval_val --model google/gemma-2-2b --compute-dtype bfloat16 --batch-size 2048 --max-batches 100 --ce-max-docs 64 --ce-dataset monology/pile-uncopyrighted --ce-dataset-config default --ce-skip-docs 50000

python scripts/eval_sae.py --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt --shard-dir activations/gemma2b_mid_pile/test --output runs/gemma2b_mid_pile_topk_8d/run_1/eval_test --model google/gemma-2-2b --compute-dtype bfloat16 --batch-size 2048 --max-batches 100 --ce-max-docs 64 --ce-dataset monology/pile-uncopyrighted --ce-dataset-config default --ce-skip-docs 52000

python scripts/plot_metrics.py --train run1=runs/gemma2b_mid_pile_topk_8d/run_1 --eval  train=runs/gemma2b_mid_pile_topk_8d/run_1/eval_train val=runs/gemma2b_mid_pile_topk_8d/run_1/eval_val test=runs/gemma2b_mid_pile_topk_8d/run_1/eval_test --output runs/gemma2b_mid_pile_topk_8d/run_1/plots
