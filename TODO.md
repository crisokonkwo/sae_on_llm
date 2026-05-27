# TODO

Roadmap for building a modular Sparse Autoencoder (SAE) stack on Gemma-2B, with a
TOFU-finetuned variant as the foundation for unlearning experiments.

## Milestones

### M1 — SAE on Gemma-2B (base model)

Goal: a modular SAE trained on a single residual-stream layer of Gemma-2B,
plus a demo of concept suppression via latent clamping.

- [x] **Activation harvesting**
  - [x] Load Gemma-2B (HF / transformer_lens) and pick one layer's residual stream
        (start mid-network, e.g. layer ~12).
  - [x] Stream a text corpus (e.g. OpenWebText / The Pile slice / FineWeb-Edu sample)
        through the model and cache `(N, d_model)` activations.
  - [x] Shuffled, sharded activation buffer for training (disk + RAM buffer).

- [x] **SAE core (modular)**
  - [x] Base `SAE` module: encoder `W_enc`, decoder `W_dec` (unit-norm cols), bias `b_dec` pre-subtracted.
  - [x] Config-driven sparsity backend: `mode ∈ {l1, topk, jumprelu, gated}`.
  - [x] **Implement Top-k first** (`k = 8·d` or `16·d`, where `d = d_model = 2304`).
  - [x] Stubs / interfaces for `l1`, `jumprelu`, `gated` so they can be added without refactor.
  - [x] Decoder weight tying option, decoder-norm constraint, dead-latent re-init.

- [ ] **Training pipeline**
  - [x] Loss = reconstruction MSE (+ sparsity term per mode).
  - [x] Auxiliary `aux_k` loss for Top-k (revives dead latents).
  - [x] Optimizer: Adam, learning-rate warmup, decoder column re-norm step.
  - [ ] Logging: recon loss, explained variance, L0, fraction dead latents,
        feature density histogram (wandb or simple TB/CSV).
  - [x] Checkpointing + resumable training.

- [x] **Evaluation / validation**
  - [x] Held-out reconstruction MSE and explained variance.
  - [x] Cross-entropy delta when SAE-reconstructed activations are spliced back in.
  - [x] L0, dead-feature count, feature-density histogram sanity checks.
  - [x] Quick interpretability pass: top-activating tokens for a handful of features.

- [ ] **Concept suppression demo**
  - [x] Pick a target concept (e.g. a named entity, a topic, or a style).
  - [x] Identify SAE features firing on that concept (top-activation search).
  - [x] Implement intervention hook: clamp chosen feature(s) to 0 (or negative)
        during the forward pass.
  - [ ] Implement conditional clamping. Paper: "Don’t Forget It! Conditional Sparse Autoencoder Clamping Works for
        Unlearning."
  - [ ] Implement Dynamic Sparse Autoencoder Guardrails Paper: "SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder
        Guardrails for Precision Unlearning in LLMs."
  - [x] Notebook showing pre/post generations and a small quantitative check
        (e.g. probability mass on concept tokens).

### M2 — Reproducible SAE training pipeline

Goal: turn M1 into a config-driven pipeline we can re-run on any checkpoint of Gemma-2B.

- [x] Single entry point `train_sae.py --config configs/gemma2b_layerX_topk.yaml`.
- [x] Config fields: model name/path, layer index, hook point, `d`, `k`, sparsity mode,
      dataset, tokens-to-train, optimizer, output dir.
- [x] Deterministic seeding; pinned deps (`requirements.txt` / `pyproject.toml`).
- [x] Smoke test: short run that verifies loss decreases and L0 ≈ k.
- [x] Validation report auto-generated per run (metrics + plots).
- [x] Documented procedure: "given a Gemma-2B checkpoint, fit an SAE in N steps."

### M3 — TOFU finetuning of Gemma-2B

Goal: a Gemma-2B checkpoint finetuned on TOFU, used as the unlearning starting point.

- [x] Pull TOFU dataset (`locuslab/TOFU`) and inspect splits (`full`, `forget*`, `retain*`).
- [x] Finetuning script (LoRA or full FT — start with LoRA for speed, then optional full FT).
- [x] Train on `full` split; eval on TOFU evaluation harness
      (forget quality / model utility metrics).
- [x] Save checkpoint as `gemma2b-tofu-ft`.
- [x] Sanity-check: model answers TOFU author questions correctly post-FT.

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

## Repo layout

```text
sae/                           # SAE library (M1 + M2)
  __init__.py                  # re-exports SAE, SAEConfig, build_sparsity
  base.py                      # SAE module: encode/decode, b_dec init, decoder renorm, loss
  sparsity/
    __init__.py                # build_sparsity(mode, n_features, **kwargs) factory + registry
    base.py                    # SparsityFn protocol (forward + optional extra_loss)
    topk.py                    # Top-K sparsity + aux-k dead-feature revival loss
    l1.py                      # placeholder (raises NotImplementedError)
    jumprelu.py                # placeholder (STE-thresholded variant; TODO)
    gated.py                   # placeholder (gate + magnitude paths; TODO)
  data.py                      # activation harvesting: stream docs through LM, shard to disk
  dataset.py                   # ActivationDataset: shard-buffered, shuffled IterableDataset
  hooks.py                     # residual-stream capture (catcher) + replace (patcher) hooks
  train.py                     # Trainer: AdamW + warmup + grad clip + decoder renorm + ckpt
  eval.py                      # recon metrics, CE-delta vs mean ablation, top-activating tokens
  intervene.py                 # make_clamp_fn (delta intervention) + find_concept_features
  plotting.py                  # training-curve and eval-comparison figures
finetune/                      # TOFU finetuning pipeline (M3)
  __init__.py
  tofu_data.py                 # load TOFU configs, format Q&A, tokenise with prompt masking
  finetune.py                  # FinetuneConfig + LoRAConfig + TOFUTrainer (PEFT LoRA)
  eval.py                      # answer log-prob, ROUGE-L, greedy generations per TOFU config
  load_ft.py                   # reload base + adapter; optional merge_and_unload for M4
configs/                       # YAML configs consumed by scripts/*.py --config
  gemma2b_mid_topk.yaml        # SAE on Gemma-2-2B mid-layer residual, Top-K k=64, 8x width
  tofu_lora_gemma2b.yaml       # TOFU LoRA FT on Gemma-2-2B, bf16, grad-ckpt, r=16
scripts/                       # CLI entry points (all support --config + CLI overrides)
  harvest_activations.py       # single-split activation harvest (ad-hoc / re-run)
  harvest_splits.py            # train/val/test in one go via disjoint stream slices
  train_sae.py                 # config-driven SAE training + auto-validation + plot
  eval_sae.py                  # held-out eval: recon + CE-delta + top tokens -> JSON report
  suppress_concept.py          # concept suppression demo: discover features + clamp + generate
  plot_metrics.py              # render train curves and train/val/test eval comparison
  inspect_tofu.py              # quick TOFU dataset inspection (no GPU, no model)
  finetune_tofu.py             # config-driven TOFU LoRA FT entry point
  eval_tofu.py                 # answer log-prob + ROUGE on retain/forget/utility configs
  smoke_sae_pipeline.py        # synthetic end-to-end SAE pipeline smoke test (CPU only)
  smoke_tofu_pipeline.py       # mocked TOFU FT pipeline smoke test (no peft, no CUDA)
docs/
  reproducible_pipeline.md     # M2: harvest -> train -> eval -> plot procedure
  tofu_finetuning.md           # M3: inspect -> smoke -> finetune -> eval -> M4 hand-off
notebooks/                     # exploratory notebooks (empty for now)
requirements.txt               # pinned deps: torch, transformers, datasets, peft, ...
TODO.md                        # roadmap (this file)
```

## Open questions / decisions to revisit

- Which layer to target first? (Mid-network is the usual default.)
- LoRA vs. full finetune for TOFU?
- Dataset for activation harvesting on the FT model — include TOFU text or keep generic?
- When to bring up `JumpReLU` / `GatedSAE` — likely after Top-k baseline is solid.

## Configuration for haversting, training, evaluating, and ploting

- python scripts/harvest_splits.py --model google/gemma-2-2b --layer -1 --root activations/gemma2b_mid_pile --train-size 50000 --val-size 2000 --test-size 2000 --seq-len 1024 --tokens-per-shard 500000 --dtype bfloat16 --dataset monology/pile-uncopyrighted --dataset-config default

- python scripts/train_sae.py --shard-dir activations/gemma2b_mid_pile --output-dir runs/gemma2b_mid_pile_topk_8d/run_1 --sparsity-mode topk --k 64 --n-features 18432 --batch-size 4096 --buffer-shards 4 --lr 3e-4 --warmup-steps 500 --max-steps 12000 --log-every 50 --ckpt-every 4000 --compute-dtype float32 --device cuda

- python scripts/eval_sae.py --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt --shard-dir activations/gemma2b_mid_pile/train --output runs/gemma2b_mid_pile_topk_8d/run_1/eval_train --model google/gemma-2-2b --compute-dtype bfloat16 --batch-size 2048 --max-batches 100 --ce-max-docs 64 --ce-dataset monology/pile-uncopyrighted --ce-dataset-config default --ce-skip-docs 0

- python scripts/eval_sae.py --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt --shard-dir activations/gemma2b_mid_pile/val --output runs/gemma2b_mid_pile_topk_8d/run_1/eval_val --model google/gemma-2-2b --compute-dtype bfloat16 --batch-size 2048 --max-batches 100 --ce-max-docs 64 --ce-dataset monology/pile-uncopyrighted --ce-dataset-config default --ce-skip-docs 50000

- python scripts/eval_sae.py --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt --shard-dir activations/gemma2b_mid_pile/test --output runs/gemma2b_mid_pile_topk_8d/run_1/eval_test --model google/gemma-2-2b --compute-dtype bfloat16 --batch-size 2048 --max-batches 100 --ce-max-docs 64 --ce-dataset monology/pile-uncopyrighted --ce-dataset-config default --ce-skip-docs 52000

- python scripts/plot_metrics.py --train run1=runs/gemma2b_mid_pile_topk_8d/run_1 --eval  train=runs/gemma2b_mid_pile_topk_8d/run_1/eval_train val=runs/gemma2b_mid_pile_topk_8d/run_1/eval_val test=runs/gemma2b_mid_pile_topk_8d/run_1/eval_test --output runs/gemma2b_mid_pile_topk_8d/run_1/plots
