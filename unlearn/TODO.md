# M5 — Mechanistic unlearning of a pretrained concept with SAE-feature stability

## Research question (advisor framing)

After we LoRA-unlearn a concept that the pretrained model already knows, does
the SAE feature we identified *before* unlearning remain a valid handle for
the same concept *after* unlearning?

Equivalently: is the SAE feature
- **still correlated** with the concept (same input → same activation)?
- **still causal** for the concept (clamping it still changes behavior)?
- or has it **drifted** in meaning?

## Why this experiment, and why not TOFU

- TOFU teaches a synthetic fact and then asks the model to forget it. That
  tests teach/un-teach symmetry, not the stability of a pretrained
  representation. The advisor's claim is that the underlying representation
  is too weak for the stability question to be meaningful.
- A pretrained concept (e.g. *Eiffel Tower → Paris*) has a real, well-formed
  internal representation in Gemma-2-2B. Unlearning it tests whether the
  representation we identified is durable under weight updates.
- The frozen SAE acts as a fixed mechanistic lens. If the lens still
  resolves the concept after weight surgery, we have evidence for
  mechanistic stability. If it doesn't, we know the apparent suppression is
  not localized to the feature we named.

## Scope

**In scope (first end-to-end result):**
- One target concept (default candidate: Eiffel Tower / Paris).
- LoRA-based unlearning on the base Gemma-2-2B (no full FT).
- One unlearning loss (gradient ascent + retain-set KL).
- Frozen SAE as the lens; no SAE retraining in this milestone.
- Six feature-stability metrics (correlation, top-token Jaccard, selectivity,
  causal clamp delta, activation-distribution KL, generation samples).

**Out of scope (deferred):**
- Multiple concepts.
- Refitting the SAE on the unlearned model (separate experiment; tests a
  different question — "do features re-emerge?" rather than "do *the same*
  features still mean the same thing?").
- Comparing clamp vs ablation vs negative steering exhaustively.
- NPO / RMU / SimPO loss comparisons.
- Adversarial relearning attacks.
- Multi-layer interventions.

## Reuse map (import, do not copy)

| Need | Existing module | New module imports from it |
|---|---|---|
| SAE module, sparsity | `sae.base`, `sae.sparsity.topk` | `unlearn.stability` |
| Activation hooks | `sae.hooks` | `unlearn.stability`, `unlearn.trainer` |
| Concept-feature discovery | `sae.intervene.find_concept_features` | `unlearn.stability` |
| Feature clamping | `sae.intervene.make_clamp_fn` | `unlearn.stability` |
| Top-activating tokens | `sae.eval.top_activating_tokens` | `unlearn.stability` |
| LoRA trainer (Adam + grad accum + PEFT save) | `finetune.finetune.TOFUTrainer` | `unlearn.trainer` (subclass / wrap, do **not** copy) |
| LoRA reload + merge | `finetune.load_ft.load_ft_model` | `unlearn.stability`, eval scripts |
| Plotting primitives | `sae.plotting` | reporting scripts |

If anything in `finetune.finetune.TOFUTrainer` is genuinely TOFU-specific
(it shouldn't be), refactor it up into a generic `LoRATrainer` rather than
duplicating logic in `unlearn/`.

## Milestones

### U1 — Concept dataset
- [x] Pick concept (decision required; see Open Questions).
- [x] Forget set: ~1k natural sentences where the concept appears
      (default source: Wikipedia article(s) about the concept).
      - implemented in `unlearn.concept_data.build_forget_corpus`
- [x] Retain set: ~1k natural sentences stylistically similar but
      concept-free (default: adjacent Wikipedia articles / random Wiki;
      `retain_source: hf_dataset` also supported for harvest-aligned text).
      - implemented in `unlearn.concept_data.build_retain_corpus`
- [x] Probe set: 20–30 base prompts × ≥3 paraphrases each, hand-written.
      - declared inline in the concept YAML (`configs/concept_eiffel_tower.yaml`).
- [x] Tokenisation + length statistics dump.
      - `scripts/build_concept_corpus.py` writes `stats.json` + `inspection.md`.
      - `scripts/inspect_concept_corpus.py` re-reads a built corpus for re-inspection.
- [x] `scripts/build_concept_corpus.py` reproducible builder.

### U2 — Pre-FT baseline (locks in the "before" snapshot)
- [ ] Load existing trained SAE on base Gemma-2-2B (frozen).
- [ ] Run `find_concept_features` on forget vs retain → pick `q` features
      to track (start with `q = 3–8`).
- [ ] Save: feature ids, score, mean activation, fire rate, top-activating
      tokens per feature.
- [ ] Baseline target-answer probability on probe set.
- [ ] Baseline clamp suppression delta on probe set
      (Δ log p(target) when chosen features are ablated).
- [ ] Baseline retain-set CE (so we can compute degradation later).
- [ ] Output: `runs/unlearn_<concept>/baseline_report.json`.

### U3 — Unlearning training
- [ ] `unlearn/unlearn_loss.py`: gradient ascent on forget + KL retain.
  - `L = -λ_f · CE(M; D_forget) + λ_r · KL(M_ref ‖ M; D_retain)`
  - Keep a frozen reference model in memory; LoRA only on the active copy.
- [ ] `unlearn/trainer.py`: thin wrapper around `TOFUTrainer` that swaps
      the loss and adds reference-model handling.
- [ ] Per-step logging: forget CE, retain CE, KL drift, grad norm,
      forget/retain ratio.
- [ ] `configs/unlearn_<concept>.yaml` reproducible config.
- [ ] Run target: 500–2000 LoRA steps; bf16; gradient checkpointing on.
- [ ] Output: `runs/unlearn_<concept>/lora_unlearn/ckpt_final/`.

### U4 — Post-FT feature stability evaluation
Frozen SAE re-encodes activations from the LoRA-merged model. For each
feature `j ∈ S_c`:
- [ ] **Pearson correlation** `ρ(z_j_pre, z_j_post)` on a fixed text set.
- [ ] **Top-activating token Jaccard** (pre vs post, on same corpus).
- [ ] **Concept selectivity drift**: `s_j(c)` before/after on forget vs retain.
- [ ] **Causal clamp delta**: target-answer probability suppression when
      `j` is clamped in the post-FT model. Compare to U2 baseline.
- [ ] **Activation-distribution KL** between binned `p^pre(z_j)` and `p^post(z_j)`.
- [ ] (Optional, deferred) Decoder-direction cosine if SAE is later refit.
- [ ] Output: `runs/unlearn_<concept>/stability_report.json`.

### U5 — Reporting
- [ ] Per-feature dashboard: pre/post stats side-by-side.
- [ ] Plots: pre/post scatter of `z_j` per feature; before/after suppression
      bar chart; top-activating-token diff view.
- [ ] Markdown summary suitable for advisor review.
- [ ] Notebook for interactive exploration.

## Success criteria

We are not chasing a single binary answer. We want each chosen feature to
fall cleanly into one of these buckets so the writeup is unambiguous:

| Outcome | Evidence pattern |
|---|---|
| **Stable handle** (best case for the mechanistic claim) | `ρ ≥ 0.9`, top-token Jaccard `≥ 0.5`, post-FT clamp still suppresses target prob comparably to pre-FT |
| **Correlated but no longer causal** | High `ρ`, but post-FT clamp delta ≈ 0 (concept moved elsewhere) |
| **Drifted** | `ρ < 0.5` or top-token Jaccard `< 0.2`; the feature now codes something else |
| **Decisively forgotten** | Target-answer prob → 0 on the unlearned model AND feature activation collapses to 0 |

Any of these is publishable. We just need the numbers to commit to one.

## Open questions (decide before code)

1. **Concept choice.** Candidates:
   - *Eiffel Tower → Paris* (clean factual triple, Wikipedia-rich, known to have Gemma Scope features).
   - *Golden Gate Bridge* (Anthropic canonical example; similar profile).
   - *Harry Potter / Rowling* (original LLMU paper used this; richer multi-prompt coverage).
   - *Python keywords* (more diffuse; harder to probe).
   - Recommendation: Eiffel Tower for cleanliness.
2. **Unlearning loss.** Start with gradient ascent + retain-KL? Or jump
   straight to NPO for stability? Recommendation: start with the simpler
   one, then add NPO as ablation in U6+.
3. **Feature set size `q`.** Pick top 3, top 8, or sweep? Recommendation:
   pick 5 by hand after looking at top-activating tokens.
4. **Frozen-SAE-only, or also refit?** The advisor framing is the frozen
   case. Recommendation: frozen for M5, defer refit comparison.
5. **Probe set size.** 30 base prompts × 3 paraphrases = 90 evals. Enough,
   or push to 100×3?
6. **LoRA rank.** r=16 (matches `finetune/`) or smaller (r=4–8) to reduce
   capacity for accidental damage?
7. **Stopping criterion for unlearning.** Fixed steps, or stop when
   target-answer prob falls below threshold while retain CE stays within
   budget? Recommendation: fixed steps + checkpoint sweep.

## Estimated cost (RTX 4070 Ti, 12 GB)

| Stage | Wall time |
|---|---|
| Concept corpus build | half day (mostly writing probes) |
| Baseline eval | ~30 min |
| LoRA unlearning training | 1–3 h per run |
| Stability eval | ~1 h |
| Reporting / plots | half day |
| **First end-to-end pass** | **2–3 focused days** |

## Planned repo layout (preview)

```
unlearn/
  feature_stability/
    __init__.py
    concept_data.py        # build forget / retain / probe sets, reproducibly
    unlearn_loss.py        # gradient-ascent + retain-KL; later: NPO / RMU
    trainer.py             # LoRA unlearning trainer (wraps TOFUTrainer)
    stability.py           # ρ, top-token Jaccard, KL, clamp causal delta, ...
    reporting.py           # per-feature dashboard + plots
  configs/
    unlearn_eiffel.yaml    # (or whichever concept we pick)
  scripts/
    build_concept_corpus.py
    baseline_features.py
    unlearn.py
    stability_eval.py
    smoke_unlearn_pipeline.py
  docs/
    unlearn_feature_stability.md
  runs/
    unlearn_<concept>/
      baseline_report.json
      lora_unlearn/ckpt_final/
      stability_report.json
      plots/
```

## Sharp risks to flag now

- **Gradient ascent without retain regularisation will brick the model in
  a few hundred steps.** The retain-KL term is mandatory, not optional.
- **PEFT model + residual hook ordering.** Merge LoRA into base weights
  before running `capture_residual_stream`; running through a wrapped
  `PeftModel` can attach hooks to a layer that bypasses the adapter.
- **Probe variance.** With small probe sets, single-paraphrase numbers
  are noisy. Always report mean ± std over paraphrases.
- **Confounded "drift".** If the SAE's reconstruction quality degrades on
  the FT'd model (because the residual distribution shifted), feature
  activations can change for reasons unrelated to the concept. Run the
  standard SAE eval (EV, NMSE, CE-delta) on the FT'd model first and gate
  the stability claims on the SAE still being faithful.
```
