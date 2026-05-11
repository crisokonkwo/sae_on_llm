"""Concept-suppression demo: discover SAE features that fire on a concept,
then clamp them to 0 during generation and compare to baseline.

Example:
    python scripts/suppress_concept.py \
        --ckpt runs/gemma2b_mid_topk_8d/run_1/ckpt_final.pt \
        --model google/gemma-2-2b \
        --shard-dir activations/gemma2b_mid/train \
        --concept-prompts "The Golden Gate Bridge spans the Golden Gate strait." \
                          "Crossing the Golden Gate Bridge into San Francisco." \
        --negative-prompts "I went to the grocery store yesterday." \
                           "The weather forecast predicts rain tomorrow." \
        --gen-prompts "The Golden Gate Bridge is" \
                      "San Francisco is famous for its" \
                      "I had pasta for dinner and" \
        --top-n 6 \
        --max-new-tokens 40 \
        --output runs/gemma2b_mid_topk_8d/run_1/suppress_golden_gate
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sae import SAE, SAEConfig
from sae.dataset import load_meta
from sae.hooks import patch_residual_stream
from sae.eval import top_activating_tokens
from sae.intervene import (
    ConceptScore,
    find_concept_features,
    make_clamp_fn,
    sequence_logprob,
)


def _read_lines(arg_values: list[str] | None) -> list[str]:
    """Each entry can be a literal string or a path to a newline-delimited file."""
    if not arg_values:
        return []
    out: list[str] = []
    for v in arg_values:
        p = Path(v)
        if p.is_file():
            out.extend(line.strip() for line in p.read_text().splitlines() if line.strip())
        else:
            out.append(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # SAE / model
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="Defaults to the layer recorded in the harvest meta or in the SAE config.")
    p.add_argument("--shard-dir", default=None,
                   help="Optional: pull resolved_layer_idx from this dir's meta.json.")
    # concept search
    p.add_argument("--concept-prompts", nargs="+", required=True,
                   help="Texts containing the concept (literal strings or file paths).") # Required: need some signal to discover features.
    p.add_argument("--negative-prompts", nargs="*", default=None,
                   help="Texts WITHOUT the concept, for differential ranking.") # Optional: if not given, just rank by mean activation on the concept prompts.
    p.add_argument("--top-n", type=int, default=6,
                   help="How many features to clamp.")
    p.add_argument("--score", default="mean_act_diff",
                   choices=["mean_act", "mean_act_diff", "fire_rate_diff"])
    p.add_argument("--feature-ids", type=int, nargs="*", default=None,
                   help="Skip discovery and clamp these features explicitly.")
    p.add_argument("--clamp-value", type=float, default=0.0,
                   help="Value to force the chosen features to (0 = ablate).")
    p.add_argument("--top-token-examples", type=int, default=5,
                   help="For each clamped feature, store this many top-activating token examples in the report.")
    p.add_argument("--top-token-context", type=int, default=8,
                   help="Left/right context window for top-activating token examples.")
    # generation
    p.add_argument("--gen-prompts", nargs="+", required=True,
                   help="Prompts to generate from for the qualitative comparison.") # Required: need some prompts to see the effect of suppression.
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy. Otherwise sampling temperature.")
    p.add_argument("--seed", type=int, default=0)
    # device
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def _load_sae(path: Path, device, dtype):
    payload = torch.load(path, map_location="cpu")
    sae_cfg = SAEConfig(**payload["sae_cfg"])
    sae = SAE(sae_cfg)
    sae.load_state_dict(payload["sae_state"])
    sae.to(device, dtype=dtype)
    sae.eval()
    return sae, payload


def _generate(model, tokenizer, prompt: str, max_new_tokens: int,
              temperature: float, device) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.compute_dtype]

    torch.manual_seed(args.seed)

    print(f"[suppress] loading SAE from {args.ckpt}")
    sae, payload = _load_sae(Path(args.ckpt), device, dtype)
    print(f"[suppress] SAE: d_model={sae.cfg.d_model} n_features={sae.cfg.n_features} mode={sae.cfg.sparsity_mode}")

    # Layer index resolution
    layer_idx = args.layer
    if layer_idx is None and args.shard_dir is not None:
        meta = load_meta(args.shard_dir)
        layer_idx = meta.get("resolved_layer_idx")
    if layer_idx is None:
        raise ValueError("--layer not given and no --shard-dir to read it from")

    # Load LM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[suppress] loading LM {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    model.eval()

    # Read concept/negative/generation prompts (literal or from files) and log counts.
    concept_texts = _read_lines(args.concept_prompts)
    negative_texts = _read_lines(args.negative_prompts) if args.negative_prompts else None
    gen_prompts = _read_lines(args.gen_prompts)
    print(f"[suppress] {len(concept_texts)} concept texts, "
          f"{len(negative_texts) if negative_texts else 0} negative texts, "
          f"{len(gen_prompts)} gen prompts")

    # 1. Discover or take features
    if args.feature_ids:
        features: list[ConceptScore] = [
            ConceptScore(feature_id=int(f), score=float("nan"),
                         mean_act_pos=float("nan"), mean_act_neg=None,
                         fire_rate_pos=float("nan"), fire_rate_neg=None)
            for f in args.feature_ids
        ]
        print(f"[suppress] using user-provided features: {[f.feature_id for f in features]}")
    else:
        print(f"[suppress] discovering top-{args.top_n} features (score={args.score}, layer={layer_idx})")
        features = find_concept_features(
            sae, model, tokenizer, layer_idx,
            positive_texts=concept_texts, negative_texts=negative_texts,
            top_n=args.top_n, score=args.score, device=device,
        )
        for f in features:
            print(f"        feature {f.feature_id}: score={f.score:.4f} "
                  f"mean_pos={f.mean_act_pos:.3f}"
                  + (f" mean_neg={f.mean_act_neg:.3f}" if f.mean_act_neg is not None else "")
                  + f" fire_pos={f.fire_rate_pos:.3f}")

    feature_ids = [f.feature_id for f in features]

    # 2. Store the top-activating tokens for the chosen features on the concept corpus.
    # This makes the JSON report self-contained: for each suppressed feature,
    # you can inspect the tokens/contexts that caused it to fire.
    print(f"[suppress] collecting top-activating token examples for features {feature_ids}")
    feature_token_examples = top_activating_tokens(
        sae, model, tokenizer, layer_idx,
        texts=concept_texts,
        feature_ids=feature_ids,
        top_k=args.top_token_examples,
        seq_len=256,
        max_docs=len(concept_texts),
        context=args.top_token_context,
        device=device,
    )
    # JSON wants string keys.
    feature_token_examples_json = {str(k): v for k, v in feature_token_examples.items()}

    # 3. Build clamp hook
    clamp_fn = make_clamp_fn(sae, feature_ids, clamp_values=args.clamp_value)

    # 4. Generate clean vs. suppressed for each prompt
    print("\n[suppress] generating ...")
    generations = []
    for prompt in gen_prompts:
        torch.manual_seed(args.seed)
        clean = _generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, device)
        torch.manual_seed(args.seed)
        with patch_residual_stream(model, layer_idx, clamp_fn):
            suppressed = _generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, device)

        # Quantitative: log-prob delta on the *clean* continuation under the
        # suppressed model. Big drop = suppression effective.
        logp_clean, n_tok = sequence_logprob(model, tokenizer, clean, device=device)
        with patch_residual_stream(model, layer_idx, clamp_fn):
            logp_suppressed, _ = sequence_logprob(model, tokenizer, clean, device=device)
        delta_per_tok = (logp_clean - logp_suppressed) / max(n_tok, 1)

        generations.append({
            "prompt": prompt,
            "clean": clean,
            "suppressed": suppressed,
            "n_tokens_scored": n_tok,
            "logprob_clean":      logp_clean,
            "logprob_suppressed": logp_suppressed,
            "logprob_delta_per_token": delta_per_tok,
        })
        print(f"\n=== PROMPT: {prompt!r}")
        print(f"--- clean      : {clean}")
        print(f"--- suppressed : {suppressed}")
        print(f"--- Δlog p(clean continuation) per token: {delta_per_tok:+.4f} nats "
              f"(over {n_tok} tokens)")

    # 5. Write report
    report = {
        "ckpt": str(args.ckpt),
        "model": args.model,
        "layer_idx": layer_idx,
        "sae_cfg": asdict(sae.cfg),
        "concept_prompts": concept_texts,
        "negative_prompts": negative_texts,
        "score_type": args.score,
        "clamp_value": args.clamp_value,
        "features": [asdict(f) for f in features],
        "feature_token_examples": feature_token_examples_json,
        "generations": generations,
        "summary": {
            "mean_logprob_delta_per_token": (
                sum(g["logprob_delta_per_token"] for g in generations) / max(len(generations), 1)
            ),
        },
    }
    out_path = out_dir / "suppression_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[suppress] wrote {out_path}")

    # Compact markdown for skimming
    md = ["# Concept suppression report\n",
          f"- ckpt: `{args.ckpt}`",
          f"- model: `{args.model}`  layer: `{layer_idx}`",
          f"- features clamped to {args.clamp_value}: `{feature_ids}`",
          f"- mean Δlog p per token: **{report['summary']['mean_logprob_delta_per_token']:+.4f} nats**",
          "",
          "## Feature token examples",
          ""]
    for f in features:
        md.append(f"### Feature {f.feature_id}")
        for hit in feature_token_examples_json.get(str(f.feature_id), [])[: args.top_token_examples]:
            md.append(
                f"- act={hit['activation']:.3f}, token=`{hit['token']}`, "
                f"context={hit['context']!r}"
            )
        md.append("")

    for g in generations:
        md += [
            f"## Prompt: `{g['prompt']!r}`",
            f"**clean**: {g['clean']}",
            "",
            f"**suppressed**: {g['suppressed']}",
            "",
            f"_Δlog p per token (over clean continuation): {g['logprob_delta_per_token']:+.4f} nats_",
            "",
        ]
    (out_dir / "suppression_report.md").write_text("\n".join(md))
    print(f"[suppress] wrote {out_dir / 'suppression_report.md'}")


if __name__ == "__main__":
    main()
