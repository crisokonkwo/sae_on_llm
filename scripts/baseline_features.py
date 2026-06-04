"""U2 - lock in the pre-FT baseline snapshot for the unlearning experiment.

Reads a built concept_data directory (forget / retain / probes) + a frozen
SAE on the base model and writes a ``baseline_report.json`` that records:

  * feature ranking (full top-N from find_concept_features) and the
    sub-list of features picked for the experiment,
  * per-feature stats and top-activating tokens drawn from the concept-
    positive corpus,
  * baseline + intervened (clamp-ablated) target-completion log-prob on
    every probe and paraphrase,
  * baseline + intervened retain-set CE (collateral-damage budget),
  * a handful of clean vs intervened greedy generations per probe.

Example:
    python scripts/baseline_features.py \\
        --ckpt runs/gemma2b_mid_pile_topk_8d/run_1/ckpt_final.pt \\
        --model google/gemma-2-2b \\
        --shard-dir activations/gemma2b_mid_pile/train \\
        --concept-data runs/unlearn_united_nations/concept_data \\
        --output runs/unlearn_united_nations/baseline_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sae import SAE, SAEConfig
from sae.dataset import load_meta
from unlearn.baseline import (
    ProbeReportRow,
    aggregate_probe_rows,
    build_probe_rows,
    feature_top_examples,
    pick_concept_features,
    probe_generations,
    probe_logprobs_clean,
    probe_logprobs_intervened,
    retain_set_ce,
)
from unlearn.concept_data import (
    ConceptDataSpec,
    ConceptProbe,
    read_corpus,
    read_probes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # SAE + LM
    p.add_argument("--ckpt", required=True, help="SAE checkpoint produced by scripts/train_sae.py")
    p.add_argument("--model", required=True, help="HF model name (must match the SAE's base model)")
    p.add_argument("--shard-dir", default=None,
                   help="Used only to read resolved_layer_idx from meta.json.")
    p.add_argument("--layer", type=int, default=None,
                   help="Override the layer index (defaults to shard meta).")
    # Concept data
    p.add_argument("--concept-data", required=True,
                   help="Directory produced by scripts/build_concept_corpus.py")
    p.add_argument("--max-positives", type=int, default=200,
                   help="Forget sentences used to score features and collect feature examples.")
    p.add_argument("--max-negatives", type=int, default=200,
                   help="Retain sentences used to score features.")
    # Feature picking
    p.add_argument("--top-n-features", type=int, default=10,
                   help="Top-N features to return from find_concept_features (full ranking).")
    p.add_argument("--q-features-to-track", type=int, default=5,
                   help="Number of features to actually clamp during U4. Default 5.")
    p.add_argument("--feature-ids", type=int, nargs="*", default=None,
                   help="Override the auto-picked feature IDs (must be q-many).")
    p.add_argument("--top-k-tokens", type=int, default=8,
                   help="Top-K activating tokens per picked feature.")
    p.add_argument("--feature-example-max-docs", type=int, default=256,
                   help="Docs of concept-positive text used for feature example collection.")
    # Intervention
    p.add_argument("--clamp-value", type=float, default=0.0,
                   help="Value to clamp the picked features to (0 = ablation).")
    # Probes
    p.add_argument("--probe-max-length", type=int, default=512)
    p.add_argument("--generation-samples", type=int, default=5,
                   help="Number of probes to capture greedy generations for (clean + intervened).")
    p.add_argument("--max-new-tokens", type=int, default=30)
    # Retain CE
    p.add_argument("--retain-eval-sentences", type=int, default=200)
    p.add_argument("--retain-eval-max-length", type=int, default=256)
    # Device / dtype
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    # Output
    p.add_argument("--output", required=True,
                   help="Output JSON path (will be overwritten).")
    return p.parse_args()


def _load_sae(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> SAE:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = SAE(SAEConfig(**payload["sae_cfg"]))
    sae.load_state_dict(payload["sae_state"])
    sae.to(device, dtype=dtype)
    sae.eval()
    return sae


def _resolve_layer(args: argparse.Namespace) -> int:
    if args.layer is not None:
        return args.layer
    if args.shard_dir is None:
        raise SystemExit("Pass either --layer or --shard-dir (to read meta).")
    layer = load_meta(args.shard_dir).get("resolved_layer_idx")
    if layer is None:
        raise SystemExit("Could not infer layer from meta.json; pass --layer.")
    return int(layer)


def _load_concept_data(d: Path) -> tuple[ConceptDataSpec, list, list, list[ConceptProbe]]:
    for name in ("spec.json", "forget.jsonl", "retain.jsonl", "probes.jsonl"):
        if not (d / name).exists():
            raise SystemExit(f"missing {name} under {d}")
    probes = read_probes(d / "probes.jsonl")
    raw = json.loads((d / "spec.json").read_text())
    raw.pop("probes", None)
    spec = ConceptDataSpec(probes=probes, **raw)
    forget = read_corpus(d / "forget.jsonl")
    retain = read_corpus(d / "retain.jsonl")
    return spec, forget, retain, probes


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.compute_dtype]
    layer_idx = _resolve_layer(args)

    print(f"[baseline] loading SAE from {args.ckpt}")
    sae = _load_sae(Path(args.ckpt), device, dtype)
    print(f"[baseline] SAE: d_model={sae.cfg.d_model} n_features={sae.cfg.n_features} "
          f"mode={sae.cfg.sparsity_mode}")

    print(f"[baseline] loading concept data from {args.concept_data}")
    spec, forget, retain, probes = _load_concept_data(Path(args.concept_data))
    print(f"[baseline] concept={spec.name!r}  forget={len(forget)}  retain={len(retain)}  "
          f"probes={len(probes)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[baseline] loading LM {args.model} (layer={layer_idx})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``device_map=...`` requires ``accelerate``; skip it on CPU to keep the
    # smoke test runnable without extra deps.
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device.type != "cpu":
        load_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()

    # 1. Feature ranking
    pos_texts = [s.text for s in forget][: args.max_positives]
    neg_texts = [s.text for s in retain][: args.max_negatives]
    print(f"[baseline] running find_concept_features (pos={len(pos_texts)}, neg={len(neg_texts)}, "
          f"top_n={args.top_n_features})")
    ranking = pick_concept_features(
        sae, model, tokenizer,
        layer_idx=layer_idx,
        positive_texts=pos_texts,
        negative_texts=neg_texts,
        top_n=args.top_n_features,
        device=device,
    )
    for i, r in enumerate(ranking, 1):
        print(f"  {i:>2}. id={r['feature_id']:<6}  score={r['score']:+.4f}  "
              f"pos={r['mean_act_pos']:.3f}  "
              f"neg={(r.get('mean_act_neg') or 0):.3f}  "
              f"fire_pos={r['fire_rate_pos']:.3f}")

    # 2. Pick q features
    if args.feature_ids:
        if len(args.feature_ids) != args.q_features_to_track:
            print(f"[baseline] WARN: --feature-ids has {len(args.feature_ids)} ids but "
                  f"--q-features-to-track is {args.q_features_to_track}; using provided IDs.")
        picked_ids = list(args.feature_ids)
    else:
        picked_ids = [r["feature_id"] for r in ranking[: args.q_features_to_track]]
    print(f"[baseline] features picked for clamp: {picked_ids}")

    # 3. Top-activating tokens per picked feature (drawn from concept-positive corpus)
    print(f"[baseline] collecting top-activating tokens for picked features")
    examples = feature_top_examples(
        sae, model, tokenizer,
        layer_idx=layer_idx, feature_ids=picked_ids,
        texts=pos_texts,
        top_k_tokens=args.top_k_tokens,
        max_docs=min(args.feature_example_max_docs, len(pos_texts)),
        device=device,
    )

    # Combine ranking entry + examples for the report
    ranking_by_id = {r["feature_id"]: r for r in ranking}
    picked_records: list[dict] = []
    for fid in picked_ids:
        rec = dict(ranking_by_id.get(fid, {"feature_id": fid}))
        rec["top_activating_tokens"] = examples.get(int(fid), [])
        picked_records.append(rec)

    # 4. Probe log-probs: clean and intervened
    print(f"[baseline] computing probe target-completion log-probs (clean)")
    clean_lp = probe_logprobs_clean(model, tokenizer, probes,
                                    device=device, max_length=args.probe_max_length)
    print(f"[baseline] computing probe target-completion log-probs (intervened, clamp={args.clamp_value})")
    int_lp = probe_logprobs_intervened(
        model, tokenizer, probes,
        sae=sae, layer_idx=layer_idx,
        feature_ids=picked_ids, clamp_value=args.clamp_value,
        device=device, max_length=args.probe_max_length,
    )
    probe_rows = build_probe_rows(probes, clean_lp, int_lp)
    probe_agg = aggregate_probe_rows(probe_rows)
    print(f"[baseline]   mean log p/clean: {probe_agg.get('mean_logprob_clean_per_token', 0):.4f}  "
          f"mean log p/intervened: {probe_agg.get('mean_logprob_intervened_per_token', 0):.4f}  "
          f"delta: {probe_agg.get('mean_delta_per_token', 0):.4f}")

    # 5. Generation samples
    print(f"[baseline] greedy generation on {args.generation_samples} probes (clean + intervened)")
    gens = probe_generations(
        model, tokenizer, probes,
        max_new_tokens=args.max_new_tokens, device=device,
        n_probes_to_sample=args.generation_samples,
        sae=sae, layer_idx=layer_idx, feature_ids=picked_ids,
        clamp_value=args.clamp_value,
    )

    # 6. Retain-set CE (clean + intervened)
    print(f"[baseline] retain-set CE on {min(args.retain_eval_sentences, len(retain))} sentences")
    retain_ce = retain_set_ce(
        model, tokenizer, retain,
        device=device,
        max_sentences=args.retain_eval_sentences,
        max_length=args.retain_eval_max_length,
        sae=sae, layer_idx=layer_idx,
        feature_ids=picked_ids, clamp_value=args.clamp_value,
    )
    print(f"[baseline]   retain CE clean={retain_ce['clean']:.4f}  "
          f"intervened={retain_ce.get('intervened', 0):.4f}  "
          f"delta={retain_ce.get('delta', 0):+.4f}")

    # 7. Write report
    report: dict[str, Any] = {
        "meta": {
            "ckpt": str(args.ckpt),
            "model": args.model,
            "shard_dir": args.shard_dir,
            "layer_idx": layer_idx,
            "concept_data_dir": str(args.concept_data),
            "sae_cfg": asdict(sae.cfg),
            "clamp_value": args.clamp_value,
            "max_positives": args.max_positives,
            "max_negatives": args.max_negatives,
            "top_n_features": args.top_n_features,
            "q_features_to_track": args.q_features_to_track,
            "compute_dtype": args.compute_dtype,
        },
        "concept": {
            "name": spec.name,
            "description": spec.description,
            "n_forget_sentences": len(forget),
            "n_retain_sentences": len(retain),
            "n_probes": len(probes),
            "n_probes_with_expected_completion": sum(1 for p in probes if p.expected_completion),
        },
        "feature_ranking": ranking,
        "features_picked": picked_records,
        "probes": {
            "per_probe": [asdict(r) for r in probe_rows],
            "aggregate": probe_agg,
        },
        "generations": gens,
        "retain_ce": retain_ce,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[baseline] wrote {out_path}")
    print(f"\n=== Headline numbers (pre-FT baseline) ===")
    print(f"  features picked       : {picked_ids}")
    print(f"  probe Δlog p / token  : {probe_agg.get('mean_delta_per_token', 0):+.4f}  "
          "(positive = clamping suppresses the target completion)")
    print(f"  retain CE Δ           : {retain_ce.get('delta', 0):+.4f}  nats/token")


if __name__ == "__main__":
    main()
