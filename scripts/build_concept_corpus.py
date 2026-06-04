"""Build forget / retain / probe artifacts for the unlearning experiment.

Reads a YAML spec, fetches Wikipedia text + a concept-negative retain corpus,
and writes:

    <output-dir>/
        spec.json            # resolved spec (provenance)
        forget.jsonl         # concept-positive sentences
        retain.jsonl         # concept-negative sentences
        probes.jsonl         # hand-written probes
        stats.json           # length / source / term-match statistics
        inspection.md        # human-skimmable summary

Example:
    python scripts/build_concept_corpus.py \\
        --config configs/concept_eiffel_tower.yaml \\
        --output-dir runs/unlearn_eiffel_tower/concept_data \\
        --tokenizer google/gemma-2-2b
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlearn.concept_data import (
    ConceptDataSpec,
    build_forget_corpus,
    build_retain_corpus,
    corpus_statistics,
    render_inspection_md,
    write_corpus,
    write_probes,
    write_spec,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML spec file.")
    p.add_argument("--output-dir", required=True)
    # Optional overrides (CLI > YAML)
    p.add_argument("--target-forget-sentences", type=int, default=None)
    p.add_argument("--target-retain-sentences", type=int, default=None)
    p.add_argument("--retain-source", default=None,
                   choices=["wikipedia_random", "hf_dataset"])
    p.add_argument("--skip-forget", action="store_true",
                   help="Re-use the previously built forget.jsonl.")
    p.add_argument("--skip-retain", action="store_true",
                   help="Re-use the previously built retain.jsonl.")
    # Stats
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer name for token-count statistics.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[concept_data] loading spec from {args.config}")
    spec = ConceptDataSpec.from_yaml(args.config)
    if args.target_forget_sentences is not None:
        spec.target_forget_sentences = args.target_forget_sentences
    if args.target_retain_sentences is not None:
        spec.target_retain_sentences = args.target_retain_sentences
    if args.retain_source is not None:
        spec.retain_source = args.retain_source
    write_spec(spec, out_dir / "spec.json")
    print(f"[concept_data] concept = {spec.name!r}  "
          f"(forget target={spec.target_forget_sentences}, "
          f"retain target={spec.target_retain_sentences}, retain_source={spec.retain_source})")

    forget_path = out_dir / "forget.jsonl"
    retain_path = out_dir / "retain.jsonl"

    # ----- Forget corpus -----
    if args.skip_forget and forget_path.exists():
        print(f"[concept_data] reusing existing {forget_path}")
        from unlearn.concept_data import read_corpus
        forget = read_corpus(forget_path)
    else:
        print(f"[concept_data] building forget corpus from {len(spec.wikipedia_titles)} "
              "Wikipedia article(s)")
        forget = build_forget_corpus(spec, progress=not args.no_progress)
        write_corpus(forget, forget_path)
        print(f"[concept_data] wrote {len(forget)} forget sentences -> {forget_path}")

    # ----- Retain corpus -----
    if args.skip_retain and retain_path.exists():
        print(f"[concept_data] reusing existing {retain_path}")
        from unlearn.concept_data import read_corpus
        retain = read_corpus(retain_path)
    else:
        print(f"[concept_data] building retain corpus (source={spec.retain_source})")
        retain = build_retain_corpus(spec, progress=not args.no_progress)
        write_corpus(retain, retain_path)
        print(f"[concept_data] wrote {len(retain)} retain sentences -> {retain_path}")

    # ----- Probes -----
    probes_path = out_dir / "probes.jsonl"
    write_probes(spec.probes, probes_path)
    print(f"[concept_data] wrote {len(spec.probes)} probes -> {probes_path}")

    # ----- Statistics -----
    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        print(f"[concept_data] loading tokenizer {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    forget_stats = corpus_statistics(
        forget, tokenizer=tokenizer, term_counts_for=spec.all_positive_terms,
    )
    retain_stats = corpus_statistics(retain, tokenizer=tokenizer)
    stats_payload = {
        "tokenizer": args.tokenizer,
        "forget": asdict(forget_stats),
        "retain": asdict(retain_stats),
        "n_probes": len(spec.probes),
        "n_probe_prompts_with_paraphrases": sum(len(p.all_prompts()) for p in spec.probes),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats_payload, f, indent=2, ensure_ascii=False)
    print(f"[concept_data] wrote {out_dir / 'stats.json'}")

    # ----- Inspection -----
    inspection_md = render_inspection_md(
        spec, forget, retain,
        forget_stats=forget_stats, retain_stats=retain_stats,
    )
    (out_dir / "inspection.md").write_text(inspection_md)
    print(f"[concept_data] wrote {out_dir / 'inspection.md'}")

    # ----- Summary -----
    print()
    print(f"=== Summary ===")
    print(f"  forget : {forget_stats.n_sentences:>5} sentences"
          + (f"  ({forget_stats.n_tokens_total:,} tokens)"
             if forget_stats.n_tokens_total is not None else ""))
    print(f"  retain : {retain_stats.n_sentences:>5} sentences"
          + (f"  ({retain_stats.n_tokens_total:,} tokens)"
             if retain_stats.n_tokens_total is not None else ""))
    print(f"  probes : {len(spec.probes):>5} base prompts "
          f"({stats_payload['n_probe_prompts_with_paraphrases']} including paraphrases)")
    if forget_stats.term_match_counts:
        print(f"  forget term coverage:")
        for term, n in sorted(forget_stats.term_match_counts.items(), key=lambda kv: -kv[1]):
            print(f"    - {term!r}: {n}")
    if len(forget) < spec.target_forget_sentences:
        print(f"  [warn] forget target was {spec.target_forget_sentences}; got {len(forget)}. "
              "Add more wikipedia_titles to the spec or relax min/max_sentence_chars.")
    if len(retain) < spec.target_retain_sentences:
        print(f"  [warn] retain target was {spec.target_retain_sentences}; got {len(retain)}. "
              "Raise max_articles or switch to retain_source: hf_dataset.")


if __name__ == "__main__":
    main()
