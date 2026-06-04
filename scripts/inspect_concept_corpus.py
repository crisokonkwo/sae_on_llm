"""Re-inspect a previously-built concept corpus.

Loads forget.jsonl, retain.jsonl, probes.jsonl from a build directory and
prints / re-renders statistics. Use this when you want to re-check the
corpus (e.g. with a different tokenizer) without rebuilding from scratch.

Example:
    python scripts/inspect_concept_corpus.py \\
        --dir runs/unlearn_eiffel_tower/concept_data \\
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
    ConceptProbe,
    CorpusSentence,
    corpus_statistics,
    read_corpus,
    read_probes,
    render_inspection_md,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="Build directory produced by build_concept_corpus.py")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer name for token-count statistics.")
    p.add_argument("--examples-per-corpus", type=int, default=8)
    p.add_argument("--re-render-md", action="store_true",
                   help="Overwrite inspection.md with the new statistics.")
    return p.parse_args()


def _load_spec(spec_path: Path, probes: list[ConceptProbe]) -> ConceptDataSpec:
    raw = json.loads(spec_path.read_text())
    raw.pop("probes", None)
    return ConceptDataSpec(probes=probes, **raw)


def main() -> None:
    args = parse_args()
    d = Path(args.dir)
    if not d.is_dir():
        raise SystemExit(f"--dir {d} does not exist")
    spec_path = d / "spec.json"
    forget_path = d / "forget.jsonl"
    retain_path = d / "retain.jsonl"
    probes_path = d / "probes.jsonl"
    for p in [spec_path, forget_path, retain_path, probes_path]:
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")

    probes = read_probes(probes_path)
    spec = _load_spec(spec_path, probes)
    forget = read_corpus(forget_path)
    retain = read_corpus(retain_path)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    forget_stats = corpus_statistics(
        forget, tokenizer=tokenizer, term_counts_for=spec.all_positive_terms,
    )
    retain_stats = corpus_statistics(retain, tokenizer=tokenizer)

    print(f"Concept: {spec.name}")
    print(f"  forget  : {forget_stats.n_sentences:>5} sentences"
          + (f"  ({forget_stats.n_tokens_total:,} tokens)"
             if forget_stats.n_tokens_total is not None else ""))
    print(f"  retain  : {retain_stats.n_sentences:>5} sentences"
          + (f"  ({retain_stats.n_tokens_total:,} tokens)"
             if retain_stats.n_tokens_total is not None else ""))
    print(f"  probes  : {len(probes):>5} base prompts "
          f"({sum(len(p.all_prompts()) for p in probes)} including paraphrases)")
    print()
    print("Sentence length (chars):")
    print(f"  forget : mean {forget_stats.n_chars_mean:.1f}  median {forget_stats.n_chars_median:.0f}  "
          f"[{forget_stats.n_chars_min}, {forget_stats.n_chars_max}]")
    print(f"  retain : mean {retain_stats.n_chars_mean:.1f}  median {retain_stats.n_chars_median:.0f}  "
          f"[{retain_stats.n_chars_min}, {retain_stats.n_chars_max}]")
    print()
    print("Sentence length (words):")
    print(f"  forget : mean {forget_stats.n_words_mean:.1f}  median {forget_stats.n_words_median:.0f}  "
          f"[{forget_stats.n_words_min}, {forget_stats.n_words_max}]")
    print(f"  retain : mean {retain_stats.n_words_mean:.1f}  median {retain_stats.n_words_median:.0f}  "
          f"[{retain_stats.n_words_min}, {retain_stats.n_words_max}]")
    if forget_stats.n_tokens_mean is not None:
        print()
        print(f"Sentence length (tokens, {args.tokenizer}):")
        print(f"  forget : mean {forget_stats.n_tokens_mean:.1f}  median {forget_stats.n_tokens_median:.0f}")
        print(f"  retain : mean {retain_stats.n_tokens_mean:.1f}  median {retain_stats.n_tokens_median:.0f}")
    if forget_stats.term_match_counts:
        print()
        print("Forget term coverage:")
        for term, n in sorted(forget_stats.term_match_counts.items(), key=lambda kv: -kv[1]):
            print(f"  - {term!r}: {n}")
    print()
    print(f"Top forget sources:")
    for src, n in forget_stats.sources_top:
        print(f"  - {src}: {n}")
    print(f"Top retain sources:")
    for src, n in retain_stats.sources_top[:5]:
        print(f"  - {src}: {n}")

    print()
    print("Sample forget sentences:")
    for s in forget[: args.examples_per_corpus]:
        print(f"  [{s.source}]  (terms: {s.matched_terms})")
        print(f"    {s.text[:200]}")
    print()
    print("Sample retain sentences:")
    for s in retain[: args.examples_per_corpus]:
        print(f"  [{s.source}]")
        print(f"    {s.text[:200]}")

    if args.re_render_md:
        md = render_inspection_md(
            spec, forget, retain,
            forget_stats=forget_stats, retain_stats=retain_stats,
            examples_per_corpus=args.examples_per_corpus,
        )
        out = d / "inspection.md"
        out.write_text(md)
        print(f"\n[concept_data] re-wrote {out}")


if __name__ == "__main__":
    main()
