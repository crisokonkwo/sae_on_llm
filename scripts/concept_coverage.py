"""Check whether candidate concepts are sufficiently represented in the
text used to harvest your activations.

Three modes:

  1. ``--shard-dir DIR``   (recommended for the unlearning experiment)
     Re-streams exactly the documents the SAE was trained on. Reads
     ``meta.json`` for dataset name/config/split/text_field/skip/take so the
     stream is reproducible. If ``DIR`` is a splits root (contains
     ``train/``, ``val/``, ``test/``), scans each present split and prints
     a per-split + per-concept comparison.

  2. ``--dataset NAME``    (ad-hoc; useful before harvesting)
     Streams the chosen HF dataset directly.

  3. ``--discover``        (concept discovery instead of checking)
     Proposes candidate concepts based on n-gram frequencies and co-occurrence patterns. 
     Only works with ``--shard-dir`` currently since it needs a stream to analyze.

Concept syntax (CLI): terms separated by ``|`` (legacy ``:``). Suffix a term
with ``?`` to mark it optional (typical for SRO relations).

Examples:
    # Use the exact slice your SAE was trained on (recommended)
    python scripts/concept_coverage.py \\
        --shard-dir activations/gemma2b_mid \\
        --concept "Eiffel Tower|Paris" \\
        --concept "Golden Gate Bridge|San Francisco" \\
        --concept "Harry Potter|Rowling" \\
        --concept "Albert Einstein|relativity" \\
        --concept "Mount Everest|Himalayas" \\
        --output runs/concept_candidates.json

    # Same but only the train split
    python scripts/concept_coverage.py \\
        --shard-dir activations/gemma2b_mid/train \\
        --concept "Eiffel Tower|Paris"

    # Ad-hoc check directly against an HF dataset (pre-harvest)
    python scripts/concept_coverage.py \\
        --dataset monology/pile-uncopyrighted --dataset-config default \\
        --concept "Eiffel Tower|Paris" --n-docs 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlearn.concept_coverage import (
    CoverageReport,
    DiscoveredConcept,
    _is_splits_root,
    coverage_check_from_dataset,
    coverage_check_from_shard_dir,
    coverage_check_from_splits_root,
    discover_concepts_from_shard_dir,
    parse_cli_concept_string,
    write_reports,
    write_split_reports,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--concept", action="append", default=[],
                   help="One concept string. Repeat the flag for multiple concepts.")
    p.add_argument("--concepts-file", default=None,
                   help="Path to a newline-delimited file of concept strings.")
    # Discovery mode
    p.add_argument("--discover", action="store_true",
                   help="Propose candidate concepts from the corpus instead of taking them as input.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of candidate concepts to propose in --discover mode.")
    p.add_argument("--ngram-max", type=int, default=5,
                   help="Max words per candidate phrase (discover mode).")
    p.add_argument("--min-doc-freq", type=int, default=100,
                   help="Minimum document frequency for a candidate (discover mode).")
    p.add_argument("--min-token-freq", type=int, default=200,
                   help="Minimum total token frequency for a candidate (discover mode).")
    p.add_argument("--max-doc-freq-ratio", type=float, default=0.02,
                   help="Discard candidates that appear in more than this fraction of docs (discover mode).")
    p.add_argument("--cooccurrence-window", type=int, default=4864,
                   help="Window (whitespace tokens) for partner suggestion (discover mode).")
    p.add_argument("--max-partners", type=int, default=3,
                   help="Number of partner suggestions per candidate (discover mode).")
    # Mode 1: shard dir (preferred)
    p.add_argument("--shard-dir", default=None,
                   help="Scan the exact text slice that produced these activation shards.")
    p.add_argument("--splits", nargs="*", default=("train", "val", "test"),
                   help="Splits to scan when --shard-dir is a splits root.")
    # Mode 2: ad-hoc HF dataset
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--n-docs", type=int, default=None,
                   help="When in shard-dir mode, override the docs-to-scan count.")
    p.add_argument("--skip-samples", type=int, default=0)
    # Scan params
    p.add_argument("--window-words", type=int, default=64)
    p.add_argument("--no-word-boundary", action="store_true",
                   help="Disable \\b anchors (allows substring matches like 'Parisian').")
    p.add_argument("--case-sensitive", action="store_true",
                   help="Disable the default case-insensitive matching.")
    p.add_argument("--max-examples", type=int, default=5)
    p.add_argument("--context-words", type=int, default=40)
    # Output
    p.add_argument("--output", default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _print_concept_report(r: CoverageReport, split_label: str | None = None) -> None:
    header = f"=== Concept: {r.concept_name!r}  ({r.kind})"
    if split_label:
        header += f"  [split={split_label}]"
    header += " ==="
    print(f"\n{header}")
    print(f"  scanned: docs={r.n_docs_scanned}  words={r.n_words_scanned:,}")
    print("  terms:")
    for t in r.terms:
        kind = "REQ" if t["required"] else "opt"
        print(f"    - [{kind}] {t['text']!r:30s}  tokens={r.term_token_freq.get(t['text'],0):>7,}  "
              f"docs={r.term_doc_freq.get(t['text'],0):>6,}")
    print(f"  docs with any required term      : {r.docs_with_any_required:>6,}")
    print(f"  docs with all required terms     : {r.docs_with_all_required:>6,}")
    print(f"  docs with window co-occurrence   : {r.docs_with_window_cooccurrence:>6,} "
          f"  (window={r.window_words} words, density={r.density_per_doc:.4g} per doc)")
    print(f"  events / million tokens (est.)   : {r.cooccurrence_per_million_tokens:>10.2f}")
    print(f"  verdict:  doc-density=[{r.verdict.upper()}]  "
          f"SAE-feature=[{r.sae_feature_likelihood.upper()}]")
    for note in r.verdict_notes:
        print(f"    - {note}")
    if r.example_contexts:
        print(f"  example contexts ({len(r.example_contexts)} shown):")
        for ex in r.example_contexts:
            print(f"    doc {ex['doc_idx']}: ...{ex['context'][:280]}...")


def _print_comparison_table(reports: list[CoverageReport], split_label: str | None = None) -> None:
    if len(reports) <= 1:
        return
    title = "Comparison" if split_label is None else f"Comparison [{split_label}]"
    print(f"\n=== {title} ===")
    print(f"  {'concept':40s}  {'verdict':10s}  {'doc-cooc':>10s}  {'density':>10s}")
    for r in reports:
        print(f"  {r.concept_name[:40]:40s}  {r.verdict:10s}  "
              f"{r.docs_with_window_cooccurrence:>10,}  {r.density_per_doc:>10.4g}")


def _print_split_summary(reports_by_split: dict[str, list[CoverageReport]]) -> None:
    """Cross-split comparison table: one row per concept, one column per split."""
    splits = list(reports_by_split.keys())
    concepts = [r.concept_name for r in next(iter(reports_by_split.values()))]
    print("\n=== Cross-split summary (window co-occurrence count / density) ===")
    header = f"  {'concept':40s}"
    for s in splits:
        header += f"  {s:>22s}"
    print(header)
    for cname in concepts:
        row = f"  {cname[:40]:40s}"
        for s in splits:
            r = next(rep for rep in reports_by_split[s] if rep.concept_name == cname)
            row += f"  {r.docs_with_window_cooccurrence:>6,} ({r.density_per_doc:>9.4g})"
        print(row)
    # Per-split verdicts: pass = all required terms present in all splits at non-trivial density.
    print("\n=== Per-concept verdict across splits ===")
    print(f"  {'concept':40s}  " + "  ".join(f"{s:>10s}" for s in splits) + "   gate")
    for cname in concepts:
        verdicts = []
        for s in splits:
            r = next(rep for rep in reports_by_split[s] if rep.concept_name == cname)
            verdicts.append((s, r.verdict))
        # Gate = train must be at least "moderate". val/test absent or weak is OK but flagged.
        gate = "PASS"
        if "train" in dict(verdicts):
            train_v = dict(verdicts)["train"]
            if train_v == "weak":
                gate = "FAIL (train weak)"
            elif any(v == "weak" for s, v in verdicts if s != "train"):
                gate = "WARN (train OK, val/test weak)"
        print(f"  {cname[:40]:40s}  "
              + "  ".join(f"{v:>10s}" for _, v in verdicts)
              + f"   {gate}")


def _print_discovered(candidates: list[DiscoveredConcept], meta: dict, split_label: str | None = None) -> None:
    title = "Discovered concept candidates"
    if split_label:
        title += f"  [split={split_label}]"
    print(f"\n=== {title} ===")
    print(f"  scanned: docs={meta['n_docs_scanned']:,}  words={meta['n_words_scanned']:,}")
    print(f"  candidates after filter: {meta['n_candidates_after_filter']:,} of {meta['n_candidates_before_filter']:,}")
    print(f"  filter: token_freq >= {meta['min_token_freq']}, "
          f"doc_freq in [{meta['min_doc_freq']}, {int(meta['max_doc_freq_ratio'] * meta['n_docs_scanned'])}]")
    if not candidates:
        print("  (no candidates passed the filter \u2014 try lowering --min-doc-freq / --min-token-freq)")
        return
    n_tokens_est = meta.get('n_tokens_estimated', 0)
    if n_tokens_est:
        print(f"  estimated tokens (words × {meta['words_to_tokens_ratio']}): {n_tokens_est:,}")
        print(f"  SAE-likelihood thresholds: likely ≥ {meta['sae_likely_threshold_per_million']:.1f}/M, "
              f"marginal ≥ {meta['sae_marginal_threshold_per_million']:.1f}/M")
    print(f"  {'rank':>4}  {'phrase':36s}  {'score':>7s}  {'tok_freq':>9s}  "
          f"{'doc_freq':>9s}  {'per_M':>8s}  {'SAE':>9s}  partners")
    for i, c in enumerate(candidates, 1):
        partners_str = ", ".join(
            f"{p['text']} ({p['cooccurrence_count']})" for p in c.suggested_partners
        ) if c.suggested_partners else "(none)"
        print(f"  {i:>4}  {c.text[:36]:36s}  {c.score:>7.3f}  {c.token_freq:>9,}  "
              f"{c.doc_freq:>9,}  {c.per_million_tokens:>8.2f}  "
              f"{c.sae_feature_likelihood:>9s}  {partners_str}")
    print("\n  example contexts for the top candidate:")
    for ex in candidates[0].example_contexts[:3]:
        if isinstance(ex, dict):
            doc_idx = ex.get('doc_idx', '?')
            ctx = ex.get('context', '')
            print(f"    - [doc {doc_idx}] ...{ctx[:280]}...")
        else:
            # Backwards compatibility: legacy string contexts.
            print(f"    - ...{ex[:280]}...")
    print("\n  CLI snippets to verify the top candidates:")
    for c in candidates:
        if c.suggested_partners:
            term = f"{c.text}|{c.suggested_partners[0]['text']}"
        else:
            term = c.text
        print(f"    --concept \"{term}\"")


def _run_discover(args: argparse.Namespace) -> None:
    discover_kwargs = dict(
        top_k=args.top_k,
        ngram_max=args.ngram_max,
        min_doc_freq=args.min_doc_freq,
        min_token_freq=args.min_token_freq,
        max_doc_freq_ratio=args.max_doc_freq_ratio,
        cooccurrence_window=args.cooccurrence_window,
        max_partners=args.max_partners,
    )
    if args.shard_dir is None:
        raise SystemExit("--discover currently requires --shard-dir.")
    path = Path(args.shard_dir)
    out_payload: dict = {}
    if _is_splits_root(path):
        splits_to_scan = [s for s in args.splits if (path / s / "meta.json").exists()]
        for split in splits_to_scan:
            cands, meta = discover_concepts_from_shard_dir(
                path / split,
                n_docs_override=args.n_docs,
                progress=not args.no_progress,
                **discover_kwargs,
            )
            _print_discovered(cands, meta, split_label=split)
            out_payload[split] = {
                "meta": meta,
                "candidates": [c.to_dict() for c in cands],
            }
    else:
        cands, meta = discover_concepts_from_shard_dir(
            path,
            n_docs_override=args.n_docs,
            progress=not args.no_progress,
            **discover_kwargs,
        )
        _print_discovered(cands, meta)
        out_payload["_"] = {
            "meta": meta,
            "candidates": [c.to_dict() for c in cands],
        }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            import json as _json
            _json.dump(out_payload, f, indent=2)
        print(f"\n[coverage] wrote {args.output}")


def main() -> None:
    args = parse_args()

    if args.discover:
        _run_discover(args)
        return

    raw_concepts: list[str] = list(args.concept)
    if args.concepts_file:
        for line in Path(args.concepts_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw_concepts.append(line)
    if not raw_concepts:
        raise SystemExit("Provide at least one --concept, --concepts-file, or --discover.")
    concepts = [parse_cli_concept_string(s) for s in raw_concepts]
    print(f"[coverage] {len(concepts)} concept(s) to scan")

    scan_kwargs = dict(
        window_words=args.window_words,
        case_insensitive=not args.case_sensitive,
        word_boundary=not args.no_word_boundary,
        max_examples=args.max_examples,
        context_words=args.context_words,
        progress=not args.no_progress,
    )

    if args.shard_dir is not None:
        path = Path(args.shard_dir)
        if _is_splits_root(path):
            print(f"[coverage] {path} is a splits root; scanning {list(args.splits)}")
            reports_by_split = coverage_check_from_splits_root(
                path, concepts, splits=tuple(args.splits),
                n_docs_override=args.n_docs, **scan_kwargs,
            )
            for split, reps in reports_by_split.items():
                print(f"\n########## split: {split} ##########")
                for r in reps:
                    _print_concept_report(r, split_label=split)
                _print_comparison_table(reps, split_label=split)
            _print_split_summary(reports_by_split)
            if args.output:
                write_split_reports(reports_by_split, args.output)
                print(f"\n[coverage] wrote {args.output}")
        else:
            print(f"[coverage] scanning single shard dir {path}")
            reports = coverage_check_from_shard_dir(
                path, concepts, n_docs_override=args.n_docs, **scan_kwargs,
            )
            for r in reports:
                _print_concept_report(r)
            _print_comparison_table(reports)
            if args.output:
                write_reports(reports, args.output)
                print(f"\n[coverage] wrote {args.output}")
    else:
        if args.dataset is None:
            raise SystemExit("Pass either --shard-dir or --dataset.")
        n_docs = args.n_docs or 5000
        print(f"[coverage] ad-hoc scan on {args.dataset}/{args.dataset_config} "
              f"({n_docs} docs, skip {args.skip_samples})")
        reports = coverage_check_from_dataset(
            concepts,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_field=args.text_field,
            n_docs=n_docs,
            skip_samples=args.skip_samples,
            **scan_kwargs,
        )
        for r in reports:
            _print_concept_report(r)
        _print_comparison_table(reports)
        if args.output:
            write_reports(reports, args.output)
            print(f"\n[coverage] wrote {args.output}")


if __name__ == "__main__":
    main()
