"""Plot SAE training curves and cross-split evaluation comparisons.

Examples:

    # one run, training curves only
    python scripts/plot_metrics.py \
        --train runs/gemma2b_layer13_topk_8d \
        --output plots/

    # compare train+val+test eval reports
    python scripts/plot_metrics.py \
        --eval train=runs/.../eval_train val=runs/.../eval_val test=runs/.../eval_test \
        --output plots/

    # both, with custom labels for training runs
    python scripts/plot_metrics.py \
        --train baseline=runs/A k64=runs/B \
        --eval train=runs/A/eval_train val=runs/A/eval_val \
        --output plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae.plotting import plot_all


def _parse_kv_or_path(items: list[str]) -> dict[str, Path]:
    """Accept both ``label=path`` pairs and bare paths (label = dir name)."""
    out: dict[str, Path] = {}
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).name
        out[label] = Path(path)
    return out


def _resolve_train(label_to_path: dict[str, Path]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for label, p in label_to_path.items():
        if p.is_dir():
            cand = p / "metrics.jsonl"
            if not cand.exists():
                raise FileNotFoundError(f"No metrics.jsonl in {p}")
            resolved[label] = cand
        else:
            resolved[label] = p
    return resolved


def _resolve_eval(label_to_path: dict[str, Path]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for label, p in label_to_path.items():
        if p.is_dir():
            cand = p / "eval_report.json"
            if not cand.exists():
                raise FileNotFoundError(f"No eval_report.json in {p}")
            resolved[label] = cand
        else:
            resolved[label] = p
    return resolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="*", default=[],
                   help="One or more 'label=run_dir' (or run_dir / metrics.jsonl path).")
    p.add_argument("--eval", nargs="*", default=[],
                   help="One or more 'label=eval_dir' (or eval_dir / eval_report.json path).")
    p.add_argument("--output", required=True, help="Output directory for the PNG figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = _resolve_train(_parse_kv_or_path(args.train)) if args.train else None
    evals = _resolve_eval(_parse_kv_or_path(args.eval)) if args.eval else None
    if not runs and not evals:
        raise SystemExit("Nothing to plot: pass --train and/or --eval")
    written = plot_all(runs, evals, args.output)
    for k, path in written.items():
        print(f"[plot] wrote {k}: {path}")


if __name__ == "__main__":
    main()
