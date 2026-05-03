"""Harvest residual-stream activations from a HF causal LM.

Harvesting from just a single split for one-off experiments, 
pulling activations from a different dataset for evaluation 
purposes, or re-running a specific split if something went wrong. 

For large-scale training/eval, see harvest_splits.py which orchestrates 
multiple runs of this script to create train/val/test splits in one go.

Example:
    python scripts/harvest_activations.py \
        --model google/gemma-2-2b \
        --layer -1 \
        --max-samples 1000 \
        --output-dir activations/gemma2b_mid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/harvest_activations.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae.data import HarvestConfig, harvest_activations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer block index to hook. Gemma‑2‑2B has 26 transformer layers indexed 0–25; -1 calls the middle layer function.",
    )
    p.add_argument("--seq-len", type=int, default=256,
                   help="Per-document truncation length (one doc per forward pass).")
    p.add_argument("--tokens-per-shard", type=int, default=500_000)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT") # 10M subset of fineweb-edu for testing. Use `--dataset-config sample-100BT` for 100M subset, or remove for full dataset.
    p.add_argument("--dataset-split", default="train") # fineweb-edu doesn't have standard train/val/test splits, so we just use the "train" split and carve out our own splits with `--max-samples` and `--skip-samples`.
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max number of documents to process (None = stream the whole dataset).")
    # Use `--skip-samples N` to carve out disjoint train/val/test splits from the same dataset split. 
    # For example, with `--max-samples 1000000` and `--skip-samples 1000000`, this script would harvest the second million documents of the stream.
    # Helps reproduce the exact same split boundaries from a multi-split harvest.
    p.add_argument("--skip-samples", type=int, default=0,
                   help="Skip the first N documents of the stream. Use to carve disjoint train/val/test slices.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HarvestConfig(
        model_name=args.model,
        layer_idx=args.layer,
        d_model=0,  # filled in from model config
        seq_len=args.seq_len,
        tokens_per_shard=args.tokens_per_shard,
        dtype=args.dtype,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_field=args.text_field,
        max_samples=args.max_samples,
        skip_samples=args.skip_samples,
        output_dir=args.output_dir,
    )
    harvest_activations(cfg)


if __name__ == "__main__":
    main()
    # HuggingFace streaming datasets keeps background threads alive after the iterator. Force exit.
    sys.exit(0)
