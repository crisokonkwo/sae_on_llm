"""Harvest train/val/test activation shards into disjoint document ranges.

Document-level split, deterministic, no overlap. Each output dir gets its
own ``meta.json`` recording the slice ``[skip_samples, skip_samples + max_samples)``
of the source stream.

Example (100k train / 2k val / 2k test docs from FineWeb-Edu sample-10BT):

    python scripts/harvest_splits.py \
        --model google/gemma-2-2b \
        --layer -1 \
        --root activations/gemma2b_mid \
        --train-size 100000 --val-size 2000 --test-size 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae.data import HarvestConfig, harvest_activations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # split sizes
    p.add_argument("--train-size", type=int, required=True, help="# documents for the train split")
    p.add_argument("--val-size", type=int, required=True)
    p.add_argument("--test-size", type=int, required=True)
    p.add_argument("--start-offset", type=int, default=0,
                   help="Skip this many documents at the very start (useful if you want to leave a buffer).")
    # passthrough
    p.add_argument("--root", required=True,
                   help="Output root. Creates ./{train,val,test} underneath.")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--tokens-per-shard", type=int, default=500_000)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.start_offset,                                              args.train_size),
        ("val",   args.start_offset + args.train_size,                            args.val_size),
        ("test",  args.start_offset + args.train_size + args.val_size,            args.test_size),
    ]

    print(f"[splits] root={root}")
    for name, skip, size in splits:
        print(f"[splits] {name:5s}: docs[{skip}, {skip + size})  size={size}")

    for name, skip, size in splits:
        if size <= 0:
            print(f"[splits] skipping {name} (size={size})")
            continue
        out = root / name
        cfg = HarvestConfig(
            model_name=args.model,
            layer_idx=args.layer,
            d_model=0,
            seq_len=args.seq_len,
            tokens_per_shard=args.tokens_per_shard,
            dtype=args.dtype,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_field=args.text_field,
            max_samples=size,
            skip_samples=skip, # important: skip is relative to the start of the stream, not relative to the split. So for val and test splits, skip includes the train docs.
            output_dir=str(out),
        )
        print(f"\n[splits] === harvesting {name} -> {out} ===")
        harvest_activations(cfg)


if __name__ == "__main__":
    main()
    sys.exit(0)
