"""Quick dataset inspection for TOFU.

Prints split sizes, columns, and a couple of example rows for each
configuration. No GPU / no model needed.

Example:
    python scripts/inspect_tofu.py
    python scripts/inspect_tofu.py --configs full forget01 retain99 real_authors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.tofu_data import TOFU_DATASET_NAME, TOFU_KNOWN_CONFIGS, load_tofu_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=["full", "forget01", "forget10",
                                                    "retain99", "retain90",
                                                    "real_authors", "world_facts"],
                   help="Configurations to inspect (default: M3-relevant ones).")
    p.add_argument("--n-examples", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"dataset: {TOFU_DATASET_NAME}")
    print(f"known configs: {TOFU_KNOWN_CONFIGS}\n")
    for cfg in args.configs:
        try:
            ds = load_tofu_split(cfg, split="train")
        except Exception as e:
            print(f"[{cfg}] FAILED to load: {e}")
            continue
        print(f"=== {cfg} ===")
        print(f"  n_rows={len(ds)}  columns={ds.column_names}")
        for i, row in enumerate(ds.select(range(min(args.n_examples, len(ds))))):
            print(f"  [{i}] Q: {row['question'][:200]}")
            print(f"      A: {row['answer'][:200]}")
        print()


if __name__ == "__main__":
    main()
