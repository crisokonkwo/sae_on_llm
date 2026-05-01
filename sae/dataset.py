"""Sharded activation dataset.

Reads ``shard_*.pt`` files written by :mod:`sae.data` and yields mini-batches
of activation rows. Mixing strategy: keep a sliding window of ``buffer_shards``
shards in RAM, concatenate them, shuffle, and yield batches until the buffer
drains; then refill with the next shards. Cheap, deterministic with a seed,
and good enough for SAE training.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


def list_shards(shard_dir: str | Path) -> list[Path]:
    paths = sorted(Path(shard_dir).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files under {shard_dir}")
    # print(f"[dataset] found {len(paths)} shards under {shard_dir} -> {paths[0]} ... {paths[-1]}")
    return paths


def load_meta(shard_dir: str | Path) -> dict:
    meta_path = Path(shard_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


class ActivationDataset(IterableDataset):
    def __init__(
        self,
        shard_dir: str | Path,
        batch_size: int,
        buffer_shards: int = 4,
        shuffle: bool = True,
        seed: int = 0,
        infinite: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.buffer_shards = max(1, buffer_shards) # at least 1 shard in buffer
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite # whether to loop infinitely over the dataset (default True, set to False for one epoch)
        self.drop_last = drop_last
        self.shards = list_shards(self.shard_dir)
        self.meta = load_meta(self.shard_dir)

    @property
    def d_model(self) -> int | None:
        return self.meta.get("resolved_d_model")

    def _shard_order(self, epoch: int) -> list[Path]:
        if not self.shuffle:
            return list(self.shards)
        rng = random.Random(self.seed + epoch)
        order = list(self.shards)
        rng.shuffle(order)
        return order

    # Iterate over shards in buffer-sized windows, yielding batches until the buffer drains; then refill with the next shards.
    def __iter__(self) -> Iterator[torch.Tensor]:
        epoch = 0
        while True:
            order = self._shard_order(epoch) # get shard order for this epoch (reshuffled each epoch if shuffle=True)
            for start in range(0, len(order), self.buffer_shards): # iterate over shards in buffer-sized windows
                window = order[start : start + self.buffer_shards]
                # print(f"[dataset] epoch {epoch} loading buffer shards {start}–{start+len(window)-1}: {[p.name for p in window]}")
                tensors = [torch.load(p, map_location="cpu") for p in window]
                buf = torch.cat(tensors, dim=0)
                if self.shuffle:
                    g = torch.Generator().manual_seed(self.seed + epoch * 10_000 + start)
                    perm = torch.randperm(buf.shape[0], generator=g)
                    buf = buf[perm]
                n = buf.shape[0]
                # print(f"[dataset] epoch {epoch} buffer loaded with {n} rows and d_model={buf.shape[1]}")
                # If drop_last is True, drop the last incomplete batch; otherwise, yield it as is.
                limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
                for i in range(0, limit, self.batch_size):
                    yield buf[i : i + self.batch_size]
            epoch += 1
            if not self.infinite:
                return

    def __len__(self) -> int:
        # Approximate: rows-per-shard * num_shards / batch_size.
        tokens = self.meta.get("tokens_written")
        if tokens is None:
            raise TypeError("Dataset length unknown (no tokens_written in meta.json)")
        return math.floor(tokens / self.batch_size)
