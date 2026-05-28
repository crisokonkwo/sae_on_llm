"""Activation harvesting: stream documents one at a time through an LM and
dump residual-stream activations to sharded files on disk.

Output format: a directory of ``shard_{idx:05d}.pt`` files, each a tensor of
shape ``(num_tokens_in_shard, d_model)``, plus a ``meta.json`` with the run
config and totals.

Design choice: simple over fast. We process one document per forward pass
(batch size = 1), truncated to ``seq_len`` tokens. No padding, no packing,
no cross-document attention masking — every captured activation row is a
real token from a real document.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch

from .hooks import capture_residual_stream, get_num_layers, middle_layer_index


@dataclass
class HarvestConfig:
    model_name: str
    layer_idx: int
    d_model: int
    seq_len: int
    tokens_per_shard: int
    dtype: str  # "float16" | "bfloat16" | "float32"
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    text_field: str
    max_samples: int | None
    skip_samples: int
    output_dir: str


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _stream_text(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    max_samples: int | None = None,
    skip_samples: int = 0,
) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    print(
        f"[harvest] streaming {dataset_name} config={dataset_config} split={split} "
        f"field='{text_field}' skip={skip_samples} take={max_samples}"
    )
    if skip_samples:
        ds = ds.skip(skip_samples)
    if max_samples is not None:
        ds = ds.take(max_samples)
    for ex in ds:
        text = ex.get(text_field)
        if text:
            yield text


class ShardWriter:
    """Buffers activation rows and flushes them to disk in fixed-size shards."""

    def __init__(self, out_dir: Path, tokens_per_shard: int, dtype: torch.dtype):
        self.out_dir = out_dir
        self.tokens_per_shard = tokens_per_shard
        self.dtype = dtype
        self.buffer: list[torch.Tensor] = []
        self.buffered_rows = 0
        self.shard_idx = 0
        self.total_written = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, acts: torch.Tensor) -> None:
        acts = acts.detach().to("cpu", dtype=self.dtype)
        self.buffer.append(acts)
        self.buffered_rows += acts.shape[0]
        while self.buffered_rows >= self.tokens_per_shard:
            self._flush_one()

    def _flush_one(self) -> None:
        cat = torch.cat(self.buffer, dim=0)
        shard, rest = cat[: self.tokens_per_shard], cat[self.tokens_per_shard :]
        path = self.out_dir / f"shard_{self.shard_idx:05d}.pt"
        torch.save(shard.contiguous(), path)
        self.shard_idx += 1
        self.total_written += shard.shape[0]
        self.buffer = [rest] if rest.numel() > 0 else []
        self.buffered_rows = rest.shape[0] if rest.numel() > 0 else 0

    def finalize(self) -> None:
        if self.buffered_rows == 0:
            return
        cat = torch.cat(self.buffer, dim=0)
        path = self.out_dir / f"shard_{self.shard_idx:05d}.pt"
        torch.save(cat.contiguous(), path)
        self.shard_idx += 1
        self.total_written += cat.shape[0]
        self.buffer = []
        self.buffered_rows = 0


@torch.no_grad()
def harvest_activations(cfg: HarvestConfig) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _torch_dtype(cfg.dtype)

    print(f"[harvest] loading {cfg.model_name} on {device} ({cfg.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        device_map=device,
    )
    model.eval()  # make sure we're in eval mode (some models have dropout in the residual stream which would mess up harvesting)
    # for name, module in model.named_modules():
    #     print(name)

    n_layers = get_num_layers(model)
    layer_idx = cfg.layer_idx if cfg.layer_idx >= 0 else middle_layer_index(model)
    d_model = model.config.hidden_size
    if cfg.d_model and cfg.d_model != d_model:
        print(f"[harvest] WARN: cfg.d_model={cfg.d_model} != model.hidden_size={d_model}")
    print(f"[harvest] model layers={n_layers}, hooking layer {layer_idx}, d_model={d_model}")

    out_dir = Path(cfg.output_dir)
    writer = ShardWriter(out_dir, cfg.tokens_per_shard, dtype)

    # Save meta upfront (also rewritten at the end with totals). 
    # This way if the run gets interrupted, we at least have the config and can see how many tokens/shards were written.
    resolved = asdict(cfg)
    resolved["resolved_layer_idx"] = layer_idx
    resolved["num_layers"] = n_layers
    resolved["resolved_d_model"] = d_model
    # print(f"[harvest] starting with config: {json.dumps(resolved, indent=2)}")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(resolved, f, indent=2)

    from tqdm.auto import tqdm

    text_iter = _stream_text(
        cfg.dataset_name, cfg.dataset_config, cfg.dataset_split, cfg.text_field,
        cfg.max_samples, cfg.skip_samples)

    docs_seen = 0
    tokens_seen = 0
    skipped_empty = 0

    pbar = tqdm(text_iter, total=cfg.max_samples, unit="doc", desc="[harvest]", smoothing=0.05)
    with capture_residual_stream(model, layer_idx) as catcher:
        for text in pbar:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=cfg.seq_len)
            if not ids:
                skipped_empty += 1
                continue
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
            model(input_ids=input_ids, use_cache=False)
            acts = catcher.activations  # (1, T, d_model)
            assert acts is not None, "Hook did not capture activations"

            flat = acts.reshape(-1, acts.shape[-1])  # (T, d_model)
            writer.add(flat)
            tokens_seen += flat.shape[0]
            docs_seen += 1

            if docs_seen % 50 == 0:
                # print(f"[harvest] docs={docs_seen} tokens={tokens_seen:,} shards={writer.shard_idx}")
                pbar.set_postfix(tokens=f"{tokens_seen:,}", shards=writer.shard_idx)
    pbar.close()

    writer.finalize()
    resolved["docs_processed"] = docs_seen
    resolved["tokens_written"] = writer.total_written
    resolved["num_shards"] = writer.shard_idx
    resolved["skipped_empty_docs"] = skipped_empty
    with open(out_dir / "meta.json", "w") as f:
        json.dump(resolved, f, indent=2)
    print(f"[harvest] done. docs={docs_seen} tokens={writer.total_written:,} "
        f"shards={writer.shard_idx} -> {out_dir}")
