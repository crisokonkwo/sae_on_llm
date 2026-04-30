"""Activation harvesting: stream text through an LM and dump residual-stream
activations to sharded files on disk.

Output format: a directory of ``shard_{idx:05d}.pt`` files, each a tensor of
shape ``(num_tokens_in_shard, d_model)`` in fp16, plus a ``meta.json`` with
the run config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader

from .hooks import capture_residual_stream, get_num_layers, middle_layer_index


@dataclass
class HarvestConfig:
    model_name: str
    layer_idx: int
    d_model: int
    seq_len: int
    batch_size: int
    total_tokens: int
    tokens_per_shard: int
    dtype: str  # "float16" | "bfloat16" | "float32"
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    text_field: str
    output_dir: str


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


# def _stream_text(
#     dataset_name: str,
#     dataset_config: str | None,
#     split: str,
#     text_field: str,
# ) -> Iterator[str]:
#     from datasets import load_dataset

#     ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
#     for ex in ds:
#         text = ex.get(text_field)
#         if text:
#             yield text


# def _tokenize_stream(
#     text_iter: Iterable[str],
#     tokenizer,
#     seq_len: int,
#     batch_size: int,
# ) -> Iterator[torch.Tensor]:
#     """Pack a stream of strings into ``(batch_size, seq_len)`` token tensors.

#     Uses simple concatenation with EOS as the separator and chunking — same
#     recipe most SAE papers use to avoid wasting tokens on padding.
#     """
#     eos = tokenizer.eos_token_id
#     if eos is None:
#         eos = tokenizer.bos_token_id  # fallback
#     buf: list[int] = []
#     needed = seq_len * batch_size
#     for text in text_iter:
#         ids = tokenizer.encode(text, add_special_tokens=False)
#         buf.extend(ids)
#         buf.append(eos)
#         while len(buf) >= needed:
#             chunk = torch.tensor(buf[:needed], dtype=torch.long).view(batch_size, seq_len)
#             yield chunk
#             buf = buf[needed:]


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

#     def add(self, acts: torch.Tensor) -> None:
#         acts = acts.detach().to("cpu", dtype=self.dtype)
#         self.buffer.append(acts)
#         self.buffered_rows += acts.shape[0]
#         while self.buffered_rows >= self.tokens_per_shard:
#             self._flush_one()

#     def _flush_one(self) -> None:
#         cat = torch.cat(self.buffer, dim=0)
#         shard, rest = cat[: self.tokens_per_shard], cat[self.tokens_per_shard :]
#         path = self.out_dir / f"shard_{self.shard_idx:05d}.pt"
#         torch.save(shard.contiguous(), path)
#         self.shard_idx += 1
#         self.total_written += shard.shape[0]
#         self.buffer = [rest] if rest.numel() > 0 else []
#         self.buffered_rows = rest.shape[0] if rest.numel() > 0 else 0

#     def finalize(self) -> None:
#         if self.buffered_rows == 0:
#             return
#         cat = torch.cat(self.buffer, dim=0)
#         path = self.out_dir / f"shard_{self.shard_idx:05d}.pt"
#         torch.save(cat.contiguous(), path)
#         self.shard_idx += 1
#         self.total_written += cat.shape[0]
#         self.buffer = []
#         self.buffered_rows = 0


@torch.no_grad()
def harvest_activations(cfg: HarvestConfig) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _torch_dtype(cfg.dtype)

    print(f"[harvest] loading {cfg.model_name} on {device} ({cfg.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval() # make sure we're in eval mode (some models have dropout in the residual stream which would mess up harvesting)
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

    # Save meta upfront (also rewritten at the end with totals). This way if the run gets interrupted, we at least have the config and can see how many tokens/shards were written.
    resolved = asdict(cfg)
    resolved["resolved_layer_idx"] = layer_idx
    resolved["num_layers"] = n_layers
    resolved["resolved_d_model"] = d_model
    # print(f"[harvest] starting with config: {json.dumps(resolved, indent=2)}")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(resolved, f, indent=2)
    
    text_iter = _stream_text(cfg.dataset_name, cfg.dataset_config, cfg.dataset_split, cfg.text_field)
    # token_iter = _tokenize_stream(text_iter, tokenizer, cfg.seq_len, cfg.batch_size)

    # tokens_seen = 0
    # with capture_residual_stream(model, layer_idx) as catcher:
    #     for batch in token_iter:
    #         batch = batch.to(device)
    #         model(input_ids=batch, use_cache=False)
    #         acts = catcher.activations  # (B, T, d_model)
    #         assert acts is not None
    #         flat = acts.reshape(-1, acts.shape[-1])
    #         writer.add(flat)
    #         tokens_seen += flat.shape[0]
    #         if tokens_seen % (cfg.batch_size * cfg.seq_len * 10) == 0:
    #             print(f"[harvest] tokens={tokens_seen:,} shards={writer.shard_idx}")
    #         if tokens_seen >= cfg.total_tokens:
    #             break

    # writer.finalize()
    # resolved["tokens_written"] = writer.total_written
    # resolved["num_shards"] = writer.shard_idx
    # with open(out_dir / "meta.json", "w") as f:
    #     json.dump(resolved, f, indent=2)
    # print(f"[harvest] done. wrote {writer.total_written:,} tokens in {writer.shard_idx} shards -> {out_dir}")
