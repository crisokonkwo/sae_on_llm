"""Config-driven LoRA finetuning of Gemma-2-2B (or similar) on TOFU.

Examples:
    # Config-only
    python scripts/finetune_tofu.py --config configs/tofu_lora_gemma2b.yaml

    # Config + CLI override
    python scripts/finetune_tofu.py --config configs/tofu_lora_gemma2b.yaml \
        --output-dir runs/gemma2b_tofu_ft/run_2 --epochs 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from finetune.finetune import FinetuneConfig, LoRAConfig, TOFUTrainer, build_model_and_tokenizer
from finetune.tofu_data import CausalLMCollator, build_tofu_train_dataset


# -----------------------------------------------------------------------------
# Config loading (mirrors scripts/train_sae.py)
# -----------------------------------------------------------------------------
def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix in {".yaml", ".yml"}:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    if p.suffix == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported config extension: {p.suffix}")


def _get(cfg: dict, path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    # model / data
    p.add_argument("--model-name", default=_get(defaults, "model.name", "google/gemma-2-2b"))
    p.add_argument("--dataset-config", default=_get(defaults, "data.config", "full"))
    p.add_argument("--dataset-split", default=_get(defaults, "data.split", "train"))
    p.add_argument("--prompt-template", default=_get(defaults, "data.prompt_template", "Question: {question}\nAnswer:"))
    p.add_argument("--max-length", type=int, default=_get(defaults, "data.max_length", 512))
    p.add_argument("--mask-prompt", action="store_true", default=_get(defaults, "data.mask_prompt", True))
    # optim
    p.add_argument("--output-dir", default=_get(defaults, "output_dir", "runs/gemma2b_tofu_ft"))
    p.add_argument("--batch-size", type=int, default=_get(defaults, "train.batch_size", 4))
    p.add_argument("--grad-accum-steps", type=int, default=_get(defaults, "train.grad_accum_steps", 4))
    p.add_argument("--lr", type=float, default=_get(defaults, "train.lr", 2e-4))
    p.add_argument("--warmup-steps", type=int, default=_get(defaults, "train.warmup_steps", 50))
    p.add_argument("--max-steps", type=int, default=_get(defaults, "train.max_steps", 0))
    p.add_argument("--epochs", type=int, default=_get(defaults, "train.epochs", 5))
    p.add_argument("--weight-decay", type=float, default=_get(defaults, "train.weight_decay", 0.0))
    p.add_argument("--grad-clip", type=float, default=_get(defaults, "train.grad_clip", 1.0))
    p.add_argument("--log-every", type=int, default=_get(defaults, "train.log_every", 25))
    p.add_argument("--ckpt-every", type=int, default=_get(defaults, "train.ckpt_every", 500))
    p.add_argument("--seed", type=int, default=_get(defaults, "train.seed", 0))
    p.add_argument("--device", default=_get(defaults, "train.device", "cuda"))
    p.add_argument("--compute-dtype", default=_get(defaults, "train.compute_dtype", "bfloat16"),
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no-progress", action="store_true", default=not _get(defaults, "train.progress", True))
    p.add_argument("--load-in-4bit", action="store_true", default=_get(defaults, "train.load_in_4bit", False))
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   default=not _get(defaults, "train.gradient_checkpointing", True))
    # LoRA
    p.add_argument("--lora-r", type=int, default=_get(defaults, "lora.r", 16)) # Relatively high rank for good performance on a strong base model, but still much smaller than the full 2B params.
    p.add_argument("--lora-alpha", type=int, default=_get(defaults, "lora.alpha", 32)) # Higher alpha can help stabilize training with higher rank, but the optimal value may depend on the specific model and dataset.
    p.add_argument("--lora-dropout", type=float, default=_get(defaults, "lora.dropout", 0.05))
    p.add_argument("--lora-target-modules", nargs="*",
                   default=_get(defaults, "lora.target_modules",
                                ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])) # Targeting all linear layers in the attention and feedforward blocks is a common choice for good performance, but you could experiment with a smaller set for faster training or a larger set for potentially better performance at the cost of more trainable parameters.
    return p


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    cfg_args, _ = pre.parse_known_args()
    defaults = _load_config(cfg_args.config)
    return _build_parser(defaults).parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = FinetuneConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        prompt_template=args.prompt_template,
        max_length=args.max_length,
        mask_prompt=args.mask_prompt,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
        device=args.device,
        compute_dtype=args.compute_dtype,
        progress=not args.no_progress,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lora=LoRAConfig(
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            target_modules=tuple(args.lora_target_modules),
        ),
    )

    print(f"[ft] loading base model {cfg.model_name}")
    model, tokenizer = build_model_and_tokenizer(cfg)
    # Print trainable param count for sanity.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[ft] trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.4f}%)")

    print(f"[ft] tokenising TOFU split={cfg.dataset_config}/{cfg.dataset_split}")
    train_ds = build_tofu_train_dataset(
        tokenizer,
        config=cfg.dataset_config,
        split=cfg.dataset_split,
        prompt_template=cfg.prompt_template,
        max_length=cfg.max_length,
        mask_prompt=cfg.mask_prompt,
    )
    print(f"[ft] tokenised {len(train_ds)} examples")

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )

    trainer = TOFUTrainer(model, tokenizer, cfg)
    trainer.train(loader)


if __name__ == "__main__":
    main()
