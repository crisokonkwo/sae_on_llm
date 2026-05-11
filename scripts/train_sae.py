"""Config-driven SAE training.

Primary reproducible entry point:

    python scripts/train_sae.py --config configs/gemma2b_mid_topk.yaml

For CLI usage use:

    python scripts/train_sae.py \
        --shard-dir activations/gemma2b_mid \
        --output-dir runs/gemma2b_mid_topk/run_1 \
        --n-features 18432 --k 64 --max-steps 20000

If ``--shard-dir`` points at a splits root produced by ``harvest_splits.py``,
training automatically uses ``train/`` and validation defaults to ``val/`` if
present.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sae import SAE, SAEConfig
from sae.dataset import ActivationDataset
from sae.eval import reconstruction_metrics
from sae.plotting import plot_training_curves
from sae.train import TrainConfig, Trainer, build_dataset, load_checkpoint


# -----------------------------------------------------------------------------
# Config loading / nested defaults
# -----------------------------------------------------------------------------
def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Install PyYAML or use a .json config: pip install pyyaml") from e
        data = yaml.safe_load(p.read_text())
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}; use .yaml/.yml/.json")
    return data or {}


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML/JSON config file. CLI flags override config values.")

    # data
    p.add_argument("--model-name", default=_get(defaults, "model.name"),
                   help="Model name/path used to harvest activations (recorded for provenance).")
    p.add_argument("--layer", type=int, default=_get(defaults, "model.layer_idx"),
                   help="Layer index for provenance; training uses shard meta for d_model.")
    p.add_argument("--hook-point", default=_get(defaults, "model.hook_point", "resid_post"))
    p.add_argument("--shard-dir", default=_get(defaults, "data.shard_dir"))
    p.add_argument("--val-shard-dir", default=_get(defaults, "data.val_shard_dir"),
                   help="Optional held-out activation shards for post-train validation.")
    p.add_argument("--output-dir", default=_get(defaults, "output_dir"))
    p.add_argument("--batch-size", type=int, default=_get(defaults, "data.batch_size", 4096))
    p.add_argument("--buffer-shards", type=int, default=_get(defaults, "data.buffer_shards", 4))

    # SAE arch
    p.add_argument("--d-model", type=int, default=_get(defaults, "sae.d_model"),
                   help="Defaults to train meta.json -> resolved_d_model.")
    p.add_argument("--n-features", type=int, default=_get(defaults, "sae.n_features"),
                   help="Dictionary size. Defaults to expansion_factor * d_model.")
    p.add_argument("--expansion-factor", type=int, default=_get(defaults, "sae.expansion_factor", 8))
    p.add_argument("--sparsity-mode", default=_get(defaults, "sae.sparsity_mode", "topk"),
                   choices=["topk", "l1", "jumprelu", "gated"])
    p.add_argument("--k", type=int, default=_get(defaults, "sae.topk.k", 64))
    p.add_argument("--k-aux", type=int, default=_get(defaults, "sae.topk.k_aux"))
    p.add_argument("--aux-coef", type=float, default=_get(defaults, "sae.topk.aux_coef", 1.0 / 32.0))
    p.add_argument("--dead-steps-threshold", type=int,
                   default=_get(defaults, "sae.topk.dead_steps_threshold", 1000))

    # optimisation
    p.add_argument("--lr", type=float, default=_get(defaults, "train.lr", 3e-4))
    p.add_argument("--warmup-steps", type=int, default=_get(defaults, "train.warmup_steps", 500))
    p.add_argument("--max-steps", type=int, default=_get(defaults, "train.max_steps", 20_000))
    p.add_argument("--grad-clip", type=float, default=_get(defaults, "train.grad_clip", 1.0))
    p.add_argument("--log-every", type=int, default=_get(defaults, "train.log_every", 50))
    p.add_argument("--ckpt-every", type=int, default=_get(defaults, "train.ckpt_every", 2000))
    p.add_argument("--seed", type=int, default=_get(defaults, "train.seed", 0))
    p.add_argument("--device", default=_get(defaults, "train.device", "cuda"))
    p.add_argument("--compute-dtype", default=_get(defaults, "train.compute_dtype", "float32"),
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--resume", default=_get(defaults, "train.resume"))
    p.add_argument("--no-progress", action="store_true", default=not _get(defaults, "train.progress", True))

    # validation / reports
    p.add_argument("--validate-after", action="store_true",
                   default=_get(defaults, "validation.validate_after", True),
                   help="Run reconstruction validation after training when val_shard_dir exists.")
    p.add_argument("--validation-max-batches", type=int,
                   default=_get(defaults, "validation.max_batches", 200))
    p.add_argument("--validation-batch-size", type=int,
                   default=_get(defaults, "validation.batch_size"),
                   help="Defaults to training batch size.")
    p.add_argument("--plot-after", action="store_true",
                   default=_get(defaults, "validation.plot_after", True),
                   help="Write training_curves.png after training.")
    return p


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    cfg_args, _ = pre.parse_known_args()
    defaults = _load_config(cfg_args.config)
    return _build_parser(defaults).parse_args()


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------
def _has_shards(p: Path) -> bool:
    return p.is_dir() and any(p.glob("shard_*.pt"))


def _resolve_train_shard_dir(shard_dir: Path) -> Path:
    """Accept either a direct shard dir or a splits root (with ``train/``)."""
    if _has_shards(shard_dir):
        return shard_dir
    train_sub = shard_dir / "train"
    if _has_shards(train_sub):
        print(f"[main] {shard_dir} looks like a splits root; using {train_sub}")
        return train_sub
    raise FileNotFoundError(
        f"No shard_*.pt files in {shard_dir} or {shard_dir}/train. "
        "Pass either a direct shard directory or a splits root."
    )


def _resolve_val_shard_dir(raw_train_arg: str | None, raw_val_arg: str | None) -> Path | None:
    if raw_val_arg:
        p = Path(raw_val_arg)
        if _has_shards(p):
            return p
        raise FileNotFoundError(f"No shard_*.pt files in val_shard_dir={p}")
    if raw_train_arg:
        val_sub = Path(raw_train_arg) / "val"
        if _has_shards(val_sub):
            print(f"[main] auto-detected validation split: {val_sub}")
            return val_sub
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_run_manifest(out_dir: Path, args: argparse.Namespace, train_meta: dict[str, Any],
                        val_dir: Path | None) -> None:
    manifest = {
        "entrypoint": "scripts/train_sae.py",
        "config": args.config,
        "model": {
            "name": args.model_name,
            "layer_idx": args.layer,
            "hook_point": args.hook_point,
        },
        "data": {
            "train_shard_dir": args.shard_dir,
            "resolved_train_shard_dir": train_meta.get("_resolved_train_shard_dir"),
            "val_shard_dir": str(val_dir) if val_dir is not None else None,
            "train_meta": {k: v for k, v in train_meta.items() if not k.startswith("_")},
        },
        "cli_args": vars(args),
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.shard_dir is None:
        raise ValueError("--shard-dir is required (or set data.shard_dir in --config)")
    if args.output_dir is None:
        raise ValueError("--output-dir is required (or set output_dir in --config)")

    _set_reproducible_seed(args.seed)

    train_shard_dir = _resolve_train_shard_dir(Path(args.shard_dir))
    val_shard_dir = _resolve_val_shard_dir(args.shard_dir, args.val_shard_dir)

    meta_path = train_shard_dir / "meta.json"
    train_meta = json.load(open(meta_path)) if meta_path.exists() else {}
    train_meta["_resolved_train_shard_dir"] = str(train_shard_dir)

    d_model = args.d_model or train_meta.get("resolved_d_model")
    if d_model is None:
        raise ValueError("--d-model not given and not present in train meta.json")
    n_features = args.n_features or (args.expansion_factor * d_model)

    print(
        f"[main] d_model={d_model}  n_features={n_features} "
        f"(expansion={n_features / d_model:.1f}x)  mode={args.sparsity_mode}"
    )

    sparsity_kwargs: dict[str, Any] = {}
    if args.sparsity_mode == "topk":
        sparsity_kwargs = {
            "k": args.k,
            "k_aux": args.k_aux,
            "aux_coef": args.aux_coef,
            "dead_steps_threshold": args.dead_steps_threshold,
        }

    sae_cfg = SAEConfig(
        d_model=d_model,
        n_features=n_features,
        sparsity_mode=args.sparsity_mode,
        sparsity_kwargs=sparsity_kwargs,
    )
    train_cfg = TrainConfig(
        shard_dir=str(train_shard_dir),
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        buffer_shards=args.buffer_shards,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
        device=args.device,
        compute_dtype=args.compute_dtype,
        progress=not args.no_progress,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(out_dir, args, train_meta, val_shard_dir)

    # Build the SAE and Trainer. The SAE constructor handles decoder init and sparsity function setup based on the config.
    sae = SAE(sae_cfg)
    trainer = Trainer(sae, sae_cfg, train_cfg)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), trainer.sae, trainer.optimizer)
        print(f"[main] resumed from {args.resume} at step {start_step}")

    dataset = build_dataset(train_cfg, infinite=True)
    dataset.progress = not args.no_progress
    print(
        f"[main] dataset: train={train_shard_dir} batch={train_cfg.batch_size} "
        f"buffer_shards={train_cfg.buffer_shards}"
    )
    print(
        f"[main] starting training for {train_cfg.max_steps} steps on "
        f"{train_cfg.device} dtype={train_cfg.compute_dtype}"
    )
    trainer.train(dataset, start_step=start_step)

    # Auto-validation report (reconstruction-only; CE requires loading the LM and is handled by eval_sae.py).
    if args.validate_after and val_shard_dir is not None:
        print(f"[main] running post-train validation on {val_shard_dir}")
        val_batch = args.validation_batch_size or args.batch_size
        val_ds = ActivationDataset(
            shard_dir=val_shard_dir,
            batch_size=val_batch,
            buffer_shards=1,
            shuffle=False,
            seed=args.seed,
            infinite=False,
            drop_last=False,
            progress=False,
        )
        report = {
            "ckpt": str(out_dir / "ckpt_final.pt"),
            "split": "val",
            "shard_dir": str(val_shard_dir),
            "max_batches": args.validation_max_batches,
            "reconstruction": reconstruction_metrics(
                trainer.sae, val_ds, max_batches=args.validation_max_batches, device=trainer.device
            ),
        }
        report_path = out_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[main] wrote {report_path}")
    elif args.validate_after:
        print("[main] validation requested but no val split found; skipping")

    if args.plot_after:
        metrics_path = out_dir / "metrics.jsonl"
        if metrics_path.exists():
            plot_path = out_dir / "training_curves.png"
            plot_training_curves({out_dir.name: metrics_path}, plot_path)
            print(f"[main] wrote {plot_path}")


if __name__ == "__main__":
    main()
