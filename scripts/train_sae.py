"""Train an SAE on harvested activations.

Example:
    python scripts/train_sae.py \
        --shard-dir activations/gemma2b_mid \
        --output-dir runs/gemma2b_mid_topk \
        --n-features 18432 \
        --k 64 \
        --max-steps 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae import SAE, SAEConfig
from sae.train import TrainConfig, Trainer, build_dataset, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--buffer-shards", type=int, default=4)
    # SAE arch
    p.add_argument("--d-model", type=int, default=None,
                   help="Defaults to meta.json -> resolved_d_model.")
    p.add_argument("--n-features", type=int, default=None,
                   help="Dictionary size. Defaults to 8 * d_model.")
    p.add_argument("--sparsity-mode", default="topk", choices=["topk", "l1", "jumprelu", "gated"])
    p.add_argument("--k", type=int, default=64, help="Top-k active features per token.")
    p.add_argument("--k-aux", type=int, default=None, 
                   help="If set, also optimize an auxiliary top-k with this many features, and add its loss to the main top-k loss with --aux-coef. This can help mitigate feature 'death' where some features never win the competition to be in the top-k.")
    p.add_argument("--aux-coef", type=float, default=1.0 / 32.0)
    p.add_argument("--dead-steps-threshold", type=int, default=1000)
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=20_000) # adjust as needed; 20k steps with batch size 4096. For full dataset or different batch sizes, adjust accordingly.
    p.add_argument("--grad-clip", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000) # save every 2000 steps by default, so that even short runs have some checkpoints. Adjust as needed.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--resume", default=None, help="Path to ckpt_*.pt to resume from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Pull d_model from harvest meta if not given.
    meta_path = Path(args.shard_dir) / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    d_model = args.d_model or meta.get("resolved_d_model")
    if d_model is None:
        raise ValueError("--d-model not given and not present in meta.json")
    n_features = args.n_features or (8 * d_model)
    print(f"[main] d_model={d_model}  n_features={n_features}  mode={args.sparsity_mode}")

    sparsity_kwargs = {}
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
        shard_dir=args.shard_dir,
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
    )

    # Initialize SAE and Trainer, optionally load checkpoint, build dataset, and start training.
    sae = SAE(sae_cfg)
    trainer = Trainer(sae, sae_cfg, train_cfg)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), trainer.sae, trainer.optimizer)
        print(f"[main] resumed from {args.resume} at step {start_step}")

    dataset = build_dataset(train_cfg, infinite=True)
    print(f"[main] dataset built with batch size {train_cfg.batch_size} and buffer shards {train_cfg.buffer_shards}")
    
    # print(f"\n[main] showing one batch from the dataset for sanity check: dtype={dataset.d_model}  batch shape={iter(dataset).__next__().shape}")
    # for batch in dataset:
    #     print(batch.shape)
    #     break
    
    print(f"\n[main] starting training for {train_cfg.max_steps} steps")
    trainer.train(dataset, start_step=start_step)


if __name__ == "__main__":
    main()
