"""SAE training loop.

Minimal, dependency-light trainer:
  * Adam optimizer with linear LR warmup.
  * Optional gradient clipping.
  * Decoder row-norm projection after each step.
  * b_dec initialised from the first batch of activations.
  * Logging to stdout + ``metrics.jsonl``.
  * Checkpoints every ``ckpt_every`` steps + a final ``ckpt_final.pt``.
  * Resume support via ``--resume``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.optim import Adam

from .base import SAE, SAEConfig
from .dataset import ActivationDataset


@dataclass
class TrainConfig:
    # data / scale
    shard_dir: str
    output_dir: str
    batch_size: int = 4096
    buffer_shards: int = 4
    # optimisation
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float | None = 1.0
    warmup_steps: int = 500
    max_steps: int = 20_000
    # bookkeeping
    log_every: int = 50
    ckpt_every: int = 2000
    seed: int = 0
    init_b_dec_from_data: bool = True
    progress: bool = True
    # device / dtype
    device: str = "cuda"
    compute_dtype: str = "float32"  # SAE is small; fp32 is fine and stable


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _lr_at(step: int, base_lr: float, warmup: int) -> float:
    if warmup <= 0:
        return base_lr
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def save_checkpoint(
    path: Path,
    sae: SAE,
    optimizer: torch.optim.Optimizer,
    step: int,
    sae_cfg: SAEConfig,
    train_cfg: TrainConfig,
) -> None:
    payload = {
        "step": step,
        "sae_state": sae.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sae_cfg": asdict(sae_cfg),
        "train_cfg": asdict(train_cfg),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def load_checkpoint(path: Path, sae: SAE, optimizer: torch.optim.Optimizer | None) -> int:
    payload = torch.load(path, map_location="cpu")
    sae.load_state_dict(payload["sae_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload.get("step", 0))


class Trainer:
    def __init__(self, sae: SAE, sae_cfg: SAEConfig, train_cfg: TrainConfig) -> None:
        self.sae = sae
        self.sae_cfg = sae_cfg
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")
        self.dtype = _torch_dtype(train_cfg.compute_dtype)
        self.sae.to(self.device, dtype=self.dtype)

        self.optimizer = Adam(self.sae.parameters(), lr=train_cfg.lr, betas=train_cfg.betas)
        self.out_dir = Path(train_cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"

        # Persist the configs for reproducibility.
        with open(self.out_dir / "config.json", "w") as f:
            json.dump({"sae": asdict(sae_cfg), "train": asdict(train_cfg)}, f, indent=2)

        self._pbar = None  # set during train(); used by _log to avoid clobbering the bar

    # ------------------------------------------------------------------
    def _log(self, record: dict[str, Any]) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        compact = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
        )
        line = f"[train] {compact}"
        if self._pbar is not None:
            # tqdm.write keeps the progress bar intact and renders cleanly above it.
            self._pbar.write(line)
        else:
            print(line)

    def _dead_fraction(self) -> float:
        s = self.sae.sparsity
        if hasattr(s, "step") and hasattr(s, "last_fired") and hasattr(s, "dead_steps_threshold"):
            dead = (s.step - s.last_fired) > s.dead_steps_threshold
            return float(dead.float().mean().item())
        return float("nan")

    # ------------------------------------------------------------------
    def train(self, batches: Iterable[torch.Tensor], start_step: int = 0) -> None:
        from tqdm.auto import tqdm

        torch.manual_seed(self.cfg.seed)
        self.sae.train()
        step = start_step
        t0 = time.time()

        total_steps = max(1, self.cfg.max_steps - start_step)
        self._pbar = (
            tqdm(total=total_steps, desc="[train]", unit="step",
                 initial=0, position=0, smoothing=0.05, dynamic_ncols=True)
            if self.cfg.progress
            else None
        )

        for raw in batches:
            x = raw.to(self.device, dtype=self.dtype, non_blocking=True) # (B, d_model)
            if step == start_step and start_step == 0 and self.cfg.init_b_dec_from_data:
                self.sae.init_b_dec_from_(x)

            x_hat, z, aux = self.sae(x)
            total, terms = self.sae.loss(x, x_hat, z, aux)

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.cfg.grad_clip)
            lr = _lr_at(step, self.cfg.lr, self.cfg.warmup_steps)
            _set_lr(self.optimizer, lr)
            self.optimizer.step()
            self.sae.normalize_decoder_()

            if step % self.cfg.log_every == 0:
                with torch.no_grad():
                    record = {
                        "step": step,
                        "lr": lr,
                        "loss": float(total.item()),
                        **{k: float(v.item()) for k, v in terms.items()},
                        "l0": float(self.sae.l0(z).item()),
                        "ev": float(self.sae.explained_variance(x, x_hat).item()),
                        "dead_frac": self._dead_fraction(),
                        "tok_per_s": float((step - start_step + 1) * x.shape[0] / max(time.time() - t0, 1e-6)),
                    }
                    self._log(record)
                    if self._pbar is not None:
                        self._pbar.set_postfix(
                            loss=f"{record['loss']:.4f}",
                            recon=f"{record['recon']:.4f}",
                            ev=f"{record['ev']:.3f}",
                            l0=f"{record['l0']:.1f}",
                            dead=f"{record['dead_frac']:.2f}",
                        )

            if step > 0 and step % self.cfg.ckpt_every == 0:
                save_checkpoint(self.out_dir / f"ckpt_step{step:07d}.pt", self.sae, self.optimizer, step, self.sae_cfg, self.cfg)

            step += 1
            if self._pbar is not None:
                self._pbar.update(1)
            if step >= self.cfg.max_steps:
                break

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        save_checkpoint(self.out_dir / "ckpt_final.pt", self.sae, self.optimizer, step, self.sae_cfg, self.cfg)
        print(f"[train] done. final step={step}  out={self.out_dir}")


# ----------------------------------------------------------------------
def build_dataset(train_cfg: TrainConfig, infinite: bool = True) -> ActivationDataset:
    return ActivationDataset(
        shard_dir=train_cfg.shard_dir,
        batch_size=train_cfg.batch_size,
        buffer_shards=train_cfg.buffer_shards,
        shuffle=True,
        seed=train_cfg.seed,
        infinite=infinite,
    )
