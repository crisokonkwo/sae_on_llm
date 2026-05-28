"""LoRA finetuning of a decoder-only LM on TOFU.

Custom training loop (matching the style of ``sae.train``) rather than
``transformers.Trainer``, so behaviour and progress reporting stay consistent
with the rest of the codebase.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Gemma attention/MLP linear layers. Same set works for Llama/Mistral.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    bias: str = "none"


@dataclass
class FinetuneConfig:
    model_name: str = "google/gemma-2-2b"
    output_dir: str = "runs/gemma2b_tofu_ft"
    dataset_config: str = "full"           # "full", "forget*", "retain*"
    dataset_split: str = "train"
    prompt_template: str = "Question: {question}\nAnswer:"
    max_length: int = 512
    mask_prompt: bool = True
    # optim
    batch_size: int = 4
    grad_accum_steps: int = 4               # Effective batch size is batch_size * grad_accum_steps. Adjust based on your GPU memory and how noisy the training is.
    lr: float = 2e-4
    warmup_steps: int = 50
    max_steps: int = 0                      # 0 -> train for `epochs` epochs
    epochs: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # logging / ckpt
    log_every: int = 25
    ckpt_every: int = 500
    seed: int = 0
    # device / dtype
    device: str = "cuda"
    compute_dtype: str = "bfloat16"
    progress: bool = True
    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    # base-model loading
    load_in_4bit: bool = False              # requires bitsandbytes
    gradient_checkpointing: bool = True


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_model_and_tokenizer(cfg: FinetuneConfig):
    """Load base LM + tokenizer and wrap with LoRA adapters via PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _torch_dtype(cfg.compute_dtype)
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": cfg.device}
    if cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs.pop("torch_dtype", None)
        except ImportError as e:
            raise ImportError("bitsandbytes is required for --load-in-4bit") from e

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Attach LoRA adapters.
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA finetuning. Install with: pip install peft"
        ) from e

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = cfg.lora
    peft_cfg = LoraConfig(
        r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules), bias=lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def _lr_at(step: int, base_lr: float, warmup: int) -> float:
    if warmup <= 0:
        return base_lr
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def save_lora_checkpoint(path: Path, model, step: int, cfg: FinetuneConfig) -> None:
    """Save PEFT adapter weights + manifest. Cheap and small (~30MB)."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    meta = {"step": step, "config": asdict(cfg)}
    with open(path / "ft_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


class TOFUTrainer:
    def __init__(self, model, tokenizer, cfg: FinetuneConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        with open(self.out_dir / "ft_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

        # Only LoRA params are trainable, so this optimiser is tiny.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
        )
        self._pbar = None

    # ------------------------------------------------------------------
    def _log(self, record: dict[str, Any]) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        line = "[ft] " + " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
        )
        if self._pbar is not None:
            self._pbar.write(line)
        else:
            print(line)

    # ------------------------------------------------------------------
    def train(self, dataloader: DataLoader) -> None:
        from tqdm.auto import tqdm

        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.model.train()

        if cfg.max_steps > 0:
            total_steps = cfg.max_steps
        else:
            steps_per_epoch = max(1, len(dataloader) // cfg.grad_accum_steps)
            total_steps = steps_per_epoch * cfg.epochs

        self._pbar = (
            tqdm(total=total_steps, desc="[ft]", unit="step", smoothing=0.05, dynamic_ncols=True)
            if cfg.progress else None
        )

        step = 0
        micro_step = 0
        running_loss = 0.0
        running_count = 0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        epoch = 0
        while step < total_steps:
            epoch += 1
            for batch in dataloader:
                if step >= total_steps:
                    break
                # Move batch to device, forward, backward.
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss / cfg.grad_accum_steps
                loss.backward()
                running_loss += float(out.loss.detach().item())
                running_count += 1

                micro_step += 1
                if micro_step % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            cfg.grad_clip,
                        )
                    lr = _lr_at(step, cfg.lr, cfg.warmup_steps)
                    _set_lr(self.optimizer, lr)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if self._pbar is not None:
                        self._pbar.update(1)

                    if step % cfg.log_every == 0:
                        avg = running_loss / max(running_count, 1)
                        record = {
                            "step": step,
                            "epoch": epoch,
                            "lr": lr,
                            "loss": avg,
                            "tok_per_s": float((step * cfg.batch_size * cfg.grad_accum_steps) / max(time.time() - t0, 1e-6)),
                        }
                        self._log(record)
                        if self._pbar is not None:
                            self._pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", epoch=epoch)
                        running_loss = 0.0
                        running_count = 0

                    if step > 0 and step % cfg.ckpt_every == 0:
                        save_lora_checkpoint(self.out_dir / f"ckpt_step{step:07d}", self.model, step, cfg)

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        save_lora_checkpoint(self.out_dir / "ckpt_final", self.model, step, cfg)
        print(f"[ft] done. final step={step} out={self.out_dir}")
