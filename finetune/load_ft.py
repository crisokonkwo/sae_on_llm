"""Reload a base LM + LoRA adapter checkpoint for evaluation or downstream
use (e.g. activation harvesting in M4).
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_ft_model(
    ckpt_dir: str | Path,
    base_model_name: str | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    merge_adapter: bool = False,
):
    """Load a PEFT adapter directory (the artifact saved by
    :func:`finetune.finetune.save_lora_checkpoint`).

    If ``merge_adapter=True`` the LoRA deltas are folded into the base weights,
    producing a plain HF model that's drop-in compatible with any pipeline
    that doesn't know about PEFT (handy for activation harvesting).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    if base_model_name is None:
        # Read PEFT config to discover the base model.
        from peft import PeftConfig
        peft_cfg = PeftConfig.from_pretrained(str(ckpt_dir))
        base_model_name = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, device_map=device,
    )
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
    if merge_adapter:
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer
