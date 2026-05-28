"""TOFU finetuning pipeline (M3).

Mirrors the style of the ``sae/`` package: small, modular, config-driven.
"""

from .tofu_data import (
    TOFU_DATASET_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    format_qa,
    load_tofu_split,
    build_tofu_train_dataset,
    TOFU_KNOWN_CONFIGS,
)

__all__ = [
    "TOFU_DATASET_NAME",
    "DEFAULT_PROMPT_TEMPLATE",
    "format_qa",
    "load_tofu_split",
    "build_tofu_train_dataset",
    "TOFU_KNOWN_CONFIGS",
]
