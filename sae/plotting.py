"""Plotting utilities for SAE training curves and eval comparisons.

Two consumer types:

* **Training metrics** \u2014 each run dir has a ``metrics.jsonl`` (one JSON
  object per logging step) produced by :class:`sae.train.Trainer`.
* **Evaluation reports** \u2014 each eval dir has an ``eval_report.json``
  produced by ``scripts/eval_sae.py``.

The functions below take dicts keyed by *split label* (e.g. ``"train"``,
``"val"``, ``"test"``) so the same code can plot one run, compare two, or
overlay arbitrarily many.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_metrics_jsonl(path: str | Path) -> dict[str, list[float]]:
    """Read a ``metrics.jsonl`` and return a column-oriented dict."""
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys()})
    cols: dict[str, list[float]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            cols[k].append(float(v) if isinstance(v, (int, float)) else float("nan"))
    return cols


def load_eval_report(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------
_DEFAULT_TRAIN_PANELS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("recon",     {"title": "Reconstruction loss", "yscale": "log"}),
    ("total",     {"title": "Total loss",          "yscale": "log"}),
    ("ev",        {"title": "Explained variance"}),
    ("l0",        {"title": "L0 (active features / token)"}),
    ("dead_frac", {"title": "Dead-feature fraction"}),
    ("lr",        {"title": "Learning rate",       "yscale": "log"}),
)


def plot_training_curves(
    runs: Mapping[str, str | Path],
    output: str | Path,
    panels: Sequence[tuple[str, dict[str, Any]]] = _DEFAULT_TRAIN_PANELS,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Plot training metrics for one or more runs on a shared step axis.

    ``runs``: ``{label: path-to-metrics.jsonl}``.
    """
    data = {label: load_metrics_jsonl(p) for label, p in runs.items()}
    n = len(panels)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    figsize = figsize or (4.5 * cols, 3.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, (key, opts) in zip(axes, panels):
        any_plotted = False
        for label, cols_data in data.items():
            if key not in cols_data or "step" not in cols_data:
                continue
            steps = cols_data["step"]
            ys = cols_data[key]
            ax.plot(steps, ys, label=label, linewidth=1.4)
            any_plotted = True
        ax.set_title(opts.get("title", key))
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        if "yscale" in opts and any_plotted:
            try:
                ax.set_yscale(opts["yscale"])
            except ValueError:
                pass
        if any_plotted and len(data) > 1:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide any unused axes.
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Eval comparisons (bar charts)
# ----------------------------------------------------------------------
_RECON_BAR_KEYS: tuple[str, ...] = (
    "explained_variance", "nmse", "l0_mean", "dead_fraction"
)
_CE_BAR_KEYS: tuple[str, ...] = (
    "ce_clean", "ce_sae", "ce_mean_ablation", "ce_recovered"
)


def _grouped_bar(
    ax,
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    title: str,
) -> None:
    n_groups = len(groups)
    n_series = len(series)
    if n_groups == 0 or n_series == 0:
        ax.set_visible(False)
        return
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for i, (label, vals) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if n_series > 1:
        ax.legend(fontsize=8)


def plot_eval_comparison(
    reports: Mapping[str, str | Path | dict],
    output: str | Path,
) -> Path:
    """Bar-chart comparison across splits (train/val/test).

    ``reports``: ``{split_label: path-to-eval_report.json | already-loaded dict}``.
    Plots reconstruction metrics, CE-delta, log-density histogram, and a
    bar of dead-feature counts.
    """
    loaded: dict[str, dict] = {}
    for label, ref in reports.items():
        loaded[label] = ref if isinstance(ref, dict) else load_eval_report(ref)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_recon, ax_ce, ax_density, ax_dead = axes.flatten()

    # 1. Reconstruction bars
    recon_series: dict[str, list[float]] = {}
    for label, rep in loaded.items():
        r = rep.get("reconstruction", {})
        recon_series[label] = [float(r.get(k, np.nan)) for k in _RECON_BAR_KEYS]
    _grouped_bar(ax_recon, list(_RECON_BAR_KEYS), recon_series, "Reconstruction metrics")

    # 2. CE-delta bars
    ce_series: dict[str, list[float]] = {}
    for label, rep in loaded.items():
        c = rep.get("ce_delta", {})
        if not c:
            continue
        ce_series[label] = [float(c.get(k, np.nan)) for k in _CE_BAR_KEYS]
    if ce_series:
        _grouped_bar(ax_ce, list(_CE_BAR_KEYS), ce_series, "CE-delta")
    else:
        ax_ce.set_visible(False)

    # 3. Log-density histogram (one line per split)
    any_density = False
    for label, rep in loaded.items():
        r = rep.get("reconstruction", {})
        edges = r.get("log10_density_bin_edges")
        hist = r.get("log10_density_hist")
        if not edges or not hist:
            continue
        edges = np.asarray(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # normalise to fraction of features so multiple n_features compare
        total = max(sum(hist), 1)
        ax_density.plot(centers, np.asarray(hist) / total, marker="o", label=label)
        any_density = True
    if any_density:
        ax_density.set_xlabel("log10(fire rate)")
        ax_density.set_ylabel("fraction of features")
        ax_density.set_title("Feature-density histogram")
        ax_density.grid(alpha=0.3)
        if len(loaded) > 1:
            ax_density.legend(fontsize=8)
    else:
        ax_density.set_visible(False)

    # 4. Dead-feature count bar
    dead_labels: list[str] = []
    dead_vals: list[float] = []
    n_feats: list[float] = []
    for label, rep in loaded.items():
        r = rep.get("reconstruction", {})
        if "n_dead" in r and "n_features" in r:
            dead_labels.append(label)
            dead_vals.append(float(r["n_dead"]))
            n_feats.append(float(r["n_features"]))
    if dead_labels:
        x = np.arange(len(dead_labels))
        ax_dead.bar(x, dead_vals, color="tab:red", alpha=0.7, label="dead")
        ax_dead.bar(x, [nf - d for nf, d in zip(n_feats, dead_vals)],
                    bottom=dead_vals, color="tab:green", alpha=0.6, label="alive")
        ax_dead.set_xticks(x)
        ax_dead.set_xticklabels(dead_labels)
        ax_dead.set_title("Dead vs alive features")
        ax_dead.legend(fontsize=8)
        ax_dead.grid(axis="y", alpha=0.3)
    else:
        ax_dead.set_visible(False)

    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Convenience: one-shot "everything" plot
# ----------------------------------------------------------------------
def plot_all(
    runs: Mapping[str, str | Path] | None,
    eval_reports: Mapping[str, str | Path | dict] | None,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Produce ``training_curves.png`` and ``eval_comparison.png`` if data is
    available. Returns paths of the figures actually written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if runs:
        written["training_curves"] = plot_training_curves(runs, out_dir / "training_curves.png")
    if eval_reports:
        written["eval_comparison"] = plot_eval_comparison(eval_reports, out_dir / "eval_comparison.png")
    return written
