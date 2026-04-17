#!/usr/bin/env python3
"""
Collapse seed-aggregated W&B CSV to one best row per (model, dataset, ...), then
optionally save a publication-style figure comparing models across datasets.

Chooses the hyperparameter row with optimal validation mean for the dataset's
``dataset.parameters.monitor_metric`` (higher-is-better vs lower-is-better from
``MONITOR_METRIC_OPTIMIZATION`` in ``utils``), using ``utils.collapse_aggregated_wandb_csv`` /
``iter_best_val_group_picks`` — the **same** best-val rule as ``table_generator`` and the
winner-row selection in ``best_rerun_sh_generator``. The output includes paired
train/val/test **mean** and **std** columns from the aggregated CSV.

Seed aggregation and reruns assume the export includes all swept config columns listed in
``utils.CONFIG_PARAM_KEYS`` (e.g. HOPSE_G ``transforms.hopse_encoding.pretrain_model``,
SANN ``transforms.sann_encoding.*``); see ``main_loader`` / ``aggregator`` docs.

The figure uses one column per dataset (max 4 per row), bars = models, error bars
from the **test** split by default (mean ± std); configs are still **chosen** using
validation. Override with ``--split``. y-axis shows the monitored metric with an arrow
for optimization direction.

Default CSV paths live under ``scripts/hopse_plotting/csvs/``; default figure path is
``plots/leaderboard/<collapsed_stem>_leaderboard.png``.

Usage::

    python scripts/hopse_plotting/main_plot.py
    python scripts/hopse_plotting/main_plot.py -i scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_seed_agg.csv \\
        -o scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_collapsed.csv
    python scripts/hopse_plotting/main_plot.py --no-plot
    python scripts/hopse_plotting/main_plot.py --split val --plot-output plots/leaderboard/fig.png
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from utils import (
    DEFAULT_AGGREGATED_EXPORT_CSV,
    DEFAULT_COLLAPSED_EXPORT_CSV,
    DEFAULT_LEADERBOARD_PLOT_DIR,
    MONITOR_METRIC_COLUMN,
    collapse_aggregated_wandb_csv,
    metric_name_tail,
    optimization_mode_for_metric_tail,
    safe_metric_col_token,
)

def _label_short(s: str, max_len: int = 28) -> str:
    t = str(s).strip()
    if "/" in t:
        t = t.rsplit("/", 1)[-1]
    return t if len(t) <= max_len else t[: max_len - 1] + "..."


def _dataset_title(dataset_path: str) -> str:
    return _label_short(dataset_path, max_len=40)


def _legend_labels_for_models(models: list[str]) -> dict[str, str]:
    """
    One legend entry per raw model id. If several models share the same short
    basename (e.g. simplicial/... and cell/...), disambiguate with the domain prefix.
    """
    uniq = sorted(set(str(m) for m in models))
    short_for = {m: _label_short(m) for m in uniq}
    short_count = Counter(short_for.values())
    seen: set[str] = set()
    out: dict[str, str] = {}
    for m in uniq:
        sh = short_for[m]
        if short_count[sh] > 1 and "/" in m:
            domain = m.split("/", 1)[0]
            label = f"{sh} ({domain})"
        else:
            label = sh
        if label in seen:
            label = m
        seen.add(label)
        out[m] = label
    return out


def _model_basename(m: str) -> str:
    return str(m).strip().rsplit("/", 1)[-1].lower()


def _model_domain(m: str) -> str:
    """Hydra-style prefix: ``cell/...``, ``simplicial/...``, ``graph/...``."""
    s = str(m).strip()
    if "/" not in s:
        return ""
    return s.split("/", 1)[0].strip().lower()


def model_category(m: str) -> str:
    """Coarse family for color + strip ordering (MPNN, TopoTune, SANN, SCCNN, CWN, HOPSE)."""
    s = str(m).lower()
    b = _model_basename(m)
    if "hopse_g" in s or b == "hopse_g":
        return "hopse_g"
    if "hopse_m" in s or b == "hopse_m":
        return "hopse_m"
    if "topotune" in s or b == "topotune":
        return "topotune"
    if b.startswith("sann"):
        return "sann"
    if b.startswith("sccnn"):
        return "sccnn"
    if b == "cwn":
        return "cwn"
    if b in ("gcn", "gin", "gat"):
        return "mpnn"
    return "other"


# Panel / legend order: MPNN, TopoTune, SANN, SCCNN, CWN, HOPSE-M, HOPSE-G, then other.
_CATEGORY_ORDER = {
    "mpnn": 0,
    "topotune": 1,
    "sann": 2,
    "sccnn": 3,
    "cwn": 4,
    "hopse_m": 5,
    "hopse_g": 6,
    "other": 7,
}
_MPNN_ORDER = {"gcn": 0, "gin": 1, "gat": 2}
# cell before simplicial within TopoTune / SANN / each HOPSE line
_DOMAIN_SORT = {"cell": 0, "simplicial": 1, "graph": 2}

# MPNN: yellow -> orange -> red
_MPNN_HEX = {"gcn": "#E6C229", "gin": "#F28C18", "gat": "#C81D25"}
# TopoTune: lighter green (cell) vs deeper green (simplicial)
_TOPOTUNE_CELL_HEX = "#8FD98A"
_TOPOTUNE_SIM_HEX = "#1E7A3E"
# HOPSE blues, light -> dark: M-cell, M-sim, G-cell, G-sim
_HOPSE_M_CELL_HEX = "#C8E6F5"
_HOPSE_M_SIM_HEX = "#7DB6E8"
_HOPSE_G_CELL_HEX = "#3D78B8"
_HOPSE_G_SIM_HEX = "#0C335C"
# SANN: warm amber (cell lighter, simplicial deeper) — distinct from TopoTune greens
_SANN_CELL_HEX = "#F39C12"
_SANN_SIM_HEX = "#B9770E"
_SCCNN_HEX = "#8E44AD"
_CWN_HEX = "#1ABC9C"
_OTHER_HEX = "#6E6E6E"


def _domain_sort_key(m: str) -> int:
    d = _model_domain(m)
    return _DOMAIN_SORT.get(d, 9)


def model_display_sort_key(m: str) -> tuple:
    cat = model_category(m)
    b = _model_basename(m)
    cat_i = _CATEGORY_ORDER[cat]
    if cat == "mpnn":
        return (cat_i, _MPNN_ORDER.get(b, 9), m.lower())
    if cat in ("topotune", "hopse_m", "hopse_g", "sann", "sccnn", "cwn"):
        return (cat_i, _domain_sort_key(m), m.lower())
    return (cat_i, b, m.lower())


def models_sorted_for_display(models) -> list[str]:
    return sorted({str(x) for x in models}, key=model_display_sort_key)


def color_for_model(m: str) -> tuple:
    cat = model_category(m)
    b = _model_basename(m)
    dom = _model_domain(m)
    if cat == "mpnn":
        h = _MPNN_HEX.get(b, _OTHER_HEX)
    elif cat == "topotune":
        if dom == "cell":
            h = _TOPOTUNE_CELL_HEX
        elif dom == "simplicial":
            h = _TOPOTUNE_SIM_HEX
        else:
            h = _TOPOTUNE_SIM_HEX
    elif cat == "hopse_m":
        if dom == "cell":
            h = _HOPSE_M_CELL_HEX
        elif dom == "simplicial":
            h = _HOPSE_M_SIM_HEX
        else:
            h = _HOPSE_M_SIM_HEX
    elif cat == "hopse_g":
        if dom == "cell":
            h = _HOPSE_G_CELL_HEX
        elif dom == "simplicial":
            h = _HOPSE_G_SIM_HEX
        else:
            h = _HOPSE_G_SIM_HEX
    elif cat == "sann":
        if dom == "cell":
            h = _SANN_CELL_HEX
        elif dom == "simplicial":
            h = _SANN_SIM_HEX
        else:
            h = _SANN_SIM_HEX
    elif cat == "sccnn":
        h = _SCCNN_HEX
    elif cat == "cwn":
        h = _CWN_HEX
    else:
        h = _OTHER_HEX
    return matplotlib.colors.to_rgb(h)


def metric_axis_label(monitor_raw: str) -> str:
    """Y-axis text: metric name + optimize direction arrow (matplotlib mathtext)."""
    tail = metric_name_tail(monitor_raw)
    if not tail:
        tail = "metric"
    mode = optimization_mode_for_metric_tail(tail)
    arrow = r"$\uparrow$" if mode == "max" else r"$\downarrow$"
    return f"{arrow} {tail}"


def _mean_std_columns_for_row(
    monitor_raw: str, split: Literal["train", "val", "test"]
) -> tuple[str, str]:
    tail = metric_name_tail(monitor_raw)
    tok = safe_metric_col_token(tail) if tail else "unknown"
    return f"{split}_{tok}_mean", f"{split}_{tok}_std"


# Touching grouped bars: unit spacing, full width 1.0 (edges meet between neighbors).
_BAR_TOUCHING_WIDTH = 1.0


def _set_ylim_from_values_with_errors(
    ax,
    means: list[float],
    stds: list[float],
    *,
    pad_frac: float = 0.06,
) -> None:
    """Tight y-axis from min(mean - std) to max(mean + std) with small padding."""
    m = np.asarray(means, dtype=float)
    s = np.asarray(stds, dtype=float)
    ok = np.isfinite(m)
    if not ok.any():
        return
    s = np.nan_to_num(s, nan=0.0)
    lo = float(np.min(m[ok] - s[ok]))
    hi = float(np.max(m[ok] + s[ok]))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    span = hi - lo
    pad = pad_frac * span if span > 1e-12 else max(abs(hi), 1e-9) * pad_frac
    ax.set_ylim(lo - pad, hi + pad)


def plot_collapsed_model_leaderboard(
    collapsed_df: pd.DataFrame,
    *,
    path: Path,
    monitor_column: str = MONITOR_METRIC_COLUMN,
    split: Literal["train", "val", "test"] = "test",
    max_cols: int = 4,
    dpi: int = 300,
) -> None:
    """
    Bar plot: facet columns = datasets (max ``max_cols`` per row), bars = models,
    heights = mean, error bars = std for ``split`` (train/val/test). Bars use fixed
    unit width and touch; no x ticks (identify models by legend). Y-limits are
    data-driven per panel.
    """
    if collapsed_df.empty:
        raise ValueError("collapsed_df is empty; nothing to plot.")
    if max_cols < 1:
        raise ValueError("max_cols must be >= 1.")

    df = collapsed_df.copy()
    if monitor_column not in df.columns:
        raise KeyError(f"missing {monitor_column!r} in collapsed dataframe")
    if "model" not in df.columns or "dataset" not in df.columns:
        raise KeyError("collapsed dataframe must contain 'model' and 'dataset' columns")

    models_all = models_sorted_for_display(df["model"].astype(str).unique())
    color_by_model = {m: color_for_model(m) for m in models_all}
    legend_label_by_model = _legend_labels_for_models(models_all)

    datasets = sorted(df["dataset"].astype(str).unique())
    n_ds = len(datasets)
    n_rows = math.ceil(n_ds / max_cols) if n_ds else 1
    n_cols_fig = min(max_cols, n_ds) if n_ds else 1

    # Publication-friendly rc
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
        }
    )

    # Wide enough for a single-row model legend when there are many models
    fig_w = min(22.0, max(6.0, n_cols_fig * 3.05, 5.0 + 0.44 * len(models_all)))
    fig_h = max(2.95, n_rows * 3.22)
    fig, axes = plt.subplots(
        n_rows,
        n_cols_fig,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    panel_idx = 0
    for row in range(n_rows):
        for col in range(n_cols_fig):
            ax = axes[row][col]
            if panel_idx >= n_ds:
                ax.set_visible(False)
                continue

            ds = datasets[panel_idx]
            sub = df[df["dataset"].astype(str) == ds].copy()
            monitor = str(sub[monitor_column].iloc[0]).strip()
            mean_c, std_c = _mean_std_columns_for_row(monitor, split)

            if mean_c not in sub.columns:
                ax.text(0.5, 0.5, f"missing column\n{mean_c}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(_dataset_title(ds))
                panel_idx += 1
                continue

            models_here = models_sorted_for_display(sub["model"].astype(str).unique())
            x = np.arange(len(models_here))
            means: list[float] = []
            stds: list[float] = []
            colors: list[tuple] = []
            for m in models_here:
                row_m = sub[sub["model"].astype(str) == m].iloc[0]
                mu = pd.to_numeric(row_m.get(mean_c, np.nan), errors="coerce")
                sg = pd.to_numeric(row_m.get(std_c, np.nan), errors="coerce") if std_c in sub.columns else np.nan
                means.append(float(mu) if pd.notna(mu) else np.nan)
                stds.append(float(sg) if pd.notna(sg) else 0.0)
                colors.append(color_by_model.get(m, (0.4, 0.4, 0.4)))

            n_b = len(models_here)
            ax.bar(
                x,
                means,
                width=_BAR_TOUCHING_WIDTH,
                align="center",
                yerr=stds,
                color=colors,
                edgecolor="0.12",
                linewidth=0.55,
                capsize=2.2,
                error_kw={"elinewidth": 0.85, "capthick": 0.85, "color": "0.22"},
            )
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            if n_b > 0:
                ax.set_xlim(-0.5, (n_b - 1) + 0.5)
            ax.set_title(_dataset_title(ds), fontweight="semibold", pad=6)
            ax.set_ylabel(metric_axis_label(monitor))
            _set_ylim_from_values_with_errors(ax, means, stds)
            ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.85)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            panel_idx += 1

    handles = [
        Patch(
            facecolor=color_by_model[m],
            edgecolor="0.15",
            linewidth=0.6,
            label=legend_label_by_model[m],
        )
        for m in models_all
    ]
    ncol = max(1, len(handles))

    # Header: title pulled down, legend pulled up (nearer title); axes top below legend box
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.08,
        top=0.795,
        wspace=0.30,
        hspace=0.72,
    )

    fig.suptitle(
        f"Best config per model (selected on val); bars: {split} mean +/- std across seeds",
        fontsize=12,
        fontweight="bold",
        y=0.918,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.888),
        ncol=ncol,
        frameon=False,
        title="Model",
        fontsize=11,
        title_fontsize=11,
        borderaxespad=0.15,
        labelspacing=0.4,
        handletextpad=0.65,
        columnspacing=1.75,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collapse seed-aggregated W&B CSV and plot model comparison across datasets."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_AGGREGATED_EXPORT_CSV,
        help=f"Seed-aggregated CSV (default: {DEFAULT_AGGREGATED_EXPORT_CSV})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_COLLAPSED_EXPORT_CSV,
        help=f"Collapsed CSV (default: {DEFAULT_COLLAPSED_EXPORT_CSV})",
    )
    p.add_argument(
        "--group-by",
        metavar="COL",
        nargs="+",
        default=["model", "dataset"],
        help="Group columns (default: model dataset)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write collapsed CSV, skip figure.",
    )
    p.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            f"Figure path (.png / .pdf). Default: {DEFAULT_LEADERBOARD_PLOT_DIR}/"
            "<collapsed_stem>_leaderboard.png"
        ),
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Which split's mean±std to plot (default: test; selection still uses val)",
    )
    p.add_argument(
        "--max-cols",
        type=int,
        default=4,
        metavar="N",
        help="Max dataset panels per row (default: 4)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300)",
    )
    args = p.parse_args()

    collapsed = collapse_aggregated_wandb_csv(
        args.input,
        args.output,
        group_cols=list(args.group_by),
    )
    print(f"Wrote {len(collapsed)} rows x {len(collapsed.columns)} columns -> {args.output}")

    if args.no_plot:
        return

    plot_path = args.plot_output
    if plot_path is None:
        plot_path = DEFAULT_LEADERBOARD_PLOT_DIR / f"{args.output.stem}_leaderboard.png"

    plot_collapsed_model_leaderboard(
        collapsed,
        path=plot_path,
        split=args.split,
        max_cols=args.max_cols,
        dpi=args.dpi,
    )
    print(f"Wrote figure -> {plot_path}")


if __name__ == "__main__":
    main()
