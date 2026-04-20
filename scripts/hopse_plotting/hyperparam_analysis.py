#!/usr/bin/env python3
"""
Hyperparameter sensitivity from a **seed-aggregated** W&B export (``aggregator``
output): rows are grouped by **(model, dataset)** so the monitored validation metric
is consistent within each panel. For each group, every config column that still varies
is plotted against validation performance.

Figures are written under ``plots/hyperparam/<model>/<dataset>/`` by default.

Validation **y** uses ``dataset.parameters.monitor_metric`` and the same
``summary_*__mean`` column resolution as ``collapse_aggregated_wandb_by_best_val``.
For metrics where lower is better (see ``MONITOR_METRIC_OPTIMIZATION`` in utils), the
y-axis label adds a bold **lower is better** line.

- Mostly numeric columns with many distinct values → scatter plot.
- Otherwise → **violin plot** per category with **jittered dots** (one point per
  aggregated row / seed-mean run) on top.

Does not modify ``main_loader``. Any model id present in the CSV (including
``simplicial/sccnn_custom``, ``cell/cwn``, ``cell/sann``, …) gets a
``(model, dataset)`` folder automatically.

Usage::

    python scripts/hopse_plotting/aggregator.py
    python scripts/hopse_plotting/hyperparam_analysis.py
    python scripts/hopse_plotting/hyperparam_analysis.py -i scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_seed_agg.csv
    python scripts/hopse_plotting/hyperparam_analysis.py --from-raw -i scripts/hopse_plotting/csvs/hopse_experiments_wandb_export.csv -o plots/out
    python scripts/hopse_plotting/hyperparam_analysis.py --models cell/hopse_m simplicial/hopse_m
    python scripts/hopse_plotting/hyperparam_analysis.py --models simplicial/sccnn_custom cell/cwn
    python scripts/hopse_plotting/hyperparam_analysis.py --datasets graph/MUTAG
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import (
    DEFAULT_AGGREGATED_EXPORT_CSV,
    DEFAULT_HYPERPARAM_PLOT_DIR,
    DEFAULT_WANDB_EXPORT_CSV,
    MONITOR_METRIC_COLUMN,
    aggregate_wandb_export_by_seed,
    hyperparam_axis_columns,
    infer_hyperparam_plot_kind,
    load_wandb_export_csv,
    metric_name_tail,
    optimization_mode_for_metric_tail,
    safe_filename_token,
    val_metric_mean_per_row,
    varied_hyperparam_columns,
)


def _candidates_for_model_dataset_groups(df: pd.DataFrame) -> list[str]:
    """Hyperparam columns to scan; ``dataset`` is fixed per group and omitted."""
    return [c for c in hyperparam_axis_columns(df) if c != "dataset"]


def _stable_rng_seed(*parts: str) -> int:
    h = zlib.adler32(b"\0".join(p.encode("utf-8", errors="replace") for p in parts)) & 0xFFFFFFFF
    return int(h % (2**31 - 1)) or 1


def _ylabel_validation_monitor(sub_all: pd.DataFrame) -> str:
    """
    Y-axis label for validation (seed-mean). When the slice's monitor metric is
    minimized (``MONITOR_METRIC_OPTIMIZATION``), append a **bold** mathtext line.
    """
    base = "Validation metric (seed-mean)\n(row monitor)"
    if MONITOR_METRIC_COLUMN not in sub_all.columns:
        return base
    mon = (
        sub_all[MONITOR_METRIC_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
    )
    mon = mon[(mon != "") & ~mon.str.lower().isin({"nan", "none"})]
    if mon.empty:
        return base
    tail = metric_name_tail(mon.iloc[0])
    if not tail:
        return base
    if optimization_mode_for_metric_tail(tail) == "min":
        return base + "\n" + r"$\bf{lower\ is\ better}$"
    return base


def _prepare_frame(
    input_path: Path,
    *,
    from_raw: bool,
) -> pd.DataFrame:
    raw = load_wandb_export_csv(input_path)
    if from_raw:
        agg, _silent = aggregate_wandb_export_by_seed(raw)
        return agg
    return raw


def _plot_one_hyperparam(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    *,
    col_name: str,
) -> None:
    y_num = pd.to_numeric(y, errors="coerce")
    mask_y = y_num.notna().to_numpy()
    x = x[mask_y]
    y = y_num[mask_y]

    if len(y) == 0:
        ax.text(0.5, 0.5, "no finite y", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    kind, xv = infer_hyperparam_plot_kind(x)

    if kind == "skip":
        ax.text(
            0.5,
            0.5,
            f"skipped (> categories)\n{col_name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()
        return

    if kind == "scatter":
        xv_num = pd.to_numeric(xv, errors="coerce")
        ok = xv_num.notna().to_numpy() & y.notna().to_numpy()
        xv = xv_num[ok]
        y = y[ok]
        ax.scatter(
            xv,
            y,
            s=26,
            alpha=0.72,
            edgecolors="0.2",
            linewidths=0.45,
            color="#2E6F9E",
        )
        ax.set_xlabel(col_name, fontsize=8)
        return

    # Violin + jittered seed-mean points per category
    sub = pd.DataFrame({"x": xv.astype(str), "y": pd.to_numeric(y, errors="coerce")})
    sub = sub[np.isfinite(sub["y"])]
    if sub.empty:
        ax.text(0.5, 0.5, "no finite y", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    medians = sub.groupby("x", dropna=False)["y"].median()
    order = medians.sort_values(ascending=False).index.tolist()
    xpos = np.arange(len(order), dtype=float)
    data_arrays = [sub.loc[sub["x"] == cat, "y"].to_numpy(dtype=float) for cat in order]

    # Violins (matplotlib KDE; single-point categories still get a narrow shape)
    parts = ax.violinplot(
        data_arrays,
        positions=xpos,
        widths=min(0.82, 0.14 * max(len(order), 1)),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for b in parts["bodies"]:
        b.set_facecolor("#4A90A4")
        b.set_alpha(0.38)
        b.set_edgecolor("0.22")
        b.set_linewidth(0.75)
    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_colors("0.15")
        parts["cmedians"].set_linewidths(1.0)

    rng = np.random.default_rng(_stable_rng_seed(col_name, *order[:8]))

    for i, cat in enumerate(order):
        ys = sub.loc[sub["x"] == cat, "y"].to_numpy(dtype=float)
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=ys.size)
        ax.scatter(
            xpos[i] + jitter,
            ys,
            s=20,
            alpha=0.88,
            c="0.12",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(l) for l in order], rotation=42, ha="right", fontsize=7)
    ax.set_xlim(xpos.min() - 0.65, xpos.max() + 0.65)
    ax.set_xlabel(col_name, fontsize=8)


def run_hyperparam_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    dpi: int = 200,
) -> None:
    if "model" not in df.columns:
        raise KeyError("expected column 'model' (seed-aggregated export)")
    if "dataset" not in df.columns:
        raise KeyError("expected column 'dataset' for (model, dataset) grouping")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    y_all = val_metric_mean_per_row(df)
    candidates = _candidates_for_model_dataset_groups(df)

    keys = df[["model", "dataset"]].astype(str).drop_duplicates()
    combos = [(r["model"], r["dataset"]) for _, r in keys.iterrows()]
    combos.sort(key=lambda t: (t[0], t[1]))

    if models:
        want_m = {str(m) for m in models}
        combos = [(m, d) for m, d in combos if m in want_m]
        missing_m = want_m - {m for m, _ in combos}
        if missing_m:
            print(f"  (warn) --models not found in CSV: {sorted(missing_m)}")

    if datasets:
        want_d = {str(x) for x in datasets}
        combos = [(m, d) for m, d in combos if d in want_d]
        missing_d = want_d - {d for _, d in combos}
        if missing_d:
            print(f"  (warn) --datasets not found in CSV: {sorted(missing_d)}")

    for m, d in combos:
        sub_all = df[(df["model"].astype(str) == m) & (df["dataset"].astype(str) == d)]
        varied = varied_hyperparam_columns(sub_all, candidate_cols=candidates)
        if not varied:
            print(f"  (skip) no varied hyperparam columns for ({m!r}, {d!r})")
            continue

        y = y_all.loc[sub_all.index]
        ylabel_str = _ylabel_validation_monitor(sub_all)

        safe_m = safe_filename_token(m.replace("/", "__"))
        safe_d = safe_filename_token(str(d).replace("/", "__"))
        combo_dir = output_dir / safe_m / safe_d
        combo_dir.mkdir(parents=True, exist_ok=True)

        for col in varied:
            _kind_w, _xv_w = infer_hyperparam_plot_kind(sub_all[col])
            if _kind_w == "scatter":
                fig_w = 5.2
            elif _kind_w == "skip":
                fig_w = 5.2
            else:
                n_lab = int(_xv_w.astype(str).nunique(dropna=False))
                fig_w = min(14.0, max(5.2, 0.42 * float(max(n_lab, 1)) + 3.0))
            fig, ax = plt.subplots(figsize=(fig_w, 3.6))
            _plot_one_hyperparam(ax, sub_all[col], y, col_name=col)
            ax.set_ylabel(ylabel_str, fontsize=8)
            ax.set_title(f"{m}\n{d}\n{col}", fontsize=9, fontweight="semibold")
            ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.85)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fn = f"{safe_filename_token(col, max_len=96)}.png"
            fig.savefig(combo_dir / fn, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)

        print(f"  Wrote {len(varied)} figure(s) -> {combo_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot val metric vs varied hyperparams per (model, dataset) (seed-aggregated W&B CSV)."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="CSV path (default: seed-aggregated export, or raw export with --from-raw)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_HYPERPARAM_PLOT_DIR,
        help=(
            "Directory root; each (model, dataset) gets a subfolder model/dataset/ "
            f"(default: {DEFAULT_HYPERPARAM_PLOT_DIR})"
        ),
    )
    p.add_argument(
        "--from-raw",
        action="store_true",
        help="Treat --input as per-run loader export; aggregate over seeds in memory (no CSV write).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Only these model ids (exact strings as in CSV 'model' column).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Only these dataset paths (exact strings as in CSV 'dataset' column).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI (default: 200)",
    )
    args = p.parse_args()

    inp = args.input
    if inp is None:
        inp = DEFAULT_WANDB_EXPORT_CSV if args.from_raw else DEFAULT_AGGREGATED_EXPORT_CSV

    df = _prepare_frame(inp, from_raw=args.from_raw)
    print(f"Loaded {'raw→aggregated' if args.from_raw else 'seed-aggregated'} table: {len(df)} rows x {len(df.columns)} cols")
    run_hyperparam_analysis(
        df,
        args.output_dir,
        models=args.models,
        datasets=args.datasets,
        dpi=args.dpi,
    )
    print(f"Done. Plots under {args.output_dir}")


if __name__ == "__main__":
    main()
