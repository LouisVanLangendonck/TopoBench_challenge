#!/usr/bin/env python3
"""
Per-model bar plots of **n_seeds** (raw-run count per hyperparameter group) vs
**n_groups** (how many groups have that count), for every dataset in the report.

Uses the same pipeline as ``aggregator.py``: drop raw runs with no finite
``summary_test_best_rerun/*`` values (silent failures), aggregate by seed, then
``build_seed_bucket_report``. Seed-count plots add a **silent fail** bar when
that count is nonzero for the subplot's (model, dataset).

Standalone (same input discovery as ``aggregator.py``)::

    python scripts/hopse_plotting/seed_n_distribution_plots.py
    python scripts/hopse_plotting/seed_n_distribution_plots.py -i shards/a.csv shards/b.csv
    python scripts/hopse_plotting/seed_n_distribution_plots.py --output-dir plots/custom

Or enable ``--plot-seed-distributions`` when running ``aggregator.py`` (writes here by default).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils import (
    DEFAULT_WANDB_EXPORT_CSV,
    DEFAULT_WANDB_EXPORT_SHARD_DIR,
    PLOTS_DIR,
    aggregate_wandb_export_by_seed,
    build_seed_bucket_report,
    load_wandb_export_csv,
    safe_filename_token,
    _union_column_order,
)

DEFAULT_SEED_N_DIST_DIR = PLOTS_DIR / "seed_n_distributions"


def _collect_input_paths(
    *,
    explicit: list[Path] | None,
    input_dir: Path | None,
    input_pattern: str,
) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        paths.extend(explicit)
    if input_dir is not None:
        d = Path(input_dir)
        if d.is_dir():
            paths.extend(sorted(d.glob(input_pattern)))
    if not paths:
        paths = [DEFAULT_WANDB_EXPORT_CSV]
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def seed_bucket_report_from_export_paths(
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match ``aggregator.py`` (without ``--required-seeds`` filtering): aggregate by
    seed, ``build_seed_bucket_report``, and return
    ``(report, silent_failure_counts)``.
    """
    if not paths:
        raise ValueError("no input paths")
    if len(paths) == 1:
        df = load_wandb_export_csv(paths[0])
        agg, silent = aggregate_wandb_export_by_seed(df)
        return build_seed_bucket_report(agg), silent

    frames: list[pd.DataFrame] = []
    silent_parts: list[pd.DataFrame] = []
    for p in paths:
        df = load_wandb_export_csv(p)
        agg_i, silent_i = aggregate_wandb_export_by_seed(df)
        frames.append(agg_i)
        silent_parts.append(silent_i)
    cols = _union_column_order(frames)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.reindex(columns=cols)
    silent_concat = pd.concat(silent_parts, ignore_index=True, sort=False)
    if silent_concat.empty:
        silent = pd.DataFrame(columns=["model", "dataset", "n_silent_failures"])
    else:
        silent = (
            silent_concat.groupby(["model", "dataset"], dropna=False)["n_silent_failures"]
            .sum()
            .reset_index()
        )
    return build_seed_bucket_report(out), silent


def write_seed_distribution_plots(
    report: pd.DataFrame,
    out_dir: Path,
    *,
    required_n_seeds: int | None = None,
    silent_failures: pd.DataFrame | None = None,
    dpi: int = 150,
) -> int:
    """
    Write one PNG per distinct ``model`` in ``report``.

    Each figure: grid of bar charts (x = ``n_seeds``, height = ``n_groups``) for
    every ``dataset`` that model appears in. Bars matching ``required_n_seeds``
    (if not None) are highlighted.

    If ``silent_failures`` is given (columns ``model``, ``dataset``,
    ``n_silent_failures``), an extra **silent fail** bar is drawn per subplot when
    the count is positive (raw runs dropped before aggregation: no finite
    ``summary_test_best_rerun/*`` values).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if report.empty or "model" not in report.columns:
        return 0

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    n_written = 0
    for model in sorted(report["model"].astype(str).unique()):
        sub_m = report[report["model"].astype(str) == model]
        datasets = sorted(sub_m["dataset"].astype(str).unique())
        n_ds = len(datasets)
        if n_ds == 0:
            continue

        sf_model = pd.DataFrame()
        if silent_failures is not None and not silent_failures.empty:
            sf_model = silent_failures[silent_failures["model"].astype(str) == str(model)]
        any_silent_for_model = (
            not sf_model.empty
            and (
                pd.to_numeric(sf_model["n_silent_failures"], errors="coerce").fillna(0) > 0
            ).any()
        )

        n_cols = min(4, n_ds)
        n_rows = math.ceil(n_ds / n_cols)
        fig_w = max(7.0, n_cols * 3.15)
        fig_h = max(3.0, n_rows * 2.75)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )

        for idx, ds in enumerate(datasets):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]
            piece = sub_m[sub_m["dataset"].astype(str) == ds].copy()
            piece = piece.sort_values("n_seeds")
            n_silent = 0
            if silent_failures is not None and not silent_failures.empty:
                sf = silent_failures
                hit = sf[
                    (sf["model"].astype(str) == model)
                    & (sf["dataset"].astype(str) == ds)
                ]
                if not hit.empty:
                    n_silent = int(pd.to_numeric(hit["n_silent_failures"], errors="coerce").fillna(0).sum())

            if piece.empty and n_silent <= 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_title(_dataset_short_title(ds), fontsize=9, fontweight="semibold")
                ax.set_axis_off()
                continue

            if piece.empty:
                x = ["silent\nfail"]
                y = [float(n_silent)]
                colors = ["#D35400"]
            else:
                piece["n_seeds"] = pd.to_numeric(piece["n_seeds"], errors="coerce")
                piece["n_groups"] = pd.to_numeric(piece["n_groups"], errors="coerce").fillna(0)
                piece = piece.dropna(subset=["n_seeds"])

                x = piece["n_seeds"].astype(int).astype(str).tolist()
                y = piece["n_groups"].astype(float).tolist()
                colors = []
                for ns in piece["n_seeds"].astype(int):
                    if required_n_seeds is not None and int(ns) == int(required_n_seeds):
                        colors.append("#E74C3C")
                    else:
                        colors.append("#4A90A4")

                if n_silent > 0:
                    x = x + ["silent\nfail"]
                    y = y + [float(n_silent)]
                    colors = colors + ["#D35400"]

            ax.bar(x, y, color=colors, edgecolor="0.2", linewidth=0.45)
            ax.set_title(_dataset_short_title(ds), fontsize=9, fontweight="semibold")
            ax.set_xlabel("n_seeds (runs / group)", fontsize=8)
            ax.set_ylabel("# groups", fontsize=8)
            ax.yaxis.grid(True, linestyle=":", linewidth=0.45, alpha=0.85)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(n_ds, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r][c].set_visible(False)

        fig.suptitle(str(model), fontsize=11, fontweight="bold", y=1.02)
        foot_lines: list[str] = []
        if required_n_seeds is not None:
            foot_lines.append(
                f"Red bars: n_seeds == {required_n_seeds} (aggregator default filter)."
            )
        if any_silent_for_model:
            foot_lines.append(
                "Terracotta bar (silent fail): raw runs dropped (no finite summary_test_best_rerun/*)."
            )
        if foot_lines:
            fig.text(0.5, 0.01, " ".join(foot_lines), ha="center", fontsize=8, style="italic")
            fig.subplots_adjust(bottom=0.12, top=0.92)
        else:
            fig.subplots_adjust(bottom=0.08, top=0.92)

        stem = safe_filename_token(str(model).replace("/", "__"), max_len=96)
        out_path = out_dir / f"{stem}_n_seeds.png"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        n_written += 1

    return n_written


def _dataset_short_title(dataset_path: str) -> str:
    t = str(dataset_path).strip()
    if "/" in t:
        t = t.rsplit("/", 1)[-1]
    return t if len(t) <= 36 else t[:33] + "..."


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bar plots of n_seeds distribution per model (all datasets as subplots)."
    )
    p.add_argument(
        "-i",
        "--input",
        action="append",
        type=Path,
        default=None,
        metavar="PATH",
        help="Per-run export CSV (repeat for shards). Default: shard dir or monolithic export.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Glob CSVs under this directory (default: shard folder if present).",
    )
    p.add_argument(
        "--input-pattern",
        default="*.csv",
        help="Glob under --input-dir (default: *.csv)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SEED_N_DIST_DIR,
        help=f"Directory for PNGs (default: {DEFAULT_SEED_N_DIST_DIR})",
    )
    p.add_argument(
        "--required-seeds",
        type=int,
        default=5,
        metavar="N",
        help="Highlight bars where n_seeds equals N (aggregator default); use -1 to disable.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    args = p.parse_args()

    input_dir = args.input_dir
    if args.input is None and input_dir is None:
        sd = DEFAULT_WANDB_EXPORT_SHARD_DIR
        if sd.is_dir() and any(sd.glob(args.input_pattern)):
            input_dir = sd

    paths = _collect_input_paths(
        explicit=args.input,
        input_dir=input_dir,
        input_pattern=args.input_pattern,
    )
    report, silent = seed_bucket_report_from_export_paths(paths)
    req = int(args.required_seeds) if int(args.required_seeds) >= 0 else None
    n = write_seed_distribution_plots(
        report,
        args.output_dir,
        required_n_seeds=req,
        silent_failures=silent,
        dpi=int(args.dpi),
    )
    print(f"Wrote {n} figure(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
