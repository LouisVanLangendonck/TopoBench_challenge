#!/usr/bin/env python3
"""
Aggregate per-run W&B export rows across ``dataset.split_params.data_seed``.

Reads per-run export CSV(s)—by default, every ``*.csv`` under
``csvs/hopse_experiments_wandb_export_shards`` when that folder exists and has
files; otherwise the monolithic ``csvs/hopse_experiments_wandb_export.csv``.
Several shard files are aggregated then concatenated into one ``-o`` CSV
(default: ``csvs/hopse_experiments_wandb_export_seed_agg.csv``).

By default only hyperparameter groups with exactly ``--required-seeds`` raw
runs (after grouping on everything except the data seed) are written to the
output CSV; see the printed per-(model, dataset) distribution for other counts.

Raw export rows with **no** finite value in any ``summary_test_best_rerun/*``
column are dropped before aggregation (silent failures). Counts are printed and
shown as a **silent fail** bar in ``--plot-seed-distributions`` figures.

Usage::

    python scripts/hopse_plotting/aggregator.py
    python scripts/hopse_plotting/aggregator.py -i path/to/export.csv -o path/to/agg.csv
    python scripts/hopse_plotting/aggregator.py --input-dir scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_shards
    python scripts/hopse_plotting/aggregator.py --keep-incomplete-seeds
    python scripts/hopse_plotting/aggregator.py --plot-seed-distributions
    python scripts/hopse_plotting/aggregator.py --plot-seed-distributions --seed-dist-dir plots/seed_n
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    DEFAULT_AGGREGATED_EXPORT_CSV,
    DEFAULT_WANDB_EXPORT_CSV,
    DEFAULT_WANDB_EXPORT_SHARD_DIR,
    PLOTS_DIR,
    aggregate_many_wandb_export_csvs,
    aggregate_wandb_export_csv,
)

# Exact raw-run count per hyperparameter group required for a row to appear in ``-o``.
DEFAULT_REQUIRED_AGGREGATED_SEEDS = 5


def _print_seed_bucket_report(
    report,
    *,
    required_n_seeds: int | None,
) -> None:
    if report.empty:
        print("Seed-count distribution: (no aggregated hyperparameter groups).")
        return
    if required_n_seeds is not None:
        print(
            f"Seed-count distribution (hyperparameter groups per model+dataset); "
            f"output CSV keeps only n_seeds=={required_n_seeds}."
        )
    else:
        print(
            "Seed-count distribution (hyperparameter groups per model+dataset); "
            "output CSV keeps all n_seeds (--keep-incomplete-seeds)."
        )
    for (model, dataset), sub in report.groupby(["model", "dataset"], dropna=False):
        print(f"\n  model={model!r}  dataset={dataset!r}")
        sub_sorted = sub.sort_values("n_seeds")
        for _, row in sub_sorted.iterrows():
            k = row["n_seeds"]
            try:
                k_int = int(k) if pd.notna(k) else k
            except (TypeError, ValueError):
                k_int = k
            mark = (
                "  <- rows written to -o"
                if required_n_seeds is not None
                and pd.notna(k)
                and int(k) == int(required_n_seeds)
                else ""
            )
            print(
                f"    n_seeds={k_int}: {int(row['n_groups'])} groups "
                f"({float(row['pct_of_groups']):.2f}% of groups for this pair){mark}"
            )


def _print_silent_failure_report(silent: pd.DataFrame) -> None:
    if silent is None or silent.empty:
        print("\nSilent failures (no summary_test_best_rerun metrics): 0 raw runs dropped.")
        return
    tot = int(pd.to_numeric(silent["n_silent_failures"], errors="coerce").fillna(0).sum())
    print(
        f"\nSilent failures (no finite summary_test_best_rerun/* on raw row): "
        f"{tot} raw run(s) dropped before seed aggregation."
    )
    print("  Per (model, dataset) counts (see terracotta 'silent fail' bar in seed distribution plots):")
    show = silent.sort_values("n_silent_failures", ascending=False).head(40)
    for _, row in show.iterrows():
        v = pd.to_numeric(row["n_silent_failures"], errors="coerce")
        n = int(v) if pd.notna(v) else 0
        print(f"    model={row['model']!r}  dataset={row['dataset']!r}  dropped={n}")
    if len(silent) > len(show):
        print(f"    ... and {len(silent) - len(show)} more (model, dataset) pair(s).")


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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate W&B export CSV(s) over data seeds; always one combined -o CSV."
    )
    p.add_argument(
        "-i",
        "--input",
        action="append",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Per-run export CSV (repeat for multiple shards). If omitted, see --input-dir / default shard folder.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Aggregate every file matching --input-pattern under this directory. "
            "If -i is not given and this is omitted, uses the shard folder when it "
            f"contains CSVs, else {DEFAULT_WANDB_EXPORT_CSV}"
        ),
    )
    p.add_argument(
        "--input-pattern",
        default="*.csv",
        help="Glob under --input-dir (default: *.csv)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_AGGREGATED_EXPORT_CSV,
        help=f"Single combined seed-aggregated CSV (default: {DEFAULT_AGGREGATED_EXPORT_CSV})",
    )
    p.add_argument(
        "--required-seeds",
        type=int,
        default=DEFAULT_REQUIRED_AGGREGATED_SEEDS,
        metavar="N",
        help=(
            "Only write hyperparameter groups aggregated from exactly this many "
            f"raw runs (default: {DEFAULT_REQUIRED_AGGREGATED_SEEDS}). "
            "Ignored with --keep-incomplete-seeds."
        ),
    )
    p.add_argument(
        "--keep-incomplete-seeds",
        action="store_true",
        help=(
            "Write all aggregated groups regardless of run count; still print the "
            "per-(model,dataset) n_seeds distribution."
        ),
    )
    p.add_argument(
        "--plot-seed-distributions",
        action="store_true",
        help=(
            "After aggregating, write per-model PNGs (bar chart of n_seeds vs #groups per "
            "dataset subplot) under --seed-dist-dir (see seed_n_distribution_plots.py)."
        ),
    )
    p.add_argument(
        "--seed-dist-dir",
        type=Path,
        default=None,
        help=f"Output directory for --plot-seed-distributions (default: {PLOTS_DIR}/seed_n_distributions).",
    )
    args = p.parse_args()
    if not args.keep_incomplete_seeds and int(args.required_seeds) < 1:
        p.error("--required-seeds must be >= 1 unless --keep-incomplete-seeds is set.")

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    req = None if args.keep_incomplete_seeds else int(args.required_seeds)

    if len(paths) == 1:
        agg, report, silent = aggregate_wandb_export_csv(
            paths[0], args.output, required_n_seeds=req
        )
        print(f"Wrote {len(agg)} aggregated rows x {len(agg.columns)} columns -> {args.output}")
    else:
        agg, report, silent = aggregate_many_wandb_export_csvs(paths, args.output, required_n_seeds=req)
        print(
            f"Combined {len(paths)} shard file(s) -> {len(agg)} aggregated rows x {len(agg.columns)} columns -> {args.output}"
        )

    _print_silent_failure_report(silent)
    _print_seed_bucket_report(report, required_n_seeds=req)

    if args.plot_seed_distributions:
        from seed_n_distribution_plots import write_seed_distribution_plots

        dist_dir = args.seed_dist_dir or (PLOTS_DIR / "seed_n_distributions")
        nfig = write_seed_distribution_plots(
            report,
            dist_dir,
            required_n_seeds=req,
            silent_failures=silent,
            dpi=150,
        )
        print(f"Seed n_seeds bar plots: wrote {nfig} figure(s) -> {dist_dir}")


if __name__ == "__main__":
    main()
