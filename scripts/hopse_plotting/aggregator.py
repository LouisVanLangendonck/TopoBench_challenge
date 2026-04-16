#!/usr/bin/env python3
"""
Aggregate per-run W&B export rows across ``dataset.split_params.data_seed``.

Reads per-run export CSV(s)—by default, every ``*.csv`` under
``csvs/hopse_experiments_wandb_export_shards`` when that folder exists and has
files; otherwise the monolithic ``csvs/hopse_experiments_wandb_export.csv``.
Several shard files are aggregated then concatenated into one ``-o`` CSV
(default: ``csvs/hopse_experiments_wandb_export_seed_agg.csv``).

Usage::

    python scripts/hopse_plotting/aggregator.py
    python scripts/hopse_plotting/aggregator.py -i path/to/export.csv -o path/to/agg.csv
    python scripts/hopse_plotting/aggregator.py --input-dir scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_shards
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    DEFAULT_AGGREGATED_EXPORT_CSV,
    DEFAULT_WANDB_EXPORT_CSV,
    DEFAULT_WANDB_EXPORT_SHARD_DIR,
    aggregate_many_wandb_export_csvs,
    aggregate_wandb_export_csv,
)


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        agg = aggregate_wandb_export_csv(paths[0], args.output)
        print(f"Wrote {len(agg)} aggregated rows x {len(agg.columns)} columns -> {args.output}")
    else:
        agg = aggregate_many_wandb_export_csvs(paths, args.output)
        print(
            f"Combined {len(paths)} shard file(s) -> {len(agg)} aggregated rows x {len(agg.columns)} columns -> {args.output}"
        )


if __name__ == "__main__":
    main()
