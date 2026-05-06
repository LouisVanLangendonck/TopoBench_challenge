#!/usr/bin/env python3
"""
Export TopoBench / HOPSE sweeps from Weights & Biases into CSV file(s).

Expected W&B project names follow the sweep scripts: ``{model}_{dataset_basename}``,
e.g. ``graph/CYP3A4_Veith`` + model ``gin`` → project ``gin_CYP3A4_Veith``.

Requires ``WANDB_API_KEY`` (or prior ``wandb login``) and the ``wandb`` package.

By default writes **multiple** smaller CSVs (one per model, all datasets) under
``csvs/hopse_experiments_wandb_export_shards/`` so memory stays bounded. Use
``--shard-by none -o path.csv`` for a single monolithic export under ``csvs/``.
Run ``aggregator`` to produce **one** combined seed-aggregated CSV.

Each raw row includes **timing** promoted into ``summary_*`` columns (train epoch mean/std from
``AvgTime/*`` in the run config, wall seconds as ``summary_Runtime`` from W&B), so you normally
do not need a second W&B fetch for timing plots or tables.

Usage (from repo root)::

    python scripts/hopse_plotting/main_loader.py
    python scripts/hopse_plotting/main_loader.py --shard-by dataset
    python scripts/hopse_plotting/main_loader.py --shard-by none \\
        -o scripts/hopse_plotting/csvs/hopse_experiments_wandb_export.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    DEFAULT_WANDB_EXPORT_CSV,
    DEFAULT_WANDB_EXPORT_SHARD_DIR,
    collect_all_runs,
    dataframe_from_rows,
    safe_filename_token,
)

# -----------------------------------------------------------------------------
# Hard-coded sweep coverage (edit here)
# -----------------------------------------------------------------------------

WANDB_ENTITY = "gbg141-hopse"

MODELS = ["hopse_g"] #["gin","gat", "gcn", "topotune", "hopse_m", "hopse_g", "sann", "sccnn", "cwn", "cccn"]

DATASETS = [    
    "graph/MUTAG",
    "graph/PROTEINS",
    "graph/NCI1",
    "graph/NCI109",
    "simplicial/mantra_name",
    "simplicial/mantra_orientation",
    "simplicial/mantra_betti_numbers",
    "graph/BBB_Martins",
    "graph/CYP3A4_Veith",
    "graph/Clearance_Hepatocyte_AZ",
    "graph/Caco2_Wang",
]

DEFAULT_OUTPUT_CSV = DEFAULT_WANDB_EXPORT_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Export W&B TopoBench sweeps to CSV.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Single output CSV path when --shard-by none (default: {DEFAULT_OUTPUT_CSV}). Ignored when sharding.",
    )
    parser.add_argument(
        "--entity",
        default=WANDB_ENTITY,
        help="W&B entity (default: hard-coded WANDB_ENTITY)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output",
    )
    parser.add_argument(
        "--run-state",
        default="finished",
        metavar="STATE",
        help='W&B run filter: "finished" (default), "running", "crashed", "failed", or "all" for no filter',
    )
    parser.add_argument(
        "--shard-by",
        choices=("none", "model", "dataset"),
        default="model",
        help='Write one CSV per "model" (all datasets, default) or per "dataset" (all models). Use "none" for a single -o file.',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for sharded CSVs (default: {DEFAULT_WANDB_EXPORT_SHARD_DIR}). Ignored when --shard-by none.",
    )
    parser.add_argument(
        "--basename",
        default="hopse_experiments_wandb_export",
        help="File stem for sharded files (default: hopse_experiments_wandb_export).",
    )
    args = parser.parse_args()

    run_state: str | None
    if str(args.run_state).lower() == "all":
        run_state = None
    else:
        run_state = str(args.run_state)

    print(f"Entity: {args.entity}")
    print(f"Models ({len(MODELS)}): {MODELS}")
    print(f"Datasets ({len(DATASETS)}): {DATASETS}")

    if args.shard_by == "none":
        print("Collecting runs …")
        rows = collect_all_runs(
            args.entity,
            MODELS,
            DATASETS,
            run_state=run_state,
            verbose=not args.quiet,
        )
        df = dataframe_from_rows(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows x {len(df.columns)} columns -> {args.output}")
        return

    out_dir = args.output_dir or DEFAULT_WANDB_EXPORT_SHARD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if args.shard_by == "model":
        print(f"Collecting runs (sharded by model) into {out_dir} …")
        for model in MODELS:
            if verbose:
                print(f"  (shard) model={model!r}")
            rows = collect_all_runs(
                args.entity,
                [model],
                DATASETS,
                run_state=run_state,
                verbose=verbose,
            )
            df = dataframe_from_rows(rows)
            stem = safe_filename_token(str(model).replace("/", "__"))
            path = out_dir / f"{args.basename}__{stem}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"    -> {len(df)} rows x {len(df.columns)} columns -> {path}")
        return

    print(f"Collecting runs (sharded by dataset) into {out_dir} …")
    for ds in DATASETS:
        if verbose:
            print(f"  (shard) dataset={ds!r}")
        rows = collect_all_runs(
            args.entity,
            MODELS,
            [ds],
            run_state=run_state,
            verbose=verbose,
        )
        df = dataframe_from_rows(rows)
        stem = safe_filename_token(str(ds).replace("/", "__"))
        path = out_dir / f"{args.basename}__{stem}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"    -> {len(df)} rows x {len(df.columns)} columns -> {path}")


if __name__ == "__main__":
    main()
