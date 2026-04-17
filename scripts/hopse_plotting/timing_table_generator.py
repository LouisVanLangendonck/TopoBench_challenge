#!/usr/bin/env python3
"""
Build a LaTeX timing table from the W&B ``best_runs_rerun`` project.

For each (model, dataset) pair, assumes one rerun (dedupes if multiple). Reads
``AvgTime/train_epoch_mean`` and ``AvgTime/train_epoch_std`` from each run's
summary. Emits a LaTeX table with:

- **bestgray** + bold: lowest mean time per epoch in the column (ties share style).
- **stdblue**: not significantly different from the column best at 95% confidence
  (two-sided Z on independent means; SE = std / sqrt(n_seeds), assumed n_seeds=10).

Model blocks: Graph (GCN/GAT/GIN), Simplicial (HOPSE-M/HOPSE-G/TopoTune),
Cell (same trio). Dataset columns use the same ordering as ``table_generator.py``.

Usage::

    python scripts/hopse_plotting/timing_table_generator.py
    python scripts/hopse_plotting/timing_table_generator.py --entity your-entity
    python scripts/hopse_plotting/timing_table_generator.py -o tables/timing_table.tex
    python scripts/hopse_plotting/timing_table_generator.py --stdout
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    print("Error: wandb package required. Install with: pip install wandb", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas package required. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

from main_loader import DATASETS as LOADER_DATASETS
from utils import (
    TABLES_DIR,
    flatten_config,
    run_with_wandb_retry,
    safe_filename_token,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

WANDB_ENTITY_DEFAULT = "gbg141-hopse"
WANDB_PROJECT_RERUNS = "best_runs_rerun"

DEFAULT_TIMING_TABLE_TEX = TABLES_DIR / "timing_table_all.tex"
DEFAULT_TIMING_TABLE_TEX_NO_TRANS = TABLES_DIR / "timing_table_no_transductive.tex"

# Same as table_generator.py
TRANSDUCTIVE_GRAPH_PATHS: tuple[str, ...] = (
    "graph/cocitation_cora",
    "graph/cocitation_citeseer",
    "graph/cocitation_pubmed",
)
TRANSDUCTIVE_GRAPH_SET: frozenset[str] = frozenset(TRANSDUCTIVE_GRAPH_PATHS)

Z_CRIT_95 = 1.959963984540054

# Assumed number of seeds for SE calculation (std / sqrt(n_seeds))
N_SEEDS_ASSUMED = 10

# -----------------------------------------------------------------------------
# Dataset column headers (same as table_generator.py)
# -----------------------------------------------------------------------------

_DATASET_COLUMN_LABEL: dict[str, str] = {
    "graph/MUTAG": "MUTAG",
    "graph/cocitation_cora": "Cora",
    "graph/PROTEINS": "PROTEINS",
    "graph/NCI1": "NCI1",
    "graph/NCI109": "NCI109",
    "graph/cocitation_citeseer": "Citeseer",
    "graph/cocitation_pubmed": "PubMed",
    "simplicial/mantra_name": "NAME",
    "simplicial/mantra_orientation": "ORIENT",
    "simplicial/mantra_betti_numbers": r"$\beta$",
    "graph/BBB_Martins": "BBB",
    "graph/CYP3A4_Veith": "CYP3A4",
    "graph/Clearance_Hepatocyte_AZ": "Cl.Hep.",
    "graph/Caco2_Wang": "Caco2",
}


def _latex_short_dataset_label(path: str) -> str:
    return _DATASET_COLUMN_LABEL.get(path, path.rsplit("/", 1)[-1].replace("_", r"\_"))


def _auto_header_for_dataset_path(path: str) -> str:
    short = _latex_short_dataset_label(path)
    # For timing, lower is always better
    return f"{short} " + r"($\downarrow$)"


def _specs_from_loader_paths() -> list[tuple[str, str]]:
    return [
        (p.strip(), _auto_header_for_dataset_path(p.strip()))
        for p in LOADER_DATASETS
        if p.strip()
    ]


def partition_specs_three_way(
    specs: list[tuple[str, str]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Graph (transductive) → Graph (inductive) → Simplicial (inductive).
    """
    by_path = {p: h for p, h in specs}
    trans = [(p, by_path[p]) for p in TRANSDUCTIVE_GRAPH_PATHS if p in by_path]
    graph_ind: list[tuple[str, str]] = []
    simplicial: list[tuple[str, str]] = []
    for p, h in specs:
        if p in TRANSDUCTIVE_GRAPH_SET:
            continue
        if p.startswith("graph/"):
            graph_ind.append((p, h))
        elif p.startswith("simplicial/"):
            simplicial.append((p, h))
    blocks = [
        ("Graph (transductive)", trans),
        ("Graph (inductive)", graph_ind),
        ("Simplicial (inductive)", simplicial),
    ]
    return [(title, blk) for title, blk in blocks if blk]


def partition_specs_two_way_no_transductive(
    specs: list[tuple[str, str]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Graph then Simplicial; omits cocitation Cora/Citeseer/PubMed."""
    graph_ind: list[tuple[str, str]] = []
    simplicial: list[tuple[str, str]] = []
    for p, h in specs:
        if p in TRANSDUCTIVE_GRAPH_SET:
            continue
        if p.startswith("graph/"):
            graph_ind.append((p, h))
        elif p.startswith("simplicial/"):
            simplicial.append((p, h))
    blocks = [
        ("Graph", graph_ind),
        ("Simplicial", simplicial),
    ]
    return [(title, blk) for title, blk in blocks if blk]


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _sem(std: float, n: int) -> float:
    if n <= 0 or not _finite(std):
        return 0.0
    return float(std) / math.sqrt(float(n))


def _z_two_sample(mu_i: float, se_i: float, mu_j: float, se_j: float) -> float:
    v = se_i * se_i + se_j * se_j
    if v <= 0.0:
        return 0.0 if abs(mu_i - mu_j) < 1e-12 else float("inf")
    return abs(mu_i - mu_j) / math.sqrt(v)


def _not_sig_diff_from_best(mu: float, se: float, best_mu: float, best_se: float) -> bool:
    return _z_two_sample(mu, se, best_mu, best_se) <= Z_CRIT_95


# -----------------------------------------------------------------------------
# W&B data collection
# -----------------------------------------------------------------------------


def collect_timing_data(
    entity: str,
    project: str,
    verbose: bool = True,
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Fetch timing data from W&B.

    Returns:
        dict[(model, dataset)] = (mean_time, std_time)
    """
    api = wandb.Api(timeout=60)

    def _fetch():
        return api.runs(
            f"{entity}/{project}",
            filters={"state": "finished"},
        )

    if verbose:
        print(f"Fetching runs from {entity}/{project} ...")

    runs = run_with_wandb_retry(_fetch, label=f"{entity}/{project}")

    # Group by (model, dataset)
    runs_by_key: dict[tuple[str, str], list[Any]] = defaultdict(list)

    for run in runs:
        # Parse model and dataset from run name: domain__model__domain__dataset
        # e.g., "graph__gat__graph__BBB_Martins" -> model="graph/gat", dataset="graph/BBB_Martins"
        run_name = run.name
        parts = run_name.split("__")
        
        if len(parts) < 4:
            if verbose:
                print(f"  (skip) run {run.name}: unexpected name format (expected domain__model__domain__dataset)")
            continue
        
        # Reconstruct with slashes: domain/model and domain/dataset
        model_domain = parts[0]
        model_name = parts[1]
        dataset_domain = parts[2]
        dataset_name = "__".join(parts[3:])  # In case dataset has __ in it
        
        model = f"{model_domain}/{model_name}"
        dataset = f"{dataset_domain}/{dataset_name}"

        runs_by_key[(model, dataset)].append(run)

    # Extract timing data
    timing_data: dict[tuple[str, str], tuple[float, float]] = {}
    
    # Track what keys we've seen for diagnostics
    timing_keys_seen: set[str] = set()
    first_run_debugged = False

    for (model, dataset), run_list in runs_by_key.items():
        if len(run_list) > 1:
            if verbose:
                print(
                    f"  (warn) Multiple runs for ({model}, {dataset}): "
                    f"{len(run_list)} runs. Using first."
                )

        run = run_list[0]

        # AvgTime/* is logged as a metric (W&B summary). Rerun projects often have an
        # empty run.config; utils.run_to_row reads metrics via dict(run.summary).
        summary = dict(run.summary)
        flat_config = flatten_config(dict(run.config))
        metrics: dict[str, Any] = {**flat_config, **summary}

        if not first_run_debugged and verbose:
            sk = sorted(summary.keys())
            timing_in_summary = [
                k for k in sk if "time" in k.lower() or "Time" in k or "AvgTime" in k
            ]
            print(
                f"\nDEBUG: first run '{run.name}': "
                f"{len(summary)} summary keys, {len(flat_config)} flattened config keys."
            )
            if timing_in_summary:
                print(f"  Timing-related summary keys: {timing_in_summary}")
            else:
                print(f"  No timing keys in summary; first 15 summary keys: {sk[:15]}")
            first_run_debugged = True

        for key in summary.keys():
            if "time" in key.lower() or "Time" in key or "AvgTime" in key:
                timing_keys_seen.add(key)

        mean_key = "AvgTime/train_epoch_mean"
        std_key = "AvgTime/train_epoch_std"
        mean_time = metrics.get(mean_key)
        std_time = metrics.get(std_key)

        if mean_time is None or std_time is None:
            if verbose:
                print(
                    f"  (skip) ({model}, {dataset}): "
                    f"missing timing data (mean={mean_time}, std={std_time})"
                )
            continue

        if not _finite(mean_time) or not _finite(std_time):
            if verbose:
                print(
                    f"  (skip) ({model}, {dataset}): "
                    f"non-finite timing data (mean={mean_time}, std={std_time})"
                )
            continue

        timing_data[(model, dataset)] = (float(mean_time), float(std_time))
    
    if verbose:
        print(f"\n{'-'*70}")
        if timing_keys_seen:
            print(f"Timing-related keys found across all runs: {sorted(timing_keys_seen)}")
        else:
            print("WARNING: No timing-related keys found in any run summary!")
        print(f"Collected timing data for {len(timing_data)} (model, dataset) pairs")
        print(f"{'-'*70}\n")

    return timing_data


# -----------------------------------------------------------------------------
# LaTeX table building
# -----------------------------------------------------------------------------


def _format_time(seconds: float, decimals: int = 2) -> str:
    """Format time in seconds, use scientific notation if >= 1000."""
    if seconds >= 1000:
        return f"{seconds:.{decimals}e}"
    return f"{seconds:.{decimals}f}"


def _make_cell(
    mean: float,
    std: float,
    is_best: bool,
    is_stat_tied: bool,
    decimals: int,
) -> str:
    """
    Build a LaTeX table cell with color/bold based on statistical ranking.

    - bestgray + bold: is_best
    - stdblue: is_stat_tied (but not best)
    - plain: neither
    """
    mean_str = _format_time(mean, decimals)
    std_str = _format_time(std, decimals)
    content = f"${mean_str} \\pm {std_str}$"

    if is_best:
        return f"\\cellcolor{{bestgray}}\\textbf{{{content}}}"
    elif is_stat_tied:
        return f"\\cellcolor{{stdblue}}{content}"
    else:
        return content


def build_latex_table(
    timing_data: dict[tuple[str, str], tuple[float, float]],
    column_groups: list[tuple[str, list[tuple[str, str]]]],
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
    decimals: int = 2,
    label: str = "tbl:timing",
) -> str:
    """
    Build the full LaTeX table.

    Args:
        timing_data: dict[(model, dataset)] = (mean_time, std_time)
        column_groups: List of (group_title, [(dataset_path, header), ...])
        graph_rows: List of (model_path, latex_label) for graph models
        simplicial_rows: List of (model_path, latex_label) for simplicial models
        cell_rows: List of (model_path, latex_label) for cell models
        decimals: Decimal places for numbers
        label: LaTeX label
    """
    lines = []

    # Preamble
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")

    # Count total columns
    total_cols = sum(len(cols) for _, cols in column_groups)

    # Table header
    col_spec = "l" + "c" * total_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Build header rows
    header_row1 = ["\\textbf{Model}"]
    header_row2 = [""]

    for group_title, cols in column_groups:
        if cols:
            header_row1.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{\\textbf{{{group_title}}}}}")
            header_row2.extend([header for _, header in cols])

    lines.append(" & ".join(header_row1) + " \\\\")
    if len(column_groups) > 1:
        # Add cmidrule for each group
        col_idx = 2  # Start after Model column
        for _, cols in column_groups:
            if cols:
                lines.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + len(cols) - 1}}}")
                col_idx += len(cols)
    lines.append(" & ".join(header_row2) + " \\\\")
    lines.append("\\midrule")

    # Collect all dataset paths in order
    all_datasets: list[str] = []
    for _, cols in column_groups:
        all_datasets.extend([path for path, _ in cols])

    # Helper to emit a model block
    def emit_model_block(block_title: str, rows: list[tuple[str, str]]) -> None:
        if not rows:
            return

        lines.append(f"\\multicolumn{{{total_cols + 1}}}{{l}}{{\\textbf{{{block_title}}}}} \\\\")

        for model_path, model_label in rows:
            # For each dataset, find best and stat-tied
            col_data: list[tuple[float, float] | None] = []
            for ds_path in all_datasets:
                key = (model_path, ds_path)
                if key in timing_data:
                    col_data.append(timing_data[key])
                else:
                    col_data.append(None)

            # Find best (minimum mean) for each column
            row_cells = [model_label]

            for col_idx, data in enumerate(col_data):
                ds_path = all_datasets[col_idx]

                if data is None:
                    row_cells.append("---")
                    continue

                mean, std = data
                se = _sem(std, N_SEEDS_ASSUMED)

                # Find best in this column across all models
                best_mean = float("inf")
                best_se = 0.0

                for all_model_path, _ in graph_rows + simplicial_rows + cell_rows:
                    key = (all_model_path, ds_path)
                    if key in timing_data:
                        m, s = timing_data[key]
                        if _finite(m) and m < best_mean:
                            best_mean = m
                            best_se = _sem(s, N_SEEDS_ASSUMED)

                is_best = abs(mean - best_mean) < 1e-12
                is_stat_tied = (
                    not is_best
                    and _finite(best_mean)
                    and _not_sig_diff_from_best(mean, se, best_mean, best_se)
                )

                cell = _make_cell(mean, std, is_best, is_stat_tied, decimals)
                row_cells.append(cell)

            lines.append(" & ".join(row_cells) + " \\\\")

    # Emit model blocks
    emit_model_block("Graph", graph_rows)
    lines.append("\\midrule")
    emit_model_block("Simplicial", simplicial_rows)
    lines.append("\\midrule")
    emit_model_block("Cell", cell_rows)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append(
        f"\\caption{{Training time per epoch (seconds, mean $\\pm$ std). "
        f"\\cellcolor{{bestgray}}\\textbf{{Bold}}: best (lowest) time; "
        f"\\cellcolor{{stdblue}}Blue: not significantly worse than best (95\\% CI).}}"
    )
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX timing table from W&B best_runs_rerun project."
    )
    parser.add_argument(
        "--entity",
        default=WANDB_ENTITY_DEFAULT,
        help=f"W&B entity (default: {WANDB_ENTITY_DEFAULT})",
    )
    parser.add_argument(
        "--project",
        default=WANDB_PROJECT_RERUNS,
        help=f"W&B project name (default: {WANDB_PROJECT_RERUNS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_TIMING_TABLE_TEX,
        help=(
            f"Output .tex for three-band table (transductive + inductive) "
            f"(default: {DEFAULT_TIMING_TABLE_TEX})"
        ),
    )
    parser.add_argument(
        "--output-without-transductive",
        type=Path,
        default=DEFAULT_TIMING_TABLE_TEX_NO_TRANS,
        help=(
            f"Second .tex: no transductive datasets "
            f"(default: {DEFAULT_TIMING_TABLE_TEX_NO_TRANS})"
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print LaTeX to stdout instead of writing files",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for timing values (default: 2)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output",
    )

    args = parser.parse_args()

    # Check for WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY"):
        print(
            "Warning: WANDB_API_KEY not set. Run 'wandb login' or set the environment variable.",
            file=sys.stderr,
        )

    # Collect timing data
    timing_data = collect_timing_data(
        args.entity,
        args.project,
        verbose=not args.quiet,
    )

    if not timing_data:
        print("Error: No timing data found. Check entity/project names.", file=sys.stderr)
        sys.exit(1)

    # Build dataset specs
    base_specs = _specs_from_loader_paths()

    # Create both table variants
    groups_three = partition_specs_three_way(base_specs)
    groups_two = partition_specs_two_way_no_transductive(base_specs)
    n_two_cols = sum(len(b) for _, b in groups_two)

    # Model rows (same as table_generator.py)
    graph_rows = [
        ("graph/gcn", "GCN"),
        ("graph/gat", "GAT"),
        ("graph/gin", "GIN"),
    ]
    simplicial_rows = [
        ("simplicial/hopse_m", "\\textbf{HOPSE-M} (Our)"),
        ("simplicial/hopse_g", "\\textbf{HOPSE-G} (Our)"),
        ("simplicial/topotune", "TopoTune"),
    ]
    cell_rows = [
        ("cell/hopse_m", "\\textbf{HOPSE-M} (Our)"),
        ("cell/hopse_g", "\\textbf{HOPSE-G} (Our)"),
        ("cell/topotune", "TopoTune"),
    ]

    # Build LaTeX tables
    tex_three = build_latex_table(
        timing_data,
        column_groups=groups_three,
        graph_rows=graph_rows,
        simplicial_rows=simplicial_rows,
        cell_rows=cell_rows,
        decimals=args.decimals,
        label="tbl:timing_all",
    )

    tex_two: str | None = None
    if n_two_cols > 0:
        tex_two = build_latex_table(
            timing_data,
            column_groups=groups_two,
            graph_rows=graph_rows,
            simplicial_rows=simplicial_rows,
            cell_rows=cell_rows,
            decimals=args.decimals,
            label="tbl:timing_no_transductive",
        )

    # Output
    if args.stdout:
        sys.stdout.write(tex_three)
        if tex_two is not None:
            sys.stdout.write(
                "\n% --- version without transductive graph (cocitation cora/citeseer/pubmed) ---\n\n"
            )
            sys.stdout.write(tex_two)
    else:
        out1 = Path(args.output)
        out1.parent.mkdir(parents=True, exist_ok=True)
        out1.write_text(tex_three, encoding="utf-8")
        print(f"Wrote {out1}")

        if tex_two is not None:
            out2 = Path(args.output_without_transductive)
            out2.parent.mkdir(parents=True, exist_ok=True)
            out2.write_text(tex_two, encoding="utf-8")
            print(f"Wrote {out2}")
        else:
            print(
                "Skipped second table (--output-without-transductive): "
                "no columns left after dropping transductive datasets."
            )


if __name__ == "__main__":
    main()