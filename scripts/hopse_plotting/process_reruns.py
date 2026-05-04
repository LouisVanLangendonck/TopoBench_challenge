#!/usr/bin/env python3
"""
Fetch **best-val rerun** runs from a single W&B project (default ``best_runs_rerun``),
aggregate across ``dataset.split_params.data_seed`` like ``aggregator.py``, write one
seed-aggregated CSV, and emit **collapsed** LaTeX tables (one GNN row per backbone), same
column layout as ``table_generator`` compact mode:

1. **Performance** — test mean ± std from ``summary_test_best_rerun/*`` (same picks as main tables).
2. **Train time per epoch** — from ``AvgTime/train_epoch_mean`` / ``AvgTime/train_epoch_std``.
   TopoBench logs these via ``log_hyperparams`` (W&B **config**, not scalar summary); this
   script copies them into ``summary_*`` for the CSV. For each (model, dataset, submodel),
   among all raw seeds with finite timing, the run with the **lowest within-run epoch std**
   (``summary_AvgTime/train_epoch_std``) is kept as the most stable timing; ± is that
   within-run variability. Two LaTeX tables use the same numbers: one bolds the column
   minimum **per domain** (graph / simplicial / cell), the other **across all models**.
3. **End-to-end wall time** — W&B's ``_runtime`` (seconds) is mapped to ``summary_Runtime`` and
   aggregated across seeds (mean ± std).

Time ``.tex`` tables (runtime + per-epoch): default **bold** = lowest mean in that **dataset
column** within the same **domain band**; **blue** = not significantly different from that
within-domain minimum (two-sided $Z$ at 95\\,\\%, SE $= \\sigma/\\sqrt{n\_\\mathrm{seeds}}$).
The extra ``rerun_time_train_epoch_bold_global.tex`` uses the same per-epoch cells but bolds
the minimum **across all models** in each column.

Run from repo root (``sys.path`` includes this directory when invoking the script)::

    python scripts/hopse_plotting/process_reruns.py
    python scripts/hopse_plotting/process_reruns.py --keep-incomplete-seeds
    python scripts/hopse_plotting/process_reruns.py --write-raw-csv scripts/hopse_plotting/csvs/best_runs_rerun_raw.csv

Requires ``wandb`` and ``WANDB_API_KEY`` (or ``wandb login``).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from table_generator import (
    TABLES_DIR,
    _finite,
    _latex_cell_body,
    _normalize_table_model_id,
    _not_sig_diff_from_best,
    _sem,
    _specs_from_loader_paths,
    _val_mean_for_pick_row,
    build_latex_table,
    cell_submodel_table_rows,
    collapse_gnn_submodel_rows_to_base,
    collect_winner_test_by_submodel,
    dataframe_with_submodel_id,
    expand_mantra_betti_specs,
    hydra_dataset_key_from_loader_identity,
    is_mantra_betti_hydra_dataset,
    optimization_mode_for_metric_tail,
    partition_specs_graph_simplicial,
    simplicial_submodel_table_rows,
)
from utils import (
    CSV_DIR,
    MANTRA_BETTI_F1_TAILS,
    MANTRA_BETTI_HYDRA_DATASET,
    MONITOR_METRIC_COLUMN,
    SEED_COLUMN,
    SUMMARY_COLUMN_PREFIX,
    aggregate_wandb_export_by_seed,
    build_seed_bucket_report,
    dataframe_from_rows,
    filter_aggregated_to_required_n_seeds,
    iter_best_val_group_picks,
    iter_runs,
    list_seed_aggregatable_summary_columns,
    metric_name_tail,
    run_to_row,
)

# Match ``best_rerun_sh_generator.py`` defaults.
DEFAULT_WANDB_ENTITY = "gbg141-hopse"
DEFAULT_WANDB_PROJECT = "best_runs_rerun"

DEFAULT_AGG_CSV = CSV_DIR / "best_runs_rerun_seed_agg.csv"

SUMMARY_RUNTIME = f"{SUMMARY_COLUMN_PREFIX}Runtime"
# ``PipelineTimer`` logs these via ``log_hyperparams`` → they live in **run.config**, not summary.
SUMMARY_EPOCH_MEAN = f"{SUMMARY_COLUMN_PREFIX}AvgTime/train_epoch_mean"
SUMMARY_EPOCH_STD = f"{SUMMARY_COLUMN_PREFIX}AvgTime/train_epoch_std"


def augment_run_row_wandb_timing(row: dict[str, Any], run) -> dict[str, Any]:
    """
    Promote timing fields into the same ``summary_*`` columns as scalar metrics so CSV
    export / seed aggregation match ``main_loader``-style tables.

    - **Per-epoch:** ``AvgTime/train_epoch_{mean,std}`` from flattened ``run.config``.
    - **Wall clock:** W&B run duration is usually ``run.summary['_runtime']`` (seconds);
      we normalize to ``summary_Runtime`` for downstream code.
    """
    from utils import _serialize_cell, flatten_config, get_from_flat

    out = dict(row)
    flat_cfg = flatten_config(dict(run.config or {}))

    for key in ("AvgTime/train_epoch_mean", "AvgTime/train_epoch_std"):
        v = get_from_flat(flat_cfg, key)
        if v is None or v == "":
            continue
        cell = _serialize_cell(v)
        if cell:
            out[f"{SUMMARY_COLUMN_PREFIX}{key}"] = cell

    try:
        summary = dict(run.summary) if run.summary is not None else {}
    except Exception:
        summary = {}

    # If anything logged AvgTime as a scalar metric, prefer filling gaps from summary.
    for key in ("AvgTime/train_epoch_mean", "AvgTime/train_epoch_std"):
        col = f"{SUMMARY_COLUMN_PREFIX}{key}"
        if col in out and str(out[col]).strip():
            continue
        if key in summary:
            out[col] = _serialize_cell(summary[key])

    wall = None
    for wb_key in ("_runtime", "Runtime", "runtime"):
        if wb_key in summary:
            wall = summary[wb_key]
            break
    if wall is None:
        v = get_from_flat(flat_cfg, "_runtime")
        if v not in (None, ""):
            wall = v
    if wall is not None:
        out[SUMMARY_RUNTIME] = _serialize_cell(wall)

    return out


def collect_runs_single_project(
    entity: str,
    project: str,
    *,
    run_state: str | None = "finished",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api(timeout=120)
    rows: list[dict[str, Any]] = []
    _filt = f"state={run_state}" if run_state else "all states"
    if verbose:
        print(f"  (fetch) {entity}/{project} ({_filt})", flush=True)
    count = 0
    runs_gen = iter_runs(api, entity, project, state=run_state)
    for run in runs_gen:
        base = run_to_row(entity=entity, project=project, run=run)
        rows.append(augment_run_row_wandb_timing(base, run))
        count += 1
        if verbose and count % 250 == 0:
            print(f"    … {count} run(s) so far", flush=True)
    if verbose:
        print(f"    -> {count} run(s)", flush=True)
        if rows:
            peek = pd.DataFrame(rows)
            if "model" in peek.columns:
                models = sorted(peek["model"].astype(str).unique())
                print(f"    Unique models in export ({len(models)}): {models}")
                print(
                    "    (If a model is missing, its reruns may still be non-finished — try "
                    "--run-state all, or confirm runs use this entity/project.)"
                )
    return rows


def _summary_metric_columns_for_rerun_export(df: pd.DataFrame) -> list[str] | None:
    cols = list_seed_aggregatable_summary_columns(df)
    extra: list[str] = []
    if SUMMARY_RUNTIME in df.columns:
        extra.append(SUMMARY_RUNTIME)
    if not cols and not extra:
        return None
    return sorted(set(cols) | set(extra))


def _print_seed_bucket_report(report: pd.DataFrame, *, required_n_seeds: int | None) -> None:
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
            "Seed-count distribution; output CSV keeps all n_seeds (--keep-incomplete-seeds)."
        )
    for (model, dataset), sub in report.groupby(["model", "dataset"], dropna=False):
        print(f"\n  model={model!r}  dataset={dataset!r}")
        for _, row in sub.sort_values("n_seeds").iterrows():
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


def collect_runtime_stats_from_agg_winners(
    df_agg: pd.DataFrame,
    *,
    mean_col: str,
    std_col: str | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Same validation picks as ``collect_winner_test_by_submodel``, but store
    ``mean_col`` / ``std_col`` in ``test_mean`` / ``test_std`` for reuse of
    ``collapse_gnn_submodel_rows_to_base``.
    """
    work = dataframe_with_submodel_id(df_agg)
    colset = set(work.columns)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    gc = ["model", "dataset", "_sub_id"]
    for keys, pick_idx, monitor_val, tail in iter_best_val_group_picks(
        work, group_cols=gc, monitor_column=MONITOR_METRIC_COLUMN
    ):
        model = _normalize_table_model_id(str(keys[0]))
        dataset_raw = str(keys[1]).strip()
        sub_id = str(keys[2]).strip()
        dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
        row_key = f"{model}|{sub_id}"
        w = work.loc[pick_idx]

        if mean_col not in w.index or mean_col not in colset:
            mu = float("nan")
        else:
            mu = float(pd.to_numeric(w.get(mean_col), errors="coerce"))
            if pd.isna(mu):
                mu = float("nan")

        sd = float("nan")
        if std_col and std_col in colset:
            v = pd.to_numeric(w.get(std_col), errors="coerce")
            if pd.notna(v):
                sd = float(v)

        if is_mantra_betti_hydra_dataset(dataset_raw):
            for fi_tail in MANTRA_BETTI_F1_TAILS:
                col_key = f"{MANTRA_BETTI_HYDRA_DATASET}#{fi_tail}"
                vm = _val_mean_for_pick_row(w, fi_tail, colset)
                out[(row_key, col_key)] = {
                    "test_mean": mu,
                    "test_std": sd,
                    "val_mean": vm,
                    "tail": fi_tail,
                    "mode": "max",
                    "monitor_raw": str(monitor_val).strip(),
                    "n_seeds": int(pd.to_numeric(w.get("n_seeds"), errors="coerce") or 0),
                }
            continue

        mode = optimization_mode_for_metric_tail(tail) if tail else "max"
        vm = _val_mean_for_pick_row(w, tail, colset)
        out[(row_key, dataset)] = {
            "test_mean": mu,
            "test_std": sd,
            "val_mean": vm,
            "tail": tail,
            "mode": mode,
            "monitor_raw": str(monitor_val).strip(),
            "n_seeds": int(pd.to_numeric(w.get("n_seeds"), errors="coerce") or 0),
        }
    return out


def _pick_raw_row_min_epoch_timing_std(
    sub: pd.DataFrame, *, mean_col: str, std_col: str
) -> pd.Series | None:
    """
    Among raw runs in ``sub``, require finite ``mean_col`` (seconds per epoch).
    Prefer the run with the lowest ``std_col`` (within-run epoch timing variability); tie-break
    by lower mean (faster), then lower data seed.
    """
    best: tuple[tuple, pd.Series] | None = None
    for _idx, sr in sub.iterrows():
        mu = pd.to_numeric(sr.get(mean_col), errors="coerce")
        if pd.isna(mu) or not math.isfinite(float(mu)):
            continue
        mu_f = float(mu)
        sig = float("nan")
        if std_col and std_col in sub.columns:
            v = pd.to_numeric(sr.get(std_col), errors="coerce")
            if pd.notna(v) and math.isfinite(float(v)):
                sig = float(v)
        seed_tie = pd.to_numeric(sr.get(SEED_COLUMN), errors="coerce")
        seed_tie = float(seed_tie) if pd.notna(seed_tie) else float("inf")
        if math.isfinite(sig):
            key = (0, sig, mu_f, seed_tie)
        else:
            key = (1, float("inf"), mu_f, seed_tie)
        if best is None or key < best[0]:
            best = (key, sr)
    return None if best is None else best[1]


def collect_epoch_time_stats_min_timing_std(
    df_raw: pd.DataFrame,
    *,
    mean_col: str,
    std_col: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    One raw row per (model, dataset, _sub_id): pick the seed whose logged **epoch-time std**
    (within-run) is smallest among rows with finite per-epoch mean. Values stored as
    ``test_mean`` / ``test_std`` for table + GNN collapse.
    """
    work = dataframe_with_submodel_id(df_raw)
    colset = set(work.columns)
    if mean_col not in work.columns:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}

    for (_model, dataset_raw, sub_id), sub in work.groupby(
        ["model", "dataset", "_sub_id"], dropna=False
    ):
        model = _normalize_table_model_id(str(_model))
        row_key = f"{model}|{sub_id}"
        ds_raw = str(dataset_raw).strip()
        dataset = hydra_dataset_key_from_loader_identity(ds_raw)

        row = _pick_raw_row_min_epoch_timing_std(sub, mean_col=mean_col, std_col=std_col)
        if row is None:
            continue

        mu = float(pd.to_numeric(row.get(mean_col), errors="coerce"))
        if not math.isfinite(mu):
            mu = float("nan")
        sd = float("nan")
        if std_col and std_col in work.columns:
            v = pd.to_numeric(row.get(std_col), errors="coerce")
            if pd.notna(v):
                sd = float(v)

        if is_mantra_betti_hydra_dataset(ds_raw):
            for fi_tail in MANTRA_BETTI_F1_TAILS:
                col_key = f"{MANTRA_BETTI_HYDRA_DATASET}#{fi_tail}"
                vm = _val_mean_for_pick_row(row, fi_tail, colset)
                out[(row_key, col_key)] = {
                    "test_mean": mu,
                    "test_std": sd,
                    "val_mean": vm,
                    "tail": fi_tail,
                    "mode": "max",
                    "monitor_raw": str(row.get(MONITOR_METRIC_COLUMN, "") or "").strip(),
                    "n_seeds": 1,
                }
            continue

        mon = str(row.get(MONITOR_METRIC_COLUMN, "") or "").strip()
        tail = metric_name_tail(mon)
        mode = optimization_mode_for_metric_tail(tail) if tail else "max"
        vm = _val_mean_for_pick_row(row, tail, colset)
        out[(row_key, dataset)] = {
            "test_mean": mu,
            "test_std": sd,
            "val_mean": vm,
            "tail": tail,
            "mode": mode,
            "monitor_raw": mon,
            "n_seeds": 1,
        }
    return out


def _domain_row_keys_for_time_table(
    row_key: str,
    *,
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
) -> list[str]:
    """Stats row keys in the same rotated band as ``row_key`` (graph / simplicial / cell)."""
    if str(row_key).startswith("graph/"):
        return [rk for rk, _ in graph_rows]
    if str(row_key).startswith("simplicial/"):
        return [rk for rk, _ in simplicial_rows]
    if str(row_key).startswith("cell/"):
        return [rk for rk, _ in cell_rows]
    return [rk for rk, _ in graph_rows + simplicial_rows + cell_rows]


def _all_row_keys_for_time_table(
    *,
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
) -> list[str]:
    """Every model row key (graph + simplicial + cell) for column-global minimum / Z comparisons."""
    return [rk for rk, _ in graph_rows + simplicial_rows + cell_rows]


def build_rerun_metric_table_plain(
    stats: dict[tuple[str, str], dict[str, Any]],
    *,
    column_groups: list[tuple[str, list[tuple[str, str]]]],
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
    caption: str,
    label: str,
    decimals: int = 2,
    comparison_scope: str = "domain",
) -> str:
    """
    Same geometry as ``build_latex_table``. **Lower mean time is better** per dataset column.

    ``comparison_scope``:

    - ``"domain"`` (default): bold / blue use the minimum **within the same band** (graph,
      simplicial, or cell) as ``row_key``.
    - ``"global"``: bold / blue use the minimum **across all model rows** in that column.
    """
    dataset_specs: list[tuple[str, str]] = []
    group_ranges: list[tuple[str, int, int]] = []
    for title, block in column_groups:
        if not block:
            continue
        i0 = len(dataset_specs)
        dataset_specs.extend(block)
        group_ranges.append((title, i0, len(dataset_specs) - 1))

    n_d = len(dataset_specs)
    colspec = "@{}ll" + "c" * n_d + "@{}"

    def _tol_eq(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-9 * (1.0 + abs(b))

    def cell_time_colored(row_key: str, ds_path: str) -> str:
        ds_key = hydra_dataset_key_from_loader_identity(ds_path)
        st = stats.get((row_key, ds_key))
        if not st or not _finite(st.get("test_mean", float("nan"))):
            return "-"
        mu = float(st["test_mean"])
        sd = float(st["test_std"]) if _finite(st.get("test_std")) else float("nan")
        n_raw = st.get("n_seeds", 0)
        n_seeds = int(pd.to_numeric(n_raw, errors="coerce")) if _finite(n_raw) else 0
        se = _sem(sd, max(n_seeds, 0))

        if comparison_scope == "global":
            band_keys = _all_row_keys_for_time_table(
                graph_rows=graph_rows,
                simplicial_rows=simplicial_rows,
                cell_rows=cell_rows,
            )
        else:
            band_keys = _domain_row_keys_for_time_table(
                row_key,
                graph_rows=graph_rows,
                simplicial_rows=simplicial_rows,
                cell_rows=cell_rows,
            )
        mus: list[float] = []
        for rk in band_keys:
            t = stats.get((rk, ds_key))
            if t and _finite(t.get("test_mean")):
                mus.append(float(t["test_mean"]))
        if not mus:
            return _latex_cell_body(
                mu, sd, se, is_best=False, blue_tie=False, decimals=decimals, scale=1.0
            )

        best_val = min(mus)
        is_best = _tol_eq(mu, best_val)

        ref_mu, ref_se = best_val, 0.0
        for rk in band_keys:
            t = stats.get((rk, ds_key))
            if not t or not _finite(t.get("test_mean")):
                continue
            if not _tol_eq(float(t["test_mean"]), best_val):
                continue
            ref_mu = float(t["test_mean"])
            ns_ref = pd.to_numeric(t.get("n_seeds", 0), errors="coerce")
            n_ref = int(ns_ref) if pd.notna(ns_ref) else 0
            ref_se = _sem(
                float(t["test_std"]) if _finite(t.get("test_std")) else 0.0,
                n_ref,
            )
            break

        blue = not is_best and _not_sig_diff_from_best(mu, se, ref_mu, ref_se)
        return _latex_cell_body(
            mu, sd, se, is_best=is_best, blue_tie=blue, decimals=decimals, scale=1.0
        )

    lines: list[str] = []
    lines.append(
        "% --- Requires: \\usepackage{booktabs,multirow,adjustbox,graphicx,xcolor,colortbl}"
    )
    lines.append(
        "\\definecolor{stdblue}{HTML}{C9DAF8}% same swatch as table_generator (non-significant vs column best)"
    )
    lines.append("\\definecolor{bestgray}{HTML}{D9D9D9}")
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{width=1.\\textwidth}")
    lines.append("\\renewcommand{\\arraystretch}{1.4}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    if n_d > 0 and group_ranges:
        multicols = []
        cmid_parts = []
        for title, i0, i1 in group_ranges:
            span = i1 - i0 + 1
            multicols.append(f"\\multicolumn{{{span}}}{{c}}{{\\mbox{{{title}}}}}")
            cmid_parts.append(f"\\cmidrule(lr){{{3 + i0}-{3 + i1}}}")
        lines.append("  &  & " + " & ".join(multicols) + " \\\\")
        lines.append(" ".join(cmid_parts))

    hdr = " & \\textbf{Model}"
    for _p, h in dataset_specs:
        hdr += f" & \\scriptsize {h}"
    hdr += " \\\\"
    lines.append(hdr)
    lines.append("\\midrule")

    def emit_block(rotate: str, rows: list[tuple[str, str]]) -> None:
        n_r = len(rows)
        rk0, lab0 = rows[0]
        row = (
            f"\\multirow{{{n_r}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{rotate}}}}}}} "
            f"& {lab0}"
        )
        for ds_path, _h in dataset_specs:
            row += " & " + cell_time_colored(rk0, ds_path)
        lines.append(row + " \\\\")
        for rk, lab in rows[1:]:
            row = f"& {lab}"
            for ds_path, _h in dataset_specs:
                row += " & " + cell_time_colored(rk, ds_path)
            lines.append(row + " \\\\")

    emit_block("Graph", graph_rows)
    lines.append("\\midrule")
    emit_block("Simplicial", simplicial_rows)
    lines.append("\\midrule")
    emit_block("Cell", cell_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="W&B best_runs_rerun: export, seed-aggregate, emit LaTeX tables (perf + times)."
    )
    p.add_argument("--entity", default=DEFAULT_WANDB_ENTITY, help="W&B entity")
    p.add_argument("--project", default=DEFAULT_WANDB_PROJECT, help="Single W&B project for all reruns")
    p.add_argument(
        "--run-state",
        default="finished",
        metavar="STATE",
        help='W&B run filter: "finished" (default), "running", "all", …',
    )
    p.add_argument("--quiet", action="store_true", help="Less console output")
    p.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=DEFAULT_AGG_CSV,
        help=f"Seed-aggregated CSV (default: {DEFAULT_AGG_CSV})",
    )
    p.add_argument(
        "--write-raw-csv",
        type=Path,
        default=None,
        help="Optional path to write per-run export before aggregation.",
    )
    p.add_argument(
        "--required-seeds",
        type=int,
        default=5,
        metavar="N",
        help="Keep only hyperparameter groups with exactly N raw runs (default: 5). Ignored with --keep-incomplete-seeds.",
    )
    p.add_argument(
        "--keep-incomplete-seeds",
        action="store_true",
        help="Do not filter on n_seeds.",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help=f"Directory for emitted .tex files (default: {TABLES_DIR})",
    )
    p.add_argument("--decimals", type=int, default=2, help="Decimal places in time tables (default: 2)")
    p.add_argument(
        "--no-scale-fractions",
        action="store_true",
        help="Performance table: do not scale accuracy-like metrics by 100.",
    )
    args = p.parse_args()

    run_state: str | None
    if str(args.run_state).lower() == "all":
        run_state = None
    else:
        run_state = str(args.run_state)

    if not args.keep_incomplete_seeds and int(args.required_seeds) < 1:
        p.error("--required-seeds must be >= 1 unless --keep-incomplete-seeds is set.")

    verbose = not args.quiet
    print(f"Entity: {args.entity}  project: {args.project!r}")
    rows = collect_runs_single_project(
        args.entity, args.project, run_state=run_state, verbose=verbose
    )
    df_raw = dataframe_from_rows(rows)
    if args.write_raw_csv is not None:
        args.write_raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(args.write_raw_csv, index=False)
        print(f"Wrote raw export: {args.write_raw_csv} ({len(df_raw)} rows)")

    sm_cols = _summary_metric_columns_for_rerun_export(df_raw)
    agg, silent = aggregate_wandb_export_by_seed(df_raw, summary_metric_columns=sm_cols)
    report = build_seed_bucket_report(agg)
    req = None if args.keep_incomplete_seeds else int(args.required_seeds)
    if req is not None:
        agg = filter_aggregated_to_required_n_seeds(agg, req)
    agg = agg.fillna("")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output_csv, index=False)
    print(f"Wrote seed-aggregated CSV: {args.output_csv} ({len(agg)} rows x {len(agg.columns)} cols)")

    _print_silent_failure_report(silent)
    _print_seed_bucket_report(report, required_n_seeds=req)

    # --- LaTeX tables (same dataset / row layout as table_generator) ---
    base_specs = expand_mantra_betti_specs(_specs_from_loader_paths())
    groups = partition_specs_graph_simplicial(base_specs)
    stats_perf = collect_winner_test_by_submodel(agg)
    simplicial_rows_sub = simplicial_submodel_table_rows()
    cell_rows_sub = cell_submodel_table_rows()
    graph_rows_compact = [
        ("graph/gcn", "GCN"),
        ("graph/gat", "GAT"),
        ("graph/gin", "GIN"),
    ]
    stats_perf_compact = collapse_gnn_submodel_rows_to_base(stats_perf)
    tex_perf_compact = build_latex_table(
        stats_perf_compact,
        column_groups=groups,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        decimals=args.decimals,
        scale_fraction_metrics=not args.no_scale_fractions,
        label="tbl:best_rerun_perf",
        caption="Rerun test, mean $\\pm$ std over seeds (bold $=$ best; blue $=$ not worse, 95\\,\\%).",
    )

    mean_r = f"{SUMMARY_RUNTIME}__mean"
    std_r = f"{SUMMARY_RUNTIME}__std"
    stats_rt_sub: dict[tuple[str, str], dict[str, Any]] = {}
    if mean_r in agg.columns:
        stats_rt_sub = collect_runtime_stats_from_agg_winners(agg, mean_col=mean_r, std_col=std_r if std_r in agg.columns else None)
    stats_rt_compact = collapse_gnn_submodel_rows_to_base(stats_rt_sub) if stats_rt_sub else {}

    cap_rt = (
        "End-to-end time (without preprocessing) in seconds (mean $\\pm$ std over seeds). "
        "\\textbf{Bold}: lowest mean per dataset (column) and domain type (graph, simplicial or cell). "
        "\\protect\\colorbox{stdblue}{blue}: not significantly slower than that minimum (95\\,\\%, two-sided $Z$)."
    )
    tex_rt_compact = build_rerun_metric_table_plain(
        stats_rt_compact,
        column_groups=groups,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        caption=cap_rt,
        label="tbl:best_rerun_runtime",
        decimals=args.decimals,
    )

    stats_ep_sub: dict[tuple[str, str], dict[str, Any]] = {}
    if SUMMARY_EPOCH_MEAN in df_raw.columns:
        stats_ep_sub = collect_epoch_time_stats_min_timing_std(
            df_raw,
            mean_col=SUMMARY_EPOCH_MEAN,
            std_col=SUMMARY_EPOCH_STD if SUMMARY_EPOCH_STD in df_raw.columns else "",
        )
        # Match compact GNN collapse to the performance table (val from seed-aggregated picks).
        for k in list(stats_ep_sub.keys()):
            if k in stats_perf:
                stats_ep_sub[k]["val_mean"] = float(stats_perf[k]["val_mean"])
    stats_ep_compact = collapse_gnn_submodel_rows_to_base(stats_ep_sub) if stats_ep_sub else {}

    cap_ep = (
        "Train seconds per epoch (mean $\\pm$ std) "
        "\\textbf{Bold}: lowest mean per dataset (column) within domain (graph, simplicial or cell). "
        "\\protect\\colorbox{stdblue}{blue}: not significantly slower than that minimum (95\\,\\%, two-sided $Z$)."
    )
    tex_ep_compact = build_rerun_metric_table_plain(
        stats_ep_compact,
        column_groups=groups,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        caption=cap_ep,
        label="tbl:best_rerun_epoch_time",
        decimals=args.decimals,
        comparison_scope="domain",
    )
    cap_ep_global = (
        "Same per-epoch times as the domain-banded table. "
        "\\textbf{Bold}: lowest mean per dataset (column) across \\textbf{all} models (not per domain). "
        "\\protect\\colorbox{stdblue}{blue}: not significantly slower than that column minimum "
        "(95\\,\\%, two-sided $Z$)."
    )
    tex_ep_global = build_rerun_metric_table_plain(
        stats_ep_compact,
        column_groups=groups,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        caption=cap_ep_global,
        label="tbl:best_rerun_epoch_time_global_min",
        decimals=args.decimals,
        comparison_scope="global",
    )

    out_dir = Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rerun_main_table.tex": tex_perf_compact,
        "rerun_time_runtime.tex": tex_rt_compact,
        "rerun_time_train_epoch.tex": tex_ep_compact,
        "rerun_time_train_epoch_bold_global.tex": tex_ep_global,
    }
    for name, body in paths.items():
        path = out_dir / name
        path.write_text(body, encoding="utf-8")
        print(f"Wrote {path}")

    if not stats_rt_sub:
        print(
            "Note: no wall-clock timing in aggregated CSV — expected ``summary_Runtime`` "
            "from W&B ``_runtime`` (or ``Runtime``) in run summary / ``_runtime`` in config."
        )
    if not stats_ep_sub:
        print(
            f"Note: no column {SUMMARY_EPOCH_MEAN!r} in raw export — "
            "``PipelineTimer`` stores AvgTime in run **config** via ``log_hyperparams``; "
            "if still empty, training may have ended before enough epochs to log averages."
        )


if __name__ == "__main__":
    main()
