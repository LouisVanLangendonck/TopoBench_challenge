#!/usr/bin/env python3
r"""
Fetch finished runs from the **preprocessing-time ablation** W&B project
(``hopse_preprocessing_time_ablation``; see ``scripts/hopse_preprocessing_time_ablation.sh``),
read scalar ``preprocessor_time`` (exported as ``summary_preprocessor_time``), and emit:

1. A flat CSV keyed by model branch, neighborhood alias, and dataset.
2. A LaTeX table: **Cell** / **Simplicial** as vertical ``\rotatebox`` band labels (compact-table
   style), then **model** (HOPSE-M-PE / HOPSE-M-F / HOPSE-G) and **neighborhood** (five rows per
   model), then dataset columns (Graph / Simplicial bands; no performance arrows). Light rules
   separate model blocks. **bestgray** + bold = fastest time in each dataset column **within that
   domain** (Cell vs Simplicial).

Run from repo root::

    python scripts/hopse_plotting/process_preprocess_ablation.py
    python scripts/hopse_plotting/process_preprocess_ablation.py --project hopse_preprocessing_time_ablation

Requires ``wandb`` and ``WANDB_API_KEY`` (or ``wandb login``).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    CSV_DIR,
    MODEL_PREPROC_ENCODINGS,
    SUMMARY_COLUMN_PREFIX,
    TABLES_DIR,
    dataframe_from_rows,
    hopse_m_encoding_f_vs_pe_sub_id,
    iter_runs,
    run_to_row,
)

DEFAULT_WANDB_ENTITY = "gbg141-hopse"
DEFAULT_WANDB_PROJECT = "hopse_preprocessing_time_ablation"

DEFAULT_EXPORT_CSV = CSV_DIR / "preprocess_ablation_wandb_export.csv"
DEFAULT_TEX = TABLES_DIR / "preprocess_ablation_preprocessor_time.tex"

SUMMARY_PREPROC = f"{SUMMARY_COLUMN_PREFIX}preprocessor_time"

# LaTeX row order: domain block → model (W&B tag still cell_m_pe / …; labels are HOPSE-M-PE / …).
DOMAIN_BLOCKS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Cell",
        [
            ("cell_m_pe", r"\textbf{HOPSE-M-PE}"),
            ("cell_m_fe", r"\textbf{HOPSE-M-F}"),
            ("cell_g", r"\textbf{HOPSE-G}"),
        ],
    ),
    (
        "Simplicial",
        [
            ("sim_m_pe", r"\textbf{HOPSE-M-PE}"),
            ("sim_m_fe", r"\textbf{HOPSE-M-F}"),
            ("sim_g", r"\textbf{HOPSE-G}"),
        ],
    ),
]

ALL_MODEL_TAGS: list[str] = [tag for _, mlist in DOMAIN_BLOCKS for tag, _ in mlist]

# Column order / headers aligned with leaderboard tables (no ↑/↓ — preprocessing time only).
TABLE_COLUMN_SPECS: list[tuple[str, str]] = [
    ("graph/MUTAG", r"MUTAG"),
    ("graph/PROTEINS", r"PROTEINS"),
    ("graph/NCI1", r"NCI1"),
    ("graph/NCI109", r"NCI109"),
    ("graph/BBB_Martins", r"BBB"),
    ("graph/CYP3A4_Veith", r"CYP3A4"),
    ("graph/Clearance_Hepatocyte_AZ", r"Cl.Hep."),
    ("graph/Caco2_Wang", r"Caco2"),
    ("graph/ZINC", r"ZINC"),
    ("simplicial/mantra_name", r"NAME"),
    ("simplicial/mantra_orientation", r"ORIENT"),
    ("simplicial/mantra_betti_numbers", r"$\beta_1$"),
    ("simplicial/mantra_betti_numbers", r"$\beta_2$"),
]

# Unique hydra paths (for W&B name parsing); order follows first occurrence in ``TABLE_COLUMN_SPECS``.
UNIQUE_SWEEP_DATASETS: list[str] = list(dict.fromkeys(h for h, _ in TABLE_COLUMN_SPECS))

N_GRAPH_COLS = sum(1 for h, _ in TABLE_COLUMN_SPECS if h.startswith("graph/"))
N_SIM_COLS = len(TABLE_COLUMN_SPECS) - N_GRAPH_COLS

NB_ORDER: list[tuple[str, str]] = [
    ("adj1", r"$\mathcal{A}_0$"),
    ("adj2", r"$\mathcal{A}_1$"),
    ("adj3", r"$\mathcal{A}_2$"),
    ("inc1", r"$\mathcal{I}_0$"),
    ("inc2", r"$\mathcal{I}_1$"),
]

_RUN_NAME_RE = re.compile(
    r"^(?P<tag>cell_m_pe|cell_m_fe|sim_m_pe|sim_m_fe|cell_g|sim_g)"
    r"_(?P<ds_und>.+)_N(?P<nb>adj[123]|inc[12])_seed(?P<seed>\d+)$"
)


def _hydra_dataset_from_underscored(ds_und: str) -> str | None:
    """Invert ``dataset.replace('/', '_')`` from the sweep script (match known list)."""
    for full in UNIQUE_SWEEP_DATASETS:
        if full.replace("/", "_") == ds_und:
            return full
    return None


def _parse_run_name(run_name: str) -> tuple[str, str, str] | None:
    """
    Map W&B run name to (model_tag, hydra_dataset, nb_alias).
    Names follow ``hopse_preprocessing_time_ablation.sh``:
    ``{tag}_{dataset.replace('/', '_')}_N{nb}_seed{seed}``.
    """
    m = _RUN_NAME_RE.match(str(run_name or "").strip())
    if not m:
        return None
    tag = m.group("tag")
    ds_und = m.group("ds_und")
    nb = m.group("nb")
    hydra_ds = _hydra_dataset_from_underscored(ds_und)
    if hydra_ds is None:
        return None
    return tag, hydra_ds, nb


def _model_tag_from_config(row: dict[str, Any]) -> str | None:
    """Recover ablation branch from Hydra config when run name does not match."""
    model = str(row.get("model") or "").strip()
    if model == "cell/hopse_g":
        return "cell_g"
    if model == "simplicial/hopse_g":
        return "sim_g"
    if model == "cell/hopse_m":
        enc = row.get(MODEL_PREPROC_ENCODINGS, "")
        return "cell_m_fe" if hopse_m_encoding_f_vs_pe_sub_id(enc) == "f" else "cell_m_pe"
    if model == "simplicial/hopse_m":
        enc = row.get(MODEL_PREPROC_ENCODINGS, "")
        return "sim_m_fe" if hopse_m_encoding_f_vs_pe_sub_id(enc) == "f" else "sim_m_pe"
    return None


def _nb_alias_from_neighborhoods_cfg(val: Any) -> str | None:
    """Map logged ``model.preprocessing_params.neighborhoods`` string to adj1..inc2."""
    s = str(val if val is not None else "").strip()
    if not s:
        return None
    canon = re.sub(r"\s+", "", s)
    signatures: list[tuple[str, str]] = [
        ("adj1", "[up_adjacency-0]"),
        ("adj2", "[up_adjacency-0,2-up_adjacency-0]"),
        (
            "adj3",
            "[up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,"
            "down_adjacency-2,2-down_adjacency-2]",
        ),
        ("inc1", "[up_incidence-0,2-up_incidence-0]"),
        (
            "inc2",
            "[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,"
            "down_incidence-2,2-down_incidence-2]",
        ),
    ]
    for alias, sig in signatures:
        if canon == re.sub(r"\s+", "", sig):
            return alias
    return None


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
        rows.append(run_to_row(entity=entity, project=project, run=run))
        count += 1
        if verbose and count % 250 == 0:
            print(f"    … {count} run(s) so far", flush=True)
    if verbose:
        print(f"    -> {count} run(s)", flush=True)
    return rows


def _to_float(cell: Any) -> float | None:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    x = pd.to_numeric(str(cell).strip(), errors="coerce")
    if pd.isna(x):
        return None
    f = float(x)
    if not math.isfinite(f):
        return None
    return f


def rows_to_long_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return long-form (model_tag, neighborhood, dataset, time_sec) and parse warnings."""
    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    if SUMMARY_PREPROC not in raw.columns:
        warnings.append(f"Column {SUMMARY_PREPROC!r} missing — nothing to aggregate.")

    for idx, row in raw.iterrows():
        run_name = str(row.get("identifiers_run_name") or "").strip()
        parsed = _parse_run_name(run_name)
        if parsed:
            tag, dataset, nb = parsed
        else:
            tag = _model_tag_from_config(row)
            dataset = str(row.get("dataset") or "").strip()
            nb = _nb_alias_from_neighborhoods_cfg(row.get("model.preprocessing_params.neighborhoods"))
            if tag is None or not nb:
                warnings.append(f"Row {idx}: could not parse model/neighborhood (run_name={run_name!r}).")
                continue

        tcell = row.get(SUMMARY_PREPROC) if SUMMARY_PREPROC in raw.columns else None
        tsec = _to_float(tcell)
        if tsec is None:
            warnings.append(f"Row {idx} ({run_name}): missing or non-finite {SUMMARY_PREPROC}.")
            continue

        records.append(
            {
                "model_tag": tag,
                "neighborhood": nb,
                "dataset": dataset,
                "preprocessor_time_sec": tsec,
                "identifiers_run_name": run_name,
                "identifiers_run_id": row.get("identifiers_run_id"),
            }
        )

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        return long_df, warnings

    dup = long_df.duplicated(subset=["model_tag", "neighborhood", "dataset"], keep=False)
    if dup.any():
        n = int(dup.sum())
        warnings.append(
            f"Found {n} row(s) with duplicate (model, neighborhood, dataset); keeping max time (one row each)."
        )
        long_df = long_df.sort_values("preprocessor_time_sec", ascending=False)
        long_df = long_df.drop_duplicates(subset=["model_tag", "neighborhood", "dataset"], keep="first")

    return long_df, warnings


def _tol_eq(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * (1.0 + abs(b))


def build_latex_table(
    pivot: dict[tuple[str, str], dict[str, float]],
    *,
    decimals: int,
    label: str,
    caption: str,
) -> str:
    """``pivot`` maps (model_tag, nb_alias) -> {hydra_dataset: time_sec}."""
    n_ds = len(TABLE_COLUMN_SPECS)
    col_last = 3 + n_ds
    colspec = "@{}lll" + "c" * n_ds + "@{}"

    # Minimum per (domain, dataset column) — bold only compares rows within Cell or Simplicial.
    col_mins_by_domain: dict[tuple[str, int], float] = {}
    for domain_name, model_list in DOMAIN_BLOCKS:
        tags_in_domain = [mk for mk, _ in model_list]
        for ci, (hydra_ds, _) in enumerate(TABLE_COLUMN_SPECS):
            vals: list[float] = []
            for mk in tags_in_domain:
                for nk, _ in NB_ORDER:
                    v = pivot.get((mk, nk), {}).get(hydra_ds)
                    if v is not None and math.isfinite(v):
                        vals.append(float(v))
            col_mins_by_domain[(domain_name, ci)] = min(vals) if vals else float("nan")

    def cell_tex(model_tag: str, nb_key: str, col_idx: int, domain_name: str) -> str:
        hydra_ds, _ = TABLE_COLUMN_SPECS[col_idx]
        v = pivot.get((model_tag, nb_key), {}).get(hydra_ds)
        if v is None or not math.isfinite(v):
            return r"{\scriptsize ---}"
        body = f"{float(v):.{decimals}f}"
        cm = col_mins_by_domain.get((domain_name, col_idx), float("nan"))
        if math.isfinite(cm) and _tol_eq(float(v), cm):
            return f"{{\\cellcolor{{bestgray}}{{\\scriptsize\\textbf{{{body}}}}}}}"
        return f"{{\\scriptsize {body}}}"

    c0 = 4
    c_graph_end = c0 + N_GRAPH_COLS - 1
    c_sim_end = c0 + n_ds - 1

    lines: list[str] = []
    lines.append("% --- Requires: \\usepackage{booktabs,multirow,adjustbox,graphicx,xcolor,colortbl}")
    lines.append("\\definecolor{bestgray}{HTML}{D9D9D9}")
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{width=1.\\textwidth}")
    lines.append("\\renewcommand{\\arraystretch}{1.4}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(
        " & & & "
        f"\\multicolumn{{{N_GRAPH_COLS}}}{{c}}{{\\mbox{{Graph}}}} & "
        f"\\multicolumn{{{N_SIM_COLS}}}{{c}}{{\\mbox{{Simplicial}}}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{c0}-{c_graph_end}}} \\cmidrule(lr){{{c_graph_end + 1}-{c_sim_end}}}"
    )
    hdr = " & \\textbf{Model} & \\textbf{Neigh.}"
    for _h, tex_h in TABLE_COLUMN_SPECS:
        hdr += f" & \\scriptsize {tex_h}"
    hdr += " \\\\"
    lines.append(hdr)
    lines.append("\\midrule")

    n_nb = len(NB_ORDER)
    for di, (domain_name, model_list) in enumerate(DOMAIN_BLOCKS):
        n_dom_rows = len(model_list) * n_nb
        for mi, (mk, model_tex) in enumerate(model_list):
            for ni, (nk, ntex) in enumerate(NB_ORDER):
                dom_cell = ""
                if mi == 0 and ni == 0:
                    dom_cell = (
                        f"\\multirow{{{n_dom_rows}}}{{*}}{{"
                        f"\\rotatebox[origin=c]{{90}}{{\\textbf{{{domain_name}}}}}"
                        f"}}"
                    )
                mod_cell = ""
                if ni == 0:
                    mod_cell = f"\\multirow{{{n_nb}}}{{*}}{{{model_tex}}}"
                row = f"{dom_cell} & {mod_cell} & {ntex}"
                for ci in range(n_ds):
                    row += " & " + cell_tex(mk, nk, ci, domain_name)
                lines.append(row + " \\\\")
                # Light separator after each model block (not through the domain \multirow column).
                if ni == n_nb - 1 and mi < len(model_list) - 1:
                    lines.append(
                        f"\\arrayrulecolor{{black!30}}\\cmidrule[0.25pt](lr){{2-{col_last}}}"
                        f"\\arrayrulecolor{{black}}"
                    )
        if di < len(DOMAIN_BLOCKS) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="W&B preprocessing ablation: export preprocessor_time + LaTeX table."
    )
    p.add_argument("--entity", default=DEFAULT_WANDB_ENTITY, help="W&B entity")
    p.add_argument("--project", default=DEFAULT_WANDB_PROJECT, help="W&B project (single project)")
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
        default=DEFAULT_EXPORT_CSV,
        help=f"Long-form CSV (default: {DEFAULT_EXPORT_CSV})",
    )
    p.add_argument(
        "--output-tex",
        type=Path,
        default=DEFAULT_TEX,
        help=f"LaTeX table path (default: {DEFAULT_TEX})",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Optional directory prefix: writes ``<dir>/<output-tex.name>`` when set.",
    )
    p.add_argument("--decimals", type=int, default=2, help="Decimal places (default: 2)")
    p.add_argument(
        "--label",
        default="tbl:preprocess_ablation_preproc_time",
        help="LaTeX \\label{...}",
    )
    args = p.parse_args()

    run_state: str | None
    if str(args.run_state).lower() == "all":
        run_state = None
    else:
        run_state = str(args.run_state)

    verbose = not args.quiet
    print(f"Entity: {args.entity}  project: {args.project!r}")
    rows = collect_runs_single_project(
        args.entity, args.project, run_state=run_state, verbose=verbose
    )
    df_raw = dataframe_from_rows(rows)

    long_df, warns = rows_to_long_dataframe(df_raw)
    for w in warns[:50]:
        print(f"⚠ {w}")
    if len(warns) > 50:
        print(f"⚠ … {len(warns) - 50} further warning(s) omitted.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output_csv, index=False)
    print(f"Wrote long CSV: {args.output_csv} ({len(long_df)} rows)")

    pivot: dict[tuple[str, str], dict[str, float]] = {}
    for _, r in long_df.iterrows():
        mk = str(r["model_tag"])
        nk = str(r["neighborhood"])
        ds = str(r["dataset"])
        pivot.setdefault((mk, nk), {})[ds] = float(r["preprocessor_time_sec"])

    tex_path = args.output_tex
    if args.tables_dir is not None:
        tex_path = Path(args.tables_dir) / args.output_tex.name

    cap = (
        "Preprocessing time in seconds. \\textbf{Bold}: fastest per dataset and domain (cell or simplicial)."
    )
    body = build_latex_table(
        pivot,
        decimals=int(args.decimals),
        label=str(args.label),
        caption=cap,
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(body, encoding="utf-8")
    print(f"Wrote LaTeX: {tex_path}")

    expected = len(ALL_MODEL_TAGS) * len(NB_ORDER) * len(UNIQUE_SWEEP_DATASETS)
    got = len(long_df)
    if got < expected:
        print(f"Note: {got} / {expected} expected rows in full grid (missing runs or parse failures).")


if __name__ == "__main__":
    main()
