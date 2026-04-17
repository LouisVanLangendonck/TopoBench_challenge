#!/usr/bin/env python3
"""
Build a LaTeX table (booktabs / multirow / cell colors) from a **seed-aggregated**
W&B CSV: for each (model, dataset), pick the hyperparameter row with best **validation**
mean — same implementation as ``main_plot`` / ``collapse_aggregated_wandb_csv`` via
``utils.iter_best_val_group_picks`` (default ``group_cols``: ``model``, ``dataset``;
``monitor_column``: ``dataset.parameters.monitor_metric``), then read **test**
``test_best_rerun`` mean ± std from that row.

The seed-aggregated CSV must include every sweep axis you care about (see
``utils.CONFIG_PARAM_KEYS`` and ``main_loader`` export), or distinct configs can collapse
during aggregation (e.g. missing ``transforms.hopse_encoding.pretrain_model`` for HOPSE_G).

- **bestgray** + bold: best test value in the column (ties share the style).
- **stdblue**: not significantly different from the column best at 95% confidence
  (two-sided Z on independent means in code; SE = seed-agg std / sqrt(n_seeds)).

Model blocks: graph GCN/GAT/GIN; simplicial HOPSE-M, HOPSE-G, TopoTune, SCCNN
(``simplicial/sccnn_custom``); cell HOPSE-M, HOPSE-G, TopoTune, CWN (``cell/cwn``).
**Dataset columns** come from ``DATASETS`` in ``main_loader.py``, reordered so **all graph
columns precede all simplicial**. By default **four** ``.tex`` files are written under ``tables/``:

- Base: ``main_table_all.tex`` / ``main_table_no_transductive.tex`` (one row per model).
- **Submodels**: ``main_table_all_submodels.tex`` / ``main_table_no_transductive_submodels.tex`` —
  GNN rows split by ``transforms`` (empty → plain name; ``combined_fe`` → ``-F``; ``combined_pe`` → ``-PE``);
  HOPSE-M split by ``model.preprocessing_params.encodings`` (**HOPSE-M-F** if HFKE/HKFE appears in the
  cell, else **HOPSE-M-PE**); HOPSE-G and TopoTune unchanged. Best validation row is chosen **within**
  each sub-row group. Use ``--skip-submodel-tables`` to emit only the base pair.

Usage::

    python scripts/hopse_plotting/table_generator.py
    python scripts/hopse_plotting/table_generator.py -o scripts/hopse_plotting/tables/main_table_all.tex \\
        --output-without-transductive scripts/hopse_plotting/tables/main_table_no_transductive.tex
    python scripts/hopse_plotting/table_generator.py --stdout
    python scripts/hopse_plotting/table_generator.py --skip-submodel-tables
    python scripts/hopse_plotting/table_generator.py --group-by model dataset
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from main_loader import DATASETS as LOADER_DATASETS
from utils import (
    DEFAULT_AGGREGATED_EXPORT_CSV,
    MONITOR_METRIC_COLUMN,
    TABLES_DIR,
    _first_existing_column,
    _paired_std_from_mean,
    _test_mean_columns_for_tail,
    hydra_dataset_key_from_loader_identity,
    iter_best_val_group_picks,
    optimization_mode_for_metric_tail,
    load_wandb_export_csv,
    safe_filename_token,
)

DEFAULT_LATEX_TABLE_TEX = TABLES_DIR / "main_table_all.tex"
DEFAULT_LATEX_TABLE_TEX_NO_TRANS = TABLES_DIR / "main_table_no_transductive.tex"
DEFAULT_LATEX_TABLE_TEX_SUBMODELS = TABLES_DIR / "main_table_all_submodels.tex"
DEFAULT_LATEX_TABLE_TEX_NO_TRANS_SUBMODELS = TABLES_DIR / "main_table_no_transductive_submodels.tex"

COL_TRANSFORMS = "transforms"
COL_PREPROC_ENC = "model.preprocessing_params.encodings"

GRAPH_MPNN = frozenset({"graph/gcn", "graph/gat", "graph/gin"})
MODEL_HOPSE_M = frozenset({"simplicial/hopse_m", "cell/hopse_m"})
MODEL_HOPSE_G_TOPO = frozenset(
    {
        "simplicial/hopse_g",
        "cell/hopse_g",
        "simplicial/topotune",
        "cell/topotune",
    }
)

# Planetoid cocitation configs (transductive); must match loader ``graph/cocitation_*`` paths.
TRANSDUCTIVE_GRAPH_PATHS: tuple[str, ...] = (
    "graph/cocitation_cora",
    "graph/cocitation_citeseer",
    "graph/cocitation_pubmed",
)
TRANSDUCTIVE_GRAPH_SET: frozenset[str] = frozenset(TRANSDUCTIVE_GRAPH_PATHS)

Z_CRIT_95 = 1.959963984540054

# W&B often stores 0–1 fractions; publication tables use 0–100 for these tails.
_DISPLAY_SCALE_100: frozenset[str] = frozenset(
    {"accuracy", "f1", "precision", "recall", "auroc", "roc_auc"}
)


def _display_scale(tail: str) -> float:
    t = (tail or "").strip().lower()
    return 100.0 if t in _DISPLAY_SCALE_100 else 1.0


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


def _parse_dataset_specs(items: list[str]) -> list[tuple[str, str]]:
    """``path:LaTeX header`` or bare ``path`` (basename used as header)."""
    out: list[tuple[str, str]] = []
    for raw in items:
        s = raw.strip()
        if not s:
            continue
        if ":" in s:
            path, hdr = s.split(":", 1)
            out.append((path.strip(), hdr.strip()))
        else:
            base = s.rsplit("/", 1)[-1]
            out.append((s.strip(), base))
    return out


# Short column titles (↑ / ↓ are heuristic for table headers; per-column optimization still comes from data).
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

_DATASET_MIN_ARROW: frozenset[str] = frozenset(
    {
        "graph/Clearance_Hepatocyte_AZ",
        "graph/Caco2_Wang",
        "simplicial/mantra_betti_numbers",
    }
)


def _latex_short_dataset_label(path: str) -> str:
    return _DATASET_COLUMN_LABEL.get(path, path.rsplit("/", 1)[-1].replace("_", r"\_"))


def _auto_header_for_dataset_path(path: str) -> str:
    short = _latex_short_dataset_label(path)
    arr = r"$\downarrow$" if path in _DATASET_MIN_ARROW else r"$\uparrow$"
    return f"{short} ({arr})"


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
    Transductive columns are only Cora / Citeseer / PubMed (``cocitation_*``), in that order.
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
    """Graph then Simplicial; omits cocitation Cora/Citeseer/PubMed. Headers omit ``(inductive)``."""
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


def _is_empty_transforms(val: Any) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    s = str(val).replace("\r", "").strip().lower()
    return s in ("", "nan", "none", "[]", "{}", "null")


def _graph_transforms_sub_id(val: Any) -> str:
    """Bucket GNN rows by ``transforms`` for separate best-val picks."""
    if _is_empty_transforms(val):
        return "base"
    s = str(val).replace("\r", "").strip().lower()
    if s == "combined_fe":
        return "fe"
    if s == "combined_pe":
        return "pe"
    tok = safe_filename_token(s, max_len=48)
    return f"other::{tok}"


def _hopse_m_enc_sub_id(val: Any) -> str:
    """
    HOPSE-M: HFKE (or HKFE as stored in exports) in ``model.preprocessing_params.encodings``
    → ``f`` (display HOPSE-M-F), else ``pe`` (HOPSE-M-PE).
    """
    s = str(val if val is not None else "").replace("\r", "")
    su = s.upper()
    if "HFKE" in su or "HKFE" in su:
        return "f"
    return "pe"


def _assign_sub_id_for_row(model: str, row: pd.Series) -> str:
    m = str(model).strip()
    if m in GRAPH_MPNN:
        tv = row[COL_TRANSFORMS] if COL_TRANSFORMS in row.index else None
        return _graph_transforms_sub_id(tv)
    if m in MODEL_HOPSE_M:
        ev = row[COL_PREPROC_ENC] if COL_PREPROC_ENC in row.index else None
        return _hopse_m_enc_sub_id(ev)
    if m in MODEL_HOPSE_G_TOPO:
        return "default"
    return "default"


def dataframe_with_submodel_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    subs: list[str] = []
    for _idx, row in out.iterrows():
        subs.append(_assign_sub_id_for_row(str(row.get("model", "")), row))
    out["_sub_id"] = subs
    return out


def _sort_graph_sub_ids(subs: set[str]) -> list[str]:
    def sk(s: str) -> tuple[int, str]:
        if s == "base":
            return (0, s)
        if s == "fe":
            return (1, s)
        if s == "pe":
            return (2, s)
        return (3, s)

    return sorted(subs, key=sk)


def _latex_graph_sub_row_label(base_short: str, sub_id: str) -> str:
    if sub_id == "base":
        return base_short
    if sub_id == "fe":
        return f"{base_short}-F"
    if sub_id == "pe":
        return f"{base_short}-PE"
    if sub_id.startswith("other::"):
        body = sub_id.split("other::", 1)[1].replace("_", r"\_")
        return f"{base_short}-\\texttt{{{body}}}"
    body = sub_id.replace("_", r"\_")
    return f"{base_short}-\\texttt{{{body}}}"


def graph_submodel_table_rows(stats: dict[tuple[str, str], Any]) -> list[tuple[str, str]]:
    """(stats_row_key, LaTeX label) for GCN/GAT/GIN sub-rows."""
    seen_keys = {k[0] for k in stats}
    templates = [("graph/gcn", "GCN"), ("graph/gat", "GAT"), ("graph/gin", "GIN")]
    rows: list[tuple[str, str]] = []
    for mid, lab in templates:
        subs = {rk.split("|", 1)[-1] for rk in seen_keys if rk.startswith(mid + "|")}
        if not subs:
            subs = {"base"}
        for sub in _sort_graph_sub_ids(subs):
            rows.append((f"{mid}|{sub}", _latex_graph_sub_row_label(lab, sub)))
    return rows


def simplicial_submodel_table_rows() -> list[tuple[str, str]]:
    return [
        (r"simplicial/hopse_m|f", r"\textbf{HOPSE-M-F} (Our)"),
        (r"simplicial/hopse_m|pe", r"\textbf{HOPSE-M-PE} (Our)"),
        (r"simplicial/hopse_g|default", r"\textbf{HOPSE-G} (Our)"),
        (r"simplicial/topotune|default", "TopoTune"),
        (r"simplicial/sccnn_custom|default", "SCCNN"),
    ]


def cell_submodel_table_rows() -> list[tuple[str, str]]:
    return [
        (r"cell/hopse_m|f", r"\textbf{HOPSE-M-F} (Our)"),
        (r"cell/hopse_m|pe", r"\textbf{HOPSE-M-PE} (Our)"),
        (r"cell/hopse_g|default", r"\textbf{HOPSE-G} (Our)"),
        (r"cell/topotune|default", "TopoTune"),
        (r"cell/cwn|default", "CWN"),
    ]


def collect_winner_test_by_model_dataset(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("model", "dataset"),
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    (model, dataset_canon) -> {test_mean, test_std, n_seeds, tail, mode, monitor_raw}.

    ``group_cols`` must include ``model`` and ``dataset`` (same contract as
    ``main_plot --group-by`` / ``collapse_aggregated_wandb_by_best_val``).
    """
    if "model" not in group_cols or "dataset" not in group_cols:
        raise ValueError("collect_winner_test_by_model_dataset: group_cols must include 'model' and 'dataset'")
    colset = set(df.columns)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for keys, pick_idx, monitor_val, tail in iter_best_val_group_picks(
        df, group_cols=list(group_cols), monitor_column=MONITOR_METRIC_COLUMN
    ):
        gk = keys if isinstance(keys, tuple) else (keys,)
        if len(gk) != len(group_cols):
            raise RuntimeError("groupby key length mismatch vs group_cols")
        zd = dict(zip(group_cols, gk, strict=True))
        model = str(zd["model"]).strip()
        dataset_raw = str(zd["dataset"]).strip()
        dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
        w = df.loc[pick_idx]
        mode: Literal["max", "min"] = optimization_mode_for_metric_tail(tail) if tail else "max"
        test_src = _first_existing_column(_test_mean_columns_for_tail(tail), colset)
        te_std = _paired_std_from_mean(test_src, colset) if test_src else None
        mu = pd.to_numeric(w.get(test_src), errors="coerce") if test_src else float("nan")
        sd = pd.to_numeric(w.get(te_std), errors="coerce") if te_std else float("nan")
        n_raw = w.get("n_seeds", float("nan"))
        n = int(pd.to_numeric(n_raw, errors="coerce")) if _finite(n_raw) else 0
        out[(model, dataset)] = {
            "test_mean": float(mu) if pd.notna(mu) else float("nan"),
            "test_std": float(sd) if pd.notna(sd) else float("nan"),
            "n_seeds": max(n, 0),
            "tail": tail,
            "mode": mode,
            "monitor_raw": str(monitor_val).strip(),
        }
    return out


def collect_winner_test_by_submodel(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Like ``collect_winner_test_by_model_dataset`` but groups by (model, dataset, _sub_id).

    Row keys in the returned map are ``f"{model}|{sub_id}"`` where ``sub_id`` comes from
    ``transforms`` (GNN) or ``model.preprocessing_params.encodings`` (HOPSE-M), or
    ``default`` for HOPSE-G / TopoTune.
    """
    work = dataframe_with_submodel_id(df)
    colset = set(work.columns)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    gc = ["model", "dataset", "_sub_id"]
    for keys, pick_idx, monitor_val, tail in iter_best_val_group_picks(
        work, group_cols=gc, monitor_column=MONITOR_METRIC_COLUMN
    ):
        model = str(keys[0]).strip()
        dataset_raw = str(keys[1]).strip()
        sub_id = str(keys[2]).strip()
        dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
        row_key = f"{model}|{sub_id}"
        w = work.loc[pick_idx]
        mode: Literal["max", "min"] = optimization_mode_for_metric_tail(tail) if tail else "max"
        test_src = _first_existing_column(_test_mean_columns_for_tail(tail), colset)
        te_std = _paired_std_from_mean(test_src, colset) if test_src else None
        mu = pd.to_numeric(w.get(test_src), errors="coerce") if test_src else float("nan")
        sd = pd.to_numeric(w.get(te_std), errors="coerce") if te_std else float("nan")
        n_raw = w.get("n_seeds", float("nan"))
        n = int(pd.to_numeric(n_raw, errors="coerce")) if _finite(n_raw) else 0
        out[(row_key, dataset)] = {
            "test_mean": float(mu) if pd.notna(mu) else float("nan"),
            "test_std": float(sd) if pd.notna(sd) else float("nan"),
            "n_seeds": max(n, 0),
            "tail": tail,
            "mode": mode,
            "monitor_raw": str(monitor_val).strip(),
        }
    return out


def _fmt_cell(mu: float, sd: float, *, decimals: int = 2, scale: float = 1.0) -> str:
    if not _finite(mu):
        return "-"
    mu *= scale
    sd = sd * scale if _finite(sd) else float("nan")
    # \pm must be in math mode (text-mode triggers "Missing $ inserted").
    if _finite(sd):
        return f"${mu:.{decimals}f} \\pm {sd:.{decimals}f}$"
    return f"${mu:.{decimals}f}$"


def _latex_cell_body(
    mu: float,
    sd: float,
    se: float,
    *,
    is_best: bool,
    blue_tie: bool,
    decimals: int,
    scale: float,
) -> str:
    body = _fmt_cell(mu, sd, decimals=decimals, scale=scale)
    if body == "-":
        return "-"
    inner = f"{{\\scriptsize {body}}}"
    if is_best:
        # \textbf does not bold math digits; \boldmath applies to the following math.
        return f"{{\\cellcolor{{bestgray}}{{\\scriptsize\\boldmath {body}}}}}"
    if blue_tie:
        return f"\\cellcolor{{stdblue}}{inner}"
    return inner


def build_latex_table(
    stats: dict[tuple[str, str], dict[str, Any]],
    *,
    column_groups: list[tuple[str, list[tuple[str, str]]]],
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
    decimals: int = 2,
    scale_fraction_metrics: bool = True,
    label: str = "tbl:hopse_wandb_graph_trans_ind_sim",
) -> str:
    """
    Return full LaTeX fragment (table env + suggested \\definecolor comments).

    Each of ``graph_rows`` / ``simplicial_rows`` / ``cell_rows`` is
    ``(stats_row_key, latex_model_label)``. Base tables use ``stats_row_key == model``
    (e.g. ``graph/gcn``); submodel tables use keys like ``graph/gcn|fe``.
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

    all_row_keys = [rk for rk, _ in graph_rows + simplicial_rows + cell_rows]

    def cell_colored(row_key: str, ds_path: str) -> str:
        ds_key = hydra_dataset_key_from_loader_identity(ds_path)
        st = stats.get((row_key, ds_key))
        if not st or not _finite(st.get("test_mean", float("nan"))):
            return "-"
        mu = float(st["test_mean"])
        sd = float(st["test_std"]) if _finite(st.get("test_std")) else float("nan")
        se = _sem(sd, int(st.get("n_seeds", 0)))
        mode = st.get("mode", "max")
        tail = str(st.get("tail", ""))
        dsc = _display_scale(tail) if scale_fraction_metrics else 1.0

        mus: list[float] = []
        for rk in all_row_keys:
            t = stats.get((rk, ds_key))
            if t and _finite(t.get("test_mean")):
                mus.append(float(t["test_mean"]))
        if not mus:
            return _latex_cell_body(
                mu, sd, se, is_best=False, blue_tie=False, decimals=decimals, scale=dsc
            )

        best_val = max(mus) if mode == "max" else min(mus)
        is_best = abs(mu - best_val) <= 1e-9 * (1 + abs(best_val))

        ref_mu, ref_se = best_val, 0.0
        for rk in all_row_keys:
            t = stats.get((rk, ds_key))
            if not t or not _finite(t.get("test_mean")):
                continue
            if abs(float(t["test_mean"]) - best_val) > 1e-9 * (1 + abs(best_val)):
                continue
            ref_mu = float(t["test_mean"])
            ref_se = _sem(
                float(t["test_std"]) if _finite(t.get("test_std")) else 0.0,
                int(t.get("n_seeds", 0)),
            )
            break

        blue = not is_best and _not_sig_diff_from_best(mu, se, ref_mu, ref_se)
        return _latex_cell_body(
            mu, sd, se, is_best=is_best, blue_tie=blue, decimals=decimals, scale=dsc
        )

    lines: list[str] = []
    lines.append("% --- Requires: \\usepackage{booktabs,multirow,adjustbox,graphicx,xcolor,colortbl}")
    lines.append(
        "\\definecolor{stdblue}{HTML}{C9DAF8}% same swatch as non-significant cells (tweak to match venue)"
    )
    lines.append("\\definecolor{bestgray}{HTML}{D9D9D9}")
    lines.append("\\begin{table}[t]")
    cap = (
        "Test mean $\\pm$ std over seeds (hyperparameters chosen on validation). "
        "Best mean result per dataset highlighted in \\textbf{Bold}. "
        "Results in \\protect\\colorbox{stdblue}{blue} are not significantly different "
        "from best model (95\\,\\% confidence)."
    )
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{width=1.\\textwidth}")
    # Must be *outside* tabular: a \\renewcommand right after \\begin{tabular} can break the
    # alignment (Misplaced \\cr / \\noalign) when the next row uses \\multicolumn + \\cmidrule.
    lines.append("\\renewcommand{\\arraystretch}{1.4}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    if n_d > 0 and group_ranges:
        multicols = []
        cmid_parts = []
        for title, i0, i1 in group_ranges:
            span = i1 - i0 + 1
            # \\mbox isolates parentheses from babel / chemistry packages that treat "(" specially.
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

    def emit_model_block(rotate: str, rows: list[tuple[str, str]]) -> None:
        n_r = len(rows)
        rk0, lab0 = rows[0]
        row = (
            f"\\multirow{{{n_r}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{rotate}}}}}}} "
            f"& {lab0}"
        )
        for ds_path, _h in dataset_specs:
            row += " & " + cell_colored(rk0, ds_path)
        lines.append(row + " \\\\")
        for rk, lab in rows[1:]:
            row = f"& {lab}"
            for ds_path, _h in dataset_specs:
                row += " & " + cell_colored(rk, ds_path)
            lines.append(row + " \\\\")

    emit_model_block("Graph", graph_rows)
    lines.append("\\midrule")
    emit_model_block("Simplicial", simplicial_rows)
    lines.append("\\midrule")
    emit_model_block("Cell", cell_rows)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Emit LaTeX leaderboard table from seed-aggregated W&B CSV.")
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
        default=DEFAULT_LATEX_TABLE_TEX,
        help=(
            "Output .tex for the three-band table: Graph (transductive), Graph (inductive), "
            f"Simplicial (inductive) (default: {DEFAULT_LATEX_TABLE_TEX})"
        ),
    )
    p.add_argument(
        "--output-without-transductive",
        type=Path,
        default=DEFAULT_LATEX_TABLE_TEX_NO_TRANS,
        help=(
            "Second .tex: no cocitation Cora/Citeseer/PubMed; band titles Graph / Simplicial "
            f"(no '(inductive)') (default: {DEFAULT_LATEX_TABLE_TEX_NO_TRANS})"
        ),
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help=(
            "Print LaTeX to stdout (three-band first, then a comment separator, then two-band "
            "if that version has at least one column)"
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="PATH:HEADER",
        help=(
            "Dataset columns as path or path:LaTeX header. "
            "Default: DATASETS from main_loader.py, reordered to "
            "transductive graph (cocitation cora/citeseer/pubmed) → other graph → simplicial."
        ),
    )
    p.add_argument("--decimals", type=int, default=2, help="Decimal places for numbers (default: 2)")
    p.add_argument(
        "--no-scale-fractions",
        action="store_true",
        help="Do not multiply accuracy/f1/... by 100 for display (W&B is often 0–1).",
    )
    p.add_argument(
        "--skip-submodel-tables",
        action="store_true",
        help="Do not emit submodel-split tables (GNN by transforms; HOPSE-M-F / HOPSE-M-PE).",
    )
    p.add_argument(
        "-o-sub",
        "--output-submodels",
        type=Path,
        default=DEFAULT_LATEX_TABLE_TEX_SUBMODELS,
        help=f"Submodel three-band .tex (default: {DEFAULT_LATEX_TABLE_TEX_SUBMODELS})",
    )
    p.add_argument(
        "--output-without-transductive-submodels",
        type=Path,
        default=DEFAULT_LATEX_TABLE_TEX_NO_TRANS_SUBMODELS,
        help=(
            "Submodel two-band .tex (no cocitation trio) "
            f"(default: {DEFAULT_LATEX_TABLE_TEX_NO_TRANS_SUBMODELS})"
        ),
    )
    p.add_argument(
        "--group-by",
        metavar="COL",
        nargs="+",
        default=["model", "dataset"],
        help=(
            "Columns for best-val hyperparameter pick (default: model dataset). "
            "Must include both model and dataset; same meaning as ``main_plot --group-by``."
        ),
    )
    args = p.parse_args()

    group_cols = tuple(args.group_by)
    df = load_wandb_export_csv(args.input)
    stats = collect_winner_test_by_model_dataset(df, group_cols=group_cols)
    stats_sub = (
        collect_winner_test_by_submodel(df) if not args.skip_submodel_tables else {}
    )

    if args.datasets:
        base_specs = _parse_dataset_specs(args.datasets)
    else:
        base_specs = _specs_from_loader_paths()

    groups_three = partition_specs_three_way(base_specs)
    groups_two = partition_specs_two_way_no_transductive(base_specs)
    n_two_cols = sum(len(b) for _, b in groups_two)

    graph_rows_base: list[tuple[str, str]] = [
        ("graph/gcn", "GCN"),
        ("graph/gat", "GAT"),
        ("graph/gin", "GIN"),
    ]
    simplicial_rows_base: list[tuple[str, str]] = [
        ("simplicial/hopse_m", "\\textbf{HOPSE-M} (Our)"),
        ("simplicial/hopse_g", "\\textbf{HOPSE-G} (Our)"),
        ("simplicial/topotune", "TopoTune"),
        ("simplicial/sccnn_custom", "SCCNN"),
    ]
    cell_rows_base: list[tuple[str, str]] = [
        ("cell/hopse_m", "\\textbf{HOPSE-M} (Our)"),
        ("cell/hopse_g", "\\textbf{HOPSE-G} (Our)"),
        ("cell/topotune", "TopoTune"),
        ("cell/cwn", "CWN"),
    ]

    graph_rows_sub = graph_submodel_table_rows(stats_sub)
    simplicial_rows_sub = simplicial_submodel_table_rows()
    cell_rows_sub = cell_submodel_table_rows()

    tex_three = build_latex_table(
        stats,
        column_groups=groups_three,
        graph_rows=graph_rows_base,
        simplicial_rows=simplicial_rows_base,
        cell_rows=cell_rows_base,
        decimals=args.decimals,
        scale_fraction_metrics=not args.no_scale_fractions,
        label="tbl:hopse_wandb_graph_trans_ind_sim",
    )
    tex_two: str | None = None
    if n_two_cols > 0:
        tex_two = build_latex_table(
            stats,
            column_groups=groups_two,
            graph_rows=graph_rows_base,
            simplicial_rows=simplicial_rows_base,
            cell_rows=cell_rows_base,
            decimals=args.decimals,
            scale_fraction_metrics=not args.no_scale_fractions,
            label="tbl:hopse_wandb_graph_ind_sim",
        )

    tex_three_sub: str | None = None
    tex_two_sub: str | None = None
    if not args.skip_submodel_tables:
        tex_three_sub = build_latex_table(
            stats_sub,
            column_groups=groups_three,
            graph_rows=graph_rows_sub,
            simplicial_rows=simplicial_rows_sub,
            cell_rows=cell_rows_sub,
            decimals=args.decimals,
            scale_fraction_metrics=not args.no_scale_fractions,
            label="tbl:hopse_wandb_graph_trans_ind_sim_sub",
        )
        if n_two_cols > 0:
            tex_two_sub = build_latex_table(
                stats_sub,
                column_groups=groups_two,
                graph_rows=graph_rows_sub,
                simplicial_rows=simplicial_rows_sub,
                cell_rows=cell_rows_sub,
                decimals=args.decimals,
                scale_fraction_metrics=not args.no_scale_fractions,
                label="tbl:hopse_wandb_graph_ind_sim_sub",
            )

    if args.stdout:
        sys.stdout.write(tex_three)
        if tex_two is not None:
            sys.stdout.write(
                "\n% --- version without transductive graph (cocitation cora/citeseer/pubmed) ---\n\n"
            )
            sys.stdout.write(tex_two)
        if tex_three_sub is not None:
            sys.stdout.write(
                "\n% --- submodels: GNN by transforms; HOPSE-M-F / HOPSE-M-PE by encodings ---\n\n"
            )
            sys.stdout.write(tex_three_sub)
        if tex_two_sub is not None:
            sys.stdout.write(
                "\n% --- submodels without transductive graph columns ---\n\n"
            )
            sys.stdout.write(tex_two_sub)
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
                "no columns left after dropping transductive graph datasets."
            )
        if tex_three_sub is not None:
            out_s1 = Path(args.output_submodels)
            out_s1.parent.mkdir(parents=True, exist_ok=True)
            out_s1.write_text(tex_three_sub, encoding="utf-8")
            print(f"Wrote {out_s1}")
        if tex_two_sub is not None:
            out_s2 = Path(args.output_without_transductive_submodels)
            out_s2.parent.mkdir(parents=True, exist_ok=True)
            out_s2.write_text(tex_two_sub, encoding="utf-8")
            print(f"Wrote {out_s2}")
        elif not args.skip_submodel_tables and n_two_cols == 0:
            print(
                "Skipped submodel two-band table: no columns left after dropping transductive graph datasets."
            )


if __name__ == "__main__":
    main()
