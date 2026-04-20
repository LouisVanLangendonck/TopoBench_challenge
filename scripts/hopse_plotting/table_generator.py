#!/usr/bin/env python3
"""
Build a LaTeX table (booktabs / multirow / cell colors) from a **seed-aggregated**
W&B CSV: for each (model, dataset), pick the hyperparameter row with best **validation**
mean — same implementation as ``main_plot`` / ``collapse_aggregated_wandb_csv`` via
``utils.iter_best_val_group_picks`` (default ``group_cols``: ``model``, ``dataset``;
``monitor_column``: ``dataset.parameters.monitor_metric``), then read **test**
``test_best_rerun`` mean ± std from that row. **MANTRA Betti numbers** use val **loss**
for the best-hparam pick but **two** table columns (``#f1-1``, ``#f1-2``; ``β₀`` omitted)
with test **F1** and per-column significance (see ``utils.MANTRA_BETTI_*``). Column titles
are ``$\\beta_1$`` / ``$\\beta_2$`` (metric is F1, not named in the header).

The seed-aggregated CSV must include every sweep axis you care about (see
``utils.CONFIG_PARAM_KEYS`` and ``main_loader`` export), or distinct configs can collapse
during aggregation (e.g. missing ``transforms.hopse_encoding.pretrain_model`` for HOPSE_G).

- **bestgray** + bold: best test value in the column (ties share the style).
- **stdblue**: not significantly different from the column best at 95% confidence
  (two-sided Z on independent means in code; SE = seed-agg std / sqrt(n_seeds)).

Model blocks: graph GCN/GAT/GIN; simplicial HOPSE-M, HOPSE-G, TopoTune, SCCNN
(``simplicial/sccnn``; legacy exports may use ``simplicial/sccnn_custom``, merged when reading);
cell HOPSE-M, HOPSE-G, TopoTune, CWN (``cell/cwn``).
**Dataset columns** come from ``DATASETS`` in ``main_loader.py``, reordered so **all graph
columns precede all simplicial**. Transductive cocitation datasets (Cora/Citeseer/PubMed) are
**never** included. Two ``.tex`` files are written:

- **``main_table_all_big.tex``** — full submodel rows: GNN split by ``transforms`` (plain / ``-F`` /
  ``-PE``); HOPSE-M split by encodings (**HOPSE-M-F** vs **HOPSE-M-PE**). Within Simplicial and Cell
  bands, **TopoTune / SCCNN** (resp. **TopoTune / CWN**) appear **above** HOPSE rows.
- **``main_table_all_compact.tex``** — same Simplicial and Cell rows; **one row per GNN backbone**
  (GCN/GAT/GIN) showing the sub-configuration that achieved the **best validation** mean for each
  dataset (test numbers from that winner).

Usage::

    python scripts/hopse_plotting/table_generator.py
    python scripts/hopse_plotting/table_generator.py -o scripts/hopse_plotting/tables/main_table_all_big.tex
    python scripts/hopse_plotting/table_generator.py --stdout
    python scripts/hopse_plotting/table_generator.py --skip-compact
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
    HOPSE_M_MODEL_PATHS,
    MANTRA_BETTI_F1_TAILS,
    MANTRA_BETTI_HYDRA_DATASET,
    MODEL_PREPROC_ENCODINGS,
    MONITOR_METRIC_COLUMN,
    TABLES_DIR,
    _first_existing_column,
    _paired_std_from_mean,
    _test_mean_columns_for_tail,
    _val_mean_columns_for_tail,
    hopse_m_encoding_f_vs_pe_sub_id,
    hydra_dataset_key_from_loader_identity,
    is_mantra_betti_hydra_dataset,
    iter_best_val_group_picks,
    optimization_mode_for_metric_tail,
    load_wandb_export_csv,
    safe_filename_token,
)

DEFAULT_LATEX_TABLE_TEX = TABLES_DIR / "main_table_all_big.tex"
DEFAULT_LATEX_TABLE_COMPACT = TABLES_DIR / "main_table_all_compact.tex"

_GNN_COLLAPSE_BASES: tuple[str, ...] = ("graph/gcn", "graph/gat", "graph/gin")

COL_TRANSFORMS = "transforms"
COL_PREPROC_ENC = MODEL_PREPROC_ENCODINGS

GRAPH_MPNN = frozenset({"graph/gcn", "graph/gat", "graph/gin"})
MODEL_HOPSE_M = HOPSE_M_MODEL_PATHS
MODEL_HOPSE_G_TOPO = frozenset(
    {
        "simplicial/hopse_g",
        "cell/hopse_g",
        "simplicial/topotune",
        "cell/topotune",
    }
)

# Planetoid cocitation (transductive); excluded from tables even if listed in ``main_loader``.
TRANSDUCTIVE_GRAPH_SET: frozenset[str] = frozenset(
    {
        "graph/cocitation_cora",
        "graph/cocitation_citeseer",
        "graph/cocitation_pubmed",
    }
)

# Hydra may expose SCCNN as ``simplicial/sccnn`` (current sweeps) or ``simplicial/sccnn_custom`` (older).
_CANONICAL_SCCNN_MODEL = "simplicial/sccnn"


def _normalize_table_model_id(model: str) -> str:
    m = str(model).strip()
    if m == "simplicial/sccnn_custom":
        return _CANONICAL_SCCNN_MODEL
    return m


Z_CRIT_95 = 1.959963984540054

# W&B often stores 0–1 fractions; publication tables use 0–100 for these tails.
_DISPLAY_SCALE_100: frozenset[str] = frozenset(
    {"accuracy", "f1", "f1-1", "f1-2", "precision", "recall", "auroc", "roc_auc"}
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


def _val_mean_for_pick_row(w: pd.Series, tail: str, colset: set[str]) -> float:
    """Seed-mean validation score used to compare GNN sub-rows (same resolution as ``iter_best_val_group_picks``)."""
    t = (tail or "").strip()
    if not t:
        return float("nan")
    val_src = _first_existing_column(_val_mean_columns_for_tail(t), colset)
    if not val_src:
        return float("nan")
    v = pd.to_numeric(w.get(val_src), errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


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
    }
)


def _latex_short_dataset_label(path: str) -> str:
    if "#" in path:
        base, _, suf = path.partition("#")
        beta_lbl = {
            "f1-1": r"$\beta_1$",
            "f1-2": r"$\beta_2$",
            "f1-0": r"$\beta_0$",
        }.get(suf, suf.replace("_", r"\_"))
        base_l = _DATASET_COLUMN_LABEL.get(base, base.rsplit("/", 1)[-1].replace("_", r"\_"))
        return f"{base_l} {beta_lbl}"
    return _DATASET_COLUMN_LABEL.get(path, path.rsplit("/", 1)[-1].replace("_", r"\_"))


def _auto_header_for_dataset_path(path: str) -> str:
    short = _latex_short_dataset_label(path)
    base_path = path.partition("#")[0] if "#" in path else path
    arr = r"$\downarrow$" if base_path in _DATASET_MIN_ARROW else r"$\uparrow$"
    return f"{short} ({arr})"


def expand_mantra_betti_specs(specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One column per ``β₁`` / ``β₂`` (test F1); ``β₀`` omitted; selection still uses val loss."""
    out: list[tuple[str, str]] = []
    for p, h in specs:
        if is_mantra_betti_hydra_dataset(p) and "#" not in p:
            out.append((f"{MANTRA_BETTI_HYDRA_DATASET}#f1-1", r"$\beta_1$ ($\uparrow$)"))
            out.append((f"{MANTRA_BETTI_HYDRA_DATASET}#f1-2", r"$\beta_2$ ($\uparrow$)"))
        else:
            out.append((p, h))
    return out


def _specs_from_loader_paths() -> list[tuple[str, str]]:
    return [
        (p.strip(), _auto_header_for_dataset_path(p.strip()))
        for p in LOADER_DATASETS
        if p.strip()
    ]


def partition_specs_graph_simplicial(
    specs: list[tuple[str, str]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    **Graph** then **Simplicial** column bands. Drops cocitation Cora/Citeseer/PubMed
    (``graph/cocitation_*``) if present in ``specs``.
    """
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
    """Delegate to ``utils.hopse_m_encoding_f_vs_pe_sub_id`` (HOPSE-M-F vs HOPSE-M-PE)."""
    return hopse_m_encoding_f_vs_pe_sub_id(val)


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
    """TopoTune / SCCNN first; HOPSE variants last."""
    return [
        (r"simplicial/topotune|default", "TopoTune"),
        (r"simplicial/sccnn|default", "SCCNN"),
        (r"simplicial/hopse_m|f", r"\textbf{HOPSE-M-F} (Our)"),
        (r"simplicial/hopse_m|pe", r"\textbf{HOPSE-M-PE} (Our)"),
        (r"simplicial/hopse_g|default", r"\textbf{HOPSE-G} (Our)"),
    ]


def cell_submodel_table_rows() -> list[tuple[str, str]]:
    """TopoTune / CWN first; HOPSE variants last."""
    return [
        (r"cell/topotune|default", "TopoTune"),
        (r"cell/cwn|default", "CWN"),
        (r"cell/hopse_m|f", r"\textbf{HOPSE-M-F} (Our)"),
        (r"cell/hopse_m|pe", r"\textbf{HOPSE-M-PE} (Our)"),
        (r"cell/hopse_g|default", r"\textbf{HOPSE-G} (Our)"),
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
        model = _normalize_table_model_id(str(zd["model"]))
        dataset_raw = str(zd["dataset"]).strip()
        dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
        w = df.loc[pick_idx]
        n_raw = w.get("n_seeds", float("nan"))
        n = int(pd.to_numeric(n_raw, errors="coerce")) if _finite(n_raw) else 0

        if is_mantra_betti_hydra_dataset(dataset_raw):
            for fi_tail in MANTRA_BETTI_F1_TAILS:
                test_src = _first_existing_column(_test_mean_columns_for_tail(fi_tail), colset)
                te_std = _paired_std_from_mean(test_src, colset) if test_src else None
                mu = pd.to_numeric(w.get(test_src), errors="coerce") if test_src else float("nan")
                sd = pd.to_numeric(w.get(te_std), errors="coerce") if te_std else float("nan")
                col_key = f"{MANTRA_BETTI_HYDRA_DATASET}#{fi_tail}"
                out[(model, col_key)] = {
                    "test_mean": float(mu) if pd.notna(mu) else float("nan"),
                    "test_std": float(sd) if pd.notna(sd) else float("nan"),
                    "n_seeds": max(n, 0),
                    "tail": fi_tail,
                    "mode": "max",
                    "monitor_raw": str(monitor_val).strip(),
                }
            continue

        mode: Literal["max", "min"] = optimization_mode_for_metric_tail(tail) if tail else "max"
        test_src = _first_existing_column(_test_mean_columns_for_tail(tail), colset)
        te_std = _paired_std_from_mean(test_src, colset) if test_src else None
        mu = pd.to_numeric(w.get(test_src), errors="coerce") if test_src else float("nan")
        sd = pd.to_numeric(w.get(te_std), errors="coerce") if te_std else float("nan")
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
        model = _normalize_table_model_id(str(keys[0]))
        dataset_raw = str(keys[1]).strip()
        sub_id = str(keys[2]).strip()
        dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
        row_key = f"{model}|{sub_id}"
        w = work.loc[pick_idx]
        n_raw = w.get("n_seeds", float("nan"))
        n = int(pd.to_numeric(n_raw, errors="coerce")) if _finite(n_raw) else 0

        if is_mantra_betti_hydra_dataset(dataset_raw):
            for fi_tail in MANTRA_BETTI_F1_TAILS:
                test_src = _first_existing_column(_test_mean_columns_for_tail(fi_tail), colset)
                te_std = _paired_std_from_mean(test_src, colset) if test_src else None
                mu = pd.to_numeric(w.get(test_src), errors="coerce") if test_src else float("nan")
                sd = pd.to_numeric(w.get(te_std), errors="coerce") if te_std else float("nan")
                col_key = f"{MANTRA_BETTI_HYDRA_DATASET}#{fi_tail}"
                vm = _val_mean_for_pick_row(w, fi_tail, colset)
                out[(row_key, col_key)] = {
                    "test_mean": float(mu) if pd.notna(mu) else float("nan"),
                    "test_std": float(sd) if pd.notna(sd) else float("nan"),
                    "n_seeds": max(n, 0),
                    "tail": fi_tail,
                    "mode": "max",
                    "monitor_raw": str(monitor_val).strip(),
                    "val_mean": vm,
                }
            continue

        mode: Literal["max", "min"] = optimization_mode_for_metric_tail(tail) if tail else "max"
        test_src = _first_existing_column(_test_mean_columns_for_tail(tail), colset)
        te_std = _paired_std_from_mean(test_src, colset) if test_src else None
        mu = pd.to_numeric(w.get(test_src), errors="coerce") if test_src else float("nan")
        sd = pd.to_numeric(w.get(te_std), errors="coerce") if te_std else float("nan")
        vm = _val_mean_for_pick_row(w, tail, colset)
        out[(row_key, dataset)] = {
            "test_mean": float(mu) if pd.notna(mu) else float("nan"),
            "test_std": float(sd) if pd.notna(sd) else float("nan"),
            "n_seeds": max(n, 0),
            "tail": tail,
            "mode": mode,
            "monitor_raw": str(monitor_val).strip(),
            "val_mean": vm,
        }
    return out


def collapse_gnn_submodel_rows_to_base(
    stats_sub: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    One row per GNN backbone (``graph/gcn``, …): for each dataset, keep the sub-row
    (plain / ``-F`` / ``-PE`` / …) whose **validation** mean is best; copy its test stats.
    Non-GNN rows are unchanged.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for k, v in stats_sub.items():
        rk, ds = k
        if not any(rk.startswith(b + "|") for b in _GNN_COLLAPSE_BASES):
            out[k] = dict(v)

    for base in _GNN_COLLAPSE_BASES:
        prefix = base + "|"
        dss = {ds for (rk, ds) in stats_sub if rk.startswith(prefix)}
        for ds in dss:
            candidates = [
                (rk, st)
                for (rk, dsk), st in stats_sub.items()
                if dsk == ds and rk.startswith(prefix)
            ]
            if not candidates:
                continue
            mode = str(candidates[0][1].get("mode", "max"))
            finite_val = [
                (rk, st)
                for rk, st in candidates
                if _finite(float(st.get("val_mean", float("nan"))))
            ]
            if finite_val:
                if mode == "min":
                    _win_rk, win_st = min(finite_val, key=lambda x: float(x[1]["val_mean"]))
                else:
                    _win_rk, win_st = max(finite_val, key=lambda x: float(x[1]["val_mean"]))
            else:
                ok_test = [
                    (rk, st)
                    for rk, st in candidates
                    if _finite(float(st.get("test_mean", float("nan"))))
                ]
                if not ok_test:
                    continue
                _win_rk, win_st = ok_test[0]

            merged = dict(win_st)
            out[(base, ds)] = merged
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
    label: str = "tbl:hopse_wandb_graph_sim",
    caption: str | None = None,
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
    cap = caption or (
        "Test mean $\\pm$ std over seeds (validation-tuned configs). "
        "\\textbf{Bold}: best per column; "
        "\\protect\\colorbox{stdblue}{blue}: not significantly different (95\\,\\%)."
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
        help=f"Full submodel table .tex (default: {DEFAULT_LATEX_TABLE_TEX})",
    )
    p.add_argument(
        "--output-compact",
        type=Path,
        default=DEFAULT_LATEX_TABLE_COMPACT,
        help=(
            "Second .tex: GNN rows collapsed to best val sub-config per backbone "
            f"(default: {DEFAULT_LATEX_TABLE_COMPACT})"
        ),
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print LaTeX to stdout (big table, then compact unless --skip-compact).",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="PATH:HEADER",
        help=(
            "Dataset columns as path or path:LaTeX header. "
            "Default: DATASETS from main_loader.py (cocitation graph datasets excluded), "
            "then graph columns → simplicial."
        ),
    )
    p.add_argument("--decimals", type=int, default=2, help="Decimal places for numbers (default: 2)")
    p.add_argument(
        "--no-scale-fractions",
        action="store_true",
        help="Do not multiply accuracy/f1/... by 100 for display (W&B is often 0–1).",
    )
    p.add_argument(
        "--skip-compact",
        action="store_true",
        help="Do not write or print the GNN-collapsed compact table.",
    )
    args = p.parse_args()
    df = load_wandb_export_csv(args.input)
    stats_sub = collect_winner_test_by_submodel(df)

    if args.datasets:
        base_specs = expand_mantra_betti_specs(_parse_dataset_specs(args.datasets))
    else:
        base_specs = expand_mantra_betti_specs(_specs_from_loader_paths())

    groups = partition_specs_graph_simplicial(base_specs)

    graph_rows_sub = graph_submodel_table_rows(stats_sub)
    simplicial_rows_sub = simplicial_submodel_table_rows()
    cell_rows_sub = cell_submodel_table_rows()

    graph_rows_compact: list[tuple[str, str]] = [
        ("graph/gcn", "GCN"),
        ("graph/gat", "GAT"),
        ("graph/gin", "GIN"),
    ]

    caption_compact = (
        "Test mean $\\pm$ std over seeds (validation-tuned configs). "
        "\\textbf{Bold}: best per column; "
        "\\protect\\colorbox{stdblue}{blue}: not significantly different (95\\,\\%)."
    )

    tex_big = build_latex_table(
        stats_sub,
        column_groups=groups,
        graph_rows=graph_rows_sub,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        decimals=args.decimals,
        scale_fraction_metrics=not args.no_scale_fractions,
        label="tbl:hopse_wandb_graph_sim_big",
    )

    stats_compact = collapse_gnn_submodel_rows_to_base(stats_sub)
    tex_compact = build_latex_table(
        stats_compact,
        column_groups=groups,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        decimals=args.decimals,
        scale_fraction_metrics=not args.no_scale_fractions,
        label="tbl:hopse_wandb_graph_sim_compact",
        caption=caption_compact,
    )

    if args.stdout:
        sys.stdout.write(tex_big)
        if not args.skip_compact:
            sys.stdout.write(
                "\n% --- compact: GNN = best val sub-config per backbone (GCN/GAT/GIN) ---\n\n"
            )
            sys.stdout.write(tex_compact)
    else:
        out_big = Path(args.output)
        out_big.parent.mkdir(parents=True, exist_ok=True)
        out_big.write_text(tex_big, encoding="utf-8")
        print(f"Wrote {out_big}")
        if not args.skip_compact:
            out_c = Path(args.output_compact)
            out_c.parent.mkdir(parents=True, exist_ok=True)
            out_c.write_text(tex_compact, encoding="utf-8")
            print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
