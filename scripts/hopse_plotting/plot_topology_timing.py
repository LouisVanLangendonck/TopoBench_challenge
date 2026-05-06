#!/usr/bin/env python3
"""
Topology vs timing plots from the **full** seed-aggregated export (all datasets, all
hyperparameter groups — not a parameter-filtered subset).

Reads ``graph_lift_topology_summary.csv`` for train-split **edges + rank-2** topological size
(triangles for simplicial lifts, 2-cells for cell lifts) and the seed-aggregated W&B export
for timing. Re-run **main_loader** then **aggregator** so timing columns exist. The export’s
``summary_AvgTime/...`` and ``summary__runtime`` aggregate columns are renamed here to short
plot column names; ``model_row_key`` / ``model_backbone`` follow the same rules as
``table_generator`` LaTeX tables.

**Grouped boxplots:** one box per ``(model_row_key, dataset)`` summarizing the distribution of
mean train seconds/epoch or mean wall runtime **across all hyperparameter training** (each point is one
seed-aggregate row).

- **X-axis:** shared label (cell vs simplicial wording for edge+2-cell vs edge+triangle);
  tick shows dataset short name; numeric topological size is drawn horizontally just under the ticks.
  Graph lifts use the same short names as tables (MUTAG, BBB, …); x-order is by topological
  size (low → high).
  MANTRA (simplicial) figures omit topological-size x-label and counts; the main title is the
  usual distribution lead sentence plus ``Simplicial Datasets``.
- **Y-axis:** per-epoch vs end-to-end training time (seconds); parameter / per-param variants analogous.
- **Colors:** TopoTune warm, HOPSE blue ramp, other models distinct hues.

PNG outputs:

- **Timing:** under ``plots/topology_timing/`` — cell train-epoch + wall-runtime; simplicial graph
  and MANTRA (``*_edges_plus_cells`` / ``*_edges_plus_triangles_*.png``). The three faceted
  **wall-runtime** figures (cell + simplicial graph + simplicial MANTRA) are also written as ``.pdf``;
  every other output remains PNG-only.
- **Parameter count:** same layout under ``plots/topology_params/``, y-axis
  ``model.params.total`` from the seed-aggregated export (``*_params_total_*.png``).
- **Time per parameter (coarse):** ``plots/topology_timing_per_param/`` — train-epoch and wall
  runtime **divided by** ``model.params.total`` (s per param); same boxplot layout.

Usage (from repo root, with ``scripts/hopse_plotting`` on ``PYTHONPATH`` or run from that
directory)::

    python scripts/hopse_plotting/plot_topology_timing.py
    python scripts/hopse_plotting/plot_topology_timing.py --dpi 200
    python scripts/hopse_plotting/plot_topology_timing.py --param-out-dir plots/my_params
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from table_generator import dataframe_with_submodel_id
from utils import (
    CSV_DIR,
    DEFAULT_AGGREGATED_EXPORT_CSV,
    PLOTS_DIR,
    SUMMARY_COLUMN_PREFIX,
    coalesce_seed_agg_wall_runtime_mean_std,
    hydra_dataset_key_from_loader_identity,
    load_wandb_export_csv,
    publication_label_hopse_from_backbone_token,
)

DEFAULT_LIFT_CSV = CSV_DIR / "graph_lift_topology_summary.csv"
DEFAULT_AGG_CSV = DEFAULT_AGGREGATED_EXPORT_CSV
DEFAULT_OUT_DIR = PLOTS_DIR / "topology_timing"
DEFAULT_PARAM_OUT_DIR = PLOTS_DIR / "topology_params"
DEFAULT_PER_PARAM_OUT_DIR = PLOTS_DIR / "topology_timing_per_param"

PARAM_COUNT_COL = "model.params.total"
TRAIN_EPOCH_SEC_PER_PARAM = "train_epoch_sec_per_param"
WALL_RUNTIME_SEC_PER_PARAM = "wall_runtime_sec_per_param"

SUMMARY_EPOCH_MEAN = f"{SUMMARY_COLUMN_PREFIX}AvgTime/train_epoch_mean"
AGG_EPOCH_MEAN = f"{SUMMARY_EPOCH_MEAN}__mean"
AGG_EPOCH_STD_ACROSS_SEEDS = f"{SUMMARY_EPOCH_MEAN}__std"

EDGES_COL = "train_total_edges_undirected"
RANK2_COL = "train_total_rank2"

# Filename stems (lift CSV encodes triangles vs 2-cells by domain)
STEM_SIMPLICIAL_LIFT = "edges_plus_triangles"
STEM_CELL_LIFT = "edges_plus_cells"

# PNG basename stems (no suffix) that also get a sibling ``.pdf`` (publication); all other outputs stay PNG-only.
FACETED_WALL_RUNTIME_PDF_STEMS: frozenset[str] = frozenset(
    {
        f"cell_wall_runtime_box_vs_{STEM_CELL_LIFT}_faceted_by_dataset",
        f"simplicial_wall_runtime_box_vs_{STEM_SIMPLICIAL_LIFT}_graph_faceted_by_dataset",
        f"simplicial_wall_runtime_box_vs_{STEM_SIMPLICIAL_LIFT}_mantra_faceted_by_dataset",
    }
)

X_AXIS_LABEL_CELL = "Topological Size (total edge+2-cell count)"
X_AXIS_LABEL_SIMPLICIAL = "Topological Size (total edge+triangle count)"
# MANTRA / native simplicial lifts: topological size is uniform; no x-metric label or counts.

# Tick labels for graph lifts (cell + simplicial on graph); x-order is by complexity, not this list.
GRAPH_TOPO_PLOT_ORDER: tuple[tuple[str, str], ...] = (
    ("graph/MUTAG", "MUTAG"),
    ("graph/PROTEINS", "PROTEINS"),
    ("graph/NCI1", "NCI1"),
    ("graph/NCI109", "NCI109"),
    ("graph/BBB_Martins", "BBB"),
    ("graph/CYP3A4_Veith", "CYP3A4"),
    ("graph/Clearance_Hepatocyte_AZ", "Cl.Hep."),
    ("graph/Caco2_Wang", "Caco2"),
)
_GRAPH_TOPO_DISPLAY = dict(GRAPH_TOPO_PLOT_ORDER)

FIG_H = 4.35
TOPO_UNDER_FONTSIZE = 13.0
TOPO_UNDER_AXES_Y = -0.016
FACET_TOPO_UNDER_AXES_Y = -0.028
TOPO_NUMBER_COLOR = "#222222"
X_LABELPAD_GROUPED = 10.0
SUPXLABEL_Y = -0.04
LEGEND_BBOX_Y = -0.038

FIG_W_PER_DS = 0.34
FIG_W_PAD = 3.2
FIG_W_MAX = 20.0
FS_TITLE = 14
FS_XY = 12.5
FS_TICK = 10.5
FS_LEGEND = 9.5

# Faceted (subplot-per-dataset) figures: slightly larger type than the grouped plots
FACET_FS_TITLE = FS_TITLE + 2
FACET_FS_XY = FS_XY + 1.5
FACET_FS_TICK = FS_TICK + 1.2
FACET_FS_LEGEND = FS_LEGEND + 5.0
FACET_LEGEND_HANDLELENGTH = 2.8
FACET_LEGEND_HANDLEHEIGHT = 1.35
FACET_LEGEND_LABELSPACING = 0.9
FACET_LEGEND_BORDERPAD = 0.75
# Figure coords; >1.0 is ok with bbox_inches="tight" on save
FACET_SUPTITLE_Y = 1.07
# Faceted boxplots: chunky boxes / medians
FACET_BOX_EDGE_WIDTH = 1.08
FACET_MEDIAN_WIDTH = 1.85

TOPOTUNE_FACE_COLOR = "#E94D35"
HOPSE_FACE_COLORS = ("#8ECAE6", "#219EBC", "#023047")
OTHER_FACE_COLORS = (
    "#239B56",
    "#8E44AD",
    "#B7950B",
    "#117A65",
    "#6C3483",
    "#CA6F1E",
    "#148F77",
    "#D35400",
    "#884EA0",
    "#27AE60",
)


def _backbone_name_from_model_path(model_path: str) -> str:
    s = str(model_path).replace("\r", "").strip()
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _display_backbone(model_path: str, sub_id: str) -> str:
    bb = _backbone_name_from_model_path(model_path)
    sid = str(sub_id).strip()
    if bb == "hopse_m":
        if sid == "f":
            return "hopse_m_F"
        if sid == "pe":
            return "hopse_m_PE"
        return f"hopse_m_{sid}" if sid and sid != "default" else "hopse_m"
    return bb


def enrich_submodel_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = dataframe_with_submodel_id(df)
    out["model_row_key"] = out["model"].astype(str) + "|" + out["_sub_id"].astype(str)
    out["model_backbone"] = [
        _display_backbone(str(m), str(s)) for m, s in zip(out["model"], out["_sub_id"], strict=True)
    ]
    return out


def map_agg_timing_to_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    rename: dict[str, str] = {}
    if AGG_EPOCH_MEAN in t.columns:
        rename[AGG_EPOCH_MEAN] = "train_epoch_sec_mean"
    if AGG_EPOCH_STD_ACROSS_SEEDS in t.columns:
        rename[AGG_EPOCH_STD_ACROSS_SEEDS] = "train_epoch_sec_std"
    t = t.rename(columns=rename)
    wm, ws = coalesce_seed_agg_wall_runtime_mean_std(t)
    t["wall_runtime_sec_mean"] = wm
    t["wall_runtime_sec_std"] = ws
    return t


def add_timing_per_param_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ``train_epoch_sec_mean / model.params.total`` and ``wall_runtime_sec_mean / …`` (s per param).
    Rows with missing or zero parameter count get NaN ratios (dropped by ``prepare_plot_frame``).
    """
    t = df.copy()
    if PARAM_COUNT_COL not in t.columns:
        return t
    params = pd.to_numeric(t[PARAM_COUNT_COL], errors="coerce").replace(0, float("nan"))
    if "train_epoch_sec_mean" in t.columns:
        num = pd.to_numeric(t["train_epoch_sec_mean"], errors="coerce")
        t[TRAIN_EPOCH_SEC_PER_PARAM] = num / params
    if "wall_runtime_sec_mean" in t.columns:
        num = pd.to_numeric(t["wall_runtime_sec_mean"], errors="coerce")
        t[WALL_RUNTIME_SEC_PER_PARAM] = num / params
    return t


def _domain_from_model(model: str) -> str | None:
    s = str(model).replace("\r", "").strip()
    if s.startswith("simplicial/"):
        return "simplicial"
    if s.startswith("cell/"):
        return "cell"
    return None


def _is_mantra_simplicial_dataset(dataset_h: str) -> bool:
    """True for hydra keys ``simplicial/mantra_*`` including ``simplicial/mantra_betti_numbers#f1-1``."""
    s = str(dataset_h).replace("\r", "").strip()
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return s.lower().startswith("simplicial/mantra")


def _pretty_legend_label(model_row_key: str, model_backbone: str) -> str:
    bb = str(model_backbone).strip()
    pub = publication_label_hopse_from_backbone_token(bb)
    if pub:
        return pub
    if bb:
        return bb.replace("_", "-")
    return str(model_row_key).replace("|", " ")


def _model_for_row_key(df: pd.DataFrame, model_row_key: str) -> str:
    sub = df.loc[df["model_row_key"].astype(str) == model_row_key, "model"]
    return str(sub.iloc[0]) if len(sub) else ""


def _is_topotune_model(model: str) -> bool:
    return "topotune" in str(model).replace("\r", "").strip().lower()


def _is_hopse_model_path(model: str) -> bool:
    m = str(model).replace("\r", "").strip().lower()
    return "/hopse_m" in m or m.endswith("hopse_m") or "/hopse_g" in m or m.endswith("hopse_g")


def _hopse_color_index(model_row_key: str, model: str) -> int:
    m = str(model).replace("\r", "").strip().lower()
    if "/hopse_g" in m or m.endswith("hopse_g"):
        return 2
    parts = str(model_row_key).replace("\r", "").strip().split("|", 1)
    sub = parts[-1].strip().lower() if len(parts) > 1 else ""
    if sub == "pe":
        return 1
    return 0


def _infer_plot_domain(df: pd.DataFrame) -> str:
    if df.empty:
        return "cell"
    dom = _domain_from_model(str(df["model"].iloc[0]))
    return dom if dom in ("cell", "simplicial") else "cell"


def _model_order_bucket(df: pd.DataFrame, mk: str, domain: str) -> tuple[int, str]:
    """
    Sort key for model row keys: hopse_g → HOPSE-M-F → HOPSE-M-C → (other hopse_m) → topotune →
    … then cell: cccn, cwn / simplicial: sann, sccnn. Unknown models last.
    """
    model = _model_for_row_key(df, mk)
    mlow = str(model).lower().strip()
    sub = df.loc[df["model_row_key"].astype(str) == str(mk)]
    bb = str(sub["model_backbone"].iloc[0]).strip().lower() if len(sub) else ""
    parts = str(mk).replace("\r", "").strip().split("|", 1)
    sub_id = parts[-1].strip().lower() if len(parts) > 1 else ""

    if "/hopse_g" in mlow or mlow.endswith("hopse_g"):
        return (0, str(mk))
    is_hopse_m = "/hopse_m" in mlow or mlow.endswith("hopse_m") or bb.startswith("hopse_m")
    if is_hopse_m:
        if sub_id == "pe" or bb == "hopse_m_pe":
            return (2, str(mk))
        if sub_id == "f" or bb == "hopse_m_f":
            return (1, str(mk))
        return (3, str(mk))
    if _is_topotune_model(model):
        return (4, str(mk))
    if domain == "cell":
        if "cccn" in mlow or bb == "cccn":
            return (5, str(mk))
        if "cwn" in mlow or bb == "cwn":
            return (6, str(mk))
    elif domain == "simplicial":
        if "sann" in mlow or bb == "sann":
            return (5, str(mk))
        if "sccnn" in mlow or bb == "sccnn" or "sccn" in mlow or bb == "sccn":
            return (6, str(mk))
    return (999, str(mk))


def _ordered_model_row_keys(df: pd.DataFrame, row_keys: list[str], domain: str) -> list[str]:
    return sorted(row_keys, key=lambda k: _model_order_bucket(df, k, domain))


def build_row_key_color_map(
    df: pd.DataFrame, row_keys: list[str], *, domain: str | None = None
) -> dict[str, tuple[float, float, float, float]]:
    color_of: dict[str, tuple[float, float, float, float]] = {}
    other_i = 0
    if domain in ("cell", "simplicial"):
        ordered = _ordered_model_row_keys(df, row_keys, domain)
    else:
        ordered = sorted(row_keys)
    for mk in ordered:
        model = _model_for_row_key(df, mk)
        if _is_topotune_model(model):
            color_of[mk] = mcolors.to_rgba(TOPOTUNE_FACE_COLOR)
        elif _is_hopse_model_path(model):
            idx = _hopse_color_index(mk, model) % len(HOPSE_FACE_COLORS)
            color_of[mk] = mcolors.to_rgba(HOPSE_FACE_COLORS[idx])
        else:
            color_of[mk] = mcolors.to_rgba(OTHER_FACE_COLORS[other_i % len(OTHER_FACE_COLORS)])
            other_i += 1
    return color_of


def lift_edges_plus_rank2_by_dataset(lift_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``(simplicial_series, cell_series)`` indexed by ``dataset_h``: edges + rank-2 train totals."""
    lift = lift_df.copy()
    lift["dataset_h"] = lift["dataset"].map(lambda x: hydra_dataset_key_from_loader_identity(str(x)))

    sim_mask = lift["lifting_domain"].astype(str).isin(("simplicial", "native_simplicial"))
    cell_mask = lift["lifting_domain"].astype(str).eq("cell")

    def _series(sub: pd.DataFrame) -> pd.Series:
        e = pd.to_numeric(sub[EDGES_COL], errors="coerce").fillna(0)
        r = pd.to_numeric(sub[RANK2_COL], errors="coerce").fillna(0)
        vals = e + r
        return pd.Series(vals.values, index=sub["dataset_h"].values)

    sim_sub = lift.loc[sim_mask].drop_duplicates("dataset_h", keep="last")
    cell_sub = lift.loc[cell_mask].drop_duplicates("dataset_h", keep="last")
    return _series(sim_sub), _series(cell_sub)


def _format_topo_size_tick(cx: float) -> str:
    try:
        x = float(cx)
    except (TypeError, ValueError):
        return str(cx)
    if not math.isfinite(x):
        return "—"
    axv = abs(x)
    if axv >= 1e6:
        return f"{x:.3g}"
    if axv >= 1000:
        return f"{x:,.0f}".replace(",", "")
    if axv >= 10:
        return f"{x:.4g}"
    return f"{x:.3g}"


def dataset_sort_order_and_tick_labels(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """``(dataset_h keys, short names for ticks/titles, formatted topo-size numbers only)``."""
    u = df.drop_duplicates(subset=["dataset_h"], keep="first")[["dataset_h", "complexity"]].copy()
    u["complexity"] = pd.to_numeric(u["complexity"], errors="coerce")
    u = u.sort_values("complexity", kind="mergesort")
    dss: list[str] = []
    short_names: list[str] = []
    topo_num_labels: list[str] = []
    for _, row in u.iterrows():
        ds = str(row["dataset_h"])
        cx = row["complexity"]
        dss.append(ds)
        short = _GRAPH_TOPO_DISPLAY.get(ds, ds.split("/")[-1] if "/" in ds else ds)
        short_names.append(short)
        topo_num_labels.append(_format_topo_size_tick(float(cx)))
    return dss, short_names, topo_num_labels


def _annotate_topo_numbers_below_axis(
    ax,
    x_data: np.ndarray,
    topo_num_labels: list[str],
    *,
    axes_frac_y: float,
    fontsize: float,
    ha: str = "center",
) -> None:
    """Draw formatted topological-size counts below the x-axis (horizontal, x in data coords)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, txt in zip(x_data, topo_num_labels, strict=True):
        ax.text(
            float(x),
            axes_frac_y,
            txt,
            transform=trans,
            rotation=0,
            ha=ha,
            va="top",
            fontsize=fontsize,
            color=TOPO_NUMBER_COLOR,
        )


def _faceted_draw_one_panel(
    ax,
    df: pd.DataFrame,
    ds: str,
    row_keys: list[str],
    color_of: dict[str, tuple[float, float, float, float]],
    y_col: str,
    *,
    fs_tick: float,
    box_width: float | None = None,
    tick_label_for_mk: Callable[[str], str] | None = None,
    label_df: pd.DataFrame | None = None,
    show_xticklabels: bool = True,
) -> None:
    label_src = label_df if label_df is not None else df
    if box_width is None:
        n = max(len(row_keys), 1)
        box_width = min(0.86, 1.05 / n * 3.55)
    data_k: list[np.ndarray] = []
    empties: list[bool] = []
    sub_ds = df.loc[df["dataset_h"].astype(str) == ds].copy()
    for mk in row_keys:
        vals = (
            sub_ds.loc[sub_ds["model_row_key"].astype(str) == mk, y_col].dropna().to_numpy(dtype=float)
        )
        empties.append(vals.size == 0)
        if vals.size == 0:
            data_k.append(np.array([np.nan, np.nan, np.nan], dtype=float))
        else:
            data_k.append(vals)

    x_positions = np.arange(len(row_keys), dtype=float)
    bp = ax.boxplot(
        data_k,
        positions=x_positions.tolist(),
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        medianprops={"color": "0.12", "linewidth": FACET_MEDIAN_WIDTH},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.55},
    )
    for k, empty in enumerate(empties):
        if empty:
            _hide_boxplot_column(bp, k)
    for k, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(color_of[row_keys[k]])
        patch.set_alpha(0.82)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(FACET_BOX_EDGE_WIDTH)
    for w in bp["whiskers"]:
        w.set_linewidth(max(FACET_BOX_EDGE_WIDTH * 0.85, 0.9))
        w.set_color("0.28")
    for cap in bp["caps"]:
        cap.set_linewidth(max(FACET_BOX_EDGE_WIDTH * 0.85, 0.9))
        cap.set_color("0.28")

    ax.set_xticks(x_positions)
    if show_xticklabels:
        pretty: list[str] = []
        for mk in row_keys:
            if tick_label_for_mk is not None:
                pretty.append(tick_label_for_mk(mk))
            else:
                sub = label_src.loc[label_src["model_row_key"].astype(str) == mk]
                lab = str(sub["legend_label"].iloc[0]) if len(sub) else mk
                pretty.append(lab)
        ax.set_xticklabels(pretty, fontsize=fs_tick, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)


def _hide_boxplot_column(bp: dict, k: int) -> None:
    if k < len(bp["boxes"]):
        bp["boxes"][k].set_visible(False)
    w = bp["whiskers"]
    i0, i1 = 2 * k, 2 * k + 1
    if i1 < len(w):
        w[i0].set_visible(False)
        w[i1].set_visible(False)
    caps = bp["caps"]
    c0, c1 = 2 * k, 2 * k + 1
    if c1 < len(caps):
        caps[c0].set_visible(False)
        caps[c1].set_visible(False)
    if k < len(bp["medians"]):
        bp["medians"][k].set_visible(False)
    fl = bp.get("fliers")
    if fl is not None and k < len(fl):
        fl[k].set_visible(False)


def prepare_plot_frame(
    timing_df: pd.DataFrame,
    sim_x: pd.Series,
    cell_x: pd.Series,
    *,
    domain: str,
    y_col: str,
    simplicial_dataset_kind: str | None = None,
) -> pd.DataFrame:
    """
    ``simplicial_dataset_kind``: for ``domain == "simplicial"``, pass ``"graph"`` (non-MANTRA
    datasets, e.g. ``graph/MUTAG``) or ``"mantra"`` (``simplicial/mantra_*``). Ignored for
    cell models.
    """
    t = timing_df.copy()
    t["dataset_h"] = t["dataset"].map(lambda x: hydra_dataset_key_from_loader_identity(str(x)))
    t["plot_domain"] = t["model"].map(_domain_from_model)
    t = t.loc[t["plot_domain"].eq(domain)].copy()

    if domain == "simplicial":
        if simplicial_dataset_kind not in ("graph", "mantra"):
            raise ValueError('simplicial_dataset_kind must be "graph" or "mantra" for simplicial domain')
        is_mantra = t["dataset_h"].map(_is_mantra_simplicial_dataset)
        t = t.loc[is_mantra if simplicial_dataset_kind == "mantra" else ~is_mantra].copy()

    comp = sim_x if domain == "simplicial" else cell_x
    t["complexity"] = t["dataset_h"].map(comp)

    n_seeds = pd.to_numeric(t["n_seeds"], errors="coerce").fillna(0)
    t = t.loc[n_seeds > 0].copy()
    t[y_col] = pd.to_numeric(t[y_col], errors="coerce")
    t = t.loc[t["complexity"].notna() & t[y_col].notna()].copy()
    t = t.loc[(pd.to_numeric(t["complexity"], errors="coerce") > 0)].copy()
    t["legend_label"] = [
        _pretty_legend_label(str(a), str(b)) for a, b in zip(t["model_row_key"], t["model_backbone"], strict=True)
    ]
    return t


def boxplot_grouped_by_dataset(
    df: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    y_label: str,
    x_axis_label: str,
    show_topo_x: bool = True,
    out_path: Path,
    dpi: int,
) -> None:
    if df.empty:
        print(f"  (skip empty) {out_path.name}")
        return

    dss, short_names, topo_num_labels = dataset_sort_order_and_tick_labels(df)
    if not dss:
        print(f"  (skip empty) {out_path.name}")
        return

    dom = _infer_plot_domain(df)
    row_keys = _ordered_model_row_keys(df, sorted(df["model_row_key"].astype(str).unique()), dom)
    color_of = build_row_key_color_map(df, row_keys, domain=dom)
    n_d, n_m = len(dss), len(row_keys)
    group_pitch = max(0.19 * n_m + 0.36, 0.88)
    centers = np.arange(n_d, dtype=float) * group_pitch
    bw = min(0.16, 0.72 / max(n_m, 1))

    fig_w = max(7.5, min(FIG_W_MAX, FIG_W_PER_DS * n_d * group_pitch + FIG_W_PAD))
    fig, ax = plt.subplots(figsize=(fig_w, FIG_H), layout="constrained")

    for j, mk in enumerate(row_keys):
        pos = centers + (j - (n_m - 1) / 2.0) * bw
        data_k: list[np.ndarray] = []
        empties: list[bool] = []
        for ds in dss:
            v = df.loc[
                (df["dataset_h"].astype(str) == ds) & (df["model_row_key"].astype(str) == mk),
                y_col,
            ].dropna().to_numpy(dtype=float)
            empties.append(v.size == 0)
            if v.size == 0:
                data_k.append(np.array([np.nan, np.nan, np.nan], dtype=float))
            else:
                data_k.append(v)

        bp = ax.boxplot(
            data_k,
            positions=pos.tolist(),
            widths=bw * 0.88,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            medianprops={"color": "0.12", "linewidth": 1.3},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.55},
        )
        for k, empty in enumerate(empties):
            if empty:
                _hide_boxplot_column(bp, k)
        for patch in bp["boxes"]:
            patch.set_facecolor(color_of[mk])
            patch.set_alpha(0.82)
            patch.set_edgecolor("0.25")
            patch.set_linewidth(0.6)

    ax.set_xticks(centers)
    ax.set_xticklabels(short_names, fontsize=FS_TICK, rotation=0, ha="center")
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel(y_label, fontsize=FS_XY)
    if show_topo_x:
        ax.set_xlabel(x_axis_label, fontsize=FS_XY, labelpad=X_LABELPAD_GROUPED)
    else:
        ax.set_xlabel("")
    ax.set_title(title, fontsize=FS_TITLE, pad=5, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)
    if show_topo_x:
        _annotate_topo_numbers_below_axis(
            ax,
            centers,
            topo_num_labels,
            axes_frac_y=TOPO_UNDER_AXES_Y,
            fontsize=TOPO_UNDER_FONTSIZE,
            ha="center",
        )

    legend_labels: list[str] = []
    legend_handles: list[Patch] = []
    for mk in row_keys:
        sub = df.loc[df["model_row_key"].astype(str) == mk]
        lab = str(sub["legend_label"].iloc[0]) if len(sub) else mk
        legend_labels.append(lab)
        legend_handles.append(
            Patch(facecolor=color_of[mk], edgecolor="0.25", linewidth=0.6, label=lab, alpha=0.82)
        )
    order = list(range(len(legend_labels)))
    ax.legend(
        [legend_handles[i] for i in order],
        [legend_labels[i] for i in order],
        loc="best",
        fontsize=FS_LEGEND,
        framealpha=0.92,
        ncol=2 if len(row_keys) > 8 else 1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _domain_title_suffix(domain: str) -> str:
    if domain == "cell":
        return "Cell Domain."
    if domain == "sim_graph":
        return "Simplicial Domain, Graph Datasets."
    if domain == "sim_mantra":
        return "Simplicial Domain, Simplicial Datasets."
    raise ValueError(f"unknown domain: {domain!r}")


def _distribution_lead(topic_phrase: str, y_col: str) -> str:
    """First sentence of the figure title (before domain / dataset phrase)."""
    if topic_phrase == "timing":
        if y_col == "wall_runtime_sec_mean":
            return "End-to-end training time distributions over all hyperparameter runs."
        if y_col == "train_epoch_sec_mean":
            return "Per-epoch training time distributions over all hyperparameter runs."
        return "Training time distributions over all hyperparameter runs."
    if topic_phrase == "parameter count":
        return "Parameter count distributions over all hyperparameter runs."
    if topic_phrase == "timing per parameter":
        if y_col == WALL_RUNTIME_SEC_PER_PARAM:
            return "End-to-end training time per parameter distributions over all hyperparameter runs."
        if y_col == TRAIN_EPOCH_SEC_PER_PARAM:
            return "Per-epoch training time per parameter distributions over all hyperparameter runs."
        return "Training time per parameter distributions over all hyperparameter runs."
    raise ValueError(f"unknown topic: {topic_phrase!r}")


def _simple_main_title(topic_phrase: str, domain: str, *, y_col: str) -> str:
    """Short main title: one lead sentence + domain phrase (same for grouped and faceted)."""
    return f"{_distribution_lead(topic_phrase, y_col)} {_domain_title_suffix(domain)}"


def _mantra_main_title(topic_phrase: str, *, y_col: str) -> str:
    """MANTRA figures: same lead as other plots, then ``Simplicial Datasets`` (no topo x-axis)."""
    return f"{_distribution_lead(topic_phrase, y_col)} Simplicial Datasets"


def _faceted_axes_at(axs, nrows: int, n_d: int, r: int, i: int):
    if nrows == 1 and n_d == 1:
        return axs
    if nrows == 1:
        return axs[i]
    if n_d == 1:
        return axs[r]
    return axs[r, i]


def _save_faceted_figure(fig, out_path: Path, *, dpi: int, pad_inches: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": pad_inches}
    fig.savefig(out_path, **save_kw)
    print(f"  wrote {out_path}")
    if out_path.suffix.lower() == ".png" and out_path.stem in FACETED_WALL_RUNTIME_PDF_STEMS:
        pdf_p = out_path.with_suffix(".pdf")
        fig.savefig(pdf_p, **save_kw)
        print(f"  wrote {pdf_p}")


def boxplot_faceted_by_dataset_columns(
    df: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    y_label: str,
    x_axis_label: str,
    show_topo_x: bool = True,
    out_path: Path,
    dpi: int,
) -> None:
    if df.empty:
        print(f"  (skip empty) {out_path.name}")
        return

    dss, short_names, topo_num_labels = dataset_sort_order_and_tick_labels(df)
    if not dss:
        print(f"  (skip empty) {out_path.name}")
        return

    dom = _infer_plot_domain(df)
    row_keys = _ordered_model_row_keys(df, sorted(df["model_row_key"].astype(str).unique()), dom)
    if not row_keys:
        print(f"  (skip empty) {out_path.name}")
        return
    color_of = build_row_key_color_map(df, row_keys, domain=dom)
    model_tick_labels = []
    for mk in row_keys:
        sub = df.loc[df["model_row_key"].astype(str) == mk]
        lab = str(sub["legend_label"].iloc[0]) if len(sub) else mk
        model_tick_labels.append(lab)

    n_d = len(dss)
    nrows = 1
    fig_w = max(8.0, min(46.0, 2.2 * n_d + 2.3))
    fig_h = max(FIG_H + 0.35, 4.85) * 1.04
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=n_d,
        figsize=(fig_w, fig_h),
        sharey=False,
        constrained_layout=True,
    )
    n_m = max(len(row_keys), 1)
    box_width = min(0.86, 1.05 / n_m * 3.55)
    for i, ds in enumerate(dss):
        ax = _faceted_axes_at(axs, nrows, n_d, 0, i)
        _faceted_draw_one_panel(
            ax,
            df,
            ds,
            row_keys,
            color_of,
            y_col,
            fs_tick=FACET_FS_TICK,
            box_width=box_width,
            show_xticklabels=False,
        )
        ax.set_title(
            short_names[i],
            fontsize=FACET_FS_TICK + 0.45,
            pad=7,
            fontweight="bold",
        )
        if i == 0:
            ax.set_ylabel(y_label, fontsize=FACET_FS_XY)
        if show_topo_x:
            n_m_k = max(len(row_keys), 1)
            x_mid = 0.5 * float(n_m_k - 1)
            _annotate_topo_numbers_below_axis(
                ax,
                np.array([x_mid], dtype=float),
                [topo_num_labels[i]],
                axes_frac_y=FACET_TOPO_UNDER_AXES_Y,
                fontsize=TOPO_UNDER_FONTSIZE,
                ha="center",
            )

    fig.suptitle(title, fontsize=FACET_FS_TITLE, y=FACET_SUPTITLE_Y, fontweight="bold")
    if show_topo_x:
        fig.supxlabel(x_axis_label, fontsize=FACET_FS_XY, y=SUPXLABEL_Y)

    legend_handles = [
        Patch(
            facecolor=color_of[mk],
            edgecolor="0.25",
            linewidth=FACET_BOX_EDGE_WIDTH,
            label=lab,
            alpha=0.82,
        )
        for mk, lab in zip(row_keys, model_tick_labels, strict=True)
    ]
    order = list(range(len(model_tick_labels)))
    n_leg = max(1, len(model_tick_labels))
    fig.legend(
        [legend_handles[i] for i in order],
        [model_tick_labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=n_leg,
        fontsize=FACET_FS_LEGEND,
        framealpha=0.92,
        handlelength=FACET_LEGEND_HANDLELENGTH,
        handleheight=FACET_LEGEND_HANDLEHEIGHT,
        labelspacing=FACET_LEGEND_LABELSPACING,
        borderpad=FACET_LEGEND_BORDERPAD,
        columnspacing=1.35,
    )

    _save_faceted_figure(fig, out_path, dpi=dpi, pad_inches=0.2)
    plt.close(fig)


def emit_topology_plots(
    plot_df: pd.DataFrame,
    sim_x: pd.Series,
    cell_x: pd.Series,
    *,
    stem_sim: str,
    stem_cell: str,
    out_dir: Path,
    dpi: int,
    y_metrics: tuple[tuple[str, str, str], ...],
    topic_phrase: str,
) -> None:
    """Grouped + faceted boxplots for each ``y_metrics`` entry (same layout as timing or param count)."""

    for y_col, stem_y, y_lab in y_metrics:
        merged_cell = prepare_plot_frame(
            plot_df, sim_x, cell_x, domain="cell", y_col=y_col, simplicial_dataset_kind=None
        )
        boxplot_grouped_by_dataset(
            merged_cell,
            y_col=y_col,
            title=_simple_main_title(topic_phrase, "cell", y_col=y_col),
            y_label=y_lab,
            x_axis_label=X_AXIS_LABEL_CELL,
            out_path=out_dir / f"cell_{stem_y}_box_vs_{stem_cell}.png",
            dpi=dpi,
        )
        boxplot_faceted_by_dataset_columns(
            merged_cell,
            y_col=y_col,
            title=_simple_main_title(topic_phrase, "cell", y_col=y_col),
            y_label=y_lab,
            x_axis_label=X_AXIS_LABEL_CELL,
            out_path=out_dir / f"cell_{stem_y}_box_vs_{stem_cell}_faceted_by_dataset.png",
            dpi=dpi,
        )

        merged_sim_graph = prepare_plot_frame(
            plot_df,
            sim_x,
            cell_x,
            domain="simplicial",
            y_col=y_col,
            simplicial_dataset_kind="graph",
        )
        boxplot_grouped_by_dataset(
            merged_sim_graph,
            y_col=y_col,
            title=_simple_main_title(topic_phrase, "sim_graph", y_col=y_col),
            y_label=y_lab,
            x_axis_label=X_AXIS_LABEL_SIMPLICIAL,
            out_path=out_dir / f"simplicial_{stem_y}_box_vs_{stem_sim}_graph.png",
            dpi=dpi,
        )
        boxplot_faceted_by_dataset_columns(
            merged_sim_graph,
            y_col=y_col,
            title=_simple_main_title(topic_phrase, "sim_graph", y_col=y_col),
            y_label=y_lab,
            x_axis_label=X_AXIS_LABEL_SIMPLICIAL,
            out_path=out_dir / f"simplicial_{stem_y}_box_vs_{stem_sim}_graph_faceted_by_dataset.png",
            dpi=dpi,
        )

    for y_col, stem_y, y_lab in y_metrics:
        merged_mantra = prepare_plot_frame(
            plot_df,
            sim_x,
            cell_x,
            domain="simplicial",
            y_col=y_col,
            simplicial_dataset_kind="mantra",
        )
        boxplot_grouped_by_dataset(
            merged_mantra,
            y_col=y_col,
            title=_mantra_main_title(topic_phrase, y_col=y_col),
            y_label=y_lab,
            x_axis_label=X_AXIS_LABEL_SIMPLICIAL,
            show_topo_x=False,
            out_path=out_dir / f"simplicial_{stem_y}_box_vs_{stem_sim}_mantra.png",
            dpi=dpi,
        )
        boxplot_faceted_by_dataset_columns(
            merged_mantra,
            y_col=y_col,
            title=_mantra_main_title(topic_phrase, y_col=y_col),
            y_label=y_lab,
            x_axis_label=X_AXIS_LABEL_SIMPLICIAL,
            show_topo_x=False,
            out_path=out_dir / f"simplicial_{stem_y}_box_vs_{stem_sim}_mantra_faceted_by_dataset.png",
            dpi=dpi,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Topological size vs timing, parameter count, and timing/param ratio "
            "(all hyperparameter configurations, boxplots)."
        )
    )
    p.add_argument("--lift-csv", type=Path, default=DEFAULT_LIFT_CSV)
    p.add_argument("--agg-csv", type=Path, default=DEFAULT_AGG_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--param-out-dir", type=Path, default=DEFAULT_PARAM_OUT_DIR)
    p.add_argument("--per-param-out-dir", type=Path, default=DEFAULT_PER_PARAM_OUT_DIR)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    lift_df = pd.read_csv(args.lift_csv, low_memory=False)
    agg = load_wandb_export_csv(args.agg_csv)
    if agg.empty:
        print(f"No rows in {args.agg_csv}")
        return
    plot_df = add_timing_per_param_columns(map_agg_timing_to_plot_columns(enrich_submodel_columns(agg)))

    sim_x, cell_x = lift_edges_plus_rank2_by_dataset(lift_df)
    stem_sim, stem_cell = STEM_SIMPLICIAL_LIFT, STEM_CELL_LIFT

    y_metrics_timing: tuple[tuple[str, str, str], ...] = (
        ("train_epoch_sec_mean", "train_epoch", "Mean train time per epoch (s)"),
        ("wall_runtime_sec_mean", "wall_runtime", "End-to-end training time (s)"),
    )

    emit_topology_plots(
        plot_df,
        sim_x,
        cell_x,
        stem_sim=stem_sim,
        stem_cell=stem_cell,
        out_dir=args.out_dir,
        dpi=args.dpi,
        y_metrics=y_metrics_timing,
        topic_phrase="timing",
    )

    if PARAM_COUNT_COL not in plot_df.columns:
        print(f"No column {PARAM_COUNT_COL!r} in aggregated export; skip parameter-count plots.")
    else:
        args.param_out_dir.mkdir(parents=True, exist_ok=True)
        y_metrics_params: tuple[tuple[str, str, str], ...] = (
            (
                PARAM_COUNT_COL,
                "params_total",
                f"Parameter count ({PARAM_COUNT_COL})",
            ),
        )
        emit_topology_plots(
            plot_df,
            sim_x,
            cell_x,
            stem_sim=stem_sim,
            stem_cell=stem_cell,
            out_dir=args.param_out_dir,
            dpi=args.dpi,
            y_metrics=y_metrics_params,
            topic_phrase="parameter count",
        )

    y_metrics_per_param: list[tuple[str, str, str]] = []
    if PARAM_COUNT_COL in plot_df.columns:
        if TRAIN_EPOCH_SEC_PER_PARAM in plot_df.columns:
            y_metrics_per_param.append(
                (
                    TRAIN_EPOCH_SEC_PER_PARAM,
                    "train_epoch_per_param",
                    f"Mean train time per epoch (s) / {PARAM_COUNT_COL}",
                )
            )
        if WALL_RUNTIME_SEC_PER_PARAM in plot_df.columns:
            y_metrics_per_param.append(
                (
                    WALL_RUNTIME_SEC_PER_PARAM,
                    "wall_runtime_per_param",
                    f"End-to-end training time (s) / {PARAM_COUNT_COL}",
                )
            )
    if not y_metrics_per_param:
        print(
            "Skip timing-per-parameter plots (need model.params.total and timing __mean columns in export)."
        )
    else:
        args.per_param_out_dir.mkdir(parents=True, exist_ok=True)
        emit_topology_plots(
            plot_df,
            sim_x,
            cell_x,
            stem_sim=stem_sim,
            stem_cell=stem_cell,
            out_dir=args.per_param_out_dir,
            dpi=args.dpi,
            y_metrics=tuple(y_metrics_per_param),
            topic_phrase="timing per parameter",
        )


if __name__ == "__main__":
    main()
