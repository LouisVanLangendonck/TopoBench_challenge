"""
plot_ssl_eval_homophily_bins.py
===============================
Plot **structural × feature** heatmaps of max gain vs a reference accuracy.

**Input:** one tidy downstream CSV (e.g. from ``analyze_downstream_wandb_results.py``)
that contains **both** transductive and inductive rows. Rows are split using the
``evaluation_setting`` column (``transductive`` vs ``inductive``; other values
such as ``unknown`` are dropped with a short note).

Homophily is binned from the mean of ``data__homophily_range``:

* **Low** — mean < ``--low-threshold`` (default 0.2)
* **High** — mean > ``--high-threshold`` (default 0.8)
* **Medium** — otherwise

**Default figure (single PNG):** one tall figure with **macro-rows** separated by
whitespace. Each macro band is one homophily bin. Inside each band:

* **Top micro-row:** transductive — one heatmap per ``pretrain_method`` (column).
* **Bottom micro-row:** inductive — same column alignment.

Values are ``max(accuracy - reference)`` over all other hyperparameters (SSL, FT,
exact homophily range inside the bin, etc.).

**Default** ``--delta-vs baseline`` matches ``plot_transductive_delta_3d.py`` (uses
``baseline_accuracy``, i.e. the matched random-init run from the tidy CSV).  Use
``--delta-vs random`` only if you want ``accuracy - random_guess`` (≈ ``1/K``), which
is *much* larger and is **not** the same as the baseline heatmaps.

Use ``--split-homophily-figures`` to restore the older layout (one PNG per bin,
SSL rows × eval columns).

**Fine-tuning slices:** figures are **not** mixed across ``ft__mode`` or
``ft__n_train``. For each ``ft__mode`` that appears in **both** settings, we take
the Cartesian product of distinct ``ft__n_train`` on the transductive side ×
distinct ``ft__n_train`` on the inductive side (e.g. trans has ``{30}``, ind has
``{5,10}`` → two plots: (30,5) and (30,10)).  Each plot still maxes over SSL and
other FT hparams **within** that slice.

Example::

    python plotting/plot_ssl_eval_homophily_bins.py -o plotting/out_compare \\
        --csv plotting/downstream_results_graphmaev2_ssl_only.csv

Legacy (two files merged with explicit labels)::

    python plotting/plot_ssl_eval_homophily_bins.py -o plotting/out_compare \\
        --pair transductive=plotting/trans.csv \\
        --pair inductive=plotting/ind.csv
"""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SIGNAL_ORDER = ("Low", "Medium", "High")

# From analyze_downstream_wandb_results.py (single tidy CSV).
EVAL_SETTING_COL = "evaluation_setting"

MIN_COLS_BASE = (
    "pretrain_method",
    "data__homophily_range",
    "data__structural_signal",
    "data__feature_signal",
    "ft__mode",
    "ft__n_train",
    "accuracy",
)


def _apply_global_style() -> None:
    """Apply larger, bolder typography for all generated figures."""
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
        }
    )


def homophily_to_mean(val: object) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    try:
        parsed = ast.literal_eval(s)
    except (SyntaxError, ValueError, TypeError):
        m = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
        if not m:
            return np.nan
        return float(np.mean([float(x) for x in m]))
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 1:
        return float(np.mean([float(x) for x in parsed]))
    return float(parsed)


def build_ordinal_mapping(
    series: pd.Series, preferred: tuple[str, ...] = DEFAULT_SIGNAL_ORDER
) -> dict[str, int]:
    present = sorted({str(x).strip() for x in series.dropna().unique()})
    ordered = [x for x in preferred if x in present]
    rest = [x for x in present if x not in preferred]
    full = ordered + sorted(rest)
    return {k: i for i, k in enumerate(full)}


def homophily_bin(mean: float, low_thr: float, high_thr: float) -> str | None:
    if not np.isfinite(mean):
        return None
    if mean < low_thr:
        return "Low"
    if mean > high_thr:
        return "High"
    return "Medium"


def _text_color_for_cell(v: float, vmin: float, vmax: float, cmap_name: str) -> str:
    if not np.isfinite(v) or vmax <= vmin:
        return "black"
    t = float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))
    rgba = plt.get_cmap(cmap_name)(t)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def parse_pair(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected eval_setting=path, got: {spec!r}"
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise argparse.ArgumentTypeError(f"Empty eval label in: {spec!r}")
    return label, path


def load_from_unified_csv(path: Path, ref_col: str) -> pd.DataFrame:
    """
    One CSV: split transductive vs inductive via ``evaluation_setting``.
    Normalizes labels to lowercase ``transductive`` / ``inductive``.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    need = tuple(MIN_COLS_BASE) + (ref_col, EVAL_SETTING_COL)
    df = pd.read_csv(path)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"{path}: missing columns {miss}")
    df = df.copy()
    raw = df[EVAL_SETTING_COL].astype(str).str.strip()
    lo = raw.str.lower()
    allowed = {"transductive", "inductive"}
    mask = lo.isin(allowed)
    dropped = int((~mask).sum())
    if dropped:
        print(
            f"Note: dropped {dropped} row(s) where {EVAL_SETTING_COL!r} is not "
            "transductive or inductive."
        )
    df = df.loc[mask].copy()
    if df.empty:
        raise SystemExit(
            f"{path}: no rows left after keeping only transductive/inductive."
        )
    df["eval_setting"] = lo[mask].values
    df["ft__n_train"] = pd.to_numeric(df["ft__n_train"], errors="coerce")
    return df


def load_merged(pairs: list[tuple[str, Path]], ref_col: str) -> pd.DataFrame:
    """Legacy: merge separate CSVs; each row's eval label comes from ``--pair``."""
    frames: list[pd.DataFrame] = []
    need = tuple(MIN_COLS_BASE) + (ref_col,)
    for eval_setting, path in pairs:
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"{path}: missing columns {miss}")
        df = df.copy()
        df["eval_setting"] = eval_setting.strip().lower()
        df["ft__n_train"] = pd.to_numeric(df["ft__n_train"], errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _modes_for_eval(df: pd.DataFrame, eval_lab: str) -> set[str]:
    s = df.loc[df["eval_setting"] == eval_lab, "ft__mode"]
    return {str(x).strip() for x in s.dropna().unique() if str(x).strip()}


def _unique_n_train_sorted(df: pd.DataFrame) -> list[float]:
    v = pd.to_numeric(df["ft__n_train"], errors="coerce").dropna().unique()
    return sorted(float(x) for x in v)


def _mask_eval_mode_nt(
    df: pd.DataFrame, eval_lab: str, mode: str, n_train: float
) -> pd.Series:
    nt = pd.to_numeric(df["ft__n_train"], errors="coerce")
    return (
        (df["eval_setting"] == eval_lab)
        & (df["ft__mode"].astype(str).str.strip() == str(mode).strip())
        & np.isclose(nt, float(n_train), rtol=0.0, atol=1e-5)
    )


def iter_trans_ind_ft_slices(
    merged: pd.DataFrame, top_ev: str, bot_ev: str
) -> Iterator[tuple[str, float, float, pd.DataFrame]]:
    """
    Yield (ft__mode, n_train_transductive, n_train_inductive, row_subset) for
    every shared mode and Cartesian product of n_train values per eval side.
    """
    modes_t = _modes_for_eval(merged, top_ev)
    modes_i = _modes_for_eval(merged, bot_ev)
    common = sorted(modes_t & modes_i)
    for mode in common:
        dt = merged.loc[
            (merged["eval_setting"] == top_ev)
            & (merged["ft__mode"].astype(str).str.strip() == mode)
        ]
        di = merged.loc[
            (merged["eval_setting"] == bot_ev)
            & (merged["ft__mode"].astype(str).str.strip() == mode)
        ]
        nts_t = _unique_n_train_sorted(dt)
        nts_i = _unique_n_train_sorted(di)
        for nt_t in nts_t:
            for nt_i in nts_i:
                mask = _mask_eval_mode_nt(merged, top_ev, mode, nt_t) | _mask_eval_mode_nt(
                    merged, bot_ev, mode, nt_i
                )
                sub = merged.loc[mask].copy()
                if not sub.empty:
                    yield mode, nt_t, nt_i, sub


def _n_train_filename_token(x: float) -> str:
    if abs(x - round(x)) < 1e-5:
        return str(int(round(x)))
    return str(x).replace(".", "p")


def aggregate_max_delta(
    df: pd.DataFrame, low_thr: float, high_thr: float, ref_col: str
) -> pd.DataFrame:
    work = df.copy()
    work["accuracy"] = pd.to_numeric(work["accuracy"], errors="coerce")
    work[ref_col] = pd.to_numeric(work[ref_col], errors="coerce")
    work = work.dropna(subset=["accuracy", ref_col])
    work["delta"] = work["accuracy"] - work[ref_col]
    work["h_mean"] = work["data__homophily_range"].map(homophily_to_mean)
    before = len(work)
    work = work.dropna(subset=["h_mean"])
    dropped = before - len(work)
    if dropped:
        print(f"Warning: dropped {dropped} rows with non-parsable homophily")
    work["homophily_bin"] = work["h_mean"].map(lambda m: homophily_bin(m, low_thr, high_thr))
    work = work.dropna(subset=["homophily_bin"])

    group_cols = [
        "homophily_bin",
        "data__structural_signal",
        "data__feature_signal",
        "pretrain_method",
        "eval_setting",
    ]
    return work.groupby(group_cols, dropna=False, as_index=False).agg(delta=("delta", "max"))


def matrix_for_cell(
    part: pd.DataFrame,
    struct_map: dict[str, int],
    feat_map: dict[str, int],
    n_struct: int,
    n_feat: int,
) -> np.ndarray:
    mat = np.full((n_struct, n_feat), np.nan, dtype=float)
    for _, row in part.iterrows():
        sk = str(row["data__structural_signal"]).strip()
        fk = str(row["data__feature_signal"]).strip()
        if sk not in struct_map or fk not in feat_map:
            continue
        mat[struct_map[sk], feat_map[fk]] = float(row["delta"])
    return mat


def order_eval_labels(labels: list[str], preferred: tuple[str, ...]) -> list[str]:
    labels = list(dict.fromkeys(labels))
    ordered = [x for x in preferred if x in labels]
    rest = sorted(x for x in labels if x not in preferred)
    return ordered + rest


def resolve_transductive_inductive(
    seen_labels: Iterable[str],
    top_key: str = "transductive",
    bottom_key: str = "inductive",
) -> tuple[str | None, str | None]:
    by_lo: dict[str, str] = {}
    for x in seen_labels:
        s = str(x).strip()
        if s:
            by_lo[s.lower()] = s
    return by_lo.get(top_key.lower()), by_lo.get(bottom_key.lower())


def bin_banner_text(name: str, low_thr: float, high_thr: float) -> tuple[str, str]:
    if name == "Low":
        return f"Homophily — {name}", f"mean of sampling range < {low_thr}"
    if name == "High":
        return f"Homophily — {name}", f"mean of sampling range > {high_thr}"
    return f"Homophily — {name}", f"{low_thr} ≤ mean of range ≤ {high_thr}"


def plot_combined_macro_figure(
    agg: pd.DataFrame,
    want_bins: list[str],
    struct_map: dict[str, int],
    feat_map: dict[str, int],
    ssl_methods: list[str],
    top_eval: str,
    bottom_eval: str,
    out_path: Path | None,
    suptitle: str,
    cmap: str,
    cell_size: float,
    dpi: int,
    show: bool,
    annotate: bool,
    low_thr: float,
    high_thr: float,
    cbar_caption: str,
) -> None:
    struct_labels = [k for k, _ in sorted(struct_map.items(), key=lambda kv: kv[1])]
    feat_labels = [k for k, _ in sorted(feat_map.items(), key=lambda kv: kv[1])]
    ns, nf = len(struct_labels), len(feat_labels)

    sub_all = agg.loc[agg["homophily_bin"].isin(want_bins)]
    vmin = float(sub_all["delta"].min()) if len(sub_all) else 0.0
    vmax = float(sub_all["delta"].max()) if len(sub_all) else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    n_ssl = len(ssl_methods)
    n_macro = len(want_bins)
    fig_w = max(2.0 + n_ssl * cell_size * 0.92, 8.0)
    fig_h = max(1.2 + n_macro * (0.55 + 2 * cell_size * 0.82) + (n_macro - 1) * 0.12, 8.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        n_macro,
        1,
        height_ratios=[1.0] * n_macro,
        hspace=0.30,
        left=0.08,
        right=0.78,
        top=0.93,
        bottom=0.07,
    )

    cmap_m = plt.get_cmap(cmap).copy()
    cmap_m.set_bad(color="#e8e8e8")

    mappable = None
    eval_rows = (top_eval, bottom_eval)
    row_titles = ("Transductive", "Inductive")

    for mi, b in enumerate(want_bins):
        inner = outer[mi, 0].subgridspec(
            3,
            n_ssl,
            height_ratios=[0.14, 1.0, 1.0],
            hspace=0.30,
            wspace=0.32,
        )

        ax_band = fig.add_subplot(inner[0, :])
        ax_band.set_axis_off()
        ax_band.set_facecolor("#e8eaed")
        for spine in ax_band.spines.values():
            spine.set_visible(False)
        line_y = 0.12
        ax_band.axhline(line_y, color="#9aa0a6", linewidth=1.0, xmin=0.02, xmax=0.98)
        title_main, _ = bin_banner_text(b, low_thr, high_thr)
        ax_band.text(
            0.5,
            0.52,
            title_main,
            transform=ax_band.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#202124",
        )

        sub_bin = agg.loc[agg["homophily_bin"] == b]
        row_axes: dict[int, list] = {0: [], 1: []}

        for ri, (ev, row_name) in enumerate(zip(eval_rows, row_titles)):
            for j, ssl in enumerate(ssl_methods):
                ax = fig.add_subplot(inner[1 + ri, j])
                row_axes[ri].append(ax)
                part = sub_bin[
                    (sub_bin["pretrain_method"] == ssl) & (sub_bin["eval_setting"] == ev)
                ]
                mat = matrix_for_cell(part, struct_map, feat_map, ns, nf)

                mappable = ax.imshow(mat, cmap=cmap_m, vmin=vmin, vmax=vmax, aspect="equal")
                ax.set_xticks(np.arange(nf))
                ax.set_yticks(np.arange(ns))

                if ri == 0:
                    ax.set_title(ssl, fontsize=14, fontweight="bold", pad=7)

                if ri == 1:
                    ax.set_xticklabels(feat_labels, rotation=28, ha="right", fontsize=11)
                else:
                    ax.set_xticklabels([])

                # Left column: Transductive / Inductive only. Rightmost: structural levels + title.
                last_col = j == n_ssl - 1
                if j == 0:
                    ax.set_ylabel(
                        row_name,
                        fontsize=13,
                        fontweight="bold",
                        labelpad=10,
                    )
                if n_ssl > 1:
                    if last_col:
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")
                        ax.set_yticklabels(struct_labels, fontsize=11)
                        ax.tick_params(
                            axis="y",
                            left=False,
                            right=True,
                            labelleft=False,
                            labelright=True,
                        )
                    else:
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.yaxis.tick_right()
                    ax.set_yticklabels(struct_labels, fontsize=11)
                    ax.tick_params(
                        axis="y",
                        left=False,
                        right=True,
                        labelleft=False,
                        labelright=True,
                    )
                if annotate:
                    for ii in range(ns):
                        for jj in range(nf):
                            v = mat[ii, jj]
                            if np.isnan(v):
                                continue
                            ax.text(
                                jj,
                                ii,
                                f"{v:.3f}",
                                ha="center",
                                va="center",
                                fontsize=6,
                                color=_text_color_for_cell(v, vmin, vmax, cmap),
                            )

        # One centered Feature/Structural label per homophily macro band.
        if row_axes[1]:
            bottom_left = row_axes[1][0].get_position()
            bottom_right = row_axes[1][-1].get_position()
            x_center = 0.5 * (bottom_left.x0 + bottom_right.x1)
            y_feat = bottom_left.y0 - 0.032
            fig.text(
                x_center,
                y_feat,
                "Feature signal",
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
            )
        if row_axes[0] and row_axes[1]:
            top_right = row_axes[0][-1].get_position()
            bottom_right = row_axes[1][-1].get_position()
            x_struct = top_right.x1 + 0.1
            y_struct = 0.5 * (top_right.y1 + bottom_right.y0)
            fig.text(
                x_struct,
                y_struct,
                "Structural signal",
                rotation=-90,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=0.99)
    if mappable is not None:
        cax = fig.add_axes([0.925, 0.16, 0.024, 0.70])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(
            cbar_caption,
            fontsize=16,
            labelpad=18,
            rotation=90,
            fontweight="normal",
        )
        cbar.ax.tick_params(labelsize=13)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bin_grid(
    agg: pd.DataFrame,
    homophily_bin: str,
    struct_map: dict[str, int],
    feat_map: dict[str, int],
    ssl_methods: list[str],
    eval_labels: list[str],
    out_path: Path | None,
    title_suffix: str,
    cmap: str,
    cell_size: float,
    dpi: int,
    show: bool,
    annotate: bool,
    low_thr: float,
    high_thr: float,
    cbar_caption: str,
) -> None:
    sub = agg.loc[agg["homophily_bin"] == homophily_bin]
    struct_labels = [k for k, _ in sorted(struct_map.items(), key=lambda kv: kv[1])]
    feat_labels = [k for k, _ in sorted(feat_map.items(), key=lambda kv: kv[1])]
    ns, nf = len(struct_labels), len(feat_labels)

    vmin = float(sub["delta"].min()) if len(sub) else 0.0
    vmax = float(sub["delta"].max()) if len(sub) else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    n_r, n_c = len(ssl_methods), len(eval_labels)
    fig_w = max(1.2 + n_c * cell_size, 6.0)
    fig_h = max(1.0 + n_r * cell_size, 5.0)
    fig, axes = plt.subplots(n_r, n_c, figsize=(fig_w, fig_h), squeeze=False)

    cmap_m = plt.get_cmap(cmap).copy()
    cmap_m.set_bad(color="#e8e8e8")

    mappable = None
    for r, ssl in enumerate(ssl_methods):
        for c, ev in enumerate(eval_labels):
            ax = axes[r, c]
            part = sub[
                (sub["pretrain_method"] == ssl) & (sub["eval_setting"] == ev)
            ]
            mat = matrix_for_cell(part, struct_map, feat_map, ns, nf)

            mappable = ax.imshow(mat, cmap=cmap_m, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks(np.arange(nf))
            ax.set_yticks(np.arange(ns))
            if r == n_r - 1:
                ax.set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=11)
                ax.set_xlabel("Feature signal", fontsize=14, fontweight="bold", labelpad=4)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_yticklabels(struct_labels, fontsize=11)
                ax.set_ylabel(
                    ssl,
                    fontsize=13,
                    fontweight="bold",
                    labelpad=4,
                )
            else:
                ax.set_yticklabels([])

            if r == 0:
                ax.set_title(ev, fontsize=14, fontweight="bold", pad=8)

            if annotate:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        v = mat[i, j]
                        if np.isnan(v):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{v:.3f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color=_text_color_for_cell(v, vmin, vmax, cmap),
                        )

    bin_desc = {
        "Low": f"mean homophily < {low_thr}",
        "Medium": f"{low_thr} ≤ mean homophily ≤ {high_thr}",
        "High": f"mean homophily > {high_thr}",
    }.get(homophily_bin, homophily_bin)

    fig.suptitle(
        "Few-shot Improvement Analysis of Pretrained Graph Models",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.84, wspace=0.25, hspace=0.28)
    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=axes.ravel().tolist(),
            shrink=0.65,
            pad=0.07,
        )
        cbar.set_label(
            cbar_caption,
            fontsize=14,
            labelpad=12,
            rotation=90,
            fontweight="normal",
        )
        cbar.ax.tick_params(labelsize=11)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    _apply_global_style()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        help=(
            "Single tidy CSV with an "
            f"{EVAL_SETTING_COL!r} column (transductive vs inductive rows)."
        ),
    )
    src.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        metavar="SETTING=CSV",
        help="Legacy: merge separate CSVs (repeat), e.g. transductive=a.csv inductive=b.csv",
    )
    p.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        required=True,
        help="Base output path (stem). Writes one PNG per (mode, n_train_trans, n_train_ind): "
        "<stem>_mode_<m>_nt_trans_<a>_ind_<b>.png. "
        "With --split-homophily-figures: ..._homophily_<Bin>.png per bin.",
    )
    p.add_argument(
        "--split-homophily-figures",
        action="store_true",
        help="Write one PNG per homophily bin (SSL rows × eval columns) instead of the combined macro layout",
    )
    p.add_argument(
        "--delta-vs",
        dest="delta_vs",
        choices=("baseline", "random"),
        default="baseline",
        help=(
            "baseline: accuracy - baseline_accuracy (same as plot_transductive_delta_3d.py; "
            "default). random: accuracy - random_guess (~1/K), not comparable to baseline heatmaps."
        ),
    )
    p.add_argument("--low-threshold", type=float, default=0.2)
    p.add_argument("--high-threshold", type=float, default=0.8)
    p.add_argument(
        "--eval-order",
        type=str,
        default="transductive,inductive",
        help="Comma-separated eval labels to pin as leftmost columns (rest sorted)",
    )
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--cell-size",
        type=float,
        default=2.4,
        help="Approx inches per heatmap cell (width/height of each subplot)",
    )
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument(
        "--bins",
        type=str,
        default="Low,Medium,High",
        help="Comma-separated subset of bins to plot",
    )

    args = p.parse_args()

    ref_col = "baseline_accuracy" if args.delta_vs == "baseline" else "random_guess"
    if args.delta_vs == "random":
        cbar_caption = (
            r"$\mathbf{\Delta\ Accuracy}$"
            + "\n"
            + r"(pretrained $-$ random guess $1/K$)"
        )
    else:
        cbar_caption = (
            r"$\mathbf{\Delta\ Accuracy}$"
            + "\n"
            + r"(pretrained $-$ random-init baseline)"
        )

    print(f"Metric: accuracy - {ref_col}  (--delta-vs {args.delta_vs})")

    if args.csv is not None:
        merged = load_from_unified_csv(args.csv, ref_col)
    else:
        pairs = args.pair or []
        if len(pairs) < 2:
            raise SystemExit("--pair legacy mode expects at least two SETTING=CSV entries.")
        merged = load_merged(pairs, ref_col)

    struct_map = build_ordinal_mapping(merged["data__structural_signal"])
    feat_map = build_ordinal_mapping(merged["data__feature_signal"])

    eval_pref = tuple(x.strip() for x in args.eval_order.split(",") if x.strip())
    eval_labels = order_eval_labels(
        merged["eval_setting"].dropna().astype(str).unique().tolist(),
        eval_pref,
    )

    top_ev, bot_ev = resolve_transductive_inductive(eval_labels)
    if top_ev is None or bot_ev is None:
        raise SystemExit(
            "Need both `transductive` and `inductive` eval labels (case-insensitive). "
            f"Found: {eval_labels}"
        )

    slices = list(iter_trans_ind_ft_slices(merged, top_ev, bot_ev))
    if not slices:
        raise SystemExit(
            "No plot slices: need at least one ft__mode present in both eval CSVs, "
            "with nonempty rows for some (n_train_trans, n_train_ind) pair."
        )
    print(f"Generating {len(slices)} figure(s) (shared mode × n_train trans × n_train ind).")

    want_bins = [x.strip() for x in args.bins.split(",") if x.strip()]

    prefix = args.output_prefix
    stem = prefix.name
    out_dir = prefix.parent
    out_name = stem if "." not in stem else stem.rsplit(".", 1)[0]
    if not out_name:
        out_name = "homophily_compare"

    for mode, nt_t, nt_i, sub in slices:
        agg = aggregate_max_delta(sub, args.low_threshold, args.high_threshold, ref_col)
        ssl_methods = sorted(agg["pretrain_method"].dropna().astype(str).unique())
        slice_eval_labels = order_eval_labels(
            sub["eval_setting"].dropna().astype(str).unique().tolist(),
            eval_pref,
        )

        nt_t_disp = int(nt_t) if abs(nt_t - round(nt_t)) < 1e-5 else nt_t
        nt_i_disp = int(nt_i) if abs(nt_i - round(nt_i)) < 1e-5 else nt_i
        suptitle = "Few-shot Improvement Analysis of Pretrained Graph Models"
        title_suffix_bins = suptitle

        safe_mode = re.sub(r"[^\w\-.]+", "_", str(mode).strip())[:60]
        file_base = (
            f"{out_name}_mode_{safe_mode}_nt_trans_{_n_train_filename_token(nt_t)}"
            f"_ind_{_n_train_filename_token(nt_i)}"
        )

        bins_ok = [b for b in want_bins if b in ("Low", "Medium", "High")]
        bins_ok = [b for b in bins_ok if not agg.loc[agg["homophily_bin"] == b].empty]
        if not bins_ok:
            print(f"  Skip {file_base}: no data in selected homophily bins")
            continue

        if args.split_homophily_figures:
            for b in want_bins:
                if b not in ("Low", "Medium", "High"):
                    print(f"Skipping unknown bin label: {b!r}")
                    continue
                if agg.loc[agg["homophily_bin"] == b].empty:
                    print(f"No rows in homophily bin {b!r} for {file_base}; skipping")
                    continue
                out_path = out_dir / f"{file_base}_homophily_{b}.png"
                plot_bin_grid(
                    agg,
                    homophily_bin=b,
                    struct_map=struct_map,
                    feat_map=feat_map,
                    ssl_methods=ssl_methods,
                    eval_labels=slice_eval_labels,
                    out_path=out_path,
                    title_suffix=title_suffix_bins,
                    cmap=args.cmap,
                    cell_size=args.cell_size,
                    dpi=args.dpi,
                    show=args.show,
                    annotate=not args.no_annotate,
                    low_thr=args.low_threshold,
                    high_thr=args.high_threshold,
                    cbar_caption=cbar_caption,
                )
                print(f"Wrote {out_path}")
        else:
            out_path = out_dir / f"{file_base}.png"
            plot_combined_macro_figure(
                agg,
                want_bins=bins_ok,
                struct_map=struct_map,
                feat_map=feat_map,
                ssl_methods=ssl_methods,
                top_eval=top_ev,
                bottom_eval=bot_ev,
                out_path=out_path,
                suptitle=suptitle,
                cmap=args.cmap,
                cell_size=args.cell_size,
                dpi=args.dpi,
                show=args.show,
                annotate=not args.no_annotate,
                low_thr=args.low_threshold,
                high_thr=args.high_threshold,
                cbar_caption=cbar_caption,
            )
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
