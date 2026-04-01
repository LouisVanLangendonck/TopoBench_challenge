"""
plot_ssl_eval_ind_trans_3d_grid.py
==================================
One tidy CSV with ``evaluation_setting`` (transductive + inductive), same inputs as
``plot_ssl_eval_homophily_bins.py``.

For each fine-tuning slice (shared ``ft__mode`` × ``n_train`` trans × ``n_train`` ind),
writes a **single figure**:

* **Rows (2):** inductive (top), transductive (bottom).
* **Columns:** one 3D scatter per ``pretrain_method`` (SSL only).
* Each scatter matches ``plot_transductive_delta_3d.py``: axes are homophily (mean of
  range), structural signal index, feature signal index; color is **max**
  ``accuracy - reference`` over remaining hyperparameters for that
  (homophily, structural, feature) cell.
* **One shared colorbar** for the whole grid.

Example::

    python plotting/plot_ssl_eval_ind_trans_3d_grid.py \\
        --csv plotting/mixed_ssl_only.csv \\
        -o plotting/out_ind_trans_3d
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Standalone run: sibling script import
_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

import plot_ssl_eval_homophily_bins as hb


def _apply_global_style() -> None:
    """Make figure typography larger and bolder across all plot elements."""
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


def aggregate_cube_delta(df: pd.DataFrame, ref_col: str) -> pd.DataFrame:
    """Max delta per (homophily range, structural, feature)."""
    work = df.copy()
    work["accuracy"] = pd.to_numeric(work["accuracy"], errors="coerce")
    work[ref_col] = pd.to_numeric(work[ref_col], errors="coerce")
    work = work.dropna(subset=["accuracy", ref_col])
    if work.empty:
        return work
    work["delta"] = work["accuracy"] - work[ref_col]
    gcols = ["data__homophily_range", "data__structural_signal", "data__feature_signal"]
    return work.groupby(gcols, dropna=False, as_index=False).agg(delta=("delta", "max"))


def _ssl_methods(sub: pd.DataFrame) -> list[str]:
    pm = sub["pretrain_method"].astype(str)
    mask = ~pm.str.lower().isin(["random_init", "cd"])
    vals = sub.loc[mask, "pretrain_method"].dropna().astype(str).unique().tolist()
    return sorted(vals)


def plot_ind_trans_3d_grid(
    sub: pd.DataFrame,
    struct_map: dict[str, int],
    feat_map: dict[str, int],
    ssl_methods: list[str],
    inductive_lab: str,
    transductive_lab: str,
    ref_col: str,
    cbar_label: str,
    suptitle: str,
    out_path: Path | None,
    cmap: str,
    figsize: tuple[float, float],
    dpi: int,
    show: bool,
    point_size: float,
) -> None:
    """
    Row 0 = inductive, row 1 = transductive. Columns = ssl_methods order.
    """
    nrows, ncols = 2, max(len(ssl_methods), 1)
    fig = plt.figure(figsize=(figsize[0] * max(ncols * 0.42, 1.5), figsize[1] * 1.35))

    # Global color scale from all cells
    all_delta: list[float] = []
    for ev in (inductive_lab, transductive_lab):
        for m in ssl_methods:
            part = sub[
                (sub["eval_setting"] == ev) & (sub["pretrain_method"].astype(str) == m)
            ]
            agg = aggregate_cube_delta(part, ref_col)
            if not agg.empty:
                all_delta.extend(agg["delta"].astype(float).tolist())
    if not all_delta:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(min(all_delta)), float(max(all_delta))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    axes_3d: list = []

    def subplot_index(row: int, col: int) -> int:
        return row * ncols + col + 1

    row_names = ("Inductive", "Transductive")
    row_labs = (inductive_lab, transductive_lab)

    for r, (rname, ev) in enumerate(zip(row_names, row_labs)):
        for c, method in enumerate(ssl_methods):
            ax = fig.add_subplot(nrows, ncols, subplot_index(r, c), projection="3d")
            axes_3d.append(ax)

            part = sub[
                (sub["eval_setting"] == ev) & (sub["pretrain_method"].astype(str) == method)
            ]
            agg = aggregate_cube_delta(part, ref_col)

            if agg.empty:
                ax.set_axis_off()
                ax.text2D(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
            else:
                agg = agg.copy()
                agg["h_mean"] = agg["data__homophily_range"].map(hb.homophily_to_mean)
                agg["y_idx"] = (
                    agg["data__structural_signal"].astype(str).str.strip().map(struct_map)
                )
                agg["z_idx"] = (
                    agg["data__feature_signal"].astype(str).str.strip().map(feat_map)
                )
                agg = agg.dropna(subset=["h_mean", "y_idx", "z_idx"])
                if agg.empty:
                    ax.set_axis_off()
                    ax.text2D(
                        0.5,
                        0.5,
                        "no data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                    )
                else:
                    x = agg["h_mean"].to_numpy(dtype=float)
                    y = agg["y_idx"].to_numpy(dtype=float)
                    z = agg["z_idx"].to_numpy(dtype=float)
                    colv = agg["delta"].to_numpy(dtype=float)
                    
                    # Draw projection lines from points to XY plane (floor) for depth
                    z_min = min(feat_map.values()) if feat_map else 0
                    for xi, yi, zi, cv in zip(x, y, z, colv):
                        color_rgba = plt.cm.get_cmap(cmap)(norm(cv))
                        ax.plot(
                            [xi, xi],
                            [yi, yi],
                            [z_min, zi],
                            color=color_rgba,
                            alpha=0.25,
                            linewidth=0.8,
                            linestyle=":",
                        )
                    
                    # Main scatter with enhanced edge for better visibility
                    ax.scatter(
                        x,
                        y,
                        z,
                        c=colv,
                        cmap=cmap,
                        norm=norm,
                        s=point_size,
                        alpha=0.85,
                        edgecolors="black",
                        linewidths=0.5,
                        depthshade=True,
                    )
                    homophily_label = "Homophily (mean)" if r == 0 else "Homophily"
                    ax.set_xlabel(homophily_label, fontsize=12, fontweight="bold", labelpad=4)
                    show_right_axis_context = c == (ncols - 1)
                    if show_right_axis_context:
                        ax.set_ylabel("Structural", fontsize=12, fontweight="bold", labelpad=7)
                        ax.set_zlabel("Feature", fontsize=12, fontweight="bold", labelpad=7)
                    else:
                        ax.set_ylabel("")
                        ax.set_zlabel("")

                    struct_labels = [
                        k for k, _ in sorted(struct_map.items(), key=lambda kv: kv[1])
                    ]
                    feat_labels = [
                        k for k, _ in sorted(feat_map.items(), key=lambda kv: kv[1])
                    ]
                    ax.set_yticks(range(len(struct_labels)))
                    if show_right_axis_context:
                        ax.set_yticklabels(struct_labels, fontsize=11)
                    else:
                        ax.set_yticklabels([])
                    ax.set_zticks(range(len(feat_labels)))
                    if show_right_axis_context:
                        ax.set_zticklabels(feat_labels, fontsize=11)
                    else:
                        ax.set_zticklabels([])
                    ax.tick_params(axis="x", pad=5)
                    ax.tick_params(axis="y", pad=4)
                    ax.tick_params(axis="z", pad=4)
                    
                    # Draw semi-transparent grid planes at structural/feature tick positions
                    x_min, x_max = ax.get_xlim()
                    for y_tick in range(len(struct_labels)):
                        for z_tick in range(len(feat_labels)):
                            # YZ plane slices (vertical slices at each structural index)
                            xx_grid = np.array([[x_min, x_max], [x_min, x_max]])
                            yy_grid = np.array([[y_tick, y_tick], [y_tick, y_tick]])
                            zz_grid = np.array([[z_tick, z_tick], [z_tick + 1, z_tick + 1]])
                            ax.plot_surface(
                                xx_grid,
                                yy_grid,
                                zz_grid,
                                color="gray",
                                alpha=0.15,
                                shade=False,
                                edgecolor="none",
                            )
                    
                    # Improve viewing angle for better depth perception
                    ax.view_init(elev=35, azim=140)

            if r == 0:
                ax.set_title(method, fontsize=14, fontweight="bold", pad=8)

            # Row label on the left column
            if c == 0:
                ax.text2D(
                    -0.18,
                    0.5,
                    rname,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=13,
                    fontweight="bold",
                )

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=0.98)

    if axes_3d:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes_3d,
            shrink=0.52,
            aspect=26,
            pad=0.10,
            fraction=0.03,
        )
        cbar.set_label(cbar_label, fontsize=17, fontweight="normal")
        cbar.ax.tick_params(labelsize=14)

    fig.subplots_adjust(
        left=0.06,
        right=0.80,
        top=0.90,
        bottom=0.06,
        wspace=0.04,
        hspace=0.18,
    )

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
    src.add_argument("--csv", type=Path, metavar="PATH", help="Unified tidy CSV")
    src.add_argument(
        "--pair",
        action="append",
        type=hb.parse_pair,
        metavar="SETTING=CSV",
        help="Legacy: two or more SETTING=CSV",
    )
    p.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path stem (directory + filename prefix); adds _mode_..._nt_....png",
    )
    p.add_argument(
        "--delta-vs",
        dest="delta_vs",
        choices=("baseline", "random"),
        default="baseline",
        help="Reference column: baseline_accuracy or random_guess",
    )
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[7.0, 5.5],
        metavar=("W", "H"),
        help="Base size per row/column heuristic (figure is expanded for many models)",
    )
    p.add_argument("--show", action="store_true")
    p.add_argument(
        "--point-size",
        type=float,
        default=55.0,
        help="Marker size for 3D scatter",
    )
    p.add_argument(
        "--eval-order",
        type=str,
        default="transductive,inductive",
        help="Same as homophily_bins (for resolving labels)",
    )

    args = p.parse_args()
    ref_col = "baseline_accuracy" if args.delta_vs == "baseline" else "random_guess"
    if args.delta_vs == "baseline":
        cbar_label = (
            r"$\mathbf{\Delta\ Accuracy}$"
            + "\n"
            + r"(pretrained $-$ random-init baseline)"
        )
    else:
        cbar_label = (
            r"$\mathbf{\Delta\ Accuracy}$"
            + "\n"
            + r"(pretrained $-$ random guess $1/K$)"
        )

    print(f"Metric: accuracy - {ref_col}")

    if args.csv is not None:
        merged = hb.load_from_unified_csv(args.csv, ref_col)
        csv_stem = args.csv.stem
    else:
        pairs = args.pair or []
        if len(pairs) < 2:
            raise SystemExit("--pair expects at least two entries.")
        merged = hb.load_merged(pairs, ref_col)
        csv_stem = "merged"

    eval_pref = tuple(x.strip() for x in args.eval_order.split(",") if x.strip())
    eval_labels = hb.order_eval_labels(
        merged["eval_setting"].dropna().astype(str).unique().tolist(),
        eval_pref,
    )
    top_ev, bot_ev = hb.resolve_transductive_inductive(eval_labels)
    if top_ev is None or bot_ev is None:
        raise SystemExit(
            "Need both transductive and inductive in the table. Found: " + str(eval_labels)
        )
    # Rows: inductive (top), transductive (bottom)
    inductive_lab = bot_ev
    transductive_lab = top_ev

    struct_map = hb.build_ordinal_mapping(merged["data__structural_signal"])
    feat_map = hb.build_ordinal_mapping(merged["data__feature_signal"])

    slices = list(hb.iter_trans_ind_ft_slices(merged, top_ev, bot_ev))
    if not slices:
        raise SystemExit("No slices: need shared ft__mode across both settings.")

    prefix = args.output_prefix
    out_dir = prefix.parent
    stem = prefix.name
    out_name = stem if "." not in stem else stem.rsplit(".", 1)[0]
    if not out_name:
        out_name = f"{csv_stem}_ind_trans_3d"

    print(f"Generating {len(slices)} figure(s).")

    for mode, nt_t, nt_i, sub in slices:
        ssl_methods = _ssl_methods(sub)
        if not ssl_methods:
            print(f"  Skip slice mode={mode} nt_t={nt_t} nt_i={nt_i}: no SSL methods")
            continue

        nt_t_disp = int(nt_t) if abs(nt_t - round(nt_t)) < 1e-5 else nt_t
        nt_i_disp = int(nt_i) if abs(nt_i - round(nt_i)) < 1e-5 else nt_i
        suptitle = "Few-shot Improvement Analysis of Pretrained Graph Models"

        safe_mode = re.sub(r"[^\w\-.]+", "_", str(mode).strip())[:60]
        file_base = (
            f"{out_name}_mode_{safe_mode}_nt_trans_{hb._n_train_filename_token(nt_t)}"
            f"_ind_{hb._n_train_filename_token(nt_i)}"
        )
        out_path = out_dir / f"{file_base}.png"

        plot_ind_trans_3d_grid(
            sub=sub,
            struct_map=struct_map,
            feat_map=feat_map,
            ssl_methods=ssl_methods,
            inductive_lab=inductive_lab,
            transductive_lab=transductive_lab,
            ref_col=ref_col,
            cbar_label=cbar_label,
            suptitle=suptitle,
            out_path=out_path,
            cmap=args.cmap,
            figsize=(args.figsize[0], args.figsize[1]),
            dpi=args.dpi,
            show=args.show,
            point_size=args.point_size,
        )
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
