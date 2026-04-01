"""
analyze_downstream_wandb_results.py
===================================
Load downstream evaluation runs from one or more wandb projects and produce
tidy CSVs suitable for plotting (inductive *or* transductive).

Inductive and transductive runs both log flattened ``pretrain/dataset/...`` keys
the same way in ``downstream_eval.py`` / ``downstream_eval_transductive.py``, so
compound ``data__feature_signal`` / ``data__structural_signal`` columns and
auto-detected pretraining hyperparameter columns work for large SSL sweeps
(e.g. thousands of runs).

``evaluation_setting`` is derived from
``pretrain/.../family_parameters/n_graphs``: ``1`` → ``transductive``, ``>1``
→ ``inductive`` (``unknown`` if missing or invalid).

``pretrain_method`` is derived from
``pretrain/dataset/loader/parameters/data_name`` by splitting on ``_`` and
taking the second segment (e.g. ``GraphUniverse_graphmaev2`` → ``graphmaev2``).
Random-init runs are still labeled ``random_init``.

Three output files
------------------
  <o>_raw_rows.csv   - one row per wandb run, flat extracted fields
  <o>.csv            - full tidy table (all methods: SSL + CD + random_init)
  <o>_ssl_only.csv   - SSL rows only, with baseline + oracle filled in as columns

Column naming convention
------------------------
  evaluation_setting   transductive | inductive | unknown (from pretrain n_graphs)
  pretrain_method      second segment of pretrain data_name after '_'; random_init for baselines
  data__*          dataset / graph-generation parameters (from pretraining config)
                   includes two derived compound params:
                     data__feature_signal    (Low / Medium / High)
                     data__structural_signal (Low / Medium / High)
  model__*         general model architecture params (feature_encoder, backbone)
  ssl__*           SSL-specific params (backbone_wrapper, readout)
  train__*         pretraining training params (trainer, optimizer, callbacks)
  ft__*            fine-tuning / downstream-eval params (mode, n_train, lr, ...)

Only parameter columns that VARY across runs are kept (auto-detected).
The two compound signal columns (feature_signal, structural_signal) are always
kept and are derived from center_variance+cluster_variance and
edge_propensity_variance+degree_separation_range respectively.

When attaching ``baseline_accuracy`` (random init) and ``oracle_accuracy`` (CD),
each SSL row is matched on **all** ``ft__*``, ``ssl__*``, and ``train__*`` columns
present in the table (plus core ``data__*`` summary fields and
``evaluation_setting``), so baselines are not collapsed across SSL, fine-tuning,
or transductive/inductive settings.  CD oracle rows omit ``ssl__*`` from the join
so CD runs still match GraphMAE SSL rows.

Join keys treat pandas/NumPy missing values (``NaN`` / ``NA`` / ``None``) as one
canonical sentinel so lookups work (plain ``float('nan')`` would never match
itself in a dict key).

Random-init rows are detected by downstream ``ft__mode`` *or* by
``pretrain_method == "random_init"`` so tidy CSVs round-trip correctly (saved
files already use canonical ``linear-probe`` / ``full-finetune`` instead of
``random-init-*``).

Usage
-----
python analyze_downstream_wandb_results.py \\
    --projects downstream_eval downstream_eval_CD \\
    --output downstream_results.csv

    # with entity prefix:
    --projects myentity/downstream_eval myentity/downstream_eval_CD

``analyze_rq1_results.py`` is a thin compatibility wrapper around this module.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants - compound signal detection
# ---------------------------------------------------------------------------

FEATURE_SIGNAL_MAP = {
    (0.01, 1.0): "Low",
    (0.2,  0.4): "Medium",
    (0.4,  0.2): "High",
}

STRUCTURAL_SIGNAL_MAP = {
    (0.0, 0.0): "Low",
    (0.5, 0.5): "Medium",
    (1.0, 1.0): "High",
    # Some sweeps use max degree separation 0.9 instead of 1.0
    (1.0, 0.9): "High",
}

# The raw keys that compose the two compound signals.
# We still emit them as data__ columns but ALSO derive the compound label from them.
_FEATURE_SIGNAL_KEYS = frozenset({
    "pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/center_variance",
    "pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/cluster_variance",
})
_STRUCTURAL_SIGNAL_KEYS = frozenset({
    "pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/edge_propensity_variance",
    "pretrain/dataset/loader/parameters/generation_parameters/family_parameters/degree_separation_range",
})
# Also treat the homophily key as "handled" so the auto-detector doesn't emit
# a duplicate data__family_parameters__homophily_range alongside data__homophily_range.
_HOMOPHILY_KEY = frozenset({
    "pretrain/dataset/loader/parameters/generation_parameters/family_parameters/homophily_range",
})
_COMPOUND_SIGNAL_KEYS = _FEATURE_SIGNAL_KEYS | _STRUCTURAL_SIGNAL_KEYS | _HOMOPHILY_KEY

# Used to infer transductive (single graph) vs inductive (many graphs) for joins.
_N_GRAPHS_KEY = (
    "pretrain/dataset/loader/parameters/generation_parameters/"
    "family_parameters/n_graphs"
)
_PRETRAIN_DATA_NAME_KEY = "pretrain/dataset/loader/parameters/data_name"

RANDOM_INIT_MODES = frozenset({"random-init-linear-probe", "random-init-full-finetune"})
RANDOM_INIT_TO_MODE = {
    "random-init-linear-probe":  "linear-probe",
    "random-init-full-finetune": "full-finetune",
}

# ---------------------------------------------------------------------------
# Parameter routing: wandb flat-key prefix -> output CSV column prefix.
# Order matters: first match wins.
# ---------------------------------------------------------------------------
_PRETRAIN_ROUTING = [
    # Data / graph generation (most specific first)
    ("pretrain/dataset/loader/parameters/generation_parameters/", "data__"),
    ("pretrain/dataset/",                                         "data__"),
    # SSL-specific (backbone_wrapper before backbone to avoid prefix clash)
    ("pretrain/model/backbone_wrapper/",                          "ssl__"),
    ("pretrain/model/readout/",                                   "ssl__"),
    # General model architecture
    ("pretrain/model/feature_encoder/",                           "model__"),
    ("pretrain/model/backbone/",                                  "model__"),
    ("pretrain/model/",                                           "model__"),
    # Pretraining training
    ("pretrain/trainer/",                                         "train__"),
    ("pretrain/optimizer/",                                       "train__"),
    ("pretrain/callbacks/",                                       "train__"),
    ("pretrain/",                                                 "train__"),  # catch-all
]

# Top-level downstream config keys that are fine-tuning params
_FINETUNE_KEYS = frozenset({
    "mode", "n_train", "epochs", "lr", "batch_size", "patience",
    "classifier_dropout", "input_dropout", "seed",
    "readout_type", "downstream_task", "n_evaluation_graphs",
    "hidden_dim", "num_classes",
    # Transductive downstream (downstream_eval_transductive)
    "n_evaluation", "data_seed", "learning_setting", "task_type", "task_level",
    "few_shot", "num_nodes", "num_train_nodes", "num_val_nodes", "num_test_nodes",
    # Compound pretrain config keys forwarded verbatim at top level
    "graphuniverse_override",
})

# Keys to ignore entirely
_SKIP_KEYS = frozenset({"_project", "_wandb_version", "wandb_version"})

# Pretrain config keys that vary *because of the pretraining method*, not because
# of a real experimental axis.  Still emitted as informational columns, but
# excluded from the cross-method join key so CD/graphmaev2/random_init can match.
_METHOD_SPECIFIC_KEYS = frozenset({
    # Dataset name encodes the method
    "pretrain/dataset/loader/parameters/data_name",
    "pretrain/dataset/loader/parameters/generation_parameters/data_name",
    "pretrain/dataset/parameters/loss_type",
    "pretrain/dataset/parameters/monitor_metric",
    "pretrain/dataset/parameters/task",
    # GPU slot is pure infrastructure noise
    "pretrain/trainer/devices",
    # Model class / wrapper name encodes the method
    "pretrain/model/backbone_wrapper/_target_",
    "pretrain/model/backbone_wrapper/wrapper_name",
    "pretrain/model/readout/_target_",
    "pretrain/model/readout/readout_name",
    "pretrain/model/model_name",
})

# Core data columns that identify the graph / task regime for matching SSL ↔
# random_init ↔ CD.  Method-specific data__ columns stay out of the join on
# purpose (see ``_build_join_key_*``).
_JOIN_KEY_DATA = (
    "evaluation_setting",
    "data__feature_signal",
    "data__structural_signal",
    "data__homophily_range",
)


def _join_key_baseline(df: pd.DataFrame) -> list[str]:
    """
    Match each SSL row to the random-init run in the *same experimental cell*:
    same graph data summary + every fine-tuning axis (ft__) + SSL sweep axes
    (ssl__) + pretraining trainer axes (train__) present in the table.

    Without ssl__/ft__/train__ in the key, every row that only differed on those
    axes incorrectly shared one constant baseline_accuracy.
    """
    keys = [c for c in _JOIN_KEY_DATA if c in df.columns]
    if "data__K" in df.columns:
        keys.append("data__K")
    keys += sorted(c for c in df.columns if c.startswith("ft__"))
    keys += sorted(c for c in df.columns if c.startswith("ssl__"))
    keys += sorted(c for c in df.columns if c.startswith("train__"))
    return keys


def _join_key_oracle(df: pd.DataFrame) -> list[str]:
    """
    CD (oracle) rows usually do not carry the same ssl__ knobs as GraphMAE runs.
    Oracle matching therefore uses data + ft + train only, not ssl__.
    """
    keys = [c for c in _JOIN_KEY_DATA if c in df.columns]
    if "data__K" in df.columns:
        keys.append("data__K")
    keys += sorted(c for c in df.columns if c.startswith("ft__"))
    keys += sorted(c for c in df.columns if c.startswith("train__"))
    return keys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_val(val: Any) -> str:
    """Normalise a config value to a compact, sortable string."""
    if val is None:
        return "null"
    if isinstance(val, (list, tuple)):
        parts = []
        for v in val:
            parts.append(str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
        return "[" + ",".join(parts) + "]"
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return re.sub(r"\s+", "", str(val))


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict with slash-separated keys."""
    out = {}
    for k, v in d.items():
        full = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full))
        else:
            out[full] = v
    return out


def _route_key(wkey: str) -> str | None:
    """
    Map a flat wandb config key to a prefixed CSV column name.
    Returns None if the key should be skipped.
    """
    if wkey in _SKIP_KEYS or wkey.startswith("_"):
        return None
    # Must contain a slash to be a pretrain param (top-level keys are ft params)
    if "/" not in wkey:
        return None
    for wprefix, csv_prefix in _PRETRAIN_ROUTING:
        if wkey.startswith(wprefix):
            suffix = wkey[len(wprefix):].replace("/", "__")
            return csv_prefix + suffix
    return None


# ---------------------------------------------------------------------------
# Compound-signal extractors
# ---------------------------------------------------------------------------

def _feature_signal(flat_cfg: dict) -> str:
    cv  = flat_cfg.get("pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/center_variance")
    clv = flat_cfg.get("pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/cluster_variance")
    if cv is None or clv is None:
        return "unknown"
    return FEATURE_SIGNAL_MAP.get(
        (round(float(cv), 4), round(float(clv), 4)),
        f"cv={cv},clv={clv}",
    )


def _structural_signal(flat_cfg: dict) -> str:
    epv = flat_cfg.get(
        "pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/edge_propensity_variance"
    )
    dsr = flat_cfg.get(
        "pretrain/dataset/loader/parameters/generation_parameters/family_parameters/degree_separation_range"
    )
    if epv is None or dsr is None:
        return "unknown"
    ds = round(float(dsr[0]), 4) if isinstance(dsr, (list, tuple)) else round(float(dsr), 4)
    return STRUCTURAL_SIGNAL_MAP.get(
        (round(float(epv), 4), ds),
        f"epv={epv},ds={ds}",
    )


def _evaluation_setting(flat_cfg: dict) -> str:
    """transductive if n_graphs == 1, inductive if n_graphs > 1, else unknown."""
    raw = flat_cfg.get(_N_GRAPHS_KEY)
    if raw is None:
        return "unknown"
    try:
        n = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)
    except (TypeError, ValueError):
        return "unknown"
    if n == 1:
        return "transductive"
    if n > 1:
        return "inductive"
    return "unknown"


def _detect_pretrain_method(flat_cfg: dict) -> str:
    """
    Second segment of data_name after splitting on '_' (e.g. Prefix_graphmaev2).
    Oracle / CD runs are normalized to 'CD' when the segment matches case-insensitively.
    """
    raw = flat_cfg.get(_PRETRAIN_DATA_NAME_KEY)
    if raw is None:
        return "unknown"
    parts = str(raw).strip().split("_")
    if len(parts) < 2:
        return "unknown"
    seg = parts[1]
    if seg.lower() == "cd":
        return "CD"
    return seg


# ---------------------------------------------------------------------------
# Auto-detection of varied parameters
# ---------------------------------------------------------------------------

def detect_varied_keys(flat_cfgs: list[dict]) -> set[str]:
    """
    Return the set of flat wandb keys that take >1 distinct value across runs.
    Compound-signal component keys are excluded here (they are always emitted
    via the derived compound column).
    """
    value_sets: dict[str, set] = defaultdict(set)
    for cfg in flat_cfgs:
        for k, v in cfg.items():
            if k in _COMPOUND_SIGNAL_KEYS:
                continue
            value_sets[k].add(_fmt_val(v))
    return {
        k for k, vs in value_sets.items()
        if len(vs) > 1 and k not in _METHOD_SPECIFIC_KEYS
    }


# ---------------------------------------------------------------------------
# Wandb loading
# ---------------------------------------------------------------------------

def fetch_runs(project_path: str, filters: dict | None = None) -> list[dict]:
    api = wandb.Api()
    if filters is None:
        filters = {"state": "finished"}
    print(f"  Fetching from {project_path} ... ", end="", flush=True)
    runs_iter = api.runs(project_path, filters=filters)
    rows = []
    for run in runs_iter:
        cfg = dict(run.config)
        cfg["_project"] = project_path
        flat = _flatten_dict(cfg)
        rows.append({
            "run_id":   run.id,
            "run_name": run.name,
            "state":    run.state,
            "config":   cfg,
            "flat_cfg": flat,
            "summary":  dict(run.summary),
            "tags":     list(run.tags or []),
            "project":  project_path,
        })
    print(f"{len(rows)} runs.")
    return rows


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def _extract_row_full(run: dict, varied_keys: set[str] | None) -> dict | None:
    """
    Extract one row. If varied_keys is None, extract all routable pretrain keys
    (used in the first pass to collect values for variance detection).
    If varied_keys is provided, only include keys in that set.
    """
    cfg      = run["config"]
    flat_cfg = run["flat_cfg"]
    summary  = run["summary"]

    mode    = cfg.get("mode")
    n_train = cfg.get("n_train")
    if mode is None:
        return None

    acc = summary.get("test/accuracy") or summary.get("test_accuracy")
    if acc is None:
        return None

    # ---- Compound signals ----
    feat_sig   = _feature_signal(flat_cfg)
    struct_sig = _structural_signal(flat_cfg)

    hom_raw = flat_cfg.get(
        "pretrain/dataset/loader/parameters/generation_parameters/family_parameters/homophily_range"
    )
    hom_range = _fmt_val(hom_raw)

    k_raw = flat_cfg.get(
        "pretrain/dataset/loader/parameters/generation_parameters/universe_parameters/K"
    )
    if k_raw is None:
        k_raw = cfg.get("num_classes")
    k_val        = int(k_raw) if k_raw is not None else None
    random_guess = (1.0 / k_val) if k_val else None

    pretrain_method = (
        "random_init" if mode in RANDOM_INIT_MODES
        else _detect_pretrain_method(flat_cfg)
    )
    eval_setting = _evaluation_setting(flat_cfg)

    # ---- Core fixed columns ----
    row: dict[str, Any] = {
        # Provenance
        "run_id":          run["run_id"],
        "run_name":        run["run_name"],
        "project":         run["project"],
        "pretrain_method": pretrain_method,
        "evaluation_setting": eval_setting,
        # Compound data signals (always present)
        "data__feature_signal":    feat_sig,
        "data__structural_signal": struct_sig,
        "data__homophily_range":   hom_range,
        "data__K":                 k_val,
        # Fine-tuning params (top-level downstream config)
        "ft__mode":               mode,
        "ft__n_train":            n_train,
        "ft__downstream_task":    cfg.get("downstream_task", "community_detection"),
        "ft__epochs":             cfg.get("epochs"),
        "ft__lr":                 cfg.get("lr"),
        "ft__batch_size":         cfg.get("batch_size"),
        "ft__patience":           cfg.get("patience"),
        "ft__classifier_dropout": cfg.get("classifier_dropout"),
        "ft__input_dropout":      cfg.get("input_dropout"),
        "ft__readout_type":       cfg.get("readout_type"),
        "ft__seed":               cfg.get("seed"),
        "ft__n_evaluation_graphs": cfg.get("n_evaluation_graphs"),
        "ft__n_evaluation":      cfg.get("n_evaluation"),
        "ft__data_seed":         cfg.get("data_seed"),
        "ft__learning_setting":   cfg.get("learning_setting"),
        "ft__task_type":          cfg.get("task_type"),
        "ft__task_level":         cfg.get("task_level"),
        "ft__few_shot":           cfg.get("few_shot"),
        # Outcome
        "accuracy":     float(acc),
        "random_guess": random_guess,
    }

    # ---- Auto-detected pretraining param columns ----
    for wkey, val in flat_cfg.items():
        # Filter to varied keys on second pass
        if varied_keys is not None and wkey not in varied_keys:
            continue
        # Compound signal components handled separately above
        if wkey in _COMPOUND_SIGNAL_KEYS:
            continue
        # Skip internal / provenance / fine-tuning keys
        if wkey in _SKIP_KEYS or wkey.startswith("_"):
            continue
        basename = wkey.split("/")[-1]
        if basename in _FINETUNE_KEYS:
            continue

        col = _route_key(wkey)
        if col is None:
            continue
        # Never overwrite the derived compound columns
        if col in ("data__feature_signal", "data__structural_signal"):
            continue

        row[col] = _fmt_val(val)

    return row


def extract_all_rows(runs: list[dict]) -> list[dict]:
    """Two-pass extraction: detect varied keys, then extract with filter."""
    # Pass 1 - collect all routable values
    flat_cfgs = [r["flat_cfg"] for r in runs]
    varied_keys = detect_varied_keys(flat_cfgs)
    print(f"  Auto-detected {len(varied_keys)} varied config keys across all runs.")

    # Show what was detected (grouped by output prefix)
    by_prefix: dict[str, list] = defaultdict(list)
    for wk in sorted(varied_keys):
        col = _route_key(wk)
        if col:
            prefix = col.split("__")[0] + "__"
            by_prefix[prefix].append(col)
        else:
            by_prefix["(skipped)"].append(wk)
    for prefix, cols in sorted(by_prefix.items()):
        print(f"    {prefix}  ->  {cols}")

    # Pass 2 - extract with filter
    rows, skipped = [], 0
    for run in runs:
        row = _extract_row_full(run, varied_keys)
        if row is not None:
            rows.append(row)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} runs (no mode or no accuracy).")
    return rows


# ---------------------------------------------------------------------------
# Build tidy dataframe
# ---------------------------------------------------------------------------

def _join_key_scalar(val: Any) -> Any:
    """
    Normalise values used in baseline/oracle join-key tuples.

    Dict lookup uses equality; ``float('nan') != float('nan')``, so two rows that
    both have missing SSL/FT floats would never match without this.
    """
    if val is None:
        return "__none__"
    if isinstance(val, str) and not val.strip():
        return "__none__"
    try:
        if pd.isna(val):
            return "__na__"
    except (ValueError, TypeError):
        pass
    return val


def _join_tuple(row: pd.Series, key_cols: list[str]) -> tuple:
    def _cell(c: str) -> Any:
        if c not in row.index:
            return _join_key_scalar(None)
        return _join_key_scalar(row[c])

    return tuple(_cell(c) for c in key_cols)


def _col_sort_key(col: str) -> tuple:
    order = [
        "run_id", "run_name", "project", "pretrain_method", "evaluation_setting",
        "data__", "model__", "ssl__", "train__", "ft__",
        "accuracy", "baseline", "oracle", "random_guess", "pct_",
    ]
    for i, p in enumerate(order):
        if col == p or col.startswith(p):
            return (i, col)
    return (len(order), col)


def build_tidy_df(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return df

    # data__ columns for display/sorting (all of them)
    data_cols = sorted(c for c in df.columns if c.startswith("data__"))
    join_key_ri = _join_key_baseline(df)
    join_key_cd = _join_key_oracle(df)
    print(
        f"\n  Baseline join key: {len(join_key_ri)} columns "
        f"(data + ft__ + ssl__ + train__)."
    )
    print(f"  Oracle join key:   {len(join_key_cd)} columns (data + ft__ + train__, no ssl__).")

    # Split: random-init rows are identified by downstream mode *or* by label (re-loaded CSVs
    # already store canonical ft__mode linear-probe / full-finetune, not random-init-*).
    ri_mask = df["ft__mode"].isin(RANDOM_INIT_MODES) | (
        df["pretrain_method"].astype(str) == "random_init"
    )
    cd_mask   = df["pretrain_method"] == "CD"
    main_mask = (~ri_mask) & (~cd_mask)

    df_ri   = df[ri_mask].copy()
    df_cd   = df[cd_mask].copy()
    df_main = df[main_mask].copy()

    # Normalise random-init mode names for matching (idempotent if already canonical).
    df_ri["ft__mode"] = df_ri["ft__mode"].map(
        lambda m: RANDOM_INIT_TO_MODE.get(m, m) if pd.notna(m) else m
    )

    def _make_lookup(sub: pd.DataFrame, key_cols: list[str]) -> dict[tuple, float]:
        lut: dict[tuple, float] = {}
        for _, r in sub.iterrows():
            k = _join_tuple(r, key_cols)
            lut[k] = (lut[k] + r["accuracy"]) / 2.0 if k in lut else r["accuracy"]
        return lut

    ri_lut = _make_lookup(df_ri, join_key_ri)
    cd_lut = _make_lookup(df_cd, join_key_cd)

    # Enrich main (SSL) rows
    records = []
    for _, row in df_main.iterrows():
        k_ri = _join_tuple(row, join_key_ri)
        k_cd = _join_tuple(row, join_key_cd)
        rg = row["random_guess"]
        bl_acc = ri_lut.get(k_ri)
        or_acc = cd_lut.get(k_cd)

        def _pct(acc):
            if acc is None or or_acc is None or rg is None:
                return None
            span = or_acc - rg
            return None if abs(span) < 1e-9 else 100.0 * (acc - rg) / span

        rec = dict(row)
        rec["baseline_accuracy"]      = bl_acc
        rec["oracle_accuracy"]        = or_acc
        rec["pct_optimal_pretrained"] = _pct(row["accuracy"])
        rec["pct_optimal_baseline"]   = _pct(bl_acc)
        records.append(rec)

    # Add placeholder outcome columns to reference pools
    for col in ["baseline_accuracy", "oracle_accuracy",
                "pct_optimal_pretrained", "pct_optimal_baseline"]:
        df_ri[col] = None
        df_cd[col] = None

    full = pd.concat(
        [pd.DataFrame(records), df_ri, df_cd],
        ignore_index=True,
    )

    # Order columns and rows
    ordered = sorted(full.columns, key=_col_sort_key)
    full = full[ordered]
    sort_cols = [c for c in ["pretrain_method", "evaluation_setting"] + data_cols
                 + ["ft__mode", "ft__n_train"]
                 if c in full.columns]
    full = full.sort_values(sort_cols).reset_index(drop=True)
    return full


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_coverage(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)

    data_cols = [c for c in df.columns if c.startswith("data__") and c != "data__K"]

    for method, grp in df.groupby("pretrain_method", sort=True):
        print(f"\n  pretrain_method = {method}  ({len(grp)} rows)")
        for col in ["evaluation_setting"] + data_cols + ["ft__mode", "ft__n_train"]:
            if col in grp.columns:
                vals = sorted(grp[col].dropna().unique().tolist(), key=str)
                print(f"    {col}: {vals}")

    ssl_df = df[~df["pretrain_method"].isin(["random_init", "CD"])]
    if not ssl_df.empty:
        print(f"\n  SSL rows: {len(ssl_df)}")
        print(f"    Missing oracle_accuracy  : {ssl_df['oracle_accuracy'].isna().sum()}")
        print(f"    Missing baseline_accuracy: {ssl_df['baseline_accuracy'].isna().sum()}")

    auto_cols = [c for c in df.columns
                 if any(c.startswith(p) for p in ("model__", "ssl__", "train__"))
                 or (c.startswith("data__") and c not in
                     ("data__feature_signal", "data__structural_signal",
                      "data__homophily_range", "data__K"))]
    if auto_cols:
        print(f"\n  Auto-detected varied param columns ({len(auto_cols)}):")
        for c in sorted(auto_cols):
            vals = sorted(df[c].dropna().unique().tolist(), key=str)
            print(f"    {c}: {vals}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate downstream wandb runs into tidy CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--projects", "-p", nargs="+", required=True,
        help="Wandb project paths ('project' or 'entity/project').",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="downstream_results.csv",
        help="Base output path (default: downstream_results.csv).",
    )
    parser.add_argument(
        "--filters", type=str, default='{"state": "finished"}',
        help='JSON wandb run filters.',
    )
    parser.add_argument(
        "--include-running", action="store_true",
        help="Also include currently running runs.",
    )
    parser.add_argument(
        "--no-save-raw", action="store_true",
        help="Skip writing the _raw_rows.csv file.",
    )
    args = parser.parse_args()

    filters = json.loads(args.filters)
    if args.include_running:
        filters.pop("state", None)

    # 1. Fetch
    print("\n" + "=" * 70)
    print("FETCHING WANDB RUNS")
    print("=" * 70)
    all_runs: list[dict] = []
    for proj in args.projects:
        all_runs.extend(fetch_runs(proj, filters=filters))
    print(f"\nTotal runs fetched: {len(all_runs)}")

    # 2. Two-pass extraction
    print("\nExtracting rows ...")
    raw_rows = extract_all_rows(all_runs)
    print(f"Rows extracted: {len(raw_rows)} / {len(all_runs)}")
    if not raw_rows:
        print("ERROR: No rows extracted.")
        sys.exit(1)

    out_path = Path(args.output)
    stem, parent = out_path.stem, out_path.parent

    # File 1: raw rows
    if not args.no_save_raw:
        raw_path = parent / (stem + "_raw_rows.csv")
        pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
        print(f"\n[1] Raw rows     -> {raw_path}")

    # 3. Build tidy df
    print("\nBuilding tidy DataFrame ...")
    df = build_tidy_df(raw_rows)
    print_coverage(df)

    # File 2: full tidy
    df.to_csv(out_path, index=False)
    print(f"\n[2] Full tidy    -> {out_path}  ({len(df)} rows, {len(df.columns)} cols)")

    # File 3: SSL only
    ssl_only = df[~df["pretrain_method"].isin(["random_init", "CD"])].copy()
    ssl_path = parent / (stem + "_ssl_only.csv")
    ssl_only.to_csv(ssl_path, index=False)
    print(f"[3] SSL only     -> {ssl_path}  ({len(ssl_only)} rows)")

    # Summary of column groups
    print("\nColumn groups:")
    for prefix, label in [
        ("data__",  "Data params   "),
        ("model__", "Model params  "),
        ("ssl__",   "SSL params    "),
        ("train__", "Train params  "),
        ("ft__",    "Fine-tune     "),
    ]:
        cols = [c for c in ssl_only.columns if c.startswith(prefix)]
        if cols:
            print(f"  {label} ({len(cols):2d}): {cols}")

    outcome = [c for c in ["accuracy", "baseline_accuracy", "oracle_accuracy",
                            "random_guess", "pct_optimal_pretrained", "pct_optimal_baseline"]
               if c in ssl_only.columns]
    print(f"  Outcome        ({len(outcome):2d}): {outcome}")

    print("\nFirst 3 SSL rows (key columns):")
    preview_cols = [c for c in [
        "pretrain_method", "evaluation_setting", "data__feature_signal",
        "data__structural_signal",
        "data__homophily_range", "ft__mode", "ft__n_train",
        "accuracy", "baseline_accuracy", "oracle_accuracy",
        "random_guess", "pct_optimal_pretrained", "pct_optimal_baseline",
    ] if c in ssl_only.columns]
    print(ssl_only[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()