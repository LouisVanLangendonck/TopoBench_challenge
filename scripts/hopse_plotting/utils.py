"""
Shared helpers for W&B TopoBench export CSVs: config constants, API helpers,
flattening, and aggregation of runs across data seeds.

Default filesystem layout (under ``scripts/hopse_plotting/``):

- ``csvs/`` — monolithic export, seed-aggregated CSV, collapsed CSV
- ``csvs/hopse_experiments_wandb_export_shards/`` — per-model or per-dataset shards from ``main_loader``
- ``plots/leaderboard/`` — collapse / leaderboard figures (``main_plot``)
- ``plots/hyperparam/`` — ``model/dataset/`` trees from ``hyperparam_analysis``
- ``tables/`` — LaTeX from ``table_generator``
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# -----------------------------------------------------------------------------
# Package paths (scripts/hopse_plotting)
# -----------------------------------------------------------------------------

_PLOT_PACKAGE_ROOT = Path(__file__).resolve().parent
CSV_DIR = _PLOT_PACKAGE_ROOT / "csvs"
WANDB_EXPORT_SHARDS_SUBDIR = "hopse_experiments_wandb_export_shards"
PLOTS_DIR = _PLOT_PACKAGE_ROOT / "plots"
TABLES_DIR = _PLOT_PACKAGE_ROOT / "tables"

DEFAULT_WANDB_EXPORT_CSV = CSV_DIR / "hopse_experiments_wandb_export.csv"
DEFAULT_WANDB_EXPORT_SHARD_DIR = CSV_DIR / WANDB_EXPORT_SHARDS_SUBDIR
DEFAULT_AGGREGATED_EXPORT_CSV = CSV_DIR / "hopse_experiments_wandb_export_seed_agg.csv"
DEFAULT_COLLAPSED_EXPORT_CSV = CSV_DIR / "hopse_experiments_wandb_export_collapsed.csv"
DEFAULT_HYPERPARAM_PLOT_DIR = PLOTS_DIR / "hyperparam"
DEFAULT_LEADERBOARD_PLOT_DIR = PLOTS_DIR / "leaderboard"

# -----------------------------------------------------------------------------
# Column layout (must match main_loader export CSV columns)
# -----------------------------------------------------------------------------

CONFIG_PARAM_KEYS: list[str] = [
    "model",
    "dataset",
    "transforms",
    "transforms.CombinedPSEs.encodings",
    "transforms.CombinedFEs.encodings",
    # SANN sweeps (``scripts/sann.sh``): k-hop transform + backbone/complex dims.
    "transforms.sann_encoding.max_hop",
    "transforms.sann_encoding.complex_dim",
    "transforms.sann_encoding.max_rank",
    # HOPSE_G / GPSE (``scripts/hopse_g.sh``): without ``pretrain_model``, molpcba vs zinc
    # runs merge in seed aggregation (2 checkpoints × 5 seeds → ``n_seeds==10``).
    "transforms.hopse_encoding.pretrain_model",
    "transforms.hopse_encoding.neighborhoods",
    "transforms.hopse_encoding.max_hop",
    "transforms.hopse_encoding.max_rank",
    "transforms.hopse_encoding.complex_dim",
    "model.feature_encoder.selected_dimensions",
    "model.backbone.complex_dim",
    "model.preprocessing_params.neighborhoods",
    "model.preprocessing_params.encodings",
    "model.backbone.neighborhoods",
    "model.backbone.num_layers",
    "model.backbone.n_layers",
    "model.backbone.GNN.num_layers",
    "model.feature_encoder.out_channels",
    "model.feature_encoder.proj_dropout",
    "optimizer.parameters.lr",
    "optimizer.parameters.weight_decay",
    "dataset.dataloader_params.batch_size",
    "dataset.split_params.data_seed",
    "dataset.parameters.monitor_metric",
]

META_COLUMNS: list[str] = [
    "wandb_entity",
    "wandb_project",
    "run_state",
    "identifiers_run_id",
    "identifiers_run_name",
    "identifiers_run_url",
    "identifiers_tags",
]

SEED_COLUMN = "dataset.split_params.data_seed"
MONITOR_METRIC_COLUMN = "dataset.parameters.monitor_metric"
IDENTIFIER_COLUMN_PREFIX = "identifiers_"
SUMMARY_COLUMN_PREFIX = "summary_"

# Hydra / PyTorch often expect ints for these; CSV aggregation yields floats (e.g. 1.0) and breaks
# e.g. torch_geometric GAT: range(num_layers - 2).
HYDRA_WHOLE_NUMBER_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "model.backbone.num_layers",
        "model.backbone.n_layers",
        "model.backbone.GNN.num_layers",
        "model.backbone.complex_dim",
        "model.feature_encoder.out_channels",
        "transforms.sann_encoding.max_hop",
        "transforms.sann_encoding.complex_dim",
        "transforms.sann_encoding.max_rank",
        "transforms.hopse_encoding.max_hop",
        "transforms.hopse_encoding.max_rank",
        "transforms.hopse_encoding.complex_dim",
        "dataset.dataloader_params.batch_size",
        "dataset.split_params.data_seed",
    }
)

# W&B CSV cells often store OmegaConf/JSON lists as ``["a","b"]``; sweep scripts pass
# bracket lists without inner quotes (``[a,b]``). Normalize for CLI reproducibility.
HYDRA_JSON_LIST_TO_BRACKET_KEYS: frozenset[str] = frozenset(
    {
        "transforms.CombinedPSEs.encodings",
        "transforms.CombinedFEs.encodings",
        "model.preprocessing_params.neighborhoods",
        "model.preprocessing_params.encodings",
        "transforms.hopse_encoding.neighborhoods",
        "model.backbone.neighborhoods",
        "model.feature_encoder.selected_dimensions",
    }
)

# W&B / flattened configs record ``dataset.loader.parameters.data_name`` (Planetoid: Cora,
# not cocitation_cora). Hydra ``dataset=`` must match ``configs/dataset/<path>`` without
# ``.yaml``. Add rows here when a loader identity does not equal that path.
# Omitted: ``graph/ZINC`` maps to both ``graph/ZINC`` and ``graph/ZINC_OGB`` (same data_name).
DATASET_LOADER_IDENTITY_TO_HYDRA: dict[str, str] = {
    "graph/Cora": "graph/cocitation_cora",
    "graph/citeseer": "graph/cocitation_citeseer",
    "graph/PubMed": "graph/cocitation_pubmed",
    "graph/manual": "graph/manual_dataset",
    "hypergraph/20newsW100": "hypergraph/20newsgroup",
    "simplicial/MANTRA_genus": "simplicial/mantra_genus",
    "simplicial/MANTRA_name": "simplicial/mantra_name",
    "simplicial/MANTRA_orientation": "simplicial/mantra_orientation",
    "simplicial/MANTRA_betti_numbers": "simplicial/mantra_betti_numbers",
}


def hydra_dataset_key_from_loader_identity(identity: str) -> str:
    """Map loader-style ``domain/data_name`` from exports to Hydra ``dataset=`` key."""
    ident = identity.replace("\r", "").strip()
    if not ident:
        return ident
    return DATASET_LOADER_IDENTITY_TO_HYDRA.get(ident, ident)


# -----------------------------------------------------------------------------
# W&B resilience helpers
# -----------------------------------------------------------------------------


def wandb_transient_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "502",
        "503",
        "504",
        "429",
        "bad gateway",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
    )
    return any(m in text for m in markers)


def run_with_wandb_retry(
    fn,
    *,
    max_retries: int = 6,
    label: str = "W&B API",
):
    last: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == max_retries - 1 or not wandb_transient_api_error(e):
                raise
            delay = min(120.0, (2**attempt) * 10 + random.uniform(0, 3))
            print(
                f"\n  {label} transient error (attempt {attempt + 1}/{max_retries}): {e!s}\n"
                f"  Retrying in {delay:.0f}s ...\n"
            )
            time.sleep(delay)
    assert last is not None
    raise last


# -----------------------------------------------------------------------------
# Config flattening & value extraction (loader)
# -----------------------------------------------------------------------------


def _unwrap_wandb_value(v: Any) -> Any:
    if isinstance(v, dict) and set(v.keys()) <= {"value", "desc", "params"}:
        if "value" in v:
            return _unwrap_wandb_value(v["value"])
    return v


def flatten_config(obj: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(obj, Mapping):
        return {parent_key: obj} if parent_key else {}

    for raw_k, raw_v in obj.items():
        k = str(raw_k)
        if not parent_key and k.startswith("_"):
            continue
        v = _unwrap_wandb_value(raw_v)
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            out.update(flatten_config(v, new_key, sep=sep))
        else:
            out[new_key] = v
    return out


def _serialize_cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int | float) and not isinstance(x, bool):
        return repr(x) if isinstance(x, float) else str(x)
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, sort_keys=True)
    except TypeError:
        return str(x)


def get_from_flat(flat: Mapping[str, Any], dotted: str) -> Any:
    """Resolve a Hydra-style dotted key from W&B ``run.config`` (after ``flatten_config``).

    Lightning/W&B sometimes flatten nested hparams with ``/`` instead of ``.``; try both so
    sweep axes like ``transforms.sann_encoding.max_hop`` are not dropped (else seed
    aggregation can merge 3×5 runs into one bucket with ``n_seeds==15``).
    """
    if dotted in flat:
        return flat[dotted]
    slashy = dotted.replace(".", "/")
    if slashy in flat:
        return flat[slashy]
    return ""


def _resolved_model_path(flat: Mapping[str, Any]) -> str:
    direct = get_from_flat(flat, "model")
    if direct not in (None, ""):
        if isinstance(direct, str):
            return direct
        return _serialize_cell(direct)
    domain = get_from_flat(flat, "model.model_domain")
    name = get_from_flat(flat, "model.model_name")
    if domain and name:
        return f"{domain}/{name}"
    return ""


def _resolved_dataset_path(flat: Mapping[str, Any]) -> str:
    direct = get_from_flat(flat, "dataset")
    if direct not in (None, ""):
        if isinstance(direct, str):
            return hydra_dataset_key_from_loader_identity(direct.strip())
        return hydra_dataset_key_from_loader_identity(_serialize_cell(direct))
    domain = get_from_flat(flat, "dataset.loader.parameters.data_domain")
    name = get_from_flat(flat, "dataset.loader.parameters.data_name")
    if domain and name:
        dd = domain if isinstance(domain, str) else _serialize_cell(domain)
        dn = name if isinstance(name, str) else _serialize_cell(name)
        return hydra_dataset_key_from_loader_identity(f"{dd}/{dn}")
    return ""


def _resolved_transforms_preset(flat: Mapping[str, Any]) -> str:
    direct = get_from_flat(flat, "transforms")
    if direct not in (None, ""):
        if isinstance(direct, str):
            return direct
        return _serialize_cell(direct)
    if get_from_flat(flat, "transforms.CombinedPSEs.encodings"):
        return "combined_pe"
    if get_from_flat(flat, "transforms.CombinedFEs.encodings"):
        return "combined_fe"
    return ""


def extract_config_params(flat: Mapping[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key in CONFIG_PARAM_KEYS:
        if key == "model":
            row[key] = _resolved_model_path(flat)
        elif key == "dataset":
            row[key] = _resolved_dataset_path(flat)
        elif key == "transforms":
            row[key] = _resolved_transforms_preset(flat)
        else:
            row[key] = _serialize_cell(get_from_flat(flat, key))
    return row


def summary_to_prefixed_row(summary: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in summary.items():
        col = f"{SUMMARY_COLUMN_PREFIX}{k}"
        out[col] = _serialize_cell(v)
    return out


def dataset_basename(dataset_path: str) -> str:
    return dataset_path.rsplit("/", 1)[-1]


def expected_project_name(model: str, dataset_path: str) -> str:
    return f"{model}_{dataset_basename(dataset_path)}"


def project_full_path(entity: str, project: str) -> str:
    return f"{entity}/{project}"


def iter_runs(api, entity: str, project: str, *, state: str | None):
    path = project_full_path(entity, project)
    filters = {"state": state} if state else None

    def _list():
        return api.runs(path, filters=filters, per_page=500)

    return run_with_wandb_retry(_list, label=f"W&B list runs {path}")


def run_to_row(
    *,
    entity: str,
    project: str,
    run,
) -> dict[str, Any]:
    flat = flatten_config(dict(run.config))
    meta = {
        "wandb_entity": entity,
        "wandb_project": project,
        "run_state": run.state,
        "identifiers_run_id": run.id,
        "identifiers_run_name": run.name or "",
        "identifiers_run_url": run.url,
        "identifiers_tags": ",".join(run.tags or []),
    }
    params = extract_config_params(flat)
    summ = summary_to_prefixed_row(dict(run.summary))

    return {**meta, **params, **summ}


def collect_all_runs(
    entity: str,
    models: list[str],
    datasets: list[str],
    *,
    run_state: str | None = "finished",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api(timeout=120)
    rows: list[dict[str, Any]] = []

    for model in models:
        for ds in datasets:
            proj = expected_project_name(model, ds)
            if verbose:
                _filt = f"state={run_state}" if run_state else "all states"
                print(f"  (fetch) {entity}/{proj} ({_filt})", flush=True)
            try:
                runs_gen = iter_runs(api, entity, proj, state=run_state)
                count = 0
                for run in runs_gen:
                    rows.append(run_to_row(entity=entity, project=proj, run=run))
                    count += 1
                    if verbose and count % 250 == 0:
                        print(f"    … {count} run(s) so far", flush=True)
            except Exception as e:
                if verbose:
                    print(f"    (skip) {e}", flush=True)
                continue
            if verbose:
                print(f"    -> {count} run(s)", flush=True)
    return rows


def dataframe_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=META_COLUMNS + CONFIG_PARAM_KEYS)

    df = pd.DataFrame(rows)
    summary_cols = sorted(c for c in df.columns if c.startswith(SUMMARY_COLUMN_PREFIX))
    ordered = META_COLUMNS + CONFIG_PARAM_KEYS + summary_cols
    rest = [c for c in df.columns if c not in ordered]
    df = df[[c for c in ordered if c in df.columns] + rest]
    df = df.fillna("")
    return df


# -----------------------------------------------------------------------------
# CSV I/O & seed aggregation
# -----------------------------------------------------------------------------


def load_wandb_export_csv(path: str | Path) -> pd.DataFrame:
    """Read an export CSV produced by ``main_loader``."""
    return pd.read_csv(path, low_memory=False)


def is_seed_aggregatable_summary_column(name: str) -> bool:
    """
    Summary columns to keep when aggregating over seeds: metrics whose W&B key
    path mentions train, val (including val_best_rerun), or test_best_rerun.
    """
    if not name.startswith(SUMMARY_COLUMN_PREFIX):
        return False
    tail = name[len(SUMMARY_COLUMN_PREFIX) :]
    if "train/" in tail or "/train/" in tail:
        return True
    if "val/" in tail or "/val/" in tail:
        return True
    if "test_best_rerun/" in tail:
        return True
    return False


def list_seed_aggregatable_summary_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if is_seed_aggregatable_summary_column(c)]
    return sorted(cols)


def hyperparam_groupby_columns(df: pd.DataFrame) -> list[str]:
    """All columns except identifiers, summary_*, and the data seed."""
    out: list[str] = []
    for c in df.columns:
        if c.startswith(IDENTIFIER_COLUMN_PREFIX):
            continue
        if c.startswith(SUMMARY_COLUMN_PREFIX):
            continue
        if c == SEED_COLUMN:
            continue
        out.append(c)
    return out


def aggregate_wandb_export_by_seed(
    df: pd.DataFrame,
    *,
    seed_column: str = SEED_COLUMN,
    summary_metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    One row per hyperparameter setting (everything equal except identifiers,
    summary columns, and ``seed_column``).

    For each group, ``n_seeds`` is the run count. Selected summary metrics
    (train/..., val/..., test_best_rerun/...) get ``<col>__mean`` and
    ``<col>__std`` (``std`` uses ``ddof=0`` so a single seed yields 0).

    Rows should include ``dataset.parameters.monitor_metric`` (from the loader)
    for downstream collapse / reporting.

    All raw ``summary_*`` columns are dropped from the output; identifier and
    seed columns are dropped. Non-aggregated context (e.g. wandb_entity) is kept.
    """
    missing = [c for c in (seed_column,) if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing expected column(s): {missing}")

    df = df.copy()
    if MONITOR_METRIC_COLUMN not in df.columns:
        df[MONITOR_METRIC_COLUMN] = ""

    group_cols = hyperparam_groupby_columns(df)
    if summary_metric_columns is None:
        summary_metric_columns = list_seed_aggregatable_summary_columns(df)

    unknown = [c for c in summary_metric_columns if c not in df.columns]
    if unknown:
        raise KeyError(f"Unknown summary column(s): {unknown}")

    sub = df[group_cols].copy()
    for c in summary_metric_columns:
        sub[c] = pd.to_numeric(df[c], errors="coerce")

    g = sub.groupby(group_cols, dropna=False)
    n_seeds = g.size().rename("n_seeds")

    mean_df = g[summary_metric_columns].mean()
    mean_df.columns = [f"{c}__mean" for c in mean_df.columns]

    std_df = g[summary_metric_columns].std(ddof=0)
    std_df.columns = [f"{c}__std" for c in std_df.columns]

    # ``n_seeds`` is a Series; concat with empty metric frames (no summary cols / no rows)
    # must be 2D+2D or pandas 2.x raises "unaligned mixed dimensional NDFrame objects".
    out = pd.concat([n_seeds.to_frame(), mean_df, std_df], axis=1).reset_index()

    # Stable metric column order: sort by base summary name, mean then std each
    metric_sorted = sorted(summary_metric_columns)
    tail = []
    for c in metric_sorted:
        tail.append(f"{c}__mean")
        tail.append(f"{c}__std")

    ordered = group_cols + ["n_seeds"] + tail
    out = out[[c for c in ordered if c in out.columns]]
    return out


def build_seed_bucket_report(
    aggregated_df: pd.DataFrame,
    *,
    model_col: str = "model",
    dataset_col: str = "dataset",
    n_seeds_col: str = "n_seeds",
) -> pd.DataFrame:
    """
    Count hyperparameter groups (rows of a seed-aggregated frame) by how many
    raw runs were merged per group, broken down by (model, dataset).

    Columns: ``model``, ``dataset``, ``n_seeds``, ``n_groups``,
    ``pct_of_groups`` (percent of groups within that model+dataset, 0--100).
    """
    if aggregated_df.empty:
        return pd.DataFrame(
            columns=[model_col, dataset_col, n_seeds_col, "n_groups", "pct_of_groups"]
        )
    missing = [c for c in (model_col, dataset_col, n_seeds_col) if c not in aggregated_df.columns]
    if missing:
        raise KeyError(f"seed bucket report: missing column(s): {missing}")

    work = aggregated_df[[model_col, dataset_col, n_seeds_col]].copy()
    work[n_seeds_col] = pd.to_numeric(work[n_seeds_col], errors="coerce").astype("Int64")
    counts = (
        work.groupby([model_col, dataset_col, n_seeds_col], dropna=False)
        .size()
        .rename("n_groups")
        .reset_index()
    )
    totals = counts.groupby([model_col, dataset_col], dropna=False)["n_groups"].transform("sum")
    counts["pct_of_groups"] = (counts["n_groups"] / totals * 100.0).round(2)
    return counts.sort_values([model_col, dataset_col, n_seeds_col]).reset_index(drop=True)


def filter_aggregated_to_required_n_seeds(
    aggregated_df: pd.DataFrame,
    required_n_seeds: int,
    *,
    n_seeds_col: str = "n_seeds",
) -> pd.DataFrame:
    """Keep only hyperparameter groups with exactly ``required_n_seeds`` runs."""
    if n_seeds_col not in aggregated_df.columns:
        raise KeyError(f"filter aggregated: missing {n_seeds_col!r}")
    ns = pd.to_numeric(aggregated_df[n_seeds_col], errors="coerce")
    return aggregated_df.loc[ns == required_n_seeds].copy()


def aggregate_wandb_export_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_metric_columns: list[str] | None = None,
    required_n_seeds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load export CSV, aggregate by seed, optionally keep only groups with an
    exact run count, write ``output_path``.

    Returns ``(written_frame, seed_bucket_report)`` where the report is built
    from the aggregate **before** filtering on ``required_n_seeds``.
    """
    df = load_wandb_export_csv(input_path)
    agg = aggregate_wandb_export_by_seed(df, summary_metric_columns=summary_metric_columns)
    report = build_seed_bucket_report(agg)
    if required_n_seeds is not None:
        agg = filter_aggregated_to_required_n_seeds(agg, required_n_seeds)
    agg = agg.fillna("")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_p, index=False)
    return agg, report


def _union_column_order(frames: list[pd.DataFrame]) -> list[str]:
    """Stable union of column names in first-seen order (for concat alignment)."""
    order: list[str] = []
    seen: set[str] = set()
    for f in frames:
        for c in f.columns:
            if c not in seen:
                seen.add(c)
                order.append(c)
    return order


def aggregate_many_wandb_export_csvs(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    summary_metric_columns: list[str] | None = None,
    required_n_seeds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load multiple per-run export CSVs (e.g. loader shards), aggregate each by
    seed, concatenate rows, optionally filter to an exact ``n_seeds``, and
    write one combined seed-aggregated CSV.

    Shards should partition runs (e.g. one file per model or per dataset) so
    hyperparameter groups are not duplicated across files.

    The seed bucket report is computed on the **concatenated** unfiltered
    aggregate (same keys as a single monolithic export).

    Returns ``(written_frame, seed_bucket_report)``.
    """
    paths = [Path(p) for p in input_paths]
    if not paths:
        raise ValueError("aggregate_many_wandb_export_csvs: no input paths")

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = load_wandb_export_csv(p)
        frames.append(aggregate_wandb_export_by_seed(df, summary_metric_columns=summary_metric_columns))

    cols = _union_column_order(frames)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.reindex(columns=cols)
    report = build_seed_bucket_report(out)
    if required_n_seeds is not None:
        out = filter_aggregated_to_required_n_seeds(out, required_n_seeds)
    out = out.fillna("")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_p, index=False)
    return out, report


# -----------------------------------------------------------------------------
# Collapse seed-aggregated CSV: best hyperparams per (model, dataset, ...)
# -----------------------------------------------------------------------------

# Metric name (last path segment, lowercased) -> "max" or "min" for val split selection.
MONITOR_METRIC_OPTIMIZATION: dict[str, str] = {
    "accuracy": "max",
    "auroc": "max",
    "roc_auc": "max",
    "f1": "max",
    "precision": "max",
    "recall": "max",
    "mae": "min",
    "mse": "min",
    "rmse": "min",
    "loss": "min",
}

DEFAULT_COLLAPSE_GROUP_COLS: list[str] = ["model", "dataset"]


def metric_name_tail(monitor_raw: str) -> str:
    """Normalize ``dataset.parameters.monitor_metric`` to a W&B metric suffix (e.g. ``accuracy``)."""
    m = str(monitor_raw).strip()
    if not m or m.lower() in ("nan", "none"):
        return ""
    if "/" in m:
        return m.rsplit("/", 1)[-1].strip().lower()
    return m.lower()


def safe_metric_col_token(tail: str) -> str:
    """Safe fragment for CSV column names such as ``train_accuracy_mean``."""
    t = re.sub(r"[^\w]+", "_", tail.strip().lower()).strip("_")
    return t or "unknown"


def optimization_mode_for_metric_tail(tail: str) -> str:
    mode = MONITOR_METRIC_OPTIMIZATION.get(tail.strip().lower(), "max")
    return mode if mode in ("max", "min") else "max"


def _first_existing_column(candidates: list[str], available: set[str]) -> str | None:
    for c in candidates:
        if c in available:
            return c
    return None


def _paired_std_from_mean(mean_col: str | None, available: set[str]) -> str | None:
    """``summary_*__mean`` -> matching ``summary_*__std`` if present in the frame."""
    if not mean_col or not str(mean_col).endswith("__mean"):
        return None
    s = str(mean_col)
    std_col = s[: -len("__mean")] + "__std"
    return std_col if std_col in available else None


def _val_mean_columns_for_tail(tail: str) -> list[str]:
    return [
        f"{SUMMARY_COLUMN_PREFIX}val/{tail}__mean",
        f"{SUMMARY_COLUMN_PREFIX}best_epoch/val/{tail}__mean",
        f"{SUMMARY_COLUMN_PREFIX}val_best_rerun/{tail}__mean",
    ]


def _train_mean_columns_for_tail(tail: str) -> list[str]:
    return [
        f"{SUMMARY_COLUMN_PREFIX}train/{tail}__mean",
        f"{SUMMARY_COLUMN_PREFIX}best_epoch/train/{tail}__mean",
    ]


def _test_mean_columns_for_tail(tail: str) -> list[str]:
    return [f"{SUMMARY_COLUMN_PREFIX}test_best_rerun/{tail}__mean"]


def iter_best_val_group_picks(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    monitor_column: str = MONITOR_METRIC_COLUMN,
):
    """
    For each ``group_cols`` group, pick the row index with best validation mean
    (same rule as ``collapse_aggregated_wandb_by_best_val``).

    Yields ``(group_key_tuple, pick_idx, monitor_val, tail)``.
    """
    if group_cols is None:
        group_cols = list(DEFAULT_COLLAPSE_GROUP_COLS)

    missing_g = [c for c in group_cols if c not in df.columns]
    if missing_g:
        raise KeyError(f"collapse: missing group column(s): {missing_g}")
    if monitor_column not in df.columns:
        raise KeyError(f"collapse: missing {monitor_column!r} (re-run loader / aggregate).")

    work = df
    colset = set(work.columns)

    for _gk, sub in work.groupby(group_cols, dropna=False):
        keys = _gk if isinstance(_gk, tuple) else (_gk,)
        if len(keys) != len(group_cols):
            raise RuntimeError("groupby key length mismatch")

        mon_series = (
            sub[monitor_column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": ""})
        )
        mon_series = mon_series[mon_series != ""]
        monitor_val = mon_series.iloc[0] if len(mon_series) else ""

        tail = metric_name_tail(monitor_val)

        pick_idx = sub.index[0]
        val_src = _first_existing_column(_val_mean_columns_for_tail(tail), colset) if tail else None
        if val_src is not None:
            scores = pd.to_numeric(sub[val_src], errors="coerce")
            if scores.notna().any():
                mode = optimization_mode_for_metric_tail(tail)
                pick_idx = scores.idxmax() if mode == "max" else scores.idxmin()

        yield keys, pick_idx, monitor_val, tail


def aggregated_rows_best_validation_per_group(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    monitor_column: str = MONITOR_METRIC_COLUMN,
) -> pd.DataFrame:
    """
    Full **seed-aggregated** rows for the best validation setting in each group
    (same picks as collapse / leaderboard), including all config columns.
    """
    work = df.copy()
    picked: list[pd.Series] = []
    for _keys, pick_idx, _monitor_val, _tail in iter_best_val_group_picks(
        work, group_cols=group_cols, monitor_column=monitor_column
    ):
        picked.append(work.loc[pick_idx])
    if not picked:
        return pd.DataFrame()
    return pd.DataFrame(picked).reset_index(drop=True)


def _serialize_hydra_cli_value(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).replace("\r", "").strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return None
    return s


def normalize_json_list_string_for_hydra_cli(s: str) -> str:
    """
    If ``s`` is a JSON array, return a Hydra-style bracket list ``[a,b,c]`` (no spaces
    after commas): string elements as in ``gat.sh`` / ``hopse_m.sh``; integer elements
    as in ``sann.sh`` (``model.feature_encoder.selected_dimensions``).

    Otherwise return ``s`` unchanged (already ``[a,b]``, not JSON, or invalid).
    """
    t = s.replace("\r", "").strip()
    if len(t) < 2 or t[0] != "[":
        return s
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        return s
    if not isinstance(parsed, list) or not parsed:
        return s
    if all(isinstance(x, str) and x for x in parsed):
        return "[" + ",".join(parsed) + "]"
    if all(isinstance(x, bool) for x in parsed):
        return s
    if all(isinstance(x, int) for x in parsed):
        return "[" + ",".join(str(x) for x in parsed) + "]"
    if all(isinstance(x, (int, float)) for x in parsed):
        try:
            ints: list[int] = []
            for x in parsed:
                xf = float(x)
                if not xf.is_integer():
                    return s
                ints.append(int(xf))
            return "[" + ",".join(str(x) for x in ints) + "]"
        except (TypeError, ValueError):
            return s
    return s


def _coerce_whole_number_override(key: str, s: str) -> str:
    """Emit 1 instead of 1.0 for keys that must be integers in YAML / native code."""
    if key not in HYDRA_WHOLE_NUMBER_OVERRIDE_KEYS or not s:
        return s
    try:
        x = float(s)
    except ValueError:
        return s
    if x.is_integer():
        return str(int(x))
    return s


def hydra_overrides_from_aggregated_row(
    row: Any,
    *,
    config_keys: list[str] | None = None,
    skip_keys: set[str] | None = None,
) -> list[str]:
    """
    Build ``key=value`` strings for ``python -m topobench`` from a loader-style
    config column set (``CONFIG_PARAM_KEYS``). Skips empty / NaN cells.
    """
    if config_keys is None:
        config_keys = list(CONFIG_PARAM_KEYS)
    skip = skip_keys or set()
    out: list[str] = []
    for key in config_keys:
        if key in skip:
            continue
        if key not in row:
            continue
        s = _serialize_hydra_cli_value(row.get(key))
        if s is None:
            continue
        if key == "dataset":
            s = hydra_dataset_key_from_loader_identity(s)
        if key in HYDRA_JSON_LIST_TO_BRACKET_KEYS:
            s = normalize_json_list_string_for_hydra_cli(s)
        s = _coerce_whole_number_override(key, s)
        out.append(f"{key}={s}")
    return out


def collapse_aggregated_wandb_by_best_val(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    monitor_column: str = MONITOR_METRIC_COLUMN,
) -> pd.DataFrame:
    """
    From a **seed-aggregated** export (``...__mean`` / ``...__std`` columns), keep one
    row per ``group_cols`` by picking the hyperparameter row with the best **validation**
    mean for the dataset's monitored metric.

    Output columns: ``group_cols``, ``monitor_column``, then a sparse wide block
    ``train_<metric>_mean``, ``train_<metric>_std``, ``val_<metric>_mean``,
    ``val_<metric>_std``, ``test_<metric>_mean``, ``test_<metric>_std`` for every
    metric tail that appears anywhere in ``monitor_column``; only the block matching
    that row's monitor is filled, others are empty. Std values come from the paired
    ``summary_*__std`` columns of the winning row.
    """
    if group_cols is None:
        group_cols = list(DEFAULT_COLLAPSE_GROUP_COLS)

    missing_g = [c for c in group_cols if c not in df.columns]
    if missing_g:
        raise KeyError(f"collapse: missing group column(s): {missing_g}")
    if monitor_column not in df.columns:
        raise KeyError(f"collapse: missing {monitor_column!r} (re-run loader / aggregate).")

    work = df.copy()
    colset = set(work.columns)

    tails_seen: set[str] = set()
    for v in work[monitor_column].fillna("").astype(str):
        t = metric_name_tail(v)
        if t:
            tails_seen.add(t)

    tokens_sorted = sorted({safe_metric_col_token(t) for t in tails_seen})
    metric_block_cols: list[str] = []
    for tok in tokens_sorted:
        metric_block_cols.extend(
            [
                f"train_{tok}_mean",
                f"train_{tok}_std",
                f"val_{tok}_mean",
                f"val_{tok}_std",
                f"test_{tok}_mean",
                f"test_{tok}_std",
            ]
        )

    out_rows: list[dict[str, Any]] = []

    for keys, pick_idx, monitor_val, tail in iter_best_val_group_picks(
        work, group_cols=group_cols, monitor_column=monitor_column
    ):
        base_row = dict(zip(group_cols, keys, strict=True))
        tok = safe_metric_col_token(tail) if tail else "unknown"

        base_row[monitor_column] = monitor_val

        for c in metric_block_cols:
            base_row[c] = ""

        winner = work.loc[pick_idx]

        if tail:
            train_src = _first_existing_column(_train_mean_columns_for_tail(tail), colset)
            val_src_w = _first_existing_column(_val_mean_columns_for_tail(tail), colset)
            test_src = _first_existing_column(_test_mean_columns_for_tail(tail), colset)

            if train_src:
                base_row[f"train_{tok}_mean"] = winner.get(train_src, "")
                tr_std = _paired_std_from_mean(train_src, colset)
                if tr_std:
                    base_row[f"train_{tok}_std"] = winner.get(tr_std, "")
            if val_src_w:
                base_row[f"val_{tok}_mean"] = winner.get(val_src_w, "")
                va_std = _paired_std_from_mean(val_src_w, colset)
                if va_std:
                    base_row[f"val_{tok}_std"] = winner.get(va_std, "")
            if test_src:
                base_row[f"test_{tok}_mean"] = winner.get(test_src, "")
                te_std = _paired_std_from_mean(test_src, colset)
                if te_std:
                    base_row[f"test_{tok}_std"] = winner.get(te_std, "")

        out_rows.append(base_row)

    out = pd.DataFrame(out_rows)
    ordered = list(group_cols) + [monitor_column] + metric_block_cols
    out = out[[c for c in ordered if c in out.columns]]
    return out.fillna("")


def collapse_aggregated_wandb_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    group_cols: list[str] | None = None,
    monitor_column: str = MONITOR_METRIC_COLUMN,
) -> pd.DataFrame:
    """Load seed-aggregated CSV, collapse to best val per group, write CSV."""
    df = load_wandb_export_csv(input_path)
    collapsed = collapse_aggregated_wandb_by_best_val(
        df, group_cols=group_cols, monitor_column=monitor_column
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(out_p, index=False)
    return collapsed


# -----------------------------------------------------------------------------
# Hyperparameter sensitivity (seed-aggregated CSV, group by model)
# -----------------------------------------------------------------------------


def hyperparam_axis_columns(df: pd.DataFrame) -> list[str]:
    """
    Config columns to treat as hyperparameters for sensitivity plots.

    Uses ``CONFIG_PARAM_KEYS`` present in ``df``, excluding ``model`` (group key)
    and the data-seed column (not present after seed aggregation).
    """
    out: list[str] = []
    for c in CONFIG_PARAM_KEYS:
        if c == "model":
            continue
        if c == SEED_COLUMN:
            continue
        if c in df.columns:
            out.append(c)
    return out


def _nonempty_str_nunique(series: pd.Series) -> int:
    t = series.astype(str).str.strip()
    t = t.mask(t.isin({"", "nan", "None", "NaN", "<NA>"}))
    return int(t.nunique(dropna=True))


def varied_hyperparam_columns(
    df: pd.DataFrame,
    *,
    candidate_cols: list[str] | None = None,
) -> list[str]:
    """Columns among ``candidate_cols`` with more than one distinct non-empty value."""
    if candidate_cols is None:
        candidate_cols = hyperparam_axis_columns(df)
    varied: list[str] = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        if _nonempty_str_nunique(df[c]) > 1:
            varied.append(c)
    return varied


def val_metric_mean_per_row(
    df: pd.DataFrame,
    *,
    monitor_column: str = MONITOR_METRIC_COLUMN,
) -> pd.Series:
    """
    For each row, validation **mean** (seed-aggregated) for that row's
    ``dataset.parameters.monitor_metric``, using the same column resolution
    order as ``collapse_aggregated_wandb_by_best_val``.
    """
    colset = set(df.columns)
    if monitor_column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype="float64")

    def _one(row: pd.Series) -> float:
        tail = metric_name_tail(str(row.get(monitor_column, "")))
        if not tail:
            return float("nan")
        src = _first_existing_column(_val_mean_columns_for_tail(tail), colset)
        if not src:
            return float("nan")
        v = pd.to_numeric(row.get(src, float("nan")), errors="coerce")
        return float(v) if pd.notna(v) else float("nan")

    return df.apply(_one, axis=1)


def infer_hyperparam_plot_kind(
    series: pd.Series,
    *,
    min_scatter_unique: int = 8,
    min_numeric_frac: float = 0.78,
    max_bar_categories: int = 48,
) -> tuple[Literal["scatter", "bar", "skip"], pd.Series]:
    """
    Decide scatter (continuous) vs bar (categorical / low cardinality).

    Returns ``(kind, x_values)`` where for ``scatter``, ``x_values`` is numeric;
    for ``bar``, ``x_values`` is string category labels; for ``skip``, too many
    categories for a readable bar chart.
    """
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    n = len(s)
    if n == 0:
        return "skip", s
    frac_num = float(num.notna().sum()) / float(n)
    n_u_num = int(num.dropna().nunique())

    if frac_num >= min_numeric_frac and n_u_num >= min_scatter_unique:
        return "scatter", num

    lab = s.astype(str).str.strip()
    lab = lab.replace({"": "«empty»", "nan": "«empty»", "None": "«empty»", "NaN": "«empty»"})
    n_u_lab = int(lab.nunique(dropna=False))
    if n_u_lab > max_bar_categories:
        return "skip", lab
    return "bar", lab


def safe_filename_token(name: str, *, max_len: int = 80) -> str:
    """Filesystem-safe fragment from a column name or model id."""
    t = re.sub(r"[^\w.\-]+", "_", str(name).strip()).strip("_")
    if not t:
        t = "unknown"
    return t[:max_len]
