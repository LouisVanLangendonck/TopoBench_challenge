#!/usr/bin/env python3
"""
From a **seed-aggregated** W&B CSV, pick the best validation row per group (default: model,
dataset, and an internal HOPSE-M F/C shard (``pe`` config bucket → **HOPSE-M-C**) — same val rule as ``utils.iter_best_val_group_picks``
used in ``collapse_aggregated_wandb_by_best_val`` / ``main_plot`` / ``table_generator``), then emit
**two** bash scripts with the same Hydra commands:

1. **Sequential** (default ``scripts/best_val_reruns_sequential.sh``): one ``python -m topobench``
   line after another (no GPU assignment; use ``--append-arg trainer.devices=[0]`` if needed).
2. **Parallel** (default ``scripts/best_val_reruns_parallel.sh``): same runs launched with
   ``&``, ``trainer.devices=[GPU]`` round-robin over ``0..7`` by default (like
   ``topotune/search_gccn_cell.sh``), then ``wait`` for all jobs.

The aggregated export usually drops ``dataset.split_params.data_seed``; this script
appends ``dataset.split_params.data_seed=...`` for each seed in ``--data-seeds`` (default
the sweep set ``0,3,5,7,9``) so reruns match the original multi-seed protocol.

Emitted ``.sh`` files use **LF** line endings only (``newline='\\n'``) so bash/WSL and Hydra
are not broken by Windows CRLF.

Dataset overrides use Hydra config stems (e.g. ``graph/cocitation_cora``): loader
``data_name`` rows in the CSV (e.g. ``graph/Cora``) are rewritten using
``DATASET_LOADER_IDENTITY_TO_HYDRA`` in ``utils``.

By default only **non-transductive** loader datasets are emitted: ``main_loader.DATASETS``
minus ``graph/cocitation_{cora,citeseer,pubmed}``. Use ``--all-datasets`` to emit every
(model, dataset) group in the CSV.

With default ``--group-by model dataset``, **HOPSE-M** (``simplicial/hopse_m`` and ``cell/hopse_m``)
is split like ``table_generator`` / ``hopse_m.sh``: encodings are bucketed into **HOPSE-M-F** vs **HOPSE-M-C**
(``utils.hopse_m_encoding_f_vs_pe_sub_id``: HKFE/HKFE-style → ``f``, else ``pe`` — same sweep
labels as ``fe::`` vs ``pse::`` in ``hopse_m.sh``), and the best-val row is chosen **separately**
per ``(model, dataset, branch)``. **HOPSE-GPSE** and other models use a single winner per
``(model, dataset)``. Override ``--group-by`` to skip the HOPSE-M split (advanced).

At the top of this file, set ``RERUN_MODEL_ALLOWLIST`` / ``RERUN_HOPSE_M_BRANCHES`` to restrict
which models (and optionally which HOPSE-M branch) get commands in the emitted ``.sh`` files.

**Training defaults** mirror the sweep scripts (``gat.sh`` / ``gcn.sh`` / ``hopse_m.sh`` /
``topotune.sh`` / ``sann.sh`` / ``sccnn.sh`` / ``cwn.sh``): ``trainer.min_epochs=50``,
``trainer.check_val_every_n_epoch=5``, plus model-specific extras (``delete_checkpoint_after_test``,
HOPSE preprocessor device, and early-stopping patience 5 vs 10) when ``--fixed-args-profile auto``
(default). TopoTune, SANN, SCCNN, and CWN share the same non-HOPSE block: patience 10 and
``delete_checkpoint_after_test=True``. Use ``--fixed-args-profile none`` to omit those extras
(not recommended for matching sweeps).

Every command includes ``trainer.max_epochs`` (default 500) and ``callbacks.early_stopping.patience``
(either ``--early-stopping-patience INT`` or, if omitted, 5 for ``graph/*{gin,gat,gcn}`` and 10
for HOPSE / TopoTune / SANN / SCCNN / CWN under ``auto``).

**Data seeds:** seed-aggregated CSVs drop ``dataset.split_params.data_seed``. By default this
script emits **one command per sweep seed** (``--data-seeds 0,3,5,7,9``) for each best-val
(model, dataset) row. Use ``--data-seeds 0`` to match the old single-seed behavior. With
``--keep-row-seed``, a per-run export that still has the seed column emits that seed only.

Every rerun also sets ``deterministic=True`` (see ``configs/run.yaml``) so reviewers can
reproduce runs; override last with ``--append-arg deterministic=False`` if needed.

W&B logging matches ``hopse_m.sh`` style: ``+logger.wandb.entity``, ``logger.wandb.project`` (same
project for every line by default: ``best_runs_rerun``), and ``+logger.wandb.name`` derived from
model/dataset (and seed when multiple). Disable with ``--no-wandb-logger`` or override via
``--append-arg`` (appended last).

Further extras: ``--append-arg`` (e.g. ``trainer.devices=[0]``; later args override earlier).
On the **parallel** script, ``--append-arg trainer.devices=...`` overrides the round-robin GPU
for that slot (still appended last).

**Hydra overrides from the winner row** use ``utils.hydra_overrides_from_aggregated_row`` with
``utils.CONFIG_PARAM_KEYS`` (same column contract as ``main_loader`` / seed aggregation; see
the list header comment in ``utils.py`` for sweep script coverage: ``hopse_m`` / ``hopse_g``,
``topotune``, ``sann``, ``sccnn``, ``cwn``, GNN ``transforms`` / ``Combined*``). Non-empty cells
become CLI overrides. **``model.params.total``** is always skipped for reruns (aggregate metric,
not a reproducibility knob). Examples:

- **HOPSE-GPSE** — ``transforms.hopse_encoding.pretrain_model`` (and related
  ``transforms.hopse_encoding.*`` keys) so molpcba vs zinc checkpoints are not dropped.
- **SANN** — ``transforms.sann_encoding.*``, ``model.feature_encoder.selected_dimensions``, etc.
- **SCCNN** — CSV / sweeps may record ``model=simplicial/sccnn`` (TopoModelX backbone). Reruns
  rewrite that override to ``simplicial/sccnn_custom`` (``topobench`` backbone), matching
  stable GPU runs used elsewhere in this repo.
- **``model.backbone.complex_dim``** — most model YAMLs keep ``complex_dim`` on
  ``backbone_wrapper`` / ``readout`` (or omit it on ``SCCNNCustom``), so stray W&B overrides
  are dropped. **Exception:** for **simplicial** SANN (``simplicial/sann``, ``simplicial/sann_online``,
  …) the HOPSE backbone still needs ``++model.backbone.complex_dim`` aligned with
  ``transforms.sann_encoding.complex_dim``; reruns add (or replace) that override accordingly.

Re-export from W&B after extending ``CONFIG_PARAM_KEYS`` so the seed-aggregated CSV carries any new sweep axes.

Usage::

    python scripts/hopse_plotting/best_rerun_sh_generator.py
    python scripts/hopse_plotting/best_rerun_sh_generator.py -i scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_seed_agg.csv \\
        -o scripts/best_val_reruns_sequential.sh \\
        --output-parallel scripts/best_val_reruns_parallel.sh
    python scripts/hopse_plotting/best_rerun_sh_generator.py --parallel-gpus 0,1,2,3
    python scripts/hopse_plotting/best_rerun_sh_generator.py --data-seeds 0
    python scripts/hopse_plotting/best_rerun_sh_generator.py --no-parallel-script
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd

from main_loader import DATASETS as LOADER_DATASETS
from utils import (
    CONFIG_PARAM_KEYS,
    DEFAULT_AGGREGATED_EXPORT_CSV,
    HOPSE_M_MODEL_PATHS,
    MODEL_PREPROC_ENCODINGS,
    SEED_COLUMN,
    aggregated_rows_best_validation_per_group,
    hopse_m_encoding_f_vs_pe_sub_id,
    hydra_dataset_key_from_loader_identity,
    hydra_overrides_from_aggregated_row,
    load_wandb_export_csv,
    safe_filename_token,
)

# -----------------------------------------------------------------------------
# Which reruns to emit (edit here)
# -----------------------------------------------------------------------------
# ``None`` → every model that survives dataset filtering gets a rerun (unchanged behaviour).
# Otherwise only rows whose CSV ``model`` matches these Hydra paths are kept (exact string).
# Exception: allowlisting ``simplicial/sccnn_custom`` also keeps CSV ``simplicial/sccnn`` (and
# the reverse), since sweeps record ``sccnn`` but emitted reruns rewrite to ``sccnn_custom``.
#
# Aligned with ``main_loader.MODELS``: graph short names → ``graph/…``; ``topotune`` /
# ``hopse_*`` / ``sann`` → both ``simplicial/`` and ``cell/`` where sweeps exist;
# ``sccnn`` → ``simplicial/sccnn`` (CSV ``simplicial/sccnn_custom`` still matches via
# ``_csv_model_matches_rerun_allowlist``); ``cwn`` / ``cccn`` → ``cell/…``.
RERUN_MODEL_ALLOWLIST = frozenset(
    {
        "graph/gat",
        "graph/gcn",
        "graph/gin",
        "cell/cccn",
        "cell/cwn",
        "cell/hopse_g",
        "cell/hopse_m",
        "cell/sann",
        "cell/topotune",
        "simplicial/hopse_g",
        "simplicial/hopse_m",
        "simplicial/sann",
        "simplicial/sccnn",
        "simplicial/topotune",
    }
)
#RERUN_MODEL_ALLOWLIST: frozenset[str] | None = None

# For ``simplicial/hopse_m`` and ``cell/hopse_m`` only (requires default ``--group-by`` so the
# FE vs PE shard exists): ``None`` → emit **both** best-val winners (FE branch ``f`` and PE ``pe``).
# ``frozenset({"f"})`` → FE encodings only (HKFE/KHopFE-style, same idea as ``fe::`` in hopse_m.sh);
# ``frozenset({"pe"})`` → PE encodings only (``pse::`` / LapPE-style sweeps).
RERUN_HOPSE_M_BRANCHES: frozenset[str] | None = None

# Repo ``scripts/`` (parent of ``hopse_plotting/``)
_DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMIT_SH_SEQUENTIAL = _DEFAULT_SCRIPTS_DIR / "best_val_reruns_sequential.sh"
DEFAULT_EMIT_SH_PARALLEL = _DEFAULT_SCRIPTS_DIR / "best_val_reruns_parallel.sh"

# Match sweep scripts ``trainer.max_epochs=500``.
DEFAULT_MAX_EPOCHS = 500

# Same order as ``DATA_SEEDS`` in ``gat.sh`` / ``hopse_m.sh`` / ``topotune.sh``.
DEFAULT_SWEEP_DATA_SEEDS = "0,3,5,7,9"

# Match scripts/hopse_m.sh wandb_entity= / logger.wandb.project (single project for all reruns).
DEFAULT_WANDB_ENTITY = "gbg141-hopse"
DEFAULT_WANDB_PROJECT = "best_runs_rerun"

# Same coverage as ``main_loader.DATASETS`` but drop Planetoid cocitation (transductive) configs.
_TRANSDUCTIVE_COCITATION_HYDRA: frozenset[str] = frozenset(
    {
        "graph/cocitation_cora",
        "graph/cocitation_citeseer",
        "graph/cocitation_pubmed",
    }
)
DEFAULT_RERUN_ALLOWED_HYDRA_DATASETS: frozenset[str] = frozenset(
    d for d in LOADER_DATASETS if d not in _TRANSDUCTIVE_COCITATION_HYDRA
)

DEFAULT_PARALLEL_GPUS = "2,3"

# Always excluded from rerun CLI (W&B summary / param count; must not override the model config).
_RERUN_SKIP_HYDRA_KEYS: frozenset[str] = frozenset({"model.params.total"})

# Internal only (not in ``CONFIG_PARAM_KEYS``); used when ``--group-by`` is default ``model dataset``.
_RERUN_HOPSE_M_BRANCH_COL = "_rerun_hopse_m_branch"
_DEFAULT_RERUN_GROUP_COLS: tuple[str, ...] = ("model", "dataset")

# Sweeps use ``simplicial/sccnn`` (``topomodelx``); reruns use the in-repo implementation.
_RERUN_SCCNN_MODEL_CSV = "simplicial/sccnn"
_RERUN_SCCNN_MODEL_HYDRA = "simplicial/sccnn_custom"


def _apply_rerun_model_hydra_rewrites(parts: list[str], *, csv_model: str) -> None:
    """In-place: adjust ``model=`` overrides where the rerun CLI should differ from the CSV row."""
    m = str(csv_model).replace("\r", "").strip()
    if m == _RERUN_SCCNN_MODEL_CSV:
        target = f"model={_RERUN_SCCNN_MODEL_HYDRA}"
        for i, p in enumerate(parts):
            if p.startswith("model="):
                parts[i] = target
                break


_SANN_ENCODING_COMPLEX_DIM_PREFIX = "transforms.sann_encoding.complex_dim="
_BACKBONE_COMPLEX_DIM_PREFIXES = ("++model.backbone.complex_dim=", "model.backbone.complex_dim=")


def _is_backbone_complex_dim_override(part: str) -> bool:
    return part.startswith(_BACKBONE_COMPLEX_DIM_PREFIXES)


def _is_simplicial_sann_model(model: str) -> bool:
    m = str(model).replace("\r", "").strip().lower()
    if not m.startswith("simplicial/"):
        return False
    tail = m.split("/")[-1]
    return tail.startswith("sann")


def _apply_backbone_complex_dim_overrides(parts: list[str], *, model: str) -> None:
    """Drop or sync ``++model.backbone.complex_dim`` (simplicial SANN) / strip legacy single ``model.*``."""
    if _is_simplicial_sann_model(model):
        enc_val: str | None = None
        for p in parts:
            if p.startswith(_SANN_ENCODING_COMPLEX_DIM_PREFIX):
                enc_val = p.split("=", 1)[1]
        parts[:] = [p for p in parts if not _is_backbone_complex_dim_override(p)]
        if enc_val is not None:
            parts.append(f"++model.backbone.complex_dim={enc_val}")
        return
    parts[:] = [p for p in parts if not _is_backbone_complex_dim_override(p)]


def dataframe_with_hopse_m_rerun_branch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``_rerun_hopse_m_branch`` (``f`` / ``pe`` / empty) so HOPSE-M best-val matches
    ``table_generator`` HOPSE-M-F vs HOPSE-M-C picks.
    """
    out = df.copy()
    branches: list[str] = []
    for _idx, row in out.iterrows():
        m = str(row.get("model", "")).strip()
        if m in HOPSE_M_MODEL_PATHS:
            ev = row[MODEL_PREPROC_ENCODINGS] if MODEL_PREPROC_ENCODINGS in row.index else None
            branches.append(hopse_m_encoding_f_vs_pe_sub_id(ev))
        else:
            branches.append("")
    out[_RERUN_HOPSE_M_BRANCH_COL] = branches
    return out


def _sort_key_rerun_row(row) -> tuple[str, str, str]:
    m = str(row.get("model", ""))
    d = str(row.get("dataset", ""))
    b = ""
    if _RERUN_HOPSE_M_BRANCH_COL in row.index:
        b = str(row.get(_RERUN_HOPSE_M_BRANCH_COL, "") or "")
    return (m, d, b)


def dataframe_filter_rerun_datasets(
    df: pd.DataFrame,
    *,
    allowed_hydra: frozenset[str],
) -> pd.DataFrame:
    """
    Keep rows whose ``dataset`` cell maps (via ``hydra_dataset_key_from_loader_identity``)
    into ``allowed_hydra`` (e.g. loader list without cocitation cora/citeseer/pubmed).
    """
    if "dataset" not in df.columns:
        raise KeyError("CSV missing 'dataset' column")

    def canon(ds_val: object) -> str:
        return hydra_dataset_key_from_loader_identity(str(ds_val).replace("\r", "").strip())

    mask = df["dataset"].map(lambda v: canon(v) in allowed_hydra)
    return df.loc[mask].copy()


def _parse_parallel_gpus(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).replace("\r", "").split(","):
        p = part.strip()
        if p:
            out.append(int(p))
    return out if out else [0]


def _parse_data_seeds(s: str) -> list[str]:
    """Comma-separated ints -> string tokens for Hydra (``3`` not ``3.0``)."""
    out: list[str] = []
    for part in str(s).replace("\r", "").split(","):
        p = part.strip()
        if not p:
            continue
        x = float(p)
        out.append(str(int(x)) if x.is_integer() else p)
    return out if out else ["0"]


def _resolve_fixed_args_profile(model: str, profile: str) -> str:
    p = str(profile).replace("\r", "").strip().lower()
    if p == "auto":
        m = str(model).lower().replace("\r", "").strip()
        if "hopse" in m:
            return "hopse"
        if "topotune" in m:
            return "topotune"
        # ``simplicial/sann``, ``cell/sann``, ``simplicial/sann_online``, …
        if m.split("/")[-1].startswith("sann"):
            return "sann"
        # ``scripts/sccnn.sh`` — same FIXED_ARGS as TopoTune / ``scripts/cwn.sh``.
        if m.split("/")[-1].startswith("sccnn"):
            return "sccnn"
        if m.split("/")[-1] == "cwn":
            return "cwn"
        if m.startswith("graph/gin") or m.startswith("graph/gat") or m.startswith("graph/gcn"):
            return "graph"
        if m.startswith("graph/"):
            return "graph"
        # Other models: TopoTune-style extras only — never HOPSE-only CUDA preprocessor.
        return "topotune"
    return p


def _benchmark_training_extras(profile: str) -> list[str]:
    """Pieces of ``FIXED_ARGS`` from sweep scripts (excluding max_epochs / early stopping)."""
    if profile in ("", "none"):
        return []
    out = [
        "trainer.min_epochs=50",
        "trainer.check_val_every_n_epoch=5",
    ]
    if profile == "hopse":
        out.extend(
            [
                "delete_checkpoint_after_test=True",
                "+combined_feature_encodings.preprocessor_device='cuda'",
            ]
        )
    elif profile in ("topotune", "sann", "sccnn", "cwn"):
        out.append("delete_checkpoint_after_test=True")
    return out


def _default_early_stopping_patience(profile: str) -> int:
    return 5 if profile == "graph" else 10


def _row_data_seeds(
    row,
    *,
    keep_row_seed: bool,
    default_seeds: list[str],
) -> list[str]:
    if not keep_row_seed:
        return list(default_seeds)
    if SEED_COLUMN not in row:
        return list(default_seeds)
    raw = row[SEED_COLUMN]
    if pd.isna(raw):
        return list(default_seeds)
    s = str(raw).replace("\r", "").strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return list(default_seeds)
    x = float(s)
    return [str(int(x)) if x.is_integer() else s]


def _base_hydra_parts_for_row(
    row,
    *,
    skip_seed: set[str],
    data_seed: str,
    max_epochs: int,
    early_stopping_patience: int | None,
    fixed_args_profile: str,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: bool,
    wandb_run_suffix: str,
) -> tuple[str, str, list[str]]:
    """Hydra overrides for one winner row (no ``--append-arg`` extras, no ``trainer.devices``)."""
    model = str(row.get("model", "")).replace("\r", "").strip()
    dataset_raw = str(row.get("dataset", "")).replace("\r", "").strip()
    dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
    resolved_profile = _resolve_fixed_args_profile(model, fixed_args_profile)

    parts = hydra_overrides_from_aggregated_row(
        row,
        config_keys=list(CONFIG_PARAM_KEYS),
        skip_keys=set(skip_seed) | set(_RERUN_SKIP_HYDRA_KEYS),
    )
    _apply_rerun_model_hydra_rewrites(parts, csv_model=model)
    _apply_backbone_complex_dim_overrides(parts, model=model)
    parts.extend(_benchmark_training_extras(resolved_profile))
    if not any(p.startswith(f"{SEED_COLUMN}=") for p in parts):
        parts.append(f"{SEED_COLUMN}={data_seed}")
    parts.append(f"trainer.max_epochs={max_epochs}")
    es = (
        early_stopping_patience
        if early_stopping_patience is not None
        else _default_early_stopping_patience(resolved_profile)
    )
    parts.append(f"callbacks.early_stopping.patience={int(es)}")
    parts.append("deterministic=True")
    if wandb_entity and wandb_project:
        parts.append(f"+logger.wandb.entity={wandb_entity}")
        parts.append(f"logger.wandb.project={wandb_project}")
        if wandb_run_name:
            base_nm = f"{model.replace('/', '__')}__{dataset.replace('/', '__')}"
            if _RERUN_HOPSE_M_BRANCH_COL in row.index:
                br = str(row.get(_RERUN_HOPSE_M_BRANCH_COL, "") or "").strip()
                if br == "f":
                    base_nm = f"{base_nm}__hopse_m_F"
                elif br == "pe":
                    base_nm = f"{base_nm}__hopse_m_C"
            if wandb_run_suffix:
                base_nm = f"{base_nm}{wandb_run_suffix}"
            wname = safe_filename_token(base_nm, max_len=120)
            parts.append(f"+logger.wandb.name={wname}")
    return model, dataset, parts


def _sorted_winner_rows(df, *, group_cols: list[str]):
    winners = aggregated_rows_best_validation_per_group(df, group_cols=group_cols)
    if winners.empty:
        raise ValueError("No rows after best-val selection (empty input?)")
    rows = [winners.iloc[i] for i in range(len(winners))]
    rows.sort(key=_sort_key_rerun_row)
    return rows


def _csv_model_matches_rerun_allowlist(m: str, allow: frozenset[str]) -> bool:
    """CSV ``model`` vs ``RERUN_MODEL_ALLOWLIST`` (see ``_RERUN_SCCNN_MODEL_*`` SCCNN rename)."""
    if m in allow:
        return True
    if m == _RERUN_SCCNN_MODEL_CSV and _RERUN_SCCNN_MODEL_HYDRA in allow:
        return True
    if m == _RERUN_SCCNN_MODEL_HYDRA and _RERUN_SCCNN_MODEL_CSV in allow:
        return True
    return False


def _filter_winner_rows_by_allowlist(rows: list) -> list:
    """
    Apply ``RERUN_MODEL_ALLOWLIST`` and ``RERUN_HOPSE_M_BRANCHES`` (see module constants).

    HOPSE-M branch filtering only applies when ``_rerun_hopse_m_branch`` is present on the row
    (default ``--group-by`` with internal HOPSE shard).
    """
    allow = RERUN_MODEL_ALLOWLIST
    branches = RERUN_HOPSE_M_BRANCHES
    if allow is None and branches is None:
        return rows

    out: list = []
    for row in rows:
        m = str(row.get("model", "")).replace("\r", "").strip()
        if allow is not None and not _csv_model_matches_rerun_allowlist(m, allow):
            continue
        if branches is not None and m in HOPSE_M_MODEL_PATHS:
            if _RERUN_HOPSE_M_BRANCH_COL not in row.index:
                continue
            br = str(row.get(_RERUN_HOPSE_M_BRANCH_COL, "") or "").strip()
            if br not in branches:
                continue
        out.append(row)
    return out


def _hopse_m_branch_comment(row) -> str:
    if _RERUN_HOPSE_M_BRANCH_COL not in row.index:
        return ""
    br = str(row.get(_RERUN_HOPSE_M_BRANCH_COL, "") or "").strip()
    if br == "f":
        return "  |  HOPSE-M-F"
    if br == "pe":
        return "  |  HOPSE-M-C"
    return ""


def emit_sequential_rerun_script(
    df,
    *,
    path: Path,
    interpreter: str,
    data_seeds: list[str],
    append_args: list[str],
    keep_row_seed: bool,
    group_cols: list[str],
    max_epochs: int,
    early_stopping_patience: int | None,
    fixed_args_profile: str,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: bool,
) -> int:
    skip_seed = set() if keep_row_seed else {SEED_COLUMN}
    rows = _sorted_winner_rows(df, group_cols=group_cols)
    rows = _filter_winner_rows_by_allowlist(rows)
    if not rows:
        raise ValueError(
            "No rerun rows after RERUN_MODEL_ALLOWLIST / RERUN_HOPSE_M_BRANCHES filter "
            "(see top of best_rerun_sh_generator.py)."
        )
    app = [a.replace("\r", "") for a in append_args]

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# Auto-generated: best val per (model, dataset[, HOPSE-M F/C branch]) — run sequentially.",
        "# Pair script: best_val_reruns_parallel.sh (GPUs in parallel, then wait).",
        "",
    ]

    n_cmd = 0
    for row in rows:
        seeds = _row_data_seeds(row, keep_row_seed=keep_row_seed, default_seeds=data_seeds)
        multi = len(seeds) > 1
        for data_seed in seeds:
            suffix = f"__ds{data_seed}" if multi and wandb_run_name else ""
            model, dataset, base = _base_hydra_parts_for_row(
                row,
                skip_seed=skip_seed,
                data_seed=data_seed,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                fixed_args_profile=fixed_args_profile,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name,
                wandb_run_suffix=suffix,
            )
            parts = list(base)
            parts.extend(app)
            cmd = shlex.join([interpreter, "-m", "topobench", *parts])
            seed_note = f"  |  data_seed={data_seed}" if multi else ""
            lines.append(f"# {model}  |  {dataset}{_hopse_m_branch_comment(row)}{seed_note}")
            lines.append(cmd)
            lines.append("")
            n_cmd += 1

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass
    return n_cmd


def emit_parallel_rerun_script(
    df,
    *,
    path: Path,
    interpreter: str,
    data_seeds: list[str],
    append_args: list[str],
    keep_row_seed: bool,
    group_cols: list[str],
    max_epochs: int,
    early_stopping_patience: int | None,
    fixed_args_profile: str,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: bool,
    gpu_ids: list[int],
    jobs_per_gpu: int = 1,
) -> int:
    skip_seed = set() if keep_row_seed else {SEED_COLUMN}
    rows = _sorted_winner_rows(df, group_cols=group_cols)
    rows = _filter_winner_rows_by_allowlist(rows)
    if not rows:
        raise ValueError(
            "No rerun rows after RERUN_MODEL_ALLOWLIST / RERUN_HOPSE_M_BRANCHES filter "
            "(see top of best_rerun_sh_generator.py)."
        )
    app = [a.replace("\r", "") for a in append_args]

    jpg = max(1, int(jobs_per_gpu))
    gpu_bash_array = " ".join(str(g) for g in gpu_ids)
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# Auto-generated: same best-val reruns as best_val_reruns_sequential.sh, with bounded parallelism.",
        "# Uses virtual GPU slots + wait -n (same pattern as scripts/hopse_m.sh): never launch all jobs at once.",
        "",
        "# Concurrent jobs per physical GPU at generation time; override without regenerating:",
        f'_JOBS_PER_GPU="${{RERUN_JOBS_PER_GPU:-{jpg}}}"',
        "",
        "# Optional: match hopse_m.sh thread limits when multiple jobs share a machine",
        "# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        "",
        f"_PHYSICAL_GPUS=({gpu_bash_array})",
        "gpus=()",
        'for gpu in "${_PHYSICAL_GPUS[@]}"; do',
        '  for ((j=1; j<=_JOBS_PER_GPU; j++)); do gpus+=("$gpu"); done',
        "done",
        'declare -a slot_pids',
        'for i in "${!gpus[@]}"; do slot_pids[$i]=0; done',
        "",
        "_acquire_rerun_slot() {",
        "    assigned_slot=-1",
        '    while [ "$assigned_slot" -eq -1 ]; do',
        '        for i in "${!gpus[@]}"; do',
        '            pid="${slot_pids[$i]}"',
        '            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then',
        "                assigned_slot=$i",
        "                break",
        "            fi",
        "        done",
        '        if [ "$assigned_slot" -eq -1 ]; then',
        "            wait -n",
        "        fi",
        "    done",
        "    _RERUN_SLOT_IDX=$assigned_slot",
        '    _gpu="${gpus[$assigned_slot]}"',
        "}",
        "",
        'echo "Parallel reruns: ${#gpus[@]} slot(s) (${_JOBS_PER_GPU} job(s)/GPU × ${#_PHYSICAL_GPUS[@]} GPU(s))."',
        "",
    ]

    n_cmd = 0
    for row in rows:
        seeds = _row_data_seeds(row, keep_row_seed=keep_row_seed, default_seeds=data_seeds)
        multi = len(seeds) > 1
        for data_seed in seeds:
            suffix = f"__ds{data_seed}" if multi and wandb_run_name else ""
            model, dataset, base = _base_hydra_parts_for_row(
                row,
                skip_seed=skip_seed,
                data_seed=data_seed,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                fixed_args_profile=fixed_args_profile,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name,
                wandb_run_suffix=suffix,
            )
            pre = shlex.join([interpreter, "-m", "topobench", *base])
            post = shlex.join(app) if app else ""
            # Bash sets _gpu then Hydra sees trainer.devices=[0] style (variable expands inside [...]).
            dev_fragment = r"trainer.devices=[${_gpu}]"
            if post:
                cmd_body = f"{pre} {dev_fragment} {post}"
            else:
                cmd_body = f"{pre} {dev_fragment}"
            seed_note = f"  |  data_seed={data_seed}" if multi else ""
            lines.append(f"# {model}  |  {dataset}{_hopse_m_branch_comment(row)}{seed_note}")
            lines.append("_acquire_rerun_slot")
            lines.append(f"{cmd_body} &")
            lines.append("slot_pids[$_RERUN_SLOT_IDX]=$!")
            lines.append("")
            n_cmd += 1

    lines.append("wait")
    lines.append('echo "All parallel reruns finished."')

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass
    return n_cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description="Emit sequential + parallel bash scripts for best-val topobench reruns."
    )
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
        default=DEFAULT_EMIT_SH_SEQUENTIAL,
        help=f"Sequential .sh path (default: {DEFAULT_EMIT_SH_SEQUENTIAL})",
    )
    p.add_argument(
        "--output-parallel",
        type=Path,
        default=DEFAULT_EMIT_SH_PARALLEL,
        help=f"Parallel .sh path (default: {DEFAULT_EMIT_SH_PARALLEL})",
    )
    p.add_argument(
        "--no-parallel-script",
        action="store_true",
        help="Only write the sequential script.",
    )
    p.add_argument(
        "--parallel-gpus",
        default=DEFAULT_PARALLEL_GPUS,
        help=f"Comma-separated GPU indices for round-robin trainer.devices (default: {DEFAULT_PARALLEL_GPUS})",
    )
    p.add_argument(
        "--parallel-jobs-per-gpu",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel script only: concurrent jobs allowed per physical GPU "
            "(virtual slots = N × len(--parallel-gpus)). Default 1. "
            "Override at run time with env RERUN_JOBS_PER_GPU."
        ),
    )
    p.add_argument(
        "--group-by",
        metavar="COL",
        nargs="+",
        default=["model", "dataset"],
        help=(
            "Group columns for best-val pick. Default ``model dataset`` adds an internal "
            "HOPSE-M F vs C shard (same as ``table_generator`` submodels). Pass explicit columns "
            "to disable that behavior (CSV must contain every named column)."
        ),
    )
    p.add_argument(
        "--interpreter",
        default="python",
        help="Python executable (default: python)",
    )
    p.add_argument(
        "--data-seeds",
        default=DEFAULT_SWEEP_DATA_SEEDS,
        help=(
            "Comma-separated dataset.split_params.data_seed values; emits one command "
            f"per best-val row per seed (default: {DEFAULT_SWEEP_DATA_SEEDS})"
        ),
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"trainer.max_epochs=... for every command (default: {DEFAULT_MAX_EPOCHS})",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "callbacks.early_stopping.patience=...; if omitted, uses 5 for graph "
            "gin/gat/gcn and 10 for HOPSE / TopoTune / SANN / SCCNN / CWN under the resolved "
            "--fixed-args-profile (see --fixed-args-profile)."
        ),
    )
    p.add_argument(
        "--fixed-args-profile",
        choices=("auto", "graph", "hopse", "topotune", "sann", "sccnn", "cwn", "none"),
        default="auto",
        help=(
            "Sweep-style extras after row overrides: min_epochs, check_val_every_n_epoch, "
            "and model-specific flags (HOPSE: delete_checkpoint + preprocessor_device; "
            "TopoTune / SANN / SCCNN / CWN: delete_checkpoint only). ``auto`` picks from the "
            "model path (default: auto)."
        ),
    )
    p.add_argument(
        "--append-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override appended after trainer/ES args (repeatable; overrides if key repeats)",
    )
    p.add_argument(
        "--keep-row-seed",
        action="store_true",
        help=(
            "If the CSV still has dataset.split_params.data_seed (per-run export), emit only "
            "that seed per row; otherwise use --data-seeds."
        ),
    )
    p.add_argument(
        "--wandb-entity",
        default=DEFAULT_WANDB_ENTITY,
        help=f"W&B entity for every command (default: {DEFAULT_WANDB_ENTITY})",
    )
    p.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project for every command (default: {DEFAULT_WANDB_PROJECT})",
    )
    p.add_argument(
        "--no-wandb-logger",
        action="store_true",
        help="Do not append logger.wandb entity/project/name overrides.",
    )
    p.add_argument(
        "--no-wandb-run-name",
        action="store_true",
        help="With W&B logging, omit +logger.wandb.name=... (entity and project still set).",
    )
    p.add_argument(
        "--all-datasets",
        action="store_true",
        help=(
            "Do not restrict to main_loader datasets without cocitation trio; include every "
            "dataset present in the CSV."
        ),
    )
    args = p.parse_args()

    wb_ent: str | None = None
    wb_proj: str | None = None
    if not args.no_wandb_logger:
        wb_ent = str(args.wandb_entity).replace("\r", "").strip()
        wb_proj = str(args.wandb_project).replace("\r", "").strip()

    df = load_wandb_export_csv(args.input)
    n_in = len(df)
    if not args.all_datasets:
        df = dataframe_filter_rerun_datasets(df, allowed_hydra=DEFAULT_RERUN_ALLOWED_HYDRA_DATASETS)
        print(
            f"Dataset filter: {n_in} -> {len(df)} rows "
            f"(main_loader.DATASETS minus cocitation cora/citeseer/pubmed; "
            f"{len(DEFAULT_RERUN_ALLOWED_HYDRA_DATASETS)} allowed Hydra paths)"
        )

    if args.keep_row_seed and SEED_COLUMN not in df.columns:
        print(
            f"Note: --keep-row-seed but CSV has no {SEED_COLUMN!r} column "
            f"(expected for seed-aggregated exports); using --data-seeds."
        )

    requested_group = tuple(str(c).strip() for c in args.group_by)
    if requested_group == _DEFAULT_RERUN_GROUP_COLS:
        df = dataframe_with_hopse_m_rerun_branch(df)
        effective_group = list(_DEFAULT_RERUN_GROUP_COLS) + [_RERUN_HOPSE_M_BRANCH_COL]
        print(
            f"Grouping: {effective_group!r} (HOPSE-M: separate best-val winner per F vs C encodings)."
        )
    else:
        effective_group = list(args.group_by)
        missing_g = [c for c in effective_group if c not in df.columns]
        if missing_g:
            raise SystemExit(
                "best_rerun_sh_generator: CSV missing column(s) for --group-by: "
                + ", ".join(repr(c) for c in missing_g)
            )

    seeds = _parse_data_seeds(str(args.data_seeds).replace("\r", ""))

    if RERUN_MODEL_ALLOWLIST is not None or RERUN_HOPSE_M_BRANCHES is not None:
        print(
            f"Rerun filter: RERUN_MODEL_ALLOWLIST={RERUN_MODEL_ALLOWLIST!r}, "
            f"RERUN_HOPSE_M_BRANCHES={RERUN_HOPSE_M_BRANCHES!r}"
        )

    common_kw = dict(
        interpreter=args.interpreter,
        data_seeds=seeds,
        append_args=list(args.append_arg),
        keep_row_seed=args.keep_row_seed,
        group_cols=effective_group,
        max_epochs=int(args.max_epochs),
        early_stopping_patience=args.early_stopping_patience,
        fixed_args_profile=str(args.fixed_args_profile),
        wandb_entity=wb_ent,
        wandb_project=wb_proj,
        wandb_run_name=not args.no_wandb_run_name,
    )

    n = emit_sequential_rerun_script(df, path=args.output, **common_kw)
    print(f"Wrote {n} sequential command(s) -> {args.output}")

    if not args.no_parallel_script:
        gpus = _parse_parallel_gpus(args.parallel_gpus)
        n2 = emit_parallel_rerun_script(
            df,
            path=args.output_parallel,
            gpu_ids=gpus,
            jobs_per_gpu=int(args.parallel_jobs_per_gpu),
            **common_kw,
        )
        slots = len(gpus) * max(1, int(args.parallel_jobs_per_gpu))
        print(
            f"Wrote {n2} parallel command(s) -> {args.output_parallel} "
            f"(GPUs {gpus}, max {slots} concurrent via slot pool)"
        )


if __name__ == "__main__":
    main()
