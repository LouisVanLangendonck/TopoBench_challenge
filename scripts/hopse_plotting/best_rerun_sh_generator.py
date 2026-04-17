#!/usr/bin/env python3
"""
From a **seed-aggregated** W&B CSV, pick the best validation row per (model, dataset)
(same rule as ``collapse_aggregated_wandb_by_best_val`` / ``main_plot`` / ``table_generator``:
``utils.iter_best_val_group_picks``), then emit **two** bash scripts with the same Hydra commands:

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
``utils.CONFIG_PARAM_KEYS`` (same column contract as ``main_loader`` / seed aggregation). That
includes sweep axes that must be present for correct reruns, for example:

- **HOPSE_G / GPSE** — ``transforms.hopse_encoding.pretrain_model`` (and related
  ``transforms.hopse_encoding.*`` keys) so molpcba vs zinc checkpoints are not dropped.
- **SANN** — ``transforms.sann_encoding.*``, ``model.feature_encoder.selected_dimensions``, etc.

Only non-empty cells become ``key=value`` flags; re-export from W&B after extending
``CONFIG_PARAM_KEYS`` so the seed-aggregated CSV carries those columns.

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
    SEED_COLUMN,
    aggregated_rows_best_validation_per_group,
    hydra_dataset_key_from_loader_identity,
    hydra_overrides_from_aggregated_row,
    load_wandb_export_csv,
    safe_filename_token,
)

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

DEFAULT_PARALLEL_GPUS = "0,1,2,3,4,5,6,7"


def _sort_key_model_dataset(row) -> tuple[str, str]:
    m = str(row.get("model", ""))
    d = str(row.get("dataset", ""))
    return (m, d)


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
        skip_keys=skip_seed,
    )
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
    rows.sort(key=_sort_key_model_dataset)
    return rows


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
    app = [a.replace("\r", "") for a in append_args]

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# Auto-generated: best val per (model, dataset) — run commands one after another.",
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
            lines.append(f"# {model}  |  {dataset}{seed_note}")
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
) -> int:
    skip_seed = set() if keep_row_seed else {SEED_COLUMN}
    rows = _sorted_winner_rows(df, group_cols=group_cols)
    app = [a.replace("\r", "") for a in append_args]

    gpu_bash_array = " ".join(str(g) for g in gpu_ids)
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# Auto-generated: same best-val reruns as best_val_reruns_sequential.sh, but launch in parallel.",
        "# trainer.devices=[GPU] round-robins over GPUS; each job runs in background; wait at end.",
        "",
        "# Optional: match hopse_m.sh thread limits when many jobs share a machine",
        "# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        "",
        f"GPUS=({gpu_bash_array})",
        "_NUM_GPUS=${#GPUS[@]}",
        "_i=0",
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
            lines.append(f"# {model}  |  {dataset}{seed_note}")
            lines.append('_gpu="${GPUS[$((_i % _NUM_GPUS))]}"; _i=$((_i + 1))')
            lines.append(f"{cmd_body} &")
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
        "--group-by",
        metavar="COL",
        nargs="+",
        default=["model", "dataset"],
        help="Group columns for best-val pick (default: model dataset)",
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

    seeds = _parse_data_seeds(str(args.data_seeds).replace("\r", ""))

    common_kw = dict(
        interpreter=args.interpreter,
        data_seeds=seeds,
        append_args=list(args.append_arg),
        keep_row_seed=args.keep_row_seed,
        group_cols=list(args.group_by),
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
        n2 = emit_parallel_rerun_script(df, path=args.output_parallel, gpu_ids=gpus, **common_kw)
        print(
            f"Wrote {n2} parallel command(s) -> {args.output_parallel} "
            f"(GPUS round-robin: {gpus})"
        )


if __name__ == "__main__":
    main()
