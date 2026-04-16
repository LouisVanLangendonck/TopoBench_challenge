#!/usr/bin/env python3
"""
From a **seed-aggregated** W&B CSV, pick the best validation row per (model, dataset)
(same rule as ``collapse_aggregated_wandb_by_best_val`` / leaderboard plot), then emit
**two** bash scripts with the same Hydra commands:

1. **Sequential** (default ``scripts/best_val_reruns_sequential.sh``): one ``python -m topobench``
   line after another (no GPU assignment; use ``--append-arg trainer.devices=[0]`` if needed).
2. **Parallel** (default ``scripts/best_val_reruns_parallel.sh``): same runs launched with
   ``&``, ``trainer.devices=[GPU]`` round-robin over ``0..7`` by default (like
   ``topotune/search_gccn_cell.sh``), then ``wait`` for all jobs.

The aggregated export usually drops ``dataset.split_params.data_seed``; this script
appends ``dataset.split_params.data_seed=...`` via ``--data-seed`` (default ``0``) so
each rerun is a single concrete job.

Emitted ``.sh`` files use **LF** line endings only (``newline='\\n'``) so bash/WSL and Hydra
are not broken by Windows CRLF.

Dataset overrides use Hydra config stems (e.g. ``graph/cocitation_cora``): loader
``data_name`` rows in the CSV (e.g. ``graph/Cora``) are rewritten using
``DATASET_LOADER_IDENTITY_TO_HYDRA`` in ``utils``.

By default only **non-transductive** loader datasets are emitted: ``main_loader.DATASETS``
minus ``graph/cocitation_{cora,citeseer,pubmed}``. Use ``--all-datasets`` to emit every
(model, dataset) group in the CSV.

Every command also includes ``trainer.max_epochs`` and ``callbacks.early_stopping.patience``
(sweep defaults: 500 / 10; override with ``--max-epochs`` / ``--early-stopping-patience``).
W&B logging matches ``hopse_m.sh`` style: ``+logger.wandb.entity``, ``logger.wandb.project`` (same
project for every line by default: ``best_runs_rerun``), and ``+logger.wandb.name`` derived from
model/dataset. Disable with ``--no-wandb-logger`` or override via ``--append-arg`` (appended last).

Further extras: ``--append-arg`` (e.g. ``trainer.devices=[0]``; later args override earlier).
On the **parallel** script, ``--append-arg trainer.devices=...`` overrides the round-robin GPU
for that slot (still appended last).

Usage::

    python scripts/hopse_plotting/best_rerun_sh_generator.py
    python scripts/hopse_plotting/best_rerun_sh_generator.py -i scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_seed_agg.csv \\
        -o scripts/best_val_reruns_sequential.sh \\
        --output-parallel scripts/best_val_reruns_parallel.sh
    python scripts/hopse_plotting/best_rerun_sh_generator.py --parallel-gpus 0,1,2,3
    python scripts/hopse_plotting/best_rerun_sh_generator.py --no-parallel-script
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd

from main_loader import DATASETS as LOADER_DATASETS
from utils import (
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

# Match scripts/hopse_m.sh FIXED_ARGS (training length + early stopping only).
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 10

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


def _base_hydra_parts_for_row(
    row,
    *,
    skip_seed: set[str],
    data_seed: str,
    max_epochs: int,
    early_stopping_patience: int,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_run_name: bool,
) -> tuple[str, str, list[str]]:
    """Hydra overrides for one winner row (no ``--append-arg`` extras, no ``trainer.devices``)."""
    model = str(row.get("model", "")).replace("\r", "").strip()
    dataset_raw = str(row.get("dataset", "")).replace("\r", "").strip()
    dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
    parts = hydra_overrides_from_aggregated_row(row, skip_keys=skip_seed)
    if not any(p.startswith(f"{SEED_COLUMN}=") for p in parts):
        parts.append(f"{SEED_COLUMN}={data_seed}")
    parts.append(f"trainer.max_epochs={max_epochs}")
    parts.append(f"callbacks.early_stopping.patience={early_stopping_patience}")
    if wandb_entity and wandb_project:
        parts.append(f"+logger.wandb.entity={wandb_entity}")
        parts.append(f"logger.wandb.project={wandb_project}")
        if wandb_run_name:
            wname = safe_filename_token(
                f"{model.replace('/', '__')}__{dataset.replace('/', '__')}",
                max_len=120,
            )
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
    data_seed: str,
    append_args: list[str],
    keep_row_seed: bool,
    group_cols: list[str],
    max_epochs: int,
    early_stopping_patience: int,
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

    for row in rows:
        model, dataset, base = _base_hydra_parts_for_row(
            row,
            skip_seed=skip_seed,
            data_seed=data_seed,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
        parts = list(base)
        parts.extend(app)
        cmd = shlex.join([interpreter, "-m", "topobench", *parts])
        lines.append(f"# {model}  |  {dataset}")
        lines.append(cmd)
        lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass
    return len(rows)


def emit_parallel_rerun_script(
    df,
    *,
    path: Path,
    interpreter: str,
    data_seed: str,
    append_args: list[str],
    keep_row_seed: bool,
    group_cols: list[str],
    max_epochs: int,
    early_stopping_patience: int,
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

    for row in rows:
        model, dataset, base = _base_hydra_parts_for_row(
            row,
            skip_seed=skip_seed,
            data_seed=data_seed,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
        pre = shlex.join([interpreter, "-m", "topobench", *base])
        post = shlex.join(app) if app else ""
        # Bash sets _gpu then Hydra sees trainer.devices=[0] style (variable expands inside [...]).
        dev_fragment = r"trainer.devices=[${_gpu}]"
        if post:
            cmd_body = f"{pre} {dev_fragment} {post}"
        else:
            cmd_body = f"{pre} {dev_fragment}"
        lines.append(f"# {model}  |  {dataset}")
        lines.append('_gpu="${GPUS[$((_i % _NUM_GPUS))]}"; _i=$((_i + 1))')
        lines.append(f"{cmd_body} &")
        lines.append("")

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
    return len(rows)


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
        "--data-seed",
        default="0",
        help="Appended as dataset.split_params.data_seed=... for each rerun (default: 0)",
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
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "callbacks.early_stopping.patience=... for every command "
            f"(default: {DEFAULT_EARLY_STOPPING_PATIENCE})"
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
        help="Emit dataset.split_params.data_seed from the CSV when present; else use --data-seed.",
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

    common_kw = dict(
        interpreter=args.interpreter,
        data_seed=str(args.data_seed).replace("\r", ""),
        append_args=list(args.append_arg),
        keep_row_seed=args.keep_row_seed,
        group_cols=list(args.group_by),
        max_epochs=int(args.max_epochs),
        early_stopping_patience=int(args.early_stopping_patience),
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
