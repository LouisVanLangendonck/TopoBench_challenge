#!/usr/bin/env python3
"""
Orchestrate GraphMAEv2 pretraining on real-world datasets (IMDB-MULTI, ZINC, NCI1).

Runs ``python -m topobench`` as **separate subprocesses**, each with a single visible GPU
(``CUDA_VISIBLE_DEVICES=<id>`` + ``trainer.devices=[0]``). That avoids Lightning DDP and the
GraphMAEv2 EMA / torchmetrics issues tied to multi-GPU single-process training.

For each dataset, runs either a full Hydra ``--multirun`` grid (default) or a **single**
training run per dataset (``--single-run``, smoke test; no multirun). Datasets run
**sequentially**.

From repo root:
  python downstream_adaptation_real_world_dataset/graphmaev2_pretrain_grid.py \\
    --wandb-project MyProject --gpu 0

Smoke test (one run per dataset; defaults: max_epochs=5, min_epochs=1, val every epoch):
  python ... --wandb-project MyProject --single-run

Dry-run:
  python ... --wandb-project MyProject --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# (short_name, dataset hydra, transforms hydra)
REAL_WORLD_DATASETS: list[tuple[str, str, str]] = [
    ("IMDB-MULTI", "graph/IMDB-MULTI_graphmaev2", "dataset_model_defaults/IMDB-MULTI_graphmaev2_gps_graphmaev2"),
    ("ZINC", "graph/ZINC_graphmaev2", "dataset_model_defaults/ZINC_graphmaev2_gps_graphmaev2"),
    ("NCI1", "graph/NCI1_graphmaev2", "model_defaults/gps_graphmaev2"),
]


def count_hydra_sweep_options(value: str) -> int:
    v = value.strip()
    v_norm = v.replace(r"\[", "[").replace(r"\]", "]")
    groups = re.findall(r"\[[^\]]+\]", v_norm)
    if len(groups) >= 2:
        return len(groups)
    if len(groups) == 1:
        return 1
    if "," in v_norm:
        parts = [p.strip() for p in v_norm.split(",") if p.strip()]
        return max(1, len(parts))
    return 1


def multirun_product(overrides: list[str]) -> int:
    product = 1
    for item in overrides:
        if item == "--multirun" or "=" not in item:
            continue
        key, _, val = item.partition("=")
        if key.startswith("#") or key == "tags":
            continue
        product *= count_hydra_sweep_options(val)
    return product


def build_overrides(
    short_name: str,
    dataset_hydra: str,
    transforms_hydra: str,
    wandb_project: str,
    *,
    max_epochs: int,
    min_epochs: int,
    check_val_every_n_epoch: int,
    early_stopping_patience: int,
    single_run: bool,
) -> list[str]:
    tag_suffix = ",single_run" if single_run else ""
    wandb_tags = f"{wandb_project},gps,graphmaev2,real_world,{short_name}{tag_suffix}"
    # Always single visible device in-process; physical GPU chosen via CUDA_VISIBLE_DEVICES.
    shared = [
        f"dataset={dataset_hydra}",
        f"transforms={transforms_hydra}",
        "model=graph/gps_graphmaev2",
        "loss=graphmaev2",
        "evaluator=graphmaev2",
        f"trainer.max_epochs={max_epochs}",
        f"trainer.min_epochs={min_epochs}",
        "model.feature_encoder.out_channels=256",
        "model.feature_encoder.proj_dropout=0.2",
        "model.backbone.num_layers=2",
        "optimizer.parameters.lr=0.001",
    ]
    tail = [
        "trainer.devices=[0]",
        f"trainer.check_val_every_n_epoch={check_val_every_n_epoch}",
        f"callbacks.early_stopping.patience={early_stopping_patience}",
        f"logger.wandb.project={wandb_project}",
        f"tags=[{wandb_tags}]",
    ]
    if single_run:
        mid = [
            "optimizer.parameters.weight_decay=0",
            "model.backbone_wrapper.momentum=0.99",
            "model.backbone_wrapper.delayed_ema_epoch=0",
            "model.backbone_wrapper.lam=0.8",
            "model.backbone_wrapper.mask_rate=0.8",
            "model.backbone_wrapper.drop_edge_rate=0.0",
            "model.backbone_wrapper.replace_rate=0.0",
            "model.readout.pooling_type=mean",
            "model.readout.decoder_type=gat",
            "model.readout.decoder_hidden_dim=128",
            "model.readout.num_remasking=6",
            "model.readout.remask_rate=0.5",
            "model.readout.remask_method=random",
        ]
        return shared + mid + tail
    mid = [
        "optimizer.parameters.weight_decay=0,0.0001",
        "model.backbone_wrapper.momentum=0.99",
        "model.backbone_wrapper.delayed_ema_epoch=0",
        "model.backbone_wrapper.lam=0.2,0.8",
        "model.backbone_wrapper.mask_rate=0.2,0.8",
        "model.backbone_wrapper.drop_edge_rate=0.0,0.5",
        "model.backbone_wrapper.replace_rate=0.0,0.5",
        "model.readout.pooling_type=mean",
        "model.readout.decoder_type=gat",
        "model.readout.decoder_hidden_dim=128",
        "model.readout.num_remasking=1,6",
        "model.readout.remask_rate=0.0,0.5",
        "model.readout.remask_method=random",
    ]
    return shared + mid + tail + ["--multirun"]


def run_one_dataset(
    repo_root: Path,
    physical_gpu: int,
    overrides: list[str],
    *,
    dry_run: bool,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    cmd = [sys.executable, "-m", "topobench", *overrides]
    print("\n" + "=" * 80)
    print(f"CUDA_VISIBLE_DEVICES={physical_gpu}  {' '.join(cmd)}")
    print("=" * 80)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=repo_root, env=env)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    parser.add_argument("--wandb-project", required=True, metavar="NAME")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="ID",
        help="Physical CUDA device index for every job (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without running.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=(
            "One training run per dataset (no Hydra --multirun). "
            "Default epochs: max=5, min=1, val every epoch (override with --max-epochs etc.)."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Default: 40 (grid) or 5 (--single-run).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="Default: 5 (grid) or 1 (--single-run).",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=None,
        help="Default: 2 (grid) or 1 (--single-run).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Default: 5 (grid or single-run).",
    )
    args = parser.parse_args()

    single = args.single_run
    max_epochs = args.max_epochs if args.max_epochs is not None else (5 if single else 40)
    min_epochs = args.min_epochs if args.min_epochs is not None else (1 if single else 5)
    check_val = (
        args.check_val_every_n_epoch
        if args.check_val_every_n_epoch is not None
        else (1 if single else 2)
    )
    patience = args.early_stopping_patience if args.early_stopping_patience is not None else 5

    sample_ov = build_overrides(
        "ZINC",
        "graph/ZINC_graphmaev2",
        "dataset_model_defaults/ZINC_graphmaev2_gps_graphmaev2",
        args.wandb_project,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        check_val_every_n_epoch=check_val,
        early_stopping_patience=patience,
        single_run=single,
    )
    per_ds = multirun_product(sample_ov)
    total = per_ds * len(REAL_WORLD_DATASETS)
    mode = "single-run test" if single else f"{per_ds} Hydra multirun jobs per dataset"
    print(
        f"Orchestrator: {len(REAL_WORLD_DATASETS)} datasets × {per_ds} run(s) each = {total} total "
        f"({mode}, sequential, GPU {args.gpu})",
        file=sys.stderr,
    )

    exit_code = 0
    for short_name, dataset_hydra, transforms_hydra in REAL_WORLD_DATASETS:
        label = "1 run" if single else f"{per_ds} Hydra jobs"
        print(f"\n>>> Dataset: {short_name} ({label}) <<<", file=sys.stderr)
        overrides = build_overrides(
            short_name,
            dataset_hydra,
            transforms_hydra,
            args.wandb_project,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            check_val_every_n_epoch=check_val,
            early_stopping_patience=patience,
            single_run=single,
        )
        rc = run_one_dataset(repo_root, args.gpu, overrides, dry_run=args.dry_run)
        if rc != 0:
            exit_code = rc
            print(f"!!! Stopping: {short_name} exited with code {rc}", file=sys.stderr)
            break
        print(f">>> Finished {short_name} <<<", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
