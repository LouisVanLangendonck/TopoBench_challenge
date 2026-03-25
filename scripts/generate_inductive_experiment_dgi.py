#!/usr/bin/env python3
"""
Generate rq1_graphmae_pretraining_grid.sh with 3x3 (feature x structural) universe blocks.

Each block is one background `python -m topobench ... --multirun &` job with a comment
labeling the feature-signal and structural-signal level.

Run:
  python scripts/generate_small_example.py --wandb-project my_project
  python scripts/generate_small_example.py --wandb-project my_project --num-devices 4 -o scripts/small_grid_graphmae.sh
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 3x3 signal presets (applied together per job; cannot express as independent sweeps)
# ---------------------------------------------------------------------------
FEATURE_SIGNALS = [
    ("Low", {"center_variance": "0.01", "cluster_variance": "1.0"}),
    ("Medium", {"center_variance": "0.2", "cluster_variance": "0.4"}),
    ("High", {"center_variance": "0.4", "cluster_variance": "0.2"}),
]

STRUCTURAL_SIGNALS = [
    ("Low", {"edge_propensity_variance": "0.0", "degree_separation_range": r"\[0.0,0.0\]"}),
    ("Medium", {"edge_propensity_variance": "0.5", "degree_separation_range": r"\[0.5,0.5\]"}),
    ("High", {"edge_propensity_variance": "1.0", "degree_separation_range": r"\[0.9,0.9\]"}),
]


def count_hydra_sweep_options(value: str) -> int:
    """
    Heuristic count of --multirun grid points for one override value.

    - Multiple bracket groups like [a,b],[c,d] -> one dimension per group (homophily sweep).
    - A single bracket group -> one literal (e.g. [30,50], devices).
    - Otherwise comma-separated scalars -> product dimension count.
    """
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


def multirun_product_from_command_block(block: str) -> int:
    """Multiply sweep sizes for all key=value overrides in a topobench invocation."""
    product = 1
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.endswith("\\") and "=" not in line:
            continue
        # Join logical lines: strip trailing \
        if line.endswith("\\"):
            line = line[:-1].strip()
        if "--multirun" in line or "python" in line.split()[0:1]:
            continue
        if "tags=" in line:
            continue
        m = re.match(r"^([a-zA-Z0-9_.]+)=(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key.startswith("#"):
            continue
        n = count_hydra_sweep_options(val)
        product *= n
    return product


# Shared template: placeholders {center_variance}, {cluster_variance},
# {edge_propensity_variance}, {degree_separation_range}, {trainer_devices},
# {wandb_project}, {wandb_tags}
# Paths match configs/dataset/graph/GraphUniverse_graphmaev2.yaml (loader.parameters.generation_parameters).
# No space after '=' before \\[...\\] — a space splits the shell word and breaks Hydra's lexer.
COMMAND_TEMPLATE = r"""python -m topobench \
    dataset=graph/GraphUniverse_dgi \
    model=graph/gps_dgi \
    loss=dgi \
    evaluator=dgi \
    dataset.loader.parameters.generation_parameters.universe_parameters.K=30 \
    dataset.loader.parameters.generation_parameters.universe_parameters.center_variance={center_variance} \
    dataset.loader.parameters.generation_parameters.universe_parameters.cluster_variance={cluster_variance} \
    dataset.loader.parameters.generation_parameters.universe_parameters.edge_propensity_variance={edge_propensity_variance} \
    dataset.loader.parameters.generation_parameters.universe_parameters.feature_dim=15 \
    dataset.loader.parameters.generation_parameters.universe_parameters.seed=42 \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=5000 \
    dataset.loader.parameters.generation_parameters.family_parameters.n_nodes_range=\[30,50\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_communities_range=\[4,6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[2.0,3.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.degree_separation_range={degree_separation_range} \
    dataset.loader.parameters.generation_parameters.family_parameters.power_law_exponent_range=\[1.5,2.5\] \
    trainer.max_epochs=40 \
    trainer.min_epochs=5 \
    model.feature_encoder.out_channels=256 \
    model.feature_encoder.proj_dropout=0.2 \
    model.backbone.num_layers=2,4 \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0 \
    dataset.dataloader_params.batch_size=256 \
    model.backbone_wrapper.corruption_type=graph_diffusion,feature_shuffle \
    model.readout.pooling_type=mean \
    model.readout.readout_type=mean \
    trainer.devices={trainer_devices} \
    trainer.check_val_every_n_epoch=2 \
    callbacks.early_stopping.patience=5 \
    logger.wandb.project={wandb_project} \
    tags="[{wandb_tags}]" \
    --multirun &""".rstrip()


def _trainer_devices_override(device_id: int) -> str:
    """Shell-escaped Hydra list literal for a single GPU index, e.g. \\[0\\]."""
    return rf"\[{device_id}\]"


def _wandb_tags_hydra(wandb_project: str) -> str:
    """Comma-separated tag list inside Hydra tags=[...] (project first, then fixed tags)."""
    return f"{wandb_project},gps,graphmaev2"


def build_block(
    feature_name: str,
    feature_params: dict[str, str],
    struct_name: str,
    struct_params: dict[str, str],
    device_id: int,
    wandb_project: str,
) -> tuple[str, str]:
    trainer_devices = _trainer_devices_override(device_id)
    wandb_tags = _wandb_tags_hydra(wandb_project)
    body = COMMAND_TEMPLATE.format(
        center_variance=feature_params["center_variance"],
        cluster_variance=feature_params["cluster_variance"],
        edge_propensity_variance=struct_params["edge_propensity_variance"],
        degree_separation_range=struct_params["degree_separation_range"],
        trainer_devices=trainer_devices,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
    )
    comment = (
        f"# Feature signal: {feature_name} "
        f"(center_variance={feature_params['center_variance']}, "
        f"cluster_variance={feature_params['cluster_variance']}) | "
        f"Structural signal: {struct_name} "
        f"(edge_propensity_variance={struct_params['edge_propensity_variance']}, "
        f"degree_separation_range={struct_params['degree_separation_range'].replace(chr(92), '')}) | "
        f"trainer.devices=[{device_id}]"
    )
    return comment, body


def generate_script(num_devices: int, wandb_project: str) -> tuple[str, int, int]:
    sections: list[str] = []
    per_job = multirun_product_from_command_block(COMMAND_TEMPLATE.format(
        center_variance="0.2",
        cluster_variance="0.4",
        edge_propensity_variance="1.0",
        degree_separation_range=r"\[0.5,0.5\]",
        trainer_devices=_trainer_devices_override(0),
        wandb_project=wandb_project,
        wandb_tags=_wandb_tags_hydra(wandb_project),
    ))

    cmd_index = 0
    for f_name, f_params in FEATURE_SIGNALS:
        for s_name, s_params in STRUCTURAL_SIGNALS:
            device_id = cmd_index % num_devices
            comment, body = build_block(
                f_name,
                f_params,
                s_name,
                s_params,
                device_id=device_id,
                wandb_project=wandb_project,
            )
            sections.append(comment)
            sections.append(body)
            sections.append("")
            cmd_index += 1

    n_jobs = len(FEATURE_SIGNALS) * len(STRUCTURAL_SIGNALS)
    total_runs = n_jobs * per_job

    header = f"""# Generated by scripts/generate_small_example.py - do not edit by hand.
# Regenerate: python scripts/generate_small_example.py --wandb-project {wandb_project} --num-devices {num_devices} -o <this file>
#
# Grid layout: {len(FEATURE_SIGNALS)} feature-signal levels x {len(STRUCTURAL_SIGNALS)} structural levels = {n_jobs} parallel jobs.
# GPU assignment: commands round-robin trainer.devices over [0..{num_devices - 1}] (wraps after each {num_devices}-th job).
# Hydra --multirun combinations per job (same hyperparameter sweeps): {per_job}
# Total training runs (jobs x per-job grid): {total_runs}
#
"""

    script = header + "\n".join(sections).rstrip() + "\n"
    return script, per_job, total_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root = Path(__file__).resolve().parents[1]
    default_out = root / "scripts" / "run_dgi_grid_inductive.sh"
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help=f"Write generated shell script (default: {default_out})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated script to stdout instead of writing a file",
    )
    parser.add_argument(
        "--wandb-project",
        required=True,
        metavar="NAME",
        help="W&B project (sets logger.wandb.project and the first Hydra tag)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Round-robin trainer.devices for each generated command: "
            "0,1,...,N-1,0,... (default: 4)"
        ),
    )
    args = parser.parse_args()

    if args.num_devices < 1:
        print("--num-devices must be >= 1", file=sys.stderr)
        return 2

    script, per_job, total_runs = generate_script(args.num_devices, args.wandb_project)
    print(
        f"Per-job Hydra grid size: {per_job}",
        file=sys.stderr,
    )
    print(
        f"Parallel jobs: {len(FEATURE_SIGNALS) * len(STRUCTURAL_SIGNALS)} | Total runs: {total_runs}",
        file=sys.stderr,
    )

    if args.stdout:
        sys.stdout.write(script)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
