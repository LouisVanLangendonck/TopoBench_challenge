"""
Load a pre-trained TopoBench model using Hydra instantiate.

This is an alternative approach that uses Hydra's instantiate function
directly, which is cleaner if you have hydra-core installed.

This works both:
- In a separate repo where topobench is installed via pip
- Locally within the topobench repo

Usage:
    python load_pretrained_model_hydra.py --run_dir /path/to/wandb/run-xxx
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from local repo
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

# TopoBench imports
from topobench.model import TBModel


def load_wandb_config(run_dir: str | Path) -> DictConfig:
    """Load the config.yaml from a wandb run directory as OmegaConf.
    
    Parameters
    ----------
    run_dir : str or Path
        Path to the wandb run directory
    
    Returns
    -------
    DictConfig
        The configuration as OmegaConf DictConfig with 'value' keys unwrapped.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "files" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # Unwrap the 'value' keys that wandb adds
    config = {}
    for key, val in raw_config.items():
        if key.startswith("_"):  # Skip wandb internal keys like _wandb
            continue
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    
    return OmegaConf.create(config)


def get_checkpoint_path_from_summary(run_dir: str | Path) -> str | None:
    """Extract the checkpoint path from wandb-summary.json."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "files" / "wandb-summary.json"
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary.get("best_epoch/checkpoint")


def load_pretrained_model(
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    strict: bool = True,
) -> TBModel:
    """Load a pre-trained model using Hydra instantiate.
    
    Parameters
    ----------
    run_dir : str or Path
        Path to the wandb run directory.
    checkpoint_path : str or Path or None
        Path to checkpoint. If None, reads from wandb-summary.json.
    device : str
        Device to load model on.
    strict : bool
        Whether to strictly match state dict keys.
    
    Returns
    -------
    TBModel
        The loaded model.
    """
    run_dir = Path(run_dir)
    
    # Load configuration
    print(f"Loading configuration from {run_dir}...")
    cfg = load_wandb_config(run_dir)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)
        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided or found in wandb-summary.json")
        print(f"Using checkpoint: {checkpoint_path}")
    
    # Build model using Hydra instantiate
    # This automatically resolves all _target_ references
    print("Building model architecture...")
    model: TBModel = instantiate(
        cfg.model,
        evaluator=instantiate(cfg.evaluator) if "evaluator" in cfg else None,
        loss=instantiate(cfg.loss) if "loss" in cfg else None,
        optimizer=None,  # Not needed for inference
    )
    
    # Load checkpoint
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    
    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Load a pre-trained TopoBench model using Hydra."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--info", action="store_true")
    
    args = parser.parse_args()
    
    model = load_pretrained_model(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    if args.info:
        print(f"\nModel: {model}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    main()

