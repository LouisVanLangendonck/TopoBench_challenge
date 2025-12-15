"""
Load a pre-trained TopoBench model from a wandb run folder.

This script demonstrates how to reconstruct a model architecture from a 
wandb run configuration and load the corresponding checkpoint weights.

This works both:
- In a separate repo where topobench is installed via pip
- Locally within the topobench repo (adds parent to sys.path)

Usage:
    python load_pretrained_model.py --run_dir /path/to/wandb/run-xxx --checkpoint /path/to/checkpoint.ckpt
    
    Or if checkpoint path is stored in wandb-summary.json:
    python load_pretrained_model.py --run_dir /path/to/wandb/run-xxx
"""

import argparse
import json
import sys
from pathlib import Path
from functools import partial

# Allow running from local repo (adds parent dir to path if topobench not installed)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml

# TopoBench imports - works both via pip install or local import
from topobench.model import TBModel
from topobench.nn.backbones import MODEL_CLASSES as BACKBONE_CLASSES
from topobench.nn.encoders import FEATURE_ENCODERS
from topobench.nn.readouts import READOUT_CLASSES
from topobench.nn.wrappers import WRAPPER_CLASSES
from topobench.loss import TBLoss
from topobench.evaluator import DGIEvaluator, TBEvaluator, GraphMAEEvaluator, GraphCLEvaluator


def load_wandb_config(run_dir: str | Path) -> dict:
    """Load the config.yaml from a wandb run directory.
    
    Parameters
    ----------
    run_dir : str or Path
        Path to the wandb run directory (e.g., run-20251212_120750-ksr6pbqe)
    
    Returns
    -------
    dict
        The configuration dictionary with 'value' keys unwrapped.
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
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    
    return config


def get_checkpoint_path_from_summary(run_dir: str | Path) -> str | None:
    """Extract the checkpoint path from wandb-summary.json.
    
    Parameters
    ----------
    run_dir : str or Path
        Path to the wandb run directory
    
    Returns
    -------
    str or None
        The checkpoint path if found, None otherwise.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "files" / "wandb-summary.json"
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary.get("best_epoch/checkpoint")


def get_class_from_target(target: str):
    """Get the class from a Hydra _target_ string.
    
    Parameters
    ----------
    target : str
        The _target_ string (e.g., 'topobench.nn.backbones.GPSEncoder')
    
    Returns
    -------
    type
        The class corresponding to the target.
    """
    parts = target.split(".")
    class_name = parts[-1]
    module_path = ".".join(parts[:-1])
    
    # Try to find in known TopoBench class registries first
    if class_name in BACKBONE_CLASSES:
        return BACKBONE_CLASSES[class_name]
    if class_name in FEATURE_ENCODERS:
        return FEATURE_ENCODERS[class_name]
    if class_name in READOUT_CLASSES:
        return READOUT_CLASSES[class_name]
    if class_name in WRAPPER_CLASSES:
        return WRAPPER_CLASSES[class_name]
    
    # Fallback: dynamic import
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_from_config(config: dict, **override_kwargs):
    """Instantiate a class from a Hydra-style config dict.
    
    Parameters
    ----------
    config : dict
        Configuration dict with '_target_' key and optional parameters.
    **override_kwargs
        Additional kwargs to override config values.
    
    Returns
    -------
    object
        The instantiated object.
    """
    if config is None:
        return None
    
    config = dict(config)  # Make a copy
    target = config.pop("_target_")
    is_partial = config.pop("_partial_", False)
    
    # Remove any nested _target_ refs that shouldn't be passed
    config.update(override_kwargs)
    
    cls = get_class_from_target(target)
    
    if is_partial:
        return partial(cls, **config)
    
    return cls(**config)


def build_model_from_config(config: dict, for_inference: bool = True) -> TBModel:
    """Build a TBModel from a wandb config dictionary.
    
    Parameters
    ----------
    config : dict
        The configuration dictionary (already unwrapped from wandb format).
    for_inference : bool
        If True, creates minimal components needed for inference only.
        If False, creates full training setup including evaluator and optimizer.
    
    Returns
    -------
    TBModel
        The instantiated model (without weights loaded).
    """
    model_config = config["model"]
    
    # 1. Build Feature Encoder
    feature_encoder_config = model_config.get("feature_encoder")
    if feature_encoder_config:
        feature_encoder = instantiate_from_config(feature_encoder_config)
    else:
        feature_encoder = None
    
    # 2. Build Backbone
    backbone_config = model_config["backbone"]
    backbone = instantiate_from_config(backbone_config)
    
    # 3. Build Backbone Wrapper (if present)
    wrapper_config = model_config.get("backbone_wrapper")
    if wrapper_config:
        backbone_wrapper = instantiate_from_config(wrapper_config)
    else:
        backbone_wrapper = None
    
    # 4. Build Readout
    readout_config = model_config["readout"]
    readout = instantiate_from_config(readout_config)
    
    # 5. Build Loss
    loss_config = config.get("loss", {})
    if loss_config:
        dataset_loss_config = loss_config.get("dataset_loss")
        if dataset_loss_config:
            dataset_loss = instantiate_from_config(dataset_loss_config)
        else:
            dataset_loss = {"task": "classification", "loss_type": "cross_entropy"}
        
        modules_losses_config = loss_config.get("modules_losses", {})
        modules_losses = {}
        for name, mloss_config in modules_losses_config.items():
            if mloss_config is not None:
                modules_losses[name] = instantiate_from_config(mloss_config)
        
        loss = TBLoss(dataset_loss=dataset_loss, modules_losses=modules_losses)
    else:
        # Minimal loss for inference
        loss = TBLoss(dataset_loss={"task": "classification", "loss_type": "cross_entropy"})
    
    # 6. Build Evaluator (optional, mainly for training/validation)
    evaluator = None
    if not for_inference:
        evaluator_config = config.get("evaluator")
        if evaluator_config:
            evaluator = instantiate_from_config(evaluator_config)
    
    # 7. Assemble the model
    model = TBModel(
        backbone=backbone,
        readout=readout,
        loss=loss,
        backbone_wrapper=backbone_wrapper,
        feature_encoder=feature_encoder,
        evaluator=evaluator,
        optimizer=None,  # Not needed for inference
        compile=False,
    )
    
    return model


def load_model_weights(model: TBModel, checkpoint_path: str | Path, strict: bool = True) -> TBModel:
    """Load weights from a checkpoint into the model.
    
    Parameters
    ----------
    model : TBModel
        The model to load weights into.
    checkpoint_path : str or Path
        Path to the checkpoint file.
    strict : bool
        Whether to strictly enforce that the keys in state_dict match.
    
    Returns
    -------
    TBModel
        The model with loaded weights.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get the state dict (PyTorch Lightning saves it under 'state_dict' key)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Load the weights
    model.load_state_dict(state_dict, strict=strict)
    
    return model


def load_pretrained_model(
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    for_inference: bool = True,
    strict: bool = True,
    device: str = "cpu",
) -> TBModel:
    """Load a pre-trained model from a wandb run directory.
    
    This is the main entry point for loading a pre-trained TopoBench model.
    
    Parameters
    ----------
    run_dir : str or Path
        Path to the wandb run directory.
    checkpoint_path : str or Path or None
        Path to the checkpoint file. If None, tries to extract from wandb-summary.json.
    for_inference : bool
        If True, creates minimal components needed for inference only.
    strict : bool
        Whether to strictly enforce that checkpoint keys match model keys.
    device : str
        Device to load the model on ('cpu', 'cuda', 'cuda:0', etc.).
    
    Returns
    -------
    TBModel
        The loaded model ready for inference or further training.
    
    Examples
    --------
    >>> model = load_pretrained_model(
    ...     run_dir="data/outputs/wandb/run-20251212_120750-ksr6pbqe",
    ...     checkpoint_path="data/outputs/checkpoints/epoch_109.ckpt",
    ... )
    >>> model.eval()
    >>> with torch.no_grad():
    ...     output = model(batch)
    """
    run_dir = Path(run_dir)
    
    # Load configuration
    print(f"Loading configuration from {run_dir}...")
    config = load_wandb_config(run_dir)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint path provided and none found in wandb-summary.json. "
                "Please provide --checkpoint argument."
            )
        print(f"Using checkpoint from wandb-summary.json: {checkpoint_path}")
    
    # Build model architecture
    print("Building model architecture...")
    model = build_model_from_config(config, for_inference=for_inference)
    
    # Load weights
    print(f"Loading weights from {checkpoint_path}...")
    model = load_model_weights(model, checkpoint_path, strict=strict)
    
    # Move to device and set to eval mode
    model = model.to(device)
    if for_inference:
        model.eval()
    
    print("Model loaded successfully!")
    return model


def extract_backbone(model: TBModel) -> torch.nn.Module:
    """Extract just the backbone (encoder) from a loaded TBModel.
    
    Useful when you want to use the pre-trained encoder for downstream tasks.
    
    Parameters
    ----------
    model : TBModel
        The loaded TBModel.
    
    Returns
    -------
    torch.nn.Module
        The backbone module.
    """
    return model.backbone


def extract_feature_encoder(model: TBModel) -> torch.nn.Module:
    """Extract the feature encoder from a loaded TBModel.
    
    Parameters
    ----------
    model : TBModel
        The loaded TBModel.
    
    Returns
    -------
    torch.nn.Module
        The feature encoder module.
    """
    return model.feature_encoder


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load a pre-trained TopoBench model from a wandb run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the wandb run directory (e.g., data/outputs/wandb/run-xxx)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file. If not provided, reads from wandb-summary.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model architecture info",
    )
    
    args = parser.parse_args()
    
    # Load the model
    model = load_pretrained_model(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    if args.info:
        print("\n" + "=" * 60)
        print("Model Architecture:")
        print("=" * 60)
        print(model)
        print("\n" + "=" * 60)
        print("Parameter Count:")
        print("=" * 60)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    main()

