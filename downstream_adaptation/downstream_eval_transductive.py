"""Downstream evaluation for transductive learning setting.

This module handles evaluation on a single large graph where nodes are split
into train/val/test masks, as opposed to the inductive setting where entire
graphs are split.

Downstream modes (``--mode``):
- ``linear-probe`` / ``random-init-linear-probe``: frozen feature encoder + backbone.
- ``full-finetune`` / ``random-init-full-finetune``: all trainable; random-init skips checkpoint.
"""

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from downstream_eval_utils import (
    DOWNSTREAM_MODES,
    LinearClassifier,
    SupervisedCDDownstreamModel,
    TBModelNodeEncoder,
    apply_transforms,
    build_tb_model_for_downstream,
    create_dataset_from_config,
    detect_learning_setting,
    detect_task_level,
    downstream_mode_freezes_encoder,
    downstream_mode_requires_checkpoint,
    freeze_batchnorm_eval_no_track,
    get_checkpoint_path_from_summary,
    load_wandb_config,
    prepare_batch_for_topobench,
    use_supervised_cd_full_tbmodel,
)


class TransductiveNodeClassifier(nn.Module):
    """Node classifier for transductive learning."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        freeze_batchnorm_eval_no_track(self.encoder)

    def forward(self, batch):
        """Forward pass - returns logits for ALL nodes."""
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                node_features = self.encoder(batch)
        else:
            node_features = self.encoder(batch)

        return self.classifier(node_features)

    def train(self, mode=True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


def train_transductive(
    model: TransductiveNodeClassifier,
    data,
    num_classes: int,
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 20,
    use_wandb: bool = False,
):
    """Train transductive node classification model."""
    model = model.to(device)
    data = data.to(device)

    # Ensure batch indices exist for TopoBench (all nodes belong to graph 0)
    if not hasattr(data, "batch"):
        data.batch = torch.zeros(
            data.num_nodes, dtype=torch.long, device=device
        )
    data = prepare_batch_for_topobench(data)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass on entire graph
        out = model(data)

        # Compute loss only on training nodes
        # train_mask contains indices, not boolean mask
        train_mask = data.train_mask.long()
        y_train = data.y[train_mask].long()
        out_train = out[train_mask]

        loss = criterion(out_train, y_train)

        # Compute training accuracy
        pred_train = out_train.argmax(dim=1)
        train_acc = (pred_train == y_train).float().mean().item()

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data)

            # Validation metrics
            # val_mask contains indices, not boolean mask
            val_mask = data.val_mask.long()
            y_val = data.y[val_mask].long()
            out_val = out[val_mask]

            val_loss = criterion(out_val, y_val).item()
            pred_val = out_val.argmax(dim=1)
            val_acc = (pred_val == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "best_val_accuracy": best_val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix(
            {
                "train_loss": f"{loss.item():.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best": f"{best_val_acc:.4f}",
            }
        )

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        best_model_state_device = {
            k: v.to(device) for k, v in best_model_state.items()
        }
        model.load_state_dict(best_model_state_device)

    return history


def evaluate_transductive(
    model: TransductiveNodeClassifier,
    data,
    num_classes: int,
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Evaluate transductive model on test set."""
    model = model.to(device)
    model.eval()
    data = data.to(device)

    # Ensure batch indices exist for TopoBench (all nodes belong to graph 0)
    if not hasattr(data, "batch"):
        data.batch = torch.zeros(
            data.num_nodes, dtype=torch.long, device=device
        )
    data = prepare_batch_for_topobench(data)

    with torch.no_grad():
        out = model(data)

        # Test metrics
        # Masks contain indices, not boolean masks
        test_mask = data.test_mask.long()
        y_test = data.y[test_mask].long()
        out_test = out[test_mask]

        pred_test = out_test.argmax(dim=1)
        test_acc = (pred_test == y_test).float().mean().item()

        # Also compute train and val for reference
        train_mask = data.train_mask.long()
        y_train = data.y[train_mask].long()
        out_train = out[train_mask]
        pred_train = out_train.argmax(dim=1)
        train_acc = (pred_train == y_train).float().mean().item()

        val_mask = data.val_mask.long()
        y_val = data.y[val_mask].long()
        out_val = out[val_mask]
        pred_val = out_val.argmax(dim=1)
        val_acc = (pred_val == y_val).float().mean().item()

    result = {
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "num_train_nodes": len(train_mask),
        "num_val_nodes": len(val_mask),
        "num_test_nodes": len(test_mask),
        "num_total_nodes": data.num_nodes,
        "predictions": pred_test.cpu().tolist(),
        "labels": y_test.cpu().tolist(),
    }

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                "test/accuracy": test_acc,
                "test/train_accuracy": train_acc,
                "test/val_accuracy": val_acc,
                "test/num_train_nodes": result["num_train_nodes"],
                "test/num_val_nodes": result["num_val_nodes"],
                "test/num_test_nodes": result["num_test_nodes"],
            }
        )

    return result


def run_downstream_evaluation_transductive(
    run_dir: str | Path,
    mode: str = "linear-probe",
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "downstream_eval_transductive",
    classifier_dropout: float = 0.5,
    input_dropout: float = None,
    n_train: int | None = None,
    n_evaluation: int | None = None,
    data_seed: int = 0,
    pretraining_config: dict | None = None,
    graphuniverse_override: dict | None = None,
    repeat_idx: int = 0,
    repeat_on_different_split_seed: int = 1,
) -> dict:
    """Run downstream evaluation in transductive setting (single graph, node-level splits).

    Args:
        run_dir: Path to pretrained model wandb run directory
        mode: Evaluation mode (linear-probe, full-finetune, random-init-*)
        n_train: Number of training nodes (if None, use all remaining after val/test)
        n_evaluation: Total val+test nodes (split 50/50)
        data_seed: Seed for train/val/test splits
        repeat_idx: Current repeat index for different train/val splits
    """
    run_dir = Path(run_dir)

    if mode not in DOWNSTREAM_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of {DOWNSTREAM_MODES}"
        )

    # Use pretraining_config if provided, otherwise load from run_dir
    if pretraining_config is not None:
        config = pretraining_config
    else:
        config = load_wandb_config(run_dir)
    task_level = detect_task_level(config)
    learning_setting = detect_learning_setting(config)

    if learning_setting != "transductive":
        raise ValueError(
            f"Config indicates '{learning_setting}' setting, but this script is for transductive. "
            f"Use downstream_eval.py for inductive."
        )

    # Get checkpoint path (if needed for this mode)
    checkpoint_path = None
    if downstream_mode_requires_checkpoint(mode):
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)
        if checkpoint_path is None:
            raise ValueError("No checkpoint path found in wandb-summary.json")

    # Extract dataset generation parameters from pretrained config
    gen_params = config["dataset"]["loader"]["parameters"].get(
        "generation_parameters", {}
    )
    pretraining_universe_seed = gen_params.get("universe_parameters", {})[
        "seed"
    ]
    pretraining_family_seed = gen_params.get("family_parameters", {})["seed"]

    # Get model name for transform reconstruction
    model_name = config["model"]["model_name"]

    print("\nLoading transductive dataset (same graph as pretraining)...")
    print(f"  Universe seed: {pretraining_universe_seed}")
    print(f"  Family seed: {pretraining_family_seed}")
    print(f"  Model name: {model_name}")

    # Load the same graph used in pretraining
    dataset, data_dir, _ = create_dataset_from_config(
        config,
        n_graphs=1,
        universe_seed=pretraining_universe_seed,
        family_seed=pretraining_family_seed,
        dataset_purpose="downstream_transductive",
        downstream_task=None,
        graphuniverse_override=graphuniverse_override,
    )

    # Apply transforms (same as pretraining) with model_name for correct ordering
    transforms_config = config.get("transforms")
    preprocessor = apply_transforms(
        dataset, data_dir, transforms_config, model_name=model_name
    )
    data_list = preprocessor.data_list

    assert len(data_list) == 1, (
        f"Expected 1 graph for transductive setting, got {len(data_list)}"
    )
    data = data_list[0]

    # Create train/val/test splits
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1

    if n_evaluation is not None:
        assert n_evaluation <= num_nodes // 2, (
            f"n_evaluation ({n_evaluation}) must be ≤ 50% of nodes ({num_nodes})"
        )

        # Split nodes: test set is fixed (based on data_seed only)
        np.random.seed(data_seed)
        all_indices = np.arange(num_nodes)
        np.random.shuffle(all_indices)

        # Fixed test set (same across all repeats)
        n_val = n_evaluation // 2
        n_test = n_evaluation - n_val
        test_indices = all_indices[:n_test]
        remaining_indices = all_indices[n_test:]

        # Variable train/val split (changes with repeat_idx)
        split_seed = data_seed + 1000 + repeat_idx
        np.random.seed(split_seed)
        np.random.shuffle(remaining_indices)

        val_indices = remaining_indices[:n_val]
        train_pool = remaining_indices[n_val:]

        if n_train is not None:
            assert n_train <= len(train_pool), (
                f"n_train ({n_train}) exceeds available nodes ({len(train_pool)})"
            )
            train_indices = train_pool[:n_train]
        else:
            train_indices = train_pool

        data.train_mask = torch.from_numpy(train_indices).long()
        data.val_mask = torch.from_numpy(val_indices).long()
        data.test_mask = torch.from_numpy(test_indices).long()

        print(f"\nSplit (data_seed={data_seed}, repeat={repeat_idx}):")
        print(
            f"  Train: {len(train_indices)} | Val: {n_val} | Test: {n_test} (fixed)"
        )
    else:
        # Use existing splits from dataset
        if not all(
            hasattr(data, m) for m in ["train_mask", "val_mask", "test_mask"]
        ):
            raise ValueError(
                "Data missing train/val/test masks. Set n_evaluation or fix config."
            )
        print(
            f"\nUsing existing splits: train={len(data.train_mask)}, val={len(data.val_mask)}, test={len(data.test_mask)}"
        )

    print(
        f"\nGraph: {num_nodes} nodes, {data.num_edges} edges, {data.x.shape[1]} features, {num_classes} classes"
    )

    # Build model: encoder + classifier
    freeze_encoder = downstream_mode_freezes_encoder(mode)
    use_full_supervised = use_supervised_cd_full_tbmodel(
        config, downstream_task=None, task_level=task_level
    )

    if use_full_supervised:
        # Supervised CD: use full TBModel with pretrained readout
        tb_model, _, _ = build_tb_model_for_downstream(
            config,
            device,
            checkpoint_path,
            load_checkpoint=downstream_mode_requires_checkpoint(mode),
            verbose=True,
        )
        downstream_model = SupervisedCDDownstreamModel(
            tb_model, freeze_encoder=freeze_encoder
        )
    else:
        # Linear probe or finetune: encoder + new linear classifier
        tb_model, hidden_dim, _ = build_tb_model_for_downstream(
            config,
            device,
            checkpoint_path,
            load_checkpoint=downstream_mode_requires_checkpoint(mode),
            verbose=True,
        )
        encoder = TBModelNodeEncoder(tb_model)
        classifier = LinearClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout if classifier_dropout else 0.0,
        )
        downstream_model = TransductiveNodeClassifier(
            encoder, classifier, freeze_encoder
        )

    # Initialize W&B logging
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            "mode": mode,
            "task_level": "node",
            "learning_setting": "transductive",
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "data_seed": data_seed,
            "num_classes": num_classes,
            "classifier_dropout": classifier_dropout,
            "num_nodes": num_nodes,
            "num_train_nodes": len(data.train_mask),
            "num_val_nodes": len(data.val_mask),
            "num_test_nodes": len(data.test_mask),
            "repeat_idx": repeat_idx,
            "repeat_on_different_split_seed": repeat_on_different_split_seed,
        }

        # Flatten and add pretrain config to wandb
        def flatten_dict(d, parent_key="", sep="/"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for key in ["dataset", "model", "optimizer", "trainer"]:
            if key in config:
                cfg = config[key]
                if isinstance(cfg, dict) and "value" in cfg:
                    cfg = cfg["value"]
                flattened = flatten_dict(cfg)
                for k, v in flattened.items():
                    wandb_config[f"pretrain/{key}/{k}"] = v

        wandb.init(project=wandb_project, config=wandb_config)

    # Train
    history = train_transductive(
        model=downstream_model,
        data=data,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        use_wandb=use_wandb,
    )

    # Evaluate
    results = evaluate_transductive(
        model=downstream_model,
        data=data,
        num_classes=num_classes,
        device=device,
        use_wandb=use_wandb,
    )

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # Add metadata to results
    results.update(
        {
            "mode": mode,
            "task_type": "classification",
            "task_level": "node",
            "learning_setting": "transductive",
            "num_classes": num_classes,
            "history": history,
            "encoder_frozen": freeze_encoder,
        }
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transductive downstream evaluation pipeline."
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Wandb run directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="linear-probe",
        choices=list(DOWNSTREAM_MODES),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb_project", type=str, default="downstream_eval_transductive"
    )
    parser.add_argument("--classifier_dropout", type=float, default=0.0)
    parser.add_argument("--input_dropout", type=float, default=None)

    # Few-shot learning parameters
    parser.add_argument(
        "--n_train",
        type=int,
        default=50,
        help="Number of training nodes for few-shot learning (default: 30)",
    )
    parser.add_argument(
        "--n_evaluation",
        type=int,
        default=500,
        help="Number of evaluation nodes (val+test, max 50%% of total). Default: 400 nodes",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=0,
        help="Seed for data splitting (keeps val/test fixed across n_train values)",
    )

    args = parser.parse_args()

    results = run_downstream_evaluation_transductive(
        run_dir=args.run_dir,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        classifier_dropout=args.classifier_dropout,
        input_dropout=args.input_dropout,
        n_train=args.n_train,
        n_evaluation=args.n_evaluation,
        data_seed=args.data_seed,
    )

    print("\n" + "=" * 60)
    print("TRANSDUCTIVE NODE CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Val Accuracy: {results['val_accuracy']:.4f}")
    print(
        f"Train nodes: {results['num_train_nodes']}, Val nodes: {results['num_val_nodes']}, Test nodes: {results['num_test_nodes']}"
    )
    print(f"Total nodes: {results['num_total_nodes']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
