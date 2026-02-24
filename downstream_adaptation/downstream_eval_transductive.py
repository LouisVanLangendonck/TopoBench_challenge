"""Downstream evaluation for transductive learning setting.

This module handles evaluation on a single large graph where nodes are split
into train/val/test masks, as opposed to the inductive setting where entire
graphs are split.
"""

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

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
    load_wandb_config,
    get_checkpoint_path_from_summary,
    create_dataset_from_config,
    apply_transforms,
    detect_pretraining_method,
    detect_task_level,
    detect_learning_setting,
    create_random_encoder,
    load_pretrained_encoder,
    prepare_batch_for_topobench,
    verify_encoder_outputs,
    PretrainedEncoder,
    LinearClassifier,
    MLPClassifier,
)

# =============================================================================
# Transductive-specific Components
# =============================================================================

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
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
    
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


# =============================================================================
# Training & Evaluation
# =============================================================================

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
    import torch
    
    model = model.to(device)
    data = data.to(device)
    data = prepare_batch_for_topobench(data)
    
    # For transductive setting, add batch indices (all nodes in same graph)
    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    if not hasattr(data, 'batch_0'):
        data.batch_0 = data.batch
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    
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
            wandb.log({
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "lr": optimizer.param_groups[0]['lr'],
            })
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({
            "train_loss": f"{loss.item():.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "best": f"{best_val_acc:.4f}",
        })
        
        if patience_counter >= patience:
            break
    
    if best_model_state is not None:
        best_model_state_device = {k: v.to(device) for k, v in best_model_state.items()}
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
    import torch
    
    model = model.to(device)
    model.eval()
    data = data.to(device)
    data = prepare_batch_for_topobench(data)
    
    # For transductive setting, add batch indices (all nodes in same graph)
    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    if not hasattr(data, 'batch_0'):
        data.batch_0 = data.batch
    
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
        wandb.log({
            "test/accuracy": test_acc,
            "test/train_accuracy": train_acc,
            "test/val_accuracy": val_acc,
            "test/num_train_nodes": result["num_train_nodes"],
            "test/num_val_nodes": result["num_val_nodes"],
            "test/num_test_nodes": result["num_test_nodes"],
        })
    
    return result


# =============================================================================
# Main Pipeline
# =============================================================================

def run_downstream_evaluation_transductive(
    run_dir: str | Path,
    mode: str = "linear",
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
) -> dict:
    """Run full downstream evaluation pipeline for transductive setting."""
    run_dir = Path(run_dir)
    
    # Load config
    config = load_wandb_config(run_dir)
    task_level = detect_task_level(config)
    learning_setting = detect_learning_setting(config)
    
    # Verify transductive setting
    if learning_setting != "transductive":
        raise ValueError(
            f"Config indicates '{learning_setting}' setting, but this script is for transductive evaluation. "
            f"Please use downstream_eval.py for inductive evaluation."
        )
    
    # Get checkpoint
    checkpoint_path = get_checkpoint_path_from_summary(run_dir)
    if checkpoint_path is None:
        raise ValueError("No checkpoint path found in wandb-summary.json")
    
    # Extract parameters
    dataset_params = config["dataset"]["parameters"]
    # NOTE: Don't use num_classes from pretraining config (it's 1 for GraphMAEv2)
    # We'll infer it from the actual labels after loading the data
    num_classes_from_config = dataset_params.get("num_classes", 10)
    
    # Get universe parameters for consistent seed
    gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
    family_params = gen_params.get("family_parameters", {})
    universe_params = gen_params.get("universe_parameters", {})
    pretraining_universe_seed = universe_params["seed"]
    pretraining_family_seed = family_params["seed"]
    
    # For transductive downstream, use SAME graph (same seeds) as pretraining
    # This is the key difference from inductive setting
    # IMPORTANT: Don't override downstream_task to keep the EXACT same graph generation
    # The labels (community detection) are always generated regardless of task type
    dataset, data_dir, dataset_info = create_dataset_from_config(
        config,
        n_graphs=1,  # Single graph for transductive
        universe_seed=pretraining_universe_seed,  # SAME universe
        family_seed=pretraining_family_seed,  # SAME family seed
        dataset_purpose="downstream_transductive",
        downstream_task=None,  # Don't override task - use pretraining task to get same features
    )
    
    # Apply transforms
    transforms_config = config.get("transforms")
    preprocessor = apply_transforms(dataset, data_dir, transforms_config)
    data_list = preprocessor.data_list
    
    # Should have exactly one graph
    assert len(data_list) == 1, f"Expected 1 graph for transductive setting, got {len(data_list)}"
    
    # Load transductive splits (creates train/val/test masks)
    from topobench.data.utils.split_utils import load_transductive_splits
    from topobench.dataloader import DataloadDataset
    from omegaconf import OmegaConf
    
    # Extract the single graph (without splits yet if doing few-shot)
    data = data_list[0]
    
    # Few-shot learning: custom split logic
    if n_train is not None or n_evaluation is not None:
        import numpy as np
        import torch
        
        num_nodes = data.num_nodes
        
        # Validate n_evaluation
        if n_evaluation is not None:
            assert n_evaluation <= num_nodes // 2, (
                f"n_evaluation ({n_evaluation}) must be at most 50% of total nodes ({num_nodes})"
            )
        else:
            # Default: use 30% for evaluation
            n_evaluation = int(0.3 * num_nodes)
        
        # Step 1: Select FIXED evaluation nodes using data_seed
        # This ensures val+test are the same across different n_train values
        np.random.seed(data_seed)
        all_indices = np.arange(num_nodes)
        np.random.shuffle(all_indices)
        
        # First n_evaluation nodes are for evaluation (val + test)
        eval_indices = all_indices[:n_evaluation]
        remaining_indices = all_indices[n_evaluation:]
        
        # Split evaluation into val and test (50/50)
        n_val = n_evaluation // 2
        n_test = n_evaluation - n_val
        val_indices = eval_indices[:n_val]
        test_indices = eval_indices[n_val:]
        
        # Step 2: Select training nodes from remaining nodes
        # Use deterministic ordering so n_train=5 is subset of n_train=15
        if n_train is not None:
            assert n_train <= len(remaining_indices), (
                f"n_train ({n_train}) must be at most {len(remaining_indices)} "
                f"(total nodes {num_nodes} - evaluation nodes {n_evaluation})"
            )
            # Use data_seed to create deterministic ordering
            # The first n_train nodes will always be the same
            np.random.seed(data_seed + 1)  # Different seed from evaluation split
            np.random.shuffle(remaining_indices)
            train_indices = remaining_indices[:n_train]
        else:
            # Use all remaining nodes for training
            train_indices = remaining_indices
        
        # Convert to torch tensors (indices, not boolean masks)
        data.train_mask = torch.from_numpy(train_indices).long()
        data.val_mask = torch.from_numpy(val_indices).long()
        data.test_mask = torch.from_numpy(test_indices).long()
        
        print(f"\n{'='*60}")
        print(f"Few-Shot Split (data_seed={data_seed}):")
        print(f"  Total nodes: {num_nodes}")
        print(f"  Evaluation nodes (fixed): {n_evaluation} (val: {n_val}, test: {n_test})")
        print(f"  Training nodes: {len(train_indices)}")
        print(f"  Remaining available for training: {len(remaining_indices)}")
        print(f"{'='*60}\n")
    else:
        # Standard split using config
        wrapped_dataset = DataloadDataset(data_list)
        split_params = config["dataset"]["split_params"]
        split_params_omega = OmegaConf.create(split_params)
        
        # Load splits - this adds train_mask, val_mask, test_mask to the data
        train_dataset, _, _ = load_transductive_splits(wrapped_dataset, split_params_omega)
        
        # Extract the single graph with masks
        data = train_dataset.data_lst[0]
        
        # Verify masks exist
        if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
            raise ValueError(
                "Data does not have train_mask, val_mask, test_mask. "
                "Make sure the dataset config has split_params.learning_setting='transductive'"
            )
    
    # Infer num_classes from actual labels (not from pretraining config which may be 1)
    num_classes = int(data.y.max().item()) + 1
    print(f"Inferred num_classes from labels: {num_classes}")
    
    # For transductive setting, add batch indices (all nodes in same graph)
    import torch
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.batch_0 = data.batch
    
    # Print graph size for debugging
    print(f"\n{'='*60}")
    print(f"Graph Statistics:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.x.shape}")
    print(f"  Train nodes: {len(data.train_mask)}")
    print(f"  Val nodes: {len(data.val_mask)}")
    print(f"  Test nodes: {len(data.test_mask)}")
    print(f"{'='*60}\n")
    
    # Load or create encoder
    if mode == "scratch" or mode == "scratch_frozen":
        feature_encoder, backbone, hidden_dim = create_random_encoder(config, device=device)
    else:
        feature_encoder, backbone, hidden_dim = load_pretrained_encoder(config, checkpoint_path, device=device)
    
    # Wrap encoder for node-level features
    encoder = PretrainedEncoder(feature_encoder, backbone)
    
    # Verify encoder (create a dummy loader for verification)
    from torch_geometric.loader import DataLoader
    dummy_loader = DataLoader([data], batch_size=1, shuffle=False)
    verify_result = verify_encoder_outputs(encoder, dummy_loader, device=device)
    print(f"Encoder verification: {verify_result['status']}")
    if verify_result['status'] != "OK":
        print(f"  Issues: {verify_result.get('issues', [])}")
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            "mode": mode,
            "task_type": "classification",
            "task_level": "node",
            "learning_setting": "transductive",
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "data_seed": data_seed,
            "num_classes": num_classes,
            "classifier_dropout": classifier_dropout,
            "input_dropout": input_dropout,
            "hidden_dim": hidden_dim,
            "num_nodes": data.num_nodes,
            "num_train_nodes": len(data.train_mask),
            "num_val_nodes": len(data.val_mask),
            "num_test_nodes": len(data.test_mask),
            "n_train": n_train,
            "n_evaluation": n_evaluation,
            "few_shot": n_train is not None or n_evaluation is not None,
        }
        
        # Add pretraining config
        def flatten_dict(d, parent_key='', sep='/'):
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
    
    # Create downstream model
    freeze_encoder = mode in ["linear", "mlp", "scratch_frozen"]
    
    if mode in ["linear", "finetune-linear", "scratch_frozen"]:
        classifier = LinearClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout if classifier_dropout else 0.0,
        )
    elif mode in ["mlp", "finetune-mlp", "scratch"]:
        classifier = MLPClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dims=[hidden_dim // 2, hidden_dim // 4],
            dropout=classifier_dropout,
            input_dropout=input_dropout,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    downstream_model = TransductiveNodeClassifier(encoder, classifier, freeze_encoder)
    
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
    
    # Cleanup wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Add metadata
    results["mode"] = mode
    results["task_type"] = "classification"
    results["task_level"] = "node"
    results["learning_setting"] = "transductive"
    results["num_classes"] = num_classes
    results["history"] = history
    results["encoder_frozen"] = freeze_encoder
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transductive downstream evaluation pipeline.")
    parser.add_argument("--run_dir", type=str, required=True, help="Wandb run directory")
    parser.add_argument("--mode", type=str, default="linear",
                       choices=["linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "scratch_frozen"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="downstream_eval_transductive")
    parser.add_argument("--classifier_dropout", type=float, default=0.0)
    parser.add_argument("--input_dropout", type=float, default=None)
    
    # Few-shot learning parameters
    parser.add_argument("--n_train", type=int, default=None,
                       help="Number of training nodes for few-shot learning (default: use all available)")
    parser.add_argument("--n_evaluation", type=int, default=500,
                       help="Number of evaluation nodes (val+test, max 50%% of total). Default: 400 nodes")
    parser.add_argument("--data_seed", type=int, default=0,
                       help="Seed for data splitting (keeps val/test fixed across n_train values)")
    
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
    
    # Print results
    print("\n" + "=" * 60)
    print("TRANSDUCTIVE NODE CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Val Accuracy: {results['val_accuracy']:.4f}")
    print(f"Train nodes: {results['num_train_nodes']}, Val nodes: {results['num_val_nodes']}, Test nodes: {results['num_test_nodes']}")
    print(f"Total nodes: {results['num_total_nodes']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

