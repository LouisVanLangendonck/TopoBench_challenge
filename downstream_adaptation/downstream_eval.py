"""Inductive downstream eval: graph-level train/val/test splits.

Transductive (single graph, masks): ``downstream_eval_transductive.py``.

Modes (``--mode``): ``linear-probe`` / ``random-init-linear-probe`` freeze encoder+backbone;
``full-finetune`` / ``random-init-full-finetune`` train all (random-init skips checkpoint).
"""

GRAPHUNIVERSE_OVERRIDE_DEFAULT = None

import argparse
import json
import random
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
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from downstream_eval_utils import (
    apply_transforms,
    build_tb_model_for_downstream,
    create_dataset_from_config,
    detect_task_level,
    downstream_mode_freezes_encoder,
    downstream_mode_requires_checkpoint,
    DownstreamModel,
    get_checkpoint_path_from_summary,
    hidden_dim_from_downstream_config,
    LinearClassifier,
    load_wandb_config,
    prepare_batch_for_topobench,
    SupervisedCDDownstreamModel,
    TBModelGraphLevelEncoder,
    TBModelNodeEncoder,
    use_supervised_cd_full_tbmodel,
    verify_downstream_logits,
    verify_encoder_outputs,
    DOWNSTREAM_MODES,
)


def train_downstream(
    model: DownstreamModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    task_level: str = "node",
    task_type: str = "classification",
    loss_type: str = "cross_entropy",
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 20,
    use_wandb: bool = False,
):
    """Train downstream model."""
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    if task_type == "regression":
        if loss_type == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        metric_name = loss_type
    elif task_type == "multilabel_bce":
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        metric_name = "mae"
    else:
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        metric_name = "accuracy"

    best_val_metric = float('inf') if task_type in ("regression", "multilabel_bce") else 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], f"train_{metric_name}": [], "val_loss": [], f"val_{metric_name}": []}

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        train_metric_sum = 0.0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            optimizer.zero_grad()

            out = model(batch)

            if task_type == "multilabel_bce":
                bs = batch.num_graphs
                y = batch.property_community_presence.float().to(device).view(bs, num_classes)
                loss = criterion(out, y)
                metric = torch.abs(torch.sigmoid(out) - y).mean()
                num_samples = bs
            elif task_level == "graph":
                if task_type == "regression":
                    y = batch.y.float().view(-1, 1)
                else:
                    y = batch.y.long()
                num_samples = y.size(0)
                loss = criterion(out, y)
                if task_type == "regression":
                    if loss_type == "mae":
                        metric = torch.abs(out - y).mean()
                    else:
                        metric = torch.pow(out - y, 2).mean()
                else:
                    pred = out.argmax(dim=1)
                    metric = (pred == y).float().mean()
            else:
                if task_type == "regression":
                    y = batch.y.view(-1, 1).float()
                else:
                    y = batch.y.view(-1).long()
                num_samples = y.size(0)
                loss = criterion(out, y)
                if task_type == "regression":
                    if loss_type == "mae":
                        metric = torch.abs(out - y).mean()
                    else:
                        metric = torch.pow(out - y, 2).mean()
                else:
                    pred = out.argmax(dim=1)
                    metric = (pred == y).float().mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * num_samples
            train_metric_sum += metric.item() * num_samples
            train_total += num_samples

        train_loss /= train_total
        train_metric = train_metric_sum / train_total

        model.eval()
        val_loss = 0.0
        val_metric_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                out = model(batch)

                if task_type == "multilabel_bce":
                    bs = batch.num_graphs
                    y = batch.property_community_presence.float().to(device).view(bs, num_classes)
                    loss = criterion(out, y)
                    metric = torch.abs(torch.sigmoid(out) - y).mean()
                    num_samples = bs
                elif task_level == "graph":
                    if task_type == "regression":
                        y = batch.y.float().view(-1, 1)
                    else:
                        y = batch.y.long()
                    num_samples = y.size(0)
                    loss = criterion(out, y)
                    if task_type == "regression":
                        if loss_type == "mae":
                            metric = torch.abs(out - y).mean()
                        else:
                            metric = torch.pow(out - y, 2).mean()
                    else:
                        pred = out.argmax(dim=1)
                        metric = (pred == y).float().mean()
                else:
                    if task_type == "regression":
                        y = batch.y.view(-1, 1).float()
                    else:
                        y = batch.y.view(-1).long()
                    num_samples = y.size(0)
                    loss = criterion(out, y)
                    if task_type == "regression":
                        if loss_type == "mae":
                            metric = torch.abs(out - y).mean()
                        else:
                            metric = torch.pow(out - y, 2).mean()
                    else:
                        pred = out.argmax(dim=1)
                        metric = (pred == y).float().mean()

                val_loss += loss.item() * num_samples
                val_metric_sum += metric.item() * num_samples
                val_total += num_samples

        val_loss /= val_total
        val_metric = val_metric_sum / val_total

        history["train_loss"].append(train_loss)
        history[f"train_{metric_name}"].append(train_metric)
        history["val_loss"].append(val_loss)
        history[f"val_{metric_name}"].append(val_metric)

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                f"train/{metric_name}": train_metric,
                "val/loss": val_loss,
                f"val/{metric_name}": val_metric,
                f"best_val_{metric_name}": best_val_metric,
                "lr": optimizer.param_groups[0]['lr'],
            })

        scheduler.step(val_metric)

        is_better = (
            (val_metric < best_val_metric)
            if task_type in ("regression", "multilabel_bce")
            else (val_metric > best_val_metric)
        )

        if is_better:
            best_val_metric = val_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            f"train_{metric_name}": f"{train_metric:.4f}",
            f"val_{metric_name}": f"{val_metric:.4f}",
            "best": f"{best_val_metric:.4f}",
        })

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def evaluate(
    model: DownstreamModel,
    test_loader: DataLoader,
    num_classes: int,
    task_type: str = "classification",
    loss_type: str = "cross_entropy",
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Evaluate model on test set."""
    model = model.to(device)
    model.eval()

    if task_type == "multilabel_bce":
        all_probs = []
        all_targets = []
        n_graphs = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                out = model(batch)
                bs = batch.num_graphs
                y = batch.property_community_presence.float().view(bs, num_classes)
                probs = torch.sigmoid(out)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())
                n_graphs += bs
        preds = torch.cat(all_probs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mae = (preds - targets).abs().mean().item()
        mse = ((preds - targets) ** 2).mean().item()
        rmse = float(np.sqrt(mse))
        result = {
            "test_mae": mae,
            "test_mae_community_presence": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "num_graphs": n_graphs,
        }
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test/mae_community_presence": mae,
                "test/mse": mse,
                "test/rmse": rmse,
                "test/num_graphs": n_graphs,
            })
        return result

    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            out = model(batch)

            if model.task_level == "graph":
                if task_type == "regression":
                    y = batch.y.float().view(-1, 1)
                else:
                    y = batch.y.long()
                num_samples = y.size(0)
            else:
                if task_type == "regression":
                    y = batch.y.view(-1, 1).float()
                else:
                    y = batch.y.view(-1).long()
                num_samples = y.size(0)

            if task_type == "classification":
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
            else:
                all_preds.extend(out.cpu().squeeze().tolist())
                all_labels.extend(y.cpu().squeeze().tolist())

            test_total += num_samples

    if task_type == "classification":
        test_correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        test_acc = test_correct / test_total

        result = {
            "test_accuracy": test_acc,
            "predictions": all_preds,
            "labels": all_labels,
            f"num_{model.task_level}s": test_total,
        }

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test/accuracy": test_acc,
                f"test/num_{model.task_level}s": test_total,
            })
    else:
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)

        mae = np.abs(all_preds_np - all_labels_np).mean()
        mse = np.power(all_preds_np - all_labels_np, 2).mean()
        rmse = np.sqrt(mse)

        result = {
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "predictions": all_preds,
            "labels": all_labels,
            f"num_{model.task_level}s": test_total,
        }

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test/mae": mae,
                "test/mse": mse,
                "test/rmse": rmse,
                f"test/num_{model.task_level}s": test_total,
            })

    return result


def run_downstream_evaluation(
    run_dir: str | Path,
    n_evaluation_graphs: int = 200,
    n_train: int = 20,
    mode: str = "full-finetune",
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "downstream_eval",
    graphuniverse_override: dict | None = None,
    classifier_dropout: float = 0.5,
    input_dropout: float = None,
    downstream_task: str | None = "community_detection",
    readout_type: str = "mean",
    pretraining_config: dict | None = None,
    verbose_checkpoint_load: bool = True,
) -> dict:
    """Run inductive downstream evaluation: community_detection (node CE) or community_presence (graph BCE)."""
    run_dir = Path(run_dir)

    if mode not in DOWNSTREAM_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {DOWNSTREAM_MODES}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = load_wandb_config(run_dir)
    task_level = detect_task_level(config)

    checkpoint_path = None
    if downstream_mode_requires_checkpoint(mode):
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)
        if checkpoint_path is None:
            raise ValueError("No checkpoint path found in wandb-summary.json")

    gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
    family_params = gen_params.get("family_parameters", {})
    universe_params = gen_params.get("universe_parameters", {})
    pretraining_universe_seed = universe_params["seed"]
    pretraining_family_seed = family_params["seed"]
    num_classes = universe_params.get("K", 10)

    family_evaluation_seed = pretraining_family_seed + 1
    family_training_seed = pretraining_family_seed + 2

    eval_dataset, eval_data_dir, _ = create_dataset_from_config(
        config,
        n_graphs=n_evaluation_graphs,
        universe_seed=pretraining_universe_seed,
        family_seed=family_evaluation_seed,
        dataset_purpose="eval",
        graphuniverse_override=graphuniverse_override,
        downstream_task=downstream_task,
    )

    transforms_config = config.get("transforms")
    eval_preprocessor = apply_transforms(eval_dataset, eval_data_dir, transforms_config)
    eval_data_list = eval_preprocessor.data_list

    train_dataset, train_data_dir, _ = create_dataset_from_config(
        config,
        n_graphs=n_train,
        universe_seed=pretraining_universe_seed,
        family_seed=family_training_seed,
        dataset_purpose="train",
        graphuniverse_override=graphuniverse_override,
        downstream_task=downstream_task,
    )
    train_preprocessor = apply_transforms(train_dataset, train_data_dir, transforms_config)
    train_data = train_preprocessor.data_list

    random.seed(pretraining_universe_seed)
    eval_indices = list(range(len(eval_data_list)))
    random.shuffle(eval_indices)
    n_val = len(eval_indices) // 2
    val_indices = eval_indices[:n_val]
    test_indices = eval_indices[n_val:]
    val_data = [eval_data_list[i] for i in val_indices]
    test_data = [eval_data_list[i] for i in test_indices]

    if downstream_task == "community_presence":
        from graph_properties import add_properties_to_dataset

        train_data = add_properties_to_dataset(
            train_data, K=num_classes, include_complex=True, verbose=False
        )
        val_data = add_properties_to_dataset(
            val_data, K=num_classes, include_complex=True, verbose=False
        )
        test_data = add_properties_to_dataset(
            test_data, K=num_classes, include_complex=True, verbose=False
        )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    freeze_encoder = downstream_mode_freezes_encoder(mode)
    use_full_supervised = use_supervised_cd_full_tbmodel(
        config, downstream_task=downstream_task, task_level=task_level
    )

    hidden_dim = hidden_dim_from_downstream_config(config)

    if use_full_supervised:
        tb_model, _, _ = build_tb_model_for_downstream(
            config,
            device,
            checkpoint_path,
            load_checkpoint=downstream_mode_requires_checkpoint(mode),
            verbose=verbose_checkpoint_load,
        )
        downstream_model = SupervisedCDDownstreamModel(
            tb_model, freeze_encoder=freeze_encoder
        )
        downstream_task_level = "node"
        eval_task_type = "classification"
        verify_result = verify_downstream_logits(downstream_model, train_loader, device=device)
        print(f"Downstream model verification: {verify_result['status']}")
        if verify_result["status"] != "OK":
            print(f"  Issues: {verify_result.get('issues', [])}")
    else:
        tb_model, hidden_dim, _ = build_tb_model_for_downstream(
            config,
            device,
            checkpoint_path,
            load_checkpoint=downstream_mode_requires_checkpoint(mode),
            verbose=verbose_checkpoint_load,
        )

        graph_level = downstream_task == "community_presence" or task_level == "graph"
        if graph_level:
            encoder = TBModelGraphLevelEncoder(tb_model, readout_type=readout_type)
            downstream_task_level = "graph"
            eval_task_type = (
                "multilabel_bce" if downstream_task == "community_presence" else "classification"
            )
        else:
            encoder = TBModelNodeEncoder(tb_model)
            downstream_task_level = "node"
            eval_task_type = "classification"

        verify_encoder_outputs(encoder, train_loader, device=device)

        classifier = LinearClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )
        downstream_model = DownstreamModel(
            encoder, classifier, freeze_encoder, task_level=downstream_task_level
        )

    if use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            "mode": mode,
            "task_type": eval_task_type,
            "task_level": downstream_task_level,
            "n_evaluation_graphs": n_evaluation_graphs,
            "n_train": n_train,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "num_classes": num_classes,
            "downstream_task": downstream_task,
            "readout_type": readout_type,
            "classifier_dropout": classifier_dropout,
            "input_dropout": input_dropout,
            "hidden_dim": hidden_dim,
            "graphuniverse_override": graphuniverse_override,
        }

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

    history = train_downstream(
        model=downstream_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        task_level=downstream_task_level,
        task_type=eval_task_type,
        loss_type="cross_entropy",
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        use_wandb=use_wandb,
    )

    results = evaluate(
        model=downstream_model,
        test_loader=test_loader,
        num_classes=num_classes,
        task_type=eval_task_type,
        device=device,
        use_wandb=use_wandb,
    )

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    results["mode"] = mode
    results["task_type"] = eval_task_type
    results["downstream_task"] = downstream_task
    results["n_train"] = n_train
    results["n_val"] = len(val_data)
    results["n_test"] = len(test_data)
    results["num_classes"] = num_classes
    results["history"] = history
    results["encoder_frozen"] = freeze_encoder

    return results


def main():
    parser = argparse.ArgumentParser(description="Inductive downstream evaluation (classification).")
    parser.add_argument("--run_dir", type=str, required=True, help="Wandb run directory")
    parser.add_argument(
        "--downstream_task",
        type=str,
        default="community_detection",
        help=(
            "community_detection: node classification | "
            "community_presence: graph-level K-vector (BCE); graphs still generated as community_detection."
        ),
    )
    parser.add_argument("--n_evaluation_graphs", type=int, default=400)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument(
        "--mode",
        type=str,
        default="linear-probe",
        choices=["full-finetune","random-init-full-finetune","linear-probe","random-init-linear-probe"],
        help="full-finetune | linear-probe: pretrained checkpoint; random-init-*: fresh weights.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="downstream_eval")
    parser.add_argument("--graphuniverse_override", type=str, default=None)
    parser.add_argument("--classifier_dropout", type=float, default=0.0)
    parser.add_argument("--input_dropout", type=float, default=None)
    parser.add_argument("--readout_type", type=str, default="mean", choices=["mean", "max", "sum"])

    args = parser.parse_args()

    graphuniverse_override = None
    if args.graphuniverse_override:
        graphuniverse_override = json.loads(args.graphuniverse_override)
    elif GRAPHUNIVERSE_OVERRIDE_DEFAULT:
        graphuniverse_override = GRAPHUNIVERSE_OVERRIDE_DEFAULT

    results = run_downstream_evaluation(
        run_dir=args.run_dir,
        n_evaluation_graphs=args.n_evaluation_graphs,
        n_train=args.n_train,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        graphuniverse_override=graphuniverse_override,
        downstream_task=args.downstream_task,
        readout_type=args.readout_type,
        classifier_dropout=args.classifier_dropout,
        input_dropout=args.input_dropout,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    if results.get("task_type") == "multilabel_bce":
        print(f"Test MAE (community presence): {results['test_mae']:.4f}")
    else:
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Train: {results['n_train']}, Val: {results['n_val']}, Test: {results['n_test']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
