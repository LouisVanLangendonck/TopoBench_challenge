"""Shared utilities for downstream evaluation (inductive and transductive)."""

import json
import random
import sys
import time
from pathlib import Path
from typing import Any
from copy import deepcopy

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

_GLOBAL_POOL = {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}

from topobench.data.preprocessor import PreProcessor
from graph_universe import GraphUniverseDataset


# =============================================================================
# Configuration Loading
# =============================================================================

def load_wandb_config(run_dir: str | Path) -> dict:
    """Load config from wandb run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / "files" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    config = {}
    for key, val in raw_config.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    
    return config


def get_checkpoint_path_from_summary(run_dir: str | Path) -> str | None:
    """Extract checkpoint path from wandb-summary.json."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "files" / "wandb-summary.json"
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary.get("best_epoch/checkpoint")


def build_wandb_local_run_dir_index(
    potential_wandb_dirs: list[Path | str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """
    Scan each wandb root once and build (1) run_id -> run directory path and
    (2) a list of (dir_name, path) for substring fallback.

    Wandb offline folders are typically ``run-YYYYMMDD_HHMMSS-<run_id>``; the
    run id is the segment after the final ``-``. Using a dict avoids O(n²)
    rescans when matching many API runs to local dirs.
    """
    by_run_id: dict[str, str] = {}
    all_named: list[tuple[str, str]] = []
    for root in potential_wandb_dirs:
        wd = Path(root)
        if not wd.is_dir():
            continue
        for run_path in wd.iterdir():
            if not run_path.is_dir():
                continue
            name = run_path.name
            resolved = str(run_path.resolve())
            all_named.append((name, resolved))
            if name.startswith("run-"):
                _prefix, tail = name.rsplit("-", 1)
                if tail:
                    by_run_id[tail] = resolved
    return by_run_id, all_named


def resolve_local_wandb_run_dir(
    run_id: str,
    by_run_id: dict[str, str],
    all_named: list[tuple[str, str]],
) -> str | None:
    """Resolve API ``run_id`` to a local run directory path."""
    p = by_run_id.get(run_id)
    if p is not None:
        return p
    for name, path in all_named:
        if run_id in name:
            return path
    return None


def wandb_transient_api_error(exc: BaseException) -> bool:
    """Return True if retrying may succeed (W&B gateway overload, timeouts)."""
    text = str(exc).lower()
    markers = (
        "502",
        "503",
        "504",
        "429",
        "bad gateway",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
    )
    return any(m in text for m in markers)


def run_with_wandb_retry(
    fn,
    *,
    max_retries: int = 6,
    label: str = "W&B API",
) -> Any:
    """Run ``fn()`` and retry on transient W&B / HTTP gateway errors."""
    last: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == max_retries - 1 or not wandb_transient_api_error(e):
                raise
            delay = min(120.0, (2**attempt) * 10 + random.uniform(0, 3))
            print(
                f"\n  {label} transient error (attempt {attempt + 1}/{max_retries}): {e!s}\n"
                f"  Retrying in {delay:.0f}s ...\n"
            )
            time.sleep(delay)
    assert last is not None
    raise last


def fetch_runs_from_wandb_project(
    project_path: str,
    filters: dict | None = None,
    min_runs: int = 1,
    *,
    max_api_retries: int = 6,
) -> list[dict[str, Any]]:
    """
    List runs from a W&B project via the API and join to local ``wandb/run-*`` dirs.

    Uses a single local directory scan, ``per_page=500``, local checkpoint/config
    reads, and retries on transient API errors (502/503/timeout).
    """
    import wandb

    def _fetch_once() -> list[dict[str, Any]]:
        api = wandb.Api(timeout=120)

        if "/" in project_path:
            entity, project = project_path.split("/", 1)
        else:
            entity = None
            project = project_path

        print(f"\n{'=' * 80}")
        print(f"FETCHING RUNS FROM WANDB PROJECT: {project_path}")
        print(f"{'=' * 80}")

        # Do not assign to ``filters`` here: that would shadow the closure and
        # trigger UnboundLocalError on ``if filters is None`` above.
        run_filters = filters if filters is not None else {"state": "finished"}

        potential_wandb_dirs = [
            Path("data/outputs/wandb"),
            Path("wandb"),
        ]
        by_run_id, all_named = build_wandb_local_run_dir_index(potential_wandb_dirs)

        print(
            "  Indexed local wandb run directories; contacting api.wandb.ai for run list "
            "(large projects may take a minute; 502/503 from W&B are often transient — retrying)."
        )

        page_size = 500
        if entity:
            runs = api.runs(
                f"{entity}/{project}", filters=run_filters, per_page=page_size
            )
        else:
            runs = api.runs(project, filters=run_filters, per_page=page_size)

        run_infos: list[dict[str, Any]] = []
        api_runs_seen = 0

        for run in runs:
            api_runs_seen += 1
            run_id = run.id
            run_name = run.name

            run_dir = resolve_local_wandb_run_dir(run_id, by_run_id, all_named)

            if run_dir is None:
                print(
                    f"  ⚠️  Skipping {run_id} ({run_name}): local run directory not found"
                )
            else:
                checkpoint_path = get_checkpoint_path_from_summary(run_dir)
                if checkpoint_path is None:
                    print(
                        f"  ⚠️  {run_id} ({run_name}): no checkpoint in local summary "
                        f"(still usable for random-init-* modes)"
                    )

                pretrain_config = load_wandb_config(run_dir)
                config = pretrain_config

                run_infos.append(
                    {
                        "run_dir": run_dir,
                        "run_id": run_id,
                        "run_name": run_name,
                        "config": config,
                        "checkpoint_path": checkpoint_path,
                        "pretrain_config": pretrain_config,
                    }
                )

                print(f"  ✓ {run_id} ({run_name})")
                print(f"    Dir: {run_dir}")
                print(f"    Checkpoint: {checkpoint_path or '—'}")

            if api_runs_seen % page_size == 0:
                n_pages = api_runs_seen // page_size
                print(
                    f"  … W&B API: fetched {api_runs_seen} runs so far "
                    f"(~page {n_pages} of {page_size}/page); "
                    f"{len(run_infos)} matched local run dirs.",
                    flush=True,
                )

        print(f"\n{'=' * 80}")
        print(f"FOUND {len(run_infos)} RUNS (checkpoint optional for random-init-*)")
        print(f"{'=' * 80}\n")

        if len(run_infos) < min_runs:
            raise ValueError(
                f"Expected at least {min_runs} runs, but only found {len(run_infos)} "
                f"in project {project_path}"
            )

        return run_infos

    return run_with_wandb_retry(_fetch_once, max_retries=max_api_retries)


# =============================================================================
# Dataset Generation
# =============================================================================

def create_dataset_from_config(
    config: dict,
    n_graphs: int | None = None,
    universe_seed: int = 42,
    family_seed: int = 43,
    dataset_purpose: str = "eval",
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,
    data_dir: str | None = None,
) -> tuple:
    """Create/load GraphUniverse dataset with optional overrides."""
    dataset_config = deepcopy(config["dataset"])
    params = dataset_config["loader"]["parameters"]
    gen_params = deepcopy(params["generation_parameters"])
    
    if downstream_task is not None:
        # Presence is derived from node community labels; GraphUniverse generates CD graphs.
        if downstream_task == "community_presence":
            gen_params["task"] = "community_detection"
        else:
            gen_params["task"] = downstream_task
    
    # Override number of graphs and seeds
    if n_graphs is not None:
        gen_params["family_parameters"]["n_graphs"] = n_graphs
    gen_params["family_parameters"]["seed"] = family_seed
    gen_params["universe_parameters"]["seed"] = universe_seed
    
    # Apply GraphUniverse overrides
    if graphuniverse_override is not None and len(graphuniverse_override) > 0:
        _deep_update(gen_params, graphuniverse_override)
    
    # Set data directory
    root_dir = data_dir if data_dir else params["data_dir"]
    if n_graphs is not None:
        root_dir = f"{root_dir}_downstream_{n_graphs}graphs_universe_seed{universe_seed}_family_seed{family_seed}"
    if downstream_task is not None:
        root_dir = f"{root_dir}_task_{downstream_task}"
    if graphuniverse_override is not None and len(graphuniverse_override) > 0:
        import hashlib
        override_hash = hashlib.md5(json.dumps(graphuniverse_override, sort_keys=True).encode()).hexdigest()[:8]
        root_dir = f"{root_dir}_override_{override_hash}"
    
    dataset = GraphUniverseDataset(root=root_dir, parameters=gen_params)
    return dataset, dataset.raw_dir, {"type": "GraphUniverse", "subsample_info": None}


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update base_dict with values from update_dict."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def apply_transforms(
    dataset,
    data_dir: str,
    transforms_config: dict | None,
) -> PreProcessor:
    """Apply transforms from config."""
    if transforms_config:
        transforms_omega = OmegaConf.create(transforms_config)
    else:
        transforms_omega = None
    
    preprocessor = PreProcessor(dataset, data_dir, transforms_omega)
    return preprocessor


def detect_pretraining_method(config: dict) -> str:
    """Detect pre-training method from config."""
    model_config = config.get("model", {})
    wrapper_config = model_config.get("backbone_wrapper", {})
    
    if wrapper_config:
        wrapper_target = wrapper_config.get("_target_", "")
        if "DGI" in wrapper_target:
            return "dgi"
        elif "GraphMAEv2" in wrapper_target:
            return "graphmaev2"
        elif "GRACE" in wrapper_target or "grace" in wrapper_target.lower():
            return "grace"
        elif "BGRL" in wrapper_target or "bgrl" in wrapper_target.lower():
            return "bgrl"
        elif "GraphMAE" in wrapper_target:
            return "graphmae"
        elif "GraphCL" in wrapper_target:
            return "graphcl"
        elif "LinkPred" in wrapper_target:
            return "linkpred"
    
    loss_config = config.get("loss", {})
    dataset_loss = loss_config.get("dataset_loss", {})
    loss_target = dataset_loss.get("_target_", "")
    
    if "DGI" in loss_target:
        return "dgi"
    elif "GraphMAEv2" in loss_target:
        return "graphmaev2"
    elif "GRACE" in loss_target or "grace" in loss_target.lower():
        return "grace"
    elif "BGRL" in loss_target or "bgrl" in loss_target.lower():
        return "bgrl"
    elif "GraphMAE" in loss_target:
        return "graphmae"
    elif "LinkPred" in loss_target:
        return "linkpred"
    
    # Fallback: detect directly from dataset task when wrapper/loss metadata is missing.
    dataset_task = str(config.get("dataset", {}).get("parameters", {}).get("task", "")).lower()
    if dataset_task == "bgrl":
        return "bgrl"

    # Check if it's supervised community detection
    dataset_config = config.get("dataset", {})
    dataset_params = dataset_config.get("parameters", {})
    task = dataset_params.get("task", "")
    loss_type = dataset_params.get("loss_type", "")
    
    # If it's supervised with classification task, check if it's CD
    if loss_type in ["cross_entropy", "classification"]:
        gen_params = dataset_config.get("loader", {}).get("parameters", {}).get("generation_parameters", {})
        task_name = gen_params.get("task", "")
        if task_name == "community_detection":
            return "supervised_cd"
    
    return "supervised"


def detect_task_level(config: dict) -> str:
    """Detect task level from config."""
    dataset_params = config.get("dataset", {}).get("parameters", {})
    task_level = dataset_params.get("task_level", "node")
    return task_level


def detect_learning_setting(config: dict) -> str:
    """Detect learning setting (inductive/transductive) from config."""
    split_params = config.get("dataset", {}).get("split_params", {})
    learning_setting = split_params.get("learning_setting", "inductive")
    return learning_setting


# =============================================================================
# Downstream evaluation modes (CLI / pipelines)
# =============================================================================

DOWNSTREAM_MODES = (
    "full-finetune",
    "random-init-full-finetune",
    "linear-probe",
    "random-init-linear-probe",
)


def downstream_mode_requires_checkpoint(mode: str) -> bool:
    """Random-init modes do not load a pretrained checkpoint."""
    return mode not in ("random-init-full-finetune", "random-init-linear-probe")


def downstream_mode_freezes_encoder(mode: str) -> bool:
    """Whether feature encoder + backbone stay frozen (readout / head still trains when applicable)."""
    return mode in ("linear-probe", "random-init-linear-probe")


def use_supervised_cd_full_tbmodel(
    config: dict,
    *,
    downstream_task: str | None = None,
    task_level: str | None = None,
) -> bool:
    """Use full TBModel + checkpoint (incl. readout) for supervised community detection."""
    if detect_pretraining_method(config) != "supervised_cd":
        return False
    tl = task_level if task_level is not None else detect_task_level(config)
    if tl != "node":
        return False
    if downstream_task == "community_presence":
        return False
    return True


def _detach_tensors_in_model_out(model_out):
    if isinstance(model_out, dict):
        return {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in model_out.items()
        }
    if torch.is_tensor(model_out):
        return model_out.detach()
    return model_out


def freeze_batchnorm_eval_no_track(root: nn.Module) -> None:
    """Set all BatchNorm submodules to eval with frozen running stats."""
    for m in root.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.track_running_stats = False


def instantiate_tbmodel_from_wandb_config(config: dict, device: str = "cpu"):
    """Build TBModel the same way as training (encoder + backbone + readout + loss + evaluator)."""
    import hydra
    from topobench.model import TBModel

    model_cfg = OmegaConf.create(config["model"])
    evaluator = hydra.utils.instantiate(OmegaConf.create(config["evaluator"]))
    loss = hydra.utils.instantiate(OmegaConf.create(config["loss"]))
    optimizer = hydra.utils.instantiate(OmegaConf.create(config["optimizer"]))
    learning_setting = config["dataset"]["split_params"]["learning_setting"]

    model: TBModel = hydra.utils.instantiate(
        model_cfg,
        evaluator=evaluator,
        learning_setting=learning_setting,
        optimizer=optimizer,
        loss=loss,
    )
    return model.to(device)


def load_tbmodel_weights_from_checkpoint(tb_model, checkpoint_path: str | Path) -> tuple:
    """Load Lightning checkpoint weights into an existing TBModel."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return tb_model.load_state_dict(state_dict, strict=False)


def sync_batch_0_from_batch(data) -> None:
    """Align ``batch_0`` with PyG ``batch`` (required e.g. for DGI ``graph_diffusion``)."""
    if getattr(data, "batch", None) is not None:
        data.batch_0 = data.batch


def hidden_dim_from_downstream_config(config: dict) -> int:
    """Hidden width for node features after backbone (same resolution as verify script)."""
    backbone_cfg = config.get("model", {}).get("backbone", {})
    return int(
        backbone_cfg.get("hidden_dim")
        or backbone_cfg.get("hidden_channels")
        or backbone_cfg.get("out_channels")
        or config.get("model", {}).get("feature_encoder", {}).get("out_channels")
        or 128
    )


def replace_ssl_backbone_with_gnn_wrapper(tb_model, *, verbose: bool = True) -> bool:
    """Use ``GNNWrapper`` for downstream: one clean GNN pass, no SSL augmentations.

    **Preserved (same ``nn.Module`` instances and tensors as after checkpoint load)**

    - Inner graph encoder: ``GPSEncoder`` (or equivalent) — the weights used for message passing.
    - Per-rank residual LayerNorms ``ln_0``, … on the wrapper — copied from the old SSL wrapper
      so we do **not** randomly re-init them (they participate in ``AbstractWrapper.__call__`` when
      ``residual_connections`` is True and ``batch`` carries ``x_0``).

    **Dropped (SSL-only; not used by ``GNNWrapper.forward``)**

    These are left out of the forward path on purpose; they only served the pretraining loss:

    - **GraphMAEv2**: ``enc_mask_token``, latent ``projector`` / ``predictor``, EMA teacher copies.
    - **GraphCL / GRACE / LinkPred**: no learnable params in the wrapper besides ``ln_*`` and backbone.
    - **DGI**: wrapper is only the GNN + corruption logic; readout/discriminator lives under
      ``tb_model.readout`` (still loaded, unused by ``TBModelNodeEncoder``).
    - **BGRL**: the **target** encoder (``bb.backbone``) is unused; we keep **online_encoder** as
      the downstream GNN (standard choice for the student branch).

    **No extra random weights**: the only new modules in a fresh ``GNNWrapper`` are ``ln_*``;
    those are filled by copying from ``bb``, not left at init.

    .. note::

        Hydra often instantiates wrappers via ``topobench.nn.wrappers`` dynamic discovery,
        which can produce a *different* class object than ``import ...graphcl_gnn_wrapper``.
        We match by ``__class__.__name__``, not ``isinstance``, so replacement still runs.
    """
    from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper

    _SSL_WRAPPER_NAMES = frozenset({
        "GraphCLGNNWrapper",
        "GraphMAEv2GNNWrapper",
        "DGIGNNWrapper",
        "LinkPredGNNWrapper",
        "BGRLGNNWrapper",
        "GRACEGNNWrapper",
    })

    bb = getattr(tb_model, "backbone", None)
    if bb is None:
        if verbose:
            print("Downstream SSL swap: tb_model has no backbone; skipping.")
        return False
    cls_name = bb.__class__.__name__
    if cls_name == "GNNWrapper":
        if verbose:
            print("Downstream SSL swap: backbone already GNNWrapper — no replacement.")
        return False
    if cls_name not in _SSL_WRAPPER_NAMES:
        if verbose:
            known = ", ".join(sorted(_SSL_WRAPPER_NAMES))
            print(
                f"Downstream SSL swap: backbone class {cls_name!r} is not in the swap list "
                f"({known}) — keeping original wrapper."
            )
        return False

    if cls_name == "BGRLGNNWrapper":
        inner = bb.online_encoder
    else:
        inner = bb.backbone

    if inner is None:
        if verbose:
            print(f"Downstream SSL swap: {cls_name} has no inner encoder; skipping.")
        return False

    out_ch = getattr(inner, "out_channels", None) or getattr(inner, "hidden_dim", None)
    if out_ch is None and hasattr(bb, "ln_0"):
        out_ch = bb.ln_0.normalized_shape[0]
    if out_ch is None:
        if verbose:
            print(
                f"Downstream SSL swap: could not infer out_channels for {cls_name}; skipping."
            )
        return False
    out_ch = int(out_ch)

    num_cell_dim = len(bb.dimensions)
    residual = getattr(bb, "residual_connections", True)

    device = next(inner.parameters()).device
    new_bb = GNNWrapper(
        inner,
        out_channels=out_ch,
        num_cell_dimensions=num_cell_dim,
        residual_connections=residual,
    )
    new_bb.to(device)
    for i in bb.dimensions:
        old_ln = getattr(bb, f"ln_{i}", None)
        new_ln = getattr(new_bb, f"ln_{i}", None)
        if old_ln is not None and new_ln is not None:
            new_ln.load_state_dict(old_ln.state_dict())
    tb_model.backbone = new_bb
    if verbose:
        print(
            f"Downstream SSL swap: replaced {cls_name} with GNNWrapper "
            f"(inner encoder module unchanged; ln_* copied)."
        )
    return True


def build_tb_model_for_downstream(
    config: dict,
    device: str,
    checkpoint_path: str | Path | None,
    *,
    load_checkpoint: bool,
    verbose: bool = True,
):
    """Instantiate full ``TBModel`` and optionally load Lightning weights (matches checkpoint verify flow)."""
    tb_model = instantiate_tbmodel_from_wandb_config(config, device=device)
    inc = None
    if load_checkpoint:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when load_checkpoint=True")
        inc = load_tbmodel_weights_from_checkpoint(tb_model, checkpoint_path)
        if verbose and (inc.missing_keys or inc.unexpected_keys):
            print(
                f"TBModel load_state_dict (non-strict): "
                f"{len(inc.missing_keys)} missing, {len(inc.unexpected_keys)} unexpected keys"
            )
    replace_ssl_backbone_with_gnn_wrapper(tb_model, verbose=verbose)
    hidden_dim = hidden_dim_from_downstream_config(config)
    return tb_model, hidden_dim, inc


class TBModelNodeEncoder(nn.Module):
    """``feature_encoder`` → ``backbone`` → node embeddings ``x_0``.

    After ``build_tb_model_for_downstream``, SSL runs use ``GNNWrapper`` (see
    ``replace_ssl_backbone_with_gnn_wrapper``), not the training-time augmenting wrapper.

    ``readout`` parameters are marked ``requires_grad=False`` and stay in ``eval()`` during
    ``train()`` on this module; the forward path does **not** call ``readout`` (SSL heads are
    irrelevant for probing). ``DownstreamModel`` then freezes the whole encoder for linear-probe
    or leaves it trainable for full-finetune — **same** for pretrained and random-init pairs.
    """

    def __init__(self, tb_model):
        super().__init__()
        self.tb_model = tb_model
        for p in self.tb_model.readout.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        self.tb_model.feature_encoder.train(mode)
        self.tb_model.backbone.train(mode)
        self.tb_model.readout.eval()
        return self

    def eval(self):
        return self.train(False)

    def encode_to_x0_and_batch(self, batch):
        batch = prepare_batch_for_topobench(batch)
        if hasattr(batch, "x"):
            batch.x_0 = batch.x
        sync_batch_0_from_batch(batch)
        encoded = self.tb_model.feature_encoder(batch)
        sync_batch_0_from_batch(encoded)
        out = self.tb_model.backbone(encoded)
        if isinstance(out, dict):
            x = out["x_0"]
            bidx = out.get("batch_0")
            if bidx is None:
                bidx = getattr(encoded, "batch_0", None)
        else:
            x = out
            bidx = getattr(encoded, "batch_0", None)
        if bidx is None:
            bidx = getattr(encoded, "batch", None)
        return x, bidx

    def forward(self, batch):
        x, _ = self.encode_to_x0_and_batch(batch)
        return x


class TBModelGraphLevelEncoder(nn.Module):
    """Graph-level: node encoder + global pool."""

    def __init__(self, tb_model, readout_type: str = "mean"):
        super().__init__()
        self.node_encoder = TBModelNodeEncoder(tb_model)
        try:
            self.pool = _GLOBAL_POOL[readout_type]
        except KeyError as e:
            raise ValueError(f"Unknown readout type: {readout_type}") from e

    def train(self, mode=True):
        self.training = mode
        self.node_encoder.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, batch, return_node_features=False):
        x, bidx = self.node_encoder.encode_to_x0_and_batch(batch)
        if bidx is None:
            raise RuntimeError(
                "Cannot pool: missing batch indices (batch_0 / batch) after backbone forward."
            )
        graph_x = self.pool(x, bidx)
        if return_node_features:
            return {"node": x, "graph": graph_x}
        return graph_x


class SupervisedCDDownstreamModel(nn.Module):
    """Full supervised TBModel for node CD: uses pretrained readout, optional frozen encoder+backbone."""

    def __init__(self, tb_model, freeze_encoder: bool = True):
        super().__init__()
        self.tb_model = tb_model
        self.freeze_encoder = freeze_encoder
        self.task_level = getattr(tb_model, "task_level", "node")
        if freeze_encoder:
            self._freeze_encoder_params()

    def _freeze_encoder_params(self):
        for p in self.tb_model.feature_encoder.parameters():
            p.requires_grad = False
        for p in self.tb_model.backbone.parameters():
            p.requires_grad = False
        for p in self.tb_model.readout.parameters():
            p.requires_grad = True

    def forward(self, batch):
        batch = prepare_batch_for_topobench(batch)
        sync_batch_0_from_batch(batch)
        if self.freeze_encoder:
            self.tb_model.feature_encoder.eval()
            self.tb_model.backbone.eval()
            with torch.no_grad():
                model_out = self.tb_model.feature_encoder(batch)
                if not isinstance(model_out, dict):
                    sync_batch_0_from_batch(model_out)
                model_out = self.tb_model.backbone(model_out)
            model_out = _detach_tensors_in_model_out(model_out)
            model_out = self.tb_model.readout(model_out=model_out, batch=batch)
        else:
            model_out = self.tb_model(batch)
        logits = model_out["logits"]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        return logits

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.tb_model.feature_encoder.eval()
            self.tb_model.backbone.eval()
            freeze_batchnorm_eval_no_track(self.tb_model.feature_encoder)
            freeze_batchnorm_eval_no_track(self.tb_model.backbone)
        return self


def prepare_batch_for_topobench(batch):
    """Ensure batch has required TopoBench attributes."""
    if not hasattr(batch, 'x_0') and hasattr(batch, 'x'):
        batch.x_0 = batch.x
    sync_batch_0_from_batch(batch)
    return batch


def verify_encoder_outputs(encoder: nn.Module, data_loader, device: str = "cpu"):
    """Verify encoder produces meaningful outputs."""
    encoder.eval()
    encoder = encoder.to(device)
    
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = prepare_batch_for_topobench(batch)
    
    input_features = batch.x_0 if hasattr(batch, 'x_0') else batch.x
    input_mean = input_features.mean().item()
    input_std = input_features.std().item()
    input_dim = input_features.shape[1]
    
    with torch.no_grad():
        try:
            features = encoder(batch)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    mean = features.mean().item()
    std = features.std().item()
    min_val = features.min().item()
    max_val = features.max().item()
    num_zeros = (features == 0).sum().item()
    total_elements = features.numel()
    
    issues = []
    if std < 1e-6:
        issues.append("Very low variance")
    if num_zeros == total_elements:
        issues.append("All outputs are zero")
    if abs(mean) > 1000:
        issues.append(f"Unusually large mean: {mean}")
    
    status = "OK" if not issues else "WARNING"
    
    return {
        "status": status,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "zero_fraction": num_zeros / total_elements,
        "shape": list(features.shape),
        "issues": issues,
        "input_mean": input_mean,
        "input_std": input_std,
        "input_dim": input_dim,
    }


def verify_downstream_logits(model: nn.Module, data_loader, device: str = "cpu") -> dict:
    """Sanity-check that a full downstream model produces reasonable logits."""
    model.eval()
    model = model.to(device)
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = prepare_batch_for_topobench(batch)
    with torch.no_grad():
        try:
            logits = model(batch)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    mean = logits.mean().item()
    std = logits.std().item()
    issues = []
    if std < 1e-6:
        issues.append("Very low logit variance")
    status = "OK" if not issues else "WARNING"
    return {
        "status": status,
        "mean": mean,
        "std": std,
        "shape": list(logits.shape),
        "issues": issues,
    }


class LinearClassifier(nn.Module):
    """Linear classifier for probing."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class DownstreamModel(nn.Module):
    """Encoder (``TBModelNodeEncoder``) + linear classifier.

    ``freeze_encoder=True`` (linear-probe / random-init-linear-probe): all encoder parameters,
    including inner GNN and wrapper ``LayerNorm``, are frozen; classifier updates only.

    ``freeze_encoder=False`` (full-finetune / random-init-full-finetune): encoder parameters train;
    submodule ``readout`` inside the nested ``TBModel`` remains frozen by ``TBModelNodeEncoder``
    because that head is not in the forward path.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        freeze_encoder: bool = True,
        task_level: str = "node",
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder
        self.task_level = task_level
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        freeze_batchnorm_eval_no_track(self.encoder)
    
    def forward(self, batch):
        """Forward pass."""
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(batch)
        else:
            features = self.encoder(batch)
        
        return self.classifier(features)
    
    def train(self, mode=True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

