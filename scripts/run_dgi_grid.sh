#!/bin/bash

# =============================================================================
# Grid search script for DGI pretraining experiments
# =============================================================================
# This script uses Hydra's --multirun to automatically generate all combinations
# of hyperparameters for DGI pretraining.
#
# Runs 2 parallel jobs on different GPUs (devices 0-1) for GCN and GAT.
#
# Fixed settings:
# - 500 graphs for training
#
# Hyperparameters tested:
# - Model architectures: GCN, GAT
# - Homophily ranges: [0.0, 0.1] (low), [0.4, 0.6] (medium), [0.9, 1.0] (high)
# - Number of layers: 2, 4
# - Dropout: 0.0, 0.2
# - Corruption types: shuffle, random
# - Discriminator types: bilinear, mlp
#
# Total experiments per model: 3 homophily × 2 layers × 2 dropout × 2 corruption × 2 discriminator = 48 experiments
# Total experiments: 2 models × 48 = 96 experiments
#
# Usage:
#   bash scripts/run_dgi_grid.sh
# =============================================================================

# Device 0: GCN model
python3 -m topobench \
    dataset=graph/GraphUniverse_dgi \
    model=graph/gcn_dgi \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0,0.2 \
    model.backbone_wrapper.corruption_type=shuffle,random \
    model.readout.discriminator_type=bilinear,mlp \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=20 \
    logger.wandb.project=dgi_pretraining \
    tags="[dgi_grid,gcn]" \
    --multirun &

sleep 10

# Device 1: GAT model
python3 -m topobench \
    dataset=graph/GraphUniverse_dgi \
    model=graph/gat_dgi \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0,0.2 \
    model.backbone_wrapper.corruption_type=shuffle,random \
    model.readout.discriminator_type=bilinear,mlp \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=20 \
    logger.wandb.project=dgi_pretraining \
    tags="[dgi_grid,gat]" \
    --multirun &

echo ""
echo "Started 2 parallel DGI grid searches on devices 0-1"
echo "Device 0 (GCN): 3 homophily × 2 layers × 2 dropout × 2 corruption × 2 discriminator = 48 experiments"
echo "Device 1 (GAT): 3 homophily × 2 layers × 2 dropout × 2 corruption × 2 discriminator = 48 experiments"
echo "Total experiments: 96"
echo ""
echo "Settings:"
echo "  Homophily ranges: [0.0, 0.1] (low), [0.4, 0.6] (medium), [0.9, 1.0] (high)"
echo "  Number of graphs: 500"
echo "  Number of layers: 2, 4"
echo "  Dropout: 0.0, 0.2"
echo "  Corruption types: shuffle, random"
echo "  Discriminator types: bilinear, mlp"
echo "  Max epochs: 100"
echo ""

