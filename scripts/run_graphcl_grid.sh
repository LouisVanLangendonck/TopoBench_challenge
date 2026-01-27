#!/bin/bash

# =============================================================================
# Grid search script for GraphCL pretraining experiments
# =============================================================================
# This script uses Hydra's --multirun to automatically generate all combinations
# of hyperparameters for GraphCL pretraining.
#
# Runs 4 parallel jobs on different GPUs (devices 0-3) to speed up training.
#
# Hyperparameters tested:
# - Augmentation combinations (aug1, aug2) with different ratios
# - Homophily ranges in the dataset
# - Training epochs
#
# Usage:
#   bash scripts/run_graphcl_grid.sh
# =============================================================================

# Device 0: drop_node augmentations
python3 -m topobench \
    dataset=graph/GraphUniverse_graphcl \
    model=graph/gps_graphcl \
    model.backbone_wrapper.aug1=drop_node \
    model.backbone_wrapper.aug2=drop_edge,mask_attr,subgraph \
    model.backbone_wrapper.aug_ratio1=0.2,0.5 \
    model.backbone_wrapper.aug_ratio2=0.2,0.5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    trainer.max_epochs=20,100 \
    trainer.min_epochs=10 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=20 \
    logger.wandb.project=graphcl_pretraining \
    tags="[graphcl_grid,aug1_drop_node]" \
    --multirun &

sleep 10

# Device 1: drop_edge augmentations
python3 -m topobench \
    dataset=graph/GraphUniverse_graphcl \
    model=graph/gps_graphcl \
    model.backbone_wrapper.aug1=drop_edge \
    model.backbone_wrapper.aug2=drop_node,mask_attr,subgraph \
    model.backbone_wrapper.aug_ratio1=0.2,0.5 \
    model.backbone_wrapper.aug_ratio2=0.2,0.5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    trainer.max_epochs=20,100 \
    trainer.min_epochs=10 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=20 \
    logger.wandb.project=graphcl_pretraining \
    tags="[graphcl_grid,aug1_drop_edge]" \
    --multirun &

# Device 2: mask_attr augmentations
python3 -m topobench \
    dataset=graph/GraphUniverse_graphcl \
    model=graph/gps_graphcl \
    model.backbone_wrapper.aug1=mask_attr \
    model.backbone_wrapper.aug2=drop_node,drop_edge,subgraph \
    model.backbone_wrapper.aug_ratio1=0.2,0.5 \
    model.backbone_wrapper.aug_ratio2=0.2,0.5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    trainer.max_epochs=20,100 \
    trainer.min_epochs=10 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=20 \
    logger.wandb.project=graphcl_pretraining \
    tags="[graphcl_grid,aug1_mask_attr]" \
    --multirun &

# Device 3: subgraph augmentations
python3 -m topobench \
    dataset=graph/GraphUniverse_graphcl \
    model=graph/gps_graphcl \
    model.backbone_wrapper.aug1=subgraph \
    model.backbone_wrapper.aug2=drop_node,drop_edge,mask_attr \
    model.backbone_wrapper.aug_ratio1=0.2,0.5 \
    model.backbone_wrapper.aug_ratio2=0.2,0.5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    trainer.max_epochs=20,100 \
    trainer.min_epochs=10 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=20 \
    logger.wandb.project=graphcl_pretraining \
    tags="[graphcl_grid,aug1_subgraph]" \
    --multirun &

