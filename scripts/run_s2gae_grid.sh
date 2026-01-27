#!/bin/bash

# S2GAE Pretraining Grid Search Script
# This script runs S2GAE pretraining experiments with different configurations
# across multiple GPUs in parallel

# Device 0: GPS model with dropout 0.0
python3 -m topobench \
    dataset=graph/GraphUniverse_s2gae \
    model=graph/gps_s2gae \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0 \
    model.backbone_wrapper.mask_ratio=0.25,0.5,0.85 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500,1000 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=s2gae_pretraining \
    tags="[s2gae_grid,gps,dropout_0.0]" \
    --multirun &

sleep 10

# Device 1: GPS model with dropout 0.2
python3 -m topobench \
    dataset=graph/GraphUniverse_s2gae \
    model=graph/gps_s2gae \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.2 \
    model.backbone_wrapper.mask_ratio=0.25,0.5,0.85 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500,1000 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=s2gae_pretraining \
    tags="[s2gae_grid,gps,dropout_0.2]" \
    --multirun &

sleep 10

# Device 2: GCN model with dropout 0.0
python3 -m topobench \
    dataset=graph/GraphUniverse_s2gae \
    model=graph/gcn_s2gae \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0 \
    model.backbone_wrapper.mask_ratio=0.25,0.5,0.85 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500,1000 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=s2gae_pretraining \
    tags="[s2gae_grid,gcn,dropout_0.0]" \
    --multirun &

sleep 10

# Device 3: GCN model with dropout 0.2
python3 -m topobench \
    dataset=graph/GraphUniverse_s2gae \
    model=graph/gcn_s2gae \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.2 \
    model.backbone_wrapper.mask_ratio=0.25,0.5,0.85 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500,1000 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=s2gae_pretraining \
    tags="[s2gae_grid,gcn,dropout_0.2]" \
    --multirun &

wait
echo "All S2GAE pretraining experiments completed!"

