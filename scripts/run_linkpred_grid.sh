#!/bin/bash

# =============================================================================
# Grid search script for Link Prediction pretraining experiments
# =============================================================================
# This script uses Hydra's --multirun to automatically generate all combinations
# of hyperparameters for Link Prediction pretraining.
#
# Runs 4 parallel jobs on different GPUs (devices 0-3) for GPS and GCN models.
#
# Fixed settings:
# - 500 graphs for training
#
# Hyperparameters tested:
# - Model architectures: GPS, GCN
# - Homophily ranges: [0.0, 0.1] (low), [0.4, 0.6] (medium), [0.9, 1.0] (high)
# - Number of layers: 2, 4
# - Dropout: 0.0, 0.2
# - Mask ratio: 0.2, 0.5
# - Decoder types: mlp, bilinear
#
# Total experiments per model: 3 homophily × 2 layers × 2 dropout × 2 mask_ratio × 2 decoder = 48 experiments
# Total experiments: 2 models × 48 = 96 experiments
#
# Usage:
#   bash scripts/run_linkpred_grid.sh
# =============================================================================

# Device 0: GPS model (first half of homophily configs)
python3 -m topobench \
    dataset=graph/GraphUniverse_linkpred \
    model=graph/gps_linkpred \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0,0.2 \
    model.backbone_wrapper.mask_ratio=0.2,0.5 \
    model.readout.decoder_type=mlp,bilinear \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=linkpred_pretraining \
    tags="[linkpred_grid,gps,low_med_homophily]" \
    --multirun &

sleep 10

# Device 1: GPS model (high homophily)
python3 -m topobench \
    dataset=graph/GraphUniverse_linkpred \
    model=graph/gps_linkpred \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0,0.2 \
    model.backbone_wrapper.mask_ratio=0.2,0.5 \
    model.readout.decoder_type=mlp,bilinear \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=linkpred_pretraining \
    tags="[linkpred_grid,gps,high_homophily]" \
    --multirun &

sleep 10

# Device 2: GCN model (first half of homophily configs)
python3 -m topobench \
    dataset=graph/GraphUniverse_linkpred \
    model=graph/gcn_linkpred \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0,0.2 \
    model.backbone_wrapper.mask_ratio=0.2,0.5 \
    model.readout.decoder_type=mlp,bilinear \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=linkpred_pretraining \
    tags="[linkpred_grid,gcn,low_med_homophily]" \
    --multirun &

sleep 10

# Device 3: GCN model (high homophily)
python3 -m topobench \
    dataset=graph/GraphUniverse_linkpred \
    model=graph/gcn_linkpred \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0,0.2 \
    model.backbone_wrapper.mask_ratio=0.2,0.5 \
    model.readout.decoder_type=mlp,bilinear \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=500 \
    trainer.max_epochs=100 \
    trainer.min_epochs=20 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=linkpred_pretraining \
    tags="[linkpred_grid,gcn,high_homophily]" \
    --multirun &

echo ""
echo "Started 4 parallel Link Prediction grid searches on devices 0-3"
echo "Device 0 (GPS): 2 homophily × 2 layers × 2 dropout × 2 mask_ratio × 2 decoder = 32 experiments"
echo "Device 1 (GPS): 1 homophily × 2 layers × 2 dropout × 2 mask_ratio × 2 decoder = 16 experiments"
echo "Device 2 (GCN): 2 homophily × 2 layers × 2 dropout × 2 mask_ratio × 2 decoder = 32 experiments"
echo "Device 3 (GCN): 1 homophily × 2 layers × 2 dropout × 2 mask_ratio × 2 decoder = 16 experiments"
echo "Total experiments: 96"
echo ""
echo "Settings:"
echo "  Homophily ranges: [0.0, 0.1] (low), [0.4, 0.6] (medium), [0.9, 1.0] (high)"
echo "  Number of graphs: 500"
echo "  Number of layers: 2, 4"
echo "  Dropout: 0.0, 0.2"
echo "  Mask ratio: 0.2, 0.5"
echo "  Decoder types: mlp, bilinear"
echo "  Max epochs: 100"
echo "  Early stopping patience: 10"
echo ""

