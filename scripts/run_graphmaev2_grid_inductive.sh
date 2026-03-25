# Device 0: GPS model
python3 -m topobench \
    dataset=graph/GraphUniverse_graphmaev2 \
    model=graph/gps_graphmaev2 \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0 \
    model.backbone_wrapper.mask_rate=0.5,0.85 \
    model.backbone_wrapper.drop_edge_rate=0.2,0.4 \
    model.feature_encoder.out_channels=128,256 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=5000 \
    trainer.max_epochs=50 \
    trainer.min_epochs=20 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=graphmaev2_pretraining_large_rerun \
    tags="[graphmaev2_grid,gps,dropout_0.0]" \
    --multirun &

sleep 10

# Device 1: GPS model (high homophily)
python3 -m topobench \
    dataset=graph/GraphUniverse_graphmaev2 \
    model=graph/gps_graphmaev2 \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.2 \
    model.backbone_wrapper.mask_rate=0.5,0.85 \
    model.backbone_wrapper.drop_edge_rate=0.2,0.4 \
    model.feature_encoder.out_channels=128,256 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=5000 \
    trainer.max_epochs=50 \
    trainer.min_epochs=20 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=graphmaev2_pretraining_large_rerun \
    tags="[graphmaev2_grid,gps,dropout_0.2]" \
    --multirun &

sleep 10

# Device 2: GCN model (first half of homophily configs)
python3 -m topobench \
    dataset=graph/GraphUniverse_graphmaev2 \
    model=graph/gcn_graphmaev2 \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.0 \
    model.backbone_wrapper.mask_rate=0.5,0.85 \
    model.backbone_wrapper.drop_edge_rate=0.2,0.4 \
    model.feature_encoder.out_channels=128,256 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=5000 \
    trainer.max_epochs=50 \
    trainer.min_epochs=20 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=graphmaev2_pretraining_large_rerun \
    tags="[graphmaev2_grid,gcn,dropout_0.0]" \
    --multirun &

sleep 10

# Device 3: GCN model (high homophily)
python3 -m topobench \
    dataset=graph/GraphUniverse_graphmaev2 \
    model=graph/gcn_graphmaev2 \
    model.backbone.num_layers=2,4 \
    model.backbone.dropout=0.2 \
    model.backbone_wrapper.mask_rate=0.5,0.85 \
    model.backbone_wrapper.drop_edge_rate=0.2,0.4 \
    model.feature_encoder.out_channels=128,256 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.0,0.1\],\[0.4,0.6\],\[0.9,1.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=5000 \
    trainer.max_epochs=50 \
    trainer.min_epochs=20 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=graphmaev2_pretraining_large_rerun \
    tags="[graphmaev2_grid,gcn,dropout_0.2]" \
    --multirun &

