#!/usr/bin/env bash
# Auto-generated: same best-val reruns as best_val_reruns_sequential.sh, with bounded parallelism.
# Uses virtual GPU slots + wait -n (same pattern as scripts/hopse_m.sh): never launch all jobs at once.

# Concurrent jobs per physical GPU at generation time; override without regenerating:
_JOBS_PER_GPU="${RERUN_JOBS_PER_GPU:-1}"

# Optional: match hopse_m.sh thread limits when multiple jobs share a machine
# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

_PHYSICAL_GPUS=(2 3)
gpus=()
for gpu in "${_PHYSICAL_GPUS[@]}"; do
  for ((j=1; j<=_JOBS_PER_GPU; j++)); do gpus+=("$gpu"); done
done
declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done

_acquire_rerun_slot() {
    assigned_slot=-1
    while [ "$assigned_slot" -eq -1 ]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i
                break
            fi
        done
        if [ "$assigned_slot" -eq -1 ]; then
            wait -n
        fi
    done
    _RERUN_SLOT_IDX=$assigned_slot
    _gpu="${gpus[$assigned_slot]}"
}

echo "Parallel reruns: ${#gpus[@]} slot(s) (${_JOBS_PER_GPU} job(s)/GPU × ${#_PHYSICAL_GPUS[@]} GPU(s))."

# simplicial/sann  |  graph/BBB_Martins  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/BBB_Martins transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__BBB_Martins__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/BBB_Martins  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/BBB_Martins transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__BBB_Martins__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/BBB_Martins  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/BBB_Martins transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__BBB_Martins__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/BBB_Martins  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/BBB_Martins transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__BBB_Martins__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/BBB_Martins  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/BBB_Martins transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__BBB_Martins__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/CYP3A4_Veith  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/CYP3A4_Veith transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__CYP3A4_Veith__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/CYP3A4_Veith  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/CYP3A4_Veith transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__CYP3A4_Veith__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/CYP3A4_Veith  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/CYP3A4_Veith transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__CYP3A4_Veith__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/CYP3A4_Veith  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/CYP3A4_Veith transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__CYP3A4_Veith__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/CYP3A4_Veith  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/CYP3A4_Veith transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__CYP3A4_Veith__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Caco2_Wang  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Caco2_Wang transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Caco2_Wang__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Caco2_Wang  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Caco2_Wang transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Caco2_Wang__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Caco2_Wang  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Caco2_Wang transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Caco2_Wang__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Caco2_Wang  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Caco2_Wang transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Caco2_Wang__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Caco2_Wang  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Caco2_Wang transforms.sann_encoding.max_hop=1 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Caco2_Wang__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Clearance_Hepatocyte_AZ transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Clearance_Hepatocyte_AZ__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Clearance_Hepatocyte_AZ transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Clearance_Hepatocyte_AZ__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Clearance_Hepatocyte_AZ transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Clearance_Hepatocyte_AZ__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Clearance_Hepatocyte_AZ transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Clearance_Hepatocyte_AZ__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/Clearance_Hepatocyte_AZ transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__Clearance_Hepatocyte_AZ__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/MUTAG  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/MUTAG transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__MUTAG__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/MUTAG  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/MUTAG transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__MUTAG__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/MUTAG  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/MUTAG transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__MUTAG__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/MUTAG  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/MUTAG transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__MUTAG__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/MUTAG  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/MUTAG transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=3 transforms.sann_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__MUTAG__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI1  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI1 transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI1__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI1  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI1 transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI1__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI1  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI1 transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI1__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI1  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI1 transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI1__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI1  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI1 transforms.sann_encoding.max_hop=2 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI1__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI109  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI109 transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI109__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI109  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI109 transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI109__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI109  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI109 transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI109__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI109  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI109 transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI109__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/NCI109  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/NCI109 transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__NCI109__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/PROTEINS  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/PROTEINS transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__PROTEINS__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/PROTEINS  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/PROTEINS transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__PROTEINS__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/PROTEINS  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/PROTEINS transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__PROTEINS__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/PROTEINS  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/PROTEINS transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__PROTEINS__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  graph/PROTEINS  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=graph/PROTEINS transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__graph__PROTEINS__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_betti_numbers  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_betti_numbers transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_betti_numbers__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_betti_numbers  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_betti_numbers transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_betti_numbers__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_betti_numbers  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_betti_numbers transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_betti_numbers__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_betti_numbers  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_betti_numbers transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_betti_numbers__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_betti_numbers  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_betti_numbers transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=4 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_betti_numbers__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_name  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_name transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_name__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_name  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_name transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_name__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_name  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_name transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_name__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_name  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_name transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_name__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_name  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_name transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_name__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_orientation  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_orientation transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_orientation__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_orientation  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_orientation transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_orientation__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_orientation  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_orientation transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_orientation__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_orientation  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_orientation transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_orientation__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sann  |  simplicial/mantra_orientation  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sann dataset=simplicial/mantra_orientation transforms.sann_encoding.max_hop=3 transforms.sann_encoding.complex_dim=2 transforms.sann_encoding.max_rank=1 'model.feature_encoder.selected_dimensions=[0,1]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sann__simplicial__mantra_orientation__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/BBB_Martins  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/BBB_Martins 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__BBB_Martins__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/BBB_Martins  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/BBB_Martins 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__BBB_Martins__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/BBB_Martins  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/BBB_Martins 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__BBB_Martins__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/BBB_Martins  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/BBB_Martins 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__BBB_Martins__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/BBB_Martins  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/BBB_Martins 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__BBB_Martins__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/CYP3A4_Veith  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/CYP3A4_Veith 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__CYP3A4_Veith__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/CYP3A4_Veith  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/CYP3A4_Veith 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__CYP3A4_Veith__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/CYP3A4_Veith  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/CYP3A4_Veith 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__CYP3A4_Veith__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/CYP3A4_Veith  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/CYP3A4_Veith 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__CYP3A4_Veith__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/CYP3A4_Veith  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/CYP3A4_Veith 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__CYP3A4_Veith__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Caco2_Wang  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Caco2_Wang 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Caco2_Wang__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Caco2_Wang  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Caco2_Wang 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Caco2_Wang__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Caco2_Wang  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Caco2_Wang 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Caco2_Wang__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Caco2_Wang  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Caco2_Wang 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Caco2_Wang__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Caco2_Wang  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Caco2_Wang 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Caco2_Wang__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Clearance_Hepatocyte_AZ 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Clearance_Hepatocyte_AZ__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Clearance_Hepatocyte_AZ 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Clearance_Hepatocyte_AZ__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Clearance_Hepatocyte_AZ 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Clearance_Hepatocyte_AZ__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Clearance_Hepatocyte_AZ 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Clearance_Hepatocyte_AZ__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/Clearance_Hepatocyte_AZ  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/Clearance_Hepatocyte_AZ 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=mae trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__Clearance_Hepatocyte_AZ__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/MUTAG  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/MUTAG 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__MUTAG__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/MUTAG  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/MUTAG 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__MUTAG__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/MUTAG  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/MUTAG 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__MUTAG__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/MUTAG  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/MUTAG 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__MUTAG__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/MUTAG  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/MUTAG 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__MUTAG__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI1  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI1 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI1__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI1  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI1 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI1__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI1  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI1 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI1__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI1  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI1 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI1__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI1  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI1 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI1__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI109  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI109 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI109__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI109  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI109 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI109__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI109  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI109 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI109__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI109  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI109 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI109__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/NCI109  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/NCI109 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=1 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__NCI109__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/PROTEINS  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/PROTEINS 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__PROTEINS__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/PROTEINS  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/PROTEINS 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__PROTEINS__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/PROTEINS  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/PROTEINS 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__PROTEINS__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/PROTEINS  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/PROTEINS 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__PROTEINS__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  graph/PROTEINS  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=graph/PROTEINS 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=2 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=accuracy trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__graph__PROTEINS__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_betti_numbers  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_betti_numbers 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_betti_numbers__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_betti_numbers  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_betti_numbers 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_betti_numbers__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_betti_numbers  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_betti_numbers 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_betti_numbers__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_betti_numbers  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_betti_numbers 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_betti_numbers__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_betti_numbers  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_betti_numbers 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.0 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_betti_numbers__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_name  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_name 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_name__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_name  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_name 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_name__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_name  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_name 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_name__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_name  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_name 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_name__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_name  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_name 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_name__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_orientation  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_orientation 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_orientation__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_orientation  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_orientation 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_orientation__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_orientation  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_orientation 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_orientation__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_orientation  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_orientation 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_orientation__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/sccnn  |  simplicial/mantra_orientation  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_orientation 'model.feature_encoder.selected_dimensions=[0,1,2]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__sccnn__simplicial__mantra_orientation__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

wait
echo "All parallel reruns finished."
