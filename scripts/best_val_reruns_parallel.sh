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

# simplicial/hopse_g  |  simplicial/mantra_betti_numbers  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_betti_numbers transforms.hopse_encoding.pretrain_model=zinc 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_betti_numbers__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_betti_numbers  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_betti_numbers transforms.hopse_encoding.pretrain_model=zinc 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_betti_numbers__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_betti_numbers  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_betti_numbers transforms.hopse_encoding.pretrain_model=zinc 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_betti_numbers__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_betti_numbers  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_betti_numbers transforms.hopse_encoding.pretrain_model=zinc 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_betti_numbers__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_betti_numbers  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_betti_numbers transforms.hopse_encoding.pretrain_model=zinc 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=4 model.feature_encoder.out_channels=128 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=128 dataset.parameters.monitor_metric=loss trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_betti_numbers__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_name  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_name transforms.hopse_encoding.pretrain_model=molpcba 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0]' model.backbone.n_layers=2 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.5 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_name__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_orientation  |  data_seed=0
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_orientation transforms.hopse_encoding.pretrain_model=molpcba 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=0 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_orientation__ds0 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_orientation  |  data_seed=3
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_orientation transforms.hopse_encoding.pretrain_model=molpcba 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=3 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_orientation__ds3 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_orientation  |  data_seed=5
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_orientation transforms.hopse_encoding.pretrain_model=molpcba 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=5 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_orientation__ds5 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_orientation  |  data_seed=7
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_orientation transforms.hopse_encoding.pretrain_model=molpcba 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=7 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_orientation__ds7 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

# simplicial/hopse_g  |  simplicial/mantra_orientation  |  data_seed=9
_acquire_rerun_slot
python -m topobench model=simplicial/hopse_g dataset=simplicial/mantra_orientation transforms.hopse_encoding.pretrain_model=molpcba 'transforms.hopse_encoding.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' transforms.hopse_encoding.max_hop=2 transforms.hopse_encoding.max_rank=2 'model.feature_encoder.selected_dimensions=[0,1,2]' 'model.preprocessing_params.neighborhoods=[up_adjacency-0,2-up_adjacency-0]' model.backbone.n_layers=1 model.feature_encoder.out_channels=256 model.feature_encoder.proj_dropout=0.25 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 dataset.dataloader_params.batch_size=256 dataset.parameters.monitor_metric=f1 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 delete_checkpoint_after_test=True '+combined_feature_encodings.preprocessor_device='"'"'cuda'"'"'' dataset.split_params.data_seed=9 trainer.max_epochs=500 callbacks.early_stopping.patience=10 deterministic=True +logger.wandb.entity=gbg141-hopse logger.wandb.project=best_runs_rerun +logger.wandb.name=simplicial__hopse_g__simplicial__mantra_orientation__ds9 trainer.devices=[${_gpu}] &
slot_pids[$_RERUN_SLOT_IDX]=$!

wait
echo "All parallel reruns finished."
