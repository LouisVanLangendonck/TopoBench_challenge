#!/usr/bin/env python3
"""Quick test to verify pretraining configs load properly."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, '/home/louisvl/graphuniverse2')

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Import and register resolvers
from topobench.utils.config_resolvers import (
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_pretraining_transform,
    get_flattened_channels,
    get_monitor_metric,
    get_monitor_mode,
    get_non_relational_out_channels,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
    infer_topotune_num_cell_dimensions,
)

# Register all resolvers
OmegaConf.register_new_resolver("get_default_metrics", get_default_metrics, replace=True)
OmegaConf.register_new_resolver("get_default_trainer", get_default_trainer, replace=True)
OmegaConf.register_new_resolver("get_default_transform", get_default_transform, replace=True)
OmegaConf.register_new_resolver("get_pretraining_transform", get_pretraining_transform, replace=True)
OmegaConf.register_new_resolver("get_flattened_channels", get_flattened_channels, replace=True)
OmegaConf.register_new_resolver("get_monitor_metric", get_monitor_metric, replace=True)
OmegaConf.register_new_resolver("get_monitor_mode", get_monitor_mode, replace=True)
OmegaConf.register_new_resolver("get_non_relational_out_channels", get_non_relational_out_channels, replace=True)
OmegaConf.register_new_resolver("get_required_lifting", get_required_lifting, replace=True)
OmegaConf.register_new_resolver("infer_in_channels", infer_in_channels, replace=True)
OmegaConf.register_new_resolver("infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True)
OmegaConf.register_new_resolver("infer_topotune_num_cell_dimensions", infer_topotune_num_cell_dimensions, replace=True)
OmegaConf.register_new_resolver("parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True)

def test_config(dataset, model, pretraining):
    """Test a specific configuration."""
    
    with initialize_config_dir(version_base="1.3", config_dir="/home/louisvl/graphuniverse2/configs"):
        cfg = compose(
            config_name="run.yaml",
            overrides=[
                f"dataset={dataset}",
                f"model={model}",
                f"pretraining={pretraining}"
            ]
        )
        
        print(f"\n{'='*80}")
        print(f"Testing: dataset={dataset}, model={model}, pretraining={pretraining}")
        print('='*80)
        
        # Print key info
        if hasattr(cfg, 'pretraining'):
            print(f"\n✓ Pretraining:")
            print(f"  - Enabled: {cfg.pretraining.enabled}")
            print(f"  - Task: {cfg.pretraining.task}")
        
        print(f"\nDataset Parameters:")
        print(f"  - Task: {cfg.dataset.parameters.task}")
        print(f"  - Loss Type: {cfg.dataset.parameters.loss_type}")
        print(f"  - Monitor Metric: {cfg.dataset.parameters.monitor_metric}")
        print(f"  - Task Level: {cfg.dataset.parameters.task_level}")
        
        print(f"\nLoss & Evaluator:")
        print(f"  - Loss Target: {cfg.loss._target_}")
        print(f"  - Evaluator Target: {cfg.evaluator._target_}")
        
        print(f"\n✓ Config loaded successfully!\n")
        
        return cfg


if __name__ == "__main__":
    # Test 1: Supervised learning (backward compatibility)
    print("\n" + "🔍 TEST 1: SUPERVISED LEARNING (pretraining=none)")
    cfg1 = test_config(
        dataset="graph/ogbg-molhiv",
        model="graph/gps",
        pretraining="none"
    )
    
    # Test 2: GraphMAEv2 pretraining
    print("\n" + "🔍 TEST 2: GRAPHMAEV2 PRETRAINING")
    cfg2 = test_config(
        dataset="graph/ogbg-molhiv",
        model="graph/gps",
        pretraining="graphmaev2"
    )
    
    # Test 3: DGI pretraining
    print("\n" + "🔍 TEST 3: DGI PRETRAINING")
    cfg3 = test_config(
        dataset="graph/ogbg-molhiv",
        model="graph/gps",
        pretraining="dgi"
    )
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80 + "\n")
