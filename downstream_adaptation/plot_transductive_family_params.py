"""
Plot transductive community detection results organized by varying family parameters.

This script fetches results from a transductive downstream wandb project and creates
comparison plots where:
- Subplots are organized by unique combinations of varying family_parameters
- If 2 params varied: one makes columns, the other makes rows
- Each subplot shows: different (mode, model_type, hyperparams) as different lines
- X-axis: n_train (number of finetuning nodes)
- Y-axis: Test accuracy
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import defaultdict
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("ERROR: wandb not installed. Please install with: pip install wandb")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Publication quality plot settings
PLOT_STYLE = {
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "font.family": "DejaVu Serif",
    "axes.linewidth": 1.1,
    "xtick.direction": 'out',
    "ytick.direction": 'out',
    "axes.titleweight": 'bold',
    "figure.titlesize": 21
}

# Mode definitions - CRITICAL for correct filtering
# Based on actual modes in data: 'linear', 'finetune-linear', 'scratch_frozen', 'scratch'
RANDOM_INIT_MODES = {'scratch', 'scratch_frozen'}
LINEAR_PROBE_MODES = {'scratch_frozen', 'linear'}  # frozen backbone
COMPLETE_TRAINING_MODES = {'scratch', 'finetune-linear'}  # unfrozen backbone (finetune-linear is full model training!)


# =============================================================================
# Data Fetching and Extraction
# =============================================================================

def fetch_runs_from_wandb(project_paths: List[str]) -> List[Dict[str, Any]]:
    """Fetch all finished runs from one or more wandb projects."""
    api = wandb.Api()
    all_run_data = []
    
    for project_path in project_paths:
        print(f"\n{'=' * 80}")
        print(f"FETCHING RUNS FROM: {project_path}")
        print(f"{'=' * 80}")
        
        runs = api.runs(project_path, filters={"state": "finished"})
        
        for run in runs:
            data = {
                "run_id": run.id,
                "run_name": run.name,
                "config": dict(run.config),
                "summary": dict(run.summary),
                "project": project_path,
            }
            all_run_data.append(data)
            print(f"  ✓ {run.id} ({run.name})")
        
        project_count = len([r for r in all_run_data if r['project'] == project_path])
        print(f"\nFound {project_count} finished runs in {project_path}")
    
    print(f"\nTOTAL RUNS: {len(all_run_data)}\n")
    return all_run_data


def get_nested_value(d: dict, key_path: str, default=None):
    """Get value from nested dict using '/' separated path."""
    if key_path in d:
        return d[key_path]
    
    keys = key_path.split('/')
    value = d
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def extract_family_parameters(config: dict) -> dict:
    """Extract all family_parameters from pretraining config."""
    family_params = {}
    
    # Try flat structure first
    prefix = 'pretrain/dataset/loader/parameters/generation_parameters/family_parameters/'
    for key in config.keys():
        if key.startswith(prefix):
            param_name = key[len(prefix):]
            family_params[param_name] = config[key]
    
    # Try nested structure
    if not family_params:
        nested_value = get_nested_value(
            config, 
            'pretrain/dataset/loader/parameters/generation_parameters/family_parameters'
        )
        if isinstance(nested_value, dict):
            family_params.update(nested_value)
    
    return family_params


def normalize_range_value(val):
    """Normalize range values to hashable format (tuple)."""
    if isinstance(val, list):
        return tuple(val)
    return val


def extract_model_hyperparams(config: dict) -> dict:
    """Extract model type, data_name, and varying hyperparameters."""
    model_type = config.get('pretrain/model/model_name', 'unknown')
    data_name = get_nested_value(
        config, 
        'pretrain/dataset/loader/parameters/data_name', 
        'unknown'
    )
    
    # Hyperparameters to extract
    hyperparam_paths = [
        'pretrain/model/backbone_wrapper/mask_rate',
        'pretrain/model/backbone_wrapper/drop_edge_rate',
        'pretrain/model/backbone_wrapper/replace_rate',
        'pretrain/model/backbone_wrapper/edge_sample_ratio',
        'pretrain/model/readout/decoder_type',
        'pretrain/optimizer/parameters/lr',
    ]
    
    hyperparams = {}
    for path in hyperparam_paths:
        val = get_nested_value(config, path)
        if val is not None:
            param_name = path.split('/')[-1]
            hyperparams[param_name] = val
    
    return {
        'model_type': model_type,
        'data_name': data_name,
        'hyperparams': hyperparams,
    }


def process_transductive_runs(run_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process transductive downstream evaluation runs into a dataframe."""
    records = []
    
    for run in run_data:
        config = run['config']
        summary = run['summary']
        
        # Check for accuracy metric
        if 'test/accuracy' not in summary:
            continue
        
        # Extract family parameters and model metadata
        family_params = extract_family_parameters(config)
        model_meta = extract_model_hyperparams(config)
        
        record = {
            'run_id': run['run_id'],
            'run_name': run['run_name'],
            'model_type': model_meta['model_type'],
            'data_name': model_meta['data_name'],
            'n_train': config.get('n_train'),
            'mode': config.get('mode'),
            'test_accuracy': summary.get('test/accuracy'),
            'train_accuracy': summary.get('test/train_accuracy'),
            'val_accuracy': summary.get('test/val_accuracy'),
        }
        
        # Add family parameters with normalized values
        for param_name, param_value in family_params.items():
            record[f'family_{param_name}'] = normalize_range_value(param_value)
        
        # Add hyperparameters
        record.update(model_meta['hyperparams'])
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} transductive runs:")
        print(f"  Model types: {df['model_type'].unique().tolist()}")
        print(f"  Modes: {df['mode'].unique().tolist()}")
        print(f"  N_train values: {sorted(df['n_train'].unique().tolist())}")
        
        # Print family parameters
        family_cols = [col for col in df.columns if col.startswith('family_')]
        print(f"  Family parameters: {[col.replace('family_', '') for col in family_cols]}")
    
    return df


# =============================================================================
# Data Analysis and Preparation
# =============================================================================

def identify_varying_family_params(df: pd.DataFrame) -> List[str]:
    """Identify which family parameters vary across runs."""
    family_cols = [col for col in df.columns if col.startswith('family_')]
    
    varying = []
    for col in family_cols:
        non_na_values = df[col].dropna()
        if len(non_na_values) > 0 and non_na_values.nunique() > 1:
            varying.append(col)
    
    return sorted(varying)


def identify_variable_hyperparams(df: pd.DataFrame, model_type: str) -> List[str]:
    """Identify which hyperparameters vary for a given model type."""
    model_df = df[df['model_type'] == model_type]
    
    # Columns to exclude from hyperparam identification
    metadata_cols = {
        'run_id', 'run_name', 'model_type', 'n_train', 'mode',
        'test_accuracy', 'train_accuracy', 'val_accuracy', 'data_name'
    }
    family_cols = {col for col in df.columns if col.startswith('family_')}
    excluded_cols = metadata_cols | family_cols
    
    # Find varying hyperparameters
    varying = []
    for col in model_df.columns:
        if col in excluded_cols:
            continue
        non_na_values = model_df[col].dropna()
        if len(non_na_values) > 0 and non_na_values.nunique() > 1:
            varying.append(col)
    
    return sorted(varying)


def create_hyperparam_id(row: pd.Series, varying_params: List[str]) -> str:
    """Create unique ID for a hyperparameter configuration."""
    if not varying_params:
        return "default"
    
    parts = []
    for param in sorted(varying_params):
        val = row.get(param)
        if val is not None and not pd.isna(val):
            # Simplify parameter name and value
            param_short = param.replace('_', '')[:6]
            if isinstance(val, float):
                val_str = f"{val:.4g}"
            else:
                val_str = str(val)[:10]
            parts.append(f"{param_short}={val_str}")
    
    return ",".join(parts) if parts else "default"


def create_config_id(row: pd.Series, model_type: str, mode: str, hyperparam_id: str) -> str:
    """Create human-readable configuration ID."""
    if mode in RANDOM_INIT_MODES:
        return f"Random Init ({mode})"
    
    # Pretrained models
    mode_display = mode.replace('_', ' ').title()
    
    if hyperparam_id == "default":
        return f"{model_type} ({mode_display})"
    else:
        return f"{model_type} ({mode_display}, {hyperparam_id})"


def create_config_id_model_comparison(row: pd.Series, model_type: str, data_name: str, mode: str) -> str:
    """Create simplified config ID for model comparison (using data_name)."""
    if mode in RANDOM_INIT_MODES:
        # Determine proper label based on mode
        if mode in LINEAR_PROBE_MODES:
            return "Random Init (Linear Probe)"
        else:
            return "Random Init (Complete Model Training)"
    
    # Determine mode display name
    if mode in LINEAR_PROBE_MODES:
        mode_display = "Linear Probe"
    elif mode in COMPLETE_TRAINING_MODES:
        mode_display = "Complete Model Training"
    else:
        mode_display = mode.replace('_', ' ').title()
    
    # Extract meaningful part of data_name
    if data_name and data_name != 'unknown':
        # Remove common prefixes/suffixes
        data_display = data_name.replace('graph_family_', '').replace('_pretrain', '')
        data_display = data_display.replace('_', ' ').title()
        
        # Remove "Graphuniverse" prefix if present
        if data_display.startswith('Graphuniverse '):
            data_display = data_display[len('Graphuniverse '):]
        
        return f"{data_display} ({mode_display})"
    else:
        return f"{model_type} ({mode_display})"


def aggregate_runs(df: pd.DataFrame, varying_family_params: List[str]) -> pd.DataFrame:
    """
    Aggregate runs by (family_params, model_type, mode, n_train, hyperparams).
    Computes mean and std of test_accuracy across seeds.
    """
    # Identify varying hyperparameters per model type
    model_varying_hyperparams = {}
    for model_type in df['model_type'].unique():
        model_varying_hyperparams[model_type] = identify_variable_hyperparams(df, model_type)
    
    # Create hyperparam_id for each row
    df = df.copy()
    df['hyperparam_id'] = df.apply(
        lambda r: create_hyperparam_id(r, model_varying_hyperparams.get(r['model_type'], [])),
        axis=1
    )
    
    # Create config_id for each row
    df['config_id'] = df.apply(
        lambda r: create_config_id(r, r['model_type'], r['mode'], r['hyperparam_id']),
        axis=1
    )
    
    # Group and aggregate
    group_cols = (
        varying_family_params + 
        ['model_type', 'mode', 'n_train', 'hyperparam_id', 'config_id', 'data_name']
    )
    
    agg_dict = {
        'test_accuracy': ['mean', 'std', 'count'],
    }
    
    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    df_agg.columns = [
        col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
        for col in df_agg.columns
    ]
    
    return df_agg


def prepare_data_for_mode_separated_plot(
    df: pd.DataFrame, 
    varying_family_params: List[str], 
    target_mode: str
) -> pd.DataFrame:
    """
    Prepare data for mode-separated plots.
    
    FIXED: Correctly filters Linear Probe vs Complete Model Training modes.
    """
    # FIXED FILTERING LOGIC
    if target_mode == 'Linear Probe':
        # Linear Probe: frozen backbone (scratch_frozen or linear_finetune)
        df_filtered = df[df['mode'].isin(LINEAR_PROBE_MODES)].copy()
    elif target_mode == 'Complete Model Training':
        # Complete Model Training: unfrozen backbone (scratch or finetune)
        df_filtered = df[df['mode'].isin(COMPLETE_TRAINING_MODES)].copy()
    else:
        raise ValueError(f"Invalid target_mode: {target_mode}")
    
    print(f"  [Mode: {target_mode}] Filtered modes: {df_filtered['mode'].unique().tolist()}")
    print(f"  [Mode: {target_mode}] Row count: {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print(f"  [Mode Separated] No data found for {target_mode}")
        return pd.DataFrame()
    
    # Aggregate the filtered data
    df_aggregated = aggregate_runs(df_filtered, varying_family_params)
    
    # Separate Random Init from pretrained
    random_init_df = df_aggregated[df_aggregated['mode'].isin(RANDOM_INIT_MODES)].copy()
    random_init_df['config_id'] = f"Random Init ({target_mode})"
    
    # Process pretrained models
    pretrained_df = df_aggregated[~df_aggregated['mode'].isin(RANDOM_INIT_MODES)].copy()
    
    if len(pretrained_df) == 0:
        print(f"  [Mode: {target_mode}] Only Random Init data found")
        return random_init_df
    
    # Select best hyperparams per (family_params, data_name) group
    best_configs = []
    group_keys = varying_family_params + ['data_name']
    
    for group_vals, group in pretrained_df.groupby(group_keys):
        hyperparam_ids = group['hyperparam_id'].unique()
        
        if len(hyperparam_ids) == 1:
            best_configs.append(group)
        else:
            # Calculate average performance for each hyperparam
            hyperparam_scores = {}
            for hp_id in hyperparam_ids:
                hp_group = group[group['hyperparam_id'] == hp_id]
                avg_score = hp_group['test_accuracy_mean'].mean()
                hyperparam_scores[hp_id] = avg_score
            
            # Select best (higher accuracy is better)
            best_hp_id = max(hyperparam_scores, key=hyperparam_scores.get)
            
            best_group = group[group['hyperparam_id'] == best_hp_id]
            best_configs.append(best_group)
            
            # Log selection
            family_vals_dict = dict(zip(varying_family_params, group_vals[:len(varying_family_params)]))
            family_str = ', '.join([f"{k.replace('family_', '')}={v}" for k, v in family_vals_dict.items()])
            data_name = group_vals[-1]
            
            print(f"  [{family_str}] {data_name}: selected {best_hp_id} "
                  f"(score={hyperparam_scores[best_hp_id]:.4f})")
    
    # Combine pretrained best configs
    if best_configs:
        pretrained_best_df = pd.concat(best_configs, ignore_index=True)
        
        # Create simplified config_id
        pretrained_best_df['config_id'] = pretrained_best_df.apply(
            lambda r: create_config_id_model_comparison(r, r['model_type'], r['data_name'], r['mode']),
            axis=1
        )
    else:
        pretrained_best_df = pd.DataFrame()
    
    # Combine Random Init + Pretrained
    result_df = pd.concat([random_init_df, pretrained_best_df], ignore_index=True)
    
    print(f"  [Mode: {target_mode}] Prepared {len(result_df)} configurations")
    print(f"  [Mode: {target_mode}] Random Init rows: {len(random_init_df)}, "
          f"Pretrained rows: {len(pretrained_best_df)}")
    
    return result_df


def check_multiple_data_names(df: pd.DataFrame) -> bool:
    """Check if there are multiple pretrained model types (data_names)."""
    # Only check pretrained models (not Random Init)
    pretrained_df = df[~df['mode'].isin(RANDOM_INIT_MODES)]
    
    if len(pretrained_df) == 0:
        return False
    
    return pretrained_df['data_name'].nunique() > 1


# =============================================================================
# Color Mapping
# =============================================================================

def create_color_map_by_model_type(config_ids: List[str], df: pd.DataFrame, mode_separated: bool = False) -> Dict[str, str]:
    """
    Create color map following these rules:
    1. Each pretrained model type (data_name) gets a distinct base color
    2. Random Init modes are gray/black
    3. If mode_separated=False (mixing LP and CMT): CMT uses more translucent/grayish version of LP color
    4. If mode_separated=True: use full colors, no translucency
    5. Multiple hyperparam variants of same model: vary color slightly within family
    """
    # Define base colors for pretrained model types (by data_name)
    base_colors_palette = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.4, 0.8),  # Blue
        (0.2, 0.7, 0.3),  # Green
        (0.8, 0.5, 0.2),  # Orange
        (0.6, 0.2, 0.7),  # Purple
        (0.9, 0.6, 0.1),  # Gold
        (0.3, 0.7, 0.7),  # Cyan
        (0.9, 0.3, 0.5),  # Pink
    ]
    
    # Gray/black for random init - ALWAYS BLACK regardless of mode
    random_init_colors = {
        'scratch': (0.0, 0.0, 0.0),         # Black
        'scratch_frozen': (0.0, 0.0, 0.0),  # Black
    }
    
    # Group configs by pretrained model type (data_name)
    model_type_groups = defaultdict(list)
    
    for config_id in config_ids:
        # Find the data_name for this config
        config_rows = df[df['config_id'] == config_id]
        if len(config_rows) == 0:
            continue
            
        # Get mode to determine if random init
        mode = config_rows['mode'].iloc[0]
        
        if mode in RANDOM_INIT_MODES:
            # Random init gets its own entry
            model_type_groups[f'__random_init_{mode}__'].append(config_id)
        else:
            # Pretrained: group by data_name
            data_name = config_rows['data_name'].iloc[0]
            model_type_groups[data_name].append(config_id)
    
    # Assign colors
    color_map = {}
    pretrained_model_idx = 0
    
    for model_type, configs in sorted(model_type_groups.items()):
        if model_type.startswith('__random_init_'):
            # Random Init: assign gray/black
            mode = model_type.replace('__random_init_', '').replace('__', '')
            base_color = random_init_colors.get(mode, (0.4, 0.4, 0.4))
            for config_id in configs:
                color_map[config_id] = base_color
        else:
            # Pretrained model: assign from palette
            base_color = base_colors_palette[pretrained_model_idx % len(base_colors_palette)]
            pretrained_model_idx += 1
            
            # Separate by mode (Linear Probe vs Complete Model Training)
            lp_configs = []
            cmt_configs = []
            
            for config_id in configs:
                config_rows = df[df['config_id'] == config_id]
                mode = config_rows['mode'].iloc[0]
                
                if mode in LINEAR_PROBE_MODES:
                    lp_configs.append(config_id)
                elif mode in COMPLETE_TRAINING_MODES:
                    cmt_configs.append(config_id)
            
            # Assign colors to Linear Probe configs
            if len(lp_configs) == 1:
                color_map[lp_configs[0]] = base_color
            else:
                # Multiple variants: vary within color family
                for i, config_id in enumerate(sorted(lp_configs)):
                    # Vary brightness: 0.7 to 1.3
                    factor = 0.75 + (i * 0.5 / max(1, len(lp_configs) - 1))
                    varied_color = tuple(min(1.0, max(0.0, c * factor)) for c in base_color)
                    color_map[config_id] = varied_color
            
            # Assign colors to Complete Model Training configs
            if mode_separated:
                # Mode separated: use same approach as LP (full colors)
                if len(cmt_configs) == 1:
                    color_map[cmt_configs[0]] = base_color
                else:
                    for i, config_id in enumerate(sorted(cmt_configs)):
                        factor = 0.75 + (i * 0.5 / max(1, len(cmt_configs) - 1))
                        varied_color = tuple(min(1.0, max(0.0, c * factor)) for c in base_color)
                        color_map[config_id] = varied_color
            else:
                # Mixed plot: CMT gets more grayish/translucent version
                # Mix with gray to make it more muted
                gray = (0.6, 0.6, 0.6)
                gray_mix = 0.3  # 30% gray
                desaturated_base = tuple(c * (1 - gray_mix) + g * gray_mix for c, g in zip(base_color, gray))
                
                if len(cmt_configs) == 1:
                    color_map[cmt_configs[0]] = desaturated_base
                else:
                    for i, config_id in enumerate(sorted(cmt_configs)):
                        factor = 0.75 + (i * 0.5 / max(1, len(cmt_configs) - 1))
                        varied_color = tuple(min(1.0, max(0.0, c * factor)) for c, g in zip(desaturated_base, gray))
                        color_map[config_id] = varied_color
    
    return color_map


# =============================================================================
# Plotting Functions
# =============================================================================

def format_family_param_value(param_name: str, param_value) -> str:
    """Format family parameter value for display."""
    param_name_clean = param_name.replace('family_', '').replace('_', ' ').title()
    
    # Special case for transductive: rename n_nodes_range to Graph Size
    if 'n_nodes_range' in param_name.lower():
        param_name_clean = 'Graph Size'
    
    if isinstance(param_value, tuple):
        if len(param_value) == 2 and param_value[0] == param_value[1]:
            return f"{param_name_clean} = {param_value[0]}"
        else:
            return f"{param_name_clean} = {list(param_value)}"
    else:
        return f"{param_name_clean} = {param_value}"


def setup_plot_style():
    """Apply publication-quality plot style."""
    mpl.rcParams.update(PLOT_STYLE)


def create_subplot_grid(varying_params: List[str], df: pd.DataFrame) -> Tuple:
    """
    Create subplot grid layout based on varying parameters.
    
    Returns:
        (row_param, col_param, row_values, col_values)
    """
    if len(varying_params) == 1:
        row_param = None
        col_param = varying_params[0]
        row_values = [None]
        col_values = sorted(df[col_param].unique())
    elif len(varying_params) >= 2:
        row_param = varying_params[0]
        col_param = varying_params[1]
        row_values = sorted(df[row_param].unique())
        col_values = sorted(df[col_param].unique())
    else:
        raise ValueError("Need at least one varying parameter")
    
    return row_param, col_param, row_values, col_values


def plot_config_line(ax, x_vals, y_vals, y_errs, config_id, color, has_std):
    """Plot a single configuration line with error bars if available."""
    is_random_init = 'Random Init' in config_id
    marker = 's' if is_random_init else 'o'
    linestyle = '--' if is_random_init else '-'
    
    if has_std and np.any(y_errs > 0):
        ax.errorbar(
            x_vals, y_vals, yerr=y_errs,
            color=color, alpha=0.85,
            linewidth=2.5, marker=marker, markersize=7,
            linestyle=linestyle,
            capsize=5, capthick=2.0, elinewidth=2.0,
            label=config_id
        )
    else:
        ax.plot(
            x_vals, y_vals,
            color=color, alpha=0.85,
            linewidth=2.5, marker=marker, markersize=7,
            linestyle=linestyle,
            label=config_id
        )


def setup_subplot_appearance(ax, row_idx, col_idx, n_rows, n_cols):
    """Configure subplot appearance (labels, grid, scaling)."""
    # X-axis label only on bottom row
    if row_idx == n_rows - 1:
        ax.set_xlabel('# Training Nodes', fontsize=13)
    
    # Y-axis label on left side (leftmost column only)
    if col_idx == 0:
        ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    
    # Grid
    ax.grid(axis='both', alpha=0.20, zorder=0, linestyle=':', linewidth=1.0)


def add_performance_bands(ax, y_min, y_max):
    """Add colored performance bands to subplot."""
    if y_max > 50:
        ax.axhspan(50, min(100, y_max), facecolor='lightgreen', alpha=0.08, zorder=0)


def setup_log_scale_if_needed(ax, x_vals):
    """Set log scale for x-axis if data spans more than 10x."""
    if len(x_vals) > 0 and np.max(x_vals) / np.min(x_vals) > 10:
        ax.set_xscale('log')
        ax.set_xticks(sorted(np.unique(x_vals)))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())


def create_legend(fig, all_configs, color_map, location='right'):
    """Create and add legend to figure."""
    from matplotlib.lines import Line2D
    
    # Separate random init from pretrained
    pretrained_configs = [c for c in all_configs if 'Random Init' not in c]
    random_init_configs = [c for c in all_configs if 'Random Init' in c]
    
    legend_handles = []
    
    # Add pretrained configs
    for config in sorted(pretrained_configs):
        color = color_map[config]
        handle = Line2D([0], [0], color=color, linestyle='-',
                       linewidth=2.5, marker='o', markersize=6, label=config)
        legend_handles.append(handle)
    
    # Add random init configs
    for config in sorted(random_init_configs):
        color = color_map[config]
        handle = Line2D([0], [0], color=color, linestyle='--',
                       linewidth=2.5, marker='s', markersize=6, label=config)
        legend_handles.append(handle)
    
    # Add legend to figure
    if location == 'top':
        fig.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.97),  # Moved up from 0.95
            ncol=4,
            fontsize=15,  # Increased from 13
            frameon=True,
            fancybox=True,
            shadow=True,
            columnspacing=1.0,
            handlelength=2.5,
            handletextpad=0.5,
        )
    else:  # right side (default)
        fig.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(0.88, 0.5),
            fontsize=16,
            frameon=True,
            fancybox=True,
            shadow=True,
            markerscale=1.8,
            handlelength=3.0,
            handletextpad=1.0,
            labelspacing=0.8
        )


def create_mode_separated_plot(
    df: pd.DataFrame,
    target_mode: str,
    output_path: str,
    project_name: Optional[str] = None,
):
    """Create plot showing only one mode (Linear Probe OR Complete Model Training)."""
    setup_plot_style()
    
    # Identify varying family parameters
    varying_family_params = identify_varying_family_params(df)
    
    print(f"\n[Mode Separated - {target_mode}]")
    print(f"  Varying family parameters: {[p.replace('family_', '') for p in varying_family_params]}")
    
    if len(varying_family_params) == 0:
        print(f"  ERROR: No varying family parameters found")
        return
    
    # Prepare data for this mode
    df_agg = prepare_data_for_mode_separated_plot(df, varying_family_params, target_mode)
    
    if len(df_agg) == 0:
        print(f"  No data to plot")
        return
    
    # Create subplot layout
    row_param, col_param, row_values, col_values = create_subplot_grid(varying_family_params, df_agg)
    n_rows, n_cols = len(row_values), len(col_values)
    
    print(f"  Creating plot with {n_rows} rows × {n_cols} columns")
    
    # Get unique configurations and create color map
    all_configs = sorted(df_agg['config_id'].unique())
    color_map = create_color_map_by_model_type(all_configs, df_agg, mode_separated=True)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.28, wspace=0.20, left=0.08, right=0.92, top=0.86, bottom=0.10)  # top from 0.88 to 0.86
    
    # Plot each subplot
    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this subplot
            if row_param:
                subplot_data = df_agg[
                    (df_agg[row_param] == row_val) & (df_agg[col_param] == col_val)
                ]
            else:
                subplot_data = df_agg[df_agg[col_param] == col_val]
            
            if len(subplot_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                continue
            
            # Plot each configuration
            for config_id in all_configs:
                config_data = subplot_data[subplot_data['config_id'] == config_id]
                if len(config_data) == 0:
                    continue
                
                config_data = config_data.sort_values('n_train')
                
                x_vals = config_data['n_train'].values
                y_vals = config_data['test_accuracy_mean'].values * 100.0
                y_errs = config_data['test_accuracy_std'].fillna(0).values * 100.0
                has_std = np.any(y_errs > 0)
                
                color = color_map.get(config_id, 'gray')
                
                plot_config_line(ax, x_vals, y_vals, y_errs, config_id, color, has_std)
            
            # Configure subplot
            setup_subplot_appearance(ax, row_idx, col_idx, n_rows, n_cols)
            
            # Column titles on top row
            if row_idx == 0:
                col_label = format_family_param_value(col_param, col_val)
                ax.set_title(col_label, fontsize=15, pad=15)
            
            # Row titles on the right (rightmost column only)
            if row_param and col_idx == n_cols - 1:
                row_label = format_family_param_value(row_param, row_val)
                # Add text on the right side of the subplot
                ax.text(1.15, 0.5, row_label, 
                       transform=ax.transAxes,
                       fontsize=15, fontweight='bold',
                       rotation=-90, va='center', ha='left')
            
            # Set y-limits and add performance bands
            y_min, y_max = ax.get_ylim()
            y_min = 0
            y_max = max(100, y_max)
            ax.set_ylim(y_min, y_max)
            add_performance_bands(ax, y_min, y_max)
            
            # Setup log scale if needed
            x_vals_all = subplot_data['n_train'].values
            setup_log_scale_if_needed(ax, x_vals_all)
    
    # Overall title at the very top
    fig.suptitle(f"Transductive: {target_mode}",
                 fontweight='bold', fontsize=20, y=0.995)  # Moved up from 0.98
    
    # Create legend at top with 4 columns
    create_legend(fig, all_configs, color_map, location='top')
    
    # Save figure
    if project_name:
        output_dir = Path(project_name)
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / Path(output_path).name)
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"  ✓ Plot saved to {output_path}")
    plt.close()


def create_hyperparam_comparison_plot(
    df: pd.DataFrame,
    data_name: str,
    output_path: str,
    project_name: Optional[str] = None,
):
    """
    Create plot comparing all hyperparameter variants for a single pretrained model.
    Shows both Linear Probe and Complete Model Training modes.
    No Random Init shown - just to see which hyperparams work best.
    """
    setup_plot_style()
    
    # Filter to this pretrained model only (no Random Init)
    df_pretrained = df[
        (df['data_name'] == data_name) & 
        (~df['mode'].isin(RANDOM_INIT_MODES))
    ].copy()
    
    if len(df_pretrained) == 0:
        print(f"  No pretrained data found for {data_name}")
        return
    
    print(f"\n[Hyperparam Comparison - {data_name}]")
    print(f"  Total rows: {len(df_pretrained)}")
    
    # Identify varying family parameters
    varying_family_params = identify_varying_family_params(df_pretrained)
    print(f"  Varying family parameters: {[p.replace('family_', '') for p in varying_family_params]}")
    
    if len(varying_family_params) == 0:
        print(f"  ERROR: No varying family parameters found")
        return
    
    # Aggregate WITHOUT selecting best hyperparams (keep all variants)
    df_agg = aggregate_runs(df_pretrained, varying_family_params)
    
    # Create subplot layout
    row_param, col_param, row_values, col_values = create_subplot_grid(varying_family_params, df_agg)
    n_rows, n_cols = len(row_values), len(col_values)
    
    print(f"  Creating plot with {n_rows} rows × {n_cols} columns")
    
    # Get unique configurations and create color map
    all_configs = sorted(df_agg['config_id'].unique())
    
    # Use mode_separated=False to get different colors for LP vs CMT
    color_map = create_color_map_by_model_type(all_configs, df_agg, mode_separated=False)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.28, wspace=0.20, left=0.08, right=0.92, top=0.86, bottom=0.10)
    
    # Plot each subplot
    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this subplot
            if row_param:
                subplot_data = df_agg[
                    (df_agg[row_param] == row_val) & (df_agg[col_param] == col_val)
                ]
            else:
                subplot_data = df_agg[df_agg[col_param] == col_val]
            
            if len(subplot_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                continue
            
            # Plot each configuration
            for config_id in all_configs:
                config_data = subplot_data[subplot_data['config_id'] == config_id]
                if len(config_data) == 0:
                    continue
                
                config_data = config_data.sort_values('n_train')
                
                x_vals = config_data['n_train'].values
                y_vals = config_data['test_accuracy_mean'].values * 100.0
                y_errs = config_data['test_accuracy_std'].fillna(0).values * 100.0
                has_std = np.any(y_errs > 0)
                
                color = color_map.get(config_id, 'gray')
                
                plot_config_line(ax, x_vals, y_vals, y_errs, config_id, color, has_std)
            
            # Configure subplot
            setup_subplot_appearance(ax, row_idx, col_idx, n_rows, n_cols)
            
            # Column titles on top row
            if row_idx == 0:
                col_label = format_family_param_value(col_param, col_val)
                ax.set_title(col_label, fontsize=15, pad=15)
            
            # Row titles on the right (rightmost column only)
            if row_param and col_idx == n_cols - 1:
                row_label = format_family_param_value(row_param, row_val)
                ax.text(1.15, 0.5, row_label, 
                       transform=ax.transAxes,
                       fontsize=15, fontweight='bold',
                       rotation=-90, va='center', ha='left')
            
            # Set y-limits and add performance bands
            y_min, y_max = ax.get_ylim()
            y_min = 0
            y_max = max(100, y_max)
            ax.set_ylim(y_min, y_max)
            add_performance_bands(ax, y_min, y_max)
            
            # Setup log scale if needed
            x_vals_all = subplot_data['n_train'].values
            setup_log_scale_if_needed(ax, x_vals_all)
    
    # Overall title
    model_display = data_name.replace('graph_family_', '').replace('_pretrain', '').replace('_', ' ').title()
    if model_display.startswith('Graphuniverse '):
        model_display = model_display[len('Graphuniverse '):]
    
    fig.suptitle(f"Transductive: {model_display} Hyperparameters",
                 fontweight='bold', fontsize=20, y=0.995)
    
    # Create legend at top
    create_legend(fig, all_configs, color_map, location='top')
    
    # Save figure
    if project_name:
        output_dir = Path(project_name)
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / Path(output_path).name)
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"  ✓ Hyperparam comparison plot saved to {output_path}")
    plt.close()


def create_combined_mode_plot(
    df: pd.DataFrame,
    output_path: str,
    project_name: Optional[str] = None,
):
    """
    Create plot with Linear Probe and Complete Model Training side-by-side as columns.
    Only uses data from largest graph size (highest n_nodes_range value).
    Rows are organized by first varying parameter (usually homophily).
    """
    setup_plot_style()
    
    # Identify varying family parameters
    varying_family_params = identify_varying_family_params(df)
    
    print(f"\n[Combined Mode Plot - Transductive]")
    print(f"  Varying family parameters: {[p.replace('family_', '') for p in varying_family_params]}")
    
    if len(varying_family_params) == 0:
        print(f"  ERROR: No varying family parameters found")
        return
    
    # Find the parameter that represents graph size
    size_param = None
    for param in varying_family_params:
        if 'n_nodes_range' in param.lower() or 'n_graphs' in param.lower():
            size_param = param
            break
    
    if size_param is None:
        print(f"  WARNING: Could not find n_nodes_range or n_graphs parameter")
        print(f"  Available params: {varying_family_params}")
        # Use all data if we can't find size param
        df_filtered = df.copy()
    else:
        # Filter to highest value of size parameter (largest graph)
        max_size = df[size_param].max()
        df_filtered = df[df[size_param] == max_size].copy()
        print(f"  Filtered to {size_param} = {max_size}")
        print(f"  Rows after filter: {len(df_filtered)}")
        
        # Remove size param from varying params
        varying_family_params = [p for p in varying_family_params if p != size_param]
    
    if len(varying_family_params) == 0:
        print(f"  ERROR: No varying family parameters remain after filtering")
        return
    
    # Use first varying param for rows
    row_param = varying_family_params[0]
    
    # Prepare data for both modes
    lp_data = prepare_data_for_mode_separated_plot(df_filtered, varying_family_params, 'Linear Probe')
    cmt_data = prepare_data_for_mode_separated_plot(df_filtered, varying_family_params, 'Complete Model Training')
    
    if len(lp_data) == 0 and len(cmt_data) == 0:
        print(f"  No data to plot")
        return
    
    # Get row values from first varying param
    all_row_values = set()
    if len(lp_data) > 0:
        all_row_values.update(lp_data[row_param].unique())
    if len(cmt_data) > 0:
        all_row_values.update(cmt_data[row_param].unique())
    row_values = sorted(all_row_values)
    
    n_rows = len(row_values)
    n_cols = 2  # Always 2 columns: LP and CMT
    
    print(f"  Creating plot with {n_rows} rows × {n_cols} columns")
    
    # Get all configs and create color maps
    all_lp_configs = sorted(lp_data['config_id'].unique()) if len(lp_data) > 0 else []
    all_cmt_configs = sorted(cmt_data['config_id'].unique()) if len(cmt_data) > 0 else []
    
    # Create combined color map (mode_separated=True for full colors)
    all_configs = list(set(all_lp_configs + all_cmt_configs))
    combined_df = pd.concat([lp_data, cmt_data], ignore_index=True) if len(lp_data) > 0 and len(cmt_data) > 0 else (lp_data if len(lp_data) > 0 else cmt_data)
    color_map = create_color_map_by_model_type(all_configs, combined_df, mode_separated=True)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 5 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.28, wspace=0.20, left=0.08, right=0.92, top=0.86, bottom=0.10)
    
    # Plot each subplot
    for row_idx, row_val in enumerate(row_values):
        for col_idx, (mode_name, mode_data, mode_configs) in enumerate([
            ('Linear Probe', lp_data, all_lp_configs),
            ('Complete Model Training', cmt_data, all_cmt_configs)
        ]):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this subplot
            if len(mode_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                continue
                
            subplot_data = mode_data[mode_data[row_param] == row_val]
            
            if len(subplot_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                continue
            
            # Plot each configuration
            for config_id in mode_configs:
                config_data = subplot_data[subplot_data['config_id'] == config_id]
                if len(config_data) == 0:
                    continue
                
                config_data = config_data.sort_values('n_train')
                
                x_vals = config_data['n_train'].values
                y_vals = config_data['test_accuracy_mean'].values * 100.0
                y_errs = config_data['test_accuracy_std'].fillna(0).values * 100.0
                has_std = np.any(y_errs > 0)
                
                color = color_map.get(config_id, 'gray')
                
                plot_config_line(ax, x_vals, y_vals, y_errs, config_id, color, has_std)
            
            # Configure subplot
            setup_subplot_appearance(ax, row_idx, col_idx, n_rows, n_cols)
            
            # Column titles on top row
            if row_idx == 0:
                ax.set_title(mode_name, fontsize=16, fontweight='bold', pad=15)
            
            # Row titles on the right (rightmost column only)
            if col_idx == n_cols - 1:
                row_label = format_family_param_value(row_param, row_val)
                ax.text(1.15, 0.5, row_label, 
                       transform=ax.transAxes,
                       fontsize=15, fontweight='bold',
                       rotation=-90, va='center', ha='left')
            
            # Set y-limits and add performance bands
            y_min, y_max = ax.get_ylim()
            y_min = 0
            y_max = max(100, y_max)
            ax.set_ylim(y_min, y_max)
            add_performance_bands(ax, y_min, y_max)
            
            # Setup log scale if needed
            x_vals_all = subplot_data['n_train'].values
            setup_log_scale_if_needed(ax, x_vals_all)
    
    # Overall title
    fig.suptitle(f"Transductive: Mode Comparison (Largest Graph)",
                 fontweight='bold', fontsize=20, y=0.995)
    
    # Create legend - simplified labels without mode in parentheses
    simplified_configs = []
    simplified_color_map = {}
    for config_id in all_configs:
        simplified = config_id.replace(' (Linear Probe)', '').replace(' (Complete Model Training)', '')
        if simplified not in simplified_configs:
            simplified_configs.append(simplified)
            simplified_color_map[simplified] = color_map[config_id]
    
    create_legend(fig, simplified_configs, simplified_color_map, location='top')
    
    # Save figure
    if project_name:
        output_dir = Path(project_name)
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / Path(output_path).name)
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"  ✓ Combined mode plot saved to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot transductive results organized by varying family parameters"
    )
    parser.add_argument(
        '--transductive_project',
        type=str,
        nargs='+',
        required=True,
        help='Wandb project(s) for transductive downstream results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='transductive_family_params_comparison.png',
        help='Output filename for plot'
    )
    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save processed dataframe to CSV'
    )
    
    args = parser.parse_args()
    
    # Ensure transductive_project is a list
    if isinstance(args.transductive_project, str):
        transductive_projects = [args.transductive_project]
    else:
        transductive_projects = args.transductive_project
    
    # Fetch and process runs
    print("\n" + "=" * 80)
    print("FETCHING TRANSDUCTIVE RUNS")
    print("=" * 80)
    transductive_runs = fetch_runs_from_wandb(transductive_projects)
    
    print("\n" + "=" * 80)
    print("PROCESSING RUNS")
    print("=" * 80)
    transductive_df = process_transductive_runs(transductive_runs)
    
    if len(transductive_df) == 0:
        print("ERROR: No valid transductive runs found")
        return
    
    # Create output directory
    project_name = transductive_projects[0].split('/')[-1]
    if len(transductive_projects) > 1:
        project_name = f"{project_name}_combined"
    
    output_dir = Path(project_name)
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir}")
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = output_dir / Path(args.output).with_suffix('.csv').name
        transductive_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved data to {csv_path}")
    
    # Create mode-separated plots
    print("\n" + "=" * 80)
    print("CREATING MODE-SEPARATED PLOTS")
    print("=" * 80)
    
    for target_mode in ['Linear Probe', 'Complete Model Training']:
        print(f"\n[Creating plot for: {target_mode}]")
        
        mode_filename = target_mode.lower().replace(' ', '_')
        output_path = output_dir / f"{Path(args.output).stem}_{mode_filename}{Path(args.output).suffix}"
        
        create_mode_separated_plot(
            df=transductive_df,
            target_mode=target_mode,
            output_path=str(output_path),
            project_name=project_name,
        )
    
    # Create hyperparam comparison plots (one per pretrained model type)
    print("\n" + "=" * 80)
    print("CREATING HYPERPARAMETER COMPARISON PLOTS")
    print("=" * 80)
    
    # Get unique pretrained model types (exclude Random Init)
    pretrained_df = transductive_df[~transductive_df['mode'].isin(RANDOM_INIT_MODES)]
    unique_data_names = pretrained_df['data_name'].unique()
    
    for data_name in unique_data_names:
        print(f"\n[Hyperparam Plot - Model: {data_name}]")
        
        # Clean data_name for filename
        clean_name = data_name.replace('graph_family_', '').replace('_pretrain', '').replace('graphuniverse_', '')
        output_path = output_dir / f"{Path(args.output).stem}_hyperparams_{clean_name}{Path(args.output).suffix}"
        
        create_hyperparam_comparison_plot(
            df=transductive_df,
            data_name=data_name,
            output_path=str(output_path),
            project_name=project_name,
        )
    
    # Create combined mode comparison plot (LP vs CMT side-by-side)
    print("\n" + "=" * 80)
    print("CREATING COMBINED MODE COMPARISON PLOT")
    print("=" * 80)
    
    output_path = output_dir / f"{Path(args.output).stem}_combined_modes{Path(args.output).suffix}"
    
    create_combined_mode_plot(
        df=transductive_df,
        output_path=str(output_path),
        project_name=project_name,
    )
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()