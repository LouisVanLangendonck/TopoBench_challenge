"""
Plot inductive downstream results organized by varying family parameters.

This script fetches results from an inductive downstream wandb project and creates
comparison plots where:
- Separate plot for each task type (community_detection, homophily, etc.)
- Subplots are organized by unique combinations of varying family_parameters
- If 2 params varied: one makes columns, the other makes rows
- Each subplot shows: different (mode, model_type, hyperparams) as different lines
- X-axis: n_train (number of finetuning graphs)
- Y-axis: Test accuracy/metric
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
# Data Fetching
# =============================================================================

def fetch_runs_from_wandb(project_path: str) -> List[Dict[str, Any]]:
    """Fetch all runs from a wandb project."""
    api = wandb.Api()
    
    print(f"\n{'=' * 80}")
    print(f"FETCHING RUNS FROM WANDB PROJECT: {project_path}")
    print(f"{'=' * 80}")
    
    runs = api.runs(project_path, filters={"state": "finished"})
    
    run_data = []
    
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        
        data = {
            "run_id": run.id,
            "run_name": run.name,
            "config": config,
            "summary": summary,
        }
        
        run_data.append(data)
        print(f"  ✓ {run.id} ({run.name})")
    
    print(f"\n{'=' * 80}")
    print(f"FOUND {len(run_data)} FINISHED RUNS")
    print(f"{'=' * 80}\n")
    
    return run_data


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
    
    prefix = 'pretrain/dataset/loader/parameters/generation_parameters/family_parameters/'
    for key in config.keys():
        if key.startswith(prefix):
            param_name = key[len(prefix):]
            family_params[param_name] = config[key]
    
    # Also try nested structure
    if not family_params:
        nested_value = get_nested_value(config, 'pretrain/dataset/loader/parameters/generation_parameters/family_parameters')
        if isinstance(nested_value, dict):
            family_params.update(nested_value)
    
    return family_params


def normalize_range_value(val):
    """Normalize range values to a hashable format."""
    if isinstance(val, list):
        # Convert list to tuple for hashing
        return tuple(val)
    return val


def extract_model_hyperparams(config: dict) -> dict:
    """Extract model type and varying hyperparameters."""
    model_type = config.get('pretrain/model/model_name', 'unknown')
    
    # Extract backbone_wrapper parameters and other model hyperparams
    hyperparam_cols = [
        'pretrain/model/backbone_wrapper/mask_rate',
        'pretrain/model/backbone_wrapper/drop_edge_rate',
        'pretrain/model/backbone_wrapper/replace_rate',
        'pretrain/model/backbone_wrapper/edge_sample_ratio',
        'pretrain/model/backbone_wrapper/corruption_type',
        'pretrain/model/backbone_wrapper/drop_scheme',
        'pretrain/model/backbone_wrapper/add_edge',
        'pretrain/optimizer/parameters/lr',
        'pretrain/optimizer/parameters/weight_decay',
    ]
    
    hyperparams = {}
    for col in hyperparam_cols:
        val = get_nested_value(config, col)
        if val is not None:
            simple_name = col.split('/')[-1]
            hyperparams[simple_name] = val
    
    return {
        'model_type': model_type,
        'hyperparams': hyperparams,
    }


def extract_task_from_summary(summary: dict) -> List[Tuple[str, str, str]]:
    """
    Extract all task types and their metrics from summary.
    
    Returns:
        List of (task_name, metric_key, metric_type) tuples
        e.g., [('community_detection', 'test/accuracy_community_detection', 'accuracy')]
    """
    tasks = []
    
    # Common task patterns in inductive setting
    # Priority order: improvement metrics > accuracy > mae
    task_patterns = [
        # Accuracy metrics
        ('community_detection', 'test/accuracy_community_detection', 'accuracy'),
        ('homophily_accuracy', 'test/accuracy_homophily', 'accuracy'),
        
        # Improvement metrics (preferred)
        ('homophily_improvement', 'test/improvement_homophily', 'improvement'),
        ('community_presence_improvement', 'test/improvement_community_presence', 'improvement'),
        
        # MAE metrics for properties
        ('homophily_mae', 'test/mae_homophily', 'mae'),
        ('community_presence_mae', 'test/mae_community_presence', 'mae'),
        ('diameter', 'test/mae_diameter', 'mae'),
        ('clustering_coefficient', 'test/mae_clustering_coefficient', 'mae'),
        ('degree_assortativity', 'test/mae_degree_assortativity', 'mae'),
        ('modularity', 'test/mae_modularity', 'mae'),
        ('transitivity', 'test/mae_transitivity', 'mae'),
        ('n_nodes', 'test/mae_n_nodes', 'mae'),
        ('n_edges', 'test/mae_n_edges', 'mae'),
    ]
    
    for task_name, metric_key, metric_type in task_patterns:
        if metric_key in summary:
            tasks.append((task_name, metric_key, metric_type))
    
    return tasks


def process_inductive_runs(run_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process inductive downstream evaluation runs."""
    records = []
    
    for run in run_data:
        config = run['config']
        summary = run['summary']
        
        # Extract all tasks present in this run
        tasks = extract_task_from_summary(summary)
        
        if not tasks:
            continue
        
        # Extract family parameters from pretraining
        family_params = extract_family_parameters(config)
        
        # Extract model metadata
        model_meta = extract_model_hyperparams(config)
        
        # Create one record per task
        for task_name, metric_key, metric_type in tasks:
            record = {
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'model_type': model_meta['model_type'],
                'n_train': config.get('n_train'),
                'mode': config.get('mode'),
                'task': task_name,
                'metric_key': metric_key,
                'metric_type': metric_type,
                'test_metric': summary.get(metric_key),
            }
            
            # Add family parameters with normalized values
            for param_name, param_value in family_params.items():
                record[f'family_{param_name}'] = normalize_range_value(param_value)
            
            # Add hyperparameters
            record.update(model_meta['hyperparams'])
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} inductive run-task combinations:")
        print(f"  Model types: {df['model_type'].unique().tolist()}")
        print(f"  Modes: {df['mode'].unique().tolist()}")
        print(f"  Tasks: {df['task'].unique().tolist()}")
        print(f"  N_train values: {sorted(df['n_train'].unique().tolist())}")
        
        # Print family parameters
        family_cols = [col for col in df.columns if col.startswith('family_')]
        print(f"  Family parameters found: {[col.replace('family_', '') for col in family_cols]}")
        for col in family_cols:
            unique_vals = df[col].dropna().unique()
            print(f"    {col}: {len(unique_vals)} unique values")
    
    return df


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
    
    metadata_cols = {
        'run_id', 'run_name', 'model_type', 'n_train', 'mode', 'task',
        'metric_key', 'metric_type', 'test_metric', 'hyperparam_id'
    }
    
    # Also exclude family parameter columns
    family_cols = [col for col in model_df.columns if col.startswith('family_')]
    metadata_cols.update(family_cols)
    
    potential_hyperparam_cols = [col for col in model_df.columns if col not in metadata_cols]
    
    varying = []
    for col in potential_hyperparam_cols:
        try:
            non_na_values = model_df[col].dropna()
            if len(non_na_values) > 0 and non_na_values.nunique() > 1:
                varying.append(col)
        except:
            pass
    
    return sorted(varying)


def create_hyperparam_id(row, varying_hyperparams: List[str]) -> str:
    """Create a unique ID for a hyperparameter configuration."""
    parts = []
    
    for hp in sorted(varying_hyperparams):
        val = row.get(hp)
        if pd.notna(val):
            display_name = hp.replace('drop_', '').replace('_rate', '').replace('_', '')
            
            if isinstance(val, (int, float)):
                if isinstance(val, float) and val != int(val):
                    parts.append(f"{display_name}={val:.2f}")
                else:
                    parts.append(f"{display_name}={int(val)}")
            else:
                parts.append(f"{display_name}={val}")
    
    return ", ".join(parts) if parts else "default"


def create_config_id(row, varying_hyperparams: List[str], model_type: str, mode: str) -> str:
    """Create a unique ID for (model_type, mode, hyperparams) configuration."""
    model_display = model_type.replace('gps_', '').upper()
    
    # Mode display
    if mode == 'scratch':
        return 'From Scratch (unfrozen)'
    elif mode == 'scratch_frozen':
        return 'From Scratch (frozen)'
    
    # For pretrained models, add hyperparams
    hyperparam_str = create_hyperparam_id(row, varying_hyperparams)
    
    mode_display = mode.replace('finetune-linear', 'Pretrained (unfrozen)').replace('linear', 'Pretrained (frozen)')
    
    if hyperparam_str and hyperparam_str != 'default':
        return f"{model_display} {mode_display} ({hyperparam_str})"
    else:
        return f"{model_display} {mode_display}"


def aggregate_runs(df: pd.DataFrame, varying_family_params: List[str]) -> pd.DataFrame:
    """Aggregate runs with same configuration (mean + std)."""
    # Identify variable hyperparams per model_type
    variable_hyperparams = {}
    for model_type in df['model_type'].unique():
        variable_hyperparams[model_type] = identify_variable_hyperparams(df, model_type)
    
    # Add hyperparam_id and config_id
    df = df.copy()
    for model_type in df['model_type'].unique():
        varying = variable_hyperparams[model_type]
        mask = df['model_type'] == model_type
        df.loc[mask, 'hyperparam_id'] = df[mask].apply(
            lambda row: create_hyperparam_id(row, varying), axis=1
        )
        df.loc[mask, 'config_id'] = df[mask].apply(
            lambda row: create_config_id(row, varying, model_type, row['mode']), axis=1
        )
    
    # Group by: task, family params, model_type, mode, n_train, hyperparam_id
    group_cols = ['task'] + varying_family_params + ['model_type', 'mode', 'n_train', 'hyperparam_id', 'config_id']
    
    # Add all varying hyperparameter columns
    metadata_cols = {
        'run_id', 'run_name', 'model_type', 'n_train', 'mode', 'task',
        'metric_key', 'metric_type', 'test_metric', 'hyperparam_id', 'config_id'
    }
    family_cols = [col for col in df.columns if col.startswith('family_')]
    metadata_cols.update(family_cols)
    
    potential_hyperparam_cols = [col for col in df.columns if col not in metadata_cols]
    for col in potential_hyperparam_cols:
        if col in df.columns and df[col].dropna().nunique() > 1:
            if col not in group_cols:
                group_cols.append(col)
    
    agg_dict = {
        'test_metric': ['mean', 'std', 'count'],
        'metric_type': 'first',
        'run_id': lambda x: list(x),
    }
    
    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    grouped.columns = [
        '_'.join(col).strip('_') if col[1] else col[0] 
        for col in grouped.columns.values
    ]
    
    # Rename for clarity
    grouped.rename(columns={
        'test_metric_mean': 'test_metric',
        'test_metric_std': 'test_metric_std',
        'test_metric_count': 'n_runs',
    }, inplace=True)
    
    # For single runs, set std to 0
    grouped['test_metric_std'] = grouped['test_metric_std'].fillna(0.0)
    
    print(f"\n  Aggregated to {len(grouped)} unique configurations")
    
    return grouped


def create_color_map_grouped_by_mode(all_configs: List[str]) -> Dict[str, np.ndarray]:
    """
    Create a color map where configs with the same base mode get similar colors.
    
    Configs with the same (model_type, mode) but different hyperparams get 
    lighter/darker variants of the same base color.
    
    Frozen and unfrozen modes get DIFFERENT color families.
    """
    # Separate scratch from pretrained
    scratch_configs = [c for c in all_configs if 'From Scratch' in c]
    pretrained_configs = [c for c in all_configs if 'From Scratch' not in c]
    
    color_map = {}
    
    # Handle scratch configs (fixed colors)
    for scratch_config in scratch_configs:
        if '(frozen)' in scratch_config.lower():
            color_map[scratch_config] = np.array([0.5, 0.5, 0.5, 1])  # Gray
        else:
            color_map[scratch_config] = np.array([0, 0, 0, 1])  # Black
    
    # Group pretrained configs by base mode (model_type + mode INCLUDING frozen/unfrozen)
    # BUT ignoring the specific hyperparams in the second set of parentheses
    mode_groups = defaultdict(list)
    for config in pretrained_configs:
        # Extract base mode: model + mode (including frozen/unfrozen)
        # E.g., "DGI Pretrained (frozen) (corruption=A)" -> "DGI Pretrained (frozen)"
        # E.g., "DGI Pretrained (unfrozen) (corruption=A)" -> "DGI Pretrained (unfrozen)"
        
        # Find all parentheses
        parts = config.split('(')
        if len(parts) >= 2:
            # Take model name + first parenthesis (which is frozen/unfrozen)
            base_mode = parts[0] + '(' + parts[1].split(')')[0] + ')'
        else:
            base_mode = config
        
        mode_groups[base_mode].append(config)
    
    # Assign colors: each base mode gets a distinct color, variants get shades
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if len(mode_groups) > 10:
        base_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(mode_groups))))
    
    for i, (base_mode, configs) in enumerate(sorted(mode_groups.items())):
        base_color = base_colors[i % len(base_colors)]
        
        if len(configs) == 1:
            # Only one config for this mode, use base color
            color_map[configs[0]] = base_color
        else:
            # Multiple configs: create lighter/darker variants
            # Use a range from 0.6 to 1.4 for color intensity variations
            intensities = np.linspace(0.65, 1.35, len(configs))
            
            for j, config in enumerate(sorted(configs)):
                # Adjust RGB channels but keep saturation
                adjusted_color = base_color[:3] * intensities[j]
                # Clip to valid range
                adjusted_color = np.clip(adjusted_color, 0, 1)
                color_map[config] = np.append(adjusted_color, 1.0)  # Add alpha
    
    return color_map


# =============================================================================
# Plotting
# =============================================================================

def create_family_param_comparison_plot(
    df: pd.DataFrame,
    task: str,
    metric_type: str,
    output_path: str = "inductive_family_params_comparison.png",
    project_name: str = None,
):
    """Create comparison plot organized by varying family parameters for a single task."""
    
    # Publication quality settings
    mpl.rcParams.update({
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
    })
    
    # Identify varying family parameters
    varying_family_params = identify_varying_family_params(df)
    
    print(f"\nVarying family parameters: {[p.replace('family_', '') for p in varying_family_params]}")
    
    if len(varying_family_params) == 0:
        print("ERROR: No varying family parameters found")
        return
    
    # Aggregate runs
    df_agg = aggregate_runs(df, varying_family_params)
    
    # Determine subplot layout
    if len(varying_family_params) == 1:
        # Single varying param: use as columns
        row_param = None
        col_param = varying_family_params[0]
        row_values = [None]
        col_values = sorted(df_agg[col_param].unique())
    elif len(varying_family_params) == 2:
        # Two varying params: one for rows, one for columns
        row_param = varying_family_params[0]
        col_param = varying_family_params[1]
        row_values = sorted(df_agg[row_param].unique())
        col_values = sorted(df_agg[col_param].unique())
    else:
        # More than 2: use first two, warn user
        print(f"WARNING: Found {len(varying_family_params)} varying family parameters. Using first 2 for plot layout.")
        row_param = varying_family_params[0]
        col_param = varying_family_params[1]
        row_values = sorted(df_agg[row_param].unique())
        col_values = sorted(df_agg[col_param].unique())
    
    n_rows = len(row_values)
    n_cols = len(col_values)
    
    print(f"\nCreating plot with {n_rows} rows × {n_cols} columns")
    if row_param:
        print(f"  Rows: {row_param.replace('family_', '')} = {row_values}")
    print(f"  Columns: {col_param.replace('family_', '')} = {col_values}")
    
    # Get unique configurations (for coloring)
    all_configs = sorted(df_agg['config_id'].unique())
    
    # Create intelligent color map grouped by mode
    color_map = create_color_map_grouped_by_mode(all_configs)
    
    print(f"\nColor map created for {len(color_map)} configurations:")
    for config, color in sorted(color_map.items()):
        print(f"  {config}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.28, wspace=0.20, left=0.08, right=0.78, top=0.92, bottom=0.10)
    
    # Determine y-axis label based on metric type
    if metric_type == 'accuracy':
        y_label = 'Test Accuracy (%)'
        metric_multiplier = 100
    elif metric_type == 'mae':
        y_label = 'Test MAE'
        metric_multiplier = 1
    elif metric_type == 'improvement':
        y_label = 'Test Improvement (%)'
        metric_multiplier = 100
    else:
        y_label = 'Test Metric'
        metric_multiplier = 1
    
    # Plot each subplot
    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this subplot
            if row_param:
                subplot_data = df_agg[(df_agg[row_param] == row_val) & (df_agg[col_param] == col_val)]
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
                y_vals = config_data['test_metric'].values * metric_multiplier
                
                # Only show error bars if we have std > 0
                has_std = config_data['test_metric_std'].fillna(0).values > 0
                y_errs = config_data['test_metric_std'].fillna(0).values * metric_multiplier
                
                color = color_map.get(config_id, 'gray')
                
                # Determine marker and linestyle
                is_scratch = 'From Scratch' in config_id
                marker = 's' if is_scratch else 'o'
                linestyle = '--' if is_scratch else '-'
                
                if np.any(has_std):
                    ax.errorbar(
                        x_vals, y_vals, yerr=y_errs,
                        color=color, alpha=0.85,
                        linewidth=2.5, marker=marker, markersize=7,
                        linestyle=linestyle,
                        capsize=5, capthick=2.0,
                        elinewidth=2.0,
                        label=config_id
                    )
                else:
                    ax.plot(x_vals, y_vals, 
                           color=color, alpha=0.85,
                           linewidth=2.5, marker=marker, markersize=7,
                           linestyle=linestyle,
                           label=config_id)
            
            # Labels
            if row_idx == n_rows - 1:
                ax.set_xlabel('# Finetuning Graphs', fontsize=13)
            
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=13)
            
            # Title
            if row_idx == 0:
                col_label = format_family_param_value(col_param, col_val)
                ax.text(0.5, 1.08, col_label,
                       transform=ax.transAxes, ha='center', va='bottom',
                       fontweight='bold', fontsize=14)
            
            if col_idx == n_cols - 1 and row_param:
                row_label = format_family_param_value(row_param, row_val)
                ax.text(1.15, 0.5, row_label,
                       transform=ax.transAxes, ha='left', va='center',
                       rotation=-90, fontweight='bold', fontsize=14)
            
            # Y-axis styling
            y_vals_all = subplot_data['test_metric'].values * metric_multiplier
            if len(y_vals_all) > 0:
                if metric_type == 'accuracy':
                    y_min = max(0, np.min(y_vals_all) - 10)
                    y_max = min(105, np.max(y_vals_all) + 10)
                elif metric_type == 'improvement':
                    # Improvement can be negative or positive
                    y_range = np.max(y_vals_all) - np.min(y_vals_all)
                    y_min = np.min(y_vals_all) - y_range * 0.1
                    y_max = np.max(y_vals_all) + y_range * 0.1
                    # Add zero line if range crosses zero
                    if y_min < 0 < y_max:
                        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
                else:  # mae
                    y_range = np.max(y_vals_all) - np.min(y_vals_all)
                    y_min = max(0, np.min(y_vals_all) - y_range * 0.1)
                    y_max = np.max(y_vals_all) + y_range * 0.1
            else:
                y_min = 0
                y_max = 100 if metric_type in ['accuracy', 'improvement'] else 1
            
            ax.set_ylim(y_min, y_max)
            
            # Color bands (only for accuracy)
            if metric_type == 'accuracy':
                if y_max > 75:
                    ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(50, 75, facecolor='lightyellow', alpha=0.08, zorder=0)
                    ax.axhspan(75, y_max, facecolor='palegreen', alpha=0.08, zorder=0)
                elif y_max > 50:
                    ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(50, y_max, facecolor='lightyellow', alpha=0.08, zorder=0)
                else:
                    ax.axhspan(max(y_min, 0), y_max, facecolor='lightcoral', alpha=0.08, zorder=0)
            elif metric_type == 'improvement':
                # Color bands for improvement: negative = bad, positive = good
                if y_min < 0 < y_max:
                    ax.axhspan(y_min, 0, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(0, y_max, facecolor='palegreen', alpha=0.08, zorder=0)
                elif y_max <= 0:
                    ax.axhspan(y_min, y_max, facecolor='lightcoral', alpha=0.08, zorder=0)
                else:
                    ax.axhspan(y_min, y_max, facecolor='palegreen', alpha=0.08, zorder=0)
            
            ax.grid(axis='both', alpha=0.20, zorder=0, linestyle=':', linewidth=1.0)
            
            # X-axis: log scale if needed
            x_vals_all = subplot_data['n_train'].values
            if len(x_vals_all) > 0 and np.max(x_vals_all) / np.min(x_vals_all) > 10:
                ax.set_xscale('log')
                ax.set_xticks(sorted(x_vals_all))
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    
    # Overall title
    task_display = task.replace('_', ' ').title()
    fig.suptitle(f"Inductive {task_display}: Family Parameter Comparison", 
                 fontweight='bold', fontsize=20, y=0.98)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_handles = []
    
    # Group configs by base mode for organized legend
    pretrained_configs = [c for c in all_configs if 'From Scratch' not in c]
    scratch_configs = [c for c in all_configs if 'From Scratch' in c]
    
    # Group pretrained by base mode (including frozen/unfrozen distinction)
    mode_groups = defaultdict(list)
    for config in pretrained_configs:
        # Extract base mode: model + mode (including frozen/unfrozen)
        parts = config.split('(')
        if len(parts) >= 2:
            base_mode = parts[0] + '(' + parts[1].split(')')[0] + ')'
        else:
            base_mode = config
        mode_groups[base_mode].append(config)
    
    # Add pretrained configs grouped by mode
    for base_mode in sorted(mode_groups.keys()):
        configs = sorted(mode_groups[base_mode])
        for config in configs:
            color = color_map[config]
            handle = Line2D([0], [0], color=color, linestyle='-', 
                           linewidth=2.5, marker='o', markersize=6, label=config)
            legend_handles.append(handle)
    
    # Add scratch configs
    for config in sorted(scratch_configs):
        color = color_map[config]
        handle = Line2D([0], [0], color=color, linestyle='--', 
                       linewidth=2.5, marker='s', markersize=6, label=config)
        legend_handles.append(handle)
    
    # Add legend outside the plot
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
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()


def format_family_param_value(param_name: str, param_value) -> str:
    """Format family parameter value for display."""
    param_name_clean = param_name.replace('family_', '').replace('_', ' ').title()
    
    if isinstance(param_value, tuple):
        # Format range tuples nicely
        if len(param_value) == 2 and param_value[0] == param_value[1]:
            # Single value range like [100, 100]
            return f"{param_name_clean} = {param_value[0]}"
        else:
            return f"{param_name_clean} = {list(param_value)}"
    else:
        return f"{param_name_clean} = {param_value}"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot inductive results organized by varying family parameters"
    )
    parser.add_argument(
        '--inductive_project',
        type=str,
        required=True,
        help='Wandb project for inductive downstream results'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='inductive_family_params_comparison',
        help='Output filename prefix for plots (task name will be appended)'
    )
    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save processed dataframe to CSV'
    )
    
    args = parser.parse_args()
    
    # Fetch runs
    print("\n" + "=" * 80)
    print("FETCHING INDUCTIVE RUNS")
    print("=" * 80)
    inductive_runs = fetch_runs_from_wandb(args.inductive_project)
    
    # Process runs
    print("\n" + "=" * 80)
    print("PROCESSING RUNS")
    print("=" * 80)
    inductive_df = process_inductive_runs(inductive_runs)
    
    if len(inductive_df) == 0:
        print("ERROR: No valid inductive runs found")
        return
    
    # Get unique tasks
    tasks = inductive_df['task'].unique()
    print(f"\nFound {len(tasks)} unique tasks: {tasks.tolist()}")
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = args.output_prefix + '_all_tasks.csv'
        inductive_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved all data to {csv_path}")
    
    # Extract project name from path (remove entity if present)
    project_name = args.inductive_project.split('/')[-1]
    
    # Create plot for each task
    for task in tasks:
        print("\n" + "=" * 80)
        print(f"CREATING PLOT FOR TASK: {task}")
        print("=" * 80)
        
        task_df = inductive_df[inductive_df['task'] == task].copy()
        metric_type = task_df['metric_type'].iloc[0]
        
        output_path = f"{args.output_prefix}_{task}_{project_name}.png"
        
        create_family_param_comparison_plot(
            df=task_df,
            task=task,
            metric_type=metric_type,
            output_path=output_path,
            project_name=project_name,
        )
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
