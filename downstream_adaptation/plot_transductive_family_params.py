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
    
    # Extract backbone_wrapper parameters
    hyperparam_cols = [
        'pretrain/model/backbone_wrapper/mask_rate',
        'pretrain/model/backbone_wrapper/drop_edge_rate',
        'pretrain/model/backbone_wrapper/replace_rate',
        'pretrain/model/backbone_wrapper/edge_sample_ratio',
        'pretrain/optimizer/parameters/lr',
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


def process_transductive_runs(run_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process transductive downstream evaluation runs."""
    records = []
    
    for run in run_data:
        config = run['config']
        summary = run['summary']
        
        # Check for accuracy metric
        if 'test/accuracy' not in summary:
            continue
        
        # Extract family parameters from pretraining
        family_params = extract_family_parameters(config)
        
        # Extract model metadata
        model_meta = extract_model_hyperparams(config)
        
        record = {
            'run_id': run['run_id'],
            'run_name': run['run_name'],
            'model_type': model_meta['model_type'],
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
        'run_id', 'run_name', 'model_type', 'n_train', 'mode', 'test_accuracy',
        'train_accuracy', 'val_accuracy', 'hyperparam_id'
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
    
    # Group by: family params, model_type, mode, n_train, hyperparam_id
    group_cols = varying_family_params + ['model_type', 'mode', 'n_train', 'hyperparam_id', 'config_id']
    
    # Add all varying hyperparameter columns
    metadata_cols = {
        'run_id', 'run_name', 'model_type', 'n_train', 'mode', 'test_accuracy',
        'train_accuracy', 'val_accuracy', 'hyperparam_id', 'config_id'
    }
    family_cols = [col for col in df.columns if col.startswith('family_')]
    metadata_cols.update(family_cols)
    
    potential_hyperparam_cols = [col for col in df.columns if col not in metadata_cols]
    for col in potential_hyperparam_cols:
        if col in df.columns and df[col].dropna().nunique() > 1:
            if col not in group_cols:
                group_cols.append(col)
    
    agg_dict = {
        'test_accuracy': ['mean', 'std', 'count'],
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
        'test_accuracy_mean': 'test_accuracy',
        'test_accuracy_std': 'test_accuracy_std',
        'test_accuracy_count': 'n_runs',
    }, inplace=True)
    
    # For single runs, set std to 0
    grouped['test_accuracy_std'] = grouped['test_accuracy_std'].fillna(0.0)
    
    print(f"\n  Aggregated to {len(grouped)} unique configurations")
    
    return grouped


# =============================================================================
# Plotting
# =============================================================================

def create_family_param_comparison_plot(
    df: pd.DataFrame,
    output_path: str = "transductive_family_params_comparison.png",
    project_name: str = None,
):
    """Create comparison plot organized by varying family parameters."""
    
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
    
    # Separate scratch from pretrained for coloring
    scratch_configs = [c for c in all_configs if 'From Scratch' in c]
    pretrained_configs = [c for c in all_configs if 'From Scratch' not in c]
    
    # Create color map
    n_pretrained = len(pretrained_configs)
    if n_pretrained <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_pretrained]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_pretrained)))
    
    color_map = {config: colors[i] for i, config in enumerate(pretrained_configs)}
    
    # Add different colors for scratch configurations
    for i, scratch_config in enumerate(scratch_configs):
        if '(frozen)' in scratch_config.lower():
            color_map[scratch_config] = np.array([0.5, 0.5, 0.5, 1])  # Gray
        else:
            color_map[scratch_config] = np.array([0, 0, 0, 1])  # Black
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.28, wspace=0.20, left=0.08, right=0.78, top=0.92, bottom=0.10)
    
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
                y_vals = config_data['test_accuracy'].values * 100
                
                # Only show error bars if we have std > 0
                has_std = config_data['test_accuracy_std'].fillna(0).values > 0
                y_errs = config_data['test_accuracy_std'].fillna(0).values * 100
                
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
                ax.set_xlabel('# Finetuning Nodes', fontsize=13)
            
            if col_idx == 0:
                ax.set_ylabel('Test Accuracy (%)', fontsize=13)
            
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
            y_vals_all = subplot_data['test_accuracy'].values * 100
            if len(y_vals_all) > 0:
                y_min = max(0, np.min(y_vals_all) - 10)
                y_max = min(105, np.max(y_vals_all) + 10)
            else:
                y_min = 0
                y_max = 100
            
            ax.set_ylim(y_min, y_max)
            
            # Color bands
            if y_max > 75:
                ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                ax.axhspan(50, 75, facecolor='lightyellow', alpha=0.08, zorder=0)
                ax.axhspan(75, y_max, facecolor='palegreen', alpha=0.08, zorder=0)
            elif y_max > 50:
                ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                ax.axhspan(50, y_max, facecolor='lightyellow', alpha=0.08, zorder=0)
            else:
                ax.axhspan(max(y_min, 0), y_max, facecolor='lightcoral', alpha=0.08, zorder=0)
            
            ax.grid(axis='both', alpha=0.20, zorder=0, linestyle=':', linewidth=1.0)
            
            # X-axis: log scale if needed
            x_vals_all = subplot_data['n_train'].values
            if len(x_vals_all) > 0 and np.max(x_vals_all) / np.min(x_vals_all) > 10:
                ax.set_xscale('log')
                ax.set_xticks(sorted(x_vals_all))
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    
    # Overall title
    fig.suptitle("Transductive Community Detection: Family Parameter Comparison", 
                 fontweight='bold', fontsize=20, y=0.98)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_handles = []
    
    # Add pretrained configs
    for config in pretrained_configs:
        color = color_map[config]
        handle = Line2D([0], [0], color=color, linestyle='-', 
                       linewidth=2.5, marker='o', markersize=6, label=config)
        legend_handles.append(handle)
    
    # Add scratch configs
    for config in scratch_configs:
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
    
    # Add project name to output path if provided
    if project_name:
        path_parts = output_path.rsplit('.', 1)
        if len(path_parts) == 2:
            output_path = f"{path_parts[0]}_{project_name}.{path_parts[1]}"
        else:
            output_path = f"{output_path}_{project_name}"
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()


def format_family_param_value(param_name: str, param_value) -> str:
    """Format family parameter value for display."""
    param_name_clean = param_name.replace('family_', '').replace('_', ' ').title()
    
    # Special case: rename n_nodes_range to Graph Size for transductive
    if 'n_nodes_range' in param_name.lower():
        param_name_clean = 'Graph Size'
    
    if isinstance(param_value, tuple):
        # Format range tuples nicely
        if len(param_value) == 2 and param_value[0] == param_value[1]:
            # Single value range like [1000, 1000]
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
        description="Plot transductive results organized by varying family parameters"
    )
    parser.add_argument(
        '--transductive_project',
        type=str,
        required=True,
        help='Wandb project for transductive downstream results'
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
    
    # Fetch runs
    print("\n" + "=" * 80)
    print("FETCHING TRANSDUCTIVE RUNS")
    print("=" * 80)
    transductive_runs = fetch_runs_from_wandb(args.transductive_project)
    
    # Process runs
    print("\n" + "=" * 80)
    print("PROCESSING RUNS")
    print("=" * 80)
    transductive_df = process_transductive_runs(transductive_runs)
    
    if len(transductive_df) == 0:
        print("ERROR: No valid transductive runs found")
        return
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = args.output.replace('.png', '.csv')
        transductive_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved data to {csv_path}")
    
    # Create plot
    print("\n" + "=" * 80)
    print("CREATING FAMILY PARAMETER COMPARISON PLOT")
    print("=" * 80)
    
    # Extract project name from path (remove entity if present)
    project_name = args.transductive_project.split('/')[-1]
    
    create_family_param_comparison_plot(
        df=transductive_df,
        output_path=args.output,
        project_name=project_name,
    )
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
