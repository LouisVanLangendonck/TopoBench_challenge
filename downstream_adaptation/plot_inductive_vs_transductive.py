"""
Plot comparison of inductive vs transductive community detection results.

This script fetches results from two wandb projects (inductive and transductive downstream)
and creates comparison plots showing:
- Row 1: Inductive results (community detection accuracy)
- Row 2: Transductive results (community detection accuracy)
- Each column: Different pretraining scale (n_graphs for inductive, graph_size for transductive)
- x-axis: n_train (graphs for inductive, nodes for transductive)
- y-axis: Test accuracy
- Lines: Different model types/hyperparams + from-scratch baseline (black)
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


def extract_pretraining_metadata(config: dict, setting: str) -> dict:
    """Extract pretraining metadata (n_graphs for inductive, graph_size for transductive)."""
    metadata = {}
    
    if setting == "inductive":
        # Get n_graphs from pretraining config
        n_graphs = get_nested_value(config, 'pretrain/dataset/loader/parameters/generation_parameters/family_parameters/n_graphs')
        metadata['pretraining_scale'] = n_graphs
        metadata['pretraining_scale_label'] = f"{int(n_graphs)} graphs" if n_graphs else "unknown"
    
    elif setting == "transductive":
        # Get graph size from pretraining config
        n_nodes_range = get_nested_value(config, 'pretrain/dataset/loader/parameters/generation_parameters/family_parameters/n_nodes_range')
        if n_nodes_range and isinstance(n_nodes_range, list) and len(n_nodes_range) > 0:
            graph_size = n_nodes_range[0]  # Take min from range
            metadata['pretraining_scale'] = graph_size
            metadata['pretraining_scale_label'] = f"{int(graph_size)} nodes"
        else:
            metadata['pretraining_scale'] = None
            metadata['pretraining_scale_label'] = "unknown"
    
    return metadata


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
            # Simplify column name
            simple_name = col.split('/')[-1]
            hyperparams[simple_name] = val
    
    return {
        'model_type': model_type,
        'hyperparams': hyperparams,
    }


def process_inductive_runs(run_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process inductive downstream evaluation runs."""
    records = []
    
    for run in run_data:
        config = run['config']
        summary = run['summary']
        
        # Only process community_detection metric
        if 'test/accuracy_community_detection' not in summary:
            continue
        
        # Extract metadata
        pretraining_meta = extract_pretraining_metadata(config, 'inductive')
        model_meta = extract_model_hyperparams(config)
        
        record = {
            'run_id': run['run_id'],
            'run_name': run['run_name'],
            'setting': 'inductive',
            'model_type': model_meta['model_type'],
            'pretraining_scale': pretraining_meta['pretraining_scale'],
            'pretraining_scale_label': pretraining_meta['pretraining_scale_label'],
            'n_train': config.get('n_train'),
            'mode': config.get('mode'),
            'test_accuracy': summary.get('test/accuracy_community_detection'),
            'train_accuracy': summary.get('train/accuracy_community_detection'),
            'val_accuracy': summary.get('val/accuracy_community_detection'),
        }
        
        # Add hyperparameters
        record.update(model_meta['hyperparams'])
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} inductive runs:")
        print(f"  Model types: {df['model_type'].unique().tolist()}")
        print(f"  Modes: {df['mode'].unique().tolist()}")
        print(f"  N_train values: {sorted(df['n_train'].unique().tolist())}")
        print(f"  Pretraining scales: {sorted(df['pretraining_scale'].dropna().unique().tolist())}")
    
    return df


def process_transductive_runs(run_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process transductive downstream evaluation runs."""
    records = []
    
    for run in run_data:
        config = run['config']
        summary = run['summary']
        
        # Check for accuracy metric
        if 'test/accuracy' not in summary:
            continue
        
        # Extract metadata
        pretraining_meta = extract_pretraining_metadata(config, 'transductive')
        model_meta = extract_model_hyperparams(config)
        
        record = {
            'run_id': run['run_id'],
            'run_name': run['run_name'],
            'setting': 'transductive',
            'model_type': model_meta['model_type'],
            'pretraining_scale': pretraining_meta['pretraining_scale'],
            'pretraining_scale_label': pretraining_meta['pretraining_scale_label'],
            'n_train': config.get('n_train'),
            'mode': config.get('mode'),
            'test_accuracy': summary.get('test/accuracy'),
            'train_accuracy': summary.get('test/train_accuracy'),
            'val_accuracy': summary.get('test/val_accuracy'),
        }
        
        # Add hyperparameters
        record.update(model_meta['hyperparams'])
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"\nProcessed {len(df)} transductive runs:")
        print(f"  Model types: {df['model_type'].unique().tolist()}")
        print(f"  Modes: {df['mode'].unique().tolist()}")
        print(f"  N_train values: {sorted(df['n_train'].unique().tolist())}")
        print(f"  Pretraining scales: {sorted(df['pretraining_scale'].dropna().unique().tolist())}")
    
    return df


def identify_variable_hyperparams(df: pd.DataFrame, model_type: str, setting: str) -> List[str]:
    """Identify which hyperparameters vary for a given model type."""
    model_df = df[(df['model_type'] == model_type) & (df['setting'] == setting)]
    
    # Candidate hyperparam columns
    metadata_cols = {
        'run_id', 'run_name', 'setting', 'model_type', 'pretraining_scale',
        'pretraining_scale_label', 'n_train', 'mode', 'test_accuracy',
        'train_accuracy', 'val_accuracy', 'hyperparam_id'
    }
    
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


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runs with same configuration (mean + std)."""
    # Group by everything except run_id, run_name, and accuracy values
    group_cols = [
        'setting', 'model_type', 'pretraining_scale', 'pretraining_scale_label',
        'n_train', 'mode', 'hyperparam_id'
    ]
    
    # Add hyperparam columns that exist
    hyperparam_cols = [col for col in df.columns if col not in group_cols + 
                      ['run_id', 'run_name', 'test_accuracy', 'train_accuracy', 'val_accuracy']]
    group_cols.extend(hyperparam_cols)
    
    # Filter to only existing columns
    group_cols = [col for col in group_cols if col in df.columns]
    
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
    
    # For single runs, set std to 0 (not NaN)
    grouped['test_accuracy_std'] = grouped['test_accuracy_std'].fillna(0.0)
    
    # Print aggregation summary
    multi_run_configs = grouped[grouped['n_runs'] > 1]
    if len(multi_run_configs) > 0:
        print(f"\n  Found {len(multi_run_configs)} configurations with multiple runs:")
        for _, row in multi_run_configs.iterrows():
            print(f"    {row['setting']} - {row['model_type']} ({row['hyperparam_id']}) - "
                  f"n_train={row['n_train']}, mode={row['mode']}: "
                  f"{row['n_runs']} runs, std={row['test_accuracy_std']*100:.2f}%")
    
    return grouped


# =============================================================================
# Plotting
# =============================================================================

def create_comparison_plot(
    inductive_df: pd.DataFrame,
    transductive_df: pd.DataFrame,
    output_path: str = "inductive_vs_transductive_comparison.png",
):
    """Create comparison plot for inductive vs transductive community detection."""
    
    # Publication quality settings
    mpl.rcParams.update({
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "font.family": "DejaVu Serif",
        "axes.linewidth": 1.1,
        "xtick.direction": 'out',
        "ytick.direction": 'out',
        "axes.titleweight": 'bold',
        "figure.titlesize": 20
    })
    
    # Identify variable hyperparams for each setting and model
    for setting, df in [('inductive', inductive_df), ('transductive', transductive_df)]:
        for model_type in df['model_type'].unique():
            varying = identify_variable_hyperparams(df, model_type, setting)
            print(f"\n{setting.capitalize()} - {model_type}: varying hyperparams = {varying}")
            
            # Add hyperparam_id
            mask = (df['model_type'] == model_type) & (df['setting'] == setting)
            df.loc[mask, 'hyperparam_id'] = df[mask].apply(
                lambda row: create_hyperparam_id(row, varying), axis=1
            )
    
    # Aggregate runs with same config
    inductive_agg = aggregate_runs(inductive_df)
    transductive_agg = aggregate_runs(transductive_df)
    
    # Collect all unique pretrained (model_type, hyperparam_id) combinations across both settings
    # This ensures consistent coloring across all panels
    all_pretrained_configs = []
    for df in [inductive_agg, transductive_agg]:
        pretrained = df[~df['mode'].isin(['scratch', 'scratch_frozen'])]
        for _, row in pretrained.iterrows():
            config = (row['model_type'], row['hyperparam_id'])
            if config not in all_pretrained_configs:
                all_pretrained_configs.append(config)
    
    # Create global color map (used by all panels)
    n_pretrained = len(all_pretrained_configs)
    if n_pretrained <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_pretrained]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_pretrained)))
    
    global_color_map = {config: colors[i] for i, config in enumerate(all_pretrained_configs)}
    
    print(f"\nGlobal color map created with {len(global_color_map)} configurations:")
    for config, color in global_color_map.items():
        print(f"  {config}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    
    # Get unique pretraining scales for columns
    inductive_scales = sorted(inductive_agg['pretraining_scale'].dropna().unique())
    transductive_scales = sorted(transductive_agg['pretraining_scale'].dropna().unique())
    
    n_cols = max(len(inductive_scales), len(transductive_scales))
    
    print(f"\nCreating plot with {n_cols} columns (pretraining scales)")
    print(f"  Inductive scales: {inductive_scales}")
    print(f"  Transductive scales: {transductive_scales}")
    
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 9), squeeze=False)
    plt.subplots_adjust(hspace=0.30, wspace=0.25, left=0.08, right=0.92, top=0.92, bottom=0.10)
    
    # Process each column
    for col_idx in range(n_cols):
        # Inductive (row 0)
        if col_idx < len(inductive_scales):
            scale = inductive_scales[col_idx]
            scale_label = inductive_agg[inductive_agg['pretraining_scale'] == scale]['pretraining_scale_label'].iloc[0]
            col_data = inductive_agg[inductive_agg['pretraining_scale'] == scale]
            
            _plot_setting_panel(
                ax=axes[0, col_idx],
                data=col_data,
                setting='inductive',
                scale_label=scale_label,
                is_top_row=True,
                is_first_col=(col_idx == 0),
                color_map=global_color_map,
            )
        else:
            axes[0, col_idx].axis('off')
        
        # Transductive (row 1)
        if col_idx < len(transductive_scales):
            scale = transductive_scales[col_idx]
            scale_label = transductive_agg[transductive_agg['pretraining_scale'] == scale]['pretraining_scale_label'].iloc[0]
            col_data = transductive_agg[transductive_agg['pretraining_scale'] == scale]
            
            _plot_setting_panel(
                ax=axes[1, col_idx],
                data=col_data,
                setting='transductive',
                scale_label=scale_label,
                is_top_row=False,
                is_first_col=(col_idx == 0),
                color_map=global_color_map,
            )
        else:
            axes[1, col_idx].axis('off')
    
    # Overall title
    fig.suptitle("Community Detection: Inductive vs Transductive", 
                 fontweight='bold', fontsize=20, y=0.98)
    
    # Create legend using the global color map
    from matplotlib.lines import Line2D
    legend_handles = []
    
    # Add pretrained configs
    for (model_type, hyperparam_id), color in global_color_map.items():
        model_display = model_type.replace('gps_', '').upper()
        if hyperparam_id and hyperparam_id != 'default':
            label = f"{model_display} ({hyperparam_id})"
        else:
            label = model_display
        
        handle = Line2D([0], [0], color=color, linestyle='-', 
                       linewidth=2.5, marker='o', markersize=6, label=label)
        legend_handles.append(handle)
    
    # Add from-scratch baseline
    scratch_handle = Line2D([0], [0], color='black', linestyle='--', 
                           linewidth=2.5, marker='s', markersize=6, label='From Scratch')
    legend_handles.append(scratch_handle)
    
    # Add legend to last subplot in top row
    axes[0, -1].legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.0),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()


def _plot_setting_panel(
    ax,
    data: pd.DataFrame,
    setting: str,
    scale_label: str,
    is_top_row: bool,
    is_first_col: bool,
    color_map: dict,
):
    """Plot a single panel (one setting, one pretraining scale)."""
    
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        return
    
    # Separate pretrained and scratch runs
    pretrained = data[~data['mode'].isin(['scratch', 'scratch_frozen'])]
    scratch = data[data['mode'].isin(['scratch', 'scratch_frozen'])]
    
    # Plot pretrained configs using the global color map
    for (model_type, hyperparam_id), group in pretrained.groupby(['model_type', 'hyperparam_id']):
        group_sorted = group.sort_values('n_train')
        
        x_vals = group_sorted['n_train'].values
        y_vals = group_sorted['test_accuracy'].values * 100  # Convert to percentage
        
        # Only show error bars if we have std > 0 (i.e., multiple runs)
        has_std = group_sorted['test_accuracy_std'].fillna(0).values > 0
        y_errs = group_sorted['test_accuracy_std'].fillna(0).values * 100
        
        # Use the global color map
        color = color_map.get((model_type, hyperparam_id), 'gray')
        
        # Only pass yerr if there's actual variation
        if np.any(has_std):
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                color=color, alpha=0.85,
                linewidth=2.5, marker='o', markersize=7,
                capsize=5, capthick=2.0,
                elinewidth=2.0,
            )
        else:
            # No error bars - just plot the line
            ax.plot(x_vals, y_vals, 
                   color=color, alpha=0.85,
                   linewidth=2.5, marker='o', markersize=7)
    
    # Plot scratch baseline
    if len(scratch) > 0:
        scratch_group = scratch.sort_values('n_train')
        
        x_vals = scratch_group['n_train'].values
        y_vals = scratch_group['test_accuracy'].values * 100
        
        has_std = scratch_group['test_accuracy_std'].fillna(0).values > 0
        y_errs = scratch_group['test_accuracy_std'].fillna(0).values * 100
        
        if np.any(has_std):
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                color='black', alpha=0.85,
                linewidth=2.5, marker='s', markersize=7,
                linestyle='--',
                capsize=5, capthick=2.0,
                elinewidth=2.0,
            )
        else:
            ax.plot(x_vals, y_vals,
                   color='black', alpha=0.85,
                   linewidth=2.5, marker='s', markersize=7,
                   linestyle='--')
    
    # Labels
    if not is_top_row:
        if setting == 'inductive':
            ax.set_xlabel('# Finetuning Graphs', fontsize=13)
        else:
            ax.set_xlabel('# Finetuning Nodes', fontsize=13)
    
    if is_first_col:
        ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    
    # Title
    if is_top_row:
        setting_display = setting.capitalize()
        ax.text(0.5, 1.18, scale_label,
               transform=ax.transAxes, ha='center', va='bottom',
               fontweight='bold', fontsize=15)
        ax.text(0.5, 1.05, setting_display,
               transform=ax.transAxes, ha='center', va='bottom',
               fontweight='normal', fontsize=13)
    else:
        setting_display = setting.capitalize()
        ax.set_title(f"{setting_display}", pad=10, fontweight='normal', fontsize=13)
    
    # Y-axis styling
    y_vals_all = data['test_accuracy'].values * 100
    y_min = max(0, np.min(y_vals_all) - 10)
    y_max = min(105, np.max(y_vals_all) + 10)
    
    ax.set_ylim(y_min, y_max)
    
    # Color bands for accuracy ranges
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
    
    # X-axis: use log scale if range is large
    x_vals_all = data['n_train'].values
    if len(x_vals_all) > 0 and np.max(x_vals_all) / np.min(x_vals_all) > 10:
        ax.set_xscale('log')
        ax.set_xticks(sorted(x_vals_all))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare inductive vs transductive community detection results"
    )
    parser.add_argument(
        '--inductive_project',
        type=str,
        required=True,
        help='Wandb project for inductive downstream results'
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
        default='inductive_vs_transductive_comparison.png',
        help='Output filename for plot'
    )
    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save processed dataframes to CSV'
    )
    
    args = parser.parse_args()
    
    # Fetch runs
    print("\n" + "=" * 80)
    print("FETCHING INDUCTIVE RUNS")
    print("=" * 80)
    inductive_runs = fetch_runs_from_wandb(args.inductive_project)
    
    print("\n" + "=" * 80)
    print("FETCHING TRANSDUCTIVE RUNS")
    print("=" * 80)
    transductive_runs = fetch_runs_from_wandb(args.transductive_project)
    
    # Process runs
    print("\n" + "=" * 80)
    print("PROCESSING RUNS")
    print("=" * 80)
    inductive_df = process_inductive_runs(inductive_runs)
    transductive_df = process_transductive_runs(transductive_runs)
    
    if len(inductive_df) == 0:
        print("ERROR: No valid inductive runs found")
        return
    
    if len(transductive_df) == 0:
        print("ERROR: No valid transductive runs found")
        return
    
    # Save CSVs if requested
    if args.save_csv:
        inductive_csv = args.output.replace('.png', '_inductive.csv')
        transductive_csv = args.output.replace('.png', '_transductive.csv')
        
        inductive_df.to_csv(inductive_csv, index=False)
        transductive_df.to_csv(transductive_csv, index=False)
        
        print(f"\n✓ Saved inductive data to {inductive_csv}")
        print(f"✓ Saved transductive data to {transductive_csv}")
    
    # Create plot
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOT")
    print("=" * 80)
    
    create_comparison_plot(
        inductive_df=inductive_df,
        transductive_df=transductive_df,
        output_path=args.output,
    )
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
