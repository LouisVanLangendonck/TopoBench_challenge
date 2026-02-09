import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Data Extraction
# =============================================================================

# Define metrics by type
METRIC_CONFIGS = {
    # MAE-based metrics (basic properties)
    "avg_degree": {"type": "mae", "category": "basic_property_reconstruction"},
    "gini": {"type": "mae", "category": "basic_property_reconstruction"},
    "diameter": {"type": "mae", "category": "basic_property_reconstruction"},
    # MAE-based metrics (community-related properties)
    "homophily": {"type": "mae", "category": "community_related_property_reconstruction"},
    "community_presence": {"type": "mae", "category": "community_related_property_reconstruction"},
    # Accuracy-based metrics
    "community_detection": {"type": "accuracy", "category": "community_related_property_reconstruction"},
}

# For backwards compatibility and grouping
TASKS_DICT = {
    "basic_property_reconstruction": ["avg_degree", "gini", "diameter"],
    "community_related_property_reconstruction": ["homophily", "community_presence", "community_detection"]
}

def fetch_runs_from_wandb(project_path: str) -> List[Dict[str, Any]]:
    """
    Fetch all runs from a wandb project.
    
    Parameters
    ----------
    project_path : str
        Wandb project path in format "entity/project" or just "project".
    
    Returns
    -------
    List[Dict[str, Any]]
        List of run data dictionaries.
    """
    api = wandb.Api()
    
    print(f"\n{'=' * 80}")
    print(f"FETCHING RUNS FROM WANDB PROJECT: {project_path}")
    print(f"{'=' * 80}")
    
    # Fetch runs
    runs = api.runs(project_path, filters={"state": "finished"})
    
    run_data = []
    
    for run in runs:
        # Get config and summary
        config = dict(run.config)
        summary = dict(run.summary)
        
        # Extract relevant data
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
    """Get value from nested dict using '/' separated path.
    
    First tries to access the key_path directly (for wandb flat keys like 'a/b/c').
    If that fails, tries to navigate through nested structure.
    """
    # First try: key might exist as-is (wandb flat config)
    if key_path in d:
        return d[key_path]
    
    # Second try: navigate through nested structure
    keys = key_path.split('/')
    value = d
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def check_metric_present(summary: dict, metric_name: str) -> bool:
    """
    Check if a specific metric is present in the summary.
    
    Parameters
    ----------
    summary : dict
        Run summary from wandb.
    metric_name : str
        Name of the metric (e.g., 'size', 'community_detection').
    
    Returns
    -------
    bool
        True if the metric is present.
    """
    metric_config = METRIC_CONFIGS.get(metric_name)
    if not metric_config:
        return False
    
    if metric_config['type'] == 'mae':
        # Check for either test/mae or test/improvement (both should be present)
        return (f'test/mae_{metric_name}' in summary or 
                f'test/improvement_{metric_name}' in summary)
    elif metric_config['type'] == 'accuracy':
        # For community_detection, the key is 'test/accuracy_community_detection'
        return f'test/accuracy_{metric_name}' in summary
    
    return False


def extract_backbone_wrapper_params(config: dict) -> dict:
    """
    Dynamically extract all parameters under pretrain/model/backbone_wrapper/, 
    pretrain/model/backbone/, and pretrain/optimizer/parameters/.
    
    Parameters
    ----------
    config : dict
        Run config from wandb.
    
    Returns
    -------
    dict
        Dictionary of backbone_wrapper, backbone, and optimizer parameters with flattened keys.
    """
    model_params = {}
    
    # Extract backbone_wrapper parameters
    wrapper_prefix = 'pretrain/model/backbone_wrapper/'
    for key in config.keys():
        if key.startswith(wrapper_prefix):
            param_name = key[len(wrapper_prefix):]
            model_params[param_name] = config[key]
    
    # Also try nested structure for backbone_wrapper
    if not model_params:
        nested_value = get_nested_value(config, 'pretrain/model/backbone_wrapper')
        if isinstance(nested_value, dict):
            model_params.update(nested_value)
    
    # Extract backbone parameters (e.g., num_layers, hidden_dim, etc.)
    backbone_prefix = 'pretrain/model/backbone/'
    for key in config.keys():
        if key.startswith(backbone_prefix):
            param_name = key[len(backbone_prefix):]
            # Use prefixed name to distinguish from wrapper params if there's overlap
            model_params[f'backbone.{param_name}'] = config[key]
    
    # Also try nested structure for backbone
    if not any(k.startswith('backbone.') for k in model_params.keys()):
        nested_value = get_nested_value(config, 'pretrain/model/backbone')
        if isinstance(nested_value, dict):
            for param_name, param_value in nested_value.items():
                model_params[f'backbone.{param_name}'] = param_value
    
    # Extract optimizer parameters
    optimizer_prefix = 'pretrain/optimizer/parameters/'
    for key in config.keys():
        if key.startswith(optimizer_prefix):
            param_name = key[len(optimizer_prefix):]
            # Use prefixed name to distinguish from other params
            model_params[f'opt.{param_name}'] = config[key]
    
    # Also try nested structure for optimizer
    if not any(k.startswith('opt.') for k in model_params.keys()):
        nested_value = get_nested_value(config, 'pretrain/optimizer/parameters')
        if isinstance(nested_value, dict):
            for param_name, param_value in nested_value.items():
                model_params[f'opt.{param_name}'] = param_value
    
    return model_params


def process_runs(run_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process run data into a structured dataframe.
    Each run produces one row per metric found in the summary.
    
    Parameters
    ----------
    run_data : List[Dict[str, Any]]
        Raw run data from wandb.
    
    Returns
    -------
    pd.DataFrame
        Processed dataframe with one row per metric per run.
    """
    records = []
    
    for run in run_data:
        config = run['config']
        summary = run['summary']
        
        # Extract backbone_wrapper hyperparameters dynamically
        wrapper_params = extract_backbone_wrapper_params(config)
        
        # Extract basic info (shared across all rows from this run)
        base_info = {
            'run_id': run['run_id'],
            'run_name': run['run_name'],
            'model_type': config.get('pretrain/model/model_name', 'unknown'),
            'n_graphs': get_nested_value(config, 'pretrain/dataset/loader/parameters/generation_parameters/family_parameters/n_graphs'),
            'n_train': get_nested_value(config, 'n_train'),
            'mode': get_nested_value(config, 'mode'),
            'readout_type': get_nested_value(config, 'readout_type'),
            'lr': get_nested_value(config, 'lr'),
            'patience': get_nested_value(config, 'patience'),
            'batch_size': get_nested_value(config, 'batch_size'),
            'classifier_dropout': get_nested_value(config, 'classifier_dropout'),
        }
        
        # Add all wrapper params to base_info
        base_info.update(wrapper_params)
        
        # Check each metric individually
        for metric_name, metric_config in METRIC_CONFIGS.items():
            if check_metric_present(summary, metric_name):
                record = dict(base_info)
                record['metric'] = metric_name
                record['metric_type'] = metric_config['type']
                record['task_type'] = metric_config['category']
                
                if metric_config['type'] == 'mae':
                    # Extract MAE-based metrics
                    # Train/val values are logged as mae during training
                    record['train_value'] = summary.get(f'train/mae_{metric_name}')
                    record['val_value'] = summary.get(f'val/mae_{metric_name}')
                    # Test values
                    record['test_value'] = summary.get(f'test/mae_{metric_name}')
                    record['test_baseline'] = summary.get(f'test/baseline_mae_{metric_name}')
                    record['test_improvement'] = summary.get(f'test/improvement_{metric_name}')
                elif metric_config['type'] == 'accuracy':
                    # Extract accuracy-based metrics
                    record['train_value'] = summary.get(f'train/accuracy_{metric_name}')
                    record['val_value'] = summary.get(f'val/accuracy_{metric_name}')
                    record['test_value'] = summary.get(f'test/accuracy_{metric_name}')
                    record['test_baseline'] = None
                    record['test_improvement'] = None
                
                records.append(record)
    
    df = pd.DataFrame(records)
    
    print(f"\nExtracted {len(records)} metric entries from {len(run_data)} runs:")
    if len(df) > 0:
        print(f"  Task types: {df['task_type'].unique().tolist()}")
        print(f"  Metric types: {df['metric_type'].unique().tolist()}")
        print(f"  Metrics: {df['metric'].unique().tolist()}")
        print(f"  Model types: {df['model_type'].unique().tolist()}")
        print(f"  Modes: {df['mode'].unique().tolist()}")
        print(f"  N_train values: {sorted(df['n_train'].unique().tolist())}")
        print(f"  N_graphs values: {sorted(df['n_graphs'].dropna().unique().tolist())} (NaN count: {df['n_graphs'].isna().sum()})")
        
        # Print hyperparameter distributions (dynamically detect all wrapper params)
        metadata_cols = {'run_id', 'run_name', 'model_type', 'n_graphs', 'n_train', 'mode', 
                        'readout_type', 'lr', 'patience', 'batch_size', 'classifier_dropout',
                        'metric', 'metric_type', 'task_type', 'train_value', 'val_value', 
                        'test_value', 'test_baseline', 'test_improvement'}
        hyperparam_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"\n  Hyperparameter distributions:")
        for hp in sorted(hyperparam_cols):
            unique_vals = df[hp].dropna().unique()
            if len(unique_vals) > 0:
                # Format values nicely
                if all(isinstance(v, (int, float)) for v in unique_vals):
                    formatted_vals = [f"{v:.3g}" if isinstance(v, float) else str(v) for v in sorted(unique_vals)]
                else:
                    formatted_vals = sorted([str(v) for v in unique_vals])
                print(f"    {hp}: {formatted_vals}")
    
    return df

def identify_variable_hyperparams(df: pd.DataFrame, model_type: str) -> List[str]:
    """
    For a given model_type, identify which hyperparameter columns vary.
    Dynamically detects all backbone_wrapper parameters and other hyperparameters.
    Also removes redundant columns that always have identical values with other columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all runs.
    model_type : str
        Model type to analyze.
    
    Returns
    -------
    List[str]
        List of variable hyperparameter column names (with redundant columns removed).
    """
    model_df = df[df['model_type'] == model_type]
    
    # Standard hyperparameter columns to check
    standard_cols = ['n_graphs', 'lr', 'batch_size', 'classifier_dropout', 'patience']
    
    # Find all columns that could be backbone_wrapper parameters
    # (any column not in known metadata columns)
    metadata_cols = {'run_id', 'run_name', 'model_type', 'n_graphs', 'n_train', 'mode', 
                     'readout_type', 'lr', 'patience', 'batch_size', 'classifier_dropout',
                     'metric', 'metric_type', 'task_type', 'train_value', 'val_value', 
                     'test_value', 'test_baseline', 'test_improvement', 'best_readout', 'run_ids',
                     'train_value_std', 'val_value_std', 'test_value_std', 
                     'test_baseline_std', 'test_improvement_std', 'hyperparam_id'}
    
    # All other columns are potential hyperparameters
    potential_hyperparam_cols = [col for col in model_df.columns if col not in metadata_cols]
    
    # Check which ones vary
    varying = []
    for col in potential_hyperparam_cols:
        try:
            # Drop NaN values before checking uniqueness
            non_na_values = model_df[col].dropna()
            if len(non_na_values) > 0 and non_na_values.nunique() > 1:
                varying.append(col)
        except:
            pass
    
    # Remove redundant columns (columns that always have identical values with another column)
    varying_deduped = _remove_redundant_columns(model_df, varying)
    
    return sorted(varying_deduped)


def _remove_redundant_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Remove columns that always have identical values with other columns.
    Keep the column with the shortest/simplest name.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data.
    columns : List[str]
        List of column names to check.
    
    Returns
    -------
    List[str]
        List of columns with redundant ones removed.
    """
    if len(columns) <= 1:
        return columns
    
    # Track which columns to keep
    to_keep = set(columns)
    redundant_groups = []
    
    # Compare each pair of columns
    for i, col1 in enumerate(columns):
        if col1 not in to_keep:
            continue
        
        for col2 in columns[i+1:]:
            if col2 not in to_keep:
                continue
            
            # Check if columns always have identical values (handling NaN)
            col1_vals = df[col1].fillna(-999999)  # Replace NaN with sentinel
            col2_vals = df[col2].fillna(-999999)
            
            if (col1_vals == col2_vals).all():
                # Columns are identical - keep the one with simpler name
                # Priority: shorter name, then alphabetically first
                if len(col1) < len(col2):
                    to_keep.discard(col2)
                    redundant_groups.append((col1, col2))
                elif len(col1) > len(col2):
                    to_keep.discard(col1)
                    redundant_groups.append((col2, col1))
                else:
                    # Same length - keep alphabetically first
                    if col1 < col2:
                        to_keep.discard(col2)
                        redundant_groups.append((col1, col2))
                    else:
                        to_keep.discard(col1)
                        redundant_groups.append((col2, col1))
    
    # Print redundancy information
    if redundant_groups:
        print(f"\n  Found redundant columns (always identical values):")
        for kept, removed in redundant_groups:
            print(f"    Keeping '{kept}', removing '{removed}'")
    
    return list(to_keep)


def create_hyperparam_id(row, varying_hyperparams: List[str]) -> str:
    """
    Create a unique ID for a hyperparameter configuration.
    
    Parameters
    ----------
    row : pd.Series
        Row from dataframe.
    varying_hyperparams : List[str]
        List of hyperparameter columns that vary.
    
    Returns
    -------
    str
        Unique ID for this hyperparameter configuration.
    """
    parts = []
    
    # Sort hyperparameters for consistent ordering
    for hp in sorted(varying_hyperparams):
        val = row.get(hp)
        if pd.notna(val):
            # Format the hyperparameter name for display
            display_name = hp.replace('drop_', '').replace('_rate', '').replace('_', '')
            
            # Format the value
            if isinstance(val, (int, float)):
                if isinstance(val, float) and val != int(val):
                    parts.append(f"{display_name}={val:.2f}")
                else:
                    parts.append(f"{display_name}={int(val)}")
            else:
                parts.append(f"{display_name}={val}")
    
    return ", ".join(parts) if parts else "default"


def select_best_readout(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique configuration (model_type, hyperparams, n_graphs, n_train, metric, mode), 
    select the best readout_type based on val_value.
    
    Aggregation strategy:
    - For MAE metrics: Select best (lowest) val_value across readout_types
    - For accuracy metrics: Select best (highest) val_value across readout_types
    - For rows with same config (including readout): Aggregate values (mean + std)
    
    Returns a dataframe with one row per (model_type, hyperparams, n_graphs, n_train, metric, mode) combination.
    """
    # Identify variable hyperparameters per model_type
    variable_hyperparams = {}
    for model_type in df['model_type'].unique():
        variable_hyperparams[model_type] = identify_variable_hyperparams(df, model_type)
    
    print("\n" + "=" * 80)
    print("VARIABLE HYPERPARAMETERS PER MODEL TYPE")
    print("=" * 80)
    for model_type, hps in variable_hyperparams.items():
        print(f"\n{model_type}:")
        print(f"  {hps}")
    
    # Group by: model_type, hyperparams, n_graphs, n_train, metric, mode
    group_cols = ['model_type', 'n_graphs', 'n_train', 'mode', 'metric', 'metric_type', 'task_type']
    
    # Add ALL varying hyperparameter columns to grouping (dynamically discovered)
    metadata_cols = {'run_id', 'run_name', 'model_type', 'n_graphs', 'n_train', 'mode', 
                     'readout_type', 'lr', 'patience', 'batch_size', 'classifier_dropout',
                     'metric', 'metric_type', 'task_type', 'train_value', 'val_value', 
                     'test_value', 'test_baseline', 'test_improvement'}
    
    potential_hyperparam_cols = [col for col in df.columns if col not in metadata_cols]
    
    for col in potential_hyperparam_cols:
        if col in df.columns and df[col].dropna().nunique() > 1:
            group_cols.append(col)
    
    value_cols = ['train_value', 'val_value', 'test_value', 'test_baseline', 'test_improvement']
    
    print("\n" + "=" * 80)
    print("GROUPING COLUMNS (excluding readout_type and metric values):")
    print("=" * 80)
    print(f"  {group_cols}")
    
    best_records = []
    grouped = df.groupby(group_cols, dropna=False)
    
    print("\n" + "=" * 80)
    print("SELECTING BEST READOUT PER METRIC FOR EACH GROUP")
    print("=" * 80)
    
    for group_key, group_df in grouped:
        if len(group_df) == 0:
            continue
        
        base_record = {k: v for k, v in zip(group_cols, group_key)}
        metric_type = base_record['metric_type']
        
        # For community_detection (accuracy), readout_type doesn't matter
        if base_record['metric'] == 'community_detection':
            valid_rows = group_df[group_df['test_value'].notna()]
            
            if len(valid_rows) == 0:
                continue
            
            record = dict(base_record)
            record['best_readout'] = None
            record['run_ids'] = valid_rows['run_id'].tolist()
            
            # Compute mean and std for all value columns
            for value_col in value_cols:
                values = valid_rows[value_col].dropna().tolist()
                if len(values) > 0:
                    record[value_col] = np.mean(values)
                    record[f'{value_col}_std'] = np.std(values) if len(values) > 1 else 0.0
                else:
                    record[value_col] = None
                    record[f'{value_col}_std'] = None
            
            best_records.append(record)
            continue
        
        # For other metrics, filter out rows with missing val_value
        valid_rows = group_df.dropna(subset=['val_value'])
        
        if len(valid_rows) == 0:
            continue
        
        # Select best readout based on val_value
        if metric_type == 'mae':
            best_idx = valid_rows['val_value'].idxmin()
        else:
            best_idx = valid_rows['val_value'].idxmax()
        
        best_row = valid_rows.loc[best_idx]
        
        # Check if there are multiple rows with the same readout_type
        same_readout = group_df[group_df['readout_type'] == best_row['readout_type']]
        
        record = dict(base_record)
        record['best_readout'] = best_row['readout_type']
        record['run_ids'] = same_readout['run_id'].tolist()
        
        # Compute mean and std for all value columns
        for value_col in value_cols:
            values = same_readout[value_col].dropna().tolist()
            if len(values) > 0:
                record[value_col] = np.mean(values)
                record[f'{value_col}_std'] = np.std(values) if len(values) > 1 else 0.0
            else:
                record[value_col] = None
                record[f'{value_col}_std'] = None
        
        best_records.append(record)
    
    best_df = pd.DataFrame(best_records)
    print(f"\nReduced from {len(df)} metric entries to {len(best_df)} unique (config, metric) selections")
    
    return best_df

# =============================================================================
# Publication Quality Plotting for Hyperparameter Comparison
# =============================================================================

import matplotlib as mpl

def is_effectively_nan(val):
    """Check if value is effectively NaN."""
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val)
        if arr.size == 0:
            return True
        arr = arr.flatten()
        return np.all(pd.isna(arr))
    return pd.isna(val)


def create_hyperparam_comparison_plots(
    df: pd.DataFrame,
    output_path: str = "hyperparam_comparison.png",
    wandb_project: str = None
):
    """
    Create comparison plots for different hyperparameter configurations.
    Creates separate plots for each unique (model_type, n_graphs) combination.
    Uses publication-quality color, sizing, legend, and axis conventions.
    """
    # Create output directory based on wandb project name
    if wandb_project:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        output_dir = Path(f"plots/{wandb_project}_hyperparam_plots")
        output_dir.mkdir(exist_ok=True)
        print(f"\n{'=' * 80}")
        print(f"OUTPUT DIRECTORY: {output_dir}")
        print(f"{'=' * 80}")
    else:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        output_dir = Path("plots")
    
    # Get unique (model_type, n_graphs) combinations
    model_n_graph_combos = df[['model_type', 'n_graphs']].drop_duplicates()
    
    print(f"\n{'=' * 80}")
    print(f"FOUND {len(model_n_graph_combos)} UNIQUE (MODEL_TYPE, N_GRAPHS) COMBINATIONS")
    print(f"{'=' * 80}")
    
    # Iterate over each combination
    for _, combo in model_n_graph_combos.iterrows():
        model_type = combo['model_type']
        n_graphs = combo['n_graphs']
        
        combo_df = df[(df['model_type'] == model_type) & (df['n_graphs'] == n_graphs)]
        
        print(f"\n{'=' * 80}")
        print(f"PROCESSING MODEL_TYPE={model_type}, N_GRAPHS={n_graphs}")
        print(f"  Found {len(combo_df)} rows")
        print(f"{'=' * 80}")
        
        # Identify variable hyperparameters for this model_type
        varying_hyperparams = identify_variable_hyperparams(combo_df, model_type)
        
        # Create hyperparam_id for each row
        combo_df = combo_df.copy()
        combo_df['hyperparam_id'] = combo_df.apply(
            lambda row: create_hyperparam_id(row, varying_hyperparams), axis=1
        )
        
        for task_type in combo_df['task_type'].unique():
            task_df = combo_df[combo_df['task_type'] == task_type]
            metrics = sorted(task_df['metric'].unique())
            
            # Special ordering for community_related: put community_detection at the end
            if task_type == 'community_related_property_reconstruction' and 'community_detection' in metrics:
                mae_metrics = [m for m in metrics if m != 'community_detection']
                metrics = mae_metrics + ['community_detection']
                print(f"  Reordered metrics for community_related: {metrics}")
            
            # Output file
            base_name = output_path.rsplit('.', 1)[0]
            ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
            model_clean = model_type.replace('gps_', '')
            filename = f"{base_name}_{model_clean}_{task_type}_n{int(n_graphs)}.{ext}"
            task_output_path = str(output_dir / filename)
            
            print(f"\nCreating plot for {model_type}, task={task_type}, n_graphs={n_graphs}")
            
            _create_hyperparam_plot(task_df, metrics, task_output_path, task_type, model_type, n_graphs)


def _create_hyperparam_plot(
    df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    task_type: str,
    model_type: str,
    n_graphs: int
):
    """Create a publication-quality plot for hyperparameter comparison."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Check if we should use bar plots or line plots
    unique_n_train = df['n_train'].dropna().unique()
    use_barplot = len(unique_n_train) == 1
    
    if use_barplot:
        print(f"  Using bar plots (single n_train value: {unique_n_train[0]})")
        _create_hyperparam_barplot(df, metrics, output_path, task_type, model_type, n_graphs)
    else:
        print(f"  Using line plots ({len(unique_n_train)} n_train values: {sorted(unique_n_train)})")
        _create_hyperparam_lineplot(df, metrics, output_path, task_type, model_type, n_graphs)


def _create_hyperparam_lineplot(
    df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    task_type: str,
    model_type: str,
    n_graphs: int
):
    """Create a publication-quality line plot for hyperparameter comparison."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Publication quality settings
    mpl.rcParams.update({
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "DejaVu Serif",
        "axes.linewidth": 1.1,
        "xtick.direction": 'out',
        "ytick.direction": 'out',
        "axes.titleweight": 'bold',
        "figure.titlesize": 22
    })
    
    n_metrics = len(metrics)
    has_community_detection = 'community_detection' in metrics
    
    if has_community_detection and len(metrics) > 1:
        fig = plt.figure(figsize=(4.6 * n_metrics, 8.6))
        gs = GridSpec(2, n_metrics, figure=fig, hspace=0.35, 
                     width_ratios=[1]*n_metrics,
                     wspace=0.21,
                     left=0.08, right=0.78, top=0.89, bottom=0.15)
        axes = []
        for row in range(2):
            row_axes = []
            for col in range(n_metrics):
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
            axes.append(row_axes)
        axes = np.array(axes)
    else:
        fig, axes = plt.subplots(2, n_metrics, figsize=(4.6 * n_metrics, 8.6), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.21, left=0.08, right=0.78, top=0.89, bottom=0.15)
    
    # Get unique hyperparam configurations
    unique_hyperparams = sorted(df['hyperparam_id'].unique())
    
    # Assign colors using a colormap (use tab10 or Set3 for distinct colors)
    n_configs = len(unique_hyperparams)
    if n_configs <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_configs]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_configs)))
    
    hyperparam_colors = {hp: colors[i] for i, hp in enumerate(unique_hyperparams)}
    
    mode_rows = [
        ('linear', 'scratch_frozen', 0, 'Frozen Encoder'),
        ('finetune-linear', 'scratch', 1, 'Unfrozen Encoder')
    ]
    
    # First pass: collect y-values for accuracy metrics (synchronized limits)
    accuracy_y_ranges = {metric: {'all_y_vals': []} for metric in metrics if METRIC_CONFIGS.get(metric, {}).get('type') == 'accuracy'}
    
    for target_mode, scratch_mode, row_idx, row_title in mode_rows:
        for metric_idx, metric in enumerate(metrics):
            metric_config = METRIC_CONFIGS.get(metric, {})
            metric_type = metric_config.get('type', 'mae')
            is_accuracy = (metric_type == 'accuracy')
            
            if not is_accuracy:
                continue
            
            mode_df = df[
                (df['mode'] == target_mode) &
                (df['metric'] == metric)
            ]
            
            if len(mode_df) == 0:
                continue
            
            for hp_id in unique_hyperparams:
                hp_df = mode_df[mode_df['hyperparam_id'] == hp_id]
                if len(hp_df) == 0:
                    continue
                
                hp_df = hp_df.sort_values('n_train')
                for _, row in hp_df.iterrows():
                    val = row['test_value']
                    if not is_effectively_nan(val):
                        accuracy_y_ranges[metric]['all_y_vals'].append(val * 100)
            
            # Collect from scratch runs (pick one hyperparam config)
            scratch_df = df[
                (df['mode'] == scratch_mode) & 
                (df['metric'] == metric)
            ]
            if len(scratch_df) > 0:
                # Just use first hyperparam config for scratch
                first_hp = scratch_df['hyperparam_id'].iloc[0]
                scratch_df = scratch_df[scratch_df['hyperparam_id'] == first_hp]
                scratch_df = scratch_df.sort_values('n_train')
                for _, row in scratch_df.iterrows():
                    val = row['test_value']
                    if not is_effectively_nan(val):
                        accuracy_y_ranges[metric]['all_y_vals'].append(val * 100)
    
    # Compute synchronized y-limits
    for metric in accuracy_y_ranges:
        all_vals = accuracy_y_ranges[metric]['all_y_vals']
        if len(all_vals) > 0:
            max_val = max(all_vals)
            min_val = min(all_vals)
            y_max = min(105, max_val + 10)
            y_min = max(0, min_val - 5)
        else:
            y_max = 105
            y_min = 0
        accuracy_y_ranges[metric]['y_min'] = y_min
        accuracy_y_ranges[metric]['y_max'] = y_max
    
    # Second pass: actual plotting
    for target_mode, scratch_mode, row_idx, row_title in mode_rows:
        for metric_idx, metric in enumerate(metrics):
            ax = axes[row_idx, metric_idx]
            
            metric_config = METRIC_CONFIGS.get(metric, {})
            metric_type = metric_config.get('type', 'mae')
            is_accuracy = (metric_type == 'accuracy')
            
            mode_df = df[
                (df['mode'] == target_mode) &
                (df['metric'] == metric)
            ]
            
            if len(mode_df) == 0:
                ax.text(0.5, 0.5, f'No data for {target_mode}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=15
                )
                continue
            
            # Plot curves for each hyperparam configuration
            for hp_id in unique_hyperparams:
                hp_df = mode_df[mode_df['hyperparam_id'] == hp_id]
                if len(hp_df) == 0:
                    continue
                
                hp_df = hp_df.sort_values('n_train')
                
                x_vals = []
                y_vals = []
                y_errs = []
                
                for _, row in hp_df.iterrows():
                    n_train = row['n_train']
                    if is_accuracy:
                        val = row['test_value']
                        val_std = row.get('test_value_std', 0.0)
                        if not is_effectively_nan(val):
                            x_vals.append(n_train)
                            y_vals.append(val * 100)
                            y_errs.append(val_std * 100 if not is_effectively_nan(val_std) else 0.0)
                    else:
                        val = row['test_improvement']
                        val_std = row.get('test_improvement_std', 0.0)
                        if not is_effectively_nan(val):
                            x_vals.append(n_train)
                            y_vals.append(np.clip(val, -100, 100))
                            y_errs.append(val_std if not is_effectively_nan(val_std) else 0.0)
                
                if len(x_vals) > 0:
                    color = hyperparam_colors[hp_id]
                    
                    ax.errorbar(
                        x_vals, y_vals, yerr=y_errs,
                        color=color, alpha=0.85,
                        linewidth=2.5, marker='o', markersize=8,
                        capsize=5, capthick=1.5,
                        label=hp_id
                    )
            
            # Plot from-scratch curve (just use first hyperparam config)
            scratch_df = df[
                (df['mode'] == scratch_mode) & 
                (df['metric'] == metric)
            ]
            if len(scratch_df) > 0:
                first_hp = scratch_df['hyperparam_id'].iloc[0]
                scratch_df = scratch_df[scratch_df['hyperparam_id'] == first_hp]
                scratch_df = scratch_df.sort_values('n_train')
                
                scratch_x = []
                scratch_y = []
                scratch_err = []
                
                for _, row in scratch_df.iterrows():
                    n_train = row['n_train']
                    if is_accuracy:
                        val = row['test_value']
                        val_std = row.get('test_value_std', 0.0)
                        if not is_effectively_nan(val):
                            scratch_x.append(n_train)
                            scratch_y.append(val * 100)
                            scratch_err.append(val_std * 100 if not is_effectively_nan(val_std) else 0.0)
                    else:
                        val = row['test_improvement']
                        val_std = row.get('test_improvement_std', 0.0)
                        if not is_effectively_nan(val):
                            scratch_x.append(n_train)
                            scratch_y.append(np.clip(val, -100, 100))
                            scratch_err.append(val_std if not is_effectively_nan(val_std) else 0.0)
                
                if len(scratch_x) > 0:
                    ax.errorbar(
                        scratch_x, scratch_y, yerr=scratch_err,
                        color='black', alpha=0.85,
                        linewidth=2.5, marker='s', markersize=8,
                        linestyle=(0, (7, 4)),
                        capsize=5, capthick=1.5,
                        label='From Scratch'
                    )
            
            # Labels and formatting
            if row_idx == 1:
                ax.set_xlabel('# Finetuning Graphs', fontsize=14)
            else:
                ax.set_xlabel('')
            
            if metric_idx == 0:
                ax.set_ylabel('Test Improvement\nover baseline (%)', fontsize=14)
            elif is_accuracy and metric_idx > 0:
                ax.set_ylabel('Test Accuracy (%)', fontsize=14)
            else:
                ax.set_ylabel('')
            
            if row_idx == 0:
                metric_display = metric.replace("_", " ").capitalize()
                if metric == "community_detection":
                    metric_display += " (node)"
                
                ax.text(0.5, 1.15, metric_display, 
                       transform=ax.transAxes, ha='center', va='bottom',
                       fontweight='bold', fontsize=16)
                ax.text(0.5, 1.02, row_title,
                       transform=ax.transAxes, ha='center', va='bottom',
                       fontweight='normal', fontsize=14)
                ax.set_title('')
            else:
                ax.set_title(f'{row_title}', pad=8, fontweight='normal', fontsize=14)
            
            # Log scale for x-axis if needed
            all_n_train = df['n_train'].dropna().unique()
            if len(all_n_train) > 0 and max(all_n_train) / min(all_n_train) > 10:
                ax.set_xscale('log')
                ax.set_xticks(sorted(all_n_train))
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
            
            # Y-axis limits and styling
            if is_accuracy:
                if metric in accuracy_y_ranges:
                    y_min = accuracy_y_ranges[metric]['y_min']
                    y_max = accuracy_y_ranges[metric]['y_max']
                else:
                    y_max = 105
                    y_min = 0
                
                ax.set_ylim(y_min, y_max)
                if y_max - y_min <= 60:
                    tick_spacing = 10
                else:
                    tick_spacing = 25
                ax.set_yticks(np.arange(y_min, y_max + 1, tick_spacing))
                
                if y_max > 75:
                    ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(50, 75, facecolor='lightyellow', alpha=0.08, zorder=0)
                    ax.axhspan(75, y_max, facecolor='palegreen', alpha=0.08, zorder=0)
                elif y_max > 50:
                    ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(50, y_max, facecolor='lightyellow', alpha=0.08, zorder=0)
                else:
                    ax.axhspan(max(y_min, 0), y_max, facecolor='lightcoral', alpha=0.08, zorder=0)
            else:
                ax.axhline(y=0, color='grey', linestyle='-', linewidth=1.1, alpha=0.65)
                ax.set_ylim(-110, 110)
                ax.set_yticks(np.arange(-100, 125, 50))
                ax.axhspan(-110, 0, facecolor='lightgray', alpha=0.08, zorder=0)
                ax.axhspan(0, 110, facecolor='palegreen', alpha=0.06, zorder=0)
            
            ax.grid(axis='both', alpha=0.20, zorder=0, linestyle=':', linewidth=1.0)
            
            # Add legend to top-right subplot
            if row_idx == 0 and metric_idx == len(metrics) - 1:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, frameon=True, 
                         fancybox=True, shadow=True)
    
    # Add extra spacing before community_detection column if present
    if has_community_detection and len(metrics) > 1:
        extra_gap = 0.04
        for row_idx in range(2):
            for col_idx in range(n_metrics):
                ax = axes[row_idx, col_idx]
                pos = ax.get_position()
                if col_idx < n_metrics - 1:
                    ax.set_position([pos.x0 - extra_gap * 0.5, pos.y0, pos.width, pos.height])
                else:
                    ax.set_position([pos.x0 + extra_gap * 0.5, pos.y0, pos.width, pos.height])
    
    # Overall title
    model_display = model_type.replace('gps_', '').upper()
    title_text = f"{model_display}: {task_type.replace('_', ' ').title()} (Pretrained on {int(n_graphs)} graphs)"
    fig.suptitle(title_text, fontweight='bold', fontsize=22, y=0.995)
    
    plt.tight_layout(rect=[0, 0.05, 0.98, 0.97])
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def _create_hyperparam_barplot(
    df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    task_type: str,
    model_type: str,
    n_graphs: int
):
    """Create a publication-quality bar plot for hyperparameter comparison (single n_train value)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Publication quality settings
    mpl.rcParams.update({
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "DejaVu Serif",
        "axes.linewidth": 1.1,
        "xtick.direction": 'out',
        "ytick.direction": 'out',
        "axes.titleweight": 'bold',
        "figure.titlesize": 22
    })
    
    n_metrics = len(metrics)
    has_community_detection = 'community_detection' in metrics
    
    if has_community_detection and len(metrics) > 1:
        fig = plt.figure(figsize=(4.6 * n_metrics, 8.6))
        gs = GridSpec(2, n_metrics, figure=fig, hspace=0.35, 
                     width_ratios=[1]*n_metrics,
                     wspace=0.21,
                     left=0.08, right=0.78, top=0.89, bottom=0.15)
        axes = []
        for row in range(2):
            row_axes = []
            for col in range(n_metrics):
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
            axes.append(row_axes)
        axes = np.array(axes)
    else:
        fig, axes = plt.subplots(2, n_metrics, figsize=(4.6 * n_metrics, 8.6), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.21, left=0.08, right=0.78, top=0.89, bottom=0.15)
    
    # Get unique hyperparam configurations
    unique_hyperparams = sorted(df['hyperparam_id'].unique())
    
    # Assign colors using a colormap
    n_configs = len(unique_hyperparams)
    if n_configs <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_configs]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_configs)))
    
    hyperparam_colors = {hp: colors[i] for i, hp in enumerate(unique_hyperparams)}
    
    mode_rows = [
        ('linear', 'scratch_frozen', 0, 'Frozen Encoder'),
        ('finetune-linear', 'scratch', 1, 'Unfrozen Encoder')
    ]
    
    # First pass: collect y-values for accuracy metrics (synchronized limits)
    accuracy_y_ranges = {metric: {'all_y_vals': []} for metric in metrics if METRIC_CONFIGS.get(metric, {}).get('type') == 'accuracy'}
    
    for target_mode, scratch_mode, row_idx, row_title in mode_rows:
        for metric_idx, metric in enumerate(metrics):
            metric_config = METRIC_CONFIGS.get(metric, {})
            metric_type = metric_config.get('type', 'mae')
            is_accuracy = (metric_type == 'accuracy')
            
            if not is_accuracy:
                continue
            
            mode_df = df[
                (df['mode'] == target_mode) &
                (df['metric'] == metric)
            ]
            
            if len(mode_df) == 0:
                continue
            
            for hp_id in unique_hyperparams:
                hp_df = mode_df[mode_df['hyperparam_id'] == hp_id]
                if len(hp_df) == 0:
                    continue
                
                for _, row in hp_df.iterrows():
                    val = row['test_value']
                    if not is_effectively_nan(val):
                        accuracy_y_ranges[metric]['all_y_vals'].append(val * 100)
            
            # Collect from scratch runs
            scratch_df = df[
                (df['mode'] == scratch_mode) & 
                (df['metric'] == metric)
            ]
            if len(scratch_df) > 0:
                first_hp = scratch_df['hyperparam_id'].iloc[0]
                scratch_df = scratch_df[scratch_df['hyperparam_id'] == first_hp]
                for _, row in scratch_df.iterrows():
                    val = row['test_value']
                    if not is_effectively_nan(val):
                        accuracy_y_ranges[metric]['all_y_vals'].append(val * 100)
    
    # Compute synchronized y-limits
    for metric in accuracy_y_ranges:
        all_vals = accuracy_y_ranges[metric]['all_y_vals']
        if len(all_vals) > 0:
            max_val = max(all_vals)
            min_val = min(all_vals)
            y_max = min(105, max_val + 10)
            y_min = max(0, min_val - 5)
        else:
            y_max = 105
            y_min = 0
        accuracy_y_ranges[metric]['y_min'] = y_min
        accuracy_y_ranges[metric]['y_max'] = y_max
    
    # Second pass: actual plotting
    for target_mode, scratch_mode, row_idx, row_title in mode_rows:
        for metric_idx, metric in enumerate(metrics):
            ax = axes[row_idx, metric_idx]
            
            metric_config = METRIC_CONFIGS.get(metric, {})
            metric_type = metric_config.get('type', 'mae')
            is_accuracy = (metric_type == 'accuracy')
            
            mode_df = df[
                (df['mode'] == target_mode) &
                (df['metric'] == metric)
            ]
            
            if len(mode_df) == 0:
                ax.text(0.5, 0.5, f'No data for {target_mode}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=15
                )
                continue
            
            # Prepare data for bar plot
            bar_data = []
            bar_labels = []
            bar_errors = []
            bar_colors = []
            
            # Add bars for each hyperparam configuration
            for hp_id in unique_hyperparams:
                hp_df = mode_df[mode_df['hyperparam_id'] == hp_id]
                if len(hp_df) == 0:
                    continue
                
                # Get the single row (or average if multiple)
                if is_accuracy:
                    val = hp_df['test_value'].mean()
                    val_std = hp_df.get('test_value_std', pd.Series([0.0])).mean()
                    if not is_effectively_nan(val):
                        bar_data.append(val * 100)
                        bar_errors.append(val_std * 100 if not is_effectively_nan(val_std) else 0.0)
                        bar_labels.append(hp_id)
                        bar_colors.append(hyperparam_colors[hp_id])
                else:
                    val = hp_df['test_improvement'].mean()
                    val_std = hp_df.get('test_improvement_std', pd.Series([0.0])).mean()
                    if not is_effectively_nan(val):
                        bar_data.append(np.clip(val, -100, 100))
                        bar_errors.append(val_std if not is_effectively_nan(val_std) else 0.0)
                        bar_labels.append(hp_id)
                        bar_colors.append(hyperparam_colors[hp_id])
            
            # Add from-scratch bar
            scratch_df = df[
                (df['mode'] == scratch_mode) & 
                (df['metric'] == metric)
            ]
            if len(scratch_df) > 0:
                first_hp = scratch_df['hyperparam_id'].iloc[0]
                scratch_df = scratch_df[scratch_df['hyperparam_id'] == first_hp]
                
                if is_accuracy:
                    val = scratch_df['test_value'].mean()
                    val_std = scratch_df.get('test_value_std', pd.Series([0.0])).mean()
                    if not is_effectively_nan(val):
                        bar_data.append(val * 100)
                        bar_errors.append(val_std * 100 if not is_effectively_nan(val_std) else 0.0)
                        bar_labels.append('From Scratch')
                        bar_colors.append('black')
                else:
                    val = scratch_df['test_improvement'].mean()
                    val_std = scratch_df.get('test_improvement_std', pd.Series([0.0])).mean()
                    if not is_effectively_nan(val):
                        bar_data.append(np.clip(val, -100, 100))
                        bar_errors.append(val_std if not is_effectively_nan(val_std) else 0.0)
                        bar_labels.append('From Scratch')
                        bar_colors.append('black')
            
            # Create bar plot
            if len(bar_data) > 0:
                x_pos = np.arange(len(bar_data))
                bars = ax.bar(x_pos, bar_data, yerr=bar_errors, 
                             color=bar_colors, alpha=0.85,
                             capsize=5, error_kw={'linewidth': 2, 'capthick': 2},
                             label=bar_labels)
                
                # Add hatching to from-scratch bar
                if 'From Scratch' in bar_labels:
                    scratch_idx = bar_labels.index('From Scratch')
                    bars[scratch_idx].set_hatch('///')
                
                # Remove x-axis labels
                ax.set_xticks(x_pos)
                ax.set_xticklabels([])
                
                # Add legend only to top-right subplot
                if row_idx == 0 and metric_idx == len(metrics) - 1:
                    # Create custom legend handles
                    from matplotlib.patches import Patch
                    legend_handles = []
                    for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
                        if label == 'From Scratch':
                            # Add hatching to scratch legend
                            patch = Patch(facecolor=color, edgecolor='black', label=label, hatch='///', alpha=0.85)
                        else:
                            patch = Patch(facecolor=color, edgecolor='black', label=label, alpha=0.85)
                        legend_handles.append(patch)
                    
                    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), 
                             fontsize=10, frameon=True, fancybox=True, shadow=True)
            
            # Labels and formatting - no x-axis label for bar plots
            ax.set_xlabel('')
            
            if metric_idx == 0:
                ax.set_ylabel('Test Improvement\nover baseline (%)', fontsize=14)
            elif is_accuracy and metric_idx > 0:
                ax.set_ylabel('Test Accuracy (%)', fontsize=14)
            else:
                ax.set_ylabel('')
            
            if row_idx == 0:
                metric_display = metric.replace("_", " ").capitalize()
                if metric == "community_detection":
                    metric_display += " (node)"
                
                ax.text(0.5, 1.15, metric_display, 
                       transform=ax.transAxes, ha='center', va='bottom',
                       fontweight='bold', fontsize=16)
                ax.text(0.5, 1.02, row_title,
                       transform=ax.transAxes, ha='center', va='bottom',
                       fontweight='normal', fontsize=14)
                ax.set_title('')
            else:
                ax.set_title(f'{row_title}', pad=8, fontweight='normal', fontsize=14)
            
            # Y-axis limits and styling
            if is_accuracy:
                if metric in accuracy_y_ranges:
                    y_min = accuracy_y_ranges[metric]['y_min']
                    y_max = accuracy_y_ranges[metric]['y_max']
                else:
                    y_max = 105
                    y_min = 0
                
                ax.set_ylim(y_min, y_max)
                if y_max - y_min <= 60:
                    tick_spacing = 10
                else:
                    tick_spacing = 25
                ax.set_yticks(np.arange(y_min, y_max + 1, tick_spacing))
                
                if y_max > 75:
                    ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(50, 75, facecolor='lightyellow', alpha=0.08, zorder=0)
                    ax.axhspan(75, y_max, facecolor='palegreen', alpha=0.08, zorder=0)
                elif y_max > 50:
                    ax.axhspan(max(y_min, 0), 50, facecolor='lightcoral', alpha=0.08, zorder=0)
                    ax.axhspan(50, y_max, facecolor='lightyellow', alpha=0.08, zorder=0)
                else:
                    ax.axhspan(max(y_min, 0), y_max, facecolor='lightcoral', alpha=0.08, zorder=0)
            else:
                ax.axhline(y=0, color='grey', linestyle='-', linewidth=1.1, alpha=0.65)
                ax.set_ylim(-110, 110)
                ax.set_yticks(np.arange(-100, 125, 50))
                ax.axhspan(-110, 0, facecolor='lightgray', alpha=0.08, zorder=0)
                ax.axhspan(0, 110, facecolor='palegreen', alpha=0.06, zorder=0)
            
            ax.grid(axis='y', alpha=0.20, zorder=0, linestyle=':', linewidth=1.0)
    
    # Add extra spacing before community_detection column if present
    if has_community_detection and len(metrics) > 1:
        extra_gap = 0.04
        for row_idx in range(2):
            for col_idx in range(n_metrics):
                ax = axes[row_idx, col_idx]
                pos = ax.get_position()
                if col_idx < n_metrics - 1:
                    ax.set_position([pos.x0 - extra_gap * 0.5, pos.y0, pos.width, pos.height])
                else:
                    ax.set_position([pos.x0 + extra_gap * 0.5, pos.y0, pos.width, pos.height])
    
    # Overall title
    model_display = model_type.replace('gps_', '').upper()
    n_train = df['n_train'].iloc[0]
    title_text = f"{model_display}: {task_type.replace('_', ' ').title()} (Pretrained on {int(n_graphs)} graphs, n_train={int(n_train)})"
    fig.suptitle(title_text, fontweight='bold', fontsize=22, y=0.995)
    
    plt.tight_layout(rect=[0, 0.05, 0.98, 0.97])
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter comparison for pretraining models")
    parser.add_argument('--wandb_project', type=str, required=True, help='Wandb project name')
    parser.add_argument('--output', type=str, default='hyperparam_comp.png', help='Output filename')
    parser.add_argument('--save_csv', action='store_true', help='Save processed dataframe to CSV')
    
    args = parser.parse_args()
    
    # Fetch runs
    run_data = fetch_runs_from_wandb(args.wandb_project)
    
    if len(run_data) == 0:
        print("ERROR: No runs found in project")
        return
    
    # Process runs
    df = process_runs(run_data)
    
    # Select best readout for each configuration
    best_df = select_best_readout(df)
    
    # Save to CSV if requested
    if args.save_csv:
        csv_path = args.output.replace('.png', '.csv')
        best_df.to_csv(csv_path, index=False)
        print(f"\n✓ Dataframe saved to {csv_path}")
    
    # Create plots
    create_hyperparam_comparison_plots(best_df, args.output, wandb_project=args.wandb_project)
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")



if __name__ == "__main__":
    main()

