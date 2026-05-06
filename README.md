# HOPSE Experiments - Reproduction Instructions

This repository contains the implementation and experiments for our paper submission. This guide will help you set up the environment and reproduce all experimental results.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Quick Start: Running a Single Experiment](#quick-start-running-a-single-experiment)
- [Reproducing Paper Results](#reproducing-paper-results)

---

## Environment Setup

### Prerequisites
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager (recommended for fast, reproducible installs)

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and navigate to the repository**:
   ```bash
   cd TopoBench
   ```

3. **Initialize environment**:
   
   Use the setup script to configure Python 3.11 and hardware-specific dependencies:
   ```bash
   # For CPU-only:
   source uv_env_setup.sh cpu
   
   # For CUDA 11.8:
   source uv_env_setup.sh cu118
   
   # For CUDA 12.1:
   source uv_env_setup.sh cu121
   ```
   
   This script automatically:
   - Creates a `.venv` with Python 3.11
   - Configures PyTorch and PyTorch Geometric for your platform
   - Generates a lock file and syncs all dependencies

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda}')"
   ```

---

## Quick Start: Running a Single Experiment

### Basic Configuration

Experiments are configured through `configs/run.yaml` and model-specific YAML files. Here's how to run a custom experiment:

1. **Select a dataset** in `configs/run.yaml`:
   ```yaml
   defaults:
     - dataset: graph/BBB_Martins  # Choose your dataset
     - model: cell/hopse_m          # Choose your model
     # ... other configs
   ```

2. **Choose a model**:
   - `cell/hopse_g` - HOPSE-GPSE (Cell domain)
   - `simplicial/hopse_g` - HOPSE-GPSE (Simplicial domain)
   - `cell/hopse_m` - HOPSE-M (Cell domain)
   - `simplicial/hopse_m` - HOPSE-M (Simplicial domain)

3. **Configure neighborhoods** in the model config file (e.g., `configs/model/cell/hopse_m.yaml`):
   ```yaml
   preprocessing_params:
     neighborhoods: 
       - 'up_adjacency-1'
       # - 'down_adjacency-1'
       # - 'up_incidence-1'
       # Uncomment desired neighborhoods
     encodings: ["RWSE"]  # Options: KHopFE, HKFE, LapPE, RWSE, ElectrostaticPE, HKdiagSE
   ```

4. **Adjust encoding hyperparameters** in `configs/transforms/data_manipulations/hopse_ps_information.yaml`

5. **Run the experiment**:
   ```bash
   python -m topobench
   ```

### Command-Line Overrides

You can also override configurations directly from the command line:
```bash
python -m topobench model=cell/hopse_m dataset=graph/MUTAG
```

---

## Reproducing Paper Results

All experiments from the paper can be reproduced using the provided shell scripts.

### 1. Main Hyperparameter Sweep

Run the full hyperparameter grid search for each model included in the paper:

```bash
# Navigate to sweep scripts directory
cd reproduce_hopse/main_hyperparameter_sweep

# Run sweep for a specific model (e.g., GCN)
bash gcn.sh

```
Each of the models in the paper have a script under reproduce_hopse/main_hyperparameter_sweep

Each script will:
- Generate all parameter combinations
- Automatically distribute jobs across available GPUs
- Log results to `./logs/<model_name>/`
- Report progress every 1% of completion

**Note**: The scripts auto-detect GPU configuration and adjust concurrency based on available VRAM.

### 2. Encoding Ablation Study

Test individual encoding contributions:

```bash
cd reproduce_hopse/ablations
bash individual_encoding_ablation.sh
```

### 3. Timing Experiments

#### Re-run Best Models (for runtime measurement)
```bash
cd reproduce_hopse/ablations
bash rerun_best_models.sh
```

#### Preprocessing Time Measurement
```bash
cd reproduce_hopse/ablations
bash preprocessing_only_timing.sh
```

---

## Experiment Monitoring

Results are logged to Weights & Biases (W&B). Each script creates a separate W&B project based on the dataset name for organized tracking.

To view logs locally without W&B, check the `./logs/` directory created by each script.

## File Structure

```
reproduce_hopse/
├── main_hyperparameter_sweep/    # Full grid search scripts
│   ├── gcn.sh
│   ├── hopse_g_cell.sh
│   ├── hopse_g_simplicial.sh
│   ├── hopse_m_cell.sh
│   └── hopse_m_simplicial.sh
└── ablations/                     # Ablation study scripts
    ├── individual_encoding_ablation.sh
    ├── rerun_best_models.sh
    └── preprocessing_only_timing.sh

configs/
├── run.yaml                       # Main configuration entry point
├── dataset
│   ├── simplicial
│   │   ├── mantra_name.yaml       # Dataset configurations
├── model/                         # Model configurations
│   ├── cell/
│   │   ├── hopse_g.yaml
│   │   └── hopse_m.yaml
│   └── simplicial/
│       ├── hopse_g.yaml
│       └── hopse_m.yaml
└── transforms/                    # Transform and encoding configs
    └── data_manipulations/
        └── hopse_ps_information.yaml
```