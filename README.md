# AQSE 3-Class Classification Workflow (AQSE 3c)

[![codecov](https://codecov.io/gh/<OWNER>/<REPO>/branch/main/graph/badge.svg)](https://codecov.io/gh/<OWNER>/<REPO>)

## Overview

The AQSE 3-Class Classification Workflow trains QSAR models for avoidome proteins using a 3-class activity classification (High/Medium/Low) with similarity-based data expansion. The workflow supports two model architectures: Random Forest and Chemprop.

**Note:** The codebase uses **Polars** as the primary data manipulation library (replacing pandas) for improved performance. Pandas is kept as a dependency only for sklearn compatibility (e.g., `train_test_split`).

### Output Organization

Outputs are organized into step-specific subdirectories:
- **01_input_preparation/**: FASTA files, sequences CSV, summary
- **02_similarity_search/**: Similarity results, BLAST database, plots
- **03_esmc_embeddings/**: Cached ESM-C embeddings
- **04_bioactivity_threshold_optimization/**: Class imbalance analyses, threshold optimization
- **05_model_training/**: Model training results, reports, metrics

### Architecture

Modular structure organized into:
- **`aqse_modelling/data_loaders/`**: Data loaders (Avoidome, similarity, sequences, bioactivity, thresholds)
- **`aqse_modelling/models/`**: Model trainers (Random Forest, Chemprop, hyperparameter optimization)
- **`aqse_modelling/utils/`**: Utilities (feature extraction, MLflow, data splitting, config loading)
- **`aqse_modelling/reporting/`**: Model reporting and metrics generation

## Prerequisites

- **Python 3.12** (required for UV workflow)
- **[UV](https://docs.astral.sh/uv/)** package manager
- **BLAST+ tools** (for Step 2)
- **Internet connection** (for UniProt API and Papyrus download)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd AQSE_v3

# Install dependencies with UV
uv sync
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`. Required packages include: polars, pandas, numpy, Bio (Biopython), papyrus-scripts, scikit-learn, chemprop, matplotlib, seaborn, pyyaml, rdkit, torch, esm, lightning, torchmetrics.

### Running Tests (UV + pytest)

After syncing dependencies, you can run the test suite from the `AQSE_v3` directory:

```bash
cd AQSE_v3

# Run tests
uv run pytest

# Run tests with coverage (no threshold enforcement)
uv run pytest --cov=aqse_modelling --cov-report=term-missing --cov-report=xml
```

## Workflow Steps

The workflow consists of 5 sequential steps:

1. **Input Preparation** (`01_input_preparation.py`) - Fetches and prepares protein sequences
2. **Similarity Search** (`02_protein_similarity_search_papyrus_blast.py`) - Identifies similar proteins in Papyrus database
3. **ESM-C Descriptor Calculation** (`03_calculate_esmc_embeddings.py`) - Computes protein embeddings (optional if cached)
4. **Bioactivity Threshold Optimization** (`04_bioactivity_threshold_optimization.py`) - Analyzes class imbalances and optimizes activity thresholds
5. **Model Training** (`05_AQSE_3c_2c_clf_th_op_parameter_optimization.py`) - Trains QSAR models using the refactored modular workflow

---

## Complete Workflow Execution (UV - Recommended)

A single UV environment runs every step (no environment switching required).

```bash
# One-time setup
cd AQSE_v3
uv sync

# Step 1: Input preparation
uv run python scripts/01_input_preparation.py

# Step 2: Similarity search
uv run python scripts/02_protein_similarity_search_papyrus_blast.py

# Step 3: ESM-C descriptors (optional, auto-calculated if missing)
uv run python scripts/03_calculate_esmc_embeddings.py

# Step 4: Bioactivity threshold optimization (optional)
uv run python scripts/04_bioactivity_threshold_optimization.py

# Step 5: Model training
uv run python scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py 
```

---

## Step 1: Input Preparation

### Purpose
Fetches protein sequences from UniProt and prepares FASTA files for similarity search.

### Script
`scripts/01_input_preparation.py`

### Inputs
- **data/avoidome_prot_list.csv**: CSV file with avoidome proteins containing:
  - `Name_2`: Protein name
  - `UniProt ID`: UniProt identifier
  - `ChEMBL_target`: ChEMBL target identifier

### Outputs
All outputs are organized in the `01_input_preparation/` directory:
- **01_input_preparation/fasta_files/**: Directory containing individual FASTA files for each protein
  - Format: `{uniprot_id}_{protein_name}.fasta`
- **01_input_preparation/avoidome_proteins_combined.fasta**: Combined FASTA file with all sequences
- **01_input_preparation/avoidome_sequences.csv**: CSV file with sequence information
  - Columns: `uniprot_id`, `protein_name`, `sequence`, `sequence_length`
- **01_input_preparation/input_preparation_summary.txt**: Summary of preparation process
- **01_input_preparation/logs/**: Directory with processing logs

### Usage
```bash
uv run python scripts/01_input_preparation.py
```

---

## Step 2: Similarity Search

### Purpose
Performs BLAST-based sequence similarity search against Papyrus database to identify similar proteins at three similarity thresholds (high: 70%, medium: 50%, low: 30%).

### Script
`scripts/02_protein_similarity_search_papyrus_blast.py`

### Inputs
- **01_input_preparation/fasta_files/**: Directory with FASTA files from Step 1
- **Papyrus database** (version 05.7): Automatically loaded via papyrus-scripts library
- **BLAST**: Requires BLAST+ tools installed

### Outputs
- **02_similarity_search/similarity_search_summary.csv**: Main output file with similarity results
  - Columns: `query_protein`, `threshold`, `similar_proteins`, `n_similar`, `max_identity`, `min_identity`, `avg_identity`
- **02_similarity_search/blast_db/**: BLAST database files
  - `papyrus_proteins.fasta`: FASTA file of Papyrus proteins
  - `papyrus_proteins_id_mapping.txt`: Mapping between BLAST indices and Papyrus IDs
- **02_similarity_search/plots/**: Visualization plots
  - `similar_proteins_count.png`: Bar plot showing number of similar proteins per threshold

### Usage
```bash
uv run python scripts/02_protein_similarity_search_papyrus_blast.py
```

### Requirements
- BLAST+ tools installed and in PATH
- papyrus-scripts library
- Internet connection for initial Papyrus dataset download

**Note**: The script automatically filters out proteins with empty or missing sequences.

---

## Step 3: ESM-C Descriptor Calculation (Optional)

### Purpose
Calculates ESM-C protein embeddings for Model B (PCM models). **Optional** if descriptors are already cached.

### Outputs
Cached in `03_esmc_embeddings/` (or `papyrus_cache_dir` in config.yaml):
- Format: `{uniprot_id}_descriptors.pkl` (960-dimensional embeddings)

### Usage

Descriptors are calculated automatically during Step 5 if missing. To pre-calculate:

```bash
uv run python scripts/03_calculate_esmc_embeddings.py
```

**Options:**
- `--force-recalculate-all`: Recalculate all embeddings (overwrites cache)
- `--config /path/to/config.yaml`: Specify custom config file

By default, the script skips proteins with existing cached embeddings.

---

## Step 4: Bioactivity Threshold Optimization

### Purpose
Analyzes class imbalances and optimizes activity thresholds for 2-class and 3-class classification.

### Script
`scripts/04_bioactivity_threshold_optimization.py`

### Inputs
- **data/avoidome_cutoff_table.csv**: Activity thresholds for each protein
- **02_similarity_search/similarity_search_summary.csv**: Similarity search results (from Step 2)
- **Papyrus database**: Bioactivity data (loaded automatically)

### Outputs
- **04_bioactivity_threshold_optimization/**: Directory containing analysis results
  - `class_imbalance_summary_only_target.csv`: Summary for target proteins only
  - `class_imbalance_summary_with_similar.csv`: Summary including similar proteins
  - `class_imbalance_summary_optimized.csv`: Summary with optimized thresholds
  - `plots/`: Visualization plots for class distributions
  - Analysis reports and statistics

### Usage
```bash
uv run python scripts/04_bioactivity_threshold_optimization.py
```

---

## Step 5: Model Training

### Purpose
Trains QSAR classification models using the AQSE approach with two model types:
- **Model A**: Simple QSAR for proteins without similar proteins
- **Model B**: PCM (Protein-Compound-Model) for proteins with similar proteins, using data expansion at three thresholds (high/medium/low)

### Script
`scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py`

Orchestrates the workflow using modular components from `aqse_modelling/`.

### Inputs
- **config.yaml**: Configuration file with paths, model type, and hyperparameters
- **data/avoidome_cutoff_table.csv**: Activity thresholds (`name2_entry`, `uniprot_id`, `cutoff_high`, `cutoff_medium`)
- **Papyrus database**: Bioactivity data (Papyrus++ default, standard Papyrus available via config)

### Outputs

All outputs saved to **05_model_training/** directory:

**Main Results:**
- `workflow_results.csv`: Summary of all training attempts
- `results_{model_type}/comprehensive_protein_report.csv`: Comprehensive report
- `results_{model_type}/model_summary_report.json`: Aggregated statistics

**Per-Model Outputs** (in `results_{model_type}/`):
- `models/`: Trained models (`{protein}_{uniprot_id}_{model_type}_{threshold}_model.pkl.gz`)
- `metrics/`: Performance metrics (accuracy, F1-macro, F1-weighted, precision, recall, confusion matrix)
- `predictions/`: Test set predictions
- `class_distributions/`: Class distribution files
- `individual_reports/`: Per-protein reports

**Additional:**
- `fingerprints/`: Cached Morgan fingerprints
- `protein_list_all_unique.csv`: All unique proteins
- Summary tables (bioactivity_table.csv, bioactivity_table_with_similar.csv, protein_summary_table.csv)

Disable table generation with `generate_summary_tables: false` in config.yaml.

### Usage

```bash
uv run python scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py [--config config.yaml]
```

The `--config` argument is optional; defaults to `config.yaml` in project root.

### Model Types

**Random Forest**: Precomputed features (Morgan fingerprints + physicochemical descriptors). Model B includes ESM-C embeddings.

**Chemprop**: Features extracted on-the-fly from SMILES. Model B includes ESM-C embeddings automatically.

### Model Status Codes
- `success`: Model trained successfully
- `insufficient_data`: Less than 30 bioactivity samples
- `insufficient_features`: Insufficient samples after feature extraction
- `no_similar_proteins`: No similar proteins found at threshold
- `no_thresholds`: No activity thresholds defined

### Papyrus Dataset Selection

Supports **Papyrus++** (default) or **standard Papyrus** per protein via `proteins_use_standard_papyrus` in config:

```yaml
proteins_use_standard_papyrus:
  - "SLCO1B1"  # Protein name or UniProt ID
  - "P59520"
```

---

## Legacy: Conda / Micromamba Workflow (Deprecated)

**Note:** The project now uses UV as the primary package manager. The conda/micromamba workflow is deprecated but kept for reference.

If you need to use conda/micromamba instead of UV:

```bash
# Step 1: Input preparation
conda activate aqse_base
python scripts/01_input_preparation.py

# Step 2: Similarity search
conda activate aqse_base
python scripts/02_protein_similarity_search_papyrus_blast.py

# Step 3: ESM-C descriptors (optional, auto-calculated if missing)
conda activate esmc
python scripts/03_calculate_esmc_embeddings.py

# Step 4: Bioactivity threshold optimization (optional)
python scripts/04_bioactivity_threshold_optimization.py

# Step 5: Model training
micromamba activate chemprop_env
python scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py --config config.yaml
```

**Prerequisites for conda/micromamba:**
- Python 3.9–3.12
- Conda or Micromamba installed
- Environment files: `environment.yml`, `esmc_environment.yml`, `chemprop_environment.yml` (deprecated, not maintained)
