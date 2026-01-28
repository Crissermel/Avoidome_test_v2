# AQSE 3-Class Classification Workflow (AQSE 3c)

## Overview

The AQSE 3-Class Classification Workflow trains QSAR models for avoidome proteins using a 3-class activity classification (High/Medium/Low) with similarity-based data expansion. The workflow supports two model architectures: Random Forest and Chemprop.

### Output Organization

All workflow outputs are organized into step-specific subdirectories for better organization:
- **01_input_preparation/**: Step 1 outputs (FASTA files, sequences CSV, summary)
- **02_similarity_search/**: Step 2 outputs (similarity results, BLAST database, plots)
- **03_esmc_embeddings/**: Step 3 outputs (cached ESM-C embeddings)
- **04_bioactivity_threshold_optimization/**: Step 4 outputs (class imbalance analyses, threshold optimization results)
- **05_model_training/**: Step 5 outputs (model training results, reports, metrics)

This organization makes it easy to identify which files belong to which step and simplifies cleanup or rerunning specific steps.

### Architecture

The codebase has been refactored into a modular structure for improved maintainability and extensibility:

- **Data Loaders** (`aqse_modelling/data_loaders/`): Specialized loaders for different data types
  - `AvoidomeDataLoader`: Loads avoidome target and threshold data
  - `SimilarityDataLoader`: Loads protein similarity search results
  - `ProteinSequenceLoader`: Loads protein sequences
  - `BioactivityDataLoader`: Loads Papyrus bioactivity data (supports both Papyrus++ and standard Papyrus)
  - `ActivityThresholdsLoader`: Loads activity classification thresholds
  - `UniqueProteinListGenerator`: Generates unique protein lists for fingerprint calculation

- **Models** (`aqse_modelling/models/`): Model training and optimization components
  - `RandomForestTrainer`: Random Forest model implementation
  - `ChempropTrainer`: Chemprop neural network model implementation
  - `ChempropHyperparameterOptimizer`: Hyperparameter optimization for Chemprop
  - `ModelTrainer`: Abstract base class for model trainers

- **Utils** (`aqse_modelling/utils/`): Utility functions and helpers
  - `feature_extraction.py`: Compound and protein feature extraction
  - `mlflow_logger.py`: MLflow experiment tracking
  - `data_splitting.py`: Stratified data splitting utilities
  - `config_loader.py`: Configuration loading and path resolution

- **Workflow** (`aqse_modelling/workflow/`): Workflow orchestration components
  - `workflow_orchestrator.py`: Main workflow logic (used by the orchestration script)
  - `data_preparation.py`: Data preparation and feature dataset creation
  - `model_trainer_wrapper.py`: Model training wrapper with reporting
  - `main.py`: Legacy module entry point (orchestration is now in the main script)

- **Reporting** (`aqse_modelling/reporting/`): Model evaluation and reporting
  - `model_reporter.py`: Model metrics calculation and report generation

## Workflow Steps

The workflow consists of 5 sequential steps:

1. **Input Preparation** (`01_input_preparation.py`) - Fetches and prepares protein sequences
2. **Similarity Search** (`02_protein_similarity_search_papyrus_blast.py`) - Identifies similar proteins in Papyrus database
3. **ESM-C Descriptor Calculation** (`03_calculate_esmc_embeddings.py`) - Computes protein embeddings (optional if cached)
4. **Bioactivity Threshold Optimization** (`04_bioactivity_threshold_optimization.py`) - Analyzes class imbalances and optimizes activity thresholds
5. **Model Training** (`05_AQSE_3c_2c_clf_th_op_parameter_optimization.py`) - Trains QSAR models using the refactored modular workflow

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
python scripts/01_input_preparation.py
```

Or programmatically:
```python
from scripts.input_preparation import AvoidomeInputPreparation

preparer = AvoidomeInputPreparation(
    avoidome_file="/path/to/avoidome_prot_list.csv",
    output_dir="/path/to/output"
)
preparer.prepare_inputs()
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
```python
python scripts/02_protein_similarity_search_papyrus_blast.py
```

Or programmatically:
```python
from scripts.protein_similarity_search_papyrus_blast import ProteinSimilaritySearchPapyrus

searcher = ProteinSimilaritySearchPapyrus(
    input_dir="/path/to/fasta_files",
    output_dir="/path/to/02_similarity_search",
    papyrus_version='05.7'
)
searcher.run_similarity_search()
```

### Requirements
- BLAST+ tools installed and in PATH
- papyrus-scripts library
- Internet connection for initial Papyrus dataset download

### Notes
- The script automatically filters out Papyrus proteins with empty or missing sequences before creating the BLAST database
- Proteins with empty sequences are skipped with a warning message
- This ensures only valid protein sequences are used for similarity search

---

## Step 3: ESM-C Descriptor Calculation (Optional)

### Purpose
Calculates ESM-C (Evolutionary Scale Modeling - Contact) protein embeddings for use in Model B (PCM models). This step is **optional** if descriptors are already cached.

### Requirements
- **esmc package**: Required for calculating ESM-C descriptors
**Python environment**: Use the `esmc` conda/environment for this step


### Inputs
- Protein sequences from Step 1
- ESM-C model (handled internally)

### Outputs
ESM-C embeddings are cached in the `03_esmc_embeddings/` directory (or as specified in `papyrus_cache_dir` in config.yaml):
- **03_esmc_embeddings/**: Directory containing cached ESM-C descriptors
  - Format: `{uniprot_id}_descriptors.pkl` for each protein
  - Contains 960-dimensional ESM-C embeddings

### Usage

ESM-C descriptors are calculated automatically during Step 5 if not found in cache. The cache directory is specified in `config.yaml` (defaults to `03_esmc_embeddings/` if not specified):
```yaml
papyrus_cache_dir: "03_esmc_embeddings"  # Relative to project root, or use absolute path
```

If descriptors are pre-calculated and cached, Step 3 can be skipped entirely.

**Note**: When running Step 3, ensure you activate the `esmc` environment if ESM-C descriptors need to be calculated:

#### Normal Mode (Skips Existing Cache)
By default, the script checks if embeddings already exist and skips calculation for cached proteins:
```bash
conda activate esmc
python scripts/03_calculate_esmc_embeddings.py
```

The script will:
- Check for existing `{uniprot_id}_descriptors.pkl` files in the cache directory
- Skip calculation for proteins that already have cached embeddings
- Only calculate embeddings for proteins that are missing from cache
- Report summary: already cached, successfully calculated, failed, and skipped (no sequence)

#### Force Recalculate All
To recalculate all embeddings (useful if embeddings have incorrect dimensions or need to be regenerated):
```bash
conda activate esmc
python scripts/03_calculate_esmc_embeddings.py --force-recalculate-all
```

This will overwrite existing cache files and recalculate embeddings for all proteins.

#### Custom Config File
To specify a custom config file path:
```bash
conda activate esmc
python scripts/03_calculate_esmc_embeddings.py --config /path/to/config.yaml
```

If not specified, the script looks for `config.yaml` in the project root.

---

## Step 4: Bioactivity Threshold Optimization

### Purpose
Analyzes class imbalances in bioactivity data and optimizes activity thresholds for each protein. This step helps determine optimal cutoff values for 2-class and 3-class classification by:
- Analyzing class distributions for target proteins only
- Analyzing class distributions with similar proteins included
- Optimizing thresholds to improve class balance
- Recommending 2-class vs 3-class classification based on imbalance ratios

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
python scripts/04_bioactivity_threshold_optimization.py
```

This step analyzes class distributions and helps optimize activity thresholds before model training. The optimized thresholds can be used to update the cutoff table for improved model performance.

---

## Step 5: Model Training

### Purpose
Trains QSAR classification models using the AQSE approach with two model types:
- **Model A**: Simple QSAR for proteins without similar proteins
- **Model B**: PCM (Protein-Compound-Model) for proteins with similar proteins, using data expansion at three thresholds (high/medium/low)

### Script
The modeling workflow has been refactored into a modular structure. The main orchestration script is:

- **Main Script**: `scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py`
- This script orchestrates the entire workflow by coordinating all modular components
- The original monolithic implementation (4,541 lines) has been refactored into a maintainable modular structure


The codebase is organized into the following modules:
- **`aqse_modelling/data_loaders/`**: Data loading components (Avoidome, similarity, sequences, bioactivity, thresholds)
- **`aqse_modelling/models/`**: Model training components (Random Forest, Chemprop, hyperparameter optimization)
- **`aqse_modelling/utils/`**: Utility functions (feature extraction, visualization, MLflow, data splitting, config loading)
- **`aqse_modelling/workflow/`**: Workflow orchestration components (used by the main script)
- **`aqse_modelling/reporting/`**: Model reporting and metrics generation

**Refactoring Complete**: The original monolithic script has been fully refactored into a maintainable modular structure, with the main script serving as the orchestration entry point.

### Requirements
- **micromamba environment**: The `chemprop_env` micromamba environment is required for Step 5 (both Random Forest and Chemprop models)

### Inputs
- **config.yaml**: Configuration file with all paths and parameters
  - `avoidome_file`: Path to avoidome_prot_list.csv
  - `similarity_file`: Path to similarity_search_summary.csv (from Step 2)
  - `sequence_file`: Path to avoidome_sequences.csv (from Step 1)
  - `activity_thresholds_file`: Path to avoidome_cutoff_table.csv
  - `output_dir`: Output directory for results (automatically set to `05_model_training/` by the script)
  - `papyrus_cache_dir`: Path to ESM-C descriptors cache
  - `model_type`: "random_forest" or "chemprop"
  - Model-specific hyperparameters

- **data/avoidome_cutoff_table.csv**: Activity thresholds for each protein
  - Columns: `name2_entry`, `uniprot_id`, `cutoff_high`, `cutoff_medium`

- **Papyrus database**: Bioactivity data (loaded automatically via papyrus-scripts)
  - **Papyrus++** (default): Enhanced version with additional data
  - **Standard Papyrus**: Available for specific proteins via configuration (see `proteins_use_standard_papyrus` in config)

### Outputs

All outputs are saved to the **05_model_training/** directory (automatically created by the script).

#### Main Results
- **workflow_results.csv**: Summary of all model training attempts
- **results_{model_type}/comprehensive_protein_report.csv**: Comprehensive report with all models and metrics
- **results_{model_type}/model_summary_report.json**: JSON summary with aggregated statistics

#### Per-Model Outputs (in `results_{model_type}/`)
- **models/**: Trained model files
  - Format: `{protein}_{uniprot_id}_{model_type}_{threshold}_model.pkl.gz`
- **metrics/**: Performance metrics (JSON files)
  - Format: `{protein}_{uniprot_id}_{model_type}_{threshold}_{split}_metrics.json`
  - Contains: accuracy, F1-macro, F1-weighted, precision-macro, recall-macro, confusion matrix
- **predictions/**: Test set predictions (CSV files)
  - Format: `{protein}_{uniprot_id}_{model_type}_{threshold}_{split}_predictions.csv`
- **class_distributions/**: Class distribution files (CSV + JSON summary)
  - Format: `{protein}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv`
- **individual_reports/**: Detailed per-protein reports (JSON + CSV)

#### Additional Outputs
- **fingerprints/**: Morgan fingerprints cache
  - `papyrus_morgan_fingerprints.parquet`: Cached fingerprints for all compounds
- **protein_list_all_unique.csv**: List of all unique proteins (avoidome + similar)
- **umap_plots/**: UMAP visualizations (if generated)
- **logs/**: Processing logs

#### Summary Tables (Generated Automatically)
After model training completes, the workflow automatically generates summary tables:
- **bioactivity_table.csv**: Comprehensive bioactivity table for target proteins only
  - Columns: Accession, Inchikey, pChembl_value, activity_threshold (binary), activity_thresholds (3-class), type
- **bioactivity_table_with_similar.csv**: Bioactivity table including target and similar proteins
  - Additional columns: protein_category (target/similar), target_protein
- **protein_summary_table.csv**: Protein-level summary with thresholds, metrics, and statistics
  - Columns: Accession, thresholds, optimized thresholds, binarization, model performance, similar proteins, datapoints

Table generation can be disabled by setting `generate_summary_tables: false` in config.yaml.

### Usage

#### Command Line

```bash
micromamba activate chemprop_env
python scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py --config config.yaml
```

Or if running from the AQSE_v3 directory:
```bash
micromamba activate chemprop_env
python scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py
```

The `--config` argument is optional. If not provided, the script will look for `config.yaml` in the project root.

#### Programmatic Usage

If you need to use the workflow components programmatically:
```python
from aqse_modelling.utils.config_loader import load_config
from aqse_modelling.workflow.workflow_orchestrator import AQSE3CWorkflow

config = load_config("config.yaml")
workflow = AQSE3CWorkflow(config)
workflow.process_all_proteins()
```

The workflow automatically:
1. Loads configuration from `config.yaml`
2. Loads all input data using modular data loaders
3. Processes each protein in the avoidome list
4. Trains Model A (if no similar proteins) or both Model A and Model B (if similar proteins exist)
5. Saves all results and generates reports

### Model Types

#### Random Forest
- Uses precomputed features: Morgan fingerprints + Physicochemical descriptors
- Model B includes ESM-C embeddings
- Configurable parameters: `rf_n_estimators`, `rf_max_depth`, `rf_class_weight`

#### Chemprop
- Extracts features on-the-fly from SMILES
- Model B includes ESM-C embeddings automatically
- Configurable architecture: `chemprop_ffn_num_layers`, `chemprop_hidden_size`, `chemprop_depth`, etc.

### Model Status Codes
- `success`: Model trained successfully
- `insufficient_data`: Less than 30 bioactivity samples
- `insufficient_features`: Insufficient samples after feature extraction
- `no_similar_proteins`: No similar proteins found at threshold
- `no_thresholds`: No activity thresholds defined

### Papyrus Dataset Selection

The workflow supports using either **Papyrus++** (default) or **standard Papyrus** datasets for different proteins. This is controlled via the `proteins_use_standard_papyrus` configuration option:

- **Papyrus++** (default): Enhanced version with additional curated data. Used for all proteins unless specified otherwise.
- **Standard Papyrus**: Original Papyrus dataset. Used for proteins listed in `proteins_use_standard_papyrus`.

**Configuration Example:**
```yaml
proteins_use_standard_papyrus:
  - "SLCO1B1"  # Protein name
  - "P59520"   # Or UniProt ID
```

The system automatically:
- Maps protein names to UniProt IDs when needed
- Lazily loads the standard Papyrus dataset only when required
- Combines data from both sources when processing multiple proteins

For more details, see `PAPYRUS_STANDARD_FEATURE.md`.

---

## Configuration File (config.yaml)

Example configuration:
```yaml
# Input files (relative to config.yaml location)
avoidome_file: "data/avoidome_prot_list.csv"
similarity_file: "02_similarity_search/similarity_search_summary.csv"
sequence_file: "01_input_preparation/avoidome_sequences.csv"
activity_thresholds_file: "data/avoidome_cutoff_table.csv"

# Output directories
output_dir: "/path/to/output"  # Automatically overridden to 05_model_training/ by the script
papyrus_cache_dir: "03_esmc_embeddings"  # For Step 3 ESM-C embeddings (relative to project root)

# Model configuration
model_type: "random_forest"  # or "chemprop"

# RandomForest parameters
rf_n_estimators: 500
rf_max_depth: null  # null means no limit
rf_class_weight: "balanced_subsample"
random_state: 42

# Chemprop parameters
chemprop_max_epochs: 100
n_classes: 3
chemprop_batch_size: 200
chemprop_val_fraction: 0.2
# ... additional architecture parameters

# Papyrus dataset selection
# Proteins that should use the standard Papyrus dataset (not Papyrus++)
# Can use protein name (e.g., "SLCO1B1") or UniProt ID (e.g., "P59520")
# All other proteins will use Papyrus++ by default
proteins_use_standard_papyrus:
  - "SLCO1B1"  # Example: uses standard Papyrus instead of Papyrus++

# MLflow configuration (optional)
mlflow_enabled: true
mlflow_tracking_uri: "file:///path/to/mlruns"

# Hyperparameter optimization (optional, for Chemprop)
enable_parameter_optimization: false
optimization_n_trials: 50
optimization_strategy: "hybrid"  # "hybrid" or "random"

# Summary table generation (optional, enabled by default)
generate_summary_tables: true  # Set to false to skip table generation after workflow
```

---

## Complete Workflow Execution

### Sequential Execution
```bash
# Step 1: Input preparation (base environment)
conda activate aqse_base
python scripts/01_input_preparation.py

# Step 2: Similarity search (base environment)
conda activate aqse_base
python scripts/02_protein_similarity_search_papyrus_blast.py

# Step 3: ESM-C descriptors (optional, auto-calculated if missing)
# Skip if descriptors are pre-cached
# If calculating descriptors, activate esmc environment:
conda activate esmc
python scripts/03_calculate_esmc_embeddings.py

# Step 4: Bioactivity threshold optimization (optional, for threshold analysis)
python scripts/04_bioactivity_threshold_optimization.py

# Step 5: Model training (uses script 05 - chemprop_env required for both Random Forest and Chemprop)
micromamba activate chemprop_env
python scripts/05_AQSE_3c_2c_clf_th_op_parameter_optimization.py --config config.yaml


```

### Prerequisites

#### General Requirements
- Python 3.9-3.12
- Required packages (see `requirements.txt` or `environment.yml`):
  - pandas, numpy
  - Bio (Biopython)
  - papyrus-scripts
  - scikit-learn (for Random Forest)
  - chemprop (for Chemprop models, optional)
  - matplotlib, seaborn (for visualizations)
  - pyyaml
  - rdkit (for molecular descriptors - install via conda-forge)
- BLAST+ tools (for Step 2)
- Internet connection (for UniProt API and Papyrus download)
