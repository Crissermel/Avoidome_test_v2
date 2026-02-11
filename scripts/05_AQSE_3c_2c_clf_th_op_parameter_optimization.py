"""
AQSE 3-Class Classification Workflow - Main Orchestration Script

This script orchestrates the entire AQSE 3-Class Classification Workflow.

All workflow logic is contained in this single script for maximum linearity and clarity.

Requires: 
    micromamba activate chemprop_env

# Initialize micromamba in your current shell: 
    eval "$(micromamba shell hook --shell bash)"
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# MLflow imports (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add parent directory to path to allow importing aqse_modelling
# This ensures the script works when run from the scripts/ directory
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from aqse_modelling.utils.config_loader import load_config, resolve_config_path
    from aqse_modelling.data_loaders import (
        AvoidomeDataLoader,
        SimilarityDataLoader,
        ProteinSequenceLoader,
        BioactivityDataLoader,
        ActivityThresholdsLoader,
        UniqueProteinListGenerator
    )
    from aqse_modelling.models.base import ModelTrainer, create_model_trainer
    from aqse_modelling.models.hyperparameter_optimizer import ChempropHyperparameterOptimizer
    from aqse_modelling.reporting.model_reporter import ModelReporter
    from aqse_modelling.utils.mlflow_logger import setup_mlflow, log_to_mlflow
    from aqse_modelling.utils.feature_extraction import (
        extract_compound_features,
        extract_protein_features,
        create_feature_dataset,
        extract_multi_protein_features,
        assign_class_labels
    )
    from aqse_modelling.utils.data_splitting import split_data_stratified
except ImportError as e:
    print(f"Error: Could not import aqse_modelling package: {e}")
    print(f"Make sure you're running from the AQSE_v3 directory or that aqse_modelling is in your Python path.")
    sys.exit(1)

# Import table generation modules from aqse_modelling.reporting
try:
    from aqse_modelling.reporting import bioactivity_table_with_similar as create_bioactivity_table_with_similar_module
    from aqse_modelling.reporting import protein_summary_table as create_protein_summary_table_module
    # Try to import the basic bioactivity table module (may not exist yet)
    try:
        from aqse_modelling.reporting import bioactivity_table as create_bioactivity_table_module
    except ImportError:
        create_bioactivity_table_module = None
except ImportError as e:
    # Logging not set up yet, use print
    print(f"Warning: Could not import table generation modules from aqse_modelling.reporting: {e}")
    create_bioactivity_table_module = None
    create_bioactivity_table_with_similar_module = None
    create_protein_summary_table_module = None


# ============================================================================
# Helper Functions - Data Preparation
# ============================================================================

def prepare_model_a_data(
    bioactivity_data: pd.DataFrame,
    uniprot_id: str,
    thresholds: Dict[str, Any],
    model_trainer,
    cached_fingerprints: Dict[str, np.ndarray],
    bioactivity_loader,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], int]:
    """
    Prepare data for Model A training.
    
    Args:
        bioactivity_data: Raw bioactivity DataFrame
        uniprot_id: UniProt ID
        thresholds: Activity thresholds dictionary
        model_trainer: Model trainer instance
        cached_fingerprints: Dictionary mapping SMILES to Morgan fingerprints
        bioactivity_loader: BioactivityDataLoader instance
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (train_df, val_df, test_df, total_samples)
        val_df will be None if not using fixed validation set
    """
    # Filter for valid data - check if columns exist first
    required_cols = ['SMILES', 'pchembl_value_Mean']
    missing_cols = [col for col in required_cols if col not in bioactivity_data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in bioactivity_data: {missing_cols}")
    
    bioactivity_data = bioactivity_data.dropna(subset=required_cols)
    bioactivity_data = bioactivity_data[bioactivity_data['SMILES'] != '']
    
    # Assign class labels
    bioactivity_data = assign_class_labels(bioactivity_data, thresholds)
    
    # Extract features (conditional based on trainer requirements)
    if model_trainer.requires_precomputed_features():
        # Model A: Extract features (no ESM-C)
        logger.info("Extracting features (Morgan + Physicochemical)...")
        features_df = create_feature_dataset(
            bioactivity_data, 
            uniprot_id, 
            cached_fingerprints,
            bioactivity_loader,
            add_esmc=False
        )
        
        if len(features_df) < 30:
            logger.warning(f"Insufficient data after feature extraction: {len(features_df)} samples")
            return None, None, None, len(features_df)
        
        # Split data: 80% train, 20% test (and optionally validation)
        use_optimization = config.get('enable_parameter_optimization', False)
        use_fixed_val = use_optimization and 'doc_id' in features_df.columns
        
        if use_fixed_val:
            train_df, val_df, test_df = split_data_stratified(
                features_df, 
                test_size=0.2,
                use_fixed_test=True,
                doc_id_column='doc_id'
            )
        else:
            train_df, test_df, _ = split_data_stratified(
                features_df, 
                test_size=0.2,
                use_fixed_test=False
            )
            val_df = None
        total_samples = len(features_df)
    else:
        # For trainers that don't need precomputed features (e.g., Chemprop)
        logger.info("Preparing data for Chemprop (features extracted on-the-fly)...")
        
        # Ensure we have the required columns for Chemprop
        columns_to_keep = ['SMILES', 'class']
        if 'doc_id' in bioactivity_data.columns:
            columns_to_keep.append('doc_id')
        if 'accession' in bioactivity_data.columns:
            columns_to_keep.append('accession')
        bioactivity_data_clean = bioactivity_data[columns_to_keep].copy()
        
        # Split data: 80% train, 20% test with stratification (and optionally validation)
        use_optimization = config.get('enable_parameter_optimization', False)
        use_fixed_val = use_optimization and 'doc_id' in bioactivity_data_clean.columns
        
        if use_fixed_val:
            train_df, val_df, test_df = split_data_stratified(
                bioactivity_data_clean, 
                test_size=0.2,
                use_fixed_test=True,
                doc_id_column='doc_id'
            )
        else:
            train_df, test_df, _ = split_data_stratified(
                bioactivity_data_clean, 
                test_size=0.2,
                use_fixed_test=False
            )
            val_df = None
        total_samples = len(bioactivity_data_clean)
    
    return train_df, val_df, test_df, total_samples


def prepare_model_b_data(
    similar_data: pd.DataFrame,
    target_data: pd.DataFrame,
    uniprot_id: str,
    thresholds: Dict[str, Any],
    model_trainer,
    cached_fingerprints: Dict[str, np.ndarray],
    bioactivity_loader,
    logger: logging.Logger
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Prepare data for Model B training.
    
    Args:
        similar_data: Bioactivity data from similar proteins
        target_data: Bioactivity data from target protein
        uniprot_id: UniProt ID of target protein
        thresholds: Activity thresholds dictionary
        model_trainer: Model trainer instance
        cached_fingerprints: Dictionary mapping SMILES to Morgan fingerprints
        bioactivity_loader: BioactivityDataLoader instance
        logger: Logger instance
        
    Returns:
        Tuple of (train_features, test_features)
    """
    # Check if DataFrames are empty first
    if len(similar_data) == 0 or len(target_data) == 0:
        logger.warning(f"Empty data: similar_data={len(similar_data)}, target_data={len(target_data)}")
        return None, None
    
    # Filter for valid data - check if columns exist
    required_cols = ['SMILES', 'pchembl_value_Mean']
    missing_cols = [col for col in required_cols if col not in similar_data.columns]
    if missing_cols:
        logger.warning(f"Missing required columns in similar_data: {missing_cols}")
        return None, None
    missing_cols = [col for col in required_cols if col not in target_data.columns]
    if missing_cols:
        logger.warning(f"Missing required columns in target_data: {missing_cols}")
        return None, None
    
    similar_data = similar_data.dropna(subset=required_cols)
    target_data = target_data.dropna(subset=required_cols)
    similar_data = similar_data[similar_data['SMILES'] != '']
    target_data = target_data[target_data['SMILES'] != '']
    
    # Check again after filtering
    if len(similar_data) == 0 or len(target_data) == 0:
        logger.warning(f"Empty data after filtering: similar_data={len(similar_data)}, target_data={len(target_data)}")
        return None, None
    
    # Assign class labels to both datasets
    similar_data = assign_class_labels(similar_data, thresholds)
    target_data = assign_class_labels(target_data, thresholds)
    
    # Split target data: 80% train, 20% test
    # split_data_stratified returns (train_df, test_df, None) - unpack all 3 values
    target_train, target_test, _ = split_data_stratified(target_data, test_size=0.2)
    
    # Combine training data: 100% similar + 80% target
    train_data = pd.concat([similar_data, target_train], ignore_index=True)
    
    # Extract features (conditional based on trainer requirements)
    if model_trainer.requires_precomputed_features():
        # Model B with Random Forest: Extract features with ESM-C
        logger.info("Extracting features (Morgan + Physicochemical + ESM-C)...")
        
        # For similar proteins - need to handle multiple proteins
        train_features = extract_multi_protein_features(
            train_data, 
            train_data['accession'].unique(),
            cached_fingerprints,
            bioactivity_loader,
            add_esmc=True
        )
        
        test_features = extract_multi_protein_features(
            target_test,
            [uniprot_id],
            cached_fingerprints,
            bioactivity_loader,
            add_esmc=True
        )
    else:
        # Model B with Chemprop: Prepare data for on-the-fly feature extraction
        logger.info("Preparing data for Chemprop (features extracted on-the-fly with ESM-C)...")
        
        # Ensure we have required columns: 'SMILES', 'class', 'accession'
        train_features = train_data[['SMILES', 'class', 'accession']].copy()
        test_features = target_test[['SMILES', 'class', 'accession']].copy()
        
        # Ensure test set has correct accession (target protein)
        test_features['accession'] = uniprot_id
    
    return train_features, test_features


# ============================================================================
# Helper Functions - Model Training
# ============================================================================

def train_model_with_reporting(
    model_trainer: ModelTrainer,
    model_reporter: ModelReporter,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = 'A',
    protein_name: Optional[str] = None,
    threshold: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    mlflow_enabled: bool = False,
    mlflow_experiment_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    cached_fingerprints: Optional[Dict[str, np.ndarray]] = None,
    bioactivity_loader=None,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Train model using modular trainer with comprehensive reporting and MLflow tracking
    
    Args:
        model_trainer: Model trainer instance
        model_reporter: Model reporter instance
        train_df: Training DataFrame
        test_df: Test DataFrame
        model_type: 'A' or 'B'
        protein_name: Optional protein name for logging
        threshold: Optional threshold for Model B
        uniprot_id: UniProt ID
        thresholds: Dictionary with thresholds, n_classes, and class_labels
        mlflow_enabled: Whether MLflow is enabled
        mlflow_experiment_name: MLflow experiment name
        output_dir: Output directory for visualizations
        cached_fingerprints: Dictionary mapping SMILES to Morgan fingerprints
        bioactivity_loader: BioactivityDataLoader instance
        logger: Logger instance
        
    Returns:
        Dictionary with training results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Prepare MLflow run name and tags
    if not uniprot_id:
        uniprot_id = protein_name if protein_name else "unknown"
    
    run_name = f"{protein_name}_{uniprot_id}_{model_type}"
    if threshold:
        run_name += f"_{threshold}"
    
    # Start MLflow run if enabled
    mlflow_run_active = False
    if mlflow_enabled and MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow.start_run(run_name=run_name)
            mlflow_run_active = True
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}. Continuing without MLflow.")
            mlflow_run_active = False
    
    try:
        # Train model using the configured trainer
        train_results = model_trainer.train(
            train_df, test_df, 
            model_type=model_type, 
            protein_name=protein_name,
            threshold=threshold,
            thresholds=thresholds  # Pass thresholds with n_classes and class_labels
        )
        
        # Check if training was successful
        if train_results['status'] != 'success':
            if mlflow_run_active:
                mlflow.end_run(status="FAILED")
            return train_results
        
        # Extract results from trainer
        model = train_results['model']
        y_true = train_results['y_true']
        y_pred = train_results['y_pred']
        y_prob = train_results['y_prob']
        
        # Calculate metrics (pass class_labels from train_results if available)
        class_labels = train_results.get('class_labels', ['Low', 'Medium', 'High'])
        metrics = model_reporter.calculate_metrics(y_true, y_pred, y_prob, class_labels=class_labels)
        
        # Save model
        model_path = model_reporter.save_model(
            model, protein_name, uniprot_id, model_type, threshold
        )
        
        # Save metrics
        metrics_path = model_reporter.save_metrics(
            metrics, protein_name, uniprot_id, model_type, threshold, 'test'
        )
        
        # Save class distributions
        train_dist_path = model_reporter.save_class_distribution(
            train_df, protein_name, uniprot_id, model_type, threshold, 'train'
        )
        test_dist_path = model_reporter.save_class_distribution(
            test_df, protein_name, uniprot_id, model_type, threshold, 'test'
        )
        
        # Save predictions (pass class_labels from train_results if available)
        class_labels = train_results.get('class_labels', ['Low', 'Medium', 'High'])
        # Use filtered test DataFrame from train_results if available (aligned with predictions)
        test_df_filtered = train_results.get('test_df_valid', test_df)
        predictions_path = model_reporter.save_predictions(
            test_df_filtered, y_pred, protein_name, uniprot_id, model_type, y_prob, threshold, 'test', class_labels=class_labels
        )
        
        # Log to MLflow if enabled (after saving artifacts so we can log them)
        if mlflow_run_active:
            try:
                log_to_mlflow(
                    model=model,
                    metrics=metrics,
                    train_df=train_df,
                    test_df=test_df,
                    protein_name=protein_name,
                    uniprot_id=uniprot_id,
                    model_type=model_type,
                    threshold=threshold,
                    train_results=train_results,
                    model_trainer=model_trainer,
                    metrics_path=metrics_path,
                    predictions_path=predictions_path,
                    train_dist_path=train_dist_path,
                    test_dist_path=test_dist_path
                )
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}. Continuing with model saving.")
        
        logger.info(f"✓ {model_trainer.get_model_type_name()} training completed successfully!")
        logger.info(f"  Model saved: {model_path}")
        logger.info(f"  Metrics saved: {metrics_path}")
        logger.info(f"  Predictions saved: {predictions_path}")
        logger.info(f"  Test accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Test F1-macro: {metrics['f1_macro']:.3f}")
        
        result = {
            'status': 'success',
            'test_accuracy': metrics['accuracy'],
            'test_f1_macro': metrics['f1_macro'],
            'test_f1_weighted': metrics['f1_weighted'],
            'model_path': model_path,
            'metrics_path': metrics_path,
            'predictions_path': predictions_path,
            'train_dist_path': train_dist_path,
            'test_dist_path': test_dist_path,
            'model_type_name': train_results.get('model_type_name', model_trainer.get_model_type_name()),
            'n_train_samples': train_results.get('n_train_samples'),
            'n_test_samples': train_results.get('n_test_samples'),
            'n_target_bioactivity_datapoints': len(test_df) + len(train_df),
            'n_features': train_results.get('feature_dimensions', 'N/A'),
            'similar_proteins': 'N/A'  # Will be filled by caller if needed
        }
        
        # End MLflow run successfully
        if mlflow_run_active:
            mlflow.end_run()
        
        return result
        
    except Exception as e:
        logger.error(f"Error training {model_trainer.get_model_type_name()} model: {e}")
        # End MLflow run with failure status
        if mlflow_run_active:
            mlflow.end_run(status="FAILED")
        return {
            'status': 'error',
            'message': str(e),
            'model_type_name': model_trainer.get_model_type_name()
        }


# ============================================================================
# Helper Functions - Result Saving
# ============================================================================

def save_workflow_results(results: List[Dict[str, Any]], output_dir: Path, model_reporter: ModelReporter, logger: logging.Logger):
    """
    Save workflow results to file and generate summary report
    
    Args:
        results: List of result dictionaries from protein processing
        output_dir: Output directory path
        model_reporter: ModelReporter instance
        logger: Logger instance
    """
    results_file = output_dir / "workflow_results.csv"
    
    # Flatten results for CSV
    csv_data = []
    for result in results:
        protein = result.get('protein', '')
        uniprot_id = result.get('uniprot_id', '')
        
        # Check if this is a combined result (has both model_a and model_b)
        if 'model_a' in result and 'model_b' in result:
            # Combined result: protein with similar proteins
            has_similar = result.get('has_similar_proteins', False)
            n_similar = result.get('n_similar_proteins', 0)
            
            # Add Model A result
            model_a = result['model_a']
            csv_data.append({
                'protein': protein,
                'uniprot_id': uniprot_id,
                'model_type': 'A',
                'status': model_a.get('status', ''),
                'n_samples': model_a.get('n_samples', ''),
                'n_train_samples': model_a.get('train_samples', 'N/A'),
                'n_test_samples': model_a.get('test_samples', 'N/A'),
                'threshold': '',
                'n_similar_activities': '',
                'n_target_activities': '',
                'n_similar_proteins': '',
                'has_similar_proteins': has_similar,
                'n_total_similar_proteins': n_similar,
                'n_features': model_a.get('n_features', 'N/A'),
                'similar_proteins': model_a.get('similar_proteins', 'N/A')
            })
            
            # Add Model B results (one per threshold)
            model_b = result['model_b']
            if 'thresholds' in model_b:
                for threshold, threshold_data in model_b['thresholds'].items():
                    csv_data.append({
                        'protein': protein,
                        'uniprot_id': uniprot_id,
                        'model_type': 'B',
                        'status': threshold_data.get('status', ''),
                        'n_samples': threshold_data.get('n_train', '') + threshold_data.get('n_test', '') if isinstance(threshold_data.get('n_train', ''), (int, float)) and isinstance(threshold_data.get('n_test', ''), (int, float)) else '',
                        'n_train_samples': threshold_data.get('n_train', 'N/A'),
                        'n_test_samples': threshold_data.get('n_test', 'N/A'),
                        'threshold': threshold,
                        'n_similar_activities': threshold_data.get('n_similar_activities', ''),
                        'n_target_activities': threshold_data.get('n_target_activities', ''),
                        'n_similar_proteins': threshold_data.get('n_similar_proteins', ''),
                        'has_similar_proteins': has_similar,
                        'n_total_similar_proteins': n_similar,
                        'n_features': threshold_data.get('n_features', 'N/A'),
                        'similar_proteins': threshold_data.get('similar_proteins', 'N/A')
                    })
        else:
            # Single model result: protein without similar proteins (Model A only)
            csv_data.append({
                'protein': protein,
                'uniprot_id': uniprot_id,
                'model_type': result.get('model_type', ''),
                'status': result.get('status', ''),
                'n_samples': result.get('n_samples', ''),
                'n_train_samples': result.get('train_samples', 'N/A'),
                'n_test_samples': result.get('test_samples', 'N/A'),
                'threshold': '',
                'n_similar_activities': '',
                'n_target_activities': '',
                'n_similar_proteins': '',
                'has_similar_proteins': False,
                'n_total_similar_proteins': 0,
                'n_features': result.get('n_features', 'N/A'),
                'similar_proteins': result.get('similar_proteins', 'N/A')
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")
    
    # Generate comprehensive summary report
    summary_path = model_reporter.generate_summary_report(df)
    logger.info(f"Generated summary report: {summary_path}")
    
    # Generate comprehensive protein report
    comprehensive_path = model_reporter.generate_comprehensive_report(df)
    logger.info(f"Generated comprehensive report: {comprehensive_path}")
    
    # Generate individual model reports
    individual_reports = model_reporter.generate_individual_model_reports(df)
    logger.info(f"Generated {len(individual_reports)} individual model reports")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main entry point for AQSE 3-Class Classification Workflow.
    
    Parses command line arguments, loads configuration, and runs the workflow.
    All orchestration logic is contained inline for maximum linearity.
    """
    # ============================================================================
    # STEP 1: Configuration and Setup
    # ============================================================================
    print("="*80)
    print("=== Loading Configuration ===")
    print("="*80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AQSE 3-Class Classification Workflow')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config.yaml file. If not provided, looks for config.yaml in project root.')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)
    
    # Resolve config file path
    config_path = resolve_config_path(args.config)
    
    # Load configuration
    try:
        config = load_config(config_path)
        # Store config directory for path resolution
        config['_config_dir'] = config_path.parent
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Override output_dir to use step-specific directory (05_model_training)
    # If output_dir is specified in config, use it as base; otherwise use project root
    original_output_dir = config.get('output_dir')
    if original_output_dir:
        # If it's an absolute path, use its parent; if relative, resolve from project root
        original_output_path = Path(original_output_dir)
        if original_output_path.is_absolute():
            base_dir = original_output_path.parent
        else:
            base_dir = project_root
    else:
        base_dir = project_root
    
    # Set step-specific output directory
    step_output_dir = base_dir / '05_model_training'
    config['output_dir'] = str(step_output_dir.resolve())
    step_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {step_output_dir}")
    
    # Validate required configuration
    avoidome_file = config.get("avoidome_file")
    if not avoidome_file:
        logger.error("No 'avoidome_file' path found in config.yaml")
        return 1
    
    logger.info(f"Avoidome input file: {avoidome_file}")
    
    # ============================================================================
    # STEP 2: Initialize Components
    # ============================================================================
    print("\n" + "="*80)
    print("=== Initializing Components ===")
    print("="*80)
    
    # Initialize loaders
    logger.info("Initializing data loaders...")
    avoidome_loader = AvoidomeDataLoader(config)
    similarity_loader = SimilarityDataLoader(config)
    bioactivity_loader = BioactivityDataLoader(config)
    thresholds_loader = ActivityThresholdsLoader(config)
    
    # Initialize model trainer (pass bioactivity_loader for Chemprop)
    logger.info("Initializing model trainer...")
    model_trainer = create_model_trainer(config, bioactivity_loader=bioactivity_loader)
    logger.info(f"Using {model_trainer.get_model_type_name()} trainer")
    
    # Initialize model reporter
    model_type = config.get("model_type", "random_forest")
    results_dir = Path(config.get("output_dir", ".")) / f"results_{model_type}"
    model_reporter = ModelReporter(results_dir)
    
    # Setup MLflow
    mlflow_config = setup_mlflow(config, model_type)
    mlflow_enabled = mlflow_config.get('mlflow_enabled', False)
    mlflow_experiment_name = mlflow_config.get('mlflow_experiment_name')
    if mlflow_enabled:
        logger.info(f"MLflow enabled: {mlflow_experiment_name}")
    else:
        logger.info("MLflow disabled")
    
    # Load data
    logger.info("Loading avoidome targets...")
    targets = avoidome_loader.load_avoidome_targets()
    logger.info(f"Loaded {len(targets)} target proteins")
    
    logger.info("Loading similarity results...")
    similarity_loader.load_similarity_results()
    
    # Cache fingerprints ONCE at initialization (only if trainer requires precomputed features)
    if model_trainer.requires_precomputed_features():
        logger.info("Loading Morgan fingerprints (will be cached for all proteins)...")
        cached_fingerprints = bioactivity_loader.load_morgan_fingerprints()
        logger.info(f"Cached {len(cached_fingerprints):,} fingerprints")
    else:
        logger.info("Model trainer does not require precomputed features - skipping fingerprint caching")
        cached_fingerprints = {}
    
    # ============================================================================
    # STEP 3: Optional Fingerprint Generation
    # ============================================================================
    generate_fingerprints = config.get('generate_fingerprints', True)
    
    if generate_fingerprints:
        print("\n" + "="*80)
        print("=== Generating Fingerprints ===")
        print("="*80)
        
        logger.info("Generating unique protein list...")
        protein_list_generator = UniqueProteinListGenerator(config)
        protein_list_df = protein_list_generator.generate_unique_protein_list()
        
        # Save to CSV
        output_file = protein_list_generator.save_protein_list()
        logger.info(f"Saved unique protein list to: {output_file}")
        
        # Calculate Morgan fingerprints filtered by unique proteins
        logger.info("Calculating Morgan fingerprints...")
        fingerprint_loader = BioactivityDataLoader(config)
        fingerprints_info = fingerprint_loader.calculate_morgan_fingerprints(protein_list_df)
        
        # Print dataset statistics
        if fingerprints_info:
            logger.info("\n=== Dataset Statistics ===")
            logger.info(f"Total activities in Papyrus: {fingerprints_info.get('total_activities_in_papyrus', 0):,}")
            logger.info(f"Activities after protein filter: {fingerprints_info.get('activities_after_protein_filter', 0):,}")
            logger.info(f"Unique proteins in filter: {fingerprints_info.get('unique_proteins_in_filter', 0)}")
            logger.info(f"Unique SMILES: {fingerprints_info.get('total_smiles', 0):,}")
            logger.info(f"Valid fingerprints calculated: {fingerprints_info.get('valid_fingerprints', 0):,}")
    
    # ============================================================================
    # STEP 4: Process All Proteins
    # ============================================================================
    print("\n" + "="*80)
    print("=== Starting Protein Processing ===")
    print("="*80)
    logger.info(f"Processing {len(targets)} proteins")
    
    results = []
    
    for i, protein in enumerate(targets):
        uniprot_id = protein['UniProt ID']
        protein_name = protein.get('Name_2', uniprot_id)
        current_idx = i + 1
        total = len(targets)
        
        # Display status
        print(f"\n[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})")
        
        try:
            # Get similarity and thresholds
            similar_proteins = similarity_loader.get_similar_proteins(uniprot_id)
            has_similar = len(similar_proteins) > 0
            
            if has_similar:
                logger.info(f"  Similar proteins found: {len(similar_proteins)}")
            else:
                logger.info("  No similar proteins")
            
            thresholds = thresholds_loader.get_thresholds(uniprot_id)
            if not thresholds:
                logger.warning(f"No activity thresholds found for {uniprot_id}")
                result = {
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'model_type': 'unknown',
                    'status': 'no_thresholds'
                }
                results.append(result)
                continue
            
            # Log classification type
            n_classes = thresholds.get('n_classes', 3)
            class_labels = thresholds.get('class_labels', ['Low', 'Medium', 'High'])
            logger.info(f"  Classification: {n_classes} classes ({', '.join(class_labels)})")
            
            if not has_similar:
                # ========================================================================
                # Case A: No similar proteins - Model A only
                # ========================================================================
                logger.info("  Training Model A (QSAR only)...")
                
                # Load bioactivity data
                bioactivity_data = bioactivity_loader.get_filtered_papyrus([uniprot_id])
                
                if len(bioactivity_data) < 30:
                    logger.warning(f"Insufficient data: {len(bioactivity_data)} samples (need ≥30)")
                    result = {
                        'protein': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': 'A',
                        'status': 'insufficient_data',
                        'n_samples': len(bioactivity_data)
                    }
                    results.append(result)
                    continue
                
                # Prepare data
                train_df, val_df, test_df, total_samples = prepare_model_a_data(
                    bioactivity_data=bioactivity_data,
                    uniprot_id=uniprot_id,
                    thresholds=thresholds,
                    model_trainer=model_trainer,
                    cached_fingerprints=cached_fingerprints,
                    bioactivity_loader=bioactivity_loader,
                    config=config,
                    logger=logger
                )
                
                if train_df is None:
                    result = {
                        'protein': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': 'A',
                        'status': 'insufficient_data',
                        'n_samples': total_samples
                    }
                    results.append(result)
                    continue
                
                logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")
                if val_df is not None:
                    logger.info(f"  Validation: {len(val_df)}")
                logger.info(f"  Class distribution - Train: {train_df['class'].value_counts().to_dict()}")
                logger.info(f"  Class distribution - Test: {test_df['class'].value_counts().to_dict()}")
                
                # Train Model A
                use_optimization = config.get('enable_parameter_optimization', False)
                model_type_config = config.get('model_type', 'random_forest').lower()
                
                if use_optimization and model_type_config == 'chemprop' and val_df is not None:
                    logger.info("  Running hyperparameter optimization...")
                    optimizer = ChempropHyperparameterOptimizer(
                        n_trials=config.get('optimization_n_trials', 50),
                        search_strategy=config.get('optimization_strategy', 'hybrid'),
                        random_state=config.get('random_state', 42),
                        bioactivity_loader=bioactivity_loader
                    )
                    
                    opt_result = optimizer.optimize(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        model_type='A',
                        protein_name=protein_name,
                        uniprot_id=uniprot_id,
                        thresholds=thresholds,
                        max_epochs=config.get('chemprop_max_epochs', 100),
                        optimization_epochs=config.get('optimization_epochs', 50),
                        n_classes=thresholds.get('n_classes', 3),
                        include_esm=False,
                        val_fraction=0.0
                    )
                    
                    if opt_result['status'] == 'success':
                        final_result = opt_result['final_test_result']
                        train_results = {
                            'status': 'success',
                            'test_accuracy': final_result.get('test_accuracy', 0.0),
                            'test_f1_macro': final_result.get('test_f1_macro', 0.0),
                            'test_f1_weighted': final_result.get('test_f1', 0.0),
                            'model_path': 'N/A',
                            'metrics_path': 'N/A',
                            'best_params': opt_result['best_params'],
                            'best_val_score': opt_result['best_val_score'],
                            'optimization_trials': opt_result['n_trials'],
                            'optimization_successful': opt_result['n_successful']
                        }
                    else:
                        logger.warning("Parameter optimization failed, falling back to default training")
                        train_results = train_model_with_reporting(
                            model_trainer=model_trainer,
                            model_reporter=model_reporter,
                            train_df=train_df,
                            test_df=test_df,
                            model_type='A',
                            protein_name=protein_name,
                            uniprot_id=uniprot_id,
                            thresholds=thresholds,
                            mlflow_enabled=mlflow_enabled,
                            mlflow_experiment_name=mlflow_experiment_name,
                            output_dir=step_output_dir,
                            cached_fingerprints=cached_fingerprints,
                            bioactivity_loader=bioactivity_loader,
                            logger=logger
                        )
                else:
                    # Standard training without optimization
                    train_results = train_model_with_reporting(
                        model_trainer=model_trainer,
                        model_reporter=model_reporter,
                        train_df=train_df,
                        test_df=test_df,
                        model_type='A',
                        protein_name=protein_name,
                        uniprot_id=uniprot_id,
                        thresholds=thresholds,
                        mlflow_enabled=mlflow_enabled,
                        mlflow_experiment_name=mlflow_experiment_name,
                        output_dir=step_output_dir,
                        cached_fingerprints=cached_fingerprints,
                        bioactivity_loader=bioactivity_loader,
                        logger=logger
                    )
                
                result = {
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'model_type': 'A',
                    'status': 'complete',
                    'n_samples': total_samples,
                    'train_samples': len(train_df),
                    'test_samples': len(test_df),
                    'similar_proteins': 'N/A',
                    'n_features': train_results.get('n_features', 'N/A'),
                    **train_results
                }
                
                # Display completion status
                if train_results.get('status') == 'success' and 'test_f1_macro' in train_results:
                    f1_score = train_results['test_f1_macro']
                    print(f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  ✓ Model A completed (F1: {f1_score:.2f})")
                
                results.append(result)
                
            else:
                # ========================================================================
                # Case B: Has similar proteins - Model A + Model B
                # ========================================================================
                logger.info("  Training Model A (before PCM)...")
                
                # Train Model A first
                bioactivity_data = bioactivity_loader.get_filtered_papyrus([uniprot_id])
                
                if len(bioactivity_data) < 30:
                    logger.warning(f"Insufficient data for Model A: {len(bioactivity_data)} samples (need ≥30)")
                    model_a_result = {
                        'protein': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': 'A',
                        'status': 'insufficient_data',
                        'n_samples': len(bioactivity_data)
                    }
                else:
                    train_df, val_df, test_df, total_samples = prepare_model_a_data(
                        bioactivity_data=bioactivity_data,
                        uniprot_id=uniprot_id,
                        thresholds=thresholds,
                        model_trainer=model_trainer,
                        cached_fingerprints=cached_fingerprints,
                        bioactivity_loader=bioactivity_loader,
                        config=config,
                        logger=logger
                    )
                    
                    if train_df is None:
                        model_a_result = {
                            'protein': protein_name,
                            'uniprot_id': uniprot_id,
                            'model_type': 'A',
                            'status': 'insufficient_data',
                            'n_samples': total_samples
                        }
                    else:
                        # Train Model A (standard, no optimization for Model A when Model B follows)
                        train_results_a = train_model_with_reporting(
                            model_trainer=model_trainer,
                            model_reporter=model_reporter,
                            train_df=train_df,
                            test_df=test_df,
                            model_type='A',
                            protein_name=protein_name,
                            uniprot_id=uniprot_id,
                            thresholds=thresholds,
                            mlflow_enabled=mlflow_enabled,
                            mlflow_experiment_name=mlflow_experiment_name,
                            output_dir=step_output_dir,
                            cached_fingerprints=cached_fingerprints,
                            bioactivity_loader=bioactivity_loader,
                            logger=logger
                        )
                        
                        model_a_result = {
                            'protein': protein_name,
                            'uniprot_id': uniprot_id,
                            'model_type': 'A',
                            'status': 'complete',
                            'n_samples': total_samples,
                            'train_samples': len(train_df),
                            'test_samples': len(test_df),
                            'similar_proteins': 'N/A',
                            'n_features': train_results_a.get('n_features', 'N/A'),
                            **train_results_a
                        }
                
                # Display Model A completion status
                model_a_f1 = None
                if model_a_result.get('status') == 'complete' and 'test_f1_macro' in model_a_result:
                    model_a_f1 = model_a_result['test_f1_macro']
                    print(f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  ✓ Model A completed (F1: {model_a_f1:.2f})  → Training Model B (high threshold)...")
                
                # Train Model B
                logger.info("  Training Model B (PCM with similar proteins)...")
                results_by_threshold = {}
                
                for threshold in ['high', 'medium', 'low']:
                    logger.info(f"    Processing {threshold.upper()} threshold...")
                    
                    # Display status for medium and low thresholds
                    if threshold != 'high':
                        status_parts = []
                        if model_a_f1 is not None:
                            status_parts.append(f"✓ Model A completed (F1: {model_a_f1:.2f})")
                        for t in ['high', 'medium']:
                            if t in results_by_threshold:
                                t_result = results_by_threshold[t]
                                if t_result.get('status') == 'complete' and 'test_f1_macro' in t_result:
                                    t_f1 = t_result['test_f1_macro']
                                    status_parts.append(f"✓ Model B ({t}) completed (F1: {t_f1:.2f})")
                        status_parts.append(f"→ Training Model B ({threshold} threshold)...")
                        print(f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  {'  '.join(status_parts)}")
                    
                    # Get similar proteins for this threshold
                    similar_proteins_thresh = similarity_loader.get_similar_proteins_for_threshold(uniprot_id, threshold)
                    
                    if not similar_proteins_thresh:
                        logger.warning(f"No similar proteins at {threshold} threshold")
                        results_by_threshold[threshold] = {
                            'status': 'no_similar_proteins',
                            'n_similar': 0
                        }
                        continue
                    
                    # Load bioactivity data
                    similar_data = bioactivity_loader.get_filtered_papyrus(similar_proteins_thresh)
                    target_data = bioactivity_loader.get_filtered_papyrus([uniprot_id])
                    
                    total_samples_b = len(similar_data) + len(target_data)
                    if total_samples_b < 30:
                        logger.warning(f"Insufficient data at {threshold}: {total_samples_b} samples")
                        results_by_threshold[threshold] = {
                            'status': 'insufficient_data',
                            'n_samples': total_samples_b,
                            'n_similar_proteins': len(similar_proteins_thresh)
                        }
                        continue
                    
                    logger.info(f"    Similar proteins data: {len(similar_data):,} activities")
                    logger.info(f"    Target protein data: {len(target_data):,} activities")
                    
                    # Prepare data
                    train_features, test_features = prepare_model_b_data(
                        similar_data=similar_data,
                        target_data=target_data,
                        uniprot_id=uniprot_id,
                        thresholds=thresholds,
                        model_trainer=model_trainer,
                        cached_fingerprints=cached_fingerprints,
                        bioactivity_loader=bioactivity_loader,
                        logger=logger
                    )
                    
                    if train_features is None or test_features is None:
                        results_by_threshold[threshold] = {
                            'status': 'insufficient_features',
                            'n_train': len(train_features) if train_features is not None else 0,
                            'n_test': len(test_features) if test_features is not None else 0
                        }
                        continue
                    
                    if len(train_features) < 30 or len(test_features) < 6:
                        logger.warning(f"Insufficient data after feature extraction")
                        results_by_threshold[threshold] = {
                            'status': 'insufficient_features',
                            'n_train': len(train_features),
                            'n_test': len(test_features)
                        }
                        continue
                    
                    logger.info(f"    Train: {len(train_features)}, Test: {len(test_features)}")
                    logger.info(f"    Class distribution - Train: {train_features['class'].value_counts().to_dict()}")
                    logger.info(f"    Class distribution - Test: {test_features['class'].value_counts().to_dict()}")
                    
                    # Train model
                    train_results_b = train_model_with_reporting(
                        model_trainer=model_trainer,
                        model_reporter=model_reporter,
                        train_df=train_features,
                        test_df=test_features,
                        model_type='B',
                        protein_name=protein_name,
                        threshold=threshold,
                        uniprot_id=uniprot_id,
                        thresholds=thresholds,
                        mlflow_enabled=mlflow_enabled,
                        mlflow_experiment_name=mlflow_experiment_name,
                        output_dir=step_output_dir,
                        cached_fingerprints=cached_fingerprints,
                        bioactivity_loader=bioactivity_loader,
                        logger=logger
                    )
                    
                    threshold_result = {
                        'status': 'complete',
                        'n_train': len(train_features),
                        'n_test': len(test_features),
                        'n_similar_proteins': len(similar_proteins_thresh),
                        'n_similar_activities': len(similar_data),
                        'n_target_activities': len(target_data),
                        'similar_proteins': similar_proteins_thresh,
                        'n_features': train_results_b.get('n_features', 'N/A'),
                        **train_results_b
                    }
                    
                    # Display completion status for this threshold
                    if train_results_b.get('status') == 'success' and 'test_f1_macro' in train_results_b:
                        f1_score = train_results_b['test_f1_macro']
                        
                        # Build status message showing Model A and completed Model B thresholds
                        status_parts = []
                        if model_a_f1 is not None:
                            status_parts.append(f"✓ Model A completed (F1: {model_a_f1:.2f})")
                        
                        # Add all completed Model B thresholds including current one
                        for t in ['high', 'medium', 'low']:
                            if t == threshold:
                                status_parts.append(f"✓ Model B ({t}) completed (F1: {f1_score:.2f})")
                                break
                            elif t in results_by_threshold:
                                t_result = results_by_threshold[t]
                                if t_result.get('status') == 'complete' and 'test_f1_macro' in t_result:
                                    t_f1 = t_result['test_f1_macro']
                                    status_parts.append(f"✓ Model B ({t}) completed (F1: {t_f1:.2f})")
                        
                        # Determine next threshold to show
                        threshold_order = ['high', 'medium', 'low']
                        current_idx_in_order = threshold_order.index(threshold)
                        if current_idx_in_order < len(threshold_order) - 1:
                            next_threshold = threshold_order[current_idx_in_order + 1]
                            status_parts.append(f"→ Training Model B ({next_threshold} threshold)...")
                        
                        print(f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  {'  '.join(status_parts)}")
                    
                    results_by_threshold[threshold] = threshold_result
                
                model_b_result = {
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'model_type': 'B',
                    'thresholds': results_by_threshold
                }
                
                # Combined results
                result = {
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'has_similar_proteins': True,
                    'n_similar_proteins': len(similar_proteins),
                    'model_a': model_a_result,
                    'model_b': model_b_result
                }
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error processing {protein_name}: {e}", exc_info=True)
            results.append({
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'error': str(e)
            })
    
    # ============================================================================
    # STEP 5: Save Results
    # ============================================================================
    print("\n" + "="*80)
    print("=== Saving Results ===")
    print("="*80)
    
    save_workflow_results(results, step_output_dir, model_reporter, logger)
    logger.info(f"Saved {len(results)} protein results")
    
    # ============================================================================
    # STEP 6: Generate Summary Tables
    # ============================================================================
    generate_tables = config.get('generate_summary_tables', True)
    if generate_tables:
        print("\n" + "="*80)
        print("=== Generating Summary Tables ===")
        print("="*80)
        
        try:
            _generate_summary_tables(config, project_root, logger)
            logger.info("\n=== Summary Tables Generated Successfully ===")
        except Exception as e:
            logger.warning(f"Failed to generate summary tables: {e}")
            logger.warning("Workflow completed successfully, but table generation failed.")
    
    print("\n" + "="*80)
    print("=== Workflow Completed Successfully ===")
    print("="*80)
    
    return 0


def _generate_summary_tables(config: dict, project_root: Path, logger: logging.Logger):
    """
    Generate summary tables after workflow completion.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory
        logger: Logger instance
    """
    # Get paths from config (already resolved to absolute paths by config_loader)
    avoidome_file = config.get('avoidome_file')
    similarity_file = config.get('similarity_file')
    activity_thresholds_file = config.get('activity_thresholds_file')
    output_dir = config.get('output_dir')
    
    if not all([avoidome_file, similarity_file, activity_thresholds_file, output_dir]):
        logger.warning("Missing required config paths for table generation. Skipping.")
        return
    
    # Convert to Path objects (paths are already absolute strings from config_loader)
    avoidome_file = Path(avoidome_file)
    similarity_file = Path(similarity_file)
    activity_thresholds_file = Path(activity_thresholds_file)
    output_dir = Path(output_dir)
    
    # Check if required modules are loaded (at least the two we have)
    if not all([create_bioactivity_table_with_similar_module, create_protein_summary_table_module]):
        logger.warning("Required table generation modules not available. Skipping table generation.")
        return
    
    # 1. Generate bioactivity table (target proteins only)
    if create_bioactivity_table_module:
        logger.info("\n--- Generating Bioactivity Table (Target Proteins) ---")
        try:
            thresholds_dict = create_bioactivity_table_module.load_activity_thresholds(str(activity_thresholds_file))
            avoidome_uniprot_ids = create_bioactivity_table_module.get_avoidome_uniprot_ids(str(avoidome_file))
            papyrus_df = create_bioactivity_table_module.load_papyrus_data()
            
            bioactivity_table = create_bioactivity_table_module.create_bioactivity_table(
                papyrus_df,
                thresholds_dict,
                avoidome_uniprot_ids
            )
            
            output_file = output_dir / "bioactivity_table.csv"
            bioactivity_table.to_csv(output_file, index=False)
            logger.info(f"Saved bioactivity table to: {output_file} ({len(bioactivity_table):,} rows)")
        except Exception as e:
            logger.error(f"Error generating bioactivity table: {e}", exc_info=True)
    
    # 2. Generate bioactivity table with similar proteins
    logger.info("\n--- Generating Bioactivity Table (With Similar Proteins) ---")
    try:
        thresholds_dict = create_bioactivity_table_with_similar_module.load_activity_thresholds(str(activity_thresholds_file))
        avoidome_uniprot_ids = create_bioactivity_table_with_similar_module.get_avoidome_uniprot_ids(str(avoidome_file))
        similar_proteins_dict = create_bioactivity_table_with_similar_module.load_similar_proteins(
            str(similarity_file), 
            avoidome_uniprot_ids
        )
        papyrus_df = create_bioactivity_table_with_similar_module.load_papyrus_data()
        
        bioactivity_table_with_similar = create_bioactivity_table_with_similar_module.create_bioactivity_table(
            papyrus_df,
            thresholds_dict,
            avoidome_uniprot_ids,
            similar_proteins_dict
        )
        
        output_file = output_dir / "bioactivity_table_with_similar.csv"
        bioactivity_table_with_similar.to_csv(output_file, index=False)
        logger.info(f"Saved bioactivity table with similar proteins to: {output_file} ({len(bioactivity_table_with_similar):,} rows)")
    except Exception as e:
        logger.error(f"Error generating bioactivity table with similar: {e}", exc_info=True)
    
    # 3. Generate protein summary table
    logger.info("\n--- Generating Protein Summary Table ---")
    try:
        thresholds_df = create_protein_summary_table_module.load_activity_thresholds(str(activity_thresholds_file))
        similar_proteins_dict = create_protein_summary_table_module.load_similar_proteins(str(similarity_file))
        two_class_proteins = create_protein_summary_table_module.get_2class_proteins()
        papyrus_df = create_protein_summary_table_module.load_papyrus_data()
        
        # Try to load metrics from workflow results
        workflow_results_file = output_dir / "workflow_results.csv"
        metrics_dict = None
        if workflow_results_file.exists():
            try:
                # Determine metrics directory based on model type
                model_type = config.get('model_type', 'chemprop')
                metrics_dir = output_dir / f"results_{model_type}" / f"results_{model_type}" / "metrics"
                rf_metrics_dir = output_dir / "results_random_forest" / "results_random_forest" / "metrics"
                metrics_dict = create_protein_summary_table_module.load_model_metrics(
                    str(workflow_results_file),
                    metrics_dir if metrics_dir.exists() else None,
                    rf_metrics_dir if rf_metrics_dir.exists() else None
                )
                logger.info(f"Loaded metrics for {len(metrics_dict)} model configurations")
            except Exception as e:
                logger.debug(f"Could not load model metrics: {e}")
        
        protein_summary_table = create_protein_summary_table_module.create_protein_summary_table(
            thresholds_df,
            similar_proteins_dict,
            two_class_proteins,
            papyrus_df,
            metrics_dict
        )
        
        output_file = output_dir / "protein_summary_table.csv"
        protein_summary_table.to_csv(output_file, index=False)
        logger.info(f"Saved protein summary table to: {output_file} ({len(protein_summary_table)} rows)")
    except Exception as e:
        logger.error(f"Error generating protein summary table: {e}", exc_info=True)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
