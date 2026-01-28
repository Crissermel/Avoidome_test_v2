"""
Model training wrapper for AQSE workflow.

This module provides a wrapper for training models with comprehensive reporting and MLflow tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# MLflow imports (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..models.base import ModelTrainer
from ..reporting.model_reporter import ModelReporter
from ..utils.mlflow_logger import log_to_mlflow

logger = logging.getLogger(__name__)


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
    bioactivity_loader=None
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
        
    Returns:
        Dictionary with training results
    """
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
        predictions_path = model_reporter.save_predictions(
            test_df, y_pred, protein_name, uniprot_id, model_type, y_prob, threshold, 'test', class_labels=class_labels
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
