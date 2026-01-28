"""
MLflow integration utilities for AQSE workflow.

This module provides functions for setting up MLflow tracking and logging
model artifacts, metrics, and parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# MLflow imports (optional)
try:
    import mlflow
    import mlflow.sklearn
    try:
        import mlflow.pytorch
        MLFLOW_PYTORCH_AVAILABLE = True
    except ImportError:
        MLFLOW_PYTORCH_AVAILABLE = False
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLFLOW_PYTORCH_AVAILABLE = False


def setup_mlflow(config: Dict[str, Any], model_type: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Setup MLflow tracking
    
    Args:
        config: Configuration dictionary
        model_type: Model type (e.g., 'random_forest', 'chemprop')
        
    Returns:
        Dictionary with MLflow configuration including:
        - mlflow_enabled: Whether MLflow is enabled
        - mlflow_experiment_name: Experiment name
        - mlflow_experiment_id: Experiment ID (if created)
    """
    result = {
        'mlflow_enabled': False,
        'mlflow_experiment_name': None,
        'mlflow_experiment_id': None
    }
    
    # Use provided logger or module logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if MLflow is available
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow is not installed. MLflow tracking will be disabled.")
        return result
    
    # Get MLflow configuration from config or use defaults
    mlflow_tracking_uri = config.get("mlflow_tracking_uri", None)
    mlflow_experiment_name = config.get("mlflow_experiment_name", f"AQSE_3C_2C_optimized_{model_type}")
    mlflow_enabled = config.get("mlflow_enabled", True)
    
    result['mlflow_enabled'] = mlflow_enabled
    
    if not mlflow_enabled:
        logger.info("MLflow tracking is disabled")
        return result
    
    try:
        # Set tracking URI if provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(mlflow_experiment_name)
            logger.info(f"Created MLflow experiment: {mlflow_experiment_name} (ID: {experiment_id})")
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {mlflow_experiment_name} (ID: {experiment_id})")
            else:
                raise
        
        result['mlflow_experiment_name'] = mlflow_experiment_name
        result['mlflow_experiment_id'] = experiment_id
        logger.info(f"MLflow tracking initialized for experiment: {mlflow_experiment_name}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}. Continuing without MLflow tracking.")
        result['mlflow_enabled'] = False
    
    return result


def log_to_mlflow(
    model: Any,
    metrics: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_name: Optional[str],
    uniprot_id: str,
    model_type: str,
    threshold: Optional[str],
    train_results: Dict[str, Any],
    model_trainer,
    metrics_path: str = "",
    predictions_path: str = "",
    train_dist_path: str = "",
    test_dist_path: str = ""
):
    """
    Log model, metrics, parameters, and artifacts to MLflow
    
    Args:
        model: Trained model object
        metrics: Dictionary of calculated metrics
        train_df: Training DataFrame
        test_df: Test DataFrame
        protein_name: Protein name
        uniprot_id: UniProt ID
        model_type: Model type ('A' or 'B')
        threshold: Threshold for Model B
        train_results: Training results dictionary
        model_trainer: Model trainer instance (for extracting parameters)
        metrics_path: Path to saved metrics JSON file
        predictions_path: Path to saved predictions CSV file
        train_dist_path: Path to saved train distribution CSV file
        test_dist_path: Path to saved test distribution CSV file
    """
    # Safety check: ensure MLflow is available
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow is not available. Skipping MLflow logging.")
        return
    
    # Log tags
    tags = {
        'protein_name': protein_name or 'unknown',
        'uniprot_id': uniprot_id,
        'model_type': model_type,
        'model_type_name': train_results.get('model_type_name', model_trainer.get_model_type_name())
    }
    if threshold:
        tags['threshold'] = threshold
    mlflow.set_tags(tags)
    
    # Log parameters
    params = {}
    
    # Model-specific parameters
    # Import here to avoid circular dependencies
    from aqse_modelling.models.random_forest_trainer import RandomForestTrainer
    from aqse_modelling.models.chemprop_trainer import ChempropTrainer
    
    if isinstance(model_trainer, RandomForestTrainer):
        params['n_estimators'] = model_trainer.n_estimators
        params['max_depth'] = model_trainer.max_depth if model_trainer.max_depth else 'None'
        params['random_state'] = model_trainer.random_state
        params['class_weight'] = model_trainer.class_weight
    elif isinstance(model_trainer, ChempropTrainer):
        params['max_epochs'] = model_trainer.max_epochs
        params['batch_size'] = model_trainer.batch_size
        params['n_classes'] = model_trainer.n_classes
        params['include_esm'] = model_trainer.include_esm
        params['val_fraction'] = model_trainer.val_fraction
        params['max_lr'] = model_trainer.max_lr
        params['init_lr'] = model_trainer.init_lr
        params['final_lr'] = model_trainer.final_lr
        params['warmup_epochs'] = model_trainer.warmup_epochs
        params['ffn_num_layers'] = model_trainer.ffn_num_layers
        params['hidden_size'] = model_trainer.hidden_size
        params['dropout'] = model_trainer.dropout
        params['activation'] = model_trainer.activation
        params['aggregation'] = model_trainer.aggregation
        params['depth'] = model_trainer.depth
        params['bias'] = model_trainer.bias
    
    # Data parameters
    params['n_train_samples'] = train_results.get('n_train_samples', len(train_df))
    params['n_test_samples'] = train_results.get('n_test_samples', len(test_df))
    params['n_features'] = train_results.get('feature_dimensions', 'N/A')
    
    # Classification parameters
    n_classes = train_results.get('n_classes', metrics.get('class_labels', ['Low', 'Medium', 'High']))
    if isinstance(n_classes, int):
        params['n_classes'] = n_classes
    else:
        params['n_classes'] = len(metrics.get('class_labels', ['Low', 'Medium', 'High']))
    params['class_labels'] = ','.join(metrics.get('class_labels', ['Low', 'Medium', 'High']))
    
    # Convert all params to strings (MLflow requirement)
    params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(params)
    
    # Log metrics
    mlflow_metrics = {
        'test_accuracy': metrics['accuracy'],
        'test_precision_macro': metrics['precision_macro'],
        'test_recall_macro': metrics['recall_macro'],
        'test_f1_macro': metrics['f1_macro'],
        'test_precision_weighted': metrics['precision_weighted'],
        'test_recall_weighted': metrics['recall_weighted'],
        'test_f1_weighted': metrics['f1_weighted']
    }
    
    # Log per-class metrics (use class_labels from metrics)
    if 'precision_per_class' in metrics:
        class_labels = metrics.get('class_labels', ['Low', 'Medium', 'High'])
        for i, label in enumerate(class_labels):
            # Sanitize label for MLflow (replace + with _)
            label_safe = label.lower().replace('+', '_')
            mlflow_metrics[f'test_precision_{label_safe}'] = metrics['precision_per_class'][i]
            mlflow_metrics[f'test_recall_{label_safe}'] = metrics['recall_per_class'][i]
            mlflow_metrics[f'test_f1_{label_safe}'] = metrics['f1_per_class'][i]
    
    mlflow.log_metrics(mlflow_metrics)
    
    # Log model
    model_type_name = train_results.get('model_type_name', model_trainer.get_model_type_name())
    registered_name = f"{protein_name}_{uniprot_id}_{model_type}" + (f"_{threshold}" if threshold else "")
    
    if model_type_name == "RandomForest":
        # Create input example from test data for signature inference
        try:
            # Get a sample from test data for input example
            X_test_sample = np.stack(test_df['features'].to_numpy()[:1]) if len(test_df) > 0 else None
            input_example = X_test_sample if X_test_sample is not None else None
        except Exception:
            input_example = None
        
        # Log sklearn model with updated API
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_name,
            input_example=input_example
        )
    elif model_type_name == "Chemprop" and MLFLOW_PYTORCH_AVAILABLE:
        # Log PyTorch model (if it's a PyTorch model)
        try:
            # Create input example if possible
            input_example = None
            try:
                if len(test_df) > 0 and 'SMILES' in test_df.columns:
                    # Use first SMILES as input example for Chemprop
                    input_example = test_df['SMILES'].iloc[0] if len(test_df) > 0 else None
            except Exception:
                pass
            
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=registered_name,
                input_example=input_example
            )
        except Exception as e:
            logger.warning(f"Failed to log Chemprop model to MLflow: {e}. Model may not be PyTorch-based.")
    
    # Log artifacts (if they exist)
    try:
        # Log metrics JSON
        if metrics_path and Path(metrics_path).exists():
            mlflow.log_artifact(metrics_path, "metrics")
        
        # Log predictions CSV
        if predictions_path and Path(predictions_path).exists():
            mlflow.log_artifact(predictions_path, "predictions")
        
        # Log class distributions
        if train_dist_path and Path(train_dist_path).exists():
            mlflow.log_artifact(train_dist_path, "class_distributions")
        
        if test_dist_path and Path(test_dist_path).exists():
            mlflow.log_artifact(test_dist_path, "class_distributions")
        
    except Exception as e:
        logger.warning(f"Failed to log some artifacts to MLflow: {e}")
