"""
Hyperparameter optimization for Chemprop models.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, Any, Optional
import logging

from .chemprop_trainer import ChempropTrainer

# MLflow imports (optional)
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChempropHyperparameterOptimizer:
    """
    Semi-randomized hyperparameter optimization for ChemProp models.
    
    Uses a hybrid approach:
    - Systematic grid search on key parameters (hidden_size, dropout)
    - Random sampling for other parameters
    - Fixed validation set based on doc_id for unbiased evaluation
    """
    
    def __init__(self, n_trials: int = 50, search_strategy: str = 'hybrid', 
                 random_state: int = 42, bioactivity_loader=None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            n_trials: Number of hyperparameter combinations to try
            search_strategy: 'hybrid' (grid + random) or 'random' (fully random)
            random_state: Random seed for reproducibility
            bioactivity_loader: BioactivityDataLoader instance
        """
        self.n_trials = n_trials
        self.search_strategy = search_strategy
        self.random_state = random_state
        self.bioactivity_loader = bioactivity_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Define parameter search spaces
        self.parameter_space = {
            'max_lr': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
            'init_lr': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
            'final_lr': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
            'dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.7},
            'hidden_size': {'type': 'int_log_uniform', 'low': 100, 'high': 2000},
            'ffn_num_layers': {'type': 'int_uniform', 'low': 2, 'high': 6},
            'depth': {'type': 'int_uniform', 'low': 3, 'high': 7},
            'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
            'activation': {'type': 'categorical', 'choices': ['ReLU', 'LeakyReLU', 'PReLU']},
            'aggregation': {'type': 'categorical', 'choices': ['mean', 'norm']},
            'warmup_epochs': {'type': 'int_uniform', 'low': 3, 'high': 10},
            'bias': {'type': 'categorical', 'choices': [True, False]},
        }
        
        # Hybrid strategy: grid search on these parameters
        self.grid_params = ['hidden_size', 'dropout']
        self.grid_values = {
            'hidden_size': [300, 600, 900, 1200, 1500],
            'dropout': [0.0, 0.2, 0.4, 0.6]
        }
    
    def _sample_parameter(self, param_name: str, param_spec: Dict[str, Any]) -> Any:
        """Sample a single parameter value according to its distribution."""
        param_type = param_spec['type']
        
        if param_type == 'log_uniform':
            low = param_spec['low']
            high = param_spec['high']
            # Log-uniform sampling
            log_low = np.log(low)
            log_high = np.log(high)
            return np.exp(random.uniform(log_low, log_high))
        
        elif param_type == 'uniform':
            return random.uniform(param_spec['low'], param_spec['high'])
        
        elif param_type == 'int_uniform':
            return random.randint(param_spec['low'], param_spec['high'])
        
        elif param_type == 'int_log_uniform':
            low = param_spec['low']
            high = param_spec['high']
            log_low = np.log(low)
            log_high = np.log(high)
            log_val = random.uniform(log_low, log_high)
            return int(np.exp(log_val))
        
        elif param_type == 'categorical':
            return random.choice(param_spec['choices'])
        
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _sample_parameters(self, trial_idx: int) -> Dict[str, Any]:
        """Sample a complete set of hyperparameters for a trial."""
        if self.search_strategy == 'hybrid':
            # Hybrid: grid search on key params, random for others
            grid_idx = trial_idx % (len(self.grid_values['hidden_size']) * len(self.grid_values['dropout']))
            hidden_idx = grid_idx // len(self.grid_values['dropout'])
            dropout_idx = grid_idx % len(self.grid_values['dropout'])
            
            params = {
                'hidden_size': self.grid_values['hidden_size'][hidden_idx],
                'dropout': self.grid_values['dropout'][dropout_idx]
            }
            
            # Random sample for other parameters
            for param_name, param_spec in self.parameter_space.items():
                if param_name not in self.grid_params:
                    params[param_name] = self._sample_parameter(param_name, param_spec)
        
        else:  # 'random' strategy
            # Fully random sampling
            params = {}
            for param_name, param_spec in self.parameter_space.items():
                params[param_name] = self._sample_parameter(param_name, param_spec)
        
        # Ensure learning rate constraints
        if params['init_lr'] > params['max_lr']:
            params['init_lr'] = params['max_lr'] * 0.1
        if params['final_lr'] > params['max_lr']:
            params['final_lr'] = params['max_lr'] * 0.1
        
        return params
    
    def optimize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 model_type: str = 'A', protein_name: Optional[str] = None,
                 uniprot_id: Optional[str] = None, thresholds: Optional[Dict[str, Any]] = None,
                 max_epochs: int = 100, optimization_epochs: Optional[int] = None,
                 n_classes: int = 3, include_esm: bool = False,
                 val_fraction: float = 0.0) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (fixed, used for hyperparameter selection)
            test_df: Test DataFrame (held out for final evaluation)
            model_type: 'A' or 'B'
            protein_name: Optional protein name
            uniprot_id: Optional UniProt ID
            thresholds: Dictionary with thresholds, n_classes, and class_labels
            max_epochs: Maximum training epochs for final model (after optimization)
            optimization_epochs: Epochs to use during optimization trials (shorter for speed)
            n_classes: Number of classes
            include_esm: Whether to include ESM embeddings
            val_fraction: Set to 0.0 since we use external validation set
            
        Returns:
            Dictionary with best parameters, best score, and all trial results
        """
        # Use shorter epochs for optimization trials if specified, otherwise use max_epochs
        trial_epochs = optimization_epochs if optimization_epochs is not None else max_epochs
        
        self.logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        self.logger.info(f"Search strategy: {self.search_strategy}")
        self.logger.info(f"Epochs per trial: {trial_epochs} (final model will use {max_epochs} epochs)")
        self.logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
        
        best_score = -np.inf
        best_params = None
        best_result = None
        all_results = []
        
        # Start MLflow parent run for optimization
        mlflow_run_active = False
        parent_run_id = None
        parent_run = None
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(f"Hyperparameter_Optimization_{protein_name or uniprot_id or 'unknown'}")
                parent_run = mlflow.start_run(run_name=f"opt_{protein_name or uniprot_id or 'unknown'}_{model_type}")
                parent_run_id = parent_run.info.run_id
                mlflow_run_active = True
                mlflow.log_param("n_trials", self.n_trials)
                mlflow.log_param("search_strategy", self.search_strategy)
                mlflow.log_param("train_samples", len(train_df))
                mlflow.log_param("val_samples", len(val_df))
                mlflow.log_param("test_samples", len(test_df))
            except Exception as e:
                self.logger.warning(f"Failed to start MLflow parent run: {e}")
                mlflow_run_active = False
        
        try:
            for trial_idx in range(self.n_trials):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Trial {trial_idx + 1}/{self.n_trials}")
                self.logger.info(f"{'='*60}")
                
                # Sample hyperparameters
                params = self._sample_parameters(trial_idx)
                self.logger.info(f"Sampled parameters: {params}")
                
                # Create trainer with sampled parameters (use shorter epochs for optimization)
                trainer = ChempropTrainer(
                    max_epochs=trial_epochs,  # Use shorter epochs during optimization
                    n_classes=n_classes,
                    batch_size=int(params['batch_size']),
                    bioactivity_loader=self.bioactivity_loader,
                    include_esm=include_esm,
                    val_fraction=val_fraction,  # Use external validation set
                    max_lr=float(params['max_lr']),
                    init_lr=float(params['init_lr']),
                    final_lr=float(params['final_lr']),
                    warmup_epochs=int(params['warmup_epochs']),
                    ffn_num_layers=int(params['ffn_num_layers']),
                    hidden_size=int(params['hidden_size']),
                    dropout=float(params['dropout']),
                    activation=str(params['activation']),
                    aggregation=str(params['aggregation']),
                    depth=int(params['depth']),
                    bias=bool(params['bias'])
                )
                
                # Train on training set, evaluate on validation set
                try:
                    result = trainer.train(
                        train_df=train_df,
                        test_df=val_df,  # Use validation set for evaluation during optimization
                        model_type=model_type,
                        protein_name=protein_name,
                        thresholds=thresholds
                    )
                    
                    if result['status'] == 'success':
                        # Use F1-macro as optimization metric
                        val_score = result.get('test_f1_macro', result.get('test_f1', 0.0))
                        
                        self.logger.info(f"Trial {trial_idx + 1} - Validation F1-macro: {val_score:.4f}")
                        
                        # Log to MLflow
                        if mlflow_run_active:
                            try:
                                with mlflow.start_run(run_name=f"trial_{trial_idx + 1}", 
                                                    nested=True, 
                                                    experiment_id=mlflow.get_experiment_by_name(
                                                        f"Hyperparameter_Optimization_{protein_name or uniprot_id or 'unknown'}"
                                                    ).experiment_id if mlflow.get_experiment_by_name(
                                                        f"Hyperparameter_Optimization_{protein_name or uniprot_id or 'unknown'}"
                                                    ) else None):
                                    # Log all parameters
                                    for key, value in params.items():
                                        mlflow.log_param(key, value)
                                    
                                    # Log metrics
                                    mlflow.log_metric("val_f1_macro", val_score)
                                    mlflow.log_metric("val_f1_weighted", result.get('test_f1', 0.0))
                                    mlflow.log_metric("val_accuracy", result.get('test_accuracy', 0.0))
                            except Exception as e:
                                self.logger.warning(f"Failed to log trial to MLflow: {e}")
                        
                        # Track best result
                        if val_score > best_score:
                            best_score = val_score
                            best_params = params.copy()
                            best_result = result
                            self.logger.info(f"*** New best score: {best_score:.4f} ***")
                        
                        all_results.append({
                            'trial': trial_idx + 1,
                            'params': params.copy(),
                            'val_f1_macro': val_score,
                            'val_f1_weighted': result.get('test_f1', 0.0),
                            'val_accuracy': result.get('test_accuracy', 0.0),
                            'status': 'success'
                        })
                    else:
                        self.logger.warning(f"Trial {trial_idx + 1} failed: {result.get('message', 'Unknown error')}")
                        all_results.append({
                            'trial': trial_idx + 1,
                            'params': params.copy(),
                            'status': 'failed',
                            'error': result.get('message', 'Unknown error')
                        })
                
                except Exception as e:
                    self.logger.error(f"Trial {trial_idx + 1} raised exception: {e}", exc_info=True)
                    all_results.append({
                        'trial': trial_idx + 1,
                        'params': params.copy(),
                        'status': 'error',
                        'error': str(e)
                    })
            
            # After optimization, retrain best model on combined train+val and evaluate on test
            if best_params is not None:
                self.logger.info(f"\n{'='*60}")
                self.logger.info("Retraining best model on combined train+val set")
                self.logger.info(f"Best parameters: {best_params}")
                self.logger.info(f"Best validation F1-macro: {best_score:.4f}")
                self.logger.info(f"{'='*60}")
                
                # Combine train and validation sets
                combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
                
                # Create final trainer with best parameters (use full max_epochs for final model)
                final_trainer = ChempropTrainer(
                    max_epochs=max_epochs,  # Use full epochs for final model
                    n_classes=n_classes,
                    batch_size=int(best_params['batch_size']),
                    bioactivity_loader=self.bioactivity_loader,
                    include_esm=include_esm,
                    val_fraction=0.0,  # No internal validation split
                    max_lr=float(best_params['max_lr']),
                    init_lr=float(best_params['init_lr']),
                    final_lr=float(best_params['final_lr']),
                    warmup_epochs=int(best_params['warmup_epochs']),
                    ffn_num_layers=int(best_params['ffn_num_layers']),
                    hidden_size=int(best_params['hidden_size']),
                    dropout=float(best_params['dropout']),
                    activation=str(best_params['activation']),
                    aggregation=str(best_params['aggregation']),
                    depth=int(best_params['depth']),
                    bias=bool(best_params['bias'])
                )
                
                # Train on combined set, evaluate on test set
                final_result = final_trainer.train(
                    train_df=combined_train_df,
                    test_df=test_df,
                    model_type=model_type,
                    protein_name=protein_name,
                    thresholds=thresholds
                )
                
                if final_result['status'] == 'success':
                    self.logger.info(f"Final test F1-macro: {final_result.get('test_f1_macro', 0.0):.4f}")
                    self.logger.info(f"Final test accuracy: {final_result.get('test_accuracy', 0.0):.4f}")
                    
                    # Log final results to MLflow
                    if mlflow_run_active and parent_run_id is not None:
                        try:
                            # Use MLflowClient to log directly to parent run (more reliable)
                            client = MlflowClient()
                            
                            # Log best parameters
                            for k, v in best_params.items():
                                client.log_param(parent_run_id, f"best_{k}", str(v))
                            
                            # Log final metrics
                            client.log_metric(parent_run_id, "best_val_f1_macro", best_score)
                            client.log_metric(parent_run_id, "final_test_f1_macro", final_result.get('test_f1_macro', 0.0))
                            client.log_metric(parent_run_id, "final_test_f1_weighted", final_result.get('test_f1', 0.0))
                            client.log_metric(parent_run_id, "final_test_accuracy", final_result.get('test_accuracy', 0.0))
                            
                            self.logger.info(f"Logged final metrics to parent run {parent_run_id}")
                        except Exception as e:
                            self.logger.warning(f"Failed to log final results to MLflow: {e}")
                            import traceback
                            self.logger.debug(traceback.format_exc())
                
                return {
                    'status': 'success',
                    'best_params': best_params,
                    'best_val_score': best_score,
                    'final_test_result': final_result,
                    'all_trials': all_results,
                    'n_trials': self.n_trials,
                    'n_successful': len([r for r in all_results if r.get('status') == 'success'])
                }
            else:
                self.logger.error("No successful trials found!")
                return {
                    'status': 'error',
                    'message': 'No successful trials',
                    'all_trials': all_results
                }
        
        finally:
            if mlflow_run_active and parent_run is not None:
                try:
                    mlflow.end_run()
                except:
                    pass
