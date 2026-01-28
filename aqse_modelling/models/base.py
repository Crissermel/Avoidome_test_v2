"""
Abstract base class for model trainers and factory for creating them from config.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class ModelTrainer(ABC):
    """Abstract base class for model trainers"""
    
    @abstractmethod
    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
              model_type: str = 'A', protein_name: Optional[str] = None,
              threshold: Optional[str] = None, thresholds: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Train a model and return results
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame  
            model_type: 'A' or 'B'
            protein_name: Protein name for logging
            threshold: Threshold for Model B
            thresholds: Dictionary with thresholds, n_classes, and class_labels
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with training results including model, metrics, etc.
        """
        pass
    
    @abstractmethod
    def requires_precomputed_features(self) -> bool:
        """
        Whether this trainer requires pre-computed features
        
        Returns:
            True if features need to be extracted beforehand, False if raw data is sufficient
        """
        pass
    
    @abstractmethod
    def get_model_type_name(self) -> str:
        """Get the name of the model type for logging and file naming"""
        pass


def create_model_trainer(config: Dict[str, Any], bioactivity_loader=None) -> "ModelTrainer":
    """
    Factory function to create model trainers based on configuration.
    
    This replaces the separate models/factory.py module so that trainer
    construction lives alongside the abstract base class.
    """
    from .random_forest_trainer import RandomForestTrainer
    from .chemprop_trainer import ChempropTrainer

    model_type = config.get('model_type', 'random_forest').lower()
    
    if model_type == 'random_forest':
        return RandomForestTrainer(
            n_estimators=config.get('rf_n_estimators', 500),
            max_depth=config.get('rf_max_depth', None),
            random_state=config.get('random_state', 42),
            class_weight=config.get('rf_class_weight', 'balanced_subsample')
        )
    elif model_type == 'chemprop':
        return ChempropTrainer(
            max_epochs=config.get('chemprop_max_epochs', 25),
            n_classes=config.get('n_classes', 3),
            batch_size=config.get('chemprop_batch_size', 32),
            bioactivity_loader=bioactivity_loader,
            include_esm=config.get('include_esm', False),
            val_fraction=config.get('chemprop_val_fraction', 0.2),
            max_lr=config.get('chemprop_max_lr', 0.0005),
            init_lr=config.get('chemprop_init_lr', 0.00005),
            final_lr=config.get('chemprop_final_lr', 0.00005),
            warmup_epochs=config.get('chemprop_warmup_epochs', 5),
            ffn_num_layers=config.get('chemprop_ffn_num_layers', 2),
            hidden_size=config.get('chemprop_hidden_size', 300),
            dropout=config.get('chemprop_dropout', 0.0),
            activation=config.get('chemprop_activation', 'ReLU'),
            aggregation=config.get('chemprop_aggregation', 'mean'),
            depth=config.get('chemprop_depth', 3),
            bias=config.get('chemprop_bias', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'random_forest', 'chemprop'")
