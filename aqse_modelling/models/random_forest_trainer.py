"""
RandomForest model trainer implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from sklearn.ensemble import RandomForestClassifier

from .base import ModelTrainer

logger = logging.getLogger(__name__)


class RandomForestTrainer(ModelTrainer):
    """RandomForest implementation of ModelTrainer"""
    
    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None, 
                 random_state: int = 42, class_weight: str = 'balanced_subsample'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
    
    def requires_precomputed_features(self) -> bool:
        return True
    
    def get_model_type_name(self) -> str:
        return "RandomForest"
    
    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
              model_type: str = 'A', protein_name: Optional[str] = None,
              threshold: Optional[str] = None, thresholds: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Train RandomForest model with comprehensive reporting
        """
        logger.info(f"Training {self.get_model_type_name()} {model_type} model...")
        
        try:
            # Get class mapping from thresholds (supports 2-class and 3-class)
            if thresholds and 'class_labels' in thresholds:
                class_labels = thresholds['class_labels']
                class_to_int = {label: i for i, label in enumerate(class_labels)}
                logger.info(f"Using class mapping: {class_to_int}")
            else:
                # Default to 3-class
                class_to_int = {'Low': 0, 'Medium': 1, 'High': 2}
                logger.info("Using default 3-class mapping")
            
            # Build X, y from pre-computed features
            X_train = np.stack(train_df['features'].to_numpy())
            y_train = train_df['class'].map(class_to_int).to_numpy()
            X_test = np.stack(test_df['features'].to_numpy())
            y_true = test_df['class'].map(class_to_int).to_numpy()

            # Train RandomForest
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight=self.class_weight
            )
            rf.fit(X_train, y_train)

            # Predict
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)
            
            logger.info(f"✓ {self.get_model_type_name()} training completed successfully!")
            logger.info(f"  Test samples: {len(y_true)}")
            logger.info(f"  Feature dimensions: {X_train.shape[1]}")
            
            # Get class_labels from thresholds if available
            class_labels = thresholds.get('class_labels', ['Low', 'Medium', 'High']) if thresholds else ['Low', 'Medium', 'High']
            n_classes = thresholds.get('n_classes', 3) if thresholds else 3
            
            return {
                'status': 'success',
                'model': rf,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'model_type_name': self.get_model_type_name(),
                'feature_dimensions': X_train.shape[1],
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_classes': n_classes,
                'class_labels': class_labels
            }
            
        except Exception as e:
            logger.error(f"Error training {self.get_model_type_name()} model: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_type_name': self.get_model_type_name()
            }
