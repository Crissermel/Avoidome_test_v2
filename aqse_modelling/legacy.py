
"""
Requires: 
micromamba activate chemprop_env

# Initialize micromamba in your current shell: 
eval "$(micromamba shell hook --shell bash)"


"""

#Imports


from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
import yaml
from pathlib import Path
import warnings
import torch
import gzip
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse
import os
import random
from collections import Counter
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

# Suppress RDKit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from rdkit import Chem
from rdkit.Chem import AllChem

# Chemprop imports
try:
    from chemprop import data, models, featurizers, nn, utils
    from lightning import pytorch as pl
    CHEMPROP_AVAILABLE = True
except ImportError as e:
    CHEMPROP_AVAILABLE = False
    logging.warning(f"Chemprop not available: {e}")
    # Create dummy classes to avoid import errors
    class pl:
        class Trainer:
            pass


# Abstract base classes for modularity
from abc import ABC, abstractmethod

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
        logger = logging.getLogger(self.__class__.__name__)
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


class ChempropTrainer(ModelTrainer):
    """Chemprop implementation of ModelTrainer for QSAR (Model A) and PCM (Model B)
    
    This trainer is used when model_type is set to 'chemprop' in the config.
    Both Model A and Model B can use Chemprop, with Model B always including ESM embeddings.
    """
    
    def __init__(self, max_epochs: int = 25, n_classes: int = 3, 
                 batch_size: int = 32, bioactivity_loader=None,
                 include_esm: bool = True, val_fraction: float = 0.2,
                 max_lr: float = 0.0005, init_lr: float = 0.00005, final_lr: float = 0.00005,
                 warmup_epochs: int = 5, ffn_num_layers: int = 2, hidden_size: int = 300,
                 dropout: float = 0.0, activation: str = "ReLU", aggregation: str = "mean",
                 depth: int = 3, bias: bool = True):
        """
        Initialize Chemprop trainer
        
        Args:
            max_epochs: Maximum number of training epochs
            n_classes: Number of classes for classification (default: 3)
            batch_size: Batch size for training
            bioactivity_loader: BioactivityDataLoader instance for feature extraction
            include_esm: Whether to include ESM embeddings (for Model B/PCM)
            val_fraction: Fraction of training data to use for validation
            max_lr: Maximum learning rate
            init_lr: Initial learning rate
            final_lr: Final learning rate
            warmup_epochs: Number of warmup epochs
            ffn_num_layers: Number of feed-forward layers
            hidden_size: Hidden layer size
            dropout: Dropout rate
            activation: Activation function name ("ReLU", "LeakyReLU", etc.)
            aggregation: Aggregation type ("mean" or "norm")
            depth: Message passing depth
            bias: Whether to use bias in layers
        """
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.bioactivity_loader = bioactivity_loader
        self.include_esm = include_esm
        self.val_fraction = val_fraction
        self.max_lr = max_lr
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.ffn_num_layers = ffn_num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.aggregation = aggregation
        self.depth = depth
        self.bias = bias
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not CHEMPROP_AVAILABLE:
            raise ImportError("Chemprop is not available. Please install it first.")
        
        if bioactivity_loader is None:
            self.logger.warning("No bioactivity_loader provided. Feature extraction may fail.")
    
    def requires_precomputed_features(self) -> bool:
        return False  # Chemprop uses SMILES directly, but we add extra features
    
    def get_model_type_name(self) -> str:
        return "Chemprop"
    
    def get_combined_features(self, df: pd.DataFrame, include_esm: bool = False) -> Tuple[np.ndarray, List[int]]:
        """
        Combines features for Chemprop from existing extractors.
        
        Args:
            df: DataFrame with 'SMILES' column and optionally 'accession' column
            include_esm: Whether to include ESM embeddings (Model B)
        
        Returns:
            Tuple of (features_array, valid_indices) where:
            - features_array: numpy array of shape (n_valid_samples, n_features)
            - valid_indices: list of original DataFrame indices that were valid
        """
        if self.bioactivity_loader is None:
            raise ValueError("bioactivity_loader is required for feature extraction")
        
        features_list = []
        valid_indices = []
        
        # Track missing accessions to avoid repeated warnings
        missing_accessions = set()
        missing_smiles_count = 0
        
        # Cache ESM embeddings per accession to avoid repeated file loads
        esm_cache = {}
        
        # Get descriptor keys in sorted order for consistency
        sample_smiles = df['SMILES'].iloc[0]
        sample_desc = self.bioactivity_loader.calculate_physicochemical_descriptors(sample_smiles)
        if not sample_desc:
            raise ValueError(f"Failed to calculate descriptors for sample SMILES: {sample_smiles}")
        descriptor_keys = sorted(sample_desc.keys())
        
        # Pre-load ESM embeddings for all unique accessions (if needed)
        if include_esm:
            unique_accessions = df['accession'].dropna().unique()
            self.logger.debug(f"Pre-loading ESM embeddings for {len(unique_accessions)} unique accessions...")
            for accession in unique_accessions:
                if accession and pd.notna(accession):
                    esm_feats = self.bioactivity_loader.load_esmc_descriptors(accession)
                    if esm_feats is not None:
                        # Ensure 1D array and cache
                        if esm_feats.ndim > 1:
                            esm_feats = esm_feats.flatten()
                        esm_cache[str(accession)] = esm_feats.astype(np.float32)
                    else:
                        missing_accessions.add(accession)
        
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            
            # Get physicochemical descriptors (compute on-the-fly)
            physchem_desc = self.bioactivity_loader.calculate_physicochemical_descriptors(smiles)
            if not physchem_desc:
                missing_smiles_count += 1
                continue
            
            # Convert dict to array (maintain consistent order)
            physchem_array = np.array([physchem_desc[k] for k in descriptor_keys], dtype=np.float32)
            
            if include_esm:
                # Get ESM embeddings from cache
                accession = row.get('accession')
                if not accession or pd.isna(accession):
                    missing_smiles_count += 1
                    continue
                
                accession_str = str(accession)
                if accession_str not in esm_cache:
                    # This shouldn't happen if pre-loading worked, but handle it
                    esm_feats = self.bioactivity_loader.load_esmc_descriptors(accession)
                    if esm_feats is None:
                        continue
                    if esm_feats.ndim > 1:
                        esm_feats = esm_feats.flatten()
                    esm_cache[accession_str] = esm_feats.astype(np.float32)
                
                esm_feats = esm_cache[accession_str]
                
                # Concatenate
                combined = np.hstack([physchem_array, esm_feats])
            else:
                combined = physchem_array
            
            features_list.append(combined)
            valid_indices.append(idx)
        
        # Log summary of missing data instead of per-row warnings
        total_rows = len(df)
        valid_rows = len(features_list)
        skipped_rows = total_rows - valid_rows
        
        if skipped_rows > 0:
            self.logger.info(f"Feature extraction: {valid_rows}/{total_rows} valid samples")
            if missing_smiles_count > 0:
                self.logger.info(f"  - {missing_smiles_count} samples skipped due to invalid SMILES or missing descriptors")
            if missing_accessions:
                accessions_str = ', '.join(sorted(list(missing_accessions))[:5])  # Show first 5
                if len(missing_accessions) > 5:
                    accessions_str += f", ... ({len(missing_accessions)} total)"
                self.logger.info(f"  - Missing ESM descriptors for accessions: {accessions_str}")
        
        if len(features_list) == 0:
            raise ValueError("No valid features extracted from DataFrame")
        
        return np.array(features_list, dtype=np.float32), valid_indices
    
    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
              model_type: str = 'A', protein_name: Optional[str] = None,
              threshold: Optional[str] = None, thresholds: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Train Chemprop model with optional physicochemical descriptors and ESM embeddings
        
        Args:
            train_df: Training DataFrame with 'SMILES', 'class' columns, and optionally 'accession'
            test_df: Test DataFrame with same structure
            model_type: 'A' for QSAR (no ESM) or 'B' for PCM (with ESM)
            protein_name: Optional protein name for logging
            threshold: Optional threshold for Model B
            thresholds: Dictionary with thresholds, n_classes, and class_labels
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with training results including model, metrics, predictions
        """
        self.logger.info(f"Starting Chemprop training (Model {model_type})")
        
        # Determine if ESM should be included based on model_type
        include_esm = model_type == 'B' or self.include_esm
        
        # Get n_classes and class mapping from thresholds
        original_n_classes = self.n_classes  # Always save original
        if thresholds and 'n_classes' in thresholds:
            n_classes = thresholds['n_classes']
            class_labels = thresholds.get('class_labels', ['Low', 'Medium', 'High'][:n_classes])
            # Temporarily update n_classes for this training run
            self.n_classes = n_classes
            self.logger.info(f"Using {n_classes}-class classification with labels: {class_labels}")
        else:
            n_classes = self.n_classes
            class_labels = ['Low', 'Medium', 'High']
            self.logger.info(f"Using default {n_classes}-class classification")
        
        try:
            # Map class labels to integers (dynamic based on thresholds)
            class_to_int = {label: i for i, label in enumerate(class_labels)}
            self.logger.info(f"Class mapping: {class_to_int}")
            
            # Extract SMILES and labels
            train_smiles = train_df['SMILES'].tolist()
            test_smiles = test_df['SMILES'].tolist()
            
            # Extract features
            self.logger.info("Extracting features for training set...")
            train_features, train_valid_indices = self.get_combined_features(train_df, include_esm=include_esm)
            train_df_valid = train_df.loc[train_valid_indices].reset_index(drop=True)
            
            # Map class labels and validate
            y_train_mapped = train_df_valid['class'].map(class_to_int)
            # Filter out any NaN values (invalid class labels)
            valid_train_mask = ~y_train_mapped.isna()
            if not valid_train_mask.all():
                n_invalid = (~valid_train_mask).sum()
                self.logger.warning(f"Filtering {n_invalid} training samples with invalid class labels")
                train_df_valid = train_df_valid[valid_train_mask].reset_index(drop=True)
                train_features = train_features[valid_train_mask.numpy() if hasattr(valid_train_mask, 'numpy') else valid_train_mask.values]
            
            train_smiles_valid = train_df_valid['SMILES'].tolist()
            y_train = y_train_mapped[valid_train_mask].values.astype(np.int64)
            
            # Validate label range
            if y_train.max() >= self.n_classes or y_train.min() < 0:
                self.logger.error(f"Invalid labels detected: min={y_train.min()}, max={y_train.max()}, expected range=[0, {self.n_classes-1}]")
                raise ValueError(f"Labels out of range: must be in [0, {self.n_classes-1}]")
            
            self.logger.info("Extracting features for test set...")
            test_features, test_valid_indices = self.get_combined_features(test_df, include_esm=include_esm)
            test_df_valid = test_df.loc[test_valid_indices].reset_index(drop=True)
            
            # Map class labels and validate
            y_test_mapped = test_df_valid['class'].map(class_to_int)
            # Filter out any NaN values (invalid class labels)
            valid_test_mask = ~y_test_mapped.isna()
            if not valid_test_mask.all():
                n_invalid = (~valid_test_mask).sum()
                self.logger.warning(f"Filtering {n_invalid} test samples with invalid class labels")
                test_df_valid = test_df_valid[valid_test_mask].reset_index(drop=True)
                test_features = test_features[valid_test_mask.numpy() if hasattr(valid_test_mask, 'numpy') else valid_test_mask.values]
            
            test_smiles_valid = test_df_valid['SMILES'].tolist()
            y_test = y_test_mapped[valid_test_mask].values.astype(np.int64)
            
            # Validate label range
            if y_test.max() >= self.n_classes or y_test.min() < 0:
                self.logger.error(f"Invalid labels detected: min={y_test.min()}, max={y_test.max()}, expected range=[0, {self.n_classes-1}]")
                raise ValueError(f"Labels out of range: must be in [0, {self.n_classes-1}]")
            
            self.logger.info(f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")
            self.logger.info(f"Train samples: {len(train_df_valid)}, Test samples: {len(test_df_valid)}")
            
            # Split train data into train + validation if needed
            if self.val_fraction > 0 and len(train_smiles_valid) > 10:
                # Split indices to maintain alignment with features and labels
                train_idx, val_idx = train_test_split(
                    range(len(train_smiles_valid)),
                    test_size=self.val_fraction,
                    random_state=42,
                    shuffle=True,
                    stratify=y_train  # Stratify by class to maintain distribution
                )
                
                # Split training data
                train_smiles_split = [train_smiles_valid[i] for i in train_idx]
                train_features_split = train_features[train_idx]
                y_train_split = y_train[train_idx]
                
                # Create validation data
                val_smiles = [train_smiles_valid[i] for i in val_idx]
                val_features = train_features[val_idx]
                y_val = y_train[val_idx]
            else:
                train_smiles_split = train_smiles_valid
                train_features_split = train_features
                y_train_split = y_train
                val_smiles = None
                val_features = None
                y_val = None
            
            # Create molecules from SMILES (needed for MoleculeDatapoint with x_d)
            # Filter out None molecules and track valid indices
            train_mols = []
            train_valid_indices = []
            train_labels_filtered = []
            train_features_filtered = []
            
            for i, smi in enumerate(train_smiles_split):
                mol = utils.make_mol(smi, keep_h=False, add_h=False)
                if mol is None:
                    self.logger.warning(f"Skipping invalid SMILES at training index {i}: {smi}")
                    continue
                y_int = int(y_train_split[i])
                if y_int < 0 or y_int >= self.n_classes:
                    self.logger.warning(f"Skipping invalid label {y_int} at training index {i}")
                    continue
                train_mols.append(mol)
                train_valid_indices.append(i)
                train_labels_filtered.append(y_int)
                train_features_filtered.append(train_features_split[i])
            
            test_mols = []
            test_valid_indices = []
            test_labels_filtered = []
            test_features_filtered = []
            
            for i, smi in enumerate(test_smiles_valid):
                mol = utils.make_mol(smi, keep_h=False, add_h=False)
                if mol is None:
                    self.logger.warning(f"Skipping invalid SMILES at test index {i}: {smi}")
                    continue
                y_int = int(y_test[i])
                if y_int < 0 or y_int >= self.n_classes:
                    self.logger.warning(f"Skipping invalid label {y_int} at test index {i}")
                    continue
                test_mols.append(mol)
                test_valid_indices.append(i)
                test_labels_filtered.append(y_int)
                test_features_filtered.append(test_features[i])
            
            val_data = None
            if val_smiles is not None:
                val_mols = []
                val_labels_filtered = []
                val_features_filtered = []
                
                for i, smi in enumerate(val_smiles):
                    mol = utils.make_mol(smi, keep_h=False, add_h=False)
                    if mol is None:
                        self.logger.warning(f"Skipping invalid SMILES at validation index {i}: {smi}")
                        continue
                    y_int = int(y_val[i])
                    if y_int < 0 or y_int >= self.n_classes:
                        self.logger.warning(f"Skipping invalid label {y_int} at validation index {i}")
                        continue
                    val_mols.append(mol)
                    val_labels_filtered.append(y_int)
                    val_features_filtered.append(val_features[i])
                
                if len(val_mols) > 0:
                    val_data = [
                        data.MoleculeDatapoint(mol, [y_int], x_d=val_features_filtered[i])
                        for i, (mol, y_int) in enumerate(zip(val_mols, val_labels_filtered))
                    ]
            
            # Create Chemprop datapoints with extra molecule features (x_d parameter)
            # According to Chemprop docs, extra molecule features are passed via x_d in MoleculeDatapoint
            train_data = [
                data.MoleculeDatapoint(mol, [y_int], x_d=train_features_filtered[i])
                for i, (mol, y_int) in enumerate(zip(train_mols, train_labels_filtered))
            ]
            
            test_data = [
                data.MoleculeDatapoint(mol, [y_int], x_d=test_features_filtered[i])
                for i, (mol, y_int) in enumerate(zip(test_mols, test_labels_filtered))
            ]
            
            # Convert filtered features to numpy arrays for validation
            if len(train_features_filtered) > 0:
                train_features_split = np.array(train_features_filtered)
            if len(test_features_filtered) > 0:
                test_features = np.array(test_features_filtered)
            
            # Create featurizer (no extra atom/bond features, only molecule-level)
            featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
            
            # Validate that we have data
            if len(train_data) == 0:
                raise ValueError("No valid training datapoints created")
            if len(test_data) == 0:
                raise ValueError("No valid test datapoints created")
            if val_data is not None and len(val_data) == 0:
                self.logger.warning("No valid validation datapoints, setting val_data to None")
                val_data = None
            
            # Verify label ranges in created datapoints BEFORE normalization
            train_labels_check = [int(dp.y[0]) for dp in train_data]
            test_labels_check = [int(dp.y[0]) for dp in test_data]
            val_labels_check = [int(dp.y[0]) for dp in val_data] if val_data is not None else []
            
            if train_labels_check:
                self.logger.info(f"Train label range: [{min(train_labels_check)}, {max(train_labels_check)}], expected: [0, {self.n_classes-1}]")
            if test_labels_check:
                self.logger.info(f"Test label range: [{min(test_labels_check)}, {max(test_labels_check)}], expected: [0, {self.n_classes-1}]")
            if val_labels_check:
                self.logger.info(f"Validation label range: [{min(val_labels_check)}, {max(val_labels_check)}], expected: [0, {self.n_classes-1}]")
            
            # Check for any invalid labels (-1 or out of range)
            invalid_train = [l for l in train_labels_check if l < 0 or l >= self.n_classes]
            invalid_test = [l for l in test_labels_check if l < 0 or l >= self.n_classes]
            invalid_val = [l for l in val_labels_check if l < 0 or l >= self.n_classes] if val_labels_check else []
            
            if invalid_train or invalid_test or invalid_val:
                error_msg = f"Invalid labels found before normalization: train={len(invalid_train)}, test={len(invalid_test)}, val={len(invalid_val)}"
                if invalid_train:
                    error_msg += f"\n  Train invalid labels: {set(invalid_train)}"
                if invalid_test:
                    error_msg += f"\n  Test invalid labels: {set(invalid_test)}"
                if invalid_val:
                    error_msg += f"\n  Val invalid labels: {set(invalid_val)}"
                raise ValueError(error_msg)
            
            self.logger.info(f"Created {len(train_data)} train, {len(test_data)} test datapoints")
            if val_data is not None:
                self.logger.info(f"Created {len(val_data)} validation datapoints")
            
            # Create datasets (features are already in datapoints via x_d)
            train_dataset = data.MoleculeDataset(train_data, featurizer)
            test_dataset = data.MoleculeDataset(test_data, featurizer)
            
            if val_data is not None:
                val_dataset = data.MoleculeDataset(val_data, featurizer)
            else:
                val_dataset = None
            
            # Normalize extra features (x_d) and targets
            # Note: For classification, we should NOT normalize targets (they're class indices)
            # Only normalize the features
            extra_features_scaler = train_dataset.normalize_inputs("X_d")
            
            # DO NOT normalize targets for classification - they're class indices [0, 1, 2], not continuous values
            # targets_scaler = train_dataset.normalize_targets()  # SKIP for classification
            
            # Apply feature scalers to validation and test sets
            if val_dataset is not None:
                val_dataset.normalize_inputs("X_d", extra_features_scaler)
                # Don't normalize validation targets either
            test_dataset.normalize_inputs("X_d", extra_features_scaler)
            # Don't normalize test targets either
            
            # Verify labels AFTER any processing (in case something modified them)
            train_labels_after = [int(dp.y[0]) for dp in train_dataset]
            test_labels_after = [int(dp.y[0]) for dp in test_dataset]
            val_labels_after = [int(dp.y[0]) for dp in val_dataset] if val_dataset is not None else []
            
            # Check for -1 values (padding/mask) or out of range
            if any(l == -1 for l in train_labels_after + test_labels_after + val_labels_after):
                invalid_indices = []
                for i, l in enumerate(train_labels_after):
                    if l == -1:
                        invalid_indices.append(f"train[{i}]")
                for i, l in enumerate(test_labels_after):
                    if l == -1:
                        invalid_indices.append(f"test[{i}]")
                for i, l in enumerate(val_labels_after):
                    if l == -1:
                        invalid_indices.append(f"val[{i}]")
                raise ValueError(f"Found -1 labels (mask/padding) after processing: {invalid_indices[:10]}")
            
            if any(l < 0 or l >= self.n_classes for l in train_labels_after + test_labels_after + val_labels_after):
                raise ValueError(f"Invalid labels after processing: some labels are out of range [0, {self.n_classes-1}]")
            
            # Enable caching for efficiency
            train_dataset.cache = True
            if val_dataset is not None:
                val_dataset.cache = True
            test_dataset.cache = True
            
            # Calculate class weights BEFORE creating data loaders (for logging and potential use)
            from collections import Counter
            class_counts = Counter(y_train_split)
            total_samples = len(y_train_split)
            
            # Compute balanced class weights (sklearn formula) for reference
            class_weights = torch.ones(self.n_classes, dtype=torch.float32)
            for class_idx in range(self.n_classes):
                count = class_counts.get(class_idx, 1)  # Avoid division by zero
                # sklearn balanced: n_samples / (n_classes * count_of_class)
                class_weights[class_idx] = total_samples / (self.n_classes * count)
            
            # Normalize weights so they sum to n_classes (optional, but helps with stability)
            class_weights = class_weights / class_weights.sum() * self.n_classes
            
            self.logger.info(f"Class distribution: {dict(class_counts)}")
            self.logger.info(f"Class weights (for reference): {class_weights.tolist()}")
            self.logger.warning("Note: Chemprop doesn't support class weights directly. Consider:")
            self.logger.warning("  - Using more training epochs (50-100)")
            self.logger.warning("  - Adjusting learning rates")
            
            # Create data loaders
            train_loader = data.build_dataloader(
                train_dataset, 
                batch_size=self.batch_size, 
                num_workers=0, 
                shuffle=True
            )
            test_loader = data.build_dataloader(
                test_dataset, 
                batch_size=self.batch_size, 
                num_workers=0, 
                shuffle=False
            )
            
            if val_dataset is not None:
                val_loader = data.build_dataloader(
                    val_dataset,
                    batch_size=self.batch_size,
                    num_workers=0,
                    shuffle=False
                )
            else:
                val_loader = None
            
            # Create transform for extra features (x_d) 
            # This applies the scaler transformation during forward pass
            X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_features_scaler)
            
            # Create MPNN components with configurable parameters
            # Message passing layer with depth
            mp = nn.BondMessagePassing(
                d_v=featurizer.atom_fdim,
                d_e=featurizer.bond_fdim,
                depth=self.depth
            )
            
            # Aggregation layer (mean or norm)
            if self.aggregation.lower() == "norm":
                agg = nn.NormAggregation()
            else:
                agg = nn.MeanAggregation()
            
            # Get feature dimension for FFN input
            # FFN input = message passing output + extra datapoint descriptors
            mp_output_dim = mp.output_dim
            extra_features_dim = train_features.shape[1]
            ffn_input_dim = mp_output_dim + extra_features_dim
            
            self.logger.info(f"MP output dim: {mp_output_dim}, Extra features dim: {extra_features_dim}, FFN input dim: {ffn_input_dim}")
            self.logger.info(f"Architecture: depth={self.depth}, hidden_size={self.hidden_size}, "
                           f"ffn_layers={self.ffn_num_layers}, dropout={self.dropout}, activation={self.activation}")
            
            # Create FFN with configurable architecture
            # Note: Check if MulticlassClassificationFFN supports these parameters
            # If not, we'll need to use a custom FFN or pass them differently
            try:
                # Try to create FFN with all parameters
                ffn = nn.MulticlassClassificationFFN(
                    n_classes=self.n_classes, 
                    input_dim=ffn_input_dim,
                    hidden_dim=self.hidden_size,
                    num_layers=self.ffn_num_layers,
                    dropout=self.dropout,
                    activation=self.activation,
                    bias=self.bias
                )
            except TypeError as e: ##### check
                # If parameters not supported, create with basic parameters and log warning
                self.logger.warning(f"Some FFN parameters not supported, using defaults: {e}")
                ffn = nn.MulticlassClassificationFFN(
                    n_classes=self.n_classes, 
                    input_dim=ffn_input_dim
                )
            
            # Create the complete model with X_d_transform for extra features
            mpnn = models.MPNN(
                message_passing=mp,
                agg=agg,
                predictor=ffn,
                batch_norm=False,
                warmup_epochs=self.warmup_epochs,
                init_lr=self.init_lr,
                max_lr=self.max_lr,
                final_lr=self.final_lr,
                X_d_transform=X_d_transform
            )
            
            # Check for CUDA errors before training - if there are any, use CPU
            # Default to CPU to avoid CUDA errors
            use_gpu = False
            
            # Only attempt GPU if explicitly needed and available
            # For now, default to CPU to avoid CUDA assertion errors until we can debug label issues
            if torch.cuda.is_available() and False:  # Temporarily disable GPU
                try:
                    # Test CUDA with a simple operation (don't clear cache if CUDA is in error state)
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor
                    torch.cuda.synchronize()
                    
                    # Only use GPU if test passed and we have enough data
                    use_gpu = len(train_data) > 10 and len(test_data) > 5
                except (RuntimeError, Exception) as e:
                    self.logger.warning(f"CUDA not available or in error state: {e}. Using CPU.")
                    use_gpu = False
            
            if not use_gpu:
                self.logger.info("Using CPU for training (GPU disabled or unavailable)")
            
            # Create PyTorch Lightning trainer
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator='gpu' if use_gpu else 'cpu',
                devices=1 if use_gpu else 1,  # CPU also needs devices=1 (number of CPU cores to use)
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False
            )
            
            # Train the model
            device_str = 'GPU' if use_gpu else 'CPU'
            self.logger.info(f"Training Chemprop model for {self.max_epochs} epochs... (using {device_str})")
            
            try:
                trainer.fit(mpnn, train_loader, val_dataloaders=val_loader if val_loader else None)
            except RuntimeError as train_err:
                if 'CUDA' in str(train_err):
                    self.logger.error(f"CUDA error during training: {train_err}")
                    self.logger.info("Falling back to CPU training...")
                    
                    # Try to reset CUDA state (but don't fail if it errors)
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except:
                        pass  # Ignore errors during cleanup
                    
                    # Retry with CPU
                    trainer = pl.Trainer(
                        max_epochs=self.max_epochs,
                        accelerator='cpu',
                        devices=1,  # CPU requires devices to be an integer
                        logger=False,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                        enable_checkpointing=False
                    )
                    
                    # Recreate model on CPU (may need to reset)
                    mpnn_cpu = models.MPNN(
                        message_passing=mp,
                        agg=agg,
                        predictor=ffn,
                        batch_norm=False,
                        warmup_epochs=2,
                        init_lr=0.0001,
                        max_lr=0.001,
                        final_lr=0.0001,
                        X_d_transform=X_d_transform
                    )
                    
                    trainer.fit(mpnn_cpu, train_loader, val_dataloaders=val_loader if val_loader else None)
                    mpnn = mpnn_cpu  # Use CPU model for evaluation
                else:
                    raise
            
            # Evaluate on test set using trainer.predict() method
            self.logger.info("Evaluating on test set...")
            
            # Get true labels from test dataset datapoints directly (before any processing)
            test_labels_from_data = [int(dp.y[0]) for dp in test_data]
            
            # Use trainer.predict() which properly handles batch conversion
            mpnn.eval()
            predictions = trainer.predict(mpnn, test_loader)
            
            # Concatenate predictions from all batches
            if isinstance(predictions, list) and len(predictions) > 0:
                # Stack predictions into a single tensor/array
                outputs = torch.cat(predictions, dim=0).cpu().numpy()
            else:
                outputs = np.array(predictions) if predictions is not None else np.array([])
            
            # Handle output shape - might be (n_samples,), (n_samples, n_classes), or (n_samples, 1, n_classes)
            if outputs.ndim == 1:
                # Single dimension output - use as class predictions
                y_pred = outputs.astype(int)
                # Create probability matrix from predictions
                y_prob = np.zeros((len(y_pred), self.n_classes))
                for i, pred_class in enumerate(y_pred):
                    y_prob[i, int(pred_class)] = 1.0
            elif outputs.ndim == 2:
                # Two dimensions - assume (n_samples, n_classes) logits
                y_pred = outputs.argmax(axis=-1).astype(int)
                y_prob = torch.softmax(torch.from_numpy(outputs), dim=-1).numpy()
            elif outputs.ndim == 3:
                # Three dimensions - likely (n_samples, 1, n_classes) - squeeze middle dimension
                if outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)  # Remove middle dimension -> (n_samples, n_classes)
                    y_pred = outputs.argmax(axis=-1).astype(int)
                    y_prob = torch.softmax(torch.from_numpy(outputs), dim=-1).numpy()
                else:
                    raise ValueError(f"Unexpected 3D prediction shape: {outputs.shape}")
            else:
                raise ValueError(f"Unexpected prediction shape: {outputs.shape}")
            
            # Get true labels (aligned with predictions)
            y_true = np.array(test_labels_from_data[:len(y_pred)])
            
            # Ensure all arrays have the same length
            min_len = min(len(y_true), len(y_pred), len(y_prob))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            y_prob = y_prob[:min_len]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            
            test_accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            confusion_mat = confusion_matrix(y_true, y_pred)
            
            self.logger.info(f"Chemprop training completed successfully!")
            self.logger.info(f"Test set size: {len(y_true)}")
            self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            self.logger.info(f"Test Precision (weighted): {precision:.4f}")
            self.logger.info(f"Test Recall (weighted): {recall:.4f}")
            self.logger.info(f"Test F1 (weighted): {f1:.4f}")
            
            # Restore original n_classes if it was changed
            if thresholds and 'n_classes' in thresholds and original_n_classes != n_classes:
                self.n_classes = original_n_classes
            
            return {
                'status': 'success',
                'model': mpnn,
                'trainer': trainer,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'test_accuracy': test_accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_precision_macro': precision_macro,
                'test_recall_macro': recall_macro,
                'test_f1_macro': f1_macro,
                'test_confusion_matrix': confusion_mat,
                'n_features': train_features.shape[1] if train_features.ndim > 1 else len(train_features[0]),
                'model_type_name': self.get_model_type_name(),
                'n_classes': n_classes,
                'class_labels': class_labels
            }
            
        except Exception as err:
            # Restore original n_classes if it was changed
            if thresholds and 'n_classes' in thresholds and original_n_classes != n_classes:
                self.n_classes = original_n_classes
            self.logger.error(f"Chemprop training failed: {err}", exc_info=True)
        return {
            'status': 'error',
                'message': f'Chemprop training failed: {str(err)}',
            'model_type_name': self.get_model_type_name()
        }


# Factory function for creating trainers
def create_model_trainer(config: Dict[str, Any], bioactivity_loader=None) -> ModelTrainer:
    """
    Factory function to create model trainers based on configuration
    
    Args:
        config: Configuration dictionary
        bioactivity_loader: Optional BioactivityDataLoader instance (required for Chemprop)
        
    Returns:
        ModelTrainer instance
    """
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


def create_fixed_test_set(df: pd.DataFrame, doc_id_column: str = 'doc_id', 
                          min_molecules_per_doc: int = 20, 
                          molecules_per_doc: int = 2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a fixed test set by selecting 1-2 molecules per doc_id that has more than min_molecules_per_doc molecules.
    
    This ensures that molecules from the same document (publication) are not split across train/test,
    which helps prevent data leakage.
    
    Args:
        df: DataFrame with bioactivity data, must contain doc_id_column
        doc_id_column: Name of the column containing document IDs
        min_molecules_per_doc: Minimum number of molecules a doc_id must have to contribute to test set
        molecules_per_doc: Number of molecules to select per doc_id (1 or 2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (remaining_df, test_df) where test_df contains fixed test molecules
    """
    logger = logging.getLogger(__name__)
    
    # Check if doc_id column exists
    if doc_id_column not in df.columns:
        logger.warning(f"Column '{doc_id_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        logger.warning("Falling back to stratified split without doc_id grouping")
        remaining_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['class'] if 'class' in df.columns else None,
            random_state=random_state
        )
        return remaining_df, test_df
    
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Count molecules per doc_id
    doc_counts = df[doc_id_column].value_counts()
    logger.info(f"Total unique doc_ids: {len(doc_counts)}")
    logger.info(f"doc_ids with >= {min_molecules_per_doc} molecules: {(doc_counts >= min_molecules_per_doc).sum()}")
    
    # Select doc_ids that have enough molecules
    eligible_docs = doc_counts[doc_counts >= min_molecules_per_doc].index.tolist()
    
    if len(eligible_docs) == 0:
        logger.warning(f"No doc_ids with >= {min_molecules_per_doc} molecules. Using stratified split instead.")
        remaining_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['class'] if 'class' in df.columns else None,
            random_state=random_state
        )
        return remaining_df, test_df
    
    # For each eligible doc_id, randomly select molecules_per_doc molecules for test set
    test_indices = []
    for doc_id in eligible_docs:
        doc_molecules = df[df[doc_id_column] == doc_id].index.tolist()
        # Randomly sample molecules_per_doc molecules (or all if fewer available)
        n_select = min(molecules_per_doc, len(doc_molecules))
        selected = random.sample(doc_molecules, n_select)
        test_indices.extend(selected)
    
    test_df = df.loc[test_indices].copy()
    remaining_df = df.drop(test_indices).copy()
    
    logger.info(f"Created fixed test set:")
    logger.info(f"  Test molecules: {len(test_df)} from {len(eligible_docs)} doc_ids")
    logger.info(f"  Remaining molecules (for train/val split): {len(remaining_df)}")
    logger.info(f"  Test set fraction: {len(test_df) / len(df):.4f}")
    
    # Log class distribution in both sets
    if 'class' in remaining_df.columns:
        remaining_class_dist = remaining_df['class'].value_counts().to_dict()
        test_class_dist = test_df['class'].value_counts().to_dict()
        logger.info(f"  Remaining class distribution: {remaining_class_dist}")
        logger.info(f"  Test class distribution: {test_class_dist}")
    
    return remaining_df, test_df


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
                
                # Create final trainer with best parameters
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
                            from mlflow.tracking import MlflowClient
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


# 2. Data loading layer
class AvoidomeDataLoader:
    """Handles avoidome protein data loading"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    
    def load_avoidome_targets(self) -> List[Dict[str, str]]:
        """Load avoidome protein targets"""
        self.logger.info("Loading avoidome protein targets...")
        filepath = self.config.get("avoidome_file")

        if not filepath:
            raise ValueError("Missing 'avoidome_file' in configuration.")

        df = pd.read_csv(filepath)

        # Clean columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        expected_cols = ["Name_2", "UniProt ID", "ChEMBL_target"]
        df = df[expected_cols]

        # Convert empty strings to NaN and drop missing
        df = df.replace("", np.nan)
        before_count = len(df)
        df = df.dropna(subset=["UniProt ID", "ChEMBL_target"])
        after_count = len(df)
        removed = before_count - after_count

        if removed > 0:
            self.logger.warning(f"Removed {removed} proteins due to missing UniProt ID or ChEMBL target.")
        else:
            self.logger.info("No proteins removed due to missing identifiers.")

        self.logger.info(f"Loaded {after_count} avoidome targets from {filepath}")
        return df.to_dict(orient="records")


    def load_avoidome_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load protein-specific thresholds
         Returns:
            Dict[str, Dict[str, float]]: protein_id → {metric_name: threshold_value}

        """
        self.logger.info("Loading avoidome thresholds...")
        filepath = self.config.get("similarity_file")
    
        if not filepath:
            raise ValueError("Missing 'similarity_file' in configuration.")
        df = pd.read_csv(filepath)

        # clean column names
        df.columns = df.columns.str.strip()


        # Identify ID column
        if "UniProt ID" in df.columns:
            id_col = "UniProt ID"
        elif "protein_id" in df.columns:
            id_col = "protein_id"
        else:
            raise ValueError("No protein ID column found in thresholds file.")

        df = df.set_index(id_col)
        thresholds = df.to_dict(orient="index")

        self.logger.info(f"Loaded threshold data for {len(thresholds)} proteins from {filepath}")
        return thresholds

class SimilarityDataLoader:
    """Handles similarity search results"""
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similarity_data: Dict[str, Dict[str, List[str]]] = {}  # will store the loaded results


    def load_similarity_results(self) -> Dict[str, Dict[str, List[str]]]:
        """Load similarity search results and store as nested dict."""
        filepath = self.config.get("similarity_file")
        if not filepath:
            raise ValueError("Missing 'similarity_file' in config.")

        self.logger.info(f"Loading similarity results from {filepath}")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        if "query_protein" not in df.columns or "threshold" not in df.columns or "similar_proteins" not in df.columns:
            raise ValueError("Expected columns: 'query_protein', 'threshold', 'similar_proteins'")

        # Build nested dictionary
        for _, row in df.iterrows():
            query = row["query_protein"]
            thresh = row["threshold"]
            similars_str = row.get("similar_proteins", "")
            similars = []

            if pd.notna(similars_str) and similars_str != "":
                for p in similars_str.split(","):
                    prot = p.strip().split(" ")[0]  # remove score in parentheses
                    prot = prot.split("_")[0]       # remove any suffix like _WT
                    if prot != query:               # remove query itself
                        similars.append(prot)

            if query not in self.similarity_data:
                self.similarity_data[query] = {}
            self.similarity_data[query][thresh] = similars
        self.logger.info(f"Loaded similarity data for {len(self.similarity_data)} proteins")
        return self.similarity_data


    def get_similar_proteins(self, uniprot_id: str) -> List[str]:
        """Get all similar proteins for a given UniProt ID across all thresholds."""
        if not self.similarity_data:
            self.load_similarity_results()
        if uniprot_id not in self.similarity_data:
            self.logger.warning(f"No similarity data found for {uniprot_id}")
            return []

        # Merge all lists from all thresholds
        similar_proteins = []
        for proteins in self.similarity_data[uniprot_id].values():
            similar_proteins.extend(proteins)
        return list(set(similar_proteins))  # remove duplicates

    def get_similar_proteins_for_threshold(self, uniprot_id: str, threshold: str) -> List[str]:
        """Get similar proteins for a specific threshold (e.g., 'high')."""
        if not self.similarity_data:
            self.load_similarity_results()
        if uniprot_id not in self.similarity_data:
            self.logger.warning(f"No similarity data found for {uniprot_id}")
            return []
        return self.similarity_data[uniprot_id].get(threshold, [])


class ProteinSequenceLoader:
    """Handles protein sequence loading"""
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.protein_sequence: Dict[str, str] = {}  # will store the loaded results
        

    def load_sequences(self) -> Dict[str, str]:
        """
        Load protein sequences from the CSV specified in config['sequence_file'].

        Returns:
            Dict[str, str]: {uniprot_id: sequence}
        """
        filepath = self.config.get("sequence_file")
        if not filepath:
            raise ValueError("Missing 'sequence_file' in config.")

        self.logger.info(f"Loading protein sequences from {filepath}")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        required_cols = ["uniprot_id", "protein_name", "sequence", "sequence_length"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Expected columns: {required_cols}")

        # Build dictionary
        self.protein_sequence = {}
        removed_count = 0
        for _, row in df.iterrows():
            seq = row["sequence"]
            uid = row["uniprot_id"]
            if pd.isna(seq) or seq == "" or pd.isna(uid) or uid == "":
                removed_count += 1
                continue
            self.protein_sequence[uid] = seq

        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} entries due to missing UniProt ID or sequence.")

        self.logger.info(f"Loaded {len(self.protein_sequence)} protein sequences")
        return self.protein_sequence

    def get_sequence(self, uniprot_id: str) -> str:
        """
        Get the protein sequence for a given UniProt ID.

        Args:
            uniprot_id (str): UniProt ID

        Returns:
            str: protein sequence or empty string if not found
        """
        if not self.protein_sequence:
            self.load_sequences()
        return self.protein_sequence.get(uniprot_id, "")



class UniqueProteinListGenerator:
    """Generates a unique list of proteins including avoidome and similar proteins"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_protein_ids_from_similar(self, similar_proteins_str: str) -> List[str]:
        """Extract UniProt IDs from the similar_proteins string
        
        Args:
            similar_proteins_str: String like "P20813_WT (100.0%), P24460_WT (78.0%)"
            
        Returns:
            List of UniProt IDs (with _WT suffix removed if present)
        """
        if pd.isna(similar_proteins_str) or similar_proteins_str == "":
            return []
        
        proteins = []
        for protein_entry in similar_proteins_str.split(","):
            # Extract protein ID (format: "P20813_WT (100.0%)")
            parts = protein_entry.strip().split(" ")[0]  # Get "P20813_WT"
            
            # Remove _WT suffix if present
            if "_WT" in parts:
                parts = parts.split("_WT")[0]
            
            proteins.append(parts)
        
        return proteins
    
    def generate_unique_protein_list(self) -> pd.DataFrame:
        """Generate a list of all unique proteins (avoidome + similar)
        
        Returns:
            DataFrame with columns: ['uniprot_id', 'protein_name', 'source']
                - uniprot_id: UniProt ID
                - protein_name: Name of the protein (if available)
                - source: Either 'avoidome' or 'similar_to_{query_protein}'
        """
        self.logger.info("Generating unique protein list...")
        
        # Load avoidome proteins
        avoidome_loader = AvoidomeDataLoader(self.config)
        avoidome_proteins = avoidome_loader.load_avoidome_targets()
        
        # Load similarity results
        similarity_loader = SimilarityDataLoader(self.config)
        similarity_data = similarity_loader.load_similarity_results()
        
        # Collect all proteins with their metadata
        unique_proteins = {}
        
        # Add avoidome proteins
        for protein in avoidome_proteins:
            uniprot_id = protein["UniProt ID"]
            protein_name = protein.get("Name_2", "")
            
            unique_proteins[uniprot_id] = {
                'uniprot_id': uniprot_id,
                'protein_name': protein_name,
                'source': 'avoidome'
            }
        
        # Add similar proteins from all thresholds
        for query_protein, thresholds in similarity_data.items():
            for threshold, similar_proteins in thresholds.items():
                for similar_protein in similar_proteins:
                    if similar_protein not in unique_proteins:
                        # Find protein name if available
                        protein_name = ""
                        for av_protein in avoidome_proteins:
                            if av_protein["UniProt ID"] == similar_protein:
                                protein_name = av_protein.get("Name_2", "")
                                break
                        
                        unique_proteins[similar_protein] = {
                            'uniprot_id': similar_protein,
                            'protein_name': protein_name,
                            'source': f'similar_to_{query_protein}'
                        }
        
        # Convert to DataFrame
        df = pd.DataFrame(list(unique_proteins.values()))
        df = df.sort_values('uniprot_id')
        
        self.logger.info(f"Generated unique protein list with {len(df)} proteins")
        self.logger.info(f"  - {len([p for p in unique_proteins.values() if p['source'] == 'avoidome'])} avoidome proteins")
        self.logger.info(f"  - {len([p for p in unique_proteins.values() if p['source'] != 'avoidome'])} similar proteins")
        
        return df
    
    def save_protein_list(self, output_path: str = None) -> str:
        """Generate and save the unique protein list to CSV
        
        Args:
            output_path: Path to save the CSV file. If None, saves to output_dir/protein_list_all_unique.csv
            
        Returns:
            Path to the saved file
        """
        df = self.generate_unique_protein_list()
        
        if output_path is None:
            output_dir = Path(self.config.get("output_dir", "."))
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "protein_list_all_unique.csv"
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved unique protein list to {output_path}")
        
        return str(output_path)



class BioactivityDataLoader:
    """Handles bioactivity data loading and processing"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up fingerprints directory
        output_dir = Path(config.get("output_dir", "."))
        self.fingerprints_dir = output_dir / "fingerprints"
        self.fingerprints_dir.mkdir(exist_ok=True)
        
        # Set up ESM-C cache directory
        self.esmc_cache_dir = Path(config.get("papyrus_cache_dir", output_dir / "esmc_cache"))
        self.esmc_cache_dir.mkdir(exist_ok=True)
        
        # Load Papyrus dataset ONCE at initialization
        self.logger.info("Loading Papyrus dataset...")
        self.logger.info("This may take ~5 minutes on first load")
        try:
            from papyrus_scripts import PapyrusDataset
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            self.papyrus_df = papyrus_data.to_dataframe()
            self.logger.info(f"Loaded {len(self.papyrus_df):,} total activities from Papyrus")
        except Exception as e:
            self.logger.error(f"Error loading Papyrus dataset: {e}")
            self.papyrus_df = pd.DataFrame()
    
    def calculate_morgan_fingerprints(self, unique_proteins: pd.DataFrame) -> Dict[str, any]:
        """
        Step 0: Calculate and save Morgan fingerprints for all compounds in Papyrus database,
        only for the compounds associated with the unique protein list
        """
        self.logger.info("Calculating Morgan fingerprints for all Papyrus compounds (filtered by unique proteins)")
        
        try:
            # Use the already-loaded Papyrus dataset
            papyrus_df = self.papyrus_df
            
            if papyrus_df.empty:
                self.logger.error("Papyrus dataset not loaded")
                return {}

            # Filter for valid SMILES
            valid_data = papyrus_df.dropna(subset=['SMILES'])
            valid_data = valid_data[valid_data['SMILES'] != '']

            # Filter by unique protein list
            unique_ids = set(unique_proteins['uniprot_id'])
            valid_data = valid_data[valid_data['accession'].isin(unique_ids)]
            self.logger.info(f"{len(valid_data)} activities after filtering by unique proteins")

            # Get unique SMILES
            unique_smiles = valid_data['SMILES'].unique()
            self.logger.info(f"Found {len(unique_smiles)} unique SMILES to process")

            # Calculate Morgan fingerprints
            fingerprints = {}
            valid_count = 0
            
            for i, smiles in enumerate(unique_smiles):
                if i % 10000 == 0:
                    self.logger.info(f"Processing SMILES {i+1}/{len(unique_smiles)}")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        morgan_array = np.array(morgan_fp, dtype=np.float32)
                        fingerprints[smiles] = morgan_array
                        valid_count += 1
                except Exception as e:
                    self.logger.warning(f"Error processing SMILES {smiles}: {e}")
                    continue

            # Save fingerprints as Parquet for efficiency
            fingerprints_file = self.fingerprints_dir / "papyrus_morgan_fingerprints.parquet"
            
            # Convert dict to DataFrame for Parquet
            fps_df = pd.DataFrame({
                'SMILES': list(fingerprints.keys()),
                'fingerprint': [fp.tolist() for fp in fingerprints.values()]  # Convert numpy array to list
            })
            
            # Save as Parquet with compression
            fps_df.to_parquet(fingerprints_file, compression='snappy')
            file_size_mb = fingerprints_file.stat().st_size / (1024 * 1024)

            self.logger.info(f"Bioactivity dataset completed: {valid_count} valid Morgan fingerprints saved to {fingerprints_file} ({file_size_mb:.2f} MB)")
            
            # Get dataset statistics
            total_activities = len(papyrus_df)
            filtered_activities = len(valid_data)
            
            return {
                'total_activities_in_papyrus': total_activities,
                'activities_after_protein_filter': filtered_activities,
                'unique_proteins_in_filter': len(unique_ids),
                'total_smiles': len(unique_smiles),
                'valid_fingerprints': valid_count,
                'fingerprints_file': str(fingerprints_file)
            }

        except Exception as e:
            self.logger.error(f"Error in Step 0: {e}")
            return {}

    def load_morgan_fingerprints(self, unique_proteins: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """Load pre-computed Morgan fingerprints
        
        Note: The fingerprints were already filtered during calculation to only include
        SMILES associated with the unique proteins list.
        """
        # Try Parquet first, fallback to pickle for backward compatibility
        fingerprints_file_parquet = self.fingerprints_dir / "papyrus_morgan_fingerprints.parquet"
        fingerprints_file_pkl = self.fingerprints_dir / "papyrus_morgan_fingerprints.pkl"
        
        try:
            # Try to load from Parquet
            if fingerprints_file_parquet.exists():
                fps_df = pd.read_parquet(fingerprints_file_parquet)
                # Convert back to dict
                fingerprints = {
                    row['SMILES']: np.array(row['fingerprint'], dtype=np.float32)
                    for _, row in fps_df.iterrows()
                }
                self.logger.info(f"Loaded {len(fingerprints)} Morgan fingerprints from Parquet")
                return fingerprints
            
            # Fallback to pickle
            elif fingerprints_file_pkl.exists():
                with open(fingerprints_file_pkl, 'rb') as f:
                    fingerprints = pickle.load(f)
                self.logger.info(f"Loaded {len(fingerprints)} Morgan fingerprints from Pickle (legacy format)")
                return fingerprints
            else:
                self.logger.error(f"Morgan fingerprints file not found (neither .parquet nor .pkl)")
                return {}
        
        except Exception as e:
            self.logger.error(f"Error loading Morgan fingerprints: {e}")
            return {}
    
    def calculate_physicochemical_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Calculate physicochemical descriptors for a single SMILES string.
        
        Uses the local physicochemical_descriptors module from aqse_modelling.utils.
        This ensures AQSE_v3 is self-contained with no external dependencies.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of descriptor names and values
        """
        try:
            from aqse_modelling.utils.physicochemical_descriptors import calculate_physicochemical_descriptors
            
            # Calculate descriptors (without SASA for speed)
            descriptors = calculate_physicochemical_descriptors(
                smiles, 
                include_sasa=False, 
                verbose=False
            )
            
            return descriptors
            
        except ImportError as e:
            self.logger.error(f"Failed to import physicochemical_descriptors from aqse_modelling.utils: {e}")
            self.logger.error("This module should be available in AQSE_v3/aqse_modelling/utils/physicochemical_descriptors.py")
            return {}
        except Exception as e:
            self.logger.error(f"Error calculating physicochemical descriptors for {smiles}: {e}")
            return {}
    
    def get_filtered_papyrus(self, uniprot_ids: List[str] = None) -> pd.DataFrame:
        """
        Get filtered Papyrus data for specific proteins.
        
        Args:
            uniprot_ids: List of UniProt IDs to filter by. If None, returns all data.
            
        Returns:
            Filtered DataFrame
        """
        if self.papyrus_df.empty:
            self.logger.error("Papyrus dataset not loaded")
            return pd.DataFrame()
        
        if uniprot_ids is None:
            return self.papyrus_df
        
        # Filter by protein IDs
        filtered_df = self.papyrus_df[self.papyrus_df['accession'].isin(uniprot_ids)]
        return filtered_df
    
    def load_esmc_descriptors(self, uniprot_id: str) -> Optional[np.ndarray]:
        """
        Load ESM-C descriptors from cache for a protein.
        
        Args:
            uniprot_id: UniProt ID of the protein
            
        Returns:
            ESM-C descriptor array or None if not found
        """
        esmc_file = self.esmc_cache_dir / f"{uniprot_id}_descriptors.pkl"
        
        if not esmc_file.exists():
            # Don't log error here - we'll summarize missing accessions at a higher level
            return None
        
        try:
            with open(esmc_file, 'rb') as f:
                esmc_data = pickle.load(f)
            
            # Extract ESM-C descriptors from the cached data
            esmc_descriptors = []
            
            if isinstance(esmc_data, dict):
                # If it's a dictionary, look for ESM-C descriptors
                for key, value in esmc_data.items():
                    if key.startswith('esm_dim_'):
                        esmc_descriptors.append(value)
                
                if esmc_descriptors:
                    esmc_embedding = np.array(esmc_descriptors)
                else:
                    # Look for any numeric keys or ESM-related keys
                    numeric_keys = [k for k in esmc_data.keys() if k.isdigit() or 'esm' in k.lower()]
                    if numeric_keys:
                        for key in sorted(numeric_keys):
                            if isinstance(esmc_data[key], (int, float)):
                                esmc_descriptors.append(esmc_data[key])
                    
                    if esmc_descriptors:
                        esmc_embedding = np.array(esmc_descriptors)
                    else:
                        # Don't log error here - we'll summarize missing accessions at a higher level
                        return None
                        
            elif isinstance(esmc_data, np.ndarray):
                esmc_embedding = esmc_data
                
            elif isinstance(esmc_data, pd.DataFrame):
                # Handle DataFrame format
                esm_cols = [col for col in esmc_data.columns if col.startswith('esm_dim_')]
                
                if esm_cols:
                    esmc_embedding = esmc_data[esm_cols].iloc[0].values
                else:
                    # Don't log error here - we'll summarize missing accessions at a higher level
                    return None
            else:
                # Log only unexpected formats as these are actual errors, not just missing files
                self.logger.warning(f"Unexpected ESM-C descriptor format for {uniprot_id}: {type(esmc_data)}")
                return None
            
            # Ensure it's a 1D array
            if esmc_embedding.ndim > 1:
                esmc_embedding = esmc_embedding.flatten()
            
            self.logger.info(f"Loaded ESM-C descriptors for {uniprot_id}: shape {esmc_embedding.shape}")
            return esmc_embedding
            
        except Exception as e:
            self.logger.error(f"Error loading ESM-C descriptors for {uniprot_id}: {e}")
            return None

class ActivityThresholdsLoader:
    """Handles activity thresholds data loading and processing with support for optimized cutoffs and 2-class classification"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load activity thresholds (optimized cutoffs)
        self._load_thresholds()
        
        # Load 2-class protein list
        self._load_2class_proteins()
    
    def _load_thresholds(self):
        """Load optimized activity thresholds from CSV file"""
        filepath = self.config.get("activity_thresholds_file")
        if not filepath:
            self.logger.warning("No 'activity_thresholds_file' in configuration")
            self.thresholds_df = pd.DataFrame()
            return
        
        try:
            self.logger.info(f"Loading optimized activity thresholds from {filepath}")
            df = pd.read_csv(filepath)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Use optimized cutoffs if available, otherwise fall back to original
            if 'optimized_cutoff_high' in df.columns:
                df['cutoff_high'] = df['optimized_cutoff_high']
                self.logger.info("Using optimized high cutoffs")
            if 'optimized_cutoff_medium' in df.columns:
                df['cutoff_medium'] = df['optimized_cutoff_medium']
                self.logger.info("Using optimized medium cutoffs")
            
            # Ensure use_2class column exists (for backward compatibility)
            if 'use_2class' not in df.columns:
                self.logger.warning("'use_2class' column not found in thresholds file. All proteins will use 3-class classification.")
                df['use_2class'] = False
            else:
                # Convert use_2class to boolean if it's stored as string
                df['use_2class'] = df['use_2class'].astype(bool)
            
            # Ensure best_2class_option and best_2class_cutoff columns exist
            if 'best_2class_option' not in df.columns:
                df['best_2class_option'] = ''
            if 'best_2class_cutoff' not in df.columns:
                df['best_2class_cutoff'] = ''
            
            # Set index for easy lookup
            df.set_index('uniprot_id', inplace=True, drop=False)
            
            self.thresholds_df = df
            self.logger.info(f"Loaded activity thresholds for {len(df)} proteins")
            
            # Count 2-class proteins
            n_2class = df['use_2class'].sum() if 'use_2class' in df.columns else 0
            self.logger.info(f"Found {n_2class} proteins recommended for 2-class classification")
        except Exception as e:
            self.logger.error(f"Error loading activity thresholds: {e}")
            self.thresholds_df = pd.DataFrame()
    
    def _load_2class_proteins(self):
        """
        Load list of proteins that should use 2-class classification.
        This method is now deprecated - use_2class flag is read directly from CSV.
        Kept for backward compatibility but no longer used.
        """
        # This method is kept for backward compatibility but is no longer used.
        # The use_2class flag is now read directly from the thresholds CSV file.
        self.two_class_proteins = set()  # Empty set - will be populated from CSV if needed
    
    def get_thresholds(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get activity thresholds for a specific protein.
        
        Args:
            uniprot_id: UniProt ID of the protein
            
        Returns:
            Dictionary with 'high' and 'medium' cutoff values, 'n_classes' (2 or 3), 
            and 'class_labels' list, or empty dict if not found
        """
        if self.thresholds_df.empty:
            return {}
        
        try:
            # Look up by UniProt ID
            protein = self.thresholds_df[self.thresholds_df['uniprot_id'] == uniprot_id]
            
            if protein.empty:
                self.logger.warning(f"No thresholds found for {uniprot_id}")
                return {}
            
            row = protein.iloc[0]
            cutoff_high = float(row['cutoff_high'])
            
            # Check if this protein should use 2-class classification (from CSV)
            use_2class = False
            if 'use_2class' in row:
                use_2class = bool(row['use_2class'])
            
            # Fallback to hardcoded list if CSV doesn't have use_2class (backward compatibility)
            if 'use_2class' not in row.index and uniprot_id in self.two_class_proteins:
                use_2class = True
                self.logger.warning(f"Using hardcoded 2-class list for {uniprot_id} (CSV missing use_2class column)")
            
            if use_2class:
                # 2-class: determine which cutoff to use and class labels
                best_2class_option = row.get('best_2class_option', 'Low+Medium vs High')
                best_2class_cutoff = row.get('best_2class_cutoff', 'high')
                
                # Determine class labels based on best option
                if best_2class_option == 'Low+Medium vs High':
                    class_labels = ['Low+Medium', 'High']
                    cutoff_to_use = cutoff_high
                elif best_2class_option == 'Low vs Medium+High':
                    class_labels = ['Low', 'Medium+High']
                    cutoff_to_use = float(row.get('cutoff_medium', cutoff_high))
                elif best_2class_option == 'Medium vs High':
                    class_labels = ['Medium', 'High']
                    cutoff_to_use = cutoff_high
                else:
                    # Default: Low+Medium vs High
                    class_labels = ['Low+Medium', 'High']
                    cutoff_to_use = cutoff_high
                
                self.logger.info(f"Using 2-class classification for {uniprot_id}: {best_2class_option} (cutoff={cutoff_to_use:.1f})")
                return {
                    'high': cutoff_high,
                    'medium': float(row.get('cutoff_medium', cutoff_high)) if best_2class_cutoff == 'medium' else None,
                    'n_classes': 2,
                    'class_labels': class_labels,
                    'best_2class_option': best_2class_option,
                    'best_2class_cutoff': best_2class_cutoff
                }
            else:
                # 3-class: use both cutoffs
                cutoff_medium = float(row['cutoff_medium'])
                return {
                    'high': cutoff_high,
                    'medium': cutoff_medium,
                    'n_classes': 3,
                    'class_labels': ['Low', 'Medium', 'High']
                }
        except Exception as e:
            self.logger.error(f"Error getting thresholds for {uniprot_id}: {e}")
            return {}
    
    def get_thresholds_by_name(self, protein_name: str) -> Dict[str, Any]:
        """
        Get activity thresholds for a specific protein by name.
        
        Args:
            protein_name: Name of the protein (name2_entry column)
            
        Returns:
            Dictionary with 'high' and 'medium' cutoff values, 'n_classes', and 'class_labels', 
            or empty dict if not found
        """
        if self.thresholds_df.empty:
            return {}
        
        try:
            protein = self.thresholds_df[self.thresholds_df['name2_entry'] == protein_name]
            
            if protein.empty:
                self.logger.warning(f"No thresholds found for {protein_name}")
                return {}
            
            row = protein.iloc[0]
            uniprot_id = row['uniprot_id']
            # Use get_thresholds to get the full dict with n_classes
            return self.get_thresholds(uniprot_id)
        except Exception as e:
            self.logger.error(f"Error getting thresholds for {protein_name}: {e}")
            return {}
    
    def get_all_proteins(self) -> List[str]:
        """Get list of all proteins with thresholds"""
        if self.thresholds_df.empty:
            return []
        return self.thresholds_df['uniprot_id'].dropna().tolist()


class AQSE3CWorkflow:
    """
    AQSE 3-Class Workflow Orchestrator
    
    Processes all proteins from avoidome list and trains models:
    - Model A: Simple QSAR for proteins without similar proteins (configurable: RF or Chemprop)
    - Model B: PCM model with similar proteins for 3 thresholds (configurable: RF or Chemprop)
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize loaders
        self.avoidome_loader = AvoidomeDataLoader(config)
        self.similarity_loader = SimilarityDataLoader(config)
        self.bioactivity_loader = BioactivityDataLoader(config)
        self.thresholds_loader = ActivityThresholdsLoader(config)
        
        # Initialize model trainer (pass bioactivity_loader for Chemprop)
        self.model_trainer = create_model_trainer(config, bioactivity_loader=self.bioactivity_loader)
        self.logger.info(f"Using {self.model_trainer.get_model_type_name()} trainer")
        
        # Initialize model reporter
        # Include model type in results directory path
        model_type = config.get("model_type", "random_forest")
        results_dir = Path(config.get("output_dir", ".")) / f"results_{model_type}"
        self.model_reporter = ModelReporter(results_dir)
        
        # Initialize MLflow
        self._setup_mlflow(config, model_type)
        
        # Load data
        self.targets = self.avoidome_loader.load_avoidome_targets()
        self.similarity_loader.load_similarity_results()
        
        # Cache fingerprints ONCE at initialization (only if trainer requires precomputed features)
        if self.model_trainer.requires_precomputed_features():
            self.logger.info("Loading Morgan fingerprints (will be cached for all proteins)...")
            self._cached_fingerprints = self.bioactivity_loader.load_morgan_fingerprints()
            self.logger.info(f"Cached {len(self._cached_fingerprints):,} fingerprints")
        else:
            self.logger.info("Model trainer does not require precomputed features - skipping fingerprint caching")
            self._cached_fingerprints = {}
        
        # Create output directory
        self.output_dir = Path(config.get("output_dir", "."))
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
    
    def _setup_mlflow(self, config: Dict[str, Any], model_type: str):
        """
        Setup MLflow tracking
        
        Args:
            config: Configuration dictionary
            model_type: Model type (e.g., 'random_forest', 'chemprop')
        """
        # Check if MLflow is available
        if not MLFLOW_AVAILABLE:
            self.logger.info("MLflow is not installed. MLflow tracking will be disabled.")
            self.mlflow_enabled = False
            return
        
        # Get MLflow configuration from config or use defaults
        mlflow_tracking_uri = config.get("mlflow_tracking_uri", None)
        mlflow_experiment_name = config.get("mlflow_experiment_name", f"AQSE_3C_2C_optimized_{model_type}")
        mlflow_enabled = config.get("mlflow_enabled", True)
        
        self.mlflow_enabled = mlflow_enabled
        
        if not mlflow_enabled:
            self.logger.info("MLflow tracking is disabled")
            return
        
        try:
            # Set tracking URI if provided
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                self.logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(mlflow_experiment_name)
                self.logger.info(f"Created MLflow experiment: {mlflow_experiment_name} (ID: {experiment_id})")
            except Exception:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                    self.logger.info(f"Using existing MLflow experiment: {mlflow_experiment_name} (ID: {experiment_id})")
                else:
                    raise
            
            self.mlflow_experiment_name = mlflow_experiment_name
            self.mlflow_experiment_id = experiment_id
            self.logger.info(f"MLflow tracking initialized for experiment: {mlflow_experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize MLflow: {e}. Continuing without MLflow tracking.")
            self.mlflow_enabled = False
    
    def _ensure_umap(self):
        try:
            import umap  # noqa: F401
            from umap import UMAP  # noqa: F401
            return True
        except Exception as e:
            self.logger.warning(f"UMAP not available ({e}); skipping visualization for this model.")
            return False

    def _compute_compound_feature_length(self, df: pd.DataFrame) -> Optional[int]:
        try:
            sample_smiles = None
            if 'SMILES' in df.columns:
                for s in df['SMILES']:
                    if isinstance(s, str) and s:
                        sample_smiles = s
                        break
            if sample_smiles is None:
                return None
            comp_vec = self.extract_compound_features(sample_smiles)
            return int(len(comp_vec)) if comp_vec is not None else None
        except Exception:
            return None

    def _compute_esmc_dim(self, protein_id: str) -> Optional[int]:
        try:
            emb = self.extract_protein_features(protein_id)
            return int(len(emb)) if emb is not None else None
        except Exception:
            return None

    def generate_umap_visualizations(
        self,
        protein_name: str,
        uniprot_id: str,
        model_type: str,
        threshold: Optional[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Optional[str]:
        if not self._ensure_umap():
            return None
        try:
            from umap import UMAP
        except Exception:
            return None

        plots_dir = Path(self.output_dir) / "umap_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        th_str = threshold if threshold else "NA"
        fig_path = plots_dir / f"{protein_name}_{uniprot_id}_{model_type}_{th_str}_umap.png"

        train_df = train_df.copy()
        test_df = test_df.copy()
        if 'features' not in train_df.columns or 'features' not in test_df.columns:
            self.logger.warning("Features not found in train/test DataFrames; skipping UMAP visualization.")
            return None

        X_train_full = np.stack(train_df['features'].to_numpy())
        X_test_full = np.stack(test_df['features'].to_numpy())
        X_full = np.vstack([X_train_full, X_test_full])

        # Build chemical-only matrix directly from SMILES (Morgan + physchem), independent of model features
        def build_chemical_matrix(train_part: pd.DataFrame, test_part: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
            """Returns (chemical_features_matrix, n_train_valid, n_test_valid)"""
            rows = []
            n_train_valid = 0
            n_test_valid = 0
            
            # Process train data
            for _, r in train_part.iterrows():
                smi = r.get('SMILES', '')
                if not isinstance(smi, str) or smi == '':
                    continue
                if smi not in self._cached_fingerprints:
                    continue
                morgan = self._cached_fingerprints[smi]
                try:
                    phys = self.bioactivity_loader.calculate_physicochemical_descriptors(smi)
                    if not phys:
                        continue
                    phys_arr = np.array(list(phys.values()), dtype=np.float32)
                    vec = np.concatenate([morgan, phys_arr]).astype(np.float32)
                    if np.isnan(vec).any():
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    rows.append(vec)
                    n_train_valid += 1
                except Exception:
                    continue
            
            # Process test data
            for _, r in test_part.iterrows():
                smi = r.get('SMILES', '')
                if not isinstance(smi, str) or smi == '':
                    continue
                if smi not in self._cached_fingerprints:
                    continue
                morgan = self._cached_fingerprints[smi]
                try:
                    phys = self.bioactivity_loader.calculate_physicochemical_descriptors(smi)
                    if not phys:
                        continue
                    phys_arr = np.array(list(phys.values()), dtype=np.float32)
                    vec = np.concatenate([morgan, phys_arr]).astype(np.float32)
                    if np.isnan(vec).any():
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    rows.append(vec)
                    n_test_valid += 1
                except Exception:
                    continue
            
            if len(rows) == 0:
                return np.empty((0, 0), dtype=np.float32), 0, 0
            X = np.vstack(rows).astype(np.float32)
            return X, n_train_valid, n_test_valid

        X_chem_all, n_train_valid, n_test_valid = build_chemical_matrix(train_df, test_df)

        # Try to infer ESM size for protein panels
        esmc_dim = self._compute_esmc_dim(uniprot_id)

        # Panel 1: Chemical space
        try:
            if X_chem_all.shape[0] < 10 or n_train_valid == 0 or n_test_valid == 0:
                chem_train = None
                chem_test = None
            else:
                umap_chem = UMAP(n_components=2, random_state=42, n_neighbors=min(15, X_chem_all.shape[0]-1), min_dist=0.1, metric='cosine')
                emb_chem = umap_chem.fit_transform(X_chem_all)
                # Split based on the known counts: first n_train_valid are train, rest are test
                chem_train = emb_chem[:n_train_valid] if n_train_valid > 0 else None
                chem_test = emb_chem[n_train_valid:] if n_test_valid > 0 else None
        except Exception as e:
            self.logger.warning(f"Chemical UMAP failed: {e}")
            chem_train = None
            chem_test = None

        # Panels 2-4: Protein space (Model B with ESM)
        prot_train = None
        prot_test = None
        prot_emb = None
        prot_colors = None
        class_colors = None
        try:
            if esmc_dim is not None and X_full.shape[1] >= esmc_dim and model_type == 'B':
                X_train_prot = X_train_full[:, -esmc_dim:]
                X_test_prot = X_test_full[:, -esmc_dim:]
                X_prot = np.vstack([X_train_prot, X_test_prot])
                
                # Handle NaN values: remove rows with any NaN
                nan_mask_prot = ~np.isnan(X_prot).any(axis=1)
                if nan_mask_prot.sum() < 6:  # Need at least 6 samples for UMAP
                    raise ValueError(f"Too few valid samples after NaN removal: {nan_mask_prot.sum()}")
                
                X_prot_clean = X_prot[nan_mask_prot]
                n_train_prot_keep = nan_mask_prot[:len(X_train_prot)].sum()
                n_test_prot_keep = nan_mask_prot[len(X_train_prot):].sum()
                
                # Replace any remaining infinite values with 0
                X_prot_clean = np.nan_to_num(X_prot_clean, nan=0.0, posinf=0.0, neginf=0.0)
                
                umap_prot = UMAP(n_components=2, random_state=42, n_neighbors=min(10, len(X_prot_clean)-1), min_dist=0.1, metric='cosine')
                emb_prot = umap_prot.fit_transform(X_prot_clean)
                prot_train = emb_prot[:n_train_prot_keep]
                prot_test = emb_prot[n_train_prot_keep:]
                prot_emb = emb_prot

                # Filter accessions and classes to match cleaned data
                all_accessions = pd.concat([train_df['accession'], test_df['accession']]).astype(str).tolist()
                all_accessions_clean = [all_accessions[i] for i in range(len(all_accessions)) if nan_mask_prot[i]]
                uniq_prots = sorted(list(set(all_accessions_clean)))
                cmap = plt.cm.get_cmap('Set3', len(uniq_prots))
                prot_to_color = {p: cmap(i) for i, p in enumerate(uniq_prots)}
                prot_colors = [prot_to_color[p] for p in all_accessions_clean]

                all_classes = pd.concat([train_df['class'], test_df['class']]).astype(str).tolist()
                all_classes_clean = [all_classes[i] for i in range(len(all_classes)) if nan_mask_prot[i]]
                class_palette = {
                    'Low': '#2ca02c',
                    'Medium': '#ff7f0e',
                    'High': '#9467bd'
                }
                class_colors = [class_palette.get(c, '#7f7f7f') for c in all_classes_clean]
        except Exception as e:
            self.logger.warning(f"Protein UMAP failed: {e}")
            prot_train = None
            prot_test = None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.ravel()

        # Panel 1
        ax1.set_title('Chemical space (train/test)')
        if chem_train is not None and chem_test is not None:
            ax1.scatter(chem_train[:,0], chem_train[:,1], s=10, c='#1f77b4', label='Train', alpha=0.7)
            ax1.scatter(chem_test[:,0], chem_test[:,1], s=10, c='#ff7f0e', label='Test', alpha=0.7)
            ax1.legend(loc='best', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')

        # Panel 2
        ax2.set_title('Protein space (by protein, train/test edges)')
        if prot_train is not None and prot_test is not None and prot_colors is not None:
            combined = np.vstack([prot_train, prot_test])
            total_len = len(combined)
            for i in range(total_len):
                edge = '#1f77b4' if i < len(prot_train) else '#ff7f0e'
                ax2.scatter(combined[i,0], combined[i,1], s=12, c=[prot_colors[i]], edgecolors=edge, linewidths=0.5, alpha=0.8)
        else:
            ax2.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')

        # Panel 3
        ax3.set_title('Protein space (by protein)')
        if prot_emb is not None and prot_colors is not None:
            ax3.scatter(prot_emb[:,0], prot_emb[:,1], s=12, c=prot_colors, alpha=0.8)
        else:
            ax3.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')

        # Panel 4
        ax4.set_title('Protein space (by activity class)')
        if prot_emb is not None and class_colors is not None:
            ax4.scatter(prot_emb[:,0], prot_emb[:,1], s=12, c=class_colors, alpha=0.8)
        else:
            ax4.text(0.5, 0.5, 'Not available', ha='center', va='center')
        ax4.set_xlabel('UMAP 1')
        ax4.set_ylabel('UMAP 2')

        fig.suptitle(f"{protein_name} {uniprot_id} • Model {model_type} • {th_str}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        self.logger.info(f"Saved UMAP visualization to {fig_path}")
        return str(fig_path)
    def process_all_proteins(self):
        """Process all proteins in the avoidome list"""
        self.logger.info(f"Starting AQSE 3-Class workflow for {len(self.targets)} proteins")
        
        for i, protein in enumerate(self.targets):
            uniprot_id = protein['UniProt ID']
            protein_name = protein.get('Name_2', uniprot_id)
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing {i+1}/{len(self.targets)}: {protein_name} ({uniprot_id})")
            self.logger.info(f"{'='*80}")
            
            try:
                result = self.process_single_protein(uniprot_id, protein_name)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {protein_name}: {e}")
                self.results.append({
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'error': str(e)
                })
        
        self._save_results()
    
    def process_single_protein(self, uniprot_id: str, protein_name: str) -> Dict[str, Any]:
        """
        Process a single protein and train appropriate model(s)
        
        Returns:
            Dictionary with results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {protein_name} ({uniprot_id})")
        self.logger.info(f"{'='*60}")
        
        # Check if protein has similar proteins
        similar_proteins = self.similarity_loader.get_similar_proteins(uniprot_id)
        has_similar = len(similar_proteins) > 0
        
        # Get activity thresholds (includes n_classes and class_labels)
        thresholds = self.thresholds_loader.get_thresholds(uniprot_id)
        if not thresholds:
            self.logger.warning(f"No activity thresholds found for {uniprot_id}")
            return {
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'model_type': 'unknown',
                'status': 'no_thresholds'
            }
        
        # Log classification type
        n_classes = thresholds.get('n_classes', 3)
        class_labels = thresholds.get('class_labels', ['Low', 'Medium', 'High'])
        self.logger.info(f"Using {n_classes}-class classification: {class_labels}")
        
        if not has_similar:
            # Model A only: Simple QSAR for proteins without similar proteins
            trainer_name = self.model_trainer.get_model_type_name()
            self.logger.info(f"No similar proteins found → Using Model A (Simple QSAR with {trainer_name})")
            return self.train_model_a(uniprot_id, protein_name, thresholds)
        else:
            # Both Model A and Model B: QSAR + PCM for proteins with similar proteins
            trainer_name = self.model_trainer.get_model_type_name()
            self.logger.info(f"Found {len(similar_proteins)} similar proteins → Using Model A (QSAR with {trainer_name}) + Model B (PCM with {trainer_name})")
            
            # Train Model A first
            self.logger.info(f"\n--- Training Model A (QSAR) for {protein_name} ---")
            model_a_result = self.train_model_a(uniprot_id, protein_name, thresholds)
            
            # Train Model B
            self.logger.info(f"\n--- Training Model B (PCM) for {protein_name} ---")
            model_b_result = self.train_model_b(uniprot_id, protein_name, thresholds)
            
            # Return combined results
            return {
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'has_similar_proteins': True,
                'n_similar_proteins': len(similar_proteins),
                'model_a': model_a_result,
                'model_b': model_b_result
            }
    
    def train_model_a(self, uniprot_id: str, protein_name: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Train Model A: Simple QSAR model for proteins without similar proteins"""
        self.logger.info("Training Model A: Simple QSAR")
        
        # Load bioactivity data
        bioactivity_data = self.bioactivity_loader.get_filtered_papyrus([uniprot_id])
        
        if len(bioactivity_data) < 30:
            self.logger.warning(f"Insufficient data: {len(bioactivity_data)} samples (need ≥30)")
            return {
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'model_type': 'A',
                'status': 'insufficient_data',
                'n_samples': len(bioactivity_data)
            }
        
        # Filter for valid data
        bioactivity_data = bioactivity_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        bioactivity_data = bioactivity_data[bioactivity_data['SMILES'] != '']
        
        # Assign class labels
        bioactivity_data = self.assign_class_labels(bioactivity_data, thresholds)
        
        # Extract features (conditional based on trainer requirements)
        if self.model_trainer.requires_precomputed_features():
            # Model A: Extract features (no ESM-C)
            self.logger.info("Extracting features (Morgan + Physicochemical)...")
            features_df = self.create_feature_dataset(bioactivity_data, uniprot_id, add_esmc=False)
            
            if len(features_df) < 30:
                self.logger.warning(f"Insufficient data after feature extraction: {len(features_df)} samples")
                return {
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'model_type': 'A',
                    'status': 'insufficient_data',
                    'n_samples': len(features_df)
                }
            
            # Split data: 80% train, 20% test (and optionally validation)
            # Check if parameter optimization is enabled and if doc_id column exists
            use_optimization = self.config.get('enable_parameter_optimization', False)
            use_fixed_val = use_optimization and 'doc_id' in features_df.columns
            
            if use_fixed_val:
                train_df, val_df, test_df = self.split_data_stratified(
                    features_df, 
                    test_size=0.2,
                    use_fixed_test=True,
                    doc_id_column='doc_id'
                )
            else:
                train_df, test_df, _ = self.split_data_stratified(
                    features_df, 
                    test_size=0.2,
                    use_fixed_test=False
                )
                val_df = None
            total_samples = len(features_df)
        else:
            # For trainers that don't need precomputed features (e.g., Chemprop)
            # Use test script approach: prepare clean DataFrames with SMILES, class columns
            # Chemprop will extract features on-the-fly during training
            self.logger.info("Preparing data for Chemprop (features extracted on-the-fly)...")
            
            # Ensure we have the required columns for Chemprop
            # ChempropTrainer expects: 'SMILES', 'class', and optionally 'accession' for Model B
            # Also preserve 'doc_id' if present (needed for fixed validation set)
            columns_to_keep = ['SMILES', 'class']
            if 'doc_id' in bioactivity_data.columns:
                columns_to_keep.append('doc_id')
            if 'accession' in bioactivity_data.columns:
                columns_to_keep.append('accession')
            bioactivity_data_clean = bioactivity_data[columns_to_keep].copy()
            
            # Split data: 80% train, 20% test with stratification (and optionally validation)
            # Check if parameter optimization is enabled and if doc_id column exists
            use_optimization = self.config.get('enable_parameter_optimization', False)
            use_fixed_val = use_optimization and 'doc_id' in bioactivity_data_clean.columns
            
            if use_fixed_val:
                train_df, val_df, test_df = self.split_data_stratified(
                    bioactivity_data_clean, 
                    test_size=0.2,
                    use_fixed_test=True,
                    doc_id_column='doc_id'
                )
            else:
                train_df, test_df, _ = self.split_data_stratified(
                    bioactivity_data_clean, 
                    test_size=0.2,
                    use_fixed_test=False
                )
                val_df = None
            total_samples = len(bioactivity_data_clean)
        
        self.logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        if val_df is not None:
            self.logger.info(f"Validation: {len(val_df)}")
        self.logger.info(f"Class distribution - Train: {train_df['class'].value_counts().to_dict()}")
        self.logger.info(f"Class distribution - Test: {test_df['class'].value_counts().to_dict()}")
        if val_df is not None:
            self.logger.info(f"Class distribution - Val: {val_df['class'].value_counts().to_dict()}")
        
        # Check if parameter optimization is enabled (only for Chemprop)
        use_optimization = self.config.get('enable_parameter_optimization', False)
        model_type_config = self.config.get('model_type', 'random_forest').lower()
        
        if use_optimization and model_type_config == 'chemprop' and val_df is not None:
            self.logger.info("Running hyperparameter optimization...")
            optimizer = ChempropHyperparameterOptimizer(
                n_trials=self.config.get('optimization_n_trials', 50),
                search_strategy=self.config.get('optimization_strategy', 'hybrid'),
                random_state=self.config.get('random_state', 42),
                bioactivity_loader=self.bioactivity_loader
            )
            
            opt_result = optimizer.optimize(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_type='A',
                protein_name=protein_name,
                uniprot_id=uniprot_id,
                thresholds=thresholds,
                max_epochs=self.config.get('chemprop_max_epochs', 100),  # Full epochs for final model
                optimization_epochs=self.config.get('optimization_epochs', 50),  # Shorter epochs for trials
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
                    'model_path': 'N/A',  # Model saved during optimization
                    'metrics_path': 'N/A',
                    'best_params': opt_result['best_params'],
                    'best_val_score': opt_result['best_val_score'],
                    'optimization_trials': opt_result['n_trials'],
                    'optimization_successful': opt_result['n_successful']
                }
            else:
                self.logger.warning("Parameter optimization failed, falling back to default training")
                train_results = self._train_model_with_reporting(
                    train_df, test_df, model_type='A', protein_name=protein_name, 
                    uniprot_id=uniprot_id, thresholds=thresholds
                )
        else:
            # Standard training without optimization
            train_results = self._train_model_with_reporting(
                train_df, test_df, model_type='A', protein_name=protein_name, 
                uniprot_id=uniprot_id, thresholds=thresholds
            )
        
        return {
            'protein': protein_name,
            'uniprot_id': uniprot_id,
            'model_type': 'A',
            'status': 'complete',
            'n_samples': total_samples,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'similar_proteins': 'N/A',  # Model A doesn't use similar proteins
            'n_features': train_results.get('n_features', 'N/A'),
            **train_results
        }
    
    def train_model_b(self, uniprot_id: str, protein_name: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Train Model B: PCM model with similar proteins at 3 thresholds
        
        Model B uses the configured trainer (Random Forest or Chemprop) and always includes
        ESM-C protein embeddings to distinguish between different proteins in the dataset.
        """
        trainer_name = self.model_trainer.get_model_type_name()
        self.logger.info(f"Training Model B: PCM with similar proteins using {trainer_name}")
        
        results_by_threshold = {}
        
        for threshold in ['high', 'medium', 'low']:
            self.logger.info(f"\n--- Processing {threshold.upper()} threshold ---")
            
            # Get similar proteins for this threshold
            similar_proteins = self.similarity_loader.get_similar_proteins_for_threshold(uniprot_id, threshold)
            
            if not similar_proteins:
                self.logger.warning(f"No similar proteins at {threshold} threshold")
                results_by_threshold[threshold] = {
                    'status': 'no_similar_proteins',
                    'n_similar': 0
                }
                continue
            
            # Load bioactivity data
            similar_data = self.bioactivity_loader.get_filtered_papyrus(similar_proteins)
            target_data = self.bioactivity_loader.get_filtered_papyrus([uniprot_id])
            
            total_samples = len(similar_data) + len(target_data)
            if total_samples < 30:
                self.logger.warning(f"Insufficient data at {threshold}: {total_samples} samples")
                results_by_threshold[threshold] = {
                    'status': 'insufficient_data',
                    'n_samples': total_samples,
                    'n_similar_proteins': len(similar_proteins)
                }
                continue
            
            self.logger.info(f"Similar proteins data: {len(similar_data):,} activities")
            self.logger.info(f"Target protein data: {len(target_data):,} activities")
            
            # Filter for valid data
            similar_data = similar_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
            target_data = target_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
            similar_data = similar_data[similar_data['SMILES'] != '']
            target_data = target_data[target_data['SMILES'] != '']
            
            # Assign class labels to both datasets
            similar_data = self.assign_class_labels(similar_data, thresholds)
            target_data = self.assign_class_labels(target_data, thresholds)
            
            # Split target data: 80% train, 20% test
            target_train, target_test = self.split_data_stratified(target_data, test_size=0.2)
            
            # Combine training data: 100% similar + 80% target
            train_data = pd.concat([similar_data, target_train], ignore_index=True)
            
            # Extract features (conditional based on trainer requirements)
            if self.model_trainer.requires_precomputed_features():
                # Model B with Random Forest: Extract features with ESM-C
                # ESM-C embeddings are required to distinguish between different proteins
                self.logger.info(f"Extracting features for {trainer_name} (Morgan + Physicochemical + ESM-C)...")
                
                # For similar proteins - need to handle multiple proteins
                train_features = self._extract_multi_protein_features(
                    train_data, 
                    train_data['accession'].unique(),
                    add_esmc=True
                )
                
                test_features = self._extract_multi_protein_features(
                    target_test,
                    [uniprot_id],
                    add_esmc=True
                )
            else:
                # Model B with Chemprop: Prepare data for on-the-fly feature extraction
                # Chemprop will extract features (physicochemical + ESM) during training
                self.logger.info(f"Preparing data for {trainer_name} (features extracted on-the-fly with ESM-C)...")
                
                # Ensure we have required columns: 'SMILES', 'class', 'accession'
                # Note: accession is needed for Model B to load ESM embeddings
                train_features = train_data[['SMILES', 'class', 'accession']].copy()
                test_features = target_test[['SMILES', 'class', 'accession']].copy()
                
                # Ensure test set has correct accession (target protein)
                test_features['accession'] = uniprot_id
            
            if len(train_features) < 30 or len(test_features) < 6:
                self.logger.warning(f"Insufficient data after feature extraction")
                results_by_threshold[threshold] = {
                    'status': 'insufficient_features',
                    'n_train': len(train_features),
                    'n_test': len(test_features)
                }
                continue
            
            self.logger.info(f"Train: {len(train_features)}, Test: {len(test_features)}")
            self.logger.info(f"Class distribution - Train: {train_features['class'].value_counts().to_dict()}")
            self.logger.info(f"Class distribution - Test: {test_features['class'].value_counts().to_dict()}")
            
            # Train model using modular trainer
            train_results = self._train_model_with_reporting(
                train_features, test_features, 
                model_type='B', 
                protein_name=protein_name,
                threshold=threshold,
                uniprot_id=uniprot_id,
                thresholds=thresholds
            )
            
            results_by_threshold[threshold] = {
                'status': 'complete',
                'n_train': len(train_features),
                'n_test': len(test_features),
                'n_similar_proteins': len(similar_proteins),
                'n_similar_activities': len(similar_data),
                'n_target_activities': len(target_data),
                'similar_proteins': similar_proteins,
                'n_features': train_results.get('n_features', 'N/A'),
                **train_results
            }
        
        return {
            'protein': protein_name,
            'uniprot_id': uniprot_id,
            'model_type': 'B',
            'thresholds': results_by_threshold
        }
    
    def _save_results(self):
        """Save workflow results to file and generate summary report"""
        results_file = self.output_dir / "workflow_results.csv"
        
        # Flatten results for CSV
        csv_data = []
        for result in self.results:
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
        self.logger.info(f"Saved results to {results_file}")
        
        # Generate comprehensive summary report
        summary_path = self.model_reporter.generate_summary_report(df)
        self.logger.info(f"Generated summary report: {summary_path}")
        
        # Generate comprehensive protein report
        comprehensive_path = self.model_reporter.generate_comprehensive_report(df)
        self.logger.info(f"Generated comprehensive report: {comprehensive_path}")
        
        # Generate individual model reports
        individual_reports = self.model_reporter.generate_individual_model_reports(df)
        self.logger.info(f"Generated {len(individual_reports)} individual model reports")
    
    def extract_compound_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        Extract features for a single compound
        
        Args:
            smiles: SMILES string
            
        Returns:
            Combined feature vector or None if extraction fails
        """
        features_list = []
        
        # 1. Morgan fingerprints (from cached dict)
        if smiles in self._cached_fingerprints:
            morgan_fp = self._cached_fingerprints[smiles]
            features_list.append(morgan_fp)
        else:
            self.logger.warning(f"Morgan fingerprint not found for {smiles}")
            return None
        
        # 2. Physicochemical descriptors
        try:
            physico_desc = self.bioactivity_loader.calculate_physicochemical_descriptors(smiles)
            if physico_desc:
                # Convert dict to numpy array
                physico_array = np.array(list(physico_desc.values()), dtype=np.float32)
                features_list.append(physico_array)
            else:
                self.logger.warning(f"Physicochemical descriptors failed for {smiles}")
                return None
        except Exception as e:
            self.logger.error(f"Error calculating physicochemical descriptors: {e}")
            return None
        
        # Combine all features
        if features_list:
            combined = np.concatenate(features_list)
            return combined
        return None
    
    def extract_protein_features(self, uniprot_id: str) -> Optional[np.ndarray]:
        """
        Extract ESM-C descriptors for a protein
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            ESM-C embedding array or None
        """
        esmc_desc = self.bioactivity_loader.load_esmc_descriptors(uniprot_id)
        return esmc_desc
    
    def assign_class_labels(self, df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
        """
        Assign class labels to bioactivity data based on pchembl_value_Mean
        Supports both 2-class and 3-class classification
        
        Args:
            df: DataFrame with pchembl_value_Mean column
            thresholds: Dictionary with 'high' and 'medium' cutoff values, 'n_classes', and 'class_labels'
            
        Returns:
            DataFrame with 'class' column added
        """
        if 'pchembl_value_Mean' not in df.columns:
            self.logger.error("pchembl_value_Mean column not found")
            return df
        
        n_classes = thresholds.get('n_classes', 3)
        cutoff_high = thresholds.get('high', 6.0)
        
        if n_classes == 2:
            # 2-class: Low+Medium vs High (combine Low and Medium)
            def assign_class(value):
                if pd.isna(value):
                    return 'unknown'
                if value <= cutoff_high:
                    return 'Low+Medium'
                else:
                    return 'High'
        else:
            # 3-class: Low, Medium, High
            cutoff_medium = thresholds.get('medium', 5.0)
            def assign_class(value):
                if pd.isna(value):
                    return 'unknown'
                if value <= cutoff_medium:
                    return 'Low'
                elif cutoff_medium < value <= cutoff_high:
                    return 'Medium'
                else:
                    return 'High'
        
        df['class'] = df['pchembl_value_Mean'].apply(assign_class)
        return df
    
    def create_feature_dataset(self, df: pd.DataFrame, uniprot_id: str, add_esmc: bool = True) -> pd.DataFrame:
        """
        Create dataset with extracted features
        
        Args:
            df: Bioactivity DataFrame with SMILES
            uniprot_id: Protein ID for ESM-C descriptors
            add_esmc: Whether to add ESM-C features (for Model B)
            
        Returns:
            DataFrame with features and labels
        """
        features_data = []
        
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            
            # Extract compound features
            compound_features = self.extract_compound_features(smiles)
            if compound_features is None:
                continue
            
            # Start with compound features
            feature_vector = compound_features.tolist()
            
            # Add ESM-C features if requested (Model B)
            if add_esmc:
                esmc_features = self.extract_protein_features(uniprot_id)
                if esmc_features is not None:
                    feature_vector.extend(esmc_features.tolist())
                else:
                    # Skip compounds if ESM-C not available
                    continue
            
            # Store the data
            features_data.append({
                'SMILES': smiles,
                'accession': row.get('accession', uniprot_id),
                'pchembl_value_Mean': row.get('pchembl_value_Mean'),
                'class': row.get('class'),
                'features': np.array(feature_vector, dtype=np.float32)
            })
        
        return pd.DataFrame(features_data)
    
    def split_data_stratified(self, df: pd.DataFrame, test_size: float = 0.2, 
                              use_fixed_test: bool = False,
                              doc_id_column: str = 'doc_id') -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: DataFrame with 'class' column and optionally doc_id_column
            test_size: Fraction for validation set (when use_fixed_test=True, this is ignored for test set)
            use_fixed_test: If True, create fixed test set based on doc_id, then split remaining into train/val
            doc_id_column: Name of column containing document IDs
            
        Returns:
            If use_fixed_test: (train_df, val_df, test_df)
            Otherwise: (train_df, test_df, None)
        """
        from sklearn.model_selection import train_test_split
        
        if use_fixed_test and doc_id_column in df.columns:
            # First, create fixed test set based on doc_id
            remaining_df, test_df = create_fixed_test_set(
                df, 
                doc_id_column=doc_id_column,
                min_molecules_per_doc=20,
                molecules_per_doc=2,
                random_state=42
            )
            
            # Then split remaining data into train and validation (80/20 stratified)
            train_df, val_df = train_test_split(
                remaining_df,
                test_size=test_size,
                stratify=remaining_df['class'] if 'class' in remaining_df.columns else None,
                random_state=42
            )
            
            self.logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, val_df, test_df
        else:
            # Standard stratified split (train/test only)
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df['class'] if 'class' in df.columns else None,
                random_state=42
            )
            
            return train_df, test_df, None
    
    def _train_model_with_reporting(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        model_type: str = 'A',
        protein_name: Optional[str] = None,
        threshold: Optional[str] = None,
        uniprot_id: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train model using modular trainer with comprehensive reporting and MLflow tracking
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            model_type: 'A' or 'B'
            protein_name: Optional protein name for logging
            threshold: Optional threshold for Model B
            uniprot_id: UniProt ID
            thresholds: Dictionary with thresholds, n_classes, and class_labels
            
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
        if self.mlflow_enabled:
            try:
                mlflow.set_experiment(self.mlflow_experiment_name)
                mlflow.start_run(run_name=run_name)
                mlflow_run_active = True
            except Exception as e:
                self.logger.warning(f"Failed to start MLflow run: {e}. Continuing without MLflow.")
                mlflow_run_active = False
        
        try:
            # Train model using the configured trainer
            train_results = self.model_trainer.train(
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
            metrics = self.model_reporter.calculate_metrics(y_true, y_pred, y_prob, class_labels=class_labels)
            
            # Save model
            model_path = self.model_reporter.save_model(
                model, protein_name, uniprot_id, model_type, threshold
            )
            
            # Save metrics
            metrics_path = self.model_reporter.save_metrics(
                metrics, protein_name, uniprot_id, model_type, threshold, 'test'
            )
            
            # Save class distributions
            train_dist_path = self.model_reporter.save_class_distribution(
                train_df, protein_name, uniprot_id, model_type, threshold, 'train'
            )
            test_dist_path = self.model_reporter.save_class_distribution(
                test_df, protein_name, uniprot_id, model_type, threshold, 'test'
            )
            
            # Save predictions (pass class_labels from train_results if available)
            class_labels = train_results.get('class_labels', ['Low', 'Medium', 'High'])
            predictions_path = self.model_reporter.save_predictions(
                test_df, y_pred, protein_name, uniprot_id, model_type, y_prob, threshold, 'test', class_labels=class_labels
            )
            
            # Log to MLflow if enabled (after saving artifacts so we can log them)
            if mlflow_run_active:
                try:
                    self._log_to_mlflow(
                        model=model,
                        metrics=metrics,
                        train_df=train_df,
                        test_df=test_df,
                        protein_name=protein_name,
                        uniprot_id=uniprot_id,
                        model_type=model_type,
                        threshold=threshold,
                        train_results=train_results,
                        metrics_path=metrics_path,
                        predictions_path=predictions_path,
                        train_dist_path=train_dist_path,
                        test_dist_path=test_dist_path
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log to MLflow: {e}. Continuing with model saving.")
            
            self.logger.info(f"✓ {self.model_trainer.get_model_type_name()} training completed successfully!")
            self.logger.info(f"  Model saved: {model_path}")
            self.logger.info(f"  Metrics saved: {metrics_path}")
            self.logger.info(f"  Predictions saved: {predictions_path}")
            self.logger.info(f"  Test accuracy: {metrics['accuracy']:.3f}")
            self.logger.info(f"  Test F1-macro: {metrics['f1_macro']:.3f}")
            # Generate UMAP visualizations (best-effort)
            try:
                self.generate_umap_visualizations(
                    protein_name=protein_name or (uniprot_id or "unknown"),
                    uniprot_id=uniprot_id or (protein_name or "unknown"),
                    model_type=model_type,
                    threshold=threshold,
                    train_df=train_df,
                    test_df=test_df
                )
            except Exception as viz_e:
                self.logger.warning(f"UMAP visualization failed: {viz_e}")
            
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
                'model_type_name': train_results.get('model_type_name', self.model_trainer.get_model_type_name()),
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
            self.logger.error(f"Error training {self.model_trainer.get_model_type_name()} model: {e}")
            # End MLflow run with failure status
            if mlflow_run_active:
                mlflow.end_run(status="FAILED")
            return {
                'status': 'error',
                'message': str(e),
                'model_type_name': self.model_trainer.get_model_type_name()
            }
    
    def _log_to_mlflow(
        self,
        model: Any,
        metrics: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        protein_name: Optional[str],
        uniprot_id: str,
        model_type: str,
        threshold: Optional[str],
        train_results: Dict[str, Any],
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
            metrics_path: Path to saved metrics JSON file
            predictions_path: Path to saved predictions CSV file
            train_dist_path: Path to saved train distribution CSV file
            test_dist_path: Path to saved test distribution CSV file
        """
        # Safety check: ensure MLflow is available
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow is not available. Skipping MLflow logging.")
            return
        
        # Log tags
        tags = {
            'protein_name': protein_name or 'unknown',
            'uniprot_id': uniprot_id,
            'model_type': model_type,
            'model_type_name': train_results.get('model_type_name', self.model_trainer.get_model_type_name())
        }
        if threshold:
            tags['threshold'] = threshold
        mlflow.set_tags(tags)
        
        # Log parameters
        params = {}
        
        # Model-specific parameters
        if isinstance(self.model_trainer, RandomForestTrainer):
            params['n_estimators'] = self.model_trainer.n_estimators
            params['max_depth'] = self.model_trainer.max_depth if self.model_trainer.max_depth else 'None'
            params['random_state'] = self.model_trainer.random_state
            params['class_weight'] = self.model_trainer.class_weight
        elif isinstance(self.model_trainer, ChempropTrainer):
            params['max_epochs'] = self.model_trainer.max_epochs
            params['batch_size'] = self.model_trainer.batch_size
            params['n_classes'] = self.model_trainer.n_classes
            params['include_esm'] = self.model_trainer.include_esm
            params['val_fraction'] = self.model_trainer.val_fraction
            params['max_lr'] = self.model_trainer.max_lr
            params['init_lr'] = self.model_trainer.init_lr
            params['final_lr'] = self.model_trainer.final_lr
            params['warmup_epochs'] = self.model_trainer.warmup_epochs
            params['ffn_num_layers'] = self.model_trainer.ffn_num_layers
            params['hidden_size'] = self.model_trainer.hidden_size
            params['dropout'] = self.model_trainer.dropout
            params['activation'] = self.model_trainer.activation
            params['aggregation'] = self.model_trainer.aggregation
            params['depth'] = self.model_trainer.depth
            params['bias'] = self.model_trainer.bias
        
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
        model_type_name = train_results.get('model_type_name', self.model_trainer.get_model_type_name())
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
                self.logger.warning(f"Failed to log Chemprop model to MLflow: {e}. Model may not be PyTorch-based.")
        
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
            self.logger.warning(f"Failed to log some artifacts to MLflow: {e}")
    
    def _extract_multi_protein_features(self, df: pd.DataFrame, protein_ids: List[str], add_esmc: bool = True) -> pd.DataFrame:
        """
        Extract features for data from multiple proteins
        
        Args:
            df: Bioactivity DataFrame with 'accession' column
            protein_ids: List of protein IDs in the dataset
            add_esmc: Whether to add ESM-C features
            
        Returns:
            DataFrame with features
        """
        features_data = []
        
        # Cache ESM-C descriptors for all proteins
        esmc_cache = {}
        if add_esmc:
            for pid in protein_ids:
                esmc_desc = self.extract_protein_features(pid)
                if esmc_desc is not None:
                    esmc_cache[pid] = esmc_desc
                else:
                    self.logger.warning(f"ESM-C descriptors not available for {pid}")
        
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            protein_id = row.get('accession', protein_ids[0])
            
            # Extract compound features
            compound_features = self.extract_compound_features(smiles)
            if compound_features is None:
                continue
            
            # Start with compound features
            feature_vector = compound_features.tolist()
            
            # Add ESM-C features if requested
            if add_esmc:
                if protein_id in esmc_cache:
                    esmc_features = esmc_cache[protein_id]
                    feature_vector.extend(esmc_features.tolist())
                else:
                    # Skip if ESM-C not available
                    continue
            
            # Store the data
            features_data.append({
                'SMILES': smiles,
                'accession': protein_id,
                'pchembl_value_Mean': row.get('pchembl_value_Mean'),
                'class': row.get('class'),
                'features': np.array(feature_vector, dtype=np.float32)
            })
        
        return pd.DataFrame(features_data)


# Main

class ModelReporter:
    """Handles comprehensive model reporting including metrics, predictions, and model saving"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.results_dir / "models"
        self.metrics_dir = self.results_dir / "metrics"
        self.predictions_dir = self.results_dir / "predictions"
        self.distributions_dir = self.results_dir / "class_distributions"
        
        for dir_path in [self.models_dir, self.metrics_dir, self.predictions_dir, self.distributions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_model(self, model, protein_name: str, uniprot_id: str, model_type: str, threshold: Optional[str] = None) -> str:
        """Save trained model in compressed format"""
        # Create filename
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_model.pkl.gz"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_model.pkl.gz"
        
        filepath = self.models_dir / filename
        
        try:
            # Handle different model types
            # For PyTorch models, save state_dict; for scikit-learn models, save the full model
            if hasattr(model, 'state_dict'):
                # PyTorch model - save state_dict
                model_data = {
                    'state_dict': model.state_dict(),
                    'model_config': {
                        'protein_name': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': model_type,
                        'threshold': threshold,
                        'timestamp': datetime.now().isoformat(),
                        'model_class': 'pytorch'
                    }
                }
            else:
                # Scikit-learn or other models - save the full model
                model_data = {
                    'model': model,
                    'model_config': {
                        'protein_name': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': model_type,
                        'threshold': threshold,
                        'timestamp': datetime.now().isoformat(),
                        'model_class': 'sklearn' if hasattr(model, 'predict') and hasattr(model, 'fit') else 'other'
                    }
                }
            
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save model {filename}: {e}")
            return ""
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], y_prob: Optional[np.ndarray] = None, class_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Use provided class_labels or default to 3-class
        if class_labels is None:
            class_labels = ['Low', 'Medium', 'High']
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class labels
        metrics['class_labels'] = class_labels
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any], protein_name: str, uniprot_id: str, 
                    model_type: str, threshold: Optional[str] = None, split: str = 'test') -> str:
        """Save metrics to JSON file"""
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_{split}_metrics.json"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{split}_metrics.json"
        
        filepath = self.metrics_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Metrics saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics {filename}: {e}")
            return ""
    
    def save_class_distribution(self, df: pd.DataFrame, protein_name: str, uniprot_id: str, 
                              model_type: str, threshold: Optional[str] = None, split: str = 'all') -> str:
        """Save class distribution analysis"""
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{split}_distribution.csv"
        
        filepath = self.distributions_dir / filename
        
        try:
            # Calculate distribution
            class_counts = df['class'].value_counts()
            class_proportions = df['class'].value_counts(normalize=True)
            
            distribution_df = pd.DataFrame({
                'class': class_counts.index,
                'count': class_counts.values,
                'proportion': class_proportions.values
            })
            
            # Add summary statistics
            summary_stats = {
                'total_samples': int(len(df)),
                'n_classes': int(len(class_counts)),
                'class_balance_ratio': float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0,
                'most_common_class': str(class_counts.index[0]) if len(class_counts) > 0 else '',
                'most_common_count': int(class_counts.iloc[0]) if len(class_counts) > 0 else 0,
                'least_common_class': str(class_counts.index[-1]) if len(class_counts) > 0 else '',
                'least_common_count': int(class_counts.iloc[-1]) if len(class_counts) > 0 else 0
            }
            
            # Save distribution
            distribution_df.to_csv(filepath, index=False)
            
            # Save summary stats
            summary_filepath = filepath.with_suffix('.summary.json')
            with open(summary_filepath, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            self.logger.info(f"Class distribution saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save class distribution {filename}: {e}")
            return ""
    
    def save_predictions(self, df: pd.DataFrame, y_pred: List[int], protein_name: str, uniprot_id: str, model_type: str, 
                        y_prob: Optional[np.ndarray] = None, threshold: Optional[str] = None, split: str = 'test', 
                        class_labels: Optional[List[str]] = None) -> str:
        """Save predictions with probabilities"""
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_{split}_predictions.csv"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{split}_predictions.csv"
        
        filepath = self.predictions_dir / filename
        
        # Use provided class_labels or default to 3-class
        if class_labels is None:
            class_labels = ['Low', 'Medium', 'High']
        
        try:
            # Create predictions DataFrame
            predictions_df = df.copy()
            predictions_df['predicted_class'] = [class_labels[i] for i in y_pred]
            predictions_df['predicted_class_int'] = y_pred
            
            # Add probabilities if available (dynamic based on n_classes)
            if y_prob is not None:
                for i, label in enumerate(class_labels):
                    predictions_df[f'prob_{label.lower().replace("+", "_")}'] = y_prob[:, i]
                predictions_df['max_prob'] = np.max(y_prob, axis=1)
                predictions_df['confidence'] = predictions_df['max_prob']
            
            # Save predictions
            predictions_df.to_csv(filepath, index=False)
            
            self.logger.info(f"Predictions saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions {filename}: {e}")
            return ""
    
    def generate_comprehensive_report(self, workflow_results: pd.DataFrame) -> str:
        """Generate comprehensive CSV report with all protein information per threshold"""
        comprehensive_filepath = self.results_dir / "comprehensive_protein_report.csv"
        
        try:
            report_data = []
            
            # Process each protein result
            for _, result in workflow_results.iterrows():
                protein = result.get('protein', '')
                uniprot_id = result.get('uniprot_id', '')
                model_type = result.get('model_type', '')
                threshold = result.get('threshold', '')
                status = result.get('status', '')
                
                # Create model_key
                model_key = f"{protein}_{uniprot_id}_{model_type}"
                if threshold and threshold != 'N/A':
                    model_key += f"_{threshold}"
                
                # Base information
                base_info = {
                    'model_key': model_key,
                    'protein': protein,
                    'uniprot_id': uniprot_id,
                    'model_type': model_type,
                    'threshold': threshold if threshold else 'N/A',
                    'qsar_model_run': 'Yes' if status == 'success' else 'No',
                    'status': status
                }
                
                # Load detailed metrics if model was successful
                if status == 'success':
                    # Try to load metrics file
                    metrics_file = self._find_metrics_file(protein, uniprot_id, model_type, threshold)
                    if metrics_file and metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            
                            base_info.update({
                                'accuracy': metrics.get('accuracy', 'N/A'),
                                'f1_macro': metrics.get('f1_macro', 'N/A'),
                                'f1_weighted': metrics.get('f1_weighted', 'N/A'),
                                'precision_macro': metrics.get('precision_macro', 'N/A'),
                                'recall_macro': metrics.get('recall_macro', 'N/A'),
                                'confusion_matrix': str(metrics.get('confusion_matrix', 'N/A'))
                            })
                        except Exception as e:
                            self.logger.warning(f"Could not load metrics for {protein}: {e}")
                            base_info.update({
                                'accuracy': 'Error loading',
                                'f1_macro': 'Error loading',
                                'f1_weighted': 'Error loading',
                                'precision_macro': 'Error loading',
                                'recall_macro': 'Error loading',
                                'confusion_matrix': 'Error loading'
                            })
                    else:
                        base_info.update({
                            'accuracy': 'Metrics file not found',
                            'f1_macro': 'Metrics file not found',
                            'f1_weighted': 'Metrics file not found',
                            'precision_macro': 'Metrics file not found',
                            'recall_macro': 'Metrics file not found',
                            'confusion_matrix': 'Metrics file not found'
                        })
                else:
                    base_info.update({
                        'accuracy': 'N/A',
                        'f1_macro': 'N/A',
                        'f1_weighted': 'N/A',
                        'precision_macro': 'N/A',
                        'recall_macro': 'N/A',
                        'confusion_matrix': 'N/A'
                    })
                
                # Add sample counts and class distribution
                # Support both Model A (train_samples/test_samples) and Model B (n_train/n_test)
                n_train_samples = result.get('train_samples') if pd.notna(result.get('train_samples', float('nan'))) else result.get('n_train', 'N/A')
                n_test_samples = result.get('test_samples') if pd.notna(result.get('test_samples', float('nan'))) else result.get('n_test', 'N/A')
                n_target_points = result.get('n_samples') if pd.notna(result.get('n_samples', float('nan'))) else result.get('n_target_activities', 'N/A')

                base_info.update({
                    'n_target_bioactivity_datapoints': n_target_points,
                    'n_expanded_proteins': result.get('n_similar_proteins', 'N/A'),
                    'n_expanded_bioactivity_datapoints': result.get('n_similar_activities', 'N/A'),
                    'n_train_samples': n_train_samples,
                    'n_test_samples': n_test_samples,
                    'train_class_distribution': 'N/A',  # Will be filled below
                    'test_class_distribution': 'N/A',   # Will be filled below
                    'n_features': 'N/A',               # Will be filled below
                    'similar_proteins': 'N/A'           # Will be filled below
                })
                
                # Load class distribution if available
                class_dist_file = self._find_class_distribution_file(protein, uniprot_id, model_type, threshold, 'test')
                if class_dist_file and class_dist_file.exists():
                    try:
                        dist_df = pd.read_csv(class_dist_file)
                        class_dist = {}
                        for _, row in dist_df.iterrows():
                            class_dist[row['class']] = f"{row['count']} ({row['proportion']:.3f})"
                        base_info['test_class_distribution'] = str(class_dist)
                    except Exception as e:
                        self.logger.warning(f"Could not load class distribution for {protein}: {e}")
                        base_info['test_class_distribution'] = 'Error loading'
                else:
                    base_info['test_class_distribution'] = 'N/A'
                
                # Load train class distribution
                train_class_dist_file = self._find_class_distribution_file(protein, uniprot_id, model_type, threshold, 'train')
                if train_class_dist_file and train_class_dist_file.exists():
                    try:
                        dist_df = pd.read_csv(train_class_dist_file)
                        class_dist = {}
                        for _, row in dist_df.iterrows():
                            class_dist[row['class']] = f"{row['count']} ({row['proportion']:.3f})"
                        base_info['train_class_distribution'] = str(class_dist)
                    except Exception as e:
                        self.logger.warning(f"Could not load train class distribution for {protein}: {e}")
                        base_info['train_class_distribution'] = 'Error loading'
                else:
                    base_info['train_class_distribution'] = 'N/A'
                
                # Add n_features and similar_proteins from workflow results
                base_info['n_features'] = result.get('n_features', 'N/A')
                base_info['similar_proteins'] = result.get('similar_proteins', 'N/A')
                
                report_data.append(base_info)
            
            # Create DataFrame and save
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(comprehensive_filepath, index=False)
            
            self.logger.info(f"Comprehensive report saved: {comprehensive_filepath}")
            return str(comprehensive_filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return ""
    
    def _find_metrics_file(self, protein: str, uniprot_id: str, model_type: str, threshold: str) -> Optional[Path]:
        """Find metrics file for a given model"""
        # Try different naming patterns
        patterns = []
        
        if threshold and threshold != 'N/A':
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_{threshold}_test_metrics.json",
                f"{protein}_{protein}_{model_type}_{threshold}_test_metrics.json",
                f"{uniprot_id}_{uniprot_id}_{model_type}_{threshold}_test_metrics.json"
            ]
        else:
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_test_metrics.json",
                f"{protein}_{protein}_{model_type}_test_metrics.json",
                f"{uniprot_id}_{uniprot_id}_{model_type}_test_metrics.json"
            ]
        
        # Try each pattern
        for pattern in patterns:
            filepath = self.metrics_dir / pattern
            if filepath.exists():
                return filepath
        
        # If no file found, return the first pattern (for error reporting)
        return self.metrics_dir / patterns[0]
    
    def _find_class_distribution_file(self, protein: str, uniprot_id: str, model_type: str, threshold: str, split: str) -> Optional[Path]:
        """Find class distribution file for a given model"""
        # Try different naming patterns
        patterns = []
        
        if threshold and threshold != 'N/A':
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv",
                f"{protein}_{protein}_{model_type}_{threshold}_{split}_distribution.csv",
                f"{uniprot_id}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv"
            ]
        else:
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_{split}_distribution.csv",
                f"{protein}_{protein}_{model_type}_{split}_distribution.csv",
                f"{uniprot_id}_{uniprot_id}_{model_type}_{split}_distribution.csv"
            ]
        
        # Try each pattern
        for pattern in patterns:
            filepath = self.distributions_dir / pattern
            if filepath.exists():
                return filepath
        
        # If no file found, return the first pattern (for error reporting)
        return self.distributions_dir / patterns[0]
    
    def generate_individual_model_reports(self, workflow_results: pd.DataFrame) -> List[str]:
        """Generate individual CSV and JSON reports for each successful model"""
        individual_reports = []
        
        successful_models = workflow_results[workflow_results['status'] == 'success']
        
        for _, result in successful_models.iterrows():
            protein = result.get('protein', '')
            uniprot_id = result.get('uniprot_id', '')
            model_type = result.get('model_type', '')
            threshold = result.get('threshold', '')
            
            try:
                # Create individual report directory
                model_dir = self.results_dir / "individual_reports" / f"{protein}_{uniprot_id}_{model_type}"
                if threshold and threshold != 'N/A':
                    model_dir = model_dir.parent / f"{protein}_{uniprot_id}_{model_type}_{threshold}"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Collect all data for this model
                model_data = {
                    'model_info': {
                        'protein': protein,
                        'uniprot_id': uniprot_id,
                        'model_type': model_type,
                        'threshold': threshold if threshold else 'N/A',
                        'generation_timestamp': datetime.now().isoformat()
                    },
                    'workflow_results': result.to_dict(),
                    'metrics': {},
                    'class_distributions': {},
                    'predictions_summary': {}
                }
                
                # Load metrics
                metrics_file = self._find_metrics_file(protein, uniprot_id, model_type, threshold)
                if metrics_file and metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        model_data['metrics'] = json.load(f)
                
                # Load class distributions
                for split in ['train', 'test']:
                    dist_file = self._find_class_distribution_file(protein, uniprot_id, model_type, threshold, split)
                    if dist_file and dist_file.exists():
                        dist_df = pd.read_csv(dist_file)
                        model_data['class_distributions'][split] = dist_df.to_dict('records')
                        
                        # Load summary if available
                        summary_file = dist_file.with_suffix('.summary.json')
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                model_data['class_distributions'][f'{split}_summary'] = json.load(f)
                
                # Load predictions summary
                pred_file = self._find_predictions_file(protein, uniprot_id, model_type, threshold)
                if pred_file and pred_file.exists():
                    pred_df = pd.read_csv(pred_file)
                    model_data['predictions_summary'] = {
                        'total_predictions': len(pred_df),
                        'prediction_distribution': pred_df['predicted_class'].value_counts().to_dict(),
                        'confidence_stats': {
                            'mean': pred_df['confidence'].mean() if 'confidence' in pred_df.columns else 'N/A',
                            'std': pred_df['confidence'].std() if 'confidence' in pred_df.columns else 'N/A',
                            'min': pred_df['confidence'].min() if 'confidence' in pred_df.columns else 'N/A',
                            'max': pred_df['confidence'].max() if 'confidence' in pred_df.columns else 'N/A'
                        }
                    }
                
                # Save JSON report
                json_file = model_dir / "model_report.json"
                with open(json_file, 'w') as f:
                    json.dump(model_data, f, indent=2)
                
                # Save CSV summary
                csv_data = []
                csv_data.append(['Model Information', ''])
                csv_data.append(['Protein', protein])
                csv_data.append(['UniProt ID', uniprot_id])
                csv_data.append(['Model Type', model_type])
                csv_data.append(['Threshold', threshold if threshold else 'N/A'])
                csv_data.append(['Generation Time', datetime.now().isoformat()])
                csv_data.append(['', ''])
                
                csv_data.append(['Performance Metrics', ''])
                if model_data['metrics']:
                    csv_data.append(['Accuracy', model_data['metrics'].get('accuracy', 'N/A')])
                    csv_data.append(['F1-Macro', model_data['metrics'].get('f1_macro', 'N/A')])
                    csv_data.append(['F1-Weighted', model_data['metrics'].get('f1_weighted', 'N/A')])
                    csv_data.append(['Precision-Macro', model_data['metrics'].get('precision_macro', 'N/A')])
                    csv_data.append(['Recall-Macro', model_data['metrics'].get('recall_macro', 'N/A')])
                csv_data.append(['', ''])
                
                csv_data.append(['Data Summary', ''])
                csv_data.append(['Target Bioactivity Points', result.get('n_samples', 'N/A')])
                csv_data.append(['Train Samples', result.get('train_samples', 'N/A')])
                csv_data.append(['Test Samples', result.get('test_samples', 'N/A')])
                csv_data.append(['Expanded Proteins', result.get('n_similar_proteins', 'N/A')])
                csv_data.append(['Expanded Bioactivity Points', result.get('n_similar_activities', 'N/A')])
                
                # Save CSV
                csv_file = model_dir / "model_summary.csv"
                with open(csv_file, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerows(csv_data)
                
                individual_reports.append(str(json_file))
                individual_reports.append(str(csv_file))
                
                self.logger.info(f"Individual report saved: {model_dir}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate individual report for {protein}: {e}")
        
        return individual_reports
    
    def _find_predictions_file(self, protein: str, uniprot_id: str, model_type: str, threshold: str) -> Optional[Path]:
        """Find predictions file for a given model"""
        if threshold and threshold != 'N/A':
            filename = f"{protein}_{uniprot_id}_{model_type}_{threshold}_test_predictions.csv"
        else:
            filename = f"{protein}_{uniprot_id}_{model_type}_test_predictions.csv"
        return self.predictions_dir / filename
    
    def generate_summary_report(self, workflow_results: pd.DataFrame) -> str:
        summary_filepath = self.results_dir / "model_summary_report.json"
        
        try:
            # Calculate overall statistics
            total_models = len(workflow_results)
            successful_models = len(workflow_results[workflow_results['status'] == 'success'])
            failed_models = total_models - successful_models
            
            # Model type breakdown
            model_a_count = len(workflow_results[workflow_results['model_type'] == 'A'])
            model_b_count = len(workflow_results[workflow_results['model_type'] == 'B'])
            
            # Status breakdown
            status_counts = workflow_results['status'].value_counts().to_dict()
            
            # Threshold breakdown for Model B
            model_b_results = workflow_results[workflow_results['model_type'] == 'B']
            threshold_counts = model_b_results['threshold'].value_counts().to_dict() if 'threshold' in model_b_results.columns else {}
            
            summary = {
                'generation_timestamp': datetime.now().isoformat(),
                'total_models_attempted': total_models,
                'successful_models': successful_models,
                'failed_models': failed_models,
                'success_rate': successful_models / total_models if total_models > 0 else 0,
                'model_type_breakdown': {
                    'model_a': model_a_count,
                    'model_b': model_b_count
                },
                'status_breakdown': status_counts,
                'model_b_threshold_breakdown': threshold_counts,
                'results_directory': str(self.results_dir),
                'subdirectories': {
                    'models': str(self.models_dir),
                    'metrics': str(self.metrics_dir),
                    'predictions': str(self.predictions_dir),
                    'class_distributions': str(self.distributions_dir)
                }
            }
            
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Summary report saved: {summary_filepath}")
            return str(summary_filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return ""


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AQSE 3-Class Classification Workflow')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config.yaml file. If not provided, looks for config.yaml in script directory.')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Determine config file path
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    elif os.getenv('CONFIG_FILE'):
        config_path = Path(os.getenv('CONFIG_FILE')).expanduser().resolve()
    else:
        # Default: look for config.yaml in project root
        config_path = project_root / "config.yaml"
    
    # Validate config file exists
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        logging.error("Please provide --config argument or set CONFIG_FILE environment variable")
        logging.error(f"Or place config.yaml in: {project_root}")
        return
    
    logging.info(f"Using config file: {config_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths in config to absolute paths (relative to config file location)
    config_dir = config_path.parent
    path_keys = ['avoidome_file', 'similarity_file', 'sequence_file', 'activity_thresholds_file', 
                 'output_dir', 'papyrus_cache_dir']
    for key in path_keys:
        if key in config and config[key]:
            # If path is relative, make it relative to config file directory
            path_value = Path(config[key])
            if not path_value.is_absolute():
                config[key] = str((config_dir / path_value).resolve())
            else:
                config[key] = str(Path(config[key]).expanduser().resolve())

    # Print the avoidome file path
    avoidome_file = config.get("avoidome_file")
    if not avoidome_file:
        logging.error("No 'avoidome_file' path found in config.yaml")
        return
    logging.info(f"Avoidome input file: {avoidome_file}")

    # Load avoidome data
    loader = AvoidomeDataLoader(config)
    targets = loader.load_avoidome_targets()

    sim_loader = SimilarityDataLoader(config)
    sim_loader.load_similarity_results()
    similarity_data = sim_loader.load_similarity_results()

    logging.info(f"Loaded similarity results for {len(similarity_data)} proteins.")
    
    #print("\n=== Similarity Results Preview ===")
    #for i, (query, thresh_dict) in enumerate(list(similarity_data.items())[:5]):  # first 5 proteins
    #    print(f"\nQuery protein: {query}")
    #    for threshold, similars in thresh_dict.items():
    #        print(f"  Threshold '{threshold}': {similars}")



    seq_loader = ProteinSequenceLoader(config)
    seq_loader.load_sequences()

    # Howto: Fetch a specific sequence
    # seq = seq_loader.get_sequence("P05177")
    # print(f"P05177 sequence (first 50 aa): {seq[:50]}")

    # Generate unique protein list (avoidome + similar proteins)
    logging.info("\n=== Generating Unique Protein List ===")
    protein_list_generator = UniqueProteinListGenerator(config)
    protein_list_df = protein_list_generator.generate_unique_protein_list()
    
    # Save to CSV
    output_file = protein_list_generator.save_protein_list()

    
    
    # Calculate Morgan fingerprints **filtered by unique proteins**
    # Make sure you pass the DataFrame of unique proteins
    fingerprint_loader = BioactivityDataLoader(config) 
    fingerprints_info = fingerprint_loader.calculate_morgan_fingerprints(protein_list_df)
    
    # Print dataset size information
    if fingerprints_info:
        logging.info("\n=== Dataset Statistics ===")
        logging.info(f"Total activities in Papyrus: {fingerprints_info.get('total_activities_in_papyrus', 0):,}")
        logging.info(f"Activities after protein filter: {fingerprints_info.get('activities_after_protein_filter', 0):,}")
        logging.info(f"Unique proteins in filter: {fingerprints_info.get('unique_proteins_in_filter', 0)}")
        logging.info(f"Unique SMILES: {fingerprints_info.get('total_smiles', 0):,}")
        logging.info(f"Valid fingerprints calculated: {fingerprints_info.get('valid_fingerprints', 0):,}")
    
    # Load fingerprints filtered by unique proteins
    fingerprints = fingerprint_loader.load_morgan_fingerprints(protein_list_df)
    
    # Calculate and print size information
    if fingerprints_info and fingerprints:
        import sys
        
        # Calculate fingerprint size in memory
        fingerprint_size = sys.getsizeof(fingerprints)
        for smiles, fp in fingerprints.items():
            fingerprint_size += sys.getsizeof(smiles) + sys.getsizeof(fp) + fp.nbytes
        
        # Convert to appropriate unit
        def format_size(size_bytes):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        
        # Estimate database size (assuming average record size)
        estimated_db_size = fingerprints_info.get('activities_after_protein_filter', 0) * 1024  # ~1KB per record
        
        logging.info("\n=== Size Information ===")
        logging.info(f"Fingerprints: {len(fingerprints):,} compounds, Memory: {format_size(fingerprint_size)}")
        logging.info(f"Fingerprint dimensions: {2048} bits each")
        logging.info(f"Estimated filtered database size: {format_size(estimated_db_size)}")
        logging.info(f"Number of proteins in dataset: {fingerprints_info.get('unique_proteins_in_filter', 0)}")
        logging.info(f"Number of unique SMILES: {fingerprints_info.get('total_smiles', 0):,}")
    
    # Test physicochemical descriptors on a few compounds
    #logging.info("\n=== Testing Physicochemical Descriptors ===")
    #if fingerprints:
    #    test_smiles = list(fingerprints.keys())[:3]  # Test with first 3 compounds
    #    
    #    for smiles in test_smiles:
    #        logging.info(f"\nCalculating descriptors for: {smiles}")
    #        descriptors = fingerprint_loader.calculate_physicochemical_descriptors(smiles)
            
    #        if descriptors:
    #            logging.info(f"  Found {len(descriptors)} descriptors")
    #            # Print first 5 descriptors as example
    #            for i, (key, value) in enumerate(descriptors.items()):
    #                if i < 5:
    #                    logging.info(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    #                else:
    #                    break
    #        else:
    #            logging.info(f"  No descriptors calculated")
    
    # Optional: preview a few fingerprints
    #for i, (smiles, fp) in enumerate(list(fingerprints.items())[:5]):
    #    logging.info(f"SMILES: {smiles}, fingerprint length: {len(fp)}")

    # Initialize Activity Thresholds Loader
    thresholds_loader = ActivityThresholdsLoader(config)

    # Test: Visualize one protein and its data sizes
    logging.info("\n" + "="*80)
    logging.info("=== Protein Data Visualization Test ===")
    logging.info("="*80)
    
    # Pick first protein from avoidome list
    if not targets:
        logging.warning("No targets loaded for test")
    else:
        test_protein = targets[5] # change to test different proteins
        test_uniprot = test_protein['UniProt ID']
        test_name = test_protein.get('Name_2', test_uniprot)
        
        logging.info(f"\nSelected protein: {test_name} ({test_uniprot})")
        
        # Get similar proteins for each threshold
        logging.info(f"\nSimilar proteins by threshold:")
        for threshold in ['high', 'medium', 'low']:
            similar_proteins = sim_loader.get_similar_proteins_for_threshold(test_uniprot, threshold)
            logging.info(f"  {threshold.upper()} : {len(similar_proteins)} proteins")
            if similar_proteins:
                logging.info(f"    {', '.join(similar_proteins)}")
        
        # Also show all merged
        all_similar = sim_loader.get_similar_proteins(test_uniprot)
        logging.info(f"  ALL MERGED: {len(all_similar)} unique proteins")
        
        # Get bioactivity points
        target_data = fingerprint_loader.get_filtered_papyrus([test_uniprot])
        target_activities = len(target_data)
        logging.info(f"\nBioactivity points:")
        logging.info(f"  Target protein: {target_activities:,}")
        
        # Get sample SMILES for later use
        sample_smiles = []
        if not target_data.empty:
            sample_smiles = target_data['SMILES'].drop_duplicates().head(3).tolist()
        
        # Similar proteins per threshold
        for threshold in ['high', 'medium', 'low']:
            similar_proteins = sim_loader.get_similar_proteins_for_threshold(test_uniprot, threshold)
            if similar_proteins:
                similar_data = fingerprint_loader.get_filtered_papyrus(similar_proteins)
                logging.info(f"  Similar ({threshold}): {len(similar_data):,} ({len(similar_proteins)} proteins)")
        
        # Calculate sizes for different descriptor types
        logging.info(f"\nDescriptor Sizes:")
        
        # 1. Morgan fingerprints
        if fingerprints:
            morgan_count = len(fingerprints)
            logging.info(f"  Morgan fingerprints: {morgan_count:,} compounds")
            if morgan_count > 0:
                sample_fp = list(fingerprints.values())[0]
                logging.info(f"    Size per compound: {2048} bits ({len(sample_fp)} float32 values)")
                logging.info(f"    Total size (approx): {format_size(morgan_count * 2048 * 4 / 8)}")
        
        # 2. Physicochemical descriptors (test on one compound)
        if sample_smiles:
            test_smiles = sample_smiles[0]
            logging.info(f"\n  Physicochemical descriptors:")
            logging.info(f"    Testing on: {test_smiles[:50]}...")
            
            physico = fingerprint_loader.calculate_physicochemical_descriptors(test_smiles)
            if physico:
                logging.info(f"    Descriptors calculated: {len(physico)}")
                logging.info(f"    Size: {len(physico)} float values (~{len(physico) * 8} bytes per compound)")
                # Show first 3 descriptors
                for i, (key, value) in enumerate(list(physico.items())[:3]):
                    logging.info(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")
            else:
                logging.info(f"    Failed to calculate")
        
        # 3. ESM-C embeddings
        logging.info(f"\n  ESM-C embeddings:")
        esmc_desc = fingerprint_loader.load_esmc_descriptors(test_uniprot)
        if esmc_desc is not None:
            logging.info(f"    Descriptors loaded: {esmc_desc.shape}")
            logging.info(f"    Size: {esmc_desc.size} float values (~{esmc_desc.nbytes / 1024:.2f} KB)")
        else:
            logging.info(f"    Not found in cache")
        
        # 4. Activity thresholds
        logging.info(f"\n  Activity thresholds:")
        thresholds = thresholds_loader.get_thresholds(test_uniprot)
        if thresholds:
            logging.info(f"    High cutoff: {thresholds['high']}")
            logging.info(f"    Medium cutoff: {thresholds['medium']}")
        else:
            logging.info(f"    No thresholds found")
    
    logging.info("\n" + "="*80)
    
    # AQSE 3-Class Workflow (commented out - uncomment to run workflow)
    # Uncomment below to process all proteins and train models
    workflow = AQSE3CWorkflow(config)
    workflow.process_all_proteins()






if __name__ == "__main__":
    main()



