"""
Chemprop model trainer implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

from .base import ModelTrainer

# Chemprop imports (optional)
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

logger = logging.getLogger(__name__)


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
            except TypeError as e:
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
            
            # Create filtered test DataFrame aligned with predictions
            # test_valid_indices tracks which samples from test_df_valid passed molecule creation
            test_df_final = test_df_valid.iloc[test_valid_indices[:min_len]].reset_index(drop=True)
            
            # Calculate metrics
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
                'test_df_valid': test_df_final,
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
                'class_labels': class_labels,
                'n_train_samples': len(train_df_valid),
                'n_test_samples': len(test_df_valid),
                'feature_dimensions': train_features.shape[1] if train_features.ndim > 1 else len(train_features[0])
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
