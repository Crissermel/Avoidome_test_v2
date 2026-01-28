"""
Data preparation utilities for AQSE workflow.

This module provides functions for preparing data for model training,
including feature extraction and data splitting coordination.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..utils.feature_extraction import (
    extract_compound_features,
    extract_protein_features,
    create_feature_dataset,
    extract_multi_protein_features,
    assign_class_labels
)
from ..utils.data_splitting import split_data_stratified

logger = logging.getLogger(__name__)


def prepare_model_a_data(
    bioactivity_data: pd.DataFrame,
    uniprot_id: str,
    thresholds: Dict[str, Any],
    model_trainer,
    cached_fingerprints: Dict[str, np.ndarray],
    bioactivity_loader,
    config: Dict[str, Any]
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
    bioactivity_loader
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
        
    Returns:
        Tuple of (train_features, test_features)
    """
    # Filter for valid data - check if columns exist first
    required_cols = ['SMILES', 'pchembl_value_Mean']
    missing_cols = [col for col in required_cols if col not in similar_data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in similar_data: {missing_cols}")
    missing_cols = [col for col in required_cols if col not in target_data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in target_data: {missing_cols}")
    
    similar_data = similar_data.dropna(subset=required_cols)
    target_data = target_data.dropna(subset=required_cols)
    similar_data = similar_data[similar_data['SMILES'] != '']
    target_data = target_data[target_data['SMILES'] != '']
    
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
