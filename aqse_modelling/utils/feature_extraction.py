"""
Feature extraction utilities for AQSE workflow.

This module provides functions for extracting compound and protein features,
creating feature datasets, and assigning class labels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def extract_compound_features(
    smiles: str,
    cached_fingerprints: Dict[str, np.ndarray],
    bioactivity_loader
) -> Optional[np.ndarray]:
    """
    Extract features for a single compound
    
    Args:
        smiles: SMILES string
        cached_fingerprints: Dictionary mapping SMILES to Morgan fingerprints
        bioactivity_loader: BioactivityDataLoader instance for physicochemical descriptors
        
    Returns:
        Combined feature vector or None if extraction fails
    """
    features_list = []
    
    # 1. Morgan fingerprints (from cached dict)
    if smiles in cached_fingerprints:
        morgan_fp = cached_fingerprints[smiles]
        features_list.append(morgan_fp)
    else:
        logger.warning(f"Morgan fingerprint not found for {smiles}")
        return None
    
    # 2. Physicochemical descriptors
    try:
        physico_desc = bioactivity_loader.calculate_physicochemical_descriptors(smiles)
        if physico_desc:
            # Convert dict to numpy array
            physico_array = np.array(list(physico_desc.values()), dtype=np.float32)
            features_list.append(physico_array)
        else:
            logger.warning(f"Physicochemical descriptors failed for {smiles}")
            return None
    except Exception as e:
        logger.error(f"Error calculating physicochemical descriptors: {e}")
        return None
    
    # Combine all features
    if features_list:
        combined = np.concatenate(features_list)
        return combined
    return None


def extract_protein_features(uniprot_id: str, bioactivity_loader) -> Optional[np.ndarray]:
    """
    Extract ESM-C descriptors for a protein
    
    Args:
        uniprot_id: UniProt ID
        bioactivity_loader: BioactivityDataLoader instance
        
    Returns:
        ESM-C embedding array or None
    """
    esmc_desc = bioactivity_loader.load_esmc_descriptors(uniprot_id)
    return esmc_desc


def assign_class_labels(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
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
        logger.error("pchembl_value_Mean column not found")
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


def create_feature_dataset(
    df: pd.DataFrame,
    uniprot_id: str,
    cached_fingerprints: Dict[str, np.ndarray],
    bioactivity_loader,
    add_esmc: bool = True
) -> pd.DataFrame:
    """
    Create dataset with extracted features
    
    Args:
        df: Bioactivity DataFrame with SMILES
        uniprot_id: Protein ID for ESM-C descriptors
        cached_fingerprints: Dictionary mapping SMILES to Morgan fingerprints
        bioactivity_loader: BioactivityDataLoader instance
        add_esmc: Whether to add ESM-C features (for Model B)
        
    Returns:
        DataFrame with features and labels
    """
    features_data = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        
        # Extract compound features
        compound_features = extract_compound_features(smiles, cached_fingerprints, bioactivity_loader)
        if compound_features is None:
            continue
        
        # Start with compound features
        feature_vector = compound_features.tolist()
        
        # Add ESM-C features if requested (Model B)
        if add_esmc:
            esmc_features = extract_protein_features(uniprot_id, bioactivity_loader)
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


def extract_multi_protein_features(
    df: pd.DataFrame,
    protein_ids: List[str],
    cached_fingerprints: Dict[str, np.ndarray],
    bioactivity_loader,
    add_esmc: bool = True
) -> pd.DataFrame:
    """
    Extract features for data from multiple proteins
    
    Args:
        df: Bioactivity DataFrame with 'accession' column
        protein_ids: List of protein IDs in the dataset
        cached_fingerprints: Dictionary mapping SMILES to Morgan fingerprints
        bioactivity_loader: BioactivityDataLoader instance
        add_esmc: Whether to add ESM-C features
        
    Returns:
        DataFrame with features
    """
    features_data = []
    
    # Cache ESM-C descriptors for all proteins
    esmc_cache = {}
    if add_esmc:
        for pid in protein_ids:
            esmc_desc = extract_protein_features(pid, bioactivity_loader)
            if esmc_desc is not None:
                esmc_cache[pid] = esmc_desc
            else:
                logger.warning(f"ESM-C descriptors not available for {pid}")
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        protein_id = row.get('accession', protein_ids[0])
        
        # Extract compound features
        compound_features = extract_compound_features(smiles, cached_fingerprints, bioactivity_loader)
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
