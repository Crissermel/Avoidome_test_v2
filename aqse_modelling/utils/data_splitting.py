"""
Data splitting utilities for AQSE workflow.

This module provides functions for splitting data into train/validation/test sets,
including support for fixed test sets based on document IDs.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import random
import logging

logger = logging.getLogger(__name__)


def create_fixed_test_set(
    df: pd.DataFrame,
    doc_id_column: str = 'doc_id',
    min_molecules_per_doc: int = 20,
    molecules_per_doc: int = 2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def split_data_stratified(
    df: pd.DataFrame,
    test_size: float = 0.2,
    use_fixed_test: bool = False,
    doc_id_column: str = 'doc_id',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: DataFrame with 'class' column and optionally doc_id_column
        test_size: Fraction for validation set (when use_fixed_test=True, this is ignored for test set)
        use_fixed_test: If True, create fixed test set based on doc_id, then split remaining into train/val
        doc_id_column: Name of column containing document IDs
        random_state: Random seed for reproducibility
        
    Returns:
        If use_fixed_test: (train_df, val_df, test_df)
        Otherwise: (train_df, test_df, None)
    """
    if use_fixed_test and doc_id_column in df.columns:
        # First, create fixed test set based on doc_id
        remaining_df, test_df = create_fixed_test_set(
            df,
            doc_id_column=doc_id_column,
            min_molecules_per_doc=20,
            molecules_per_doc=2,
            random_state=random_state
        )
        
        # Then split remaining data into train and validation (80/20 stratified)
        train_df, val_df = train_test_split(
            remaining_df,
            test_size=test_size,
            stratify=remaining_df['class'] if 'class' in remaining_df.columns else None,
            random_state=random_state
        )
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    else:
        # Standard stratified split (train/test only)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['class'] if 'class' in df.columns else None,
            random_state=random_state
        )
        
        return train_df, test_df, None
