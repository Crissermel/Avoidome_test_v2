#!/usr/bin/env python3
"""
Script to create a comprehensive bioactivity table from Papyrus dataset for avoidome proteins
AND their similar proteins.

The table includes:
- Accession (UniProt ID)
- Inchikey (calculated from SMILES)
- pChembl_value (mean pChEMBL value)
- activity threshold (binary) - active/inactive
- activity thresholds (low/med/high) - 3-class classification
- type (EC50/IC50/Ki/Kd) - activity measurement type
- protein_category (target/similar) - indicates if protein is target or similar
- target_protein (for similar proteins) - which avoidome protein this is similar to
"""

import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Set
from rdkit import Chem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_inchikey(smiles: str) -> Optional[str]:
    """
    Calculate InChIKey from SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        InChIKey string or None if conversion fails
    """
    try:
        if pd.isna(smiles) or smiles == '':
            return None
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        inchikey = Chem.MolToInchiKey(mol)
        return inchikey
    except Exception as e:
        logger.debug(f"Error calculating InChIKey for SMILES {smiles[:50]}: {e}")
        return None


def assign_binary_label(pchembl_value: float, cutoff_high: float) -> str:
    """
    Assign binary activity label based on pChEMBL value and high cutoff.
    
    Args:
        pchembl_value: pChEMBL value
        cutoff_high: High activity cutoff threshold
        
    Returns:
        'active' if pchembl_value > cutoff_high, else 'inactive'
    """
    if pd.isna(pchembl_value):
        return 'unknown'
    if pchembl_value > cutoff_high:
        return 'active'
    else:
        return 'inactive'


def assign_3class_label(pchembl_value: float, cutoff_medium: float, cutoff_high: float) -> str:
    """
    Assign 3-class activity label based on pChEMBL value and thresholds.
    
    Args:
        pchembl_value: pChEMBL value
        cutoff_medium: Medium activity cutoff threshold
        cutoff_high: High activity cutoff threshold
        
    Returns:
        'Low', 'Medium', or 'High' based on thresholds
    """
    if pd.isna(pchembl_value):
        return 'unknown'
    if pchembl_value <= cutoff_medium:
        return 'Low'
    elif cutoff_medium < pchembl_value <= cutoff_high:
        return 'Medium'
    else:
        return 'High'


def load_activity_thresholds(thresholds_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load activity thresholds from CSV file.
    
    Args:
        thresholds_file: Path to thresholds CSV file
        
    Returns:
        Dictionary mapping uniprot_id to thresholds dict with 'cutoff_high' and 'cutoff_medium'
    """
    logger.info(f"Loading activity thresholds from {thresholds_file}")
    df = pd.read_csv(thresholds_file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Create mapping
    thresholds_dict = {}
    for _, row in df.iterrows():
        uniprot_id = row.get('uniprot_id')
        if pd.isna(uniprot_id):
            continue
        
        cutoff_high = row.get('cutoff_high')
        cutoff_medium = row.get('cutoff_medium')
        
        # Check for optimized cutoffs
        if 'optimized_cutoff_high' in df.columns and pd.notna(row.get('optimized_cutoff_high')):
            cutoff_high = row.get('optimized_cutoff_high')
        if 'optimized_cutoff_medium' in df.columns and pd.notna(row.get('optimized_cutoff_medium')):
            cutoff_medium = row.get('optimized_cutoff_medium')
        
        if pd.notna(cutoff_high) and pd.notna(cutoff_medium):
            thresholds_dict[str(uniprot_id)] = {
                'cutoff_high': float(cutoff_high),
                'cutoff_medium': float(cutoff_medium)
            }
    
    logger.info(f"Loaded thresholds for {len(thresholds_dict)} proteins")
    return thresholds_dict


def load_papyrus_data() -> pd.DataFrame:
    """
    Load Papyrus dataset.
    
    Returns:
        DataFrame with Papyrus bioactivity data
    """
    logger.info("Loading Papyrus dataset...")
    logger.info("This may take ~5 minutes on first load")
    try:
        from papyrus_scripts import PapyrusDataset
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        papyrus_df = papyrus_data.to_dataframe()
        logger.info(f"Loaded {len(papyrus_df):,} total activities from Papyrus")
        return papyrus_df
    except Exception as e:
        logger.error(f"Error loading Papyrus dataset: {e}")
        raise


def get_avoidome_uniprot_ids(avoidome_file: str) -> Set[str]:
    """
    Get list of UniProt IDs from avoidome file.
    
    Args:
        avoidome_file: Path to avoidome protein list CSV
        
    Returns:
        Set of UniProt IDs
    """
    logger.info(f"Loading avoidome protein list from {avoidome_file}")
    df = pd.read_csv(avoidome_file)
    
    # Clean columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    
    # Get UniProt IDs
    uniprot_ids = set()
    if 'UniProt ID' in df.columns:
        uniprot_ids = set(df['UniProt ID'].dropna().astype(str))
    elif 'uniprot_id' in df.columns:
        uniprot_ids = set(df['uniprot_id'].dropna().astype(str))
    
    logger.info(f"Found {len(uniprot_ids)} unique UniProt IDs in avoidome list")
    return uniprot_ids


def load_similar_proteins(similarity_file: str, avoidome_ids: Set[str]) -> Dict[str, Set[str]]:
    """
    Load similar proteins from similarity search results.
    
    Args:
        similarity_file: Path to similarity search summary CSV
        avoidome_ids: Set of avoidome UniProt IDs
        
    Returns:
        Dictionary mapping avoidome_id to set of similar protein IDs
    """
    logger.info(f"Loading similar proteins from {similarity_file}")
    df = pd.read_csv(similarity_file)
    df.columns = df.columns.str.strip()
    
    # Dictionary to store similar proteins for each target
    similar_proteins_dict = {}
    
    for _, row in df.iterrows():
        query_protein = str(row['query_protein']).strip()
        
        # Only process avoidome proteins
        if query_protein not in avoidome_ids:
            continue
        
        if query_protein not in similar_proteins_dict:
            similar_proteins_dict[query_protein] = set()
        
        # Parse similar_proteins column
        similar_proteins_str = row.get('similar_proteins', '')
        if pd.notna(similar_proteins_str) and similar_proteins_str.strip():
            # Extract UniProt IDs from strings like "P05177 (100.0%), P04799 (75.1%)"
            # Pattern matches UniProt IDs: P/Q/O followed by 5 alphanumeric characters
            matches = re.findall(r'([OPQ][0-9A-Z]{5})(?:_WT)?\s*\(', str(similar_proteins_str))
            for protein_id in matches:
                if protein_id and protein_id != query_protein:  # Exclude query itself
                    similar_proteins_dict[query_protein].add(protein_id)
    
    # Log statistics
    total_similar = sum(len(sims) for sims in similar_proteins_dict.values())
    unique_similar = set()
    for sims in similar_proteins_dict.values():
        unique_similar.update(sims)
    
    logger.info(f"Found {total_similar} similar protein relationships")
    logger.info(f"Found {len(unique_similar)} unique similar proteins")
    logger.info(f"Similar proteins found for {len(similar_proteins_dict)} target proteins")
    
    return similar_proteins_dict


def create_bioactivity_table(
    papyrus_df: pd.DataFrame,
    thresholds_dict: Dict[str, Dict[str, float]],
    avoidome_uniprot_ids: Set[str],
    similar_proteins_dict: Dict[str, Set[str]]
) -> pd.DataFrame:
    """
    Create the comprehensive bioactivity table with target and similar proteins.
    
    Args:
        papyrus_df: Papyrus DataFrame
        thresholds_dict: Dictionary mapping uniprot_id to thresholds
        avoidome_uniprot_ids: Set of avoidome UniProt IDs
        similar_proteins_dict: Dictionary mapping target_id to set of similar protein IDs
        
    Returns:
        DataFrame with all required columns
    """
    # Collect all protein IDs to include (target + similar)
    all_protein_ids = set(avoidome_uniprot_ids)
    for similar_set in similar_proteins_dict.values():
        all_protein_ids.update(similar_set)
    
    logger.info(f"Filtering Papyrus data for {len(avoidome_uniprot_ids)} target proteins and {len(all_protein_ids) - len(avoidome_uniprot_ids)} similar proteins...")
    
    # Filter for all proteins (target + similar)
    filtered_df = papyrus_df[papyrus_df['accession'].isin(all_protein_ids)].copy()
    logger.info(f"Found {len(filtered_df):,} bioactivity records")
    
    # Filter for valid data
    logger.info("Filtering for valid SMILES and pchembl values...")
    filtered_df = filtered_df.dropna(subset=['SMILES', 'pchembl_value_Mean'])
    filtered_df = filtered_df[filtered_df['SMILES'] != '']
    logger.info(f"After filtering: {len(filtered_df):,} records")
    
    # Initialize result columns
    logger.info("Calculating InChIKeys from SMILES...")
    filtered_df['Inchikey'] = filtered_df['SMILES'].apply(calculate_inchikey)
    
    # Remove rows where InChIKey calculation failed
    initial_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Inchikey'].notna()]
    logger.info(f"Removed {initial_count - len(filtered_df)} records with failed InChIKey calculation")
    
    # Add protein category and target protein columns
    logger.info("Categorizing proteins (target vs similar)...")
    def categorize_protein(row):
        accession = str(row['accession'])
        if accession in avoidome_uniprot_ids:
            return pd.Series({
                'protein_category': 'target',
                'target_protein': accession
            })
        else:
            # Find which target protein(s) this is similar to
            target_proteins = []
            for target_id, similar_set in similar_proteins_dict.items():
                if accession in similar_set:
                    target_proteins.append(target_id)
            
            if target_proteins:
                # If similar to multiple targets, join with semicolon
                return pd.Series({
                    'protein_category': 'similar',
                    'target_protein': ';'.join(sorted(target_proteins))
                })
            else:
                # Should not happen, but handle gracefully
                return pd.Series({
                    'protein_category': 'unknown',
                    'target_protein': ''
                })
    
    category_df = filtered_df.apply(categorize_protein, axis=1)
    filtered_df['protein_category'] = category_df['protein_category']
    filtered_df['target_protein'] = category_df['target_protein']
    
    # Convert activity type binary columns to single type column
    logger.info("Converting activity type binary columns to single type column...")
    def get_activity_type(row):
        """Determine activity type from binary columns (handles string values and semicolon-separated)"""
        def has_type(col_name):
            val = row.get(col_name, 0)
            if pd.isna(val):
                return False
            val_str = str(val)
            return '1' in val_str.split(';') if ';' in val_str else val_str == '1'
        
        if has_type('type_IC50'):
            return 'IC50'
        elif has_type('type_EC50'):
            return 'EC50'
        elif has_type('type_KD'):
            return 'Kd'
        elif has_type('type_Ki'):
            return 'Ki'
        elif has_type('type_other'):
            return 'other'
        else:
            return 'unknown'
    
    type_columns = ['type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other']
    has_type_columns = any(col in filtered_df.columns for col in type_columns)
    
    if has_type_columns:
        filtered_df['activity_type'] = filtered_df.apply(get_activity_type, axis=1)
        logger.info(f"Activity type distribution: {filtered_df['activity_type'].value_counts().to_dict()}")
    else:
        logger.warning("Activity type columns not found. Setting all to 'unknown'")
        filtered_df['activity_type'] = 'unknown'
    
    # Assign activity labels
    logger.info("Assigning activity labels...")
    
    def assign_labels(row):
        accession = str(row['accession'])
        pchembl_value = row['pchembl_value_Mean']
        
        # For similar proteins, use thresholds from their target protein
        # If similar to multiple targets, use the first one
        target_id = row['target_protein']
        if ';' in target_id:
            target_id = target_id.split(';')[0]
        
        # Get thresholds for this protein (or its target)
        thresholds = thresholds_dict.get(target_id, {})
        cutoff_high = thresholds.get('cutoff_high', 6.0)  # Default
        cutoff_medium = thresholds.get('cutoff_medium', 5.0)  # Default
        
        binary_label = assign_binary_label(pchembl_value, cutoff_high)
        three_class_label = assign_3class_label(pchembl_value, cutoff_medium, cutoff_high)
        
        return pd.Series({
            'activity_threshold_binary': binary_label,
            'activity_thresholds_3class': three_class_label
        })
    
    label_df = filtered_df.apply(assign_labels, axis=1)
    filtered_df['activity_threshold_binary'] = label_df['activity_threshold_binary']
    filtered_df['activity_thresholds_3class'] = label_df['activity_thresholds_3class']
    
    # Create final table with required columns
    result_columns = {
        'Accession': filtered_df['accession'],
        'Inchikey': filtered_df['Inchikey'],
        'pChembl_value': filtered_df['pchembl_value_Mean'],
        'activity_threshold_binary': filtered_df['activity_threshold_binary'],
        'activity_thresholds_3class': filtered_df['activity_thresholds_3class'],
        'type': filtered_df['activity_type'],
        'protein_category': filtered_df['protein_category'],
        'target_protein': filtered_df['target_protein']
    }
    
    result_df = pd.DataFrame(result_columns)
    
    # Sort by protein_category (target first), then Accession, then pChembl_value
    result_df['sort_order'] = result_df['protein_category'].map({'target': 0, 'similar': 1, 'unknown': 2})
    result_df = result_df.sort_values(['sort_order', 'Accession', 'pChembl_value'], ascending=[True, True, False])
    result_df = result_df.drop(columns=['sort_order'])
    
    logger.info(f"Final table contains {len(result_df):,} records")
    logger.info(f"Target proteins: {result_df[result_df['protein_category'] == 'target']['Accession'].nunique()}")
    logger.info(f"Similar proteins: {result_df[result_df['protein_category'] == 'similar']['Accession'].nunique()}")
    
    return result_df
