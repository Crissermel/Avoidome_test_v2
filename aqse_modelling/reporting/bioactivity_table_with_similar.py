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

import polars as pl
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
        if smiles is None or smiles == '':
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
    if pchembl_value is None:
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
    if pchembl_value is None:
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
    df = pl.read_csv(thresholds_file)
    
    # Clean column names - strip whitespace
    rename_dict = {col: col.strip() for col in df.columns}
    df = df.rename(rename_dict)
    
    # Create mapping
    thresholds_dict = {}
    for row in df.iter_rows(named=True):
        uniprot_id = row.get('uniprot_id')
        if uniprot_id is None:
            continue
        
        cutoff_high = row.get('cutoff_high')
        cutoff_medium = row.get('cutoff_medium')
        
        # Check for optimized cutoffs
        if 'optimized_cutoff_high' in df.columns and row.get('optimized_cutoff_high') is not None:
            cutoff_high = row.get('optimized_cutoff_high')
        if 'optimized_cutoff_medium' in df.columns and row.get('optimized_cutoff_medium') is not None:
            cutoff_medium = row.get('optimized_cutoff_medium')
        
        if cutoff_high is not None and cutoff_medium is not None:
            thresholds_dict[str(uniprot_id)] = {
                'cutoff_high': float(cutoff_high),
                'cutoff_medium': float(cutoff_medium)
            }
    
    logger.info(f"Loaded thresholds for {len(thresholds_dict)} proteins")
    return thresholds_dict


def load_papyrus_data() -> pl.DataFrame:
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
        papyrus_pd_df = papyrus_data.to_dataframe()
        papyrus_df = pl.from_pandas(papyrus_pd_df)
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
    df = pl.read_csv(avoidome_file)
    
    # Clean columns - remove Unnamed columns and strip whitespace
    df = df.select([col for col in df.columns if not col.startswith('Unnamed')])
    rename_dict = {col: col.strip() for col in df.columns}
    df = df.rename(rename_dict)
    
    # Get UniProt IDs
    uniprot_ids = set()
    if 'UniProt ID' in df.columns:
        uniprot_ids = set(df['UniProt ID'].drop_nulls().cast(pl.Utf8).to_list())
    elif 'uniprot_id' in df.columns:
        uniprot_ids = set(df['uniprot_id'].drop_nulls().cast(pl.Utf8).to_list())
    
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
    df = pl.read_csv(similarity_file)
    # Clean column names - strip whitespace
    rename_dict = {col: col.strip() for col in df.columns}
    df = df.rename(rename_dict)
    
    # Dictionary to store similar proteins for each target
    similar_proteins_dict = {}
    
    for row in df.iter_rows(named=True):
        query_protein = str(row['query_protein']).strip()
        
        # Only process avoidome proteins
        if query_protein not in avoidome_ids:
            continue
        
        if query_protein not in similar_proteins_dict:
            similar_proteins_dict[query_protein] = set()
        
        # Parse similar_proteins column
        similar_proteins_str = row.get('similar_proteins', '')
        if similar_proteins_str is not None and similar_proteins_str.strip():
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
    papyrus_df: pl.DataFrame,
    thresholds_dict: Dict[str, Dict[str, float]],
    avoidome_uniprot_ids: Set[str],
    similar_proteins_dict: Dict[str, Set[str]]
) -> pl.DataFrame:
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
    filtered_df = papyrus_df.filter(pl.col('accession').is_in(list(all_protein_ids)))
    logger.info(f"Found {len(filtered_df):,} bioactivity records")
    
    # Filter for valid data
    logger.info("Filtering for valid SMILES and pchembl values...")
    filtered_df = filtered_df.drop_nulls(subset=['SMILES', 'pchembl_value_Mean'])
    filtered_df = filtered_df.filter(pl.col('SMILES') != '')
    logger.info(f"After filtering: {len(filtered_df):,} records")
    
    # Initialize result columns
    logger.info("Calculating InChIKeys from SMILES...")
    # Convert to pandas temporarily for apply, then back to polars
    filtered_df_pd = filtered_df.to_pandas()
    filtered_df_pd['Inchikey'] = filtered_df_pd['SMILES'].apply(calculate_inchikey)
    filtered_df = pl.from_pandas(filtered_df_pd)
    
    # Remove rows where InChIKey calculation failed
    initial_count = len(filtered_df)
    filtered_df = filtered_df.filter(pl.col('Inchikey').is_not_null())
    logger.info(f"Removed {initial_count - len(filtered_df)} records with failed InChIKey calculation")
    
    # Add protein category and target protein columns
    logger.info("Categorizing proteins (target vs similar)...")
    def categorize_protein(row):
        accession = str(row['accession'])
        if accession in avoidome_uniprot_ids:
            return {'protein_category': 'target', 'target_protein': accession}
        else:
            # Find which target protein(s) this is similar to
            target_proteins = []
            for target_id, similar_set in similar_proteins_dict.items():
                if accession in similar_set:
                    target_proteins.append(target_id)
            
            if target_proteins:
                # If similar to multiple targets, join with semicolon
                return {'protein_category': 'similar', 'target_protein': ';'.join(sorted(target_proteins))}
            else:
                # Should not happen, but handle gracefully
                return {'protein_category': 'unknown', 'target_protein': ''}
    
    # Convert to pandas for apply, then back to polars
    filtered_df_pd = filtered_df.to_pandas()
    category_data = filtered_df_pd.apply(categorize_protein, axis=1, result_type='expand')
    filtered_df_pd['protein_category'] = category_data[0] if isinstance(category_data, tuple) else category_data['protein_category']
    filtered_df_pd['target_protein'] = category_data[1] if isinstance(category_data, tuple) else category_data['target_protein']
    filtered_df = pl.from_pandas(filtered_df_pd)
    
    # Convert activity type binary columns to single type column
    logger.info("Converting activity type binary columns to single type column...")
    def get_activity_type(row):
        """Determine activity type from binary columns (handles string values and semicolon-separated)"""
        def has_type(col_name):
            val = row.get(col_name, 0)
            if val is None:
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
        # Convert to pandas for apply, then back to polars
        filtered_df_pd = filtered_df.to_pandas()
        filtered_df_pd['activity_type'] = filtered_df_pd.apply(get_activity_type, axis=1)
        filtered_df = pl.from_pandas(filtered_df_pd)
        activity_type_counts = filtered_df.group_by('activity_type').agg(pl.count().alias('count'))
        logger.info(f"Activity type distribution: {activity_type_counts.to_dicts()}")
    else:
        logger.warning("Activity type columns not found. Setting all to 'unknown'")
        filtered_df = filtered_df.with_columns(pl.lit('unknown').alias('activity_type'))
    
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
        
        return {'activity_threshold_binary': binary_label, 'activity_thresholds_3class': three_class_label}
    
    # Convert to pandas for apply, then back to polars
    filtered_df_pd = filtered_df.to_pandas()
    label_data = filtered_df_pd.apply(assign_labels, axis=1, result_type='expand')
    filtered_df_pd['activity_threshold_binary'] = label_data[0] if isinstance(label_data, tuple) else label_data['activity_threshold_binary']
    filtered_df_pd['activity_thresholds_3class'] = label_data[1] if isinstance(label_data, tuple) else label_data['activity_thresholds_3class']
    filtered_df = pl.from_pandas(filtered_df_pd)
    
    # Create final table with required columns
    result_df = filtered_df.select([
        pl.col('accession').alias('Accession'),
        pl.col('Inchikey'),
        pl.col('pchembl_value_Mean').alias('pChembl_value'),
        pl.col('activity_threshold_binary'),
        pl.col('activity_thresholds_3class'),
        pl.col('activity_type').alias('type'),
        pl.col('protein_category'),
        pl.col('target_protein')
    ])
    
    # Sort by protein_category (target first), then Accession, then pChembl_value
    sort_order_map = {'target': 0, 'similar': 1, 'unknown': 2}
    result_df = result_df.with_columns(
        pl.col('protein_category').map_dict(sort_order_map).alias('sort_order')
    )
    result_df = result_df.sort(['sort_order', 'Accession', 'pChembl_value'], descending=[False, False, True])
    result_df = result_df.drop('sort_order')
    
    logger.info(f"Final table contains {len(result_df):,} records")
    logger.info(f"Target proteins: {result_df[result_df['protein_category'] == 'target']['Accession'].nunique()}")
    logger.info(f"Similar proteins: {result_df[result_df['protein_category'] == 'similar']['Accession'].nunique()}")
    
    return result_df
