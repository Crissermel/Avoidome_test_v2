#!/usr/bin/env python3
"""
Script to create a protein-level summary table with:
- Accession (UniProt ID)
- Thresholds (cutoff_high, cutoff_medium)
- Optimized thresholds (optimized_cutoff_high, optimized_cutoff_medium)
- Binarization (2-class vs 3-class)
- RF F1 macro performance
- Best model
- Similar proteins
- Datapoints of target protein
- Datapoints of similar proteins
- Total datapoints
"""

import pandas as pd
import numpy as np
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Set, List
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_activity_thresholds(thresholds_file: str) -> pd.DataFrame:
    """
    Load activity thresholds from CSV file.
    
    Args:
        thresholds_file: Path to thresholds CSV file
        
    Returns:
        DataFrame with threshold information
    """
    logger.info(f"Loading activity thresholds from {thresholds_file}")
    df = pd.read_csv(thresholds_file)
    df.columns = df.columns.str.strip()
    return df


def load_similar_proteins(similarity_file: str) -> Dict[str, List[str]]:
    """
    Load similar proteins from similarity search results.
    
    Args:
        similarity_file: Path to similarity search summary CSV
        
    Returns:
        Dictionary mapping target_id to list of similar protein IDs
    """
    logger.info(f"Loading similar proteins from {similarity_file}")
    df = pd.read_csv(similarity_file)
    df.columns = df.columns.str.strip()
    
    similar_proteins_dict = {}
    
    for _, row in df.iterrows():
        query_protein = str(row['query_protein']).strip()
        
        if query_protein not in similar_proteins_dict:
            similar_proteins_dict[query_protein] = []
        
        # Parse similar_proteins column
        similar_proteins_str = row.get('similar_proteins', '')
        if pd.notna(similar_proteins_str) and similar_proteins_str.strip():
            # Extract UniProt IDs from strings like "P05177 (100.0%), P04799 (75.1%)"
            matches = re.findall(r'([OPQ][0-9A-Z]{5})(?:_WT)?\s*\(', str(similar_proteins_str))
            for protein_id in matches:
                if protein_id and protein_id != query_protein:
                    if protein_id not in similar_proteins_dict[query_protein]:
                        similar_proteins_dict[query_protein].append(protein_id)
    
    logger.info(f"Found similar proteins for {len(similar_proteins_dict)} target proteins")
    return similar_proteins_dict


def get_2class_proteins() -> Set[str]:
    """
    Get list of proteins that use 2-class classification.
    Based on the code in AQSE_3c_2c_clf_th_op.py
    
    Returns:
        Set of UniProt IDs that use 2-class classification
    """
    # From the code in AQSE_3c_2c_clf_th_op.py line 1670-1676
    two_class_proteins = {
        'P28845',  # HSD11B1
        'P31645',  # SLC6A4
        'P35348',  # ADRA1A
        'P20309',  # CHRM3
        'P07550',  # ADRB2
    }
    return two_class_proteins


def load_model_metrics(workflow_results_file: str, metrics_dir: Optional[Path] = None, rf_metrics_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Load model performance metrics from workflow results and/or metrics files.
    Prioritizes Random Forest metrics if available.
    
    Args:
        workflow_results_file: Path to workflow_results.csv
        metrics_dir: Optional directory containing metrics JSON files (Chemprop)
        rf_metrics_dir: Optional directory containing Random Forest metrics JSON files
        
    Returns:
        Dictionary mapping (uniprot_id, model_type, threshold) to metrics
    """
    logger.info(f"Loading model metrics from {workflow_results_file}")
    metrics_dict = {}
    
    # Try to load Random Forest metrics first (priority)
    if rf_metrics_dir and rf_metrics_dir.exists():
        logger.info(f"Scanning Random Forest metrics directory: {rf_metrics_dir}")
        metrics_files = list(rf_metrics_dir.glob("*_test_metrics.json"))
        logger.info(f"Found {len(metrics_files)} Random Forest metrics files")
        
        for metrics_file in metrics_files:
            try:
                # Parse filename: {protein}_{uniprot_id}_{model_type}_{threshold}_test_metrics.json
                parts = metrics_file.stem.replace('_test_metrics', '').split('_')
                if len(parts) >= 3:
                    uniprot_id = parts[-2] if parts[-2].startswith(('P', 'Q', 'O')) else parts[-3]
                    model_type = parts[-1] if parts[-1] in ['A', 'B'] else (parts[-2] if parts[-2] in ['A', 'B'] else 'A')
                    threshold = None
                    if model_type == 'B' and len(parts) >= 4:
                        threshold = parts[-1] if parts[-1] in ['high', 'medium', 'low'] else None
                    
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    key = (uniprot_id, model_type, threshold)
                    metrics_dict[key] = metrics
            except Exception as e:
                logger.debug(f"Error loading {metrics_file}: {e}")
    
    # Fall back to Chemprop metrics if RF not available
    elif metrics_dir and metrics_dir.exists():
        logger.info(f"Scanning metrics directory: {metrics_dir}")
        metrics_files = list(metrics_dir.glob("*_test_metrics.json"))
        logger.info(f"Found {len(metrics_files)} metrics files (Chemprop)")
        
        for metrics_file in metrics_files:
            try:
                # Parse filename: {protein}_{uniprot_id}_{model_type}_{threshold}_test_metrics.json
                parts = metrics_file.stem.replace('_test_metrics', '').split('_')
                if len(parts) >= 3:
                    uniprot_id = parts[-2] if parts[-2].startswith(('P', 'Q', 'O')) else parts[-3]
                    model_type = parts[-1] if parts[-1] in ['A', 'B'] else (parts[-2] if parts[-2] in ['A', 'B'] else 'A')
                    threshold = None
                    if model_type == 'B' and len(parts) >= 4:
                        threshold = parts[-1] if parts[-1] in ['high', 'medium', 'low'] else None
                    
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    key = (uniprot_id, model_type, threshold)
                    metrics_dict[key] = metrics
            except Exception as e:
                logger.debug(f"Error loading {metrics_file}: {e}")
    
    return metrics_dict


def count_datapoints(papyrus_df: pd.DataFrame, protein_ids: Set[str]) -> int:
    """
    Count bioactivity datapoints for given proteins.
    
    Args:
        papyrus_df: Papyrus DataFrame
        protein_ids: Set of UniProt IDs
        
    Returns:
        Number of valid datapoints
    """
    filtered = papyrus_df[papyrus_df['accession'].isin(protein_ids)]
    filtered = filtered.dropna(subset=['SMILES', 'pchembl_value_Mean'])
    filtered = filtered[filtered['SMILES'] != '']
    return len(filtered)


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


def create_protein_summary_table(
    thresholds_df: pd.DataFrame,
    similar_proteins_dict: Dict[str, List[str]],
    two_class_proteins: Set[str],
    papyrus_df: pd.DataFrame,
    metrics_dict: Dict = None
) -> pd.DataFrame:
    """
    Create the protein-level summary table.
    
    Args:
        thresholds_df: DataFrame with threshold information
        similar_proteins_dict: Dictionary mapping target_id to list of similar proteins
        two_class_proteins: Set of UniProt IDs using 2-class classification
        papyrus_df: Papyrus DataFrame
        metrics_dict: Dictionary with model metrics
        
    Returns:
        DataFrame with protein summary
    """
    logger.info("Creating protein summary table...")
    
    results = []
    
    for _, row in thresholds_df.iterrows():
        uniprot_id = str(row['uniprot_id']).strip()
        if pd.isna(uniprot_id) or uniprot_id == '':
            continue
        
        # Get thresholds
        cutoff_high = row.get('original_cutoff_high', row.get('cutoff_high', np.nan))
        cutoff_medium = row.get('original_cutoff_medium', row.get('cutoff_medium', np.nan))
        optimized_cutoff_high = row.get('optimized_cutoff_high', cutoff_high)
        optimized_cutoff_medium = row.get('optimized_cutoff_medium', cutoff_medium)
        
        # Determine binarization
        binarization = '2-class' if uniprot_id in two_class_proteins else '3-class'
        
        # Get similar proteins
        similar_proteins_list = similar_proteins_dict.get(uniprot_id, [])
        similar_proteins_str = ', '.join(sorted(similar_proteins_list)) if similar_proteins_list else 'None'
        
        # Count datapoints
        target_datapoints = count_datapoints(papyrus_df, {uniprot_id})
        similar_datapoints = count_datapoints(papyrus_df, set(similar_proteins_list)) if similar_proteins_list else 0
        total_datapoints = target_datapoints + similar_datapoints
        
        # Get RF F1 macro and best model
        rf_f1_macro = np.nan
        best_model = 'N/A'
        
        if metrics_dict:
            # Look for Random Forest metrics (model_type could be 'A' or 'B')
            # Try to find best performing model
            best_f1 = -1
            best_key = None
            
            for key, metrics in metrics_dict.items():
                key_uniprot, model_type, threshold = key
                if key_uniprot == uniprot_id:
                    f1 = metrics.get('f1_macro', -1)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_key = key
            
            if best_key:
                key_uniprot, model_type, threshold = best_key
                metrics = metrics_dict[best_key]
                rf_f1_macro = metrics.get('f1_macro', np.nan)
                best_model = f"Model {model_type}"
                if threshold:
                    best_model += f" ({threshold})"
        
        results.append({
            'Accession': uniprot_id,
            'cutoff_high': cutoff_high,
            'cutoff_medium': cutoff_medium,
            'optimized_cutoff_high': optimized_cutoff_high,
            'optimized_cutoff_medium': optimized_cutoff_medium,
            'binarization': binarization,
            'RF_F1_macro': rf_f1_macro,
            'best_model': best_model,
            'similar_proteins': similar_proteins_str,
            'target_datapoints': target_datapoints,
            'similar_datapoints': similar_datapoints,
            'total_datapoints': total_datapoints
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Accession')
    
    logger.info(f"Created summary for {len(result_df)} proteins")
    return result_df
