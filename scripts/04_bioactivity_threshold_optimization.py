#!/usr/bin/env python3
"""
Class Imbalance Analysis Script

This script analyzes class imbalances for bioactivity data across 53 proteins from the avoidome list.
It loads bioactivity data from Papyrus++ and assigns classes (Low, Medium, High) based on protein-specific
cutoffs defined in the cutoff table.

The script performs the following analyses:
1. Loads protein list and cutoff thresholds from avoidome_cutoff_table.csv
2. Loads similarity search results to get similar proteins for each target
3. Loads bioactivity data from Papyrus++ for all proteins (target + similar)
4. Assigns class labels based on pchembl_value_Mean and protein-specific cutoffs
5. Calculates class distributions for each protein (target-only and target+similar)
6. Generates summary statistics and visualizations
7. Saves results to 04_bioactivity_threshold_optimization subdirectory


"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import yaml
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Analyze all proteins from the cutoff table (no filtering)
# Previously restricted to ALLOWED_PROTEINS, now includes all proteins


def load_cutoff_table(cutoff_table_path: str) -> pd.DataFrame:
    """
    Load the cutoff table with protein thresholds
    
    Args:
        cutoff_table_path: Path to the cutoff table CSV file
        
    Returns:
        DataFrame with protein information and cutoffs
    """
    logger.info(f"Loading cutoff table from {cutoff_table_path}")
    cutoff_df = pd.read_csv(cutoff_table_path)
    
    # Filter out proteins without uniprot_id
    cutoff_df = cutoff_df[cutoff_df['uniprot_id'].notna()]
    cutoff_df = cutoff_df[cutoff_df['uniprot_id'] != '']
    
    logger.info(f"Loaded {len(cutoff_df)} proteins from cutoff table")
    return cutoff_df


def load_similarity_summary(similarity_file_path: str) -> Dict[str, List[str]]:
    """
    Load similarity search summary and extract similar proteins for each target
    
    Args:
        similarity_file_path: Path to similarity_search_summary.csv
        
    Returns:
        Dictionary mapping target uniprot_id to list of similar protein uniprot_ids
    """
    logger.info(f"Loading similarity summary from {similarity_file_path}")
    
    try:
        similarity_df = pd.read_csv(similarity_file_path)
        
        # Dictionary to store similar proteins for each target
        # Key: target uniprot_id, Value: list of similar protein uniprot_ids (including target)
        similar_proteins_dict = {}
        
        for _, row in similarity_df.iterrows():
            query_protein = row['query_protein']
            threshold = row['threshold']
            similar_proteins_str = row.get('similar_proteins', '')
            
            # Use 'high' threshold similar proteins
            if threshold == 'high' and pd.notna(similar_proteins_str):
                # Extract protein IDs from string like "P05177 (100.0%), P04799 (75.1%)"
                protein_ids = []
                for protein_entry in similar_proteins_str.split(","):
                    # Extract protein ID (format: "P05177 (100.0%)" or "P20813_WT (100.0%)")
                    parts = protein_entry.strip().split(" ")[0]  # Get "P05177" or "P20813_WT"
                    
                    # Remove _WT suffix if present
                    if "_WT" in parts:
                        parts = parts.split("_WT")[0]
                    
                    if parts:  # Only add non-empty IDs
                        protein_ids.append(parts)
                
                # Store similar proteins (includes target protein itself)
                similar_proteins_dict[query_protein] = protein_ids
                logger.debug(f"  {query_protein}: {len(protein_ids)} similar proteins (including target)")
        
        logger.info(f"Loaded similarity data for {len(similar_proteins_dict)} target proteins")
        return similar_proteins_dict
        
    except Exception as e:
        logger.error(f"Error loading similarity summary: {e}")
        return {}


def load_papyrus_data() -> pd.DataFrame:
    """
    Load Papyrus++ dataset using papyrus_scripts
    
    Returns:
        DataFrame with bioactivity data from Papyrus++
    """
    try:
        from papyrus_scripts import PapyrusDataset
        
        logger.info("Loading Papyrus++ dataset...")
        logger.info("This may take ~5 minutes on first load")
        
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        papyrus_df = papyrus_data.to_dataframe()
        
        logger.info(f"Loaded {len(papyrus_df):,} total activities from Papyrus++")
        return papyrus_df
        
    except ImportError as e:
        logger.error(f"Error importing papyrus_scripts: {e}")
        logger.error("Make sure papyrus_scripts is installed: pip install papyrus-scripts")
        raise
    except Exception as e:
        logger.error(f"Error loading Papyrus++ dataset: {e}")
        raise


def assign_class_labels(
    pchembl_value: float,
    cutoff_medium: float,
    cutoff_high: float
) -> str:
    """
    Assign class label based on pchembl_value and cutoffs
    
    Args:
        pchembl_value: pchembl_value_Mean value
        cutoff_medium: Medium class cutoff threshold
        cutoff_high: High class cutoff threshold
        
    Returns:
        Class label: 'Low', 'Medium', or 'High'
    """
    if pd.isna(pchembl_value):
        return 'Unknown'
    
    if pchembl_value <= cutoff_medium:
        return 'Low'
    elif cutoff_medium < pchembl_value <= cutoff_high:
        return 'Medium'
    else:
        return 'High'


def analyze_protein_class_distribution(
    protein_data: pd.DataFrame,
    cutoff_medium: float,
    cutoff_high: float
) -> Dict[str, int]:
    """
    Analyze class distribution for a single protein
    
    Args:
        protein_data: DataFrame with bioactivity data for the protein
        cutoff_medium: Medium class cutoff threshold
        cutoff_high: High class cutoff threshold
        
    Returns:
        Dictionary with class counts
    """
    if protein_data.empty or 'pchembl_value_Mean' not in protein_data.columns:
        return {'Low': 0, 'Medium': 0, 'High': 0, 'Unknown': 0, 'Total': 0}
    
    # Assign classes
    protein_data = protein_data.copy()
    protein_data['class'] = protein_data['pchembl_value_Mean'].apply(
        lambda x: assign_class_labels(x, cutoff_medium, cutoff_high)
    )
    
    # Count classes
    class_counts = protein_data['class'].value_counts().to_dict()
    
    # Ensure all classes are present
    result = {
        'Low': class_counts.get('Low', 0),
        'Medium': class_counts.get('Medium', 0),
        'High': class_counts.get('High', 0),
        'Unknown': class_counts.get('Unknown', 0),
        'Total': len(protein_data)
    }
    
    return result


def calculate_imbalance_ratio_from_data(
    protein_data: pd.DataFrame,
    cutoff_medium: float,
    cutoff_high: float
) -> float:
    """
    Calculate imbalance ratio for given cutoffs and data
    
    Args:
        protein_data: DataFrame with bioactivity data
        cutoff_medium: Medium class cutoff threshold
        cutoff_high: High class cutoff threshold
        
    Returns:
        Imbalance ratio (max/min class count), or inf if any class is empty
    """
    if protein_data.empty:
        return float('inf')
    
    class_counts = analyze_protein_class_distribution(protein_data, cutoff_medium, cutoff_high)
    
    low = class_counts.get('Low', 0)
    medium = class_counts.get('Medium', 0)
    high = class_counts.get('High', 0)
    
    # If any class is empty, return inf (penalty)
    if min(low, medium, high) == 0:
        return float('inf')
    
    return max(low, medium, high) / min(low, medium, high)


def greedy_search_optimal_cutoffs(
    protein_data: pd.DataFrame,
    original_cutoff_medium: float,
    original_cutoff_high: float,
    step_size: float = 0.1,
    max_iterations: int = 100
) -> Tuple[float, float, float]:
    """
    Greedy search to find optimal cutoffs that minimize imbalance ratio
    
    Args:
        protein_data: DataFrame with bioactivity data
        original_cutoff_medium: Original medium cutoff
        original_cutoff_high: Original high cutoff
        step_size: Step size for greedy search (default 0.1)
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (optimized_cutoff_medium, optimized_cutoff_high, best_imbalance_ratio)
    """
    # Define constraints
    medium_min = max(4.5, original_cutoff_medium - 1.5) # before 3  ## gerard 5
    medium_max = min(6.0, original_cutoff_medium + 1.5) # before 5.7
    high_min = max(5.0, original_cutoff_high - 1.5) # before 4.5 ## gerard 6.5
    high_max = min(8.5, original_cutoff_high + 1.5) # before 7.5
    min_gap = 0.5
    max_gap = 2.5
    
    # Skip optimization if no data
    if protein_data.empty:
        return original_cutoff_medium, original_cutoff_high, float('inf')
    
    # Initialize with original cutoffs
    current_medium = original_cutoff_medium
    current_high = original_cutoff_high
    current_ratio = calculate_imbalance_ratio_from_data(protein_data, current_medium, current_high)
    
    best_medium = current_medium
    best_high = current_high
    best_ratio = current_ratio
    
    logger.debug(f"  Starting greedy search: original ratio={current_ratio:.2f}")
    
    # Greedy search with coordinate descent
    for iteration in range(max_iterations):
        improvement = False
        
        # Try optimizing cutoff_medium (fix cutoff_high)
        for direction in [-step_size, step_size]:
            candidate_medium = current_medium + direction
            
            # Check constraints
            if candidate_medium < medium_min or candidate_medium > medium_max:
                continue
            
            gap = current_high - candidate_medium
            if gap < min_gap or gap > max_gap:
                continue
            
            candidate_ratio = calculate_imbalance_ratio_from_data(
                protein_data, candidate_medium, current_high
            )
            
            if candidate_ratio < best_ratio:
                best_ratio = candidate_ratio
                best_medium = candidate_medium
                improvement = True
        
        # Update if better
        if best_medium != current_medium:
            current_medium = best_medium
            current_ratio = best_ratio
        
        # Try optimizing cutoff_high (fix cutoff_medium)
        for direction in [-step_size, step_size]:
            candidate_high = current_high + direction
            
            # Check constraints
            if candidate_high < high_min or candidate_high > high_max:
                continue
            
            gap = candidate_high - current_medium
            if gap < min_gap or gap > max_gap:
                continue
            
            candidate_ratio = calculate_imbalance_ratio_from_data(
                protein_data, current_medium, candidate_high
            )
            
            if candidate_ratio < best_ratio:
                best_ratio = candidate_ratio
                best_high = candidate_high
                improvement = True
        
        # Update if better
        if best_high != current_high:
            current_high = best_high
            current_ratio = best_ratio
        
        # Stop if no improvement
        if not improvement:
            break
    
    # Round to 1 decimal place and ensure constraints are still met
    best_medium = round(best_medium, 1)
    best_high = round(best_high, 1)
    
    # Ensure constraints after rounding
    best_medium = max(medium_min, min(medium_max, best_medium))
    best_high = max(high_min, min(high_max, best_high))
    
    # Ensure gap constraint
    gap = best_high - best_medium
    if gap < min_gap:
        # Adjust to meet minimum gap
        best_high = best_medium + min_gap
        best_high = min(high_max, best_high)
    elif gap > max_gap:
        # Adjust to meet maximum gap
        best_high = best_medium + max_gap
        best_high = min(high_max, best_high)
    
    # Recalculate ratio with final rounded values
    final_ratio = calculate_imbalance_ratio_from_data(protein_data, best_medium, best_high)
    
    logger.debug(f"  Optimized: medium={best_medium:.1f}, high={best_high:.1f}, ratio={final_ratio:.2f}")
    
    return best_medium, best_high, final_ratio


def calculate_imbalance_metrics(class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate imbalance metrics for class distribution
    
    Args:
        class_counts: Dictionary with class counts
        
    Returns:
        Dictionary with imbalance metrics
    """
    low = class_counts.get('Low', 0)
    medium = class_counts.get('Medium', 0)
    high = class_counts.get('High', 0)
    total = class_counts.get('Total', 0)
    
    if total == 0:
        return {
            'low_pct': 0.0,
            'medium_pct': 0.0,
            'high_pct': 0.0,
            'low_to_high_ratio': 0.0,
            'medium_to_high_ratio': 0.0,
            'imbalance_ratio': 0.0,
            'max_class': 'N/A',
            'min_class': 'N/A'
        }
    
    # Calculate percentages
    low_pct = (low / total) * 100 if total > 0 else 0.0
    medium_pct = (medium / total) * 100 if total > 0 else 0.0
    high_pct = (high / total) * 100 if total > 0 else 0.0
    
    # Calculate ratios
    low_to_high_ratio = low / high if high > 0 else float('inf') if low > 0 else 0.0
    medium_to_high_ratio = medium / high if high > 0 else float('inf') if medium > 0 else 0.0
    
    # Imbalance ratio: max class / min class (excluding zeros)
    class_values = [low, medium, high]
    non_zero_values = [v for v in class_values if v > 0]
    if len(non_zero_values) > 1:
        imbalance_ratio = max(non_zero_values) / min(non_zero_values)
    else:
        imbalance_ratio = 0.0
    
    # Identify max and min classes
    class_dict = {'Low': low, 'Medium': medium, 'High': high}
    max_class = max(class_dict, key=class_dict.get) if any(class_dict.values()) else 'N/A'
    min_class = min([k for k, v in class_dict.items() if v > 0], key=lambda k: class_dict[k]) if any(class_dict.values()) else 'N/A'
    
    return {
        'low_pct': low_pct,
        'medium_pct': medium_pct,
        'high_pct': high_pct,
        'low_to_high_ratio': low_to_high_ratio,
        'medium_to_high_ratio': medium_to_high_ratio,
        'imbalance_ratio': imbalance_ratio,
        'max_class': max_class,
        'min_class': min_class
    }


def analyze_2class_options(
    protein_data: pd.DataFrame,
    cutoff_medium: float,
    cutoff_high: float
) -> Dict[str, Any]:
    """
    Analyze all 2-class classification options and find the best one
    
    Args:
        protein_data: DataFrame with bioactivity data
        cutoff_medium: Medium cutoff threshold
        cutoff_high: High cutoff threshold
        
    Returns:
        Dictionary with best 2-class option and metrics
    """
    if protein_data.empty or 'pchembl_value_Mean' not in protein_data.columns:
        return {
            'best_option': None,
            'best_ratio': float('inf'),
            'best_cutoff_to_use': None,
            'lowmedium_vs_high_ratio': float('inf'),
            'low_vs_mediumhigh_ratio': float('inf'),
            'medium_vs_high_ratio': float('inf')
        }
    
    # Get 3-class distribution
    class_counts_3class = analyze_protein_class_distribution(protein_data, cutoff_medium, cutoff_high)
    low = class_counts_3class.get('Low', 0)
    medium = class_counts_3class.get('Medium', 0)
    high = class_counts_3class.get('High', 0)
    
    # Option 1: Low+Medium vs High (remove medium cutoff, use only high)
    low_medium = low + medium
    if min(low_medium, high) > 0:
        ratio_lowmedium_vs_high = max(low_medium, high) / min(low_medium, high)
    else:
        ratio_lowmedium_vs_high = float('inf')
    
    # Option 2: Low vs Medium+High (remove high cutoff, use only medium)
    medium_high = medium + high
    if min(low, medium_high) > 0:
        ratio_low_vs_mediumhigh = max(low, medium_high) / min(low, medium_high)
    else:
        ratio_low_vs_mediumhigh = float('inf')
    
    # Option 3: Medium vs High (remove low cutoff)
    if min(medium, high) > 0:
        ratio_medium_vs_high = max(medium, high) / min(medium, high)
    else:
        ratio_medium_vs_high = float('inf')
    
    # Find best option
    ratios = {
        'Low+Medium vs High': ratio_lowmedium_vs_high,
        'Low vs Medium+High': ratio_low_vs_mediumhigh,
        'Medium vs High': ratio_medium_vs_high
    }
    
    # Filter out infinite ratios
    finite_ratios = {k: v for k, v in ratios.items() if np.isfinite(v)}
    
    if not finite_ratios:
        return {
            'best_option': None,
            'best_ratio': float('inf'),
            'best_cutoff_to_use': None,
            'lowmedium_vs_high_ratio': ratio_lowmedium_vs_high,
            'low_vs_mediumhigh_ratio': ratio_low_vs_mediumhigh,
            'medium_vs_high_ratio': ratio_medium_vs_high
        }
    
    best_option = min(finite_ratios, key=finite_ratios.get)
    best_ratio = finite_ratios[best_option]
    
    # Determine which cutoff to use
    if best_option == 'Low+Medium vs High':
        best_cutoff_to_use = 'high'
    elif best_option == 'Low vs Medium+High':
        best_cutoff_to_use = 'medium'
    else:  # Medium vs High
        best_cutoff_to_use = 'high'  # Use high cutoff, ignore low
    
    return {
        'best_option': best_option,
        'best_ratio': best_ratio,
        'best_cutoff_to_use': best_cutoff_to_use,
        'lowmedium_vs_high_ratio': ratio_lowmedium_vs_high,
        'low_vs_mediumhigh_ratio': ratio_low_vs_mediumhigh,
        'medium_vs_high_ratio': ratio_medium_vs_high
    }


def should_use_2class(
    optimized_3class_ratio: float,
    best_2class_ratio: float,
    class_counts: Dict[str, int],
    improvement_threshold: float = 20.0,
    max_2class_ratio: float = 2.5,
    min_samples: int = 30
) -> bool:
    """
    Determine if a protein should use 2-class classification
    
    Args:
        optimized_3class_ratio: Optimized 3-class imbalance ratio
        best_2class_ratio: Best 2-class imbalance ratio
        class_counts: Class counts dictionary
        improvement_threshold: Minimum % improvement to use 2-class (default 20%)
        max_2class_ratio: Maximum acceptable 2-class ratio (default 2.5)
        min_samples: Minimum samples required (default 30)
        
    Returns:
        True if protein should use 2-class, False otherwise
    """
    total = class_counts.get('Total', 0)
    low = class_counts.get('Low', 0)
    medium = class_counts.get('Medium', 0)
    high = class_counts.get('High', 0)
    
    # Check minimum samples
    if total < min_samples:
        return False
    
    # Check if any class is empty (automatic 2-class)
    if min(low, medium, high) == 0:
        return True
    
    # Check if 2-class ratio is infinite
    if not np.isfinite(best_2class_ratio):
        return False
    
    # Check if 2-class ratio exceeds maximum
    if best_2class_ratio > max_2class_ratio:
        return False
    
    # Check improvement threshold
    if np.isfinite(optimized_3class_ratio) and optimized_3class_ratio > 0:
        improvement_pct = ((optimized_3class_ratio - best_2class_ratio) / optimized_3class_ratio) * 100.0
        if improvement_pct >= improvement_threshold:
            return True
    
    # If 3-class ratio is infinite but 2-class is finite, use 2-class
    if not np.isfinite(optimized_3class_ratio) and np.isfinite(best_2class_ratio):
        return True
    
    return False


def create_summary_statistics(
    results: List[Dict],
    output_dir: Path,
    suffix: str = ""
) -> pd.DataFrame:
    """
    Create summary statistics DataFrame and save to CSV
    
    Args:
        results: List of dictionaries with protein analysis results
        output_dir: Output directory for saving results
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for result in results:
        summary_row = {
            'name2_entry': result['name2_entry'],
            'uniprot_id': result['uniprot_id'],
            'protein_family': result.get('protein_family', 'N/A'),
            'cutoff_high': result['cutoff_high'],
            'cutoff_medium': result['cutoff_medium'],
            'total_datapoints': result['class_counts']['Total'],
            'low_count': result['class_counts']['Low'],
            'medium_count': result['class_counts']['Medium'],
            'high_count': result['class_counts']['High'],
            'unknown_count': result['class_counts']['Unknown'],
            'low_pct': result['metrics']['low_pct'],
            'medium_pct': result['metrics']['medium_pct'],
            'high_pct': result['metrics']['high_pct'],
            'low_to_high_ratio': result['metrics']['low_to_high_ratio'],
            'medium_to_high_ratio': result['metrics']['medium_to_high_ratio'],
            'imbalance_ratio': result['metrics']['imbalance_ratio'],
            'max_class': result['metrics']['max_class'],
            'min_class': result['metrics']['min_class']
        }
        
        # Add n_similar_proteins if available
        if 'n_similar_proteins' in result:
            summary_row['n_similar_proteins'] = result['n_similar_proteins']
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    if suffix:
        summary_file = output_dir / f'class_imbalance_summary_{suffix}.csv'
    else:
        summary_file = output_dir / 'class_imbalance_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary statistics to {summary_file}")
    
    return summary_df


def create_visualizations(
    summary_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = ""
):
    """
    Create visualization plots for class imbalance analysis
    
    Args:
        summary_df: DataFrame with summary statistics
        output_dir: Output directory for saving plots
        suffix: Suffix for plot directory name
    """
    if suffix:
        plots_dir = output_dir / 'plots' / suffix
    else:
        plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both original and optimized dataframes
    # Check if we have optimized columns
    has_optimized = 'optimized_cutoff_high' in summary_df.columns
    if has_optimized:
        # Use optimized cutoffs for display
        cutoff_high_col = 'optimized_cutoff_high'
        cutoff_medium_col = 'optimized_cutoff_medium'
    else:
        cutoff_high_col = 'cutoff_high'
        cutoff_medium_col = 'cutoff_medium'
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Class distribution bar plot (counts)
    fig, ax = plt.subplots(figsize=(14, 8))
    proteins_with_data = summary_df[summary_df['total_datapoints'] > 0]
    
    if not proteins_with_data.empty:
        x_pos = np.arange(len(proteins_with_data))
        width = 0.25
        
        ax.bar(x_pos - width, proteins_with_data['low_count'], width, label='Low', color='#3498db')
        ax.bar(x_pos, proteins_with_data['medium_count'], width, label='Medium', color='#f39c12')
        ax.bar(x_pos + width, proteins_with_data['high_count'], width, label='High', color='#e74c3c')
        
        ax.set_xlabel('Protein', fontsize=12)
        ax.set_ylabel('Number of Datapoints', fontsize=12)
        ax.set_title('Class Distribution by Protein (Counts)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(proteins_with_data['name2_entry'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = plots_dir / 'class_distribution_counts.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created class distribution counts plot: {plot_file}")
    
    # 2. Class distribution percentage plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if not proteins_with_data.empty:
        x_pos = np.arange(len(proteins_with_data))
        width = 0.25
        
        ax.bar(x_pos - width, proteins_with_data['low_pct'], width, label='Low', color='#3498db')
        ax.bar(x_pos, proteins_with_data['medium_pct'], width, label='Medium', color='#f39c12')
        ax.bar(x_pos + width, proteins_with_data['high_pct'], width, label='High', color='#e74c3c')
        
        ax.set_xlabel('Protein', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Class Distribution by Protein (Percentages)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(proteins_with_data['name2_entry'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plot_file = plots_dir / 'class_distribution_percentages.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created class distribution percentages plot: {plot_file}")
    
    # 3. Imbalance ratio distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    imbalance_ratios = proteins_with_data['imbalance_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if not imbalance_ratios.empty:
        ax.hist(imbalance_ratios, bins=30, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax.set_xlabel('Imbalance Ratio (Max/Min)', fontsize=12)
        ax.set_ylabel('Number of Proteins', fontsize=12)
        ax.set_title('Distribution of Class Imbalance Ratios', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = plots_dir / 'imbalance_ratio_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created imbalance ratio distribution plot: {plot_file}")
    
    # 4. Total datapoints per protein
    fig, ax = plt.subplots(figsize=(14, 8))
    
    proteins_sorted = summary_df.sort_values('total_datapoints', ascending=False)
    
    ax.barh(range(len(proteins_sorted)), proteins_sorted['total_datapoints'], color='#2ecc71')
    ax.set_yticks(range(len(proteins_sorted)))
    ax.set_yticklabels(proteins_sorted['name2_entry'])
    ax.set_xlabel('Total Number of Datapoints', fontsize=12)
    ax.set_ylabel('Protein', fontsize=12)
    ax.set_title('Total Bioactivity Datapoints per Protein', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file = plots_dir / 'total_datapoints_per_protein.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created total datapoints plot: {plot_file}")
    
    # 5. Class percentage heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(proteins_with_data) * 0.3)))
    
    if not proteins_with_data.empty:
        heatmap_data = proteins_with_data[['low_pct', 'medium_pct', 'high_pct']].T
        heatmap_data.columns = proteins_with_data['name2_entry']
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_ylabel('Class', fontsize=12)
        ax.set_xlabel('Protein', fontsize=12)
        ax.set_title('Class Distribution Heatmap (Percentages)', fontsize=14, fontweight='bold')
        ax.set_yticklabels(['Low', 'Medium', 'High'], rotation=0)
        
        plt.tight_layout()
        plot_file = plots_dir / 'class_distribution_heatmap.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created class distribution heatmap: {plot_file}")


def create_comparison_visualizations(
    summary_df_original: pd.DataFrame,
    summary_df_optimized: pd.DataFrame,
    output_dir: Path
):
    """
    Create comparison visualizations between original and optimized cutoffs
    
    Args:
        summary_df_original: DataFrame with original cutoff results
        summary_df_optimized: DataFrame with optimized cutoff results
        output_dir: Output directory for saving plots
    """
    plots_dir = output_dir / 'plots' / 'comparison'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge dataframes for comparison
    comparison_df = summary_df_original[['name2_entry', 'uniprot_id', 'imbalance_ratio']].copy()
    comparison_df = comparison_df.rename(columns={'imbalance_ratio': 'original_imbalance_ratio'})
    
    comparison_df = comparison_df.merge(
        summary_df_optimized[['uniprot_id', 'imbalance_ratio', 'improvement_pct']],
        on='uniprot_id',
        how='left'
    )
    comparison_df = comparison_df.rename(columns={'imbalance_ratio': 'optimized_imbalance_ratio'})
    
    # Filter out proteins with no data
    comparison_df = comparison_df[
        (comparison_df['original_imbalance_ratio'] > 0) & 
        (comparison_df['original_imbalance_ratio'] != np.inf) &
        (comparison_df['optimized_imbalance_ratio'] > 0) &
        (comparison_df['optimized_imbalance_ratio'] != np.inf)
    ]
    
    if comparison_df.empty:
        logger.warning("No data available for comparison plots")
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Imbalance ratio comparison (scatter plot)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(
        comparison_df['original_imbalance_ratio'],
        comparison_df['optimized_imbalance_ratio'],
        alpha=0.6,
        s=100,
        color='#3498db'
    )
    
    # Add diagonal line (y=x)
    max_val = max(
        comparison_df['original_imbalance_ratio'].max(),
        comparison_df['optimized_imbalance_ratio'].max()
    )
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No improvement')
    ax.set_xlabel('Original Imbalance Ratio', fontsize=12)
    ax.set_ylabel('Optimized Imbalance Ratio', fontsize=12)
    ax.set_title('Imbalance Ratio: Original vs Optimized', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = plots_dir / 'imbalance_ratio_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created imbalance ratio comparison plot: {plot_file}")
    
    # 2. Improvement percentage bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    comparison_sorted = comparison_df.sort_values('improvement_pct', ascending=True)
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in comparison_sorted['improvement_pct']]
    ax.barh(
        range(len(comparison_sorted)),
        comparison_sorted['improvement_pct'],
        color=colors,
        alpha=0.7
    )
    
    ax.set_yticks(range(len(comparison_sorted)))
    ax.set_yticklabels(comparison_sorted['name2_entry'])
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_ylabel('Protein', fontsize=12)
    ax.set_title('Imbalance Ratio Improvement by Protein', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file = plots_dir / 'improvement_percentage.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created improvement percentage plot: {plot_file}")
    
    # 3. Before/After imbalance ratio distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(
        comparison_df['original_imbalance_ratio'],
        bins=30,
        alpha=0.6,
        label='Original',
        color='#3498db',
        edgecolor='black'
    )
    ax.hist(
        comparison_df['optimized_imbalance_ratio'],
        bins=30,
        alpha=0.6,
        label='Optimized',
        color='#2ecc71',
        edgecolor='black'
    )
    
    ax.set_xlabel('Imbalance Ratio', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Distribution of Imbalance Ratios: Original vs Optimized', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = plots_dir / 'imbalance_ratio_distribution_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created imbalance ratio distribution comparison plot: {plot_file}")
    
    # 4. Cutoff changes visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    cutoff_changes = summary_df_optimized.copy()
    cutoff_changes['medium_change'] = (
        cutoff_changes['optimized_cutoff_medium'] - cutoff_changes['original_cutoff_medium']
    )
    cutoff_changes['high_change'] = (
        cutoff_changes['optimized_cutoff_high'] - cutoff_changes['original_cutoff_high']
    )
    cutoff_changes = cutoff_changes[cutoff_changes['total_datapoints'] > 0]
    cutoff_changes = cutoff_changes.sort_values('medium_change')
    
    # Medium cutoff changes
    colors_medium = ['#e74c3c' if x < 0 else '#2ecc71' if x > 0 else '#95a5a6' 
                     for x in cutoff_changes['medium_change']]
    ax1.barh(range(len(cutoff_changes)), cutoff_changes['medium_change'], color=colors_medium, alpha=0.7)
    ax1.set_yticks(range(len(cutoff_changes)))
    ax1.set_yticklabels(cutoff_changes['name2_entry'])
    ax1.set_xlabel('Change in Medium Cutoff', fontsize=12)
    ax1.set_ylabel('Protein', fontsize=12)
    ax1.set_title('Medium Cutoff Changes (Optimized - Original)', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # High cutoff changes
    cutoff_changes = cutoff_changes.sort_values('high_change')
    colors_high = ['#e74c3c' if x < 0 else '#2ecc71' if x > 0 else '#95a5a6' 
                   for x in cutoff_changes['high_change']]
    ax2.barh(range(len(cutoff_changes)), cutoff_changes['high_change'], color=colors_high, alpha=0.7)
    ax2.set_yticks(range(len(cutoff_changes)))
    ax2.set_yticklabels(cutoff_changes['name2_entry'])
    ax2.set_xlabel('Change in High Cutoff', fontsize=12)
    ax2.set_ylabel('Protein', fontsize=12)
    ax2.set_title('High Cutoff Changes (Optimized - Original)', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file = plots_dir / 'cutoff_changes.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created cutoff changes plot: {plot_file}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main function to run class imbalance analysis
    """
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Determine config file path
    if os.getenv('CONFIG_FILE'):
        config_path = Path(os.getenv('CONFIG_FILE')).expanduser().resolve()
    else:
        # Default: look for config.yaml in project root
        config_path = project_root / "config.yaml"
    
    # Validate config file exists
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.error("Please provide CONFIG_FILE environment variable or place config.yaml in project root")
        return
    
    logger.info(f"Using config file: {config_path}")
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Convert relative paths in config to absolute paths (relative to config file location)
    config_dir = config_path.parent
    
    # Get paths from config
    avoidome_file = config.get('avoidome_file')
    similarity_file = config.get('similarity_file')
    
    if not avoidome_file:
        logger.error("No 'avoidome_file' path found in config.yaml")
        return
    
    if not similarity_file:
        logger.error("No 'similarity_file' path found in config.yaml")
        return
    
    # Convert to absolute paths
    if not Path(avoidome_file).is_absolute():
        avoidome_file = str((config_dir / avoidome_file).resolve())
    else:
        avoidome_file = str(Path(avoidome_file).expanduser().resolve())
    
    if not Path(similarity_file).is_absolute():
        similarity_file = str((config_dir / similarity_file).resolve())
    else:
        similarity_file = str(Path(similarity_file).expanduser().resolve())
    
    # Determine cutoff table path (assume it's in the same directory as avoidome_file)
    avoidome_file_path = Path(avoidome_file)
    cutoff_table_path = avoidome_file_path.parent / 'avoidome_cutoff_table.csv'
    
    # If cutoff table doesn't exist in same directory, try data directory
    if not cutoff_table_path.exists():
        cutoff_table_path = avoidome_file_path.parent / 'data' / 'avoidome_cutoff_table.csv'
    
    # If still doesn't exist, try project root data directory
    if not cutoff_table_path.exists():
        cutoff_table_path = project_root / 'data' / 'avoidome_cutoff_table.csv'
    
    if not cutoff_table_path.exists():
        logger.error(f"Cutoff table not found. Tried:")
        logger.error(f"  - {avoidome_file_path.parent / 'avoidome_cutoff_table.csv'}")
        logger.error(f"  - {avoidome_file_path.parent / 'data' / 'avoidome_cutoff_table.csv'}")
        logger.error(f"  - {project_root / 'data' / 'avoidome_cutoff_table.csv'}")
        return
    
    # Set output directory (use project root / 04_bioactivity_threshold_optimization)
    output_dir = project_root / '04_bioactivity_threshold_optimization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cutoff table: {cutoff_table_path}")
    logger.info(f"Similarity file: {similarity_file}")
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("=" * 80)
    logger.info("Starting Class Imbalance Analysis")
    logger.info("=" * 80)
    
    # Step 1: Load cutoff table
    cutoff_df = load_cutoff_table(str(cutoff_table_path))
    
    # Filter out proteins with insufficient data points for modeling (<30)
    proteins_to_exclude = ['AHR', 'AOX1', 'CHRNA3', 'GSTA1', 'SLCO1B1', 'SULT1A1', 'SLCO2B3']
    initial_count = len(cutoff_df)
    cutoff_df = cutoff_df[~cutoff_df['name2_entry'].isin(proteins_to_exclude)]
    excluded_count = initial_count - len(cutoff_df)
    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} proteins with insufficient data points: {', '.join(proteins_to_exclude)}")
        logger.info(f"Remaining proteins for analysis: {len(cutoff_df)}")
    
    # Step 2: Load similarity summary
    similar_proteins_dict = load_similarity_summary(similarity_file)
    
    # Step 3: Load Papyrus++ data
    papyrus_df = load_papyrus_data()
    
    # Step 4: Analyze each protein - TARGET ONLY
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 1: Target Proteins Only")
    logger.info("=" * 80)
    
    results_target_only = []
    uniprot_ids = cutoff_df['uniprot_id'].tolist()
    
    logger.info(f"\nAnalyzing class distributions for {len(uniprot_ids)} target proteins (target only)...")
    
    for idx, row in cutoff_df.iterrows():
        name2_entry = row['name2_entry']
        uniprot_id = row['uniprot_id']
        cutoff_high = float(row['cutoff_high'])
        cutoff_medium = float(row['cutoff_medium'])
        protein_family = row.get('protein_family', 'N/A')
        
        logger.info(f"\nProcessing {name2_entry} ({uniprot_id})...")
        
        # Filter bioactivity data for this protein only
        protein_data = papyrus_df[papyrus_df['accession'] == uniprot_id].copy()
        
        # Filter for valid data
        protein_data = protein_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        protein_data = protein_data[protein_data['SMILES'] != '']
        
        logger.info(f"  Found {len(protein_data)} bioactivity datapoints (target only)")
        
        # Analyze class distribution
        class_counts = analyze_protein_class_distribution(
            protein_data,
            cutoff_medium,
            cutoff_high
        )
        
        # Calculate imbalance metrics
        metrics = calculate_imbalance_metrics(class_counts)
        
        # Store results
        results_target_only.append({
            'name2_entry': name2_entry,
            'uniprot_id': uniprot_id,
            'protein_family': protein_family,
            'cutoff_high': cutoff_high,
            'cutoff_medium': cutoff_medium,
            'class_counts': class_counts,
            'metrics': metrics,
            'n_similar_proteins': 0
        })
        
        logger.info(f"  Class distribution: Low={class_counts['Low']}, "
                   f"Medium={class_counts['Medium']}, High={class_counts['High']}")
        logger.info(f"  Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    
    # Step 5: Analyze each protein - TARGET + SIMILAR PROTEINS
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 2: Target Proteins + Similar Proteins")
    logger.info("=" * 80)
    
    results_with_similar = []
    
    logger.info(f"\nAnalyzing class distributions for {len(uniprot_ids)} target proteins (with similar proteins)...")
    
    for idx, row in cutoff_df.iterrows():
        name2_entry = row['name2_entry']
        uniprot_id = row['uniprot_id']
        cutoff_high = float(row['cutoff_high'])
        cutoff_medium = float(row['cutoff_medium'])
        protein_family = row.get('protein_family', 'N/A')
        
        logger.info(f"\nProcessing {name2_entry} ({uniprot_id})...")
        
        # Get similar proteins for this target
        similar_proteins = similar_proteins_dict.get(uniprot_id, [])
        
        if not similar_proteins:
            # No similar proteins found, use target only
            logger.info(f"  No similar proteins found, using target only")
            similar_proteins = [uniprot_id]
        else:
            logger.info(f"  Found {len(similar_proteins)} similar proteins (including target)")
        
        # Filter bioactivity data for target + similar proteins
        protein_data = papyrus_df[papyrus_df['accession'].isin(similar_proteins)].copy()
        
        # Filter for valid data
        protein_data = protein_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        protein_data = protein_data[protein_data['SMILES'] != '']
        
        logger.info(f"  Found {len(protein_data)} bioactivity datapoints (target + similar)")
        
        # Analyze class distribution
        class_counts = analyze_protein_class_distribution(
            protein_data,
            cutoff_medium,
            cutoff_high
        )
        
        # Calculate imbalance metrics
        metrics = calculate_imbalance_metrics(class_counts)
        
        # Store results
        results_with_similar.append({
            'name2_entry': name2_entry,
            'uniprot_id': uniprot_id,
            'protein_family': protein_family,
            'cutoff_high': cutoff_high,
            'cutoff_medium': cutoff_medium,
            'class_counts': class_counts,
            'metrics': metrics,
            'n_similar_proteins': len(similar_proteins) - 1  # Exclude target itself
        })
        
        logger.info(f"  Class distribution: Low={class_counts['Low']}, "
                   f"Medium={class_counts['Medium']}, High={class_counts['High']}")
        logger.info(f"  Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    
    # Step 6: Create summary statistics for target only
    logger.info("\n" + "=" * 80)
    logger.info("Creating summary statistics (target only)...")
    summary_df_target = create_summary_statistics(results_target_only, output_dir, "only_target")
    
    # Step 7: Create summary statistics for target + similar
    logger.info("\nCreating summary statistics (target + similar)...")
    summary_df_with_similar = create_summary_statistics(results_with_similar, output_dir, "with_similar")
    
    # Step 8: Create visualizations for target only
    logger.info("\nCreating visualizations (target only)...")
    create_visualizations(summary_df_target, output_dir, "only_target")
    
    # Step 9: Create visualizations for target + similar
    logger.info("\nCreating visualizations (target + similar)...")
    create_visualizations(summary_df_with_similar, output_dir, "with_similar")
    
    # Step 10: Optimize cutoffs using greedy search (target + similar proteins)
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 3: Optimizing Cutoffs (Greedy Search) - Target + Similar Proteins")
    logger.info("=" * 80)
    
    results_optimized = []
    
    logger.info(f"\nOptimizing cutoffs for {len(uniprot_ids)} target proteins...")
    
    for idx, row in cutoff_df.iterrows():
        name2_entry = row['name2_entry']
        uniprot_id = row['uniprot_id']
        original_cutoff_high = float(row['cutoff_high'])
        original_cutoff_medium = float(row['cutoff_medium'])
        protein_family = row.get('protein_family', 'N/A')
        
        logger.info(f"\nOptimizing {name2_entry} ({uniprot_id})...")
        logger.info(f"  Original cutoffs: medium={original_cutoff_medium:.1f}, high={original_cutoff_high:.1f}")
        
        # Get similar proteins for this target
        similar_proteins = similar_proteins_dict.get(uniprot_id, [])
        
        if not similar_proteins:
            similar_proteins = [uniprot_id]
        
        # Filter bioactivity data for target + similar proteins
        protein_data = papyrus_df[papyrus_df['accession'].isin(similar_proteins)].copy()
        
        # Filter for valid data
        protein_data = protein_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        protein_data = protein_data[protein_data['SMILES'] != '']
        
        if len(protein_data) == 0:
            logger.info(f"  No data available, skipping optimization")
            # Store original cutoffs if no data
            class_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Unknown': 0, 'Total': 0}
            metrics = calculate_imbalance_metrics(class_counts)
            results_optimized.append({
                'name2_entry': name2_entry,
                'uniprot_id': uniprot_id,
                'protein_family': protein_family,
                'cutoff_high': original_cutoff_high,
                'cutoff_medium': original_cutoff_medium,
                'optimized_cutoff_high': original_cutoff_high,
                'optimized_cutoff_medium': original_cutoff_medium,
                'class_counts': class_counts,
                'metrics': metrics,
                'n_similar_proteins': len(similar_proteins) - 1,
                'original_imbalance_ratio': float('inf'),
                'optimized_imbalance_ratio': float('inf'),
                'use_2class': False,
                'best_2class_option': None,
                'best_2class_ratio': float('inf'),
                'best_2class_cutoff': None,
                'two_class_improvement_pct': 0.0,
                'lowmedium_vs_high_ratio': float('inf'),
                'low_vs_mediumhigh_ratio': float('inf'),
                'medium_vs_high_ratio': float('inf')
            })
            continue
        
        # Calculate original imbalance ratio
        original_class_counts = analyze_protein_class_distribution(
            protein_data, original_cutoff_medium, original_cutoff_high
        )
        original_metrics = calculate_imbalance_metrics(original_class_counts)
        original_ratio = original_metrics['imbalance_ratio']
        
        logger.info(f"  Original imbalance ratio: {original_ratio:.2f}")
        
        # Run greedy search optimization
        optimized_medium, optimized_high, optimized_ratio = greedy_search_optimal_cutoffs(
            protein_data,
            original_cutoff_medium,
            original_cutoff_high
        )
        
        logger.info(f"  Greedy proposal: medium={optimized_medium:.1f}, high={optimized_high:.1f}, "
                    f"ratio={optimized_ratio:.2f}")
        
        # Decide whether to accept optimized cutoffs
        use_optimized = False
        if np.isfinite(optimized_ratio):
            if not np.isfinite(original_ratio) or optimized_ratio < original_ratio:
                use_optimized = True
        
        if use_optimized:
            final_medium = optimized_medium
            final_high = optimized_high
            final_ratio = optimized_ratio
            final_class_counts = analyze_protein_class_distribution(
                protein_data, final_medium, final_high
            )
            final_metrics = calculate_imbalance_metrics(final_class_counts)
            logger.info(f"  Accepted optimized cutoffs")
        else:
            # Keep original cutoffs if optimization does not improve imbalance
            final_medium = original_cutoff_medium
            final_high = original_cutoff_high
            final_ratio = original_ratio
            final_class_counts = original_class_counts
            final_metrics = original_metrics
            logger.info("  Optimization did not improve imbalance ratio → keeping original cutoffs")
        
        if np.isfinite(original_ratio) and original_ratio > 0:
            improvement_pct = ((original_ratio - final_ratio) / original_ratio) * 100.0
        else:
            improvement_pct = 0.0
        
        logger.info(f"  Final cutoffs: medium={final_medium:.1f}, high={final_high:.1f}")
        logger.info(f"  Final imbalance ratio: {final_ratio:.2f}")
        logger.info(f"  Improvement: {improvement_pct:.1f}%")
        
        # Analyze 2-class options
        logger.info(f"  Analyzing 2-class classification options...")
        two_class_analysis = analyze_2class_options(protein_data, final_medium, final_high)
        best_2class_ratio = two_class_analysis['best_ratio']
        best_2class_option = two_class_analysis['best_option']
        best_2class_cutoff = two_class_analysis['best_cutoff_to_use']
        
        # Determine if should use 2-class
        use_2class = should_use_2class(
            optimized_3class_ratio=final_ratio,
            best_2class_ratio=best_2class_ratio,
            class_counts=final_class_counts,
            improvement_threshold=20.0,  # 20% improvement threshold
            max_2class_ratio=2.5,  # Maximum acceptable 2-class ratio
            min_samples=30  # Minimum samples required
        )
        
        if use_2class:
            if np.isfinite(final_ratio) and final_ratio > 0:
                two_class_improvement = ((final_ratio - best_2class_ratio) / final_ratio) * 100.0
            else:
                two_class_improvement = 100.0 if np.isfinite(best_2class_ratio) else 0.0
            logger.info(f"  → Recommended for 2-class: {best_2class_option} (ratio={best_2class_ratio:.2f}, improvement={two_class_improvement:.1f}%)")
        else:
            two_class_improvement = 0.0
            logger.info(f"  → Keep 3-class (2-class ratio={best_2class_ratio:.2f} doesn't improve significantly)")
        
        # Store results
        results_optimized.append({
            'name2_entry': name2_entry,
            'uniprot_id': uniprot_id,
            'protein_family': protein_family,
            'cutoff_high': original_cutoff_high,
            'cutoff_medium': original_cutoff_medium,
            'optimized_cutoff_high': final_high,
            'optimized_cutoff_medium': final_medium,
            'class_counts': final_class_counts,
            'metrics': final_metrics,
            'n_similar_proteins': len(similar_proteins) - 1,
            'original_imbalance_ratio': original_ratio,
            'optimized_imbalance_ratio': final_ratio,
            'use_2class': use_2class,
            'best_2class_option': best_2class_option,
            'best_2class_ratio': best_2class_ratio,
            'best_2class_cutoff': best_2class_cutoff,
            'two_class_improvement_pct': two_class_improvement,
            'lowmedium_vs_high_ratio': two_class_analysis['lowmedium_vs_high_ratio'],
            'low_vs_mediumhigh_ratio': two_class_analysis['low_vs_mediumhigh_ratio'],
            'medium_vs_high_ratio': two_class_analysis['medium_vs_high_ratio']
        })
        
        logger.info(f"  Final class distribution: Low={final_class_counts['Low']}, "
                   f"Medium={final_class_counts['Medium']}, High={final_class_counts['High']}")
    
    # Step 11: Create summary statistics for optimized cutoffs
    logger.info("\n" + "=" * 80)
    logger.info("Creating summary statistics (optimized cutoffs)...")
    
    # Create summary with both original and optimized cutoffs, including 2-class recommendations
    summary_data_optimized = []
    for result in results_optimized:
        # Calculate improvement percentage
        if (result['original_imbalance_ratio'] > 0 and 
            result['original_imbalance_ratio'] != np.inf and
            result['original_imbalance_ratio'] != float('inf')):
            improvement_pct = ((result['original_imbalance_ratio'] - result['optimized_imbalance_ratio']) / 
                              result['original_imbalance_ratio'] * 100)
        else:
            improvement_pct = 0.0
        
        summary_data_optimized.append({
            'name2_entry': result['name2_entry'],
            'uniprot_id': result['uniprot_id'],
            'protein_family': result.get('protein_family', 'N/A'),
            'original_cutoff_high': result['cutoff_high'],
            'original_cutoff_medium': result['cutoff_medium'],
            'optimized_cutoff_high': result['optimized_cutoff_high'],
            'optimized_cutoff_medium': result['optimized_cutoff_medium'],
            'total_datapoints': result['class_counts']['Total'],
            'low_count': result['class_counts']['Low'],
            'medium_count': result['class_counts']['Medium'],
            'high_count': result['class_counts']['High'],
            'unknown_count': result['class_counts']['Unknown'],
            'low_pct': result['metrics']['low_pct'],
            'medium_pct': result['metrics']['medium_pct'],
            'high_pct': result['metrics']['high_pct'],
            'low_to_high_ratio': result['metrics']['low_to_high_ratio'],
            'medium_to_high_ratio': result['metrics']['medium_to_high_ratio'],
            'imbalance_ratio': result['metrics']['imbalance_ratio'],
            'original_imbalance_ratio': result['original_imbalance_ratio'],
            'optimized_imbalance_ratio': result['optimized_imbalance_ratio'],
            'improvement_pct': improvement_pct,
            'max_class': result['metrics']['max_class'],
            'min_class': result['metrics']['min_class'],
            'n_similar_proteins': result.get('n_similar_proteins', 0),
            # 2-class classification fields
            'use_2class': result.get('use_2class', False),
            'best_2class_option': result.get('best_2class_option', ''),
            'best_2class_ratio': result.get('best_2class_ratio', float('inf')),
            'best_2class_cutoff': result.get('best_2class_cutoff', ''),
            'two_class_improvement_pct': result.get('two_class_improvement_pct', 0.0),
            'lowmedium_vs_high_ratio': result.get('lowmedium_vs_high_ratio', float('inf')),
            'low_vs_mediumhigh_ratio': result.get('low_vs_mediumhigh_ratio', float('inf')),
            'medium_vs_high_ratio': result.get('medium_vs_high_ratio', float('inf'))
        })
    
    summary_df_optimized = pd.DataFrame(summary_data_optimized)
    summary_file_optimized = output_dir / 'class_imbalance_summary_optimized.csv'
    summary_df_optimized.to_csv(summary_file_optimized, index=False)
    logger.info(f"Saved optimized summary statistics to {summary_file_optimized}")
    
    # Step 12: Create visualizations for optimized cutoffs
    logger.info("\nCreating visualizations (optimized cutoffs)...")
    create_visualizations(summary_df_optimized, output_dir, "optimized")
    
    """# Step 13: Create comparison visualizations (original vs optimized)
    logger.info("\nCreating comparison visualizations (original vs optimized)...")
    create_comparison_visualizations(
        summary_df_with_similar, summary_df_optimized, output_dir
    )
    
    # Step 14: Create UMAP plots for proteins with sufficient datapoints
    logger.info("\n" + "=" * 80)
    logger.info("Creating UMAP plots for proteins with >= 50 datapoints")
    logger.info("=" * 80)
    
    # Use optimized results for UMAP plots (they have the best cutoffs)
    # Determine if we should use 2-class or 3-class based on whether medium class has data
    # For now, create 3-class plots (can be extended to support 2-class)
    create_umap_plots_for_proteins(
        results=results_optimized,
        papyrus_df=papyrus_df,
        similar_proteins_dict=similar_proteins_dict,
        cutoff_df=cutoff_df,
        output_dir=output_dir,
        min_datapoints=50,
        use_2class=False  # Set to True for 2-class plots (Low+Medium vs High)
    )"""
    
    # Step 10: Print overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Summary - Target Only")
    logger.info("=" * 80)
    
    total_proteins = len(summary_df_target)
    proteins_with_data = summary_df_target[summary_df_target['total_datapoints'] > 0]
    proteins_without_data = summary_df_target[summary_df_target['total_datapoints'] == 0]
    
    logger.info(f"Total proteins analyzed: {total_proteins}")
    logger.info(f"Proteins with bioactivity data: {len(proteins_with_data)}")
    logger.info(f"Proteins without bioactivity data: {len(proteins_without_data)}")
    
    if not proteins_without_data.empty:
        logger.info(f"\nProteins without data:")
        for _, row in proteins_without_data.iterrows():
            logger.info(f"  - {row['name2_entry']} ({row['uniprot_id']})")
    
    if not proteins_with_data.empty:
        total_datapoints = proteins_with_data['total_datapoints'].sum()
        avg_imbalance = proteins_with_data['imbalance_ratio'].mean()
        max_imbalance = proteins_with_data['imbalance_ratio'].max()
        min_imbalance = proteins_with_data['imbalance_ratio'].min()
        
        logger.info(f"\nTotal bioactivity datapoints: {total_datapoints:,}")
        logger.info(f"Average imbalance ratio: {avg_imbalance:.2f}")
        logger.info(f"Maximum imbalance ratio: {max_imbalance:.2f}")
        logger.info(f"Minimum imbalance ratio: {min_imbalance:.2f}")
        
        # Most imbalanced proteins
        most_imbalanced = proteins_with_data.nlargest(5, 'imbalance_ratio')
        logger.info(f"\nTop 5 most imbalanced proteins:")
        for _, row in most_imbalanced.iterrows():
            logger.info(f"  - {row['name2_entry']}: ratio={row['imbalance_ratio']:.2f}, "
                       f"Low={row['low_count']}, Medium={row['medium_count']}, High={row['high_count']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Overall Summary - Target + Similar Proteins")
    logger.info("=" * 80)
    
    total_proteins = len(summary_df_with_similar)
    proteins_with_data = summary_df_with_similar[summary_df_with_similar['total_datapoints'] > 0]
    proteins_without_data = summary_df_with_similar[summary_df_with_similar['total_datapoints'] == 0]
    
    logger.info(f"Total proteins analyzed: {total_proteins}")
    logger.info(f"Proteins with bioactivity data: {len(proteins_with_data)}")
    logger.info(f"Proteins without bioactivity data: {len(proteins_without_data)}")
    
    if not proteins_without_data.empty:
        logger.info(f"\nProteins without data:")
        for _, row in proteins_without_data.iterrows():
            logger.info(f"  - {row['name2_entry']} ({row['uniprot_id']})")
    
    if not proteins_with_data.empty:
        total_datapoints = proteins_with_data['total_datapoints'].sum()
        avg_imbalance = proteins_with_data['imbalance_ratio'].mean()
        max_imbalance = proteins_with_data['imbalance_ratio'].max()
        min_imbalance = proteins_with_data['imbalance_ratio'].min()
        
        logger.info(f"\nTotal bioactivity datapoints: {total_datapoints:,}")
        logger.info(f"Average imbalance ratio: {avg_imbalance:.2f}")
        logger.info(f"Maximum imbalance ratio: {max_imbalance:.2f}")
        logger.info(f"Minimum imbalance ratio: {min_imbalance:.2f}")
        
        # Most imbalanced proteins
        most_imbalanced = proteins_with_data.nlargest(5, 'imbalance_ratio')
        logger.info(f"\nTop 5 most imbalanced proteins:")
        for _, row in most_imbalanced.iterrows():
            logger.info(f"  - {row['name2_entry']}: ratio={row['imbalance_ratio']:.2f}, "
                       f"Low={row['low_count']}, Medium={row['medium_count']}, High={row['high_count']}")
    
    # Step 14: Print overall summary for optimized cutoffs
    logger.info("\n" + "=" * 80)
    logger.info("Overall Summary - Optimized Cutoffs (Target + Similar Proteins)")
    logger.info("=" * 80)
    
    total_proteins = len(summary_df_optimized)
    proteins_with_data = summary_df_optimized[summary_df_optimized['total_datapoints'] > 0]
    proteins_without_data = summary_df_optimized[summary_df_optimized['total_datapoints'] == 0]
    
    logger.info(f"Total proteins analyzed: {total_proteins}")
    logger.info(f"Proteins with bioactivity data: {len(proteins_with_data)}")
    logger.info(f"Proteins without bioactivity data: {len(proteins_without_data)}")
    
    if not proteins_with_data.empty:
        total_datapoints = proteins_with_data['total_datapoints'].sum()
        avg_imbalance_original = proteins_with_data['original_imbalance_ratio'].mean()
        avg_imbalance_optimized = proteins_with_data['optimized_imbalance_ratio'].mean()
        avg_improvement = proteins_with_data['improvement_pct'].mean()
        max_imbalance = proteins_with_data['optimized_imbalance_ratio'].max()
        min_imbalance = proteins_with_data['optimized_imbalance_ratio'].min()
        
        logger.info(f"\nTotal bioactivity datapoints: {total_datapoints:,}")
        logger.info(f"Average imbalance ratio (original): {avg_imbalance_original:.2f}")
        logger.info(f"Average imbalance ratio (optimized): {avg_imbalance_optimized:.2f}")
        logger.info(f"Average improvement: {avg_improvement:.1f}%")
        logger.info(f"Maximum imbalance ratio (optimized): {max_imbalance:.2f}")
        logger.info(f"Minimum imbalance ratio (optimized): {min_imbalance:.2f}")
        
        # Most improved proteins
        most_improved = proteins_with_data.nlargest(5, 'improvement_pct')
        logger.info(f"\nTop 5 most improved proteins:")
        for _, row in most_improved.iterrows():
            logger.info(f"  - {row['name2_entry']}: "
                       f"original={row['original_imbalance_ratio']:.2f}, "
                       f"optimized={row['optimized_imbalance_ratio']:.2f}, "
                       f"improvement={row['improvement_pct']:.1f}%")
        
        # Most imbalanced proteins (after optimization)
        most_imbalanced = proteins_with_data.nlargest(5, 'optimized_imbalance_ratio')
        logger.info(f"\nTop 5 most imbalanced proteins (after optimization):")
        for _, row in most_imbalanced.iterrows():
            logger.info(f"  - {row['name2_entry']}: ratio={row['optimized_imbalance_ratio']:.2f}, "
                       f"Low={row['low_count']}, Medium={row['medium_count']}, High={row['high_count']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("  - Target only: class_imbalance_summary_only_target.csv")
    logger.info("  - With similar: class_imbalance_summary_with_similar.csv")
    logger.info("  - Optimized: class_imbalance_summary_optimized.csv")
    logger.info("=" * 80)

    # Compact stdout report (2–3 lines)
    try:
        tgt_with_data = (summary_df_target['total_datapoints'] > 0).sum()
        sim_with_data = (summary_df_with_similar['total_datapoints'] > 0).sum()
        opt_with_data = summary_df_optimized[summary_df_optimized['total_datapoints'] > 0]

        print(f"[AQSE Step 4] Proteins with data (target only / with similar): {tgt_with_data} / {sim_with_data}")

        if not opt_with_data.empty:
            avg_orig = opt_with_data['original_imbalance_ratio'].mean()
            avg_opt = opt_with_data['optimized_imbalance_ratio'].mean()
            print(f"[AQSE Step 4] Avg imbalance ratio (original / optimized): {avg_orig:.2f} / {avg_opt:.2f}")

        print(f"[AQSE Step 4] Results directory: {output_dir}")
    except Exception as e:
        logger.debug(f"Failed to print compact stdout report: {e}")


if __name__ == "__main__":
    main()

