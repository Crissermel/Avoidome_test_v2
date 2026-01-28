"""
Main entry point for AQSE 3-Class Classification Workflow.

This module provides the main() function that orchestrates the entire workflow.
"""

import argparse
import logging
import os
from pathlib import Path

from ..utils.config_loader import load_config, resolve_config_path
from ..workflow.workflow_orchestrator import AQSE3CWorkflow
from ..data_loaders import (
    AvoidomeDataLoader,
    SimilarityDataLoader,
    ProteinSequenceLoader,
    BioactivityDataLoader,
    UniqueProteinListGenerator
)


def main():
    """
    Main entry point for AQSE 3-Class Classification Workflow.
    
    Parses command line arguments, loads configuration, and runs the workflow.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AQSE 3-Class Classification Workflow')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config.yaml file. If not provided, looks for config.yaml in project root.')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)
    
    # Resolve config file path
    config_path = resolve_config_path(args.config)
    
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Validate required configuration
    avoidome_file = config.get("avoidome_file")
    if not avoidome_file:
        logger.error("No 'avoidome_file' path found in config.yaml")
        return 1
    
    logger.info(f"Avoidome input file: {avoidome_file}")
    
    # Optional: Generate unique protein list and calculate fingerprints
    # This step can be skipped if fingerprints are already cached
    generate_fingerprints = config.get('generate_fingerprints', True)
    
    if generate_fingerprints:
        logger.info("\n=== Generating Unique Protein List ===")
        protein_list_generator = UniqueProteinListGenerator(config)
        protein_list_df = protein_list_generator.generate_unique_protein_list()
        
        # Save to CSV
        output_file = protein_list_generator.save_protein_list()
        logger.info(f"Saved unique protein list to: {output_file}")
        
        # Calculate Morgan fingerprints filtered by unique proteins
        logger.info("\n=== Calculating Morgan Fingerprints ===")
        fingerprint_loader = BioactivityDataLoader(config)
        fingerprints_info = fingerprint_loader.calculate_morgan_fingerprints(protein_list_df)
        
        # Print dataset statistics
        if fingerprints_info:
            logger.info("\n=== Dataset Statistics ===")
            logger.info(f"Total activities in Papyrus: {fingerprints_info.get('total_activities_in_papyrus', 0):,}")
            logger.info(f"Activities after protein filter: {fingerprints_info.get('activities_after_protein_filter', 0):,}")
            logger.info(f"Unique proteins in filter: {fingerprints_info.get('unique_proteins_in_filter', 0)}")
            logger.info(f"Unique SMILES: {fingerprints_info.get('total_smiles', 0):,}")
            logger.info(f"Valid fingerprints calculated: {fingerprints_info.get('valid_fingerprints', 0):,}")
    
    # Run AQSE 3-Class Workflow
    logger.info("\n" + "="*80)
    logger.info("=== Starting AQSE 3-Class Classification Workflow ===")
    logger.info("="*80)
    
    try:
        workflow = AQSE3CWorkflow(config)
        workflow.process_all_proteins()
        logger.info("\n=== Workflow Completed Successfully ===")
        return 0
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
