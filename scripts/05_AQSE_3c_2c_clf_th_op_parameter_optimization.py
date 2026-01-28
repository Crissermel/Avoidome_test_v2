"""
AQSE 3-Class Classification Workflow - Main Orchestration Script

This script orchestrates the entire AQSE 3-Class Classification Workflow.

The modular structure includes:
- aqse_modelling/data_loaders/    - Data loading components
- aqse_modelling/models/          - Model training components  
- aqse_modelling/utils/           - Utility functions
- aqse_modelling/workflow/        - Workflow orchestration components
- aqse_modelling/reporting/       - Model reporting

Requires: 
    micromamba activate chemprop_env

# Initialize micromamba in your current shell: 
    eval "$(micromamba shell hook --shell bash)"
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add parent directory to path to allow importing aqse_modelling
# This ensures the script works when run from the scripts/ directory
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from aqse_modelling.utils.config_loader import load_config, resolve_config_path
    from aqse_modelling.workflow.workflow_orchestrator import AQSE3CWorkflow
    from aqse_modelling.data_loaders import (
        AvoidomeDataLoader,
        SimilarityDataLoader,
        ProteinSequenceLoader,
        BioactivityDataLoader,
        UniqueProteinListGenerator
    )
except ImportError as e:
    print(f"Error: Could not import aqse_modelling package: {e}")
    print(f"Make sure you're running from the AQSE_v3 directory or that aqse_modelling is in your Python path.")
    sys.exit(1)

# Import table generation modules from aqse_modelling.reporting
try:
    from aqse_modelling.reporting import bioactivity_table_with_similar as create_bioactivity_table_with_similar_module
    from aqse_modelling.reporting import protein_summary_table as create_protein_summary_table_module
    # Try to import the basic bioactivity table module (may not exist yet)
    try:
        from aqse_modelling.reporting import bioactivity_table as create_bioactivity_table_module
    except ImportError:
        create_bioactivity_table_module = None
except ImportError as e:
    # Logging not set up yet, use print
    print(f"Warning: Could not import table generation modules from aqse_modelling.reporting: {e}")
    create_bioactivity_table_module = None
    create_bioactivity_table_with_similar_module = None
    create_protein_summary_table_module = None


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
        # Store config directory for path resolution
        config['_config_dir'] = config_path.parent
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Override output_dir to use step-specific directory (05_model_training)
    # If output_dir is specified in config, use it as base; otherwise use project root
    original_output_dir = config.get('output_dir')
    if original_output_dir:
        # If it's an absolute path, use its parent; if relative, resolve from project root
        original_output_path = Path(original_output_dir)
        if original_output_path.is_absolute():
            base_dir = original_output_path.parent
        else:
            base_dir = project_root
    else:
        base_dir = project_root
    
    # Set step-specific output directory
    step_output_dir = base_dir / '05_model_training'
    config['output_dir'] = str(step_output_dir.resolve())
    step_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model training outputs will be saved to: {step_output_dir}")
    
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
        
        # Generate summary tables after workflow completion
        generate_tables = config.get('generate_summary_tables', True)
        if generate_tables:
            logger.info("\n" + "="*80)
            logger.info("=== Generating Summary Tables ===")
            logger.info("="*80)
            
            try:
                _generate_summary_tables(config, project_root, logger)
                logger.info("\n=== Summary Tables Generated Successfully ===")
            except Exception as e:
                logger.warning(f"Failed to generate summary tables: {e}")
                logger.warning("Workflow completed successfully, but table generation failed.")
        
        return 0
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}", exc_info=True)
        return 1


def _generate_summary_tables(config: dict, project_root: Path, logger: logging.Logger):
    """
    Generate summary tables after workflow completion.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory
        logger: Logger instance
    """
    # Get paths from config (already resolved to absolute paths by config_loader)
    avoidome_file = config.get('avoidome_file')
    similarity_file = config.get('similarity_file')
    activity_thresholds_file = config.get('activity_thresholds_file')
    output_dir = config.get('output_dir')
    
    if not all([avoidome_file, similarity_file, activity_thresholds_file, output_dir]):
        logger.warning("Missing required config paths for table generation. Skipping.")
        return
    
    # Convert to Path objects (paths are already absolute strings from config_loader)
    avoidome_file = Path(avoidome_file)
    similarity_file = Path(similarity_file)
    activity_thresholds_file = Path(activity_thresholds_file)
    output_dir = Path(output_dir)
    
    # Check if required modules are loaded (at least the two we have)
    if not all([create_bioactivity_table_with_similar_module, create_protein_summary_table_module]):
        logger.warning("Required table generation modules not available. Skipping table generation.")
        return
    
    # 1. Generate bioactivity table (target proteins only)
    if create_bioactivity_table_module:
        logger.info("\n--- Generating Bioactivity Table (Target Proteins) ---")
        try:
            thresholds_dict = create_bioactivity_table_module.load_activity_thresholds(str(activity_thresholds_file))
            avoidome_uniprot_ids = create_bioactivity_table_module.get_avoidome_uniprot_ids(str(avoidome_file))
            papyrus_df = create_bioactivity_table_module.load_papyrus_data()
            
            bioactivity_table = create_bioactivity_table_module.create_bioactivity_table(
                papyrus_df,
                thresholds_dict,
                avoidome_uniprot_ids
            )
            
            output_file = output_dir / "bioactivity_table.csv"
            bioactivity_table.to_csv(output_file, index=False)
            logger.info(f"Saved bioactivity table to: {output_file} ({len(bioactivity_table):,} rows)")
        except Exception as e:
            logger.error(f"Error generating bioactivity table: {e}", exc_info=True)
    
    # 2. Generate bioactivity table with similar proteins
    logger.info("\n--- Generating Bioactivity Table (With Similar Proteins) ---")
    try:
        thresholds_dict = create_bioactivity_table_with_similar_module.load_activity_thresholds(str(activity_thresholds_file))
        avoidome_uniprot_ids = create_bioactivity_table_with_similar_module.get_avoidome_uniprot_ids(str(avoidome_file))
        similar_proteins_dict = create_bioactivity_table_with_similar_module.load_similar_proteins(
            str(similarity_file), 
            avoidome_uniprot_ids
        )
        papyrus_df = create_bioactivity_table_with_similar_module.load_papyrus_data()
        
        bioactivity_table_with_similar = create_bioactivity_table_with_similar_module.create_bioactivity_table(
            papyrus_df,
            thresholds_dict,
            avoidome_uniprot_ids,
            similar_proteins_dict
        )
        
        output_file = output_dir / "bioactivity_table_with_similar.csv"
        bioactivity_table_with_similar.to_csv(output_file, index=False)
        logger.info(f"Saved bioactivity table with similar proteins to: {output_file} ({len(bioactivity_table_with_similar):,} rows)")
    except Exception as e:
        logger.error(f"Error generating bioactivity table with similar: {e}", exc_info=True)
    
    # 3. Generate protein summary table
    logger.info("\n--- Generating Protein Summary Table ---")
    try:
        thresholds_df = create_protein_summary_table_module.load_activity_thresholds(str(activity_thresholds_file))
        similar_proteins_dict = create_protein_summary_table_module.load_similar_proteins(str(similarity_file))
        two_class_proteins = create_protein_summary_table_module.get_2class_proteins()
        papyrus_df = create_protein_summary_table_module.load_papyrus_data()
        
        # Try to load metrics from workflow results
        workflow_results_file = output_dir / "workflow_results.csv"
        metrics_dict = None
        if workflow_results_file.exists():
            try:
                # Determine metrics directory based on model type
                model_type = config.get('model_type', 'chemprop')
                metrics_dir = output_dir / f"results_{model_type}" / f"results_{model_type}" / "metrics"
                rf_metrics_dir = output_dir / "results_random_forest" / "results_random_forest" / "metrics"
                metrics_dict = create_protein_summary_table_module.load_model_metrics(
                    str(workflow_results_file),
                    metrics_dir if metrics_dir.exists() else None,
                    rf_metrics_dir if rf_metrics_dir.exists() else None
                )
                logger.info(f"Loaded metrics for {len(metrics_dict)} model configurations")
            except Exception as e:
                logger.debug(f"Could not load model metrics: {e}")
        
        protein_summary_table = create_protein_summary_table_module.create_protein_summary_table(
            thresholds_df,
            similar_proteins_dict,
            two_class_proteins,
            papyrus_df,
            metrics_dict
        )
        
        output_file = output_dir / "protein_summary_table.csv"
        protein_summary_table.to_csv(output_file, index=False)
        logger.info(f"Saved protein summary table to: {output_file} ({len(protein_summary_table)} rows)")
    except Exception as e:
        logger.error(f"Error generating protein summary table: {e}", exc_info=True)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
