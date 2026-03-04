#!/usr/bin/env python3
"""
AQSE Workflow - Step 1: Input Preparation

This script prepares the input data for the Avoidome QSAR Similarity Expansion (AQSE) workflow.
It loads avoidome protein sequences and prepares them for BLAST similarity search against Papyrus.

"""

import polars as pl
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import requests
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AvoidomeInputPreparation:
    """Handles input preparation for AQSE workflow"""
    
    def __init__(self, avoidome_file: str, output_dir: str):
        """
        Initialize the input preparation class
        
        Args:
            avoidome_file: Path to input_clean.csv
            output_dir: Output directory for prepared files
        """
        self.avoidome_file = avoidome_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.fasta_dir = self.output_dir / "fasta_files"
        self.fasta_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def load_avoidome_proteins(self) -> pl.DataFrame:
        """
        Load and clean avoidome protein list from input_clean.csv.
        Expected columns: uniprot_id, protein_name, alternative_names (optional).
        ChEMBL_target is optional; only uniprot_id is required.
        
        Returns:
            DataFrame with uniprot_id, protein_name, alternative_names
        """
        logger.info(f"Loading avoidome proteins from {self.avoidome_file}")
        
        try:
            df = pl.read_csv(self.avoidome_file)
            rename_dict = {col: col.strip() for col in df.columns}
            df = df.rename(rename_dict)
            if 'uniprot_id' not in df.columns:
                raise ValueError("input_clean.csv must have a 'uniprot_id' column.")
            logger.info(f"Loaded {len(df)} avoidome proteins")
            
            name_col = 'protein_name' if 'protein_name' in df.columns else 'gene_name'
            alt_col = 'alternative_names' if 'alternative_names' in df.columns else None
            select_exprs = [
                pl.col('uniprot_id'),
                pl.col(name_col).alias('protein_name'),
            ]
            if alt_col and alt_col in df.columns:
                select_exprs.append(pl.col(alt_col).fill_null('').alias('alternative_names'))
            else:
                select_exprs.append(pl.lit('').alias('alternative_names'))
            df = df.select(select_exprs)
            
            df = df.drop_nulls(subset=['uniprot_id'])
            df = df.filter(pl.col('uniprot_id').cast(pl.Utf8) != '')
            df = df.filter(pl.col('uniprot_id').cast(pl.Utf8) != 'no info')
            df = df.unique(subset=['uniprot_id'])
            
            logger.info(f"After cleaning: {len(df)} unique proteins with UniProt IDs")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading avoidome proteins: {e}")
            raise
    
    def fetch_protein_sequences(self, uniprot_ids: List[str]) -> Dict[str, str]:
        """
        Fetch protein sequences from UniProt
        
        Args:
            uniprot_ids: List of UniProt IDs
            
        Returns:
            Dictionary mapping UniProt ID to sequence
        """
        logger.info(f"Fetching sequences for {len(uniprot_ids)} proteins")
        
        sequences = {}
        failed_ids = []
        
        for i, uniprot_id in enumerate(uniprot_ids):
            try:
                # UniProt REST API
                url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Parse FASTA
                    fasta_text = response.text
                    if fasta_text.startswith('>'):
                        # Extract sequence
                        lines = fasta_text.strip().split('\n')
                        sequence = ''.join(lines[1:])
                        sequences[uniprot_id] = sequence
                        logger.info(f"Fetched sequence for {uniprot_id} ({len(sequence)} aa)")
                    else:
                        logger.warning(f"No sequence found for {uniprot_id}")
                        failed_ids.append(uniprot_id)
                else:
                    logger.warning(f"Failed to fetch {uniprot_id}: HTTP {response.status_code}")
                    failed_ids.append(uniprot_id)
                    
            except Exception as e:
                logger.warning(f"Error fetching {uniprot_id}: {e}")
                failed_ids.append(uniprot_id)
            
            # Add delay to avoid overwhelming UniProt
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)
        
        logger.info(f"Successfully fetched {len(sequences)} sequences")
        logger.info(f"Failed to fetch {len(failed_ids)} sequences: {failed_ids}")
        
        return sequences, failed_ids
    
    def create_fasta_files(self, df: pl.DataFrame, sequences: Dict[str, str]) -> List[str]:
        """
        Create FASTA files for BLAST search
        
        Args:
            df: Avoidome proteins DataFrame
            sequences: Dictionary of UniProt ID to sequence
            
        Returns:
            List of created FASTA file paths
        """
        logger.info("Creating FASTA files for BLAST search")
        
        fasta_files = []
        
        # Create individual FASTA files for each protein (df has uniprot_id, protein_name, alternative_names)
        for row in df.iter_rows(named=True):
            uniprot_id = row['uniprot_id']
            protein_name = row['protein_name']
            
            if uniprot_id in sequences:
                # Create FASTA record
                seq_record = SeqRecord(
                    Seq(sequences[uniprot_id]),
                    id=uniprot_id,
                    description=f"{protein_name} - {row.get('alternative_names', '')}"
                )
                
                # Write to file
                fasta_file = self.fasta_dir / f"{uniprot_id}_{protein_name}.fasta"
                with open(fasta_file, 'w') as f:
                    SeqIO.write(seq_record, f, 'fasta')
                
                fasta_files.append(str(fasta_file))
                logger.info(f"Created FASTA file: {fasta_file}")
        
        # Create combined FASTA file for all proteins
        combined_fasta = self.output_dir / "avoidome_proteins_combined.fasta"
        with open(combined_fasta, 'w') as f:
            for row in df.iter_rows(named=True):
                uniprot_id = row['uniprot_id']
                protein_name = row['protein_name']
                
                if uniprot_id in sequences:
                    seq_record = SeqRecord(
                        Seq(sequences[uniprot_id]),
                        id=uniprot_id,
                        description=f"{protein_name} - {row.get('alternative_names', '')}"
                    )
                    SeqIO.write(seq_record, f, 'fasta')
        
        fasta_files.append(str(combined_fasta))
        logger.info(f"Created combined FASTA file: {combined_fasta}")
        
        return fasta_files
    
    def save_summary(self, df: pl.DataFrame, sequences: Dict[str, str], failed_ids: List[str]) -> str:
        """
        Save input preparation summary
        
        Args:
            df: Avoidome proteins DataFrame
            sequences: Dictionary of sequences
            failed_ids: List of failed UniProt IDs
            
        Returns:
            Path to summary file
        """
        summary_file = self.output_dir / "input_preparation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AQSE Input Preparation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total avoidome proteins: {len(df)}\n")
            f.write(f"Proteins with sequences: {len(sequences)}\n")
            f.write(f"Failed to fetch: {len(failed_ids)}\n\n")
            
            f.write("Proteins with sequences:\n")
            f.write("-" * 30 + "\n")
            for uniprot_id, seq in sequences.items():
                protein_row = df.filter(pl.col('uniprot_id') == uniprot_id)
                if len(protein_row) > 0:
                    protein_name = protein_row['protein_name'][0]
                    f.write(f"{uniprot_id} ({protein_name}): {len(seq)} aa\n")
            
            if failed_ids:
                f.write(f"\nFailed to fetch sequences:\n")
                f.write("-" * 30 + "\n")
                for uniprot_id in failed_ids:
                    f.write(f"{uniprot_id}\n")
        
        logger.info(f"Saved summary to: {summary_file}")
        return str(summary_file)
    
    def run_preparation(self) -> Dict[str, str]:
        """
        Run the complete input preparation workflow
        
        Returns:
            Dictionary with output file paths
        """
        logger.info("Starting AQSE input preparation")
        
        # Load avoidome proteins
        df = self.load_avoidome_proteins()
        
        # Get UniProt IDs
        uniprot_ids = df['uniprot_id'].to_list()
        
        # Fetch sequences
        sequences, failed_ids = self.fetch_protein_sequences(uniprot_ids)
        
        # Create FASTA files
        fasta_files = self.create_fasta_files(df, sequences)
        
        # Save summary
        summary_file = self.save_summary(df, sequences, failed_ids)
        
        # Save sequences to CSV for easy access
        sequences_data = []
        for uniprot_id, seq in sequences.items():
            protein_row = df.filter(pl.col('uniprot_id') == uniprot_id)
            protein_name = protein_row['protein_name'][0] if len(protein_row) > 0 else ''
            sequences_data.append({
                'uniprot_id': uniprot_id,
                'protein_name': protein_name,
                'sequence': seq,
                'sequence_length': len(seq)
            })
        
        sequences_df = pl.DataFrame(sequences_data)
        sequences_csv = self.output_dir / "avoidome_sequences.csv"
        sequences_df.write_csv(sequences_csv)
        
        logger.info("Input preparation completed successfully")
        
        return {
            'sequences_csv': str(sequences_csv),
            'summary_file': summary_file,
            'fasta_files': fasta_files,
            'failed_ids': failed_ids
        }

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to run input preparation"""
    
    # Optional: override config file path via CLI or environment
    parser = argparse.ArgumentParser(description='AQSE Workflow - Step 1: Input Preparation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml. If not provided, uses CONFIG_FILE env or project root config.yaml.')
    args = parser.parse_args()
    
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Resolve config file path
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    elif os.getenv('CONFIG_FILE'):
        config_path = Path(os.getenv('CONFIG_FILE')).expanduser().resolve()
    else:
        config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Using config file: {config_path}")
    config = load_config(str(config_path))
    config_dir = config_path.parent
    
    # Get avoidome_file from config and resolve to absolute path
    avoidome_file = config.get('avoidome_file')
    if not avoidome_file:
        raise ValueError("No 'avoidome_file' found in config.yaml")
    if not Path(avoidome_file).is_absolute():
        avoidome_file = (config_dir / avoidome_file).resolve()
    else:
        avoidome_file = Path(avoidome_file).expanduser().resolve()
    
    # Step 1 output directory: fixed relative to project root
    output_dir = project_root / "01_input_preparation"
    
    # Validate input file exists
    if not avoidome_file.exists():
        raise FileNotFoundError(f"Avoidome file not found: {avoidome_file}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler so logs go to both stderr and a log file in the step output directory
    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to {log_file}")
    
    logger.info(f"Using avoidome file: {avoidome_file}")
    logger.info(f"Using output directory: {output_dir}")
    
    # Initialize preparation class
    preparer = AvoidomeInputPreparation(str(avoidome_file), str(output_dir))
    
    # Run preparation
    results = preparer.run_preparation()
    
    print("\n" + "="*60)
    print("AQSE INPUT PREPARATION COMPLETED")
    print("="*60)
    print(f"Sequences CSV: {results['sequences_csv']}")
    print(f"Summary: {results['summary_file']}")
    print(f"FASTA files: {len(results['fasta_files'])} created")
    print(f"Failed sequences: {len(results['failed_ids'])}")
    print("="*60)

if __name__ == "__main__":
    main()