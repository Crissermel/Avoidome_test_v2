#!/usr/bin/env python3
"""
AQSE Workflow - Step 1: Input Preparation

This script prepares the input data for the Avoidome QSAR Similarity Expansion (AQSE) workflow.
It loads avoidome protein sequences and prepares them for BLAST similarity search against Papyrus.

"""

import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import requests
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
            avoidome_file: Path to avoidome_prot_list.csv
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
        
    def load_avoidome_proteins(self) -> pd.DataFrame:
        """
        Load and clean avoidome protein list
        
        Returns:
            DataFrame with avoidome proteins
        """
        logger.info(f"Loading avoidome proteins from {self.avoidome_file}")
        
        try:
            df = pd.read_csv(self.avoidome_file)
            logger.info(f"Loaded {len(df)} avoidome proteins")
            
            # Clean the data
            df = df.dropna(subset=['UniProt ID'])
            df = df[df['UniProt ID'] != 'no info']
            
            # Remove duplicates based on UniProt ID
            df = df.drop_duplicates(subset=['UniProt ID'])
            
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
    
    def create_fasta_files(self, df: pd.DataFrame, sequences: Dict[str, str]) -> List[str]:
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
        
        # Create individual FASTA files for each protein
        for _, row in df.iterrows():
            uniprot_id = row['UniProt ID']
            protein_name = row['Name']
            
            if uniprot_id in sequences:
                # Create FASTA record
                seq_record = SeqRecord(
                    Seq(sequences[uniprot_id]),
                    id=uniprot_id,
                    description=f"{protein_name} - {row.get('Alternative Names', '')}"
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
            for _, row in df.iterrows():
                uniprot_id = row['UniProt ID']
                protein_name = row['Name']
                
                if uniprot_id in sequences:
                    seq_record = SeqRecord(
                        Seq(sequences[uniprot_id]),
                        id=uniprot_id,
                        description=f"{protein_name} - {row.get('Alternative Names', '')}"
                    )
                    SeqIO.write(seq_record, f, 'fasta')
        
        fasta_files.append(str(combined_fasta))
        logger.info(f"Created combined FASTA file: {combined_fasta}")
        
        return fasta_files
    
    def save_summary(self, df: pd.DataFrame, sequences: Dict[str, str], failed_ids: List[str]) -> str:
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
                protein_name = df[df['UniProt ID'] == uniprot_id]['Name'].iloc[0]
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
        uniprot_ids = df['UniProt ID'].tolist()
        
        # Fetch sequences
        sequences, failed_ids = self.fetch_protein_sequences(uniprot_ids)
        
        # Create FASTA files
        fasta_files = self.create_fasta_files(df, sequences)
        
        # Save summary
        summary_file = self.save_summary(df, sequences, failed_ids)
        
        # Save sequences to CSV for easy access
        sequences_df = pd.DataFrame([
            {
                'uniprot_id': uniprot_id,
                'protein_name': df[df['UniProt ID'] == uniprot_id]['Name'].iloc[0],
                'sequence': seq,
                'sequence_length': len(seq)
            }
            for uniprot_id, seq in sequences.items()
        ])
        
        sequences_csv = self.output_dir / "avoidome_sequences.csv"
        sequences_df.to_csv(sequences_csv, index=False)
        
        logger.info("Input preparation completed successfully")
        
        return {
            'sequences_csv': str(sequences_csv),
            'summary_file': summary_file,
            'fasta_files': fasta_files,
            'failed_ids': failed_ids
        }

def main():
    """Main function to run input preparation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AQSE Workflow - Step 1: Input Preparation')
    parser.add_argument('--avoidome-file', type=str, 
                       default=None,
                       help='Path to avoidome_prot_list.csv file. If not provided, looks for avoidome_prot_list.csv in script directory.')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Output directory for prepared files. If not provided, uses script directory.')
    args = parser.parse_args()
    
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Set up paths - use command line args, environment variables, or defaults
    if args.avoidome_file:
        avoidome_file = Path(args.avoidome_file).expanduser().resolve()
    elif os.getenv('AVOIDOME_FILE'):
        avoidome_file = Path(os.getenv('AVOIDOME_FILE')).expanduser().resolve()
    else:
        # Default: look for avoidome_prot_list.csv in data directory
        avoidome_file = project_root / "data" / "avoidome_prot_list.csv"
    
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif os.getenv('OUTPUT_DIR'):
        output_dir = Path(os.getenv('OUTPUT_DIR')).expanduser().resolve()
    else:
        # Default: create 01_input_preparation in project root
        output_dir = project_root / "01_input_preparation"
    
    # Validate input file exists
    if not avoidome_file.exists():
        raise FileNotFoundError(f"Avoidome file not found: {avoidome_file}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
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