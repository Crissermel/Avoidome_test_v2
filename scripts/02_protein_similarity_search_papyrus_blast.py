#!/usr/bin/env python3
"""
AQSE Workflow - Step 2: Protein Similarity Search using Papyrus-scripts with BLAST

This script uses the papyrus-scripts library to find similar proteins in Papyrus
database based on sequence similarity using BLAST and creates similarity matrices.

Hybrid approach:
1. Creates a list of Papyrus proteins (possible queries)
2. Uses their sequences directly from Papyrus
3. Builds a local BLAST database from these sequences
4. Queries the database with target sequences using actual BLAST

"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio.Blast.Applications import NcbiblastpCommandline, NcbimakeblastdbCommandline
from Bio.Blast import NCBIXML
import requests
from io import StringIO
import subprocess
import tempfile
import shutil
import re
import argparse

# Import papyrus-scripts
from papyrus_scripts import PapyrusDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinSimilaritySearchPapyrus:
    """Handles protein similarity search using papyrus-scripts library"""
    
    def __init__(self, input_dir: str, output_dir: str, papyrus_version: str = '05.7'):
        """
        Initialize the similarity search class
        
        Args:
            input_dir: Directory with prepared FASTA files
            output_dir: Output directory for results
            papyrus_version: Papyrus dataset version to use
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.papyrus_version = papyrus_version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.blast_dir = self.output_dir / "blast_db"
        self.blast_dir.mkdir(exist_ok=True)
        
        # Similarity thresholds
        self.thresholds = {
            'high': 70.0,
            'medium': 50.0,
            'low': 30.0
        }
        
        # BLAST database paths
        self.blast_db_path = self.blast_dir / "papyrus_proteins"
        self.blast_db_fasta = self.blast_dir / "papyrus_proteins.fasta"
        self.blast_db_mapping = self.blast_dir / "papyrus_proteins_id_mapping.txt"
        
        # Initialize Papyrus dataset
        logger.info(f"Initializing Papyrus dataset version {papyrus_version}")
        self.papyrus_data = PapyrusDataset(version=papyrus_version, plusplus=False)
        
    def load_avoidome_sequences(self) -> Dict[str, str]:
        """
        Load avoidome protein sequences from FASTA files
        
        Returns:
            Dictionary mapping protein IDs to sequences
        """
        logger.info("Loading avoidome protein sequences")
        
        sequences = {}
        fasta_files = list(self.input_dir.glob("*.fasta"))
        
        for fasta_file in fasta_files:
            logger.info(f"Loading sequences from {fasta_file}")
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_id = record.id
                sequence = str(record.seq)
                sequences[protein_id] = sequence
                logger.info(f"Loaded {protein_id}: {len(sequence)} amino acids")
        
        logger.info(f"Loaded {len(sequences)} avoidome protein sequences")
        return sequences
    
    def get_papyrus_protein_sequences(self) -> pd.DataFrame:
        """
        Get protein sequences from Papyrus database
        
        Returns:
            DataFrame with Papyrus protein sequences
        """
        logger.info("Loading Papyrus protein sequences")
        
        try:
            # Get protein data from Papyrus
            protein_data = self.papyrus_data.proteins()
            logger.info("Loaded protein data from Papyrus")
            
            # Convert to DataFrame if it's a PapyrusProteinSet
            if hasattr(protein_data, 'to_dataframe'):
                protein_df = protein_data.to_dataframe()
            else:
                protein_df = protein_data
            
            # Filter for proteins with sequences (non-null and non-empty)
            protein_df = protein_df.dropna(subset=['Sequence'])
            # Also filter out empty strings
            protein_df = protein_df[protein_df['Sequence'].astype(str).str.strip() != '']
            logger.info(f"Found {len(protein_df)} proteins with sequences")
            
            return protein_df
            
        except Exception as e:
            logger.error(f"Error loading Papyrus protein data: {e}")
            return pd.DataFrame()
    
    def _clean_sequence(self, sequence: str) -> str:
        """
        Clean protein sequence by removing/replacing non-standard amino acids
        
        Standard 20 amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
        Common non-standard: X (unknown), B (D/N), Z (E/Q), J (I/L), O, U, * (stop), - (gap)
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Cleaned sequence with only standard amino acids
        """
        if not sequence:
            return ""
        
        # Define standard amino acids
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        # Replace ambiguous amino acids with common alternatives
        # B (D or N) -> N (more common)
        # Z (E or Q) -> E (more common)
        # J (I or L) -> L (more common)
        replacements = {
            'B': 'N',  # Asparagine or Aspartic acid -> Asparagine
            'Z': 'E',  # Glutamine or Glutamic acid -> Glutamic acid
            'J': 'L',  # Isoleucine or Leucine -> Leucine
            'X': 'A',  # Unknown -> Alanine (most common)
            'O': 'K',  # Pyrrolysine -> Lysine
            'U': 'C',  # Selenocysteine -> Cysteine
            '*': '',   # Stop codon -> remove
        }
        
        cleaned = []
        for aa in sequence.upper():
            if aa in standard_aa:
                cleaned.append(aa)
            elif aa in replacements:
                replacement = replacements[aa]
                if replacement:  # Only add if replacement is not empty
                    cleaned.append(replacement)
            # Skip gaps, spaces, and other non-amino acid characters
        
        return ''.join(cleaned)
    
    def create_blast_database(self, papyrus_proteins: pd.DataFrame, force_rebuild: bool = False) -> bool:
        """
        Create a BLAST database from Papyrus protein sequences
        
        Args:
            papyrus_proteins: DataFrame with Papyrus protein sequences
            force_rebuild: If True, rebuild database even if it exists
            
        Returns:
            True if database was created successfully, False otherwise
        """
        # Check if database already exists
        db_files = [
            self.blast_db_path.with_suffix('.phr'),
            self.blast_db_path.with_suffix('.pin'),
            self.blast_db_path.with_suffix('.psq')
        ]
        
        if not force_rebuild and all(f.exists() for f in db_files) and self.blast_db_mapping.exists():
            logger.info(f"BLAST database already exists at {self.blast_db_path}, skipping rebuild")
            logger.info("Use force_rebuild=True to rebuild the database")
            return True
        
        logger.info("Creating BLAST database from Papyrus sequences")
        
        try:
            # Write sequences to FASTA file and create ID mapping
            logger.info(f"Writing {len(papyrus_proteins)} sequences to FASTA file")
            sequence_index = 0
            id_mapping = {}  # Maps BLAST sequence index to original Papyrus target_id
            
            with open(self.blast_db_fasta, 'w') as f:
                for _, row in papyrus_proteins.iterrows():
                    protein_id = row['target_id']
                    sequence = row['Sequence']
                    
                    # Clean sequence for BLAST
                    cleaned_seq = self._clean_sequence(sequence)
                    if len(cleaned_seq) == 0:
                        logger.warning(f"Skipping {protein_id} - empty sequence after cleaning")
                        continue
                    
                    # Write FASTA entry with index as identifier to ensure proper mapping
                    # BLAST will assign gnl|BL_ORD_ID|X where X is the sequence index (0-based)
                    f.write(f">{sequence_index} {protein_id}\n")
                    # Write sequence in 60-character lines (FASTA format)
                    for i in range(0, len(cleaned_seq), 60):
                        f.write(f"{cleaned_seq[i:i+60]}\n")
                    
                    # Store mapping: BLAST index -> original Papyrus ID
                    id_mapping[sequence_index] = protein_id
                    sequence_index += 1
            
            # Save ID mapping to file
            with open(self.blast_db_mapping, 'w') as f:
                for blast_idx, papyrus_id in id_mapping.items():
                    f.write(f"{blast_idx}\t{papyrus_id}\n")
            
            logger.info(f"FASTA file created: {self.blast_db_fasta}")
            logger.info(f"ID mapping file created: {self.blast_db_mapping} ({len(id_mapping)} sequences)")
            
            # Check if BLAST tools are available
            # Try to find BLAST in common locations
            blast_bin_paths = [
                os.path.expanduser("~/bin/ncbi-blast-2.15.0+/bin"),
                "/usr/local/ncbi-blast/bin",
                "/usr/bin",
                "/usr/local/bin"
            ]
            
            blast_found = False
            blast_path = None
            
            # First check if BLAST is in PATH
            try:
                result = subprocess.run(['makeblastdb', '-version'], 
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    blast_found = True
                    logger.info("BLAST tools found in PATH")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # If not in PATH, check common installation locations
            if not blast_found:
                for path in blast_bin_paths:
                    makeblastdb_path = os.path.join(path, 'makeblastdb')
                    if os.path.exists(makeblastdb_path) and os.access(makeblastdb_path, os.X_OK):
                        blast_path = path
                        blast_found = True
                        # Add to PATH for this process
                        os.environ['PATH'] = f"{path}:{os.environ.get('PATH', '')}"
                        logger.info(f"BLAST tools found at {path}, added to PATH")
                        break
            
            if not blast_found:
                logger.error("BLAST tools not found. Please install BLAST+ (https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)")
                logger.error(f"Checked locations: {blast_bin_paths}")
                return False
            
            # Create BLAST database with parse_seqids to preserve IDs
            logger.info("Building BLAST database...")
            makeblastdb_cline = NcbimakeblastdbCommandline(
                dbtype="prot",
                input_file=str(self.blast_db_fasta),
                out=str(self.blast_db_path),
                parse_seqids=True  # Try to preserve sequence IDs
            )
            
            stdout, stderr = makeblastdb_cline()
            
            if stderr and "Building a new DB" not in stderr:
                logger.warning(f"BLAST database creation warnings: {stderr}")
            
            logger.info(f"BLAST database created successfully: {self.blast_db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating BLAST database: {e}")
            return False
    
    def _load_id_mapping(self) -> Dict[int, str]:
        """
        Load the mapping from BLAST sequence indices to Papyrus target IDs
        
        Returns:
            Dictionary mapping BLAST index to Papyrus target_id
        """
        mapping = {}
        if not self.blast_db_mapping.exists():
            logger.warning(f"ID mapping file not found: {self.blast_db_mapping}")
            return mapping
        
        try:
            with open(self.blast_db_mapping, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            blast_idx = int(parts[0])
                            papyrus_id = parts[1]
                            mapping[blast_idx] = papyrus_id
        except Exception as e:
            logger.error(f"Error loading ID mapping: {e}")
        
        return mapping
    
    def _ensure_blast_in_path(self):
        """Ensure BLAST tools are in PATH"""
        # Check if BLAST is already in PATH
        try:
            subprocess.run(['blastp', '-version'], 
                          capture_output=True, text=True, timeout=5, check=True)
            return  # Already in PATH
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        
        # Try to find BLAST in common locations
        blast_bin_paths = [
            os.path.expanduser("~/bin/ncbi-blast-2.15.0+/bin"),
            "/usr/local/ncbi-blast/bin",
            "/usr/bin",
            "/usr/local/bin"
        ]
        
        for path in blast_bin_paths:
            blastp_path = os.path.join(path, 'blastp')
            if os.path.exists(blastp_path) and os.access(blastp_path, os.X_OK):
                os.environ['PATH'] = f"{path}:{os.environ.get('PATH', '')}"
                logger.debug(f"Added BLAST path to environment: {path}")
                return
        
        logger.warning("BLAST tools not found in PATH or common locations")
    
    def query_blast_database(self, query_id: str, query_sequence: str, 
                            min_identity: float = 0.0) -> List[Dict]:
        """
        Query BLAST database with a target sequence
        
        Args:
            query_id: ID of the query protein
            query_sequence: Protein sequence to search
            min_identity: Minimum percent identity threshold
            
        Returns:
            List of dictionaries with BLAST hit information
        """
        hits = []
        
        # Ensure BLAST is in PATH
        self._ensure_blast_in_path()
        
        # Load ID mapping
        id_mapping = self._load_id_mapping()
        
        try:
            # Clean query sequence
            cleaned_query = self._clean_sequence(query_sequence)
            if len(cleaned_query) == 0:
                logger.warning(f"Query sequence for {query_id} is empty after cleaning")
                return hits
            
            # Create temporary FASTA file for query
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_query:
                tmp_query.write(f">{query_id}\n")
                for i in range(0, len(cleaned_query), 60):
                    tmp_query.write(f"{cleaned_query[i:i+60]}\n")
                tmp_query_path = tmp_query.name
            
            # Create temporary output file for BLAST results
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_output:
                tmp_output_path = tmp_output.name
            
            try:
                # Run BLAST
                blastp_cline = NcbiblastpCommandline(
                    query=tmp_query_path,
                    db=str(self.blast_db_path),
                    out=tmp_output_path,
                    outfmt=5,  # XML format
                    evalue=10.0,  # E-value threshold
                    max_target_seqs=10000,  # Maximum number of hits
                    num_threads=1
                )
                
                stdout, stderr = blastp_cline()
                
                if stderr:
                    logger.warning(f"BLAST warnings for {query_id}: {stderr}")
                
                # Parse BLAST results
                with open(tmp_output_path, 'r') as result_handle:
                    blast_records = NCBIXML.parse(result_handle)
                    
                    for blast_record in blast_records:
                        for alignment in blast_record.alignments:
                            for hsp in alignment.hsps:
                                # Calculate percent identity
                                percent_identity = (hsp.identities / hsp.align_length) * 100
                                
                                # Filter by minimum identity
                                if percent_identity >= min_identity:
                                    # Extract protein ID from alignment title
                                    # BLAST returns "gnl|BL_ORD_ID|X" where X is the sequence index
                                    # We need to extract X and map it to the original Papyrus ID
                                    papyrus_id = "unknown"
                                    
                                    if alignment.title:
                                        title = alignment.title
                                        # Try to extract BLAST sequence index from title
                                        # Format: "gnl|BL_ORD_ID|X description" or just "gnl|BL_ORD_ID|X"
                                        if "BL_ORD_ID" in title:
                                            # Extract the number after BL_ORD_ID|
                                            match = re.search(r'BL_ORD_ID\|(\d+)', title)
                                            if match:
                                                blast_idx = int(match.group(1))
                                                # Map to original Papyrus ID
                                                papyrus_id = id_mapping.get(blast_idx, f"unknown_{blast_idx}")
                                        else:
                                            # If parse_seqids worked, title might be the original ID
                                            # Try to get the first part before space
                                            first_part = title.split()[0].lstrip('>')
                                            # Check if it's a numeric index (needs mapping)
                                            if first_part.isdigit():
                                                blast_idx = int(first_part)
                                                papyrus_id = id_mapping.get(blast_idx, f"unknown_{blast_idx}")
                                            else:
                                                # Already a protein ID (e.g., "P05177_WT")
                                                papyrus_id = first_part
                                    else:
                                        # Fallback: try to get from hit_id if available
                                        if hasattr(alignment, 'hit_id') and alignment.hit_id:
                                            # hit_id format might be "gnl|BL_ORD_ID|X"
                                            match = re.search(r'BL_ORD_ID\|(\d+)', str(alignment.hit_id))
                                            if match:
                                                blast_idx = int(match.group(1))
                                                papyrus_id = id_mapping.get(blast_idx, f"unknown_{blast_idx}")
                                    
                                    hits.append({
                                        'papyrus_id': papyrus_id,
                                        'similarity': percent_identity,
                                        'evalue': hsp.expect,
                                        'bitscore': hsp.bits,
                                        'query_start': hsp.query_start,
                                        'query_end': hsp.query_end,
                                        'subject_start': hsp.sbjct_start,
                                        'subject_end': hsp.sbjct_end,
                                        'align_length': hsp.align_length,
                                        'identities': hsp.identities,
                                        'positives': hsp.positives
                                    })
                
                # Sort by similarity (descending)
                hits.sort(key=lambda x: x['similarity'], reverse=True)
                
            finally:
                # Clean up temporary files
                if os.path.exists(tmp_query_path):
                    os.unlink(tmp_query_path)
                if os.path.exists(tmp_output_path):
                    os.unlink(tmp_output_path)
                    
        except Exception as e:
            logger.error(f"Error querying BLAST database for {query_id}: {e}")
        
        return hits
    
    def find_similar_proteins(self, avoidome_sequences: Dict[str, str], 
                            papyrus_proteins: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Find similar proteins in Papyrus for each avoidome protein using BLAST
        
        Args:
            avoidome_sequences: Dictionary of avoidome protein sequences
            papyrus_proteins: DataFrame of Papyrus proteins (for metadata lookup)
            
        Returns:
            Dictionary of similar proteins by avoidome protein
        """
        logger.info("Finding similar proteins in Papyrus database using BLAST")
        
        # Create a lookup dictionary for Papyrus protein metadata
        papyrus_metadata = {}
        for _, row in papyrus_proteins.iterrows():
            papyrus_metadata[row['target_id']] = {
                'organism': row.get('Organism', 'Unknown'),
                'protein_class': row.get('Classification', 'Unknown'),
                'sequence': row.get('Sequence', '')
            }
        
        similar_proteins = {}
        min_threshold = min(self.thresholds.values())
        
        for avoidome_id, avoidome_seq in avoidome_sequences.items():
            logger.info(f"BLAST searching for proteins similar to {avoidome_id}")
            
            # Query BLAST database
            hits = self.query_blast_database(avoidome_id, avoidome_seq, min_identity=min_threshold)
            
            similar_proteins[avoidome_id] = []
            
            for hit in hits:
                papyrus_id = hit['papyrus_id']
                
                # Get metadata from lookup
                metadata = papyrus_metadata.get(papyrus_id, {
                    'organism': 'Unknown',
                    'protein_class': 'Unknown',
                    'sequence': ''
                })
                
                similar_proteins[avoidome_id].append({
                        'papyrus_id': papyrus_id,
                    'similarity': hit['similarity'],
                    'evalue': hit['evalue'],
                    'bitscore': hit['bitscore'],
                    'organism': metadata['organism'],
                    'protein_class': metadata['protein_class'],
                    'sequence': metadata['sequence']
                })
            
            logger.info(f"Found {len(similar_proteins[avoidome_id])} similar proteins for {avoidome_id}")
        
        return similar_proteins
    
    def create_summary_statistics(self, similar_proteins: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Create summary statistics for similarity search
        
        Args:
            similar_proteins: Dictionary of similar proteins
            
        Returns:
            DataFrame with summary statistics
        """
        logger.info("Creating summary statistics")
        
        summary_data = []
        
        for avoidome_id, similar_list in similar_proteins.items():
            for threshold_name, threshold_value in self.thresholds.items():
                # Count proteins above threshold
                above_threshold = [p for p in similar_list if p['similarity'] >= threshold_value]
                
                # Get top 10 most similar
                top_similar = sorted(above_threshold, key=lambda x: x['similarity'], reverse=True)#[:10]
                # Extract UniProt ID from target_id (remove _WT suffix if present)
                # Load ID mapping for fallback lookup if needed
                id_mapping = self._load_id_mapping()
                
                similar_names = []
                for p in top_similar:
                    papyrus_id = p['papyrus_id']
                    uniprot_id = None
                    
                    # If it's in format "P05177_WT", extract just "P05177"
                    if papyrus_id.endswith('_WT'):
                        uniprot_id = papyrus_id[:-3]  # Remove _WT suffix
                    # If it's already a UniProt ID format (P/Q/O followed by 5 alphanumeric)
                    elif re.match(r'^[OPQ][0-9A-Z]{5}$', papyrus_id):
                        uniprot_id = papyrus_id
                    # If it's numeric (mapping failed earlier), try to look it up now
                    elif papyrus_id.isdigit():
                        blast_idx = int(papyrus_id)
                        mapped_id = id_mapping.get(blast_idx)
                        if mapped_id:
                            # Remove _WT suffix if present
                            if mapped_id.endswith('_WT'):
                                uniprot_id = mapped_id[:-3]
                            else:
                                uniprot_id = mapped_id
                        else:
                            logger.warning(f"Failed to map BLAST index {blast_idx} to protein ID")
                            uniprot_id = papyrus_id  # Keep numeric as fallback
                    elif papyrus_id.startswith('unknown_'):
                        # Extract the index from "unknown_X"
                        try:
                            blast_idx = int(papyrus_id.replace('unknown_', ''))
                            mapped_id = id_mapping.get(blast_idx)
                            if mapped_id:
                                if mapped_id.endswith('_WT'):
                                    uniprot_id = mapped_id[:-3]
                                else:
                                    uniprot_id = mapped_id
                            else:
                                logger.warning(f"Failed to map BLAST index {blast_idx}")
                                uniprot_id = papyrus_id
                        except ValueError:
                            uniprot_id = papyrus_id
                    else:
                        # Try to extract UniProt ID from the string
                        uniprot_match = re.search(r'([OPQ][0-9A-Z]{5})', papyrus_id)
                        if uniprot_match:
                            uniprot_id = uniprot_match.group(1)
                        else:
                            # Last resort: use as is
                            uniprot_id = papyrus_id
                    
                    similar_names.append(f"{uniprot_id} ({p['similarity']:.1f}%)")
                
                summary_data.append({
                    'query_protein': avoidome_id,
                    'threshold': threshold_name,
                    'threshold_value': threshold_value,
                    'num_similar_proteins': len(above_threshold),
                    'max_similarity': max([p['similarity'] for p in above_threshold]) if above_threshold else 0,
                    'similar_proteins': ', '.join(similar_names)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / "similarity_search_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved summary statistics to {summary_file}")
        return summary_df
    
    def create_visualizations(self, summary_df: pd.DataFrame):
        """
        Create visualizations for similarity search results
        
        Args:
            summary_df: Summary statistics DataFrame
        """
        logger.info("Creating visualizations")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Number of similar proteins per avoidome protein
        if not summary_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Pivot for plotting
            pivot_df = summary_df.pivot(index='query_protein', columns='threshold', values='num_similar_proteins')
            pivot_df.plot(kind='bar', ax=ax)
            
            ax.set_title('Number of Similar Proteins per Avoidome Protein')
            ax.set_xlabel('Avoidome Protein')
            ax.set_ylabel('Number of Similar Proteins')
            ax.legend(title='Threshold')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'similar_proteins_count.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def run_similarity_search(self) -> Dict[str, any]:
        """
        Run the complete similarity search workflow using Papyrus and BLAST
        
        Hybrid approach:
        1. Creates a list of Papyrus proteins (possible queries)
        2. Uses their sequences directly from Papyrus
        3. Builds a local BLAST database from these sequences
        4. Queries the database with target sequences using actual BLAST
        
        Returns:
            Dictionary with results
        """
        logger.info("Starting AQSE protein similarity search with Papyrus and BLAST")
        
        # Load avoidome sequences (targets)
        avoidome_sequences = self.load_avoidome_sequences()
        if not avoidome_sequences:
            logger.error("No avoidome sequences loaded")
            return {}
        
        # Get Papyrus protein sequences (possible queries)
        papyrus_proteins = self.get_papyrus_protein_sequences()
        if papyrus_proteins.empty:
            logger.error("No Papyrus protein sequences loaded")
            return {}
        
        # Create BLAST database from Papyrus sequences
        logger.info("Step 1: Creating BLAST database from Papyrus sequences")
        if not self.create_blast_database(papyrus_proteins):
            logger.error("Failed to create BLAST database")
            return {}
        
        # Find similar proteins using BLAST
        logger.info("Step 2: Querying BLAST database with target sequences")
        similar_proteins = self.find_similar_proteins(avoidome_sequences, papyrus_proteins)
        
        # Create summary statistics
        logger.info("Step 3: Creating summary statistics")
        summary_df = self.create_summary_statistics(similar_proteins)
        
        # Create visualizations
        logger.info("Step 4: Creating visualizations")
        self.create_visualizations(summary_df)
        
        logger.info("Protein similarity search completed successfully")
        
        return {
            'avoidome_sequences': avoidome_sequences,
            'papyrus_proteins': papyrus_proteins,
            'similar_proteins': similar_proteins,
            'summary_df': summary_df
        }

def main():
    """Main function to run protein similarity search"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AQSE Workflow - Step 2: Protein Similarity Search')
    parser.add_argument('--input-dir', type=str,
                       default=None,
                       help='Directory with prepared FASTA files. If not provided, uses script directory.')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Output directory for similarity search results. If not provided, uses script_dir/02_similarity_search.')
    parser.add_argument('--papyrus-version', type=str,
                       default='05.7',
                       help='Papyrus dataset version to use (default: 05.7)')
    args = parser.parse_args()
    
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Set up paths - use command line args, environment variables, or defaults
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
    elif os.getenv('INPUT_DIR'):
        input_dir = Path(os.getenv('INPUT_DIR')).expanduser().resolve()
    else:
        # Default: use 01_input_preparation directory (where fasta_files should be from Step 1)
        input_dir = project_root / "01_input_preparation"
    
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif os.getenv('OUTPUT_DIR'):
        output_dir = Path(os.getenv('OUTPUT_DIR')).expanduser().resolve()
    else:
        # Default: create 02_similarity_search in project root
        output_dir = project_root / "02_similarity_search"
    
    # Validate input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Check for fasta_files directory
    fasta_dir = input_dir / "fasta_files"
    if not fasta_dir.exists():
        logger.warning(f"FASTA files directory not found: {fasta_dir}")
        logger.info(f"Looking for FASTA files in: {input_dir}")
    
    logger.info(f"Using input directory: {input_dir}")
    logger.info(f"Using output directory: {output_dir}")
    
    # Initialize similarity search class
    searcher = ProteinSimilaritySearchPapyrus(str(input_dir), str(output_dir), papyrus_version=args.papyrus_version)
    
    # Run similarity search
    results = searcher.run_similarity_search()
    
    if results:
        print("\n" + "="*60)
        print("AQSE PROTEIN SIMILARITY SEARCH COMPLETED")
        print("="*60)
        print(f"Avoidome sequences: {len(results['avoidome_sequences'])}")
        print(f"Papyrus proteins: {len(results['papyrus_proteins'])}")
        print(f"Summary statistics: {len(results['summary_df'])} rows")
        print("="*60)
    else:
        print("Similarity search failed - check logs for details")

if __name__ == "__main__":
    main()