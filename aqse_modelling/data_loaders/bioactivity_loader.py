"""
Bioactivity data loader.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class BioactivityDataLoader:
    """Handles bioactivity data loading and processing"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up fingerprints directory
        output_dir = Path(config.get("output_dir", "."))
        self.fingerprints_dir = output_dir / "fingerprints"
        self.fingerprints_dir.mkdir(exist_ok=True)
        
        # Set up ESM-C cache directory
        self.esmc_cache_dir = Path(config.get("papyrus_cache_dir", output_dir / "esmc_cache"))
        self.esmc_cache_dir.mkdir(exist_ok=True)
        
        # Get list of proteins that should use standard Papyrus (not ++)
        # Can contain protein names (e.g., "SLCO1B1") or UniProt IDs (e.g., "P59520")
        self.proteins_use_standard_papyrus = config.get("proteins_use_standard_papyrus", [])
        if self.proteins_use_standard_papyrus:
            self.logger.info(f"Proteins using standard Papyrus: {self.proteins_use_standard_papyrus}")
        
        # Store mapping of protein names to UniProt IDs (will be populated if avoidome loader is available)
        self._protein_name_to_uniprot = {}
        
        # Load Papyrus++ dataset ONCE at initialization (default)
        self.logger.info("Loading Papyrus++ dataset...")
        self.logger.info("This may take ~5 minutes on first load")
        try:
            from papyrus_scripts import PapyrusDataset
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            self.papyrus_df = papyrus_data.to_dataframe()
            self.logger.info(f"Loaded {len(self.papyrus_df):,} total activities from Papyrus++")
        except Exception as e:
            self.logger.error(f"Error loading Papyrus++ dataset: {e}")
            self.papyrus_df = pd.DataFrame()
        
        # Load standard Papyrus dataset if needed (lazy loading - only if proteins_use_standard_papyrus is not empty)
        self.papyrus_standard_df = pd.DataFrame()
        self._papyrus_standard_loaded = False
    
    def calculate_morgan_fingerprints(self, unique_proteins: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 0: Calculate and save Morgan fingerprints for all compounds in Papyrus database,
        only for the compounds associated with the unique protein list
        """
        self.logger.info("Calculating Morgan fingerprints for all Papyrus compounds (filtered by unique proteins)")
        
        try:
            # Use the already-loaded Papyrus dataset
            papyrus_df = self.papyrus_df
            
            if papyrus_df.empty:
                self.logger.error("Papyrus dataset not loaded")
                return {}

            # Filter for valid SMILES
            valid_data = papyrus_df.dropna(subset=['SMILES'])
            valid_data = valid_data[valid_data['SMILES'] != '']

            # Filter by unique protein list
            unique_ids = set(unique_proteins['uniprot_id'])
            valid_data = valid_data[valid_data['accession'].isin(unique_ids)]
            self.logger.info(f"{len(valid_data)} activities after filtering by unique proteins")

            # Get unique SMILES
            unique_smiles = valid_data['SMILES'].unique()
            self.logger.info(f"Found {len(unique_smiles)} unique SMILES to process")

            # Calculate Morgan fingerprints
            fingerprints = {}
            valid_count = 0
            
            for i, smiles in enumerate(unique_smiles):
                if i % 10000 == 0:
                    self.logger.info(f"Processing SMILES {i+1}/{len(unique_smiles)}")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        morgan_array = np.array(morgan_fp, dtype=np.float32)
                        fingerprints[smiles] = morgan_array
                        valid_count += 1
                except Exception as e:
                    self.logger.warning(f"Error processing SMILES {smiles}: {e}")
                    continue

            # Save fingerprints as Parquet for efficiency
            fingerprints_file = self.fingerprints_dir / "papyrus_morgan_fingerprints.parquet"
            
            # Convert dict to DataFrame for Parquet
            fps_df = pd.DataFrame({
                'SMILES': list(fingerprints.keys()),
                'fingerprint': [fp.tolist() for fp in fingerprints.values()]  # Convert numpy array to list
            })
            
            # Save as Parquet with compression
            fps_df.to_parquet(fingerprints_file, compression='snappy')
            file_size_mb = fingerprints_file.stat().st_size / (1024 * 1024)

            self.logger.info(f"Bioactivity dataset completed: {valid_count} valid Morgan fingerprints saved to {fingerprints_file} ({file_size_mb:.2f} MB)")
            
            # Get dataset statistics
            total_activities = len(papyrus_df)
            filtered_activities = len(valid_data)
            
            return {
                'total_activities_in_papyrus': total_activities,
                'activities_after_protein_filter': filtered_activities,
                'unique_proteins_in_filter': len(unique_ids),
                'total_smiles': len(unique_smiles),
                'valid_fingerprints': valid_count,
                'fingerprints_file': str(fingerprints_file)
            }

        except Exception as e:
            self.logger.error(f"Error in Step 0: {e}")
            return {}

    def load_morgan_fingerprints(self, unique_proteins: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """Load pre-computed Morgan fingerprints
        
        Note: The fingerprints were already filtered during calculation to only include
        SMILES associated with the unique proteins list.
        """
        # Try Parquet first, fallback to pickle for backward compatibility
        fingerprints_file_parquet = self.fingerprints_dir / "papyrus_morgan_fingerprints.parquet"
        fingerprints_file_pkl = self.fingerprints_dir / "papyrus_morgan_fingerprints.pkl"
        
        try:
            # Try to load from Parquet
            if fingerprints_file_parquet.exists():
                fps_df = pd.read_parquet(fingerprints_file_parquet)
                # Convert back to dict
                fingerprints = {
                    row['SMILES']: np.array(row['fingerprint'], dtype=np.float32)
                    for _, row in fps_df.iterrows()
                }
                self.logger.info(f"Loaded {len(fingerprints)} Morgan fingerprints from Parquet")
                return fingerprints
            
            # Fallback to pickle
            elif fingerprints_file_pkl.exists():
                with open(fingerprints_file_pkl, 'rb') as f:
                    fingerprints = pickle.load(f)
                self.logger.info(f"Loaded {len(fingerprints)} Morgan fingerprints from Pickle (legacy format)")
                return fingerprints
            else:
                self.logger.error(f"Morgan fingerprints file not found (neither .parquet nor .pkl)")
                return {}
        
        except Exception as e:
            self.logger.error(f"Error loading Morgan fingerprints: {e}")
            return {}
    
    def calculate_physicochemical_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Calculate physicochemical descriptors for a single SMILES string.
        
        Uses the local physicochemical_descriptors module from aqse_modelling.utils.
        This ensures AQSE_v3 is self-contained with no external dependencies.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of descriptor names and values
        """
        try:
            from aqse_modelling.utils.physicochemical_descriptors import calculate_physicochemical_descriptors
            
            # Calculate descriptors (without SASA for speed)
            descriptors = calculate_physicochemical_descriptors(
                smiles, 
                include_sasa=False, 
                verbose=False
            )
            
            return descriptors
            
        except ImportError as e:
            self.logger.error(f"Failed to import physicochemical_descriptors from aqse_modelling.utils: {e}")
            self.logger.error("This module should be available in AQSE_v3/aqse_modelling/utils/physicochemical_descriptors.py")
            return {}
        except Exception as e:
            self.logger.error(f"Error calculating physicochemical descriptors for {smiles}: {e}")
            return {}
    
    def _load_standard_papyrus(self):
        """Lazy load standard Papyrus dataset if not already loaded"""
        if self._papyrus_standard_loaded:
            return
        
        if not self.proteins_use_standard_papyrus:
            # No proteins need standard Papyrus, skip loading
            return
        
        self.logger.info("Loading standard Papyrus dataset (not ++)...")
        self.logger.info("This may take ~5 minutes on first load")
        try:
            from papyrus_scripts import PapyrusDataset
            papyrus_data = PapyrusDataset(version='latest', plusplus=False)
            self.papyrus_standard_df = papyrus_data.to_dataframe()
            self._papyrus_standard_loaded = True
            self.logger.info(f"Loaded {len(self.papyrus_standard_df):,} total activities from standard Papyrus")
        except Exception as e:
            self.logger.error(f"Error loading standard Papyrus dataset: {e}")
            self.papyrus_standard_df = pd.DataFrame()
            self._papyrus_standard_loaded = True  # Mark as loaded to avoid retrying
    
    def _build_protein_name_mapping(self):
        """Build mapping from protein names to UniProt IDs using avoidome loader if available"""
        if self._protein_name_to_uniprot:
            return  # Already built
        
        try:
            # Try to get avoidome loader from config or create one
            from .avoidome_loader import AvoidomeDataLoader
            avoidome_loader = AvoidomeDataLoader(self.config)
            targets = avoidome_loader.load_avoidome_targets()
            
            # Build mapping: name -> uniprot_id
            for target in targets:
                name = target.get('Name_2', '')
                uniprot_id = target.get('UniProt ID', '')
                if name and uniprot_id:
                    self._protein_name_to_uniprot[name.upper()] = uniprot_id.upper()
                    # Also map UniProt ID to itself for direct lookups
                    self._protein_name_to_uniprot[uniprot_id.upper()] = uniprot_id.upper()
            
            if self._protein_name_to_uniprot:
                self.logger.debug(f"Built protein name mapping for {len(self._protein_name_to_uniprot)} entries")
        except Exception as e:
            self.logger.debug(f"Could not build protein name mapping: {e}")
    
    def _should_use_standard_papyrus(self, uniprot_id: str) -> bool:
        """
        Check if a protein should use standard Papyrus based on config.
        
        Args:
            uniprot_id: UniProt ID or protein name to check
            
        Returns:
            True if protein should use standard Papyrus, False for Papyrus++
        """
        if not self.proteins_use_standard_papyrus:
            return False
        
        # Build name mapping if not already done
        self._build_protein_name_mapping()
        
        # Normalize input
        check_id = uniprot_id.upper()
        
        # Check direct match first
        for protein_entry in self.proteins_use_standard_papyrus:
            if isinstance(protein_entry, str):
                entry_upper = protein_entry.upper()
                # Direct match
                if entry_upper == check_id:
                    return True
                # Check if entry is a protein name that maps to this UniProt ID
                if entry_upper in self._protein_name_to_uniprot:
                    mapped_id = self._protein_name_to_uniprot[entry_upper]
                    if mapped_id == check_id:
                        return True
                # Check if check_id is a protein name that maps to entry
                if check_id in self._protein_name_to_uniprot:
                    mapped_id = self._protein_name_to_uniprot[check_id]
                    if mapped_id == entry_upper:
                        return True
        
        return False
    
    def get_filtered_papyrus(self, uniprot_ids: List[str] = None) -> pd.DataFrame:
        """
        Get filtered Papyrus data for specific proteins.
        Automatically selects Papyrus++ or standard Papyrus based on config.
        
        Args:
            uniprot_ids: List of UniProt IDs to filter by. If None, returns all data from Papyrus++.
            
        Returns:
            Filtered DataFrame from appropriate Papyrus version
        """
        if uniprot_ids is None:
            # Return all data from Papyrus++ (default)
            if self.papyrus_df.empty:
                self.logger.error("Papyrus++ dataset not loaded")
                return pd.DataFrame()
            return self.papyrus_df
        
        # Check if any of the requested proteins need standard Papyrus
        needs_standard = any(self._should_use_standard_papyrus(uid) for uid in uniprot_ids)
        needs_plusplus = any(not self._should_use_standard_papyrus(uid) for uid in uniprot_ids)
        
        results = []
        
        # Load standard Papyrus if needed
        if needs_standard:
            self._load_standard_papyrus()
            standard_ids = [uid for uid in uniprot_ids if self._should_use_standard_papyrus(uid)]
            if standard_ids and not self.papyrus_standard_df.empty:
                standard_df = self.papyrus_standard_df[self.papyrus_standard_df['accession'].isin(standard_ids)]
                if not standard_df.empty:
                    self.logger.info(f"Using standard Papyrus for {len(standard_ids)} protein(s): {standard_ids}")
                    results.append(standard_df)
        
        # Get Papyrus++ data for remaining proteins
        if needs_plusplus:
            if self.papyrus_df.empty:
                self.logger.error("Papyrus++ dataset not loaded")
            else:
                plusplus_ids = [uid for uid in uniprot_ids if not self._should_use_standard_papyrus(uid)]
                if plusplus_ids:
                    plusplus_df = self.papyrus_df[self.papyrus_df['accession'].isin(plusplus_ids)]
                    if not plusplus_df.empty:
                        if needs_standard:
                            self.logger.info(f"Using Papyrus++ for {len(plusplus_ids)} protein(s): {plusplus_ids}")
                        results.append(plusplus_df)
        
        # Combine results if both datasets were used
        if len(results) == 0:
            return pd.DataFrame()
        elif len(results) == 1:
            return results[0]
        else:
            # Combine DataFrames from both sources
            combined_df = pd.concat(results, ignore_index=True)
            self.logger.info(f"Combined data from both Papyrus versions: {len(combined_df)} total activities")
            return combined_df
    
    def load_esmc_descriptors(self, uniprot_id: str) -> Optional[np.ndarray]:
        """
        Load ESM-C descriptors from cache for a protein.
        
        Args:
            uniprot_id: UniProt ID of the protein
            
        Returns:
            ESM-C descriptor array or None if not found
        """
        esmc_file = self.esmc_cache_dir / f"{uniprot_id}_descriptors.pkl"
        
        if not esmc_file.exists():
            # Don't log error here - we'll summarize missing accessions at a higher level
            return None
        
        try:
            with open(esmc_file, 'rb') as f:
                esmc_data = pickle.load(f)
            
            # Extract ESM-C descriptors from the cached data
            esmc_descriptors = []
            
            if isinstance(esmc_data, dict):
                # If it's a dictionary, look for ESM-C descriptors
                for key, value in esmc_data.items():
                    if key.startswith('esm_dim_'):
                        esmc_descriptors.append(value)
                
                if esmc_descriptors:
                    esmc_embedding = np.array(esmc_descriptors)
                else:
                    # Look for any numeric keys or ESM-related keys
                    numeric_keys = [k for k in esmc_data.keys() if k.isdigit() or 'esm' in k.lower()]
                    if numeric_keys:
                        for key in sorted(numeric_keys):
                            if isinstance(esmc_data[key], (int, float)):
                                esmc_descriptors.append(esmc_data[key])
                    
                    if esmc_descriptors:
                        esmc_embedding = np.array(esmc_descriptors)
                    else:
                        # Don't log error here - we'll summarize missing accessions at a higher level
                        return None
                        
            elif isinstance(esmc_data, np.ndarray):
                esmc_embedding = esmc_data
                
            elif isinstance(esmc_data, pd.DataFrame):
                # Handle DataFrame format
                esm_cols = [col for col in esmc_data.columns if col.startswith('esm_dim_')]
                
                if esm_cols:
                    esmc_embedding = esmc_data[esm_cols].iloc[0].values
                else:
                    # Don't log error here - we'll summarize missing accessions at a higher level
                    return None
            else:
                # Log only unexpected formats as these are actual errors, not just missing files
                self.logger.warning(f"Unexpected ESM-C descriptor format for {uniprot_id}: {type(esmc_data)}")
                return None
            
            # Ensure it's a 1D array
            if esmc_embedding.ndim > 1:
                esmc_embedding = esmc_embedding.flatten()
            
            self.logger.info(f"Loaded ESM-C descriptors for {uniprot_id}: shape {esmc_embedding.shape}")
            return esmc_embedding
            
        except Exception as e:
            self.logger.error(f"Error loading ESM-C descriptors for {uniprot_id}: {e}")
            return None
