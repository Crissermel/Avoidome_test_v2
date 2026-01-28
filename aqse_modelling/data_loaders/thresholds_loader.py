"""
Activity thresholds loader.
"""

import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ActivityThresholdsLoader:
    """Handles activity thresholds data loading and processing with support for optimized cutoffs and 2-class classification"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load activity thresholds (optimized cutoffs)
        self._load_thresholds()
        
        # Load 2-class protein list
        self._load_2class_proteins()
    
    def _load_thresholds(self):
        """Load optimized activity thresholds from CSV file"""
        filepath = self.config.get("activity_thresholds_file")
        if not filepath:
            self.logger.warning("No 'activity_thresholds_file' in configuration")
            self.thresholds_df = pd.DataFrame()
            return
        
        try:
            self.logger.info(f"Loading optimized activity thresholds from {filepath}")
            df = pd.read_csv(filepath)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Use optimized cutoffs if available, otherwise fall back to original
            if 'optimized_cutoff_high' in df.columns:
                df['cutoff_high'] = df['optimized_cutoff_high']
                self.logger.info("Using optimized high cutoffs")
            if 'optimized_cutoff_medium' in df.columns:
                df['cutoff_medium'] = df['optimized_cutoff_medium']
                self.logger.info("Using optimized medium cutoffs")
            
            # Ensure use_2class column exists (for backward compatibility)
            if 'use_2class' not in df.columns:
                self.logger.warning("'use_2class' column not found in thresholds file. All proteins will use 3-class classification.")
                df['use_2class'] = False
            else:
                # Convert use_2class to boolean if it's stored as string
                df['use_2class'] = df['use_2class'].astype(bool)
            
            # Ensure best_2class_option and best_2class_cutoff columns exist
            if 'best_2class_option' not in df.columns:
                df['best_2class_option'] = ''
            if 'best_2class_cutoff' not in df.columns:
                df['best_2class_cutoff'] = ''
            
            # Set index for easy lookup
            df.set_index('uniprot_id', inplace=True, drop=False)
            
            self.thresholds_df = df
            self.logger.info(f"Loaded activity thresholds for {len(df)} proteins")
            
            # Count 2-class proteins
            n_2class = df['use_2class'].sum() if 'use_2class' in df.columns else 0
            self.logger.info(f"Found {n_2class} proteins recommended for 2-class classification")
        except Exception as e:
            self.logger.error(f"Error loading activity thresholds: {e}")
            self.thresholds_df = pd.DataFrame()
    
    def _load_2class_proteins(self):
        """
        Load list of proteins that should use 2-class classification.
        This method is now deprecated - use_2class flag is read directly from CSV.
        Kept for backward compatibility but no longer used.
        """
        # This method is kept for backward compatibility but is no longer used.
        # The use_2class flag is now read directly from the thresholds CSV file.
        self.two_class_proteins = set()  # Empty set - will be populated from CSV if needed
    
    def get_thresholds(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get activity thresholds for a specific protein.
        
        Args:
            uniprot_id: UniProt ID of the protein
            
        Returns:
            Dictionary with 'high' and 'medium' cutoff values, 'n_classes' (2 or 3), 
            and 'class_labels' list, or empty dict if not found
        """
        if self.thresholds_df.empty:
            return {}
        
        try:
            # Look up by UniProt ID
            protein = self.thresholds_df[self.thresholds_df['uniprot_id'] == uniprot_id]
            
            if protein.empty:
                self.logger.warning(f"No thresholds found for {uniprot_id}")
                return {}
            
            row = protein.iloc[0]
            cutoff_high = float(row['cutoff_high'])
            
            # Check if this protein should use 2-class classification (from CSV)
            use_2class = False
            if 'use_2class' in row:
                use_2class = bool(row['use_2class'])
            
            # Fallback to hardcoded list if CSV doesn't have use_2class (backward compatibility)
            if 'use_2class' not in row.index and uniprot_id in self.two_class_proteins:
                use_2class = True
                self.logger.warning(f"Using hardcoded 2-class list for {uniprot_id} (CSV missing use_2class column)")
            
            if use_2class:
                # 2-class: determine which cutoff to use and class labels
                best_2class_option = row.get('best_2class_option', 'Low+Medium vs High')
                best_2class_cutoff = row.get('best_2class_cutoff', 'high')
                
                # Determine class labels based on best option
                if best_2class_option == 'Low+Medium vs High':
                    class_labels = ['Low+Medium', 'High']
                    cutoff_to_use = cutoff_high
                elif best_2class_option == 'Low vs Medium+High':
                    class_labels = ['Low', 'Medium+High']
                    cutoff_to_use = float(row.get('cutoff_medium', cutoff_high))
                elif best_2class_option == 'Medium vs High':
                    class_labels = ['Medium', 'High']
                    cutoff_to_use = cutoff_high
                else:
                    # Default: Low+Medium vs High
                    class_labels = ['Low+Medium', 'High']
                    cutoff_to_use = cutoff_high
                
                self.logger.info(f"Using 2-class classification for {uniprot_id}: {best_2class_option} (cutoff={cutoff_to_use:.1f})")
                return {
                    'high': cutoff_high,
                    'medium': float(row.get('cutoff_medium', cutoff_high)) if best_2class_cutoff == 'medium' else None,
                    'n_classes': 2,
                    'class_labels': class_labels,
                    'best_2class_option': best_2class_option,
                    'best_2class_cutoff': best_2class_cutoff
                }
            else:
                # 3-class: use both cutoffs
                cutoff_medium = float(row['cutoff_medium'])
                return {
                    'high': cutoff_high,
                    'medium': cutoff_medium,
                    'n_classes': 3,
                    'class_labels': ['Low', 'Medium', 'High']
                }
        except Exception as e:
            self.logger.error(f"Error getting thresholds for {uniprot_id}: {e}")
            return {}
    
    def get_thresholds_by_name(self, protein_name: str) -> Dict[str, Any]:
        """
        Get activity thresholds for a specific protein by name.
        
        Args:
            protein_name: Name of the protein (name2_entry column)
            
        Returns:
            Dictionary with 'high' and 'medium' cutoff values, 'n_classes', and 'class_labels', 
            or empty dict if not found
        """
        if self.thresholds_df.empty:
            return {}
        
        try:
            protein = self.thresholds_df[self.thresholds_df['name2_entry'] == protein_name]
            
            if protein.empty:
                self.logger.warning(f"No thresholds found for {protein_name}")
                return {}
            
            row = protein.iloc[0]
            uniprot_id = row['uniprot_id']
            # Use get_thresholds to get the full dict with n_classes
            return self.get_thresholds(uniprot_id)
        except Exception as e:
            self.logger.error(f"Error getting thresholds for {protein_name}: {e}")
            return {}
    
    def get_all_proteins(self) -> List[str]:
        """Get list of all proteins with thresholds"""
        if self.thresholds_df.empty:
            return []
        return self.thresholds_df['uniprot_id'].dropna().tolist()
