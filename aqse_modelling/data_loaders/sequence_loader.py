"""
Protein sequence loader.
"""

import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ProteinSequenceLoader:
    """Handles protein sequence loading"""
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.protein_sequence: Dict[str, str] = {}  # will store the loaded results
        

    def load_sequences(self) -> Dict[str, str]:
        """
        Load protein sequences from the CSV specified in config['sequence_file'].

        Returns:
            Dict[str, str]: {uniprot_id: sequence}
        """
        filepath = self.config.get("sequence_file")
        if not filepath:
            raise ValueError("Missing 'sequence_file' in config.")

        self.logger.info(f"Loading protein sequences from {filepath}")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        required_cols = ["uniprot_id", "protein_name", "sequence", "sequence_length"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Expected columns: {required_cols}")

        # Build dictionary
        self.protein_sequence = {}
        removed_count = 0
        for _, row in df.iterrows():
            seq = row["sequence"]
            uid = row["uniprot_id"]
            if pd.isna(seq) or seq == "" or pd.isna(uid) or uid == "":
                removed_count += 1
                continue
            self.protein_sequence[uid] = seq

        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} entries due to missing UniProt ID or sequence.")

        self.logger.info(f"Loaded {len(self.protein_sequence)} protein sequences")
        return self.protein_sequence

    def get_sequence(self, uniprot_id: str) -> str:
        """
        Get the protein sequence for a given UniProt ID.

        Args:
            uniprot_id (str): UniProt ID

        Returns:
            str: protein sequence or empty string if not found
        """
        if not self.protein_sequence:
            self.load_sequences()
        return self.protein_sequence.get(uniprot_id, "")
