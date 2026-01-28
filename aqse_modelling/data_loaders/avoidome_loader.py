"""
Avoidome data loader.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class AvoidomeDataLoader:
    """Handles avoidome protein data loading"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    
    def load_avoidome_targets(self) -> List[Dict[str, str]]:
        """Load avoidome protein targets"""
        self.logger.info("Loading avoidome protein targets...")
        filepath = self.config.get("avoidome_file")

        if not filepath:
            raise ValueError("Missing 'avoidome_file' in configuration.")

        df = pd.read_csv(filepath)

        # Clean columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        expected_cols = ["Name_2", "UniProt ID", "ChEMBL_target"]
        df = df[expected_cols]

        # Convert empty strings to NaN and drop missing
        df = df.replace("", np.nan)
        before_count = len(df)
        df = df.dropna(subset=["UniProt ID", "ChEMBL_target"])
        after_count = len(df)
        removed = before_count - after_count

        if removed > 0:
            self.logger.warning(f"Removed {removed} proteins due to missing UniProt ID or ChEMBL target.")
        else:
            self.logger.info("No proteins removed due to missing identifiers.")

        self.logger.info(f"Loaded {after_count} avoidome targets from {filepath}")
        return df.to_dict(orient="records")


    def load_avoidome_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load protein-specific thresholds
         Returns:
            Dict[str, Dict[str, float]]: protein_id → {metric_name: threshold_value}

        """
        self.logger.info("Loading avoidome thresholds...")
        filepath = self.config.get("similarity_file")
    
        if not filepath:
            raise ValueError("Missing 'similarity_file' in configuration.")
        df = pd.read_csv(filepath)

        # clean column names
        df.columns = df.columns.str.strip()


        # Identify ID column
        if "UniProt ID" in df.columns:
            id_col = "UniProt ID"
        elif "protein_id" in df.columns:
            id_col = "protein_id"
        else:
            raise ValueError("No protein ID column found in thresholds file.")

        df = df.set_index(id_col)
        thresholds = df.to_dict(orient="index")

        self.logger.info(f"Loaded threshold data for {len(thresholds)} proteins from {filepath}")
        return thresholds
