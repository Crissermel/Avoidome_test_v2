"""
Avoidome data loader.
"""

import polars as pl
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
        """Load avoidome protein targets from input_clean.csv.
        Expected columns: uniprot_id, gene_name or protein_name, ChEMBL_target (optional).
        Returned dicts have keys 'UniProt ID' and 'Name_2' for downstream compatibility.
        """
        self.logger.info("Loading avoidome protein targets...")
        filepath = self.config.get("avoidome_file")

        if not filepath:
            raise ValueError("Missing 'avoidome_file' in configuration.")

        df = pl.read_csv(filepath)
        df = df.select([col for col in df.columns if not col.startswith('Unnamed')])
        rename_dict = {col: col.strip() for col in df.columns}
        df = df.rename(rename_dict)

        if "uniprot_id" not in df.columns:
            raise ValueError("Avoidome file (input_clean.csv) must have 'uniprot_id' column.")
        name_col = "gene_name" if "gene_name" in df.columns else "protein_name"
        cols = ["uniprot_id", name_col]
        if "ChEMBL_target" in df.columns:
            cols.append("ChEMBL_target")
        df = df.select([c for c in cols if c in df.columns])
        df = df.with_columns(pl.when(pl.col("uniprot_id") == "").then(None).otherwise(pl.col("uniprot_id")).alias("uniprot_id"))
        df = df.drop_nulls(subset=["uniprot_id"])
        result = []
        for row in df.iter_rows(named=True):
            result.append({
                "UniProt ID": row["uniprot_id"],
                "Name_2": row.get(name_col, row["uniprot_id"]),
                "ChEMBL_target": row.get("ChEMBL_target", ""),
            })
        self.logger.info(f"Loaded {len(result)} avoidome targets from {filepath}")
        return result


    def load_avoidome_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load protein-specific thresholds
         Returns:
            Dict[str, Dict[str, float]]: protein_id → {metric_name: threshold_value}

        """
        self.logger.info("Loading avoidome thresholds...")
        filepath = self.config.get("similarity_file")
    
        if not filepath:
            raise ValueError("Missing 'similarity_file' in configuration.")
        df = pl.read_csv(filepath)

        # Clean column names - strip whitespace
        rename_dict = {col: col.strip() for col in df.columns}
        df = df.rename(rename_dict)

        # Identify ID column
        if "UniProt ID" in df.columns:
            id_col = "UniProt ID"
        elif "protein_id" in df.columns:
            id_col = "protein_id"
        else:
            raise ValueError("No protein ID column found in thresholds file.")

        # Convert to dict with id_col as key
        thresholds = {}
        for row in df.iter_rows(named=True):
            protein_id = row[id_col]
            thresholds[protein_id] = {k: v for k, v in row.items() if k != id_col}

        self.logger.info(f"Loaded threshold data for {len(thresholds)} proteins from {filepath}")
        return thresholds
