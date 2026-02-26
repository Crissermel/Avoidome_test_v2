"""
Avoidome data loader.
"""

import logging
import logging as lg  # temporary
from pathlib import Path
from typing import Dict, List

import polars as pl

logger = logging.getLogger(__name__)


def load_avoidome_targets(avoidome_file) -> list[dict[str, str]]:
    df = pl.read_csv(avoidome_file)
    df = df.select([col for col in df.columns if not col.startswith("Unnamed")])
    # Rename columns to strip whitespace
    rename_dict = {col: col.strip() for col in df.columns}
    df = df.rename(rename_dict)

    expected_cols = ["Name_2", "UniProt ID", "ChEMBL_target"]
    df = df.select(expected_cols)

    # Convert empty strings to null and drop missing
    df = df.with_columns(
        [
            pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
            for col in ["UniProt ID", "ChEMBL_target"]
        ]
    )
    before_count = len(df)
    df = df.drop_nulls(subset=["UniProt ID", "ChEMBL_target"])
    after_count = len(df)
    removed = before_count - after_count

    if removed > 0:
        lg.warning(
            f"Removed {removed} proteins due to missing UniProt ID or ChEMBL target."
        )
    else:
        lg.info("No proteins removed due to missing identifiers.")

    lg.info(f"Loaded {after_count} avoidome targets from {avoidome_file}")
    return df.to_dicts()


def load_avoidome_thresholds(
    similarity_file_path: Path | str,
) -> Dict[str, Dict[str, float]]:
    """Load protein-specific thresholds
    Args:
        similarity_file_path : path to similarity file csv
    Returns:
       Dict[str, Dict[str, float]]: protein_id → {metric_name: threshold_value}

    """
    lg.info("Loading avoidome thresholds...")

    df = pl.read_csv(similarity_file_path)

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

    lg.info(
        f"Loaded threshold data for {len(thresholds)} proteins from {similarity_file_path}"
    )
    return thresholds


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

        df = pl.read_csv(filepath)

        # Clean columns - remove Unnamed columns and strip whitespace
        df = df.select([col for col in df.columns if not col.startswith("Unnamed")])
        # Rename columns to strip whitespace
        rename_dict = {col: col.strip() for col in df.columns}
        df = df.rename(rename_dict)

        expected_cols = ["Name_2", "UniProt ID", "ChEMBL_target"]
        df = df.select(expected_cols)

        # Convert empty strings to null and drop missing
        df = df.with_columns(
            [
                pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
                for col in ["UniProt ID", "ChEMBL_target"]
            ]
        )
        before_count = len(df)
        df = df.drop_nulls(subset=["UniProt ID", "ChEMBL_target"])
        after_count = len(df)
        removed = before_count - after_count

        if removed > 0:
            self.logger.warning(
                f"Removed {removed} proteins due to missing UniProt ID or ChEMBL target."
            )
        else:
            self.logger.info("No proteins removed due to missing identifiers.")

        self.logger.info(f"Loaded {after_count} avoidome targets from {filepath}")
        return df.to_dicts()

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

        self.logger.info(
            f"Loaded threshold data for {len(thresholds)} proteins from {filepath}"
        )
        self.logger.info(
            f"Loaded threshold data for {len(thresholds)} proteins from {filepath}"
        )
        return thresholds
