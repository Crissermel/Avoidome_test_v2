"""
Similarity data loader.
"""

import logging
from typing import Dict, List

import polars as pl

logger = logging.getLogger(__name__)


def load_similarity_results(
    similarity_file, similarity_data
) -> Dict[str, Dict[str, List[str]]]:
    """Load similarity search results and store as nested dict."""

    logger.info(f"Loading similarity results from {similarity_file}")
    df = pl.read_csv(similarity_file)
    # Clean column names - strip whitespace
    rename_dict = {col: col.strip() for col in df.columns}
    df = df.rename(rename_dict)

    if (
        "query_protein" not in df.columns
        or "threshold" not in df.columns
        or "similar_proteins" not in df.columns
    ):
        raise ValueError(
            "Expected columns: 'query_protein', 'threshold', 'similar_proteins'"
        )

    # Build nested dictionary
    for row in df.iter_rows(named=True):
        query = row["query_protein"]
        thresh = row["threshold"]
        similars_str = row.get("similar_proteins", "")
        similars = []

        if similars_str is not None and similars_str != "":
            for p in similars_str.split(","):
                prot = p.strip().split(" ")[0]  # remove score in parentheses
                prot = prot.split("_")[0]  # remove any suffix like _WT
                if prot != query:  # remove query itself
                    similars.append(prot)

        if query not in similarity_data:
            similarity_data[query] = {}
        similarity_data[query][thresh] = similars
    logger.info(f"Loaded similarity data for {len(similarity_data)} proteins")
    return similarity_data


def get_similar_proteins(
    similarity_file, similarity_data, uniprot_id: str
) -> List[str]:
    """Get all similar proteins for a given UniProt ID across all thresholds."""
    if not similarity_data:
        load_similarity_results(similarity_file, similarity_data)
    if uniprot_id not in similarity_data:
        logger.warning(f"No similarity data found for {uniprot_id}")
        return []

    # Merge all lists from all thresholds
    similar_proteins = []
    for proteins in similarity_data[uniprot_id].values():
        similar_proteins.extend(proteins)
    return list(set(similar_proteins))  # remove duplicates


def get_similar_proteins_for_threshold(
    similarity_file, similarity_data, uniprot_id: str, threshold: str
) -> List[str]:
    """Get similar proteins for a specific threshold (e.g., 'high')."""
    if not similarity_data:
        load_similarity_results(similarity_file, similarity_data)
    if uniprot_id not in similarity_data:
        logger.warning(f"No similarity data found for {uniprot_id}")
        return []
    return similarity_data[uniprot_id].get(threshold, [])


class SimilarityDataLoader:
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similarity_data: Dict[
            str, Dict[str, List[str]]
        ] = {}  # will store the loaded results

    def load_similarity_results(self) -> Dict[str, Dict[str, List[str]]]:
        """Load similarity search results and store as nested dict."""
        filepath = self.config.get("similarity_file")
        if not filepath:
            raise ValueError("Missing 'similarity_file' in config.")

        self.logger.info(f"Loading similarity results from {filepath}")
        df = pl.read_csv(filepath)
        # Clean column names - strip whitespace
        rename_dict = {col: col.strip() for col in df.columns}
        df = df.rename(rename_dict)

        if (
            "query_protein" not in df.columns
            or "threshold" not in df.columns
            or "similar_proteins" not in df.columns
        ):
            raise ValueError(
                "Expected columns: 'query_protein', 'threshold', 'similar_proteins'"
            )

        # Build nested dictionary
        for row in df.iter_rows(named=True):
            query = row["query_protein"]
            thresh = row["threshold"]
            similars_str = row.get("similar_proteins", "")
            similars = []

            if similars_str is not None and similars_str != "":
                for p in similars_str.split(","):
                    prot = p.strip().split(" ")[0]  # remove score in parentheses
                    prot = prot.split("_")[0]  # remove any suffix like _WT
                    if prot != query:  # remove query itself
                        similars.append(prot)

            if query not in self.similarity_data:
                self.similarity_data[query] = {}
            self.similarity_data[query][thresh] = similars
        self.logger.info(
            f"Loaded similarity data for {len(self.similarity_data)} proteins"
        )
        return self.similarity_data

    def get_similar_proteins(self, uniprot_id: str) -> List[str]:
        """Get all similar proteins for a given UniProt ID across all thresholds."""
        if not self.similarity_data:
            self.load_similarity_results()
        if uniprot_id not in self.similarity_data:
            self.logger.warning(f"No similarity data found for {uniprot_id}")
            return []

        # Merge all lists from all thresholds
        similar_proteins = []
        for proteins in self.similarity_data[uniprot_id].values():
            similar_proteins.extend(proteins)
        return list(set(similar_proteins))  # remove duplicates

    def get_similar_proteins_for_threshold(
        self, uniprot_id: str, threshold: str
    ) -> List[str]:
        """Get similar proteins for a specific threshold (e.g., 'high')."""
        if not self.similarity_data:
            self.load_similarity_results()
        if uniprot_id not in self.similarity_data:
            self.logger.warning(f"No similarity data found for {uniprot_id}")
            return []
        return self.similarity_data[uniprot_id].get(threshold, [])
