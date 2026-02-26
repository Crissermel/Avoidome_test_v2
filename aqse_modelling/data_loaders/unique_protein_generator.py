"""
Unique protein list generator.
"""

import logging
from pathlib import Path
from typing import List

import polars as pl

from .avoidome_loader import AvoidomeDataLoader, load_avoidome_targets
from .similarity_loader import SimilarityDataLoader, load_similarity_results

logger = logging.getLogger(__name__)


def extract_protein_ids_from_similar(similar_proteins_str: str) -> List[str]:
    """Extract UniProt IDs from the similar_proteins string

    Args:
        similar_proteins_str: String like "P20813_WT (100.0%), P24460_WT (78.0%)"

    Returns:
        List of UniProt IDs (with _WT suffix removed if present)
    """
    if similar_proteins_str is None or similar_proteins_str == "":
        return []

    proteins = []
    for protein_entry in similar_proteins_str.split(","):
        # Extract protein ID (format: "P20813_WT (100.0%)")
        parts = protein_entry.strip().split(" ")[0]  # Get "P20813_WT"

        # Remove _WT suffix if present
        if "_WT" in parts:
            parts = parts.split("_WT")[0]

        proteins.append(parts)

    return proteins


def generate_unique_protein_list(
    avoidome_file: Path, similarity_file: Path, similarity_data
) -> pl.DataFrame:
    """Generate a list of all unique proteins (avoidome + similar)

    Returns:
        DataFrame with columns: ['uniprot_id', 'protein_name', 'source']
            - uniprot_id: UniProt ID
            - protein_name: Name of the protein (if available)
            - source: Either 'avoidome' or 'similar_to_{query_protein}'
    """
    logger.info("Generating unique protein list...")

    # Load avoidome proteins
    avoidome_proteins = load_avoidome_targets(avoidome_file)

    # Load similarity results
    similarity_data = load_similarity_results(similarity_file, similarity_data)

    # Collect all proteins with their metadata
    unique_proteins = {}

    # Add avoidome proteins
    for protein in avoidome_proteins:
        uniprot_id = protein["UniProt ID"]
        protein_name = protein.get("Name_2", "")

        unique_proteins[uniprot_id] = {
            "uniprot_id": uniprot_id,
            "protein_name": protein_name,
            "source": "avoidome",
        }

    # Add similar proteins from all thresholds
    for query_protein, thresholds in similarity_data.items():
        for threshold, similar_proteins in thresholds.items():
            for similar_protein in similar_proteins:
                if similar_protein not in unique_proteins:
                    # Find protein name if available
                    protein_name = ""
                    for av_protein in avoidome_proteins:
                        if av_protein["UniProt ID"] == similar_protein:
                            protein_name = av_protein.get("Name_2", "")
                            break

                    unique_proteins[similar_protein] = {
                        "uniprot_id": similar_protein,
                        "protein_name": protein_name,
                        "source": f"similar_to_{query_protein}",
                    }

    # Convert to DataFrame
    df = pl.DataFrame(list(unique_proteins.values()))
    df = df.sort("uniprot_id")

    logger.info(f"Generated unique protein list with {len(df)} proteins")
    logger.info(
        f"  - {len([p for p in unique_proteins.values() if p['source'] == 'avoidome'])} avoidome proteins"
    )

    return df


def save_protein_list(unique_protein_list: pl.DataFrame, output_path: Path) -> str:
    """Generate and save the unique protein list to CSV

    Args:
        output_path: Path to save the CSV file. If None, saves to output_dir/protein_list_all_unique.csv

    Returns:
        Path to the saved file
    """
    unique_protein_list.write_csv(output_path)
    logger.info(f"Saved unique protein list to {output_path}")

    return str(output_path)


class UniqueProteinListGenerator:
    """Generates a unique list of proteins including avoidome and similar proteins"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_protein_ids_from_similar(self, similar_proteins_str: str) -> List[str]:
        """Extract UniProt IDs from the similar_proteins string

        Args:
            similar_proteins_str: String like "P20813_WT (100.0%), P24460_WT (78.0%)"

        Returns:
            List of UniProt IDs (with _WT suffix removed if present)
        """
        if similar_proteins_str is None or similar_proteins_str == "":
            return []

        proteins = []
        for protein_entry in similar_proteins_str.split(","):
            # Extract protein ID (format: "P20813_WT (100.0%)")
            parts = protein_entry.strip().split(" ")[0]  # Get "P20813_WT"

            # Remove _WT suffix if present
            if "_WT" in parts:
                parts = parts.split("_WT")[0]

            proteins.append(parts)

        return proteins

    def generate_unique_protein_list(self) -> pl.DataFrame:
        """Generate a list of all unique proteins (avoidome + similar)

        Returns:
            DataFrame with columns: ['uniprot_id', 'protein_name', 'source']
                - uniprot_id: UniProt ID
                - protein_name: Name of the protein (if available)
                - source: Either 'avoidome' or 'similar_to_{query_protein}'
        """
        self.logger.info("Generating unique protein list...")

        # Load avoidome proteins
        avoidome_loader = AvoidomeDataLoader(self.config)
        avoidome_proteins = avoidome_loader.load_avoidome_targets()

        # Load similarity results
        similarity_loader = SimilarityDataLoader(self.config)
        similarity_data = similarity_loader.load_similarity_results()

        # Collect all proteins with their metadata
        unique_proteins = {}

        # Add avoidome proteins
        for protein in avoidome_proteins:
            uniprot_id = protein["UniProt ID"]
            protein_name = protein.get("Name_2", "")

            unique_proteins[uniprot_id] = {
                "uniprot_id": uniprot_id,
                "protein_name": protein_name,
                "source": "avoidome",
            }

        # Add similar proteins from all thresholds
        for query_protein, thresholds in similarity_data.items():
            for threshold, similar_proteins in thresholds.items():
                for similar_protein in similar_proteins:
                    if similar_protein not in unique_proteins:
                        # Find protein name if available
                        protein_name = ""
                        for av_protein in avoidome_proteins:
                            if av_protein["UniProt ID"] == similar_protein:
                                protein_name = av_protein.get("Name_2", "")
                                break

                        unique_proteins[similar_protein] = {
                            "uniprot_id": similar_protein,
                            "protein_name": protein_name,
                            "source": f"similar_to_{query_protein}",
                        }

        # Convert to DataFrame
        df = pl.DataFrame(list(unique_proteins.values()))
        df = df.sort("uniprot_id")

        self.logger.info(f"Generated unique protein list with {len(df)} proteins")
        self.logger.info(
            f"  - {len([p for p in unique_proteins.values() if p['source'] == 'avoidome'])} avoidome proteins"
        )
        self.logger.info(
            f"  - {len([p for p in unique_proteins.values() if p['source'] != 'avoidome'])} similar proteins"
        )

        return df

    def save_protein_list(self, output_path: str | Path | None = None) -> str:
        """Generate and save the unique protein list to CSV

        Args:
            output_path: Path to save the CSV file. If None, saves to output_dir/protein_list_all_unique.csv

        Returns:
            Path to the saved file
        """
        df = self.generate_unique_protein_list()

        if output_path is None:
            output_dir = Path(self.config.get("output_dir", "."))
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "protein_list_all_unique.csv"

        df.write_csv(output_path)
        self.logger.info(f"Saved unique protein list to {output_path}")

        return str(output_path)
