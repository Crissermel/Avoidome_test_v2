import os
from pathlib import Path

import numpy as np
import polars as pl

from aqse_modelling.utils.config_loader import load_config, resolve_config_path
from aqse_modelling.utils.data_splitting import split_data_stratified
from aqse_modelling.utils.physicochemical_descriptors import (
    calculate_physicochemical_descriptors,
    validate_descriptors,
)


def test_load_config_resolves_relative_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "avoidome_file: data/avoidome.csv",
                "similarity_file: data/similarity.csv",
                "sequence_file: data/sequences.csv",
                "activity_thresholds_file: data/cutoffs.csv",
                "output_dir: outputs",
                "papyrus_cache_dir: cache/papyrus",
            ]
        )
    )

    config = load_config(config_path=config_path)

    for key in [
        "avoidome_file",
        "similarity_file",
        "sequence_file",
        "activity_thresholds_file",
        "output_dir",
        "papyrus_cache_dir",
    ]:
        assert key in config
        path_value = Path(config[key])
        assert path_value.is_absolute()
        assert str(path_value).startswith(str(config_path.parent.resolve()))


def test_resolve_config_path_prefers_arg_over_env(monkeypatch, tmp_path):
    arg_path = tmp_path / "from_arg.yaml"
    env_path = tmp_path / "from_env.yaml"

    arg_path.write_text("key: value_from_arg\n")
    env_path.write_text("key: value_from_env\n")

    monkeypatch.setenv("CONFIG_FILE", str(env_path))

    resolved = resolve_config_path(config_arg=str(arg_path))
    assert resolved == arg_path.resolve()


def test_resolve_config_path_uses_env_when_no_arg(monkeypatch, tmp_path):
    env_path = tmp_path / "from_env.yaml"
    env_path.write_text("key: value_from_env\n")

    monkeypatch.setenv("CONFIG_FILE", str(env_path))

    resolved = resolve_config_path(config_arg=None)
    assert resolved == env_path.resolve()


def test_calculate_physicochemical_descriptors_basic_properties():
    smiles = "CCO"  # ethanol
    descriptors = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)

    required_keys = [
        "ALogP",
        "Molecular_Weight",
        "Num_H_Donors",
        "Num_H_Acceptors",
        "Num_Rotatable_Bonds",
        "Num_Atoms",
        "Num_Rings",
        "Num_Aromatic_Rings",
        "LogS",
        "Molecular_Surface_Area",
        "Molecular_Polar_Surface_Area",
        "SASA",
        "Num_Heavy_Atoms",
        "Formal_Charge",
        "Num_Saturated_Rings",
    ]

    for key in required_keys:
        assert key in descriptors
        assert isinstance(descriptors[key], (float, int, np.floating, np.integer))


def test_validate_descriptors_returns_drug_like_flags():
    smiles = "CCO"
    descriptors = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)

    validation = validate_descriptors(descriptors)

    expected_keys = [
        "Lipinski_MW",
        "Lipinski_LogP",
        "Lipinski_HBD",
        "Lipinski_HBA",
        "TPSA_Range",
        "Rotatable_Bonds",
        "Aromatic_Rings",
        "Drug_Like",
    ]

    for key in expected_keys:
        assert key in validation
        assert isinstance(validation[key], bool)


def _build_synthetic_class_dataframe(n_rows: int = 60) -> pl.DataFrame:
    classes = np.random.choice([0, 1, 2], size=n_rows)
    # Create a few doc_ids with enough molecules to exercise fixed-test logic
    doc_ids = []
    for i in range(n_rows):
        if i < 40:
            doc_ids.append(f"doc_{i // 20}")  # two docs with 20 molecules each
        else:
            doc_ids.append(f"doc_{2 + (i - 40) // 10}")

    return pl.DataFrame(
        {
            "doc_id": doc_ids,
            "class": classes.tolist(),
            "value": np.arange(n_rows).tolist(),
        }
    )


def test_split_data_stratified_with_fixed_test_creates_disjoint_sets():
    df = _build_synthetic_class_dataframe()

    train_df, val_df, test_df = split_data_stratified(
        df,
        test_size=0.2,
        use_fixed_test=True,
        doc_id_column="doc_id",
        random_state=123,
    )

    # Basic sanity checks
    assert test_df is not None
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Ensure no overlaps between splits
    train_ids = set(train_df["value"].to_list())
    val_ids = set(val_df["value"].to_list())
    test_ids = set(test_df["value"].to_list())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Ensure we did not lose or duplicate rows
    combined_ids = train_ids | val_ids | test_ids
    original_ids = set(df["value"].to_list())
    assert combined_ids == original_ids

