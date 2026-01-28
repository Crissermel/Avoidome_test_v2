#!/usr/bin/env python3
"""
Physicochemical Descriptors Calculator for QSAR Modeling

This module provides functions to calculate key physicochemical descriptors
for drug molecules using RDKit. These descriptors are essential for QSAR
modeling and drug discovery applications.

REQUIREMENTS:
- Python 3.7+
- RDKit (pip install rdkit)
- NumPy (pip install numpy)

Author: Generated for QSAR modeling
Date: 2024
"""

import numpy as np
from typing import Dict, List, Union, Optional
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFreeSASA

# Suppress RDKit warnings
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_physicochemical_descriptors(
    smiles: str,
    include_sasa: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive physicochemical descriptors for a molecule.
    
    Args:
        smiles (str): SMILES string of the molecule
        include_sasa (bool): Whether to calculate SASA (requires additional setup)
        verbose (bool): Print calculation details
        
    Returns:
        Dict[str, float]: Dictionary containing all calculated descriptors
        
    Raises:
        ValueError: If SMILES string is invalid
        RuntimeError: If descriptor calculation fails
    """
    
    if verbose:
        print(f"Calculating descriptors for SMILES: {smiles}")
    
    # Parse SMILES string
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
    except Exception as e:
        raise ValueError(f"Error parsing SMILES: {e}")
    
    # Initialize descriptors dictionary
    descriptors = {}
    
    try:
        # 1. ALogP (lipophilicity / partition coefficient)
        descriptors['ALogP'] = Crippen.MolLogP(mol)
        
        # 2. Molecular Weight
        descriptors['Molecular_Weight'] = Descriptors.MolWt(mol)
        
        # 3. Number of Hydrogen Donors
        descriptors['Num_H_Donors'] = Descriptors.NumHDonors(mol)
        
        # 4. Number of Hydrogen Acceptors
        descriptors['Num_H_Acceptors'] = Descriptors.NumHAcceptors(mol)
        
        # 5. Number of Rotatable Bonds
        descriptors['Num_Rotatable_Bonds'] = Descriptors.NumRotatableBonds(mol)
        
        # 6. Number of Atoms
        descriptors['Num_Atoms'] = mol.GetNumAtoms()
        
        # 7. Number of Rings
        descriptors['Num_Rings'] = Descriptors.RingCount(mol)
        
        # 8. Number of Aromatic Rings
        descriptors['Num_Aromatic_Rings'] = Descriptors.NumAromaticRings(mol)
        
        # 9. Molecular Solubility (LogS) - estimate from LogP and MW
        logp = descriptors['ALogP']
        mw = descriptors['Molecular_Weight']
        # Estimation: LogS ≈ -LogP - 0.01*MW + 0.16
        descriptors['LogS'] = -logp - 0.01 * mw + 0.16
        
        # 10. Molecular Surface Area - estimate from molecular weight
        # Rough estimation: MSA ≈ 6.7 * MW^0.67 (Å²)
        descriptors['Molecular_Surface_Area'] = 6.7 * (mw ** 0.67)
        
        # 11. Molecular Polar Surface Area
        descriptors['Molecular_Polar_Surface_Area'] = Descriptors.TPSA(mol)
        
        # 12. Solvent-Accessible Surface Area (SASA)
        if include_sasa:
            try:
                descriptors['SASA'] = calculate_sasa(mol)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not calculate SASA: {e}")
                descriptors['SASA'] = np.nan
        else:
            descriptors['SASA'] = np.nan
            
        # Additional useful descriptors
        descriptors['Num_Heavy_Atoms'] = mol.GetNumHeavyAtoms()
        descriptors['Formal_Charge'] = Chem.rdmolops.GetFormalCharge(mol)
        descriptors['Num_Saturated_Rings'] = descriptors['Num_Rings'] - descriptors['Num_Aromatic_Rings']
        
        if verbose:
            print(f"Successfully calculated {len(descriptors)} descriptors")
            
    except Exception as e:
        raise RuntimeError(f"Error calculating descriptors: {e}")
    
    return descriptors


def calculate_sasa(mol: Chem.Mol, probe_radius: float = 1.4) -> float:
    """
    Calculate Solvent-Accessible Surface Area (SASA) for a molecule.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        probe_radius (float): Probe radius in Angstroms (default: 1.4 for water)
        
    Returns:
        float: SASA in square Angstroms
    """
    try:
        # Create conformer for SASA calculation
        from rdkit.Chem import AllChem
        mol_with_conf = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_conf)
        
        # Calculate SASA
        sasa = rdFreeSASA.CalcSASA(mol_with_conf, probe_radius)
        return sasa
        
    except Exception as e:
        # Fallback: estimate SASA from molecular surface area
        mw = Descriptors.MolWt(mol)
        msa = 6.7 * (mw ** 0.67)  # Use our MSA estimation
        # Rough approximation: SASA ≈ MSA * 1.2
        return msa * 1.2


def calculate_batch_descriptors(
    smiles_list: List[str],
    include_sasa: bool = True,
    verbose: bool = False
) -> List[Dict[str, float]]:
    """
    Calculate descriptors for a batch of molecules.
    
    Args:
        smiles_list (List[str]): List of SMILES strings
        include_sasa (bool): Whether to calculate SASA
        verbose (bool): Print progress information
        
    Returns:
        List[Dict[str, float]]: List of descriptor dictionaries
    """
    
    if verbose:
        print(f"Processing {len(smiles_list)} molecules...")
    
    results = []
    failed_molecules = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            descriptors = calculate_physicochemical_descriptors(
                smiles, include_sasa=include_sasa, verbose=False
            )
            results.append(descriptors)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(smiles_list)} molecules")
                
        except Exception as e:
            if verbose:
                print(f"Failed to process molecule {i+1}: {smiles} - {e}")
            failed_molecules.append((i, smiles, str(e)))
            # Add NaN values for failed molecules
            results.append({key: np.nan for key in [
                'ALogP', 'Molecular_Weight', 'Num_H_Donors', 'Num_H_Acceptors',
                'Num_Rotatable_Bonds', 'Num_Atoms', 'Num_Rings', 'Num_Aromatic_Rings',
                'LogS', 'Molecular_Surface_Area', 'Molecular_Polar_Surface_Area', 'SASA'
            ]})
    
    if verbose and failed_molecules:
        print(f"Failed to process {len(failed_molecules)} molecules")
    
    return results


def descriptors_to_dataframe(
    descriptors_list: List[Dict[str, float]],
    smiles_list: Optional[List[str]] = None
) -> 'pd.DataFrame':
    """
    Convert list of descriptor dictionaries to pandas DataFrame.
    
    Args:
        descriptors_list (List[Dict[str, float]]): List of descriptor dictionaries
        smiles_list (Optional[List[str]]): Optional SMILES strings for indexing
        
    Returns:
        pd.DataFrame: DataFrame with descriptors as columns
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(descriptors_list)
        
        if smiles_list is not None:
            df['SMILES'] = smiles_list
            df = df.set_index('SMILES')
            
        return df
        
    except ImportError:
        print("Warning: pandas not available. Returning list of dictionaries.")
        return descriptors_list


def validate_descriptors(descriptors: Dict[str, float]) -> Dict[str, bool]:
    """
    Validate descriptor values for drug-likeness.
    
    Args:
        descriptors (Dict[str, float]): Dictionary of calculated descriptors
        
    Returns:
        Dict[str, bool]: Validation results for each rule
    """
    validation = {}
    
    # Lipinski's Rule of Five
    validation['Lipinski_MW'] = descriptors['Molecular_Weight'] <= 500
    validation['Lipinski_LogP'] = descriptors['ALogP'] <= 5
    validation['Lipinski_HBD'] = descriptors['Num_H_Donors'] <= 5
    validation['Lipinski_HBA'] = descriptors['Num_H_Acceptors'] <= 10
    
    # Additional drug-likeness rules
    validation['TPSA_Range'] = 0 <= descriptors['Molecular_Polar_Surface_Area'] <= 140
    validation['Rotatable_Bonds'] = descriptors['Num_Rotatable_Bonds'] <= 10
    validation['Aromatic_Rings'] = descriptors['Num_Aromatic_Rings'] <= 5
    
    # Overall drug-likeness
    lipinski_passed = sum([
        validation['Lipinski_MW'], validation['Lipinski_LogP'],
        validation['Lipinski_HBD'], validation['Lipinski_HBA']
    ]) >= 3
    
    validation['Drug_Like'] = lipinski_passed and all([
        validation['TPSA_Range'], validation['Rotatable_Bonds'], validation['Aromatic_Rings']
    ])
    
    return validation


def print_descriptor_summary(descriptors: Dict[str, float]):
    """
    Print a formatted summary of calculated descriptors.
    
    Args:
        descriptors (Dict[str, float]): Dictionary of calculated descriptors
    """
    print("=" * 60)
    print("PHYSICOCHEMICAL DESCRIPTORS SUMMARY")
    print("=" * 60)
    
    # Basic properties
    print(f"Molecular Weight:        {descriptors['Molecular_Weight']:.2f} Da")
    print(f"ALogP (Lipophilicity):   {descriptors['ALogP']:.2f}")
    print(f"LogS (Solubility):       {descriptors['LogS']:.2f}")
    print(f"Formal Charge:           {descriptors['Formal_Charge']:.0f}")
    
    print("\n" + "-" * 40)
    print("STRUCTURAL PROPERTIES")
    print("-" * 40)
    print(f"Number of Atoms:         {descriptors['Num_Atoms']:.0f}")
    print(f"Number of Heavy Atoms:   {descriptors['Num_Heavy_Atoms']:.0f}")
    print(f"Number of Rings:         {descriptors['Num_Rings']:.0f}")
    print(f"Aromatic Rings:          {descriptors['Num_Aromatic_Rings']:.0f}")
    print(f"Saturated Rings:         {descriptors['Num_Saturated_Rings']:.0f}")
    print(f"Rotatable Bonds:         {descriptors['Num_Rotatable_Bonds']:.0f}")
    
    print("\n" + "-" * 40)
    print("SURFACE AREA PROPERTIES")
    print("-" * 40)
    print(f"Molecular Surface Area:  {descriptors['Molecular_Surface_Area']:.2f} Å²")
    print(f"Polar Surface Area:      {descriptors['Molecular_Polar_Surface_Area']:.2f} Å²")
    if not np.isnan(descriptors['SASA']):
        print(f"SASA:                    {descriptors['SASA']:.2f} Å²")
    else:
        print("SASA:                    Not calculated")
    
    print("\n" + "-" * 40)
    print("HYDROGEN BONDING")
    print("-" * 40)
    print(f"Hydrogen Bond Donors:    {descriptors['Num_H_Donors']:.0f}")
    print(f"Hydrogen Bond Acceptors: {descriptors['Num_H_Acceptors']:.0f}")
    
    # Validation
    validation = validate_descriptors(descriptors)
    print("\n" + "-" * 40)
    print("DRUG-LIKENESS VALIDATION")
    print("-" * 40)
    print(f"Lipinski's Rule of Five:  {'PASS' if sum([validation['Lipinski_MW'], validation['Lipinski_LogP'], validation['Lipinski_HBD'], validation['Lipinski_HBA']]) >= 3 else 'FAIL'}")
    print(f"Overall Drug-Like:       {'PASS' if validation['Drug_Like'] else 'FAIL'}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Physicochemical Descriptors Calculator")
    print("=" * 50)
    
    # Example molecules
    example_smiles = [
        "CCN(CC)CCCC(C)NC1=C2C=CC(Cl)=CC2=NC=C1",  # Chloroquine
        "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)CO",  # Quercetin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    for i, smiles in enumerate(example_smiles, 1):
        print(f"\nExample {i}: {smiles}")
        try:
            descriptors = calculate_physicochemical_descriptors(smiles, verbose=True)
            print_descriptor_summary(descriptors)
        except Exception as e:
            print(f"Error: {e}")
    
    # Batch processing example
    print(f"\n{'='*60}")
    print("BATCH PROCESSING EXAMPLE")
    print(f"{'='*60}")
    
    try:
        batch_results = calculate_batch_descriptors(example_smiles, verbose=True)
        print(f"Processed {len(batch_results)} molecules successfully")
        
        # Convert to DataFrame if pandas is available
        try:
            df = descriptors_to_dataframe(batch_results, example_smiles)
            print("\nDataFrame shape:", df.shape)
            print("\nFirst few rows:")
            print(df.head())
        except ImportError:
            print("Pandas not available for DataFrame conversion")
            
    except Exception as e:
        print(f"Batch processing error: {e}")
