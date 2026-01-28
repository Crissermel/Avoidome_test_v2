#!/usr/bin/env python3
"""
Single ESM C Embedding Function

This module provides a clean function to compute ESM C embeddings for a single protein sequence.
ESM C is a more efficient replacement for ESM2 with better performance and lower memory requirements.

Based on the ESM C API from https://pypi.org/project/esm/

REQUIREMENTS:
- Python 3.10 or higher
- ESM C package (pip install esm)
- The 'esmc' conda environment must be activated before running this script
  To activate: conda activate esmc

"""

import torch
import numpy as np
import time
import warnings
from typing import Union, Dict, List, Optional

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    raise ImportError(
        "ESM C is not installed. Please install it with: pip install esm\n"
        "Make sure you are in the 'esmc' conda environment: conda activate esmc"
    )


def get_single_esmc_embedding(
    protein_sequence: str,
    model_name: str = "esmc_300m",
    device: Optional[str] = None,
    return_per_residue: bool = False,
    verbose: bool = True,
    use_flash_attn: bool = True
) -> Union[np.ndarray, Dict]:
    """
    Compute ESM C embedding for a single protein sequence.
    
    Args:
        protein_sequence (str): Protein sequence in single-letter amino acid code
        model_name (str): ESM C model name. Options: "esmc_300m", "esmc_600m"
        device (str, optional): Device to run on ("cuda", "cpu", or None for auto)
        return_per_residue (bool): If True, return per-residue embeddings
        verbose (bool): Print progress information
        use_flash_attn (bool): Use Flash Attention for faster computation
        
    Returns:
        Union[np.ndarray, Dict]: 
            - If return_per_residue=False: 1D array of sequence-level embedding
            - If return_per_residue=True: Dict with 'sequence_embedding' and 'per_residue_embeddings'
    """
    
    if verbose:
        print(f"Computing ESM C embedding for sequence of length {len(protein_sequence)}")
        print(f"Using model: {model_name}")
    
    # Input validation
    if not protein_sequence or not isinstance(protein_sequence, str):
        raise ValueError("Protein sequence must be a non-empty string")
    
    # Validate amino acid sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_aa for aa in protein_sequence.upper()):
        raise ValueError("Sequence contains invalid amino acid characters")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"Using device: {device}")
    
    start_time = time.time()
    
    try:
        # Load ESM C model
        if verbose:
            print("Loading ESM C model...")
        
        client = ESMC.from_pretrained(model_name).to(device)
        
        # Create ESMProtein object
        protein = ESMProtein(sequence=protein_sequence.upper())
        
        # Encode the protein
        if verbose:
            print("Encoding protein sequence...")
        
        protein_tensor = client.encode(protein)
        
        # Get logits and embeddings
        if verbose:
            print("Computing embeddings...")
        
        logits_config = LogitsConfig(
            sequence=True, 
            return_embeddings=True
        )
        
        # Disable flash attention if requested
        if not use_flash_attn:
            # Note: This would need to be passed to the model initialization
            # For now, we'll proceed with the default behavior
            pass
        
        logits_output = client.logits(protein_tensor, logits_config)
        
        # Extract embeddings
        embeddings = logits_output.embeddings
        
        if verbose:
            print(f"Embedding shape: {embeddings.shape}")
        
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Handle batch dimension if present (embeddings might be (1, seq_len, dim) or (seq_len, dim))
        # Squeeze out batch dimension if it's size 1
        if embeddings.ndim == 3 and embeddings.shape[0] == 1:
            embeddings = embeddings.squeeze(0)  # Remove batch dimension: (1, seq_len, dim) -> (seq_len, dim)
        
        # Process embeddings based on return type
        if return_per_residue:
            # Return both sequence-level and per-residue embeddings
            # Average along sequence dimension (axis 0) to get sequence-level embedding
            sequence_embedding = np.mean(embeddings, axis=0)  # (seq_len, dim) -> (dim,)
            
            result = {
                'sequence_embedding': sequence_embedding,
                'per_residue_embeddings': embeddings,
                'sequence_length': len(protein_sequence),
                'model_name': model_name,
                'embedding_dimension': embeddings.shape[-1],
                'computation_time': time.time() - start_time
            }
        else:
            # Return only sequence-level embedding (average pooled)
            # Average along sequence dimension (axis 0) to get sequence-level embedding
            sequence_embedding = np.mean(embeddings, axis=0)  # (seq_len, dim) -> (dim,)
            result = sequence_embedding
        
        if verbose:
            print(f"Computation completed in {time.time() - start_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error computing ESM C embedding: {str(e)}")


def batch_esmc_embeddings(
    protein_sequences: List[str],
    model_name: str = "esmc_300m",
    device: Optional[str] = None,
    batch_size: int = 1,
    verbose: bool = True,
    use_flash_attn: bool = True
) -> List[np.ndarray]:
    """
    Compute ESM C embeddings for multiple protein sequences.
    
    Args:
        protein_sequences (List[str]): List of protein sequences
        model_name (str): ESM C model name
        device (str, optional): Device to run on
        batch_size (int): Batch size for processing (currently limited to 1 for ESM C)
        verbose (bool): Print progress information
        use_flash_attn (bool): Use Flash Attention
        
    Returns:
        List[np.ndarray]: List of sequence-level embeddings
    """
    
    if verbose:
        print(f"Processing {len(protein_sequences)} sequences with ESM C")
    
    # ESM C currently processes one sequence at a time
    if batch_size != 1:
        if verbose:
            print("Warning: ESM C currently supports batch_size=1 only. Processing sequences individually.")
        batch_size = 1
    
    embeddings = []
    
    for i, sequence in enumerate(protein_sequences):
        if verbose:
            print(f"Processing sequence {i+1}/{len(protein_sequences)}")
        
        embedding = get_single_esmc_embedding(
            protein_sequence=sequence,
            model_name=model_name,
            device=device,
            return_per_residue=False,
            verbose=False,
            use_flash_attn=use_flash_attn
        )
        
        embeddings.append(embedding)
    
    return embeddings


