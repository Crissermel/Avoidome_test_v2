"""
Configuration loading utilities for AQSE workflow.

This module provides functions for loading and validating configuration files.
"""

import yaml
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in project root.
        
    Returns:
        Configuration dictionary with resolved paths
    """
    # Get project root directory (parent of scripts directory)
    if config_path is None:
        # Try to find config.yaml in common locations
        script_dir = Path(__file__).parent.parent.parent  # Go up from utils to aqse_modelling to AQSE_v3
        project_root = script_dir
        config_path = project_root / "config.yaml"
    
    # Validate config file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please provide --config argument or set CONFIG_FILE environment variable\n"
            f"Or place config.yaml in: {project_root}"
        )
    
    logger.info(f"Using config file: {config_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths in config to absolute paths (relative to config file location)
    config_dir = config_path.parent
    path_keys = ['avoidome_file', 'similarity_file', 'sequence_file', 'activity_thresholds_file', 
                 'output_dir', 'papyrus_cache_dir']
    for key in path_keys:
        if key in config and config[key]:
            # If path is relative, make it relative to config file directory
            path_value = Path(config[key])
            if not path_value.is_absolute():
                config[key] = str((config_dir / path_value).resolve())
            else:
                config[key] = str(Path(config[key]).expanduser().resolve())
    
    return config


def resolve_config_path(config_arg: str = None) -> Path:
    """
    Resolve configuration file path from command line argument or environment variable.
    
    Args:
        config_arg: Path provided via command line argument
        
    Returns:
        Resolved Path to config file
    """
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent.parent.parent  # Go up from utils to aqse_modelling to AQSE_v3
    project_root = script_dir
    
    # Determine config file path
    if config_arg:
        config_path = Path(config_arg).expanduser().resolve()
    elif os.getenv('CONFIG_FILE'):
        config_path = Path(os.getenv('CONFIG_FILE')).expanduser().resolve()
    else:
        # Default: look for config.yaml in project root
        config_path = project_root / "config.yaml"
    
    return config_path
