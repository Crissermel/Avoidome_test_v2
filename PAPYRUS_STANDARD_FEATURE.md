# Papyrus Standard Dataset Feature

## Overview

This feature allows you to specify which proteins should use the standard Papyrus dataset (not Papyrus++) for modeling. This is useful when you want to use the standard dataset for specific proteins while keeping Papyrus++ as the default for others.

## Configuration

Add a `proteins_use_standard_papyrus` list to your `config.yaml`:

```yaml
# Proteins to use standard Papyrus dataset (not Papyrus++)
# List protein names or UniProt IDs of proteins that should use standard Papyrus instead of Papyrus++
# All other proteins will use Papyrus++ by default
# Examples: "SLCO1B1" (protein name) or "P59520" (UniProt ID)
proteins_use_standard_papyrus:
  - "SLCO1B1"  # Uses standard Papyrus (not ++)
```

## How It Works

1. **Default Behavior**: All proteins use Papyrus++ by default
2. **Standard Papyrus**: Proteins listed in `proteins_use_standard_papyrus` will use the standard Papyrus dataset


## Example:  

```yaml
proteins_use_standard_papyrus:
  - "SLCO1B1"
  - "P59520"  # UniProt ID format also works
  - "SLCO2B1"
```
or empty list: 

```yaml
proteins_use_standard_papyrus: []  # All proteins use Papyrus++
```

Both protein names (e.g., "SLCO1B1") and UniProt IDs (e.g., "P59520") are supported

