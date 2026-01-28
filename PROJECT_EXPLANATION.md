# Comprehensive Project Explanation: AQSE 3-Class Classification Workflow

## Executive Summary

**AQSE** stands for **Avoidome QSAR Similarity Expansion**. This project develops machine learning models to predict the binding activity of small molecules (drug candidates) against "avoidome" proteins - a curated set of proteins that should be avoided as drug targets due to their critical physiological functions and potential for adverse effects.

The project implements a sophisticated **proteochemometric (PCM) modeling** approach that combines:
- **QSAR (Quantitative Structure-Activity Relationship)** modeling for individual proteins
- **Similarity-based data expansion** using sequence-similar proteins from the Papyrus database
- **3-class activity classification** (High/Medium/Low binding activity)
- **Deep learning** (Chemprop) and **ensemble methods** (Random Forest) for model training

---

## 1. Biological and Pharmaceutical Context

### 1.1 What is the Avoidome?

The **Avoidome** is a curated list of approximately 52 proteins that should be avoided as drug targets in pharmaceutical development. These proteins fall into several critical functional categories:

#### **Drug Metabolism Enzymes**
- **Cytochrome P450 (CYP) enzymes**: CYP1A2, CYP2B6, CYP2C9, CYP2C19, CYP2D6, CYP3A4
  - These enzymes metabolize ~75% of all drugs
  - Inhibition can cause drug-drug interactions and toxicity
  - Induction can reduce drug efficacy

- **Oxidoreductases**: Aldehyde oxidase (AOX1), Xanthine oxidase (XDH), Monoamine oxidases (MAOA, MAOB)
  - Critical for xenobiotic metabolism
  - Inhibition can lead to accumulation of toxic metabolites

- **Phase II enzymes**: Sulfotransferases (SULT1A1), Glutathione S-transferases (GSTA1)
  - Conjugation reactions essential for drug elimination

#### **Nuclear Receptors**
- **AHR (Aryl hydrocarbon receptor)**: Regulates xenobiotic metabolism
- **CAR (NR1I3) and PXR (NR1I2)**: Control drug-metabolizing enzyme expression
- Binding to these can cause unintended transcriptional changes

#### **Transporters**
- **OATP transporters** (SLCO1B1, SLCO2B1, SLCO2B3): Hepatic drug uptake
- **SLC6 transporters** (SERT, DAT, NET): Neurotransmitter reuptake
- Inhibition can cause drug accumulation or neurotransmitter imbalances

#### **Ion Channels**
- **KCNH2 (hERG)**: Cardiac potassium channel - blocking causes fatal arrhythmias
- **SCN5A (NaV1.5)**: Cardiac sodium channel - critical for heart function
- **Calcium channels**: Essential for muscle contraction and neurotransmission

#### **Receptors**
- **Adrenergic receptors** (ADRA1A, ADRA2A, ADRB1, ADRB2): Cardiovascular regulation
- **Muscarinic and nicotinic receptors**: Neuromuscular function
- **Serotonin receptors** (HTR2B): Mood and cardiovascular regulation
- **Histamine receptors** (HRH1): Allergic and inflammatory responses

### 1.2 Why Predict Binding to Avoidome Proteins?

**Drug Safety Assessment**: Before a drug candidate enters clinical trials, pharmaceutical companies must assess its potential to bind to avoidome proteins. Unintended binding can cause:
- **Drug-drug interactions**: One drug inhibiting metabolism of another
- **Toxicity**: Accumulation of drugs or metabolites
- **Cardiac arrhythmias**: hERG channel blockade (major cause of drug withdrawals)
- **Neurological side effects**: Binding to neurotransmitter transporters/receptors
- **Metabolic disorders**: Disruption of essential metabolic pathways

**Regulatory Requirements**: Regulatory agencies (FDA, EMA) require extensive off-target profiling. Computational predictions can:
- Reduce experimental costs (binding assays are expensive)
- Prioritize compounds for experimental testing
- Guide medicinal chemistry optimization
- Identify potential safety issues early in drug development

---

## 2. Chemical and Computational Theory

### 2.1 QSAR (Quantitative Structure-Activity Relationship)

**QSAR** is a fundamental concept in computational chemistry that relates molecular structure to biological activity.

#### **Core Principle**
The activity of a compound is a function of its molecular properties:
```
Activity = f(Molecular Descriptors)
```

#### **Molecular Descriptors Used in This Project**

**1. Morgan Fingerprints (ECFP-like)**


**2. Physicochemical Descriptors**
- **Molecular weight**: Affects drug-likeness (Lipinski's Rule of Five)
- **LogP (partition coefficient)**: Hydrophobicity - affects membrane permeability
- **Topological Polar Surface Area (TPSA)**: Oral bioavailability predictor
- **Number of rotatable bonds**: Molecular flexibility
- **Number of hydrogen bond donors/acceptors**: Drug-likeness criteria
- **Aromatic ring count**: Aromaticity affects binding
- **Formal charge**: Electrostatic interactions

**3. RDKit Descriptors**
- Additional molecular properties calculated using RDKit
- Include electronic, steric, and topological features

#### **QSAR Model Types**

**Model A: Simple QSAR**
- Uses only compound features (Morgan fingerprints + physicochemical descriptors)
- Trained on bioactivity data for a single target protein

### 2.2 Proteochemometric (PCM) Modeling

**PCM** extends QSAR by incorporating **both** compound and protein information.


#### **Protein Descriptors: ESM-C Embeddings**

**ESM-C (Evolutionary Scale Modeling - Contact)**
- **What**: Deep learning protein language model embeddings
- **How**: Transformer-based model trained on millions of protein sequences
- **Output**: 960-dimensional vector encoding:
  - **Evolutionary information**: Conservation patterns across species
  - **Structural information**: Predicted contact maps and secondary structure
  - **Functional information**: Domain architecture and active site motifs
  - **Sequence patterns**: Amino acid composition and motifs


#### **PCM Model (Model B)**

**Model B: Proteochemometric Model**
- Combines compound features + protein ESM-C embeddings
- Trained on bioactivity data from **multiple similar proteins**
- Advantage: Can predict activity for new proteins by using their ESM embeddings

**Key Insight**: If two proteins have similar sequences (and thus similar ESM embeddings), compounds that bind to one may bind to the other with similar activity.

### 2.3 Similarity-Based Data Expansion

#### **The Data Sparsity Problem**

Many avoidome proteins have **limited experimental bioactivity data**:
- Some proteins are difficult to assay experimentally
- Historical data may be sparse or proprietary
- New proteins may have no data at all

#### **The Similarity Expansion Solution**

**Hypothesis**: Proteins with similar sequences share similar binding properties.

**Strategy**:
1. Find proteins similar to each avoidome protein using BLAST
2. Collect bioactivity data from similar proteins
3. Combine data from target + similar proteins
4. Train PCM model on expanded dataset

**Similarity Thresholds**:
- **High similarity (≥70% identity)**: Very similar binding sites, high confidence
- **Medium similarity (≥50% identity)**: Moderately similar, moderate confidence
- **Low similarity (≥30% identity)**: Distantly related, lower confidence but more data

**Example**: If CYP2C9 has 50 experimental data points but CYP2C19 (90% similar) has 500 data points, we can use CYP2C19's data to improve CYP2C9's model.

### 2.4 Activity Classification

#### **pChEMBL Values**

Bioactivity is measured as **pChEMBL** = -log₁₀(IC₅₀ or Ki in M)
- **IC₅₀**: Concentration inhibiting 50% of activity
- **Ki**: Inhibition constant
- **Higher pChEMBL = Stronger binding**

**Example**:
- pChEMBL = 6.0 → IC₅₀ = 1 μM (micromolar)
- pChEMBL = 7.0 → IC₅₀ = 100 nM (nanomolar) - 10x stronger
- pChEMBL = 8.0 → IC₅₀ = 10 nM - 100x stronger

#### **3-Class Classification**

**Low Activity**: pChEMBL ≤ 5.0 (IC₅₀ ≥ 10 μM)
- Weak binding, unlikely to cause issues at therapeutic doses
- **Interpretation**: Safe, low risk

**Medium Activity**: 5.0 < pChEMBL ≤ 6.0 (1-10 μM)
- Moderate binding, may cause issues at high doses
- **Interpretation**: Caution, monitor in development

**High Activity**: pChEMBL > 6.0 (IC₅₀ < 1 μM)
- Strong binding, high risk of adverse effects
- **Interpretation**: High risk, may need structural modification

#### **2-Class Classification (Alternative)**

Some proteins use **2-class**:
- **Low+Medium**: pChEMBL ≤ cutoff_high (combined inactive class)
- **High**: pChEMBL > cutoff_high (active class)

**Why 2-class?**
- Better for imbalanced datasets
- Focuses on "binding vs. non-binding" rather than fine-grained activity levels

---

## 3. Workflow Architecture

### 3.1 Step 1: Input Preparation

**Purpose**: Fetch protein sequences from UniProt database

**Process**:
1. Load avoidome protein list (CSV with UniProt IDs)
2. Query UniProt API for each protein sequence
3. Create FASTA files for BLAST search
4. Generate sequence metadata

**Outputs**:
- Individual FASTA files per protein
- Combined FASTA file
- Sequence metadata CSV

### 3.2 Step 2: Similarity Search

**Purpose**: Find sequence-similar proteins in Papyrus database

**Method**: **BLAST (Basic Local Alignment Search Tool)**
- Compares protein sequences using local alignment
- Calculates sequence identity percentage
- Identifies similar proteins at three thresholds

**Papyrus Database**:
- Large-scale bioactivity database (version 05.7)
- Contains millions of compound-protein interactions
- Includes experimental IC₅₀, Ki, and other activity measurements
- Covers thousands of proteins

**Process**:
1. Build BLAST database from Papyrus protein sequences
2. Query each avoidome protein against database
3. Filter results by identity thresholds (70%, 50%, 30%)
4. Generate similarity summary

**Outputs**:
- Similarity search summary CSV
- BLAST database files
- Visualization plots

### 3.3 Step 3: ESM-C Descriptor Calculation (Optional)

**Purpose**: Generate protein embeddings for PCM models

**Process**:
1. Load protein sequences
2. Run ESM-C model inference
3. Extract 960-dimensional embeddings
4. Cache descriptors for reuse

**Why Cached?**
- ESM-C inference is computationally expensive
- Embeddings are deterministic (same sequence → same embedding)
- Can be reused across different model training runs

**Outputs**:
- Cached ESM-C descriptor files (`.pkl` format)

### 3.4 Step 4: Model Training

**Purpose**: Train QSAR/PCM classification models

**Two Model Types**:

#### **Model A: Simple QSAR**
- **Input**: Compound features only
- **Data**: Bioactivity for target protein only
- **Use case**: No similar proteins found, or baseline comparison

#### **Model B: PCM Model**
- **Input**: Compound features + Protein ESM-C embeddings
- **Data**: Bioactivity from target + similar proteins (at each threshold)
- **Use case**: Similar proteins found, data expansion beneficial

**Training Process** (per protein):

1. **Load bioactivity data** from Papyrus
   - Query by UniProt ID or ChEMBL target ID
   - Filter by activity type (IC₅₀, Ki, etc.)
   - Convert to pChEMBL values

2. **Assign class labels**
   - Apply activity thresholds (High/Medium/Low)
   - Handle missing values

3. **Extract features**
   - **Compounds**: Morgan fingerprints + physicochemical descriptors
   - **Proteins** (Model B only): ESM-C embeddings

4. **Split data**
   - Stratified train/test split (80/20)
   - Maintains class balance

5. **Train model**
   - **Random Forest**: Ensemble of decision trees
   - **Chemprop**: Graph neural network on molecular graphs

6. **Evaluate**
   - Test set predictions
   - Metrics: Accuracy, Precision, Recall, F1-score (macro and weighted)
   - Per-class metrics

7. **Save results**
   - Trained models
   - Predictions
   - Metrics
   - MLflow tracking (experiment tracking)

---

## 4. Machine Learning Approaches

### 4.1 Random Forest

**Algorithm**: Ensemble of decision trees

**Features**:
- **Morgan fingerprints**: Binary vectors (2048 bits)
- **Physicochemical descriptors**: Continuous values (~20-30 features)
- **ESM-C embeddings** (Model B): 960-dimensional vectors

**Total feature dimensions**:
- Model A: ~2070 features
- Model B: ~3030 features

**Hyperparameters**:
- `n_estimators`: 500 trees
- `max_depth`: None (unlimited)
- `class_weight`: "balanced_subsample" (handles class imbalance)

**Advantages**:
- Fast training and prediction
- Handles mixed feature types well
- Interpretable (feature importance)
- Robust to overfitting

### 4.2 Chemprop (Graph Neural Network)

**Algorithm**: Message-passing neural network on molecular graphs

**Architecture**:
1. **Molecular graph construction**: SMILES → graph (atoms = nodes, bonds = edges)
2. **Message passing**: Information propagates through bonds (depth = 5)
3. **Atom embeddings**: Each atom gets a learned representation
4. **Aggregation**: Combine atom embeddings to molecule embedding ("mean" or "norm")
5. **Feed-forward network**: 5 layers, 1700 hidden units
6. **Additional features**: Physicochemical descriptors + ESM-C (Model B) concatenated
7. **Classification head**: Output probabilities for 3 classes

**Hyperparameters** (from config):
- `max_epochs`: 800 (increased for imbalanced data)
- `batch_size`: 200
- `hidden_size`: 1700
- `depth`: 5 (message passing layers)
- `dropout`: 0.5 (regularization)
- `learning_rate`: Cosine schedule (init: 0.00005, max: 0.00025, final: 0.00005)
- `warmup_epochs`: 5

**Advantages**:
- Learns molecular representations automatically
- Captures 3D-like information from 2D structure
- Can generalize to novel chemical structures
- State-of-the-art performance on molecular property prediction

**Why 800 Epochs?**
- Imbalanced datasets (few High activity examples)
- Model needs more time to learn minority classes
- Early stopping prevents overfitting

### 4.3 Class Imbalance Handling

**Problem**: Many proteins have few "High" activity compounds

**Solutions**:
1. **Class weights**: Penalize misclassifying minority classes more
2. **Stratified sampling**: Ensure train/test splits maintain class proportions
3. **Extended training**: More epochs for Chemprop to learn rare patterns
4. **Data expansion**: Similar proteins provide more examples

---

## 5. Data Sources and Integration

### 5.1 Papyrus Database

**What**: Large-scale bioactivity database
- **Version**: 05.7
- **Size**: Millions of compound-protein interactions
- **Coverage**: Thousands of proteins, millions of compounds
- **Activity types**: IC₅₀, Ki, EC₅₀, etc.
- **Standardization**: pChEMBL values (normalized activity measures)

**Access**: Via `papyrus-scripts` Python library
- Automatically downloads and manages database
- Provides query interface
- Handles data standardization

### 5.2 UniProt Database

**What**: Universal protein knowledgebase
- **Purpose**: Protein sequences and metadata
- **Access**: Via REST API
- **Usage**: Fetch sequences for BLAST search

### 5.3 ChEMBL Database

**What**: Bioactivity database (via Papyrus integration)
- **Purpose**: Standardized activity measurements
- **pChEMBL values**: Normalized binding affinities

---

## 6. Model Evaluation and Metrics

### 6.1 Classification Metrics

**Accuracy**: Overall correct predictions
- `(TP + TN) / (TP + TN + FP + FN)`

**Precision (Macro)**: Average precision across classes
- `(Precision_Low + Precision_Medium + Precision_High) / 3`

**Recall (Macro)**: Average recall across classes
- `(Recall_Low + Recall_Medium + Recall_High) / 3`

**F1-Score (Macro)**: Harmonic mean of precision and recall
- `2 × (Precision × Recall) / (Precision + Recall)`

**Weighted Metrics**: Weighted by class frequency
- Accounts for class imbalance

**Per-Class Metrics**: Individual performance for Low/Medium/High
- Important for understanding model behavior on each activity level

### 6.2 Confusion Matrix

Shows prediction vs. true labels:
```
              Predicted
           Low  Medium  High
True Low    X     X      X
     Medium  X     X      X
     High    X     X      X
```

**Interpretation**:
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- **High → Low misclassification**: Most dangerous (misses strong binders)
- **Low → High misclassification**: False positives (may reject safe compounds)

### 6.3 MLflow Tracking

**Purpose**: Experiment tracking and model versioning

**Tracked**:
- **Parameters**: Hyperparameters, model configuration
- **Metrics**: Test set performance
- **Artifacts**: Trained models, predictions, metrics files
- **Tags**: Protein name, model type, threshold

**Benefits**:
- Compare models across runs
- Reproducibility
- Model registry
- Performance monitoring

---

## 7. Practical Applications

### 7.1 Early-Stage Drug Discovery

**Virtual Screening**: Predict binding of large compound libraries
- Screen millions of compounds computationally
- Prioritize for experimental testing
- Reduce costs

**Lead Optimization**: Guide chemical modifications
- Predict activity of modified compounds
- Identify structural features causing binding
- Optimize away from avoidome targets

### 7.2 Safety Assessment

**Off-Target Profiling**: Predict binding to multiple avoidome proteins
- Comprehensive safety profile
- Identify potential drug-drug interactions
- Guide dose selection

**Regulatory Submission**: Computational evidence for safety
- Support experimental data
- Justify compound selection
- Risk assessment

### 7.3 Chemical Space Exploration

**Novel Compounds**: Predict activity for compounds not in training data
- Generalization to new chemical structures
- Explore chemical space efficiently
- Identify promising scaffolds

---

## 8. Technical Implementation Details

### 8.1 Feature Engineering Pipeline

**Compound Features**:
1. **SMILES parsing**: RDKit converts SMILES to molecular object
2. **Morgan fingerprints**: `AllChem.GetMorganFingerprintAsBitVect(radius=2, nBits=2048)`
3. **Physicochemical descriptors**: RDKit descriptor calculators
4. **Normalization**: Standard scaling (optional)

**Protein Features** (Model B):
1. **ESM-C loading**: Load cached 960-dimensional embeddings
2. **Concatenation**: Append to compound features

### 8.2 Data Preprocessing

**Missing Values**:
- Remove compounds with missing SMILES
- Remove compounds with missing activity values
- Handle missing ESM-C embeddings (skip compound)

**Class Balancing**:
- Stratified train/test split
- Class weights in Random Forest
- Extended training for Chemprop

**Feature Scaling**:
- Random Forest: No scaling needed (tree-based)
- Chemprop: Optional normalization

### 8.3 Model Training Workflow

**For Each Protein**:

1. **Check prerequisites**:
   - Activity thresholds defined?
   - Sufficient data? (minimum 30 samples)
   - Similar proteins found?

2. **Model A Training** (always):
   - Load target protein bioactivity
   - Extract compound features
   - Train QSAR model
   - Evaluate on test set

3. **Model B Training** (if similar proteins found):
   - For each threshold (high/medium/low):
     - Load target + similar proteins bioactivity
     - Extract compound + protein features
     - Train PCM model
     - Evaluate on test set

4. **Save Results**:
   - Model files
   - Predictions
   - Metrics
   - MLflow logging

### 8.4 Output Structure

```
results_chemprop_3c_2c/
├── models/
│   ├── CYP2C9_P11712_A_model.pkl.gz
│   ├── CYP2C9_P11712_B_high_model.pkl.gz
│   └── ...
├── metrics/
│   ├── CYP2C9_P11712_A_test_metrics.json
│   └── ...
├── predictions/
│   ├── CYP2C9_P11712_A_test_predictions.csv
│   └── ...
├── class_distributions/
│   └── ...
├── comprehensive_protein_report.csv
└── model_summary_report.json
```

---

## 9. Challenges and Solutions

### 9.1 Data Sparsity

**Challenge**: Many proteins have limited experimental data

**Solution**: Similarity-based expansion
- Use data from similar proteins
- Increases training set size
- Improves model generalization

### 9.2 Class Imbalance

**Challenge**: Few "High" activity examples

**Solution**:
- Class weights
- Extended training (800 epochs)
- Stratified sampling
- 2-class classification for severely imbalanced cases

### 9.3 Computational Cost

**Challenge**: Training many models is expensive

**Solution**:
- Caching (fingerprints, ESM-C embeddings)
- Parallel processing where possible
- MLflow for experiment tracking (avoid redundant runs)

### 9.4 Model Selection

**Challenge**: Which model performs best?

**Solution**:
- Compare Model A vs. Model B
- Compare Random Forest vs. Chemprop
- Compare different similarity thresholds
- Use comprehensive metrics and visualizations

---

## 10. Future Directions

### 10.1 Model Improvements

- **Transfer learning**: Pre-train on large datasets, fine-tune on avoidome
- **Multi-task learning**: Train on multiple proteins simultaneously
- **Attention mechanisms**: Identify important molecular substructures
- **Uncertainty quantification**: Predict confidence intervals

### 10.2 Data Expansion

- **Additional databases**: Integrate more bioactivity sources
- **Experimental collaboration**: Generate new data for sparse proteins
- **Active learning**: Prioritize compounds for experimental testing

### 10.3 Interpretability

- **Feature importance**: Which molecular features drive binding?
- **SHAP values**: Explain individual predictions
- **Attention visualization**: Highlight important atoms/bonds
- **Chemical rules**: Extract interpretable binding rules

---

## 11. Summary

This project implements a sophisticated computational pipeline for predicting small molecule binding to avoidome proteins. By combining:

- **QSAR modeling** (compound structure → activity)
- **PCM modeling** (compound + protein structure → activity)
- **Similarity-based data expansion** (leveraging related proteins)
- **Deep learning** (Chemprop) and **ensemble methods** (Random Forest)
- **3-class classification** (High/Medium/Low activity)

The system provides valuable predictions for drug safety assessment, enabling pharmaceutical companies to identify potential off-target binding early in drug development, reducing costs and improving safety.

The workflow is fully automated, tracks experiments with MLflow, and generates comprehensive reports for analysis and decision-making.

---

## References and Key Concepts

**QSAR**: Hansch, C. & Fujita, T. (1964). p-σ-π Analysis. A Method for the Correlation of Biological Activity and Chemical Structure.

**Proteochemometrics**: Lapinsh, M. et al. (2001). Proteochemometric Modeling of the Interaction of Amine G-Protein Coupled Receptors with a Diverse Set of Ligands.

**ESM-C**: Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.

**Chemprop**: Yang, K. et al. (2019). Analyzing Learned Molecular Representations for Property Prediction.

**Papyrus Database**: Sterling, T. & Irwin, J. J. (2015). ZINC 15 – Ligand Discovery for Everyone.

**Avoidome Concept**: Based on pharmaceutical industry best practices for off-target profiling and drug safety assessment.

