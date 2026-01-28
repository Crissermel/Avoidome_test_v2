"""
Workflow orchestrator for AQSE 3-Class Classification.

This module provides the main workflow orchestrator that coordinates all components
to process proteins and train models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..data_loaders import (
    AvoidomeDataLoader,
    SimilarityDataLoader,
    BioactivityDataLoader,
    ActivityThresholdsLoader
)
from ..models.base import create_model_trainer
from ..models.hyperparameter_optimizer import ChempropHyperparameterOptimizer
from ..reporting.model_reporter import ModelReporter
from ..utils.mlflow_logger import setup_mlflow, log_to_mlflow
from ..workflow.model_trainer_wrapper import train_model_with_reporting
from ..workflow.data_preparation import prepare_model_a_data, prepare_model_b_data

logger = logging.getLogger(__name__)


class AQSE3CWorkflow:
    """
    AQSE 3-Class Workflow Orchestrator
    
    Processes all proteins from avoidome list and trains models:
    - Model A: Simple QSAR for proteins without similar proteins (configurable: RF or Chemprop)
    - Model B: PCM model with similar proteins for 3 thresholds (configurable: RF or Chemprop)
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize loaders
        self.avoidome_loader = AvoidomeDataLoader(config)
        self.similarity_loader = SimilarityDataLoader(config)
        self.bioactivity_loader = BioactivityDataLoader(config)
        self.thresholds_loader = ActivityThresholdsLoader(config)
        
        # Initialize model trainer (pass bioactivity_loader for Chemprop)
        self.model_trainer = create_model_trainer(config, bioactivity_loader=self.bioactivity_loader)
        self.logger.info(f"Using {self.model_trainer.get_model_type_name()} trainer")
        
        # Initialize model reporter
        # Include model type in results directory path
        model_type = config.get("model_type", "random_forest")
        results_dir = Path(config.get("output_dir", ".")) / f"results_{model_type}"
        self.model_reporter = ModelReporter(results_dir)
        
        # Initialize MLflow
        mlflow_config = setup_mlflow(config, model_type)
        self.mlflow_enabled = mlflow_config.get('mlflow_enabled', False)
        self.mlflow_experiment_name = mlflow_config.get('mlflow_experiment_name')
        
        # Load data
        self.targets = self.avoidome_loader.load_avoidome_targets()
        self.similarity_loader.load_similarity_results()
        
        # Cache fingerprints ONCE at initialization (only if trainer requires precomputed features)
        if self.model_trainer.requires_precomputed_features():
            self.logger.info("Loading Morgan fingerprints (will be cached for all proteins)...")
            self._cached_fingerprints = self.bioactivity_loader.load_morgan_fingerprints()
            self.logger.info(f"Cached {len(self._cached_fingerprints):,} fingerprints")
        else:
            self.logger.info("Model trainer does not require precomputed features - skipping fingerprint caching")
            self._cached_fingerprints = {}
        
        # Create output directory
        self.output_dir = Path(config.get("output_dir", "."))
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
    
    def process_all_proteins(self):
        """Process all proteins in the avoidome list"""
        self.logger.info(f"Starting AQSE 3-Class workflow for {len(self.targets)} proteins")
        
        for i, protein in enumerate(self.targets):
            uniprot_id = protein['UniProt ID']
            protein_name = protein.get('Name_2', uniprot_id)
            
            # Display compact status at start
            self._print_status(f"[{i+1}/{len(self.targets)}] Processing: {protein_name} ({uniprot_id})")
            
            try:
                current_idx = i + 1
                total = len(self.targets)

                # Check if protein has similar proteins
                similar_proteins = self.similarity_loader.get_similar_proteins(uniprot_id)
                has_similar = len(similar_proteins) > 0
                
                # Get activity thresholds (includes n_classes and class_labels)
                thresholds = self.thresholds_loader.get_thresholds(uniprot_id)
                if not thresholds:
                    self.logger.warning(f"No activity thresholds found for {uniprot_id}")
                    result = {
                        'protein': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': 'unknown',
                        'status': 'no_thresholds'
                    }
                else:
                    # Log classification type
                    n_classes = thresholds.get('n_classes', 3)
                    class_labels = thresholds.get('class_labels', ['Low', 'Medium', 'High'])
                    
                    if not has_similar:
                        # Model A only: Simple QSAR for proteins without similar proteins
                        result = self.train_model_a(uniprot_id, protein_name, thresholds, current_idx, total)
                    else:
                        # Both Model A and Model B: QSAR + PCM for proteins with similar proteins
                        # Train Model A first
                        model_a_result = self.train_model_a(
                            uniprot_id,
                            protein_name,
                            thresholds,
                            current_idx,
                            total,
                            track_for_model_b=True
                        )
                        
                        # Display Model A completion status and start of Model B
                        model_a_f1 = None
                        if model_a_result.get('status') == 'complete' and 'test_f1_macro' in model_a_result:
                            model_a_f1 = model_a_result['test_f1_macro']
                            status_msg = (
                                f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  "
                                f"✓ Model A completed (F1: {model_a_f1:.2f})  → Training Model B (high threshold)..."
                            )
                            self._print_status(status_msg)
                        
                        # Train Model B (pass Model A F1 for status display)
                        model_b_result = self.train_model_b(
                            uniprot_id,
                            protein_name,
                            thresholds,
                            current_idx,
                            total,
                            model_a_f1=model_a_f1
                        )
                        
                        # Combined results
                        result = {
                            'protein': protein_name,
                            'uniprot_id': uniprot_id,
                            'has_similar_proteins': True,
                            'n_similar_proteins': len(similar_proteins),
                            'model_a': model_a_result,
                            'model_b': model_b_result
                        }

                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {protein_name}: {e}")
                self.results.append({
                    'protein': protein_name,
                    'uniprot_id': uniprot_id,
                    'error': str(e)
                })
        
        self._save_results()
    
    def _print_status(self, message: str, end: str = '\n'):
        """Print compact status message"""
        print(f"\r{message}", end=end, flush=True)
    
    def train_model_a(self, uniprot_id: str, protein_name: str, thresholds: Dict[str, Any], current_idx: int = None, total: int = None, track_for_model_b: bool = False) -> Dict[str, Any]:
        """Train Model A: Simple QSAR model for proteins without similar proteins"""
        self.logger.info("Training Model A: Simple QSAR")
        
        # Load bioactivity data
        bioactivity_data = self.bioactivity_loader.get_filtered_papyrus([uniprot_id])
        
        if len(bioactivity_data) < 30:
            self.logger.warning(f"Insufficient data: {len(bioactivity_data)} samples (need ≥30)")
            return {
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'model_type': 'A',
                'status': 'insufficient_data',
                'n_samples': len(bioactivity_data)
            }
        
        # Prepare data using the data preparation module
        from ..workflow.data_preparation import prepare_model_a_data
        
        train_df, val_df, test_df, total_samples = prepare_model_a_data(
            bioactivity_data=bioactivity_data,
            uniprot_id=uniprot_id,
            thresholds=thresholds,
            model_trainer=self.model_trainer,
            cached_fingerprints=self._cached_fingerprints,
            bioactivity_loader=self.bioactivity_loader,
            config=self.config
        )
        
        if train_df is None:
            return {
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'model_type': 'A',
                'status': 'insufficient_data',
                'n_samples': total_samples
            }
        
        self.logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        if val_df is not None:
            self.logger.info(f"Validation: {len(val_df)}")
        self.logger.info(f"Class distribution - Train: {train_df['class'].value_counts().to_dict()}")
        self.logger.info(f"Class distribution - Test: {test_df['class'].value_counts().to_dict()}")
        if val_df is not None:
            self.logger.info(f"Class distribution - Val: {val_df['class'].value_counts().to_dict()}")
        
        # Check if parameter optimization is enabled (only for Chemprop)
        use_optimization = self.config.get('enable_parameter_optimization', False)
        model_type_config = self.config.get('model_type', 'random_forest').lower()
        
        if use_optimization and model_type_config == 'chemprop' and val_df is not None:
            self.logger.info("Running hyperparameter optimization...")
            optimizer = ChempropHyperparameterOptimizer(
                n_trials=self.config.get('optimization_n_trials', 50),
                search_strategy=self.config.get('optimization_strategy', 'hybrid'),
                random_state=self.config.get('random_state', 42),
                bioactivity_loader=self.bioactivity_loader
            )
            
            opt_result = optimizer.optimize(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_type='A',
                protein_name=protein_name,
                uniprot_id=uniprot_id,
                thresholds=thresholds,
                max_epochs=self.config.get('chemprop_max_epochs', 100),
                optimization_epochs=self.config.get('optimization_epochs', 50),
                n_classes=thresholds.get('n_classes', 3),
                include_esm=False,
                val_fraction=0.0
            )
            
            if opt_result['status'] == 'success':
                final_result = opt_result['final_test_result']
                train_results = {
                    'status': 'success',
                    'test_accuracy': final_result.get('test_accuracy', 0.0),
                    'test_f1_macro': final_result.get('test_f1_macro', 0.0),
                    'test_f1_weighted': final_result.get('test_f1', 0.0),
                    'model_path': 'N/A',
                    'metrics_path': 'N/A',
                    'best_params': opt_result['best_params'],
                    'best_val_score': opt_result['best_val_score'],
                    'optimization_trials': opt_result['n_trials'],
                    'optimization_successful': opt_result['n_successful']
                }
            else:
                self.logger.warning("Parameter optimization failed, falling back to default training")
                train_results = train_model_with_reporting(
                    model_trainer=self.model_trainer,
                    model_reporter=self.model_reporter,
                    train_df=train_df,
                    test_df=test_df,
                    model_type='A',
                    protein_name=protein_name,
                    uniprot_id=uniprot_id,
                    thresholds=thresholds,
                    mlflow_enabled=self.mlflow_enabled,
                    mlflow_experiment_name=self.mlflow_experiment_name,
                    output_dir=self.output_dir,
                    cached_fingerprints=self._cached_fingerprints,
                    bioactivity_loader=self.bioactivity_loader
                )
        else:
            # Standard training without optimization
            train_results = train_model_with_reporting(
                model_trainer=self.model_trainer,
                model_reporter=self.model_reporter,
                train_df=train_df,
                test_df=test_df,
                model_type='A',
                protein_name=protein_name,
                uniprot_id=uniprot_id,
                thresholds=thresholds,
                mlflow_enabled=self.mlflow_enabled,
                mlflow_experiment_name=self.mlflow_experiment_name,
                output_dir=self.output_dir,
                cached_fingerprints=self._cached_fingerprints,
                bioactivity_loader=self.bioactivity_loader
            )
        
        result = {
            'protein': protein_name,
            'uniprot_id': uniprot_id,
            'model_type': 'A',
            'status': 'complete',
            'n_samples': total_samples,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'similar_proteins': 'N/A',
            'n_features': train_results.get('n_features', 'N/A'),
            **train_results
        }
        
        # Display completion status if this is a standalone Model A (no similar proteins)
        # Don't display if track_for_model_b is True (will be displayed with Model B start)
        if current_idx and total and not track_for_model_b and train_results.get('status') == 'success' and 'test_f1_macro' in train_results:
            f1_score = train_results['test_f1_macro']
            status_msg = f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  ✓ Model A completed (F1: {f1_score:.2f})"
            self._print_status(status_msg)
        
        return result
    
    def train_model_b(self, uniprot_id: str, protein_name: str, thresholds: Dict[str, Any], current_idx: int = None, total: int = None, model_a_f1: float = None) -> Dict[str, Any]:
        """Train Model B: PCM model with similar proteins at 3 thresholds"""
        trainer_name = self.model_trainer.get_model_type_name()
        self.logger.info(f"Training Model B: PCM with similar proteins using {trainer_name}")
        
        results_by_threshold = {}
        
        for threshold in ['high', 'medium', 'low']:
            self.logger.info(f"\n--- Processing {threshold.upper()} threshold ---")
            
            # Note: Status for starting Model B (high) is already displayed in process_single_protein
            # Only display status updates for medium and low thresholds when starting
            if current_idx and total and threshold != 'high':
                # Build status with Model A and completed Model B thresholds
                status_parts = []
                if model_a_f1 is not None:
                    status_parts.append(f"✓ Model A completed (F1: {model_a_f1:.2f})")
                
                # Add completed Model B thresholds
                for t in ['high', 'medium']:
                    if t in results_by_threshold:
                        t_result = results_by_threshold[t]
                        if t_result.get('status') == 'complete' and 'test_f1_macro' in t_result:
                            t_f1 = t_result['test_f1_macro']
                            status_parts.append(f"✓ Model B ({t}) completed (F1: {t_f1:.2f})")
                
                status_parts.append(f"→ Training Model B ({threshold} threshold)...")
                status_msg = f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  {'  '.join(status_parts)}"
                self._print_status(status_msg)
            
            # Get similar proteins for this threshold
            similar_proteins = self.similarity_loader.get_similar_proteins_for_threshold(uniprot_id, threshold)
            
            if not similar_proteins:
                self.logger.warning(f"No similar proteins at {threshold} threshold")
                results_by_threshold[threshold] = {
                    'status': 'no_similar_proteins',
                    'n_similar': 0
                }
                continue
            
            # Load bioactivity data
            similar_data = self.bioactivity_loader.get_filtered_papyrus(similar_proteins)
            target_data = self.bioactivity_loader.get_filtered_papyrus([uniprot_id])
            
            total_samples = len(similar_data) + len(target_data)
            if total_samples < 30:
                self.logger.warning(f"Insufficient data at {threshold}: {total_samples} samples")
                results_by_threshold[threshold] = {
                    'status': 'insufficient_data',
                    'n_samples': total_samples,
                    'n_similar_proteins': len(similar_proteins)
                }
                continue
            
            self.logger.info(f"Similar proteins data: {len(similar_data):,} activities")
            self.logger.info(f"Target protein data: {len(target_data):,} activities")
            
            # Prepare data using the data preparation module
            from ..workflow.data_preparation import prepare_model_b_data
            
            train_features, test_features = prepare_model_b_data(
                similar_data=similar_data,
                target_data=target_data,
                uniprot_id=uniprot_id,
                thresholds=thresholds,
                model_trainer=self.model_trainer,
                cached_fingerprints=self._cached_fingerprints,
                bioactivity_loader=self.bioactivity_loader
            )
            
            if train_features is None or test_features is None:
                results_by_threshold[threshold] = {
                    'status': 'insufficient_features',
                    'n_train': len(train_features) if train_features is not None else 0,
                    'n_test': len(test_features) if test_features is not None else 0
                }
                continue
            
            if len(train_features) < 30 or len(test_features) < 6:
                self.logger.warning(f"Insufficient data after feature extraction")
                results_by_threshold[threshold] = {
                    'status': 'insufficient_features',
                    'n_train': len(train_features),
                    'n_test': len(test_features)
                }
                continue
            
            self.logger.info(f"Train: {len(train_features)}, Test: {len(test_features)}")
            self.logger.info(f"Class distribution - Train: {train_features['class'].value_counts().to_dict()}")
            self.logger.info(f"Class distribution - Test: {test_features['class'].value_counts().to_dict()}")
            
            # Train model using modular trainer
            train_results = train_model_with_reporting(
                model_trainer=self.model_trainer,
                model_reporter=self.model_reporter,
                train_df=train_features,
                test_df=test_features,
                model_type='B',
                protein_name=protein_name,
                threshold=threshold,
                uniprot_id=uniprot_id,
                thresholds=thresholds,
                mlflow_enabled=self.mlflow_enabled,
                mlflow_experiment_name=self.mlflow_experiment_name,
                output_dir=self.output_dir,
                cached_fingerprints=self._cached_fingerprints,
                bioactivity_loader=self.bioactivity_loader
            )
            
            threshold_result = {
                'status': 'complete',
                'n_train': len(train_features),
                'n_test': len(test_features),
                'n_similar_proteins': len(similar_proteins),
                'n_similar_activities': len(similar_data),
                'n_target_activities': len(target_data),
                'similar_proteins': similar_proteins,
                'n_features': train_results.get('n_features', 'N/A'),
                **train_results
            }
            
            # Display completion status for this threshold
            if current_idx and total and train_results.get('status') == 'success' and 'test_f1_macro' in train_results:
                f1_score = train_results['test_f1_macro']
                
                # Build status message showing Model A (if exists) and completed Model B thresholds
                status_parts = []
                
                # Add Model A completion if available
                if model_a_f1 is not None:
                    status_parts.append(f"✓ Model A completed (F1: {model_a_f1:.2f})")
                
                # Add all completed Model B thresholds including current one
                for t in ['high', 'medium', 'low']:
                    if t == threshold:
                        status_parts.append(f"✓ Model B ({t}) completed (F1: {f1_score:.2f})")
                        break
                    elif t in results_by_threshold:
                        t_result = results_by_threshold[t]
                        if t_result.get('status') == 'complete' and 'test_f1_macro' in t_result:
                            t_f1 = t_result['test_f1_macro']
                            status_parts.append(f"✓ Model B ({t}) completed (F1: {t_f1:.2f})")
                
                # Determine next threshold to show
                threshold_order = ['high', 'medium', 'low']
                current_idx_in_order = threshold_order.index(threshold)
                if current_idx_in_order < len(threshold_order) - 1:
                    next_threshold = threshold_order[current_idx_in_order + 1]
                    status_parts.append(f"→ Training Model B ({next_threshold} threshold)...")
                
                status_msg = f"[{current_idx}/{total}] Processing: {protein_name} ({uniprot_id})  {'  '.join(status_parts)}"
                self._print_status(status_msg)
            
            results_by_threshold[threshold] = threshold_result
        
        return {
            'protein': protein_name,
            'uniprot_id': uniprot_id,
            'model_type': 'B',
            'thresholds': results_by_threshold
        }
    
    def _save_results(self):
        """Save workflow results to file and generate summary report"""
        results_file = self.output_dir / "workflow_results.csv"
        
        # Flatten results for CSV
        csv_data = []
        for result in self.results:
            protein = result.get('protein', '')
            uniprot_id = result.get('uniprot_id', '')
            
            # Check if this is a combined result (has both model_a and model_b)
            if 'model_a' in result and 'model_b' in result:
                # Combined result: protein with similar proteins
                has_similar = result.get('has_similar_proteins', False)
                n_similar = result.get('n_similar_proteins', 0)
                
                # Add Model A result
                model_a = result['model_a']
                csv_data.append({
                    'protein': protein,
                    'uniprot_id': uniprot_id,
                    'model_type': 'A',
                    'status': model_a.get('status', ''),
                    'n_samples': model_a.get('n_samples', ''),
                    'n_train_samples': model_a.get('train_samples', 'N/A'),
                    'n_test_samples': model_a.get('test_samples', 'N/A'),
                    'threshold': '',
                    'n_similar_activities': '',
                    'n_target_activities': '',
                    'n_similar_proteins': '',
                    'has_similar_proteins': has_similar,
                    'n_total_similar_proteins': n_similar,
                    'n_features': model_a.get('n_features', 'N/A'),
                    'similar_proteins': model_a.get('similar_proteins', 'N/A')
                })
                
                # Add Model B results (one per threshold)
                model_b = result['model_b']
                if 'thresholds' in model_b:
                    for threshold, threshold_data in model_b['thresholds'].items():
                        csv_data.append({
                            'protein': protein,
                            'uniprot_id': uniprot_id,
                            'model_type': 'B',
                            'status': threshold_data.get('status', ''),
                            'n_samples': threshold_data.get('n_train', '') + threshold_data.get('n_test', '') if isinstance(threshold_data.get('n_train', ''), (int, float)) and isinstance(threshold_data.get('n_test', ''), (int, float)) else '',
                            'n_train_samples': threshold_data.get('n_train', 'N/A'),
                            'n_test_samples': threshold_data.get('n_test', 'N/A'),
                            'threshold': threshold,
                            'n_similar_activities': threshold_data.get('n_similar_activities', ''),
                            'n_target_activities': threshold_data.get('n_target_activities', ''),
                            'n_similar_proteins': threshold_data.get('n_similar_proteins', ''),
                            'has_similar_proteins': has_similar,
                            'n_total_similar_proteins': n_similar,
                            'n_features': threshold_data.get('n_features', 'N/A'),
                            'similar_proteins': threshold_data.get('similar_proteins', 'N/A')
                        })
            else:
                # Single model result: protein without similar proteins (Model A only)
                csv_data.append({
                    'protein': protein,
                    'uniprot_id': uniprot_id,
                    'model_type': result.get('model_type', ''),
                    'status': result.get('status', ''),
                    'n_samples': result.get('n_samples', ''),
                    'n_train_samples': result.get('train_samples', 'N/A'),
                    'n_test_samples': result.get('test_samples', 'N/A'),
                    'threshold': '',
                    'n_similar_activities': '',
                    'n_target_activities': '',
                    'n_similar_proteins': '',
                    'has_similar_proteins': False,
                    'n_total_similar_proteins': 0,
                    'n_features': result.get('n_features', 'N/A'),
                    'similar_proteins': result.get('similar_proteins', 'N/A')
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(results_file, index=False)
        self.logger.info(f"Saved results to {results_file}")
        
        # Generate comprehensive summary report
        summary_path = self.model_reporter.generate_summary_report(df)
        self.logger.info(f"Generated summary report: {summary_path}")
        
        # Generate comprehensive protein report
        comprehensive_path = self.model_reporter.generate_comprehensive_report(df)
        self.logger.info(f"Generated comprehensive report: {comprehensive_path}")
        
        # Generate individual model reports
        individual_reports = self.model_reporter.generate_individual_model_reports(df)
        self.logger.info(f"Generated {len(individual_reports)} individual model reports")
