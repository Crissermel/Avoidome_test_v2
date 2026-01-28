"""
Model reporting utilities for AQSE workflow.

This module provides comprehensive model reporting including metrics, predictions, and model saving.
"""

import pandas as pd
import numpy as np
import pickle
import gzip
import json
import csv
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class ModelReporter:
    """Handles comprehensive model reporting including metrics, predictions, and model saving"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.results_dir / "models"
        self.metrics_dir = self.results_dir / "metrics"
        self.predictions_dir = self.results_dir / "predictions"
        self.distributions_dir = self.results_dir / "class_distributions"
        
        for dir_path in [self.models_dir, self.metrics_dir, self.predictions_dir, self.distributions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_model(self, model, protein_name: str, uniprot_id: str, model_type: str, threshold: Optional[str] = None) -> str:
        """Save trained model in compressed format"""
        # Create filename
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_model.pkl.gz"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_model.pkl.gz"
        
        filepath = self.models_dir / filename
        
        try:
            # Handle different model types
            # For PyTorch models, save state_dict; for scikit-learn models, save the full model
            if hasattr(model, 'state_dict'):
                # PyTorch model - save state_dict
                model_data = {
                    'state_dict': model.state_dict(),
                    'model_config': {
                        'protein_name': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': model_type,
                        'threshold': threshold,
                        'timestamp': datetime.now().isoformat(),
                        'model_class': 'pytorch'
                    }
                }
            else:
                # Scikit-learn or other models - save the full model
                model_data = {
                    'model': model,
                    'model_config': {
                        'protein_name': protein_name,
                        'uniprot_id': uniprot_id,
                        'model_type': model_type,
                        'threshold': threshold,
                        'timestamp': datetime.now().isoformat(),
                        'model_class': 'sklearn' if hasattr(model, 'predict') and hasattr(model, 'fit') else 'other'
                    }
                }
            
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save model {filename}: {e}")
            return ""
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], y_prob: Optional[np.ndarray] = None, class_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Use provided class_labels or default to 3-class
        if class_labels is None:
            class_labels = ['Low', 'Medium', 'High']
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class labels
        metrics['class_labels'] = class_labels
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any], protein_name: str, uniprot_id: str, 
                    model_type: str, threshold: Optional[str] = None, split: str = 'test') -> str:
        """Save metrics to JSON file"""
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_{split}_metrics.json"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{split}_metrics.json"
        
        filepath = self.metrics_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Metrics saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics {filename}: {e}")
            return ""
    
    def save_class_distribution(self, df: pd.DataFrame, protein_name: str, uniprot_id: str, 
                              model_type: str, threshold: Optional[str] = None, split: str = 'all') -> str:
        """Save class distribution analysis"""
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{split}_distribution.csv"
        
        filepath = self.distributions_dir / filename
        
        try:
            # Calculate distribution
            class_counts = df['class'].value_counts()
            class_proportions = df['class'].value_counts(normalize=True)
            
            distribution_df = pd.DataFrame({
                'class': class_counts.index,
                'count': class_counts.values,
                'proportion': class_proportions.values
            })
            
            # Add summary statistics
            summary_stats = {
                'total_samples': int(len(df)),
                'n_classes': int(len(class_counts)),
                'class_balance_ratio': float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0,
                'most_common_class': str(class_counts.index[0]) if len(class_counts) > 0 else '',
                'most_common_count': int(class_counts.iloc[0]) if len(class_counts) > 0 else 0,
                'least_common_class': str(class_counts.index[-1]) if len(class_counts) > 0 else '',
                'least_common_count': int(class_counts.iloc[-1]) if len(class_counts) > 0 else 0
            }
            
            # Save distribution
            distribution_df.to_csv(filepath, index=False)
            
            # Save summary stats
            summary_filepath = filepath.with_suffix('.summary.json')
            with open(summary_filepath, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            self.logger.info(f"Class distribution saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save class distribution {filename}: {e}")
            return ""
    
    def save_predictions(self, df: pd.DataFrame, y_pred: List[int], protein_name: str, uniprot_id: str, model_type: str, 
                        y_prob: Optional[np.ndarray] = None, threshold: Optional[str] = None, split: str = 'test', 
                        class_labels: Optional[List[str]] = None) -> str:
        """Save predictions with probabilities"""
        if threshold:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{threshold}_{split}_predictions.csv"
        else:
            filename = f"{protein_name}_{uniprot_id}_{model_type}_{split}_predictions.csv"
        
        filepath = self.predictions_dir / filename
        
        # Use provided class_labels or default to 3-class
        if class_labels is None:
            class_labels = ['Low', 'Medium', 'High']
        
        try:
            # Create predictions DataFrame
            predictions_df = df.copy()
            predictions_df['predicted_class'] = [class_labels[i] for i in y_pred]
            predictions_df['predicted_class_int'] = y_pred
            
            # Add probabilities if available (dynamic based on n_classes)
            # Check that y_prob shape matches number of class_labels
            if y_prob is not None:
                n_prob_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 1
                n_labels = len(class_labels)
                
                # Only add probabilities for classes that exist in y_prob
                for i, label in enumerate(class_labels):
                    prob_col_name = f'prob_{label.lower().replace("+", "_")}'
                    if i < n_prob_classes:
                        predictions_df[prob_col_name] = y_prob[:, i]
                    else:
                        # If model has fewer classes than labels, set to 0
                        predictions_df[prob_col_name] = 0.0
                
                predictions_df['max_prob'] = np.max(y_prob, axis=1)
                predictions_df['confidence'] = predictions_df['max_prob']
            
            # Save predictions
            predictions_df.to_csv(filepath, index=False)
            
            self.logger.info(f"Predictions saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions {filename}: {e}")
            return ""
    
    def generate_comprehensive_report(self, workflow_results: pd.DataFrame) -> str:
        """Generate comprehensive CSV report with all protein information per threshold"""
        comprehensive_filepath = self.results_dir / "comprehensive_protein_report.csv"
        
        try:
            report_data = []
            
            # Process each protein result
            for _, result in workflow_results.iterrows():
                protein = result.get('protein', '')
                uniprot_id = result.get('uniprot_id', '')
                model_type = result.get('model_type', '')
                threshold = result.get('threshold', '')
                status = result.get('status', '')
                
                # Create model_key
                model_key = f"{protein}_{uniprot_id}_{model_type}"
                if threshold and threshold != 'N/A':
                    model_key += f"_{threshold}"
                
                # Base information
                base_info = {
                    'model_key': model_key,
                    'protein': protein,
                    'uniprot_id': uniprot_id,
                    'model_type': model_type,
                    'threshold': threshold if threshold else 'N/A',
                    'qsar_model_run': 'Yes' if status == 'success' else 'No',
                    'status': status
                }
                
                # Load detailed metrics if model was successful
                if status == 'success':
                    # Try to load metrics file
                    metrics_file = self._find_metrics_file(protein, uniprot_id, model_type, threshold)
                    if metrics_file and metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            
                            base_info.update({
                                'accuracy': metrics.get('accuracy', 'N/A'),
                                'f1_macro': metrics.get('f1_macro', 'N/A'),
                                'f1_weighted': metrics.get('f1_weighted', 'N/A'),
                                'precision_macro': metrics.get('precision_macro', 'N/A'),
                                'recall_macro': metrics.get('recall_macro', 'N/A'),
                                'confusion_matrix': str(metrics.get('confusion_matrix', 'N/A'))
                            })
                        except Exception as e:
                            self.logger.warning(f"Could not load metrics for {protein}: {e}")
                            base_info.update({
                                'accuracy': 'Error loading',
                                'f1_macro': 'Error loading',
                                'f1_weighted': 'Error loading',
                                'precision_macro': 'Error loading',
                                'recall_macro': 'Error loading',
                                'confusion_matrix': 'Error loading'
                            })
                    else:
                        base_info.update({
                            'accuracy': 'Metrics file not found',
                            'f1_macro': 'Metrics file not found',
                            'f1_weighted': 'Metrics file not found',
                            'precision_macro': 'Metrics file not found',
                            'recall_macro': 'Metrics file not found',
                            'confusion_matrix': 'Metrics file not found'
                        })
                else:
                    base_info.update({
                        'accuracy': 'N/A',
                        'f1_macro': 'N/A',
                        'f1_weighted': 'N/A',
                        'precision_macro': 'N/A',
                        'recall_macro': 'N/A',
                        'confusion_matrix': 'N/A'
                    })
                
                # Add sample counts and class distribution
                # Support both Model A (train_samples/test_samples) and Model B (n_train/n_test)
                n_train_samples = result.get('train_samples') if pd.notna(result.get('train_samples', float('nan'))) else result.get('n_train', 'N/A')
                n_test_samples = result.get('test_samples') if pd.notna(result.get('test_samples', float('nan'))) else result.get('n_test', 'N/A')
                n_target_points = result.get('n_samples') if pd.notna(result.get('n_samples', float('nan'))) else result.get('n_target_activities', 'N/A')

                base_info.update({
                    'n_target_bioactivity_datapoints': n_target_points,
                    'n_expanded_proteins': result.get('n_similar_proteins', 'N/A'),
                    'n_expanded_bioactivity_datapoints': result.get('n_similar_activities', 'N/A'),
                    'n_train_samples': n_train_samples,
                    'n_test_samples': n_test_samples,
                    'train_class_distribution': 'N/A',  # Will be filled below
                    'test_class_distribution': 'N/A',   # Will be filled below
                    'n_features': 'N/A',               # Will be filled below
                    'similar_proteins': 'N/A'           # Will be filled below
                })
                
                # Load class distribution if available
                class_dist_file = self._find_class_distribution_file(protein, uniprot_id, model_type, threshold, 'test')
                if class_dist_file and class_dist_file.exists():
                    try:
                        dist_df = pd.read_csv(class_dist_file)
                        class_dist = {}
                        for _, row in dist_df.iterrows():
                            class_dist[row['class']] = f"{row['count']} ({row['proportion']:.3f})"
                        base_info['test_class_distribution'] = str(class_dist)
                    except Exception as e:
                        self.logger.warning(f"Could not load class distribution for {protein}: {e}")
                        base_info['test_class_distribution'] = 'Error loading'
                else:
                    base_info['test_class_distribution'] = 'N/A'
                
                # Load train class distribution
                train_class_dist_file = self._find_class_distribution_file(protein, uniprot_id, model_type, threshold, 'train')
                if train_class_dist_file and train_class_dist_file.exists():
                    try:
                        dist_df = pd.read_csv(train_class_dist_file)
                        class_dist = {}
                        for _, row in dist_df.iterrows():
                            class_dist[row['class']] = f"{row['count']} ({row['proportion']:.3f})"
                        base_info['train_class_distribution'] = str(class_dist)
                    except Exception as e:
                        self.logger.warning(f"Could not load train class distribution for {protein}: {e}")
                        base_info['train_class_distribution'] = 'Error loading'
                else:
                    base_info['train_class_distribution'] = 'N/A'
                
                # Add n_features and similar_proteins from workflow results
                base_info['n_features'] = result.get('n_features', 'N/A')
                base_info['similar_proteins'] = result.get('similar_proteins', 'N/A')
                
                report_data.append(base_info)
            
            # Create DataFrame and save
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(comprehensive_filepath, index=False)
            
            self.logger.info(f"Comprehensive report saved: {comprehensive_filepath}")
            return str(comprehensive_filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return ""
    
    def _find_metrics_file(self, protein: str, uniprot_id: str, model_type: str, threshold: str) -> Optional[Path]:
        """Find metrics file for a given model"""
        # Try different naming patterns
        patterns = []
        
        if threshold and threshold != 'N/A':
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_{threshold}_test_metrics.json",
                f"{protein}_{protein}_{model_type}_{threshold}_test_metrics.json",
                f"{uniprot_id}_{uniprot_id}_{model_type}_{threshold}_test_metrics.json"
            ]
        else:
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_test_metrics.json",
                f"{protein}_{protein}_{model_type}_test_metrics.json",
                f"{uniprot_id}_{uniprot_id}_{model_type}_test_metrics.json"
            ]
        
        # Try each pattern
        for pattern in patterns:
            filepath = self.metrics_dir / pattern
            if filepath.exists():
                return filepath
        
        # If no file found, return the first pattern (for error reporting)
        return self.metrics_dir / patterns[0]
    
    def _find_class_distribution_file(self, protein: str, uniprot_id: str, model_type: str, threshold: str, split: str) -> Optional[Path]:
        """Find class distribution file for a given model"""
        # Try different naming patterns
        patterns = []
        
        if threshold and threshold != 'N/A':
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv",
                f"{protein}_{protein}_{model_type}_{threshold}_{split}_distribution.csv",
                f"{uniprot_id}_{uniprot_id}_{model_type}_{threshold}_{split}_distribution.csv"
            ]
        else:
            patterns = [
                f"{protein}_{uniprot_id}_{model_type}_{split}_distribution.csv",
                f"{protein}_{protein}_{model_type}_{split}_distribution.csv",
                f"{uniprot_id}_{uniprot_id}_{model_type}_{split}_distribution.csv"
            ]
        
        # Try each pattern
        for pattern in patterns:
            filepath = self.distributions_dir / pattern
            if filepath.exists():
                return filepath
        
        # If no file found, return the first pattern (for error reporting)
        return self.distributions_dir / patterns[0]
    
    def generate_individual_model_reports(self, workflow_results: pd.DataFrame) -> List[str]:
        """Generate individual CSV and JSON reports for each successful model"""
        individual_reports = []
        
        successful_models = workflow_results[workflow_results['status'] == 'success']
        
        for _, result in successful_models.iterrows():
            protein = result.get('protein', '')
            uniprot_id = result.get('uniprot_id', '')
            model_type = result.get('model_type', '')
            threshold = result.get('threshold', '')
            
            try:
                # Create individual report directory
                model_dir = self.results_dir / "individual_reports" / f"{protein}_{uniprot_id}_{model_type}"
                if threshold and threshold != 'N/A':
                    model_dir = model_dir.parent / f"{protein}_{uniprot_id}_{model_type}_{threshold}"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Collect all data for this model
                model_data = {
                    'model_info': {
                        'protein': protein,
                        'uniprot_id': uniprot_id,
                        'model_type': model_type,
                        'threshold': threshold if threshold else 'N/A',
                        'generation_timestamp': datetime.now().isoformat()
                    },
                    'workflow_results': result.to_dict(),
                    'metrics': {},
                    'class_distributions': {},
                    'predictions_summary': {}
                }
                
                # Load metrics
                metrics_file = self._find_metrics_file(protein, uniprot_id, model_type, threshold)
                if metrics_file and metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        model_data['metrics'] = json.load(f)
                
                # Load class distributions
                for split in ['train', 'test']:
                    dist_file = self._find_class_distribution_file(protein, uniprot_id, model_type, threshold, split)
                    if dist_file and dist_file.exists():
                        dist_df = pd.read_csv(dist_file)
                        model_data['class_distributions'][split] = dist_df.to_dict('records')
                        
                        # Load summary if available
                        summary_file = dist_file.with_suffix('.summary.json')
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                model_data['class_distributions'][f'{split}_summary'] = json.load(f)
                
                # Load predictions summary
                pred_file = self._find_predictions_file(protein, uniprot_id, model_type, threshold)
                if pred_file and pred_file.exists():
                    pred_df = pd.read_csv(pred_file)
                    model_data['predictions_summary'] = {
                        'total_predictions': len(pred_df),
                        'prediction_distribution': pred_df['predicted_class'].value_counts().to_dict(),
                        'confidence_stats': {
                            'mean': pred_df['confidence'].mean() if 'confidence' in pred_df.columns else 'N/A',
                            'std': pred_df['confidence'].std() if 'confidence' in pred_df.columns else 'N/A',
                            'min': pred_df['confidence'].min() if 'confidence' in pred_df.columns else 'N/A',
                            'max': pred_df['confidence'].max() if 'confidence' in pred_df.columns else 'N/A'
                        }
                    }
                
                # Save JSON report
                json_file = model_dir / "model_report.json"
                with open(json_file, 'w') as f:
                    json.dump(model_data, f, indent=2)
                
                # Save CSV summary
                csv_data = []
                csv_data.append(['Model Information', ''])
                csv_data.append(['Protein', protein])
                csv_data.append(['UniProt ID', uniprot_id])
                csv_data.append(['Model Type', model_type])
                csv_data.append(['Threshold', threshold if threshold else 'N/A'])
                csv_data.append(['Generation Time', datetime.now().isoformat()])
                csv_data.append(['', ''])
                
                csv_data.append(['Performance Metrics', ''])
                if model_data['metrics']:
                    csv_data.append(['Accuracy', model_data['metrics'].get('accuracy', 'N/A')])
                    csv_data.append(['F1-Macro', model_data['metrics'].get('f1_macro', 'N/A')])
                    csv_data.append(['F1-Weighted', model_data['metrics'].get('f1_weighted', 'N/A')])
                    csv_data.append(['Precision-Macro', model_data['metrics'].get('precision_macro', 'N/A')])
                    csv_data.append(['Recall-Macro', model_data['metrics'].get('recall_macro', 'N/A')])
                csv_data.append(['', ''])
                
                csv_data.append(['Data Summary', ''])
                csv_data.append(['Target Bioactivity Points', result.get('n_samples', 'N/A')])
                csv_data.append(['Train Samples', result.get('train_samples', 'N/A')])
                csv_data.append(['Test Samples', result.get('test_samples', 'N/A')])
                csv_data.append(['Expanded Proteins', result.get('n_similar_proteins', 'N/A')])
                csv_data.append(['Expanded Bioactivity Points', result.get('n_similar_activities', 'N/A')])
                
                # Save CSV
                csv_file = model_dir / "model_summary.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_data)
                
                individual_reports.append(str(json_file))
                individual_reports.append(str(csv_file))
                
                self.logger.info(f"Individual report saved: {model_dir}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate individual report for {protein}: {e}")
        
        return individual_reports
    
    def _find_predictions_file(self, protein: str, uniprot_id: str, model_type: str, threshold: str) -> Optional[Path]:
        """Find predictions file for a given model"""
        if threshold and threshold != 'N/A':
            filename = f"{protein}_{uniprot_id}_{model_type}_{threshold}_test_predictions.csv"
        else:
            filename = f"{protein}_{uniprot_id}_{model_type}_test_predictions.csv"
        return self.predictions_dir / filename
    
    def generate_summary_report(self, workflow_results: pd.DataFrame) -> str:
        summary_filepath = self.results_dir / "model_summary_report.json"
        
        try:
            # Calculate overall statistics
            total_models = len(workflow_results)
            successful_models = len(workflow_results[workflow_results['status'] == 'success'])
            failed_models = total_models - successful_models
            
            # Model type breakdown
            model_a_count = len(workflow_results[workflow_results['model_type'] == 'A'])
            model_b_count = len(workflow_results[workflow_results['model_type'] == 'B'])
            
            # Status breakdown
            status_counts = workflow_results['status'].value_counts().to_dict()
            
            # Threshold breakdown for Model B
            model_b_results = workflow_results[workflow_results['model_type'] == 'B']
            threshold_counts = model_b_results['threshold'].value_counts().to_dict() if 'threshold' in model_b_results.columns else {}
            
            summary = {
                'generation_timestamp': datetime.now().isoformat(),
                'total_models_attempted': total_models,
                'successful_models': successful_models,
                'failed_models': failed_models,
                'success_rate': successful_models / total_models if total_models > 0 else 0,
                'model_type_breakdown': {
                    'model_a': model_a_count,
                    'model_b': model_b_count
                },
                'status_breakdown': status_counts,
                'model_b_threshold_breakdown': threshold_counts,
                'results_directory': str(self.results_dir),
                'subdirectories': {
                    'models': str(self.models_dir),
                    'metrics': str(self.metrics_dir),
                    'predictions': str(self.predictions_dir),
                    'class_distributions': str(self.distributions_dir)
                }
            }
            
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Summary report saved: {summary_filepath}")
            return str(summary_filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return ""
