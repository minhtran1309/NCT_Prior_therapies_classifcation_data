"""
Evaluation utilities for clinical trial prior therapy classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve
)
from pathlib import Path

from ..models.clinical_trial_data import (
    ClinicalTrialDataset, ClassificationResult, EvaluationMetrics, TherapyType
)
from ..classification.prior_therapy_classifier import BaseClassifier


class ModelEvaluator:
    """Comprehensive evaluation of prior therapy classification models."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def evaluate_model(self, 
                      classifier: BaseClassifier,
                      test_dataset: ClinicalTrialDataset,
                      save_results: bool = True) -> EvaluationMetrics:
        """
        Comprehensive evaluation of a classification model.
        """
        predictions = classifier.predict_dataset(test_dataset)
        
        # Extract true labels and predictions
        y_true = [sentence.onco_annotation.has_prior_therapy for sentence in test_dataset.sentences]
        y_pred = [pred.predicted_has_prior_therapy for pred in predictions]
        y_proba = [pred.confidence_score for pred in predictions]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create evaluation metrics object
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm.tolist(),
            classification_report=report
        )
        
        if save_results:
            self._save_evaluation_results(classifier.model_name, metrics, y_true, y_pred, y_proba)
        
        return metrics
    
    def compare_models(self, 
                      models_and_datasets: List[tuple],
                      save_results: bool = True) -> pd.DataFrame:
        """
        Compare multiple models on the same test dataset.
        
        Args:
            models_and_datasets: List of tuples (classifier, test_dataset, model_display_name)
        """
        results = []
        
        for classifier, test_dataset, display_name in models_and_datasets:
            metrics = self.evaluate_model(classifier, test_dataset, save_results=False)
            
            results.append({
                'Model': display_name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1-Score': metrics.f1_score
            })
        
        comparison_df = pd.DataFrame(results)
        
        if save_results:
            self._save_model_comparison(comparison_df)
        
        return comparison_df
    
    def analyze_errors(self, 
                      classifier: BaseClassifier,
                      test_dataset: ClinicalTrialDataset,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze classification errors to identify patterns.
        """
        predictions = classifier.predict_dataset(test_dataset)
        
        error_analysis = {
            'false_positives': [],
            'false_negatives': [],
            'correct_predictions': []
        }
        
        for sentence, prediction in zip(test_dataset.sentences, predictions):
            true_label = sentence.onco_annotation.has_prior_therapy
            pred_label = prediction.predicted_has_prior_therapy
            
            error_info = {
                'nct_id': sentence.nct_id,
                'sentence_id': sentence.sentence_id,
                'sentence_text': sentence.sentence_text,
                'section': sentence.section.value,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': prediction.confidence_score,
                'therapy_type': sentence.onco_annotation.therapy_type.value if sentence.onco_annotation.therapy_type else None,
                'therapy_name': sentence.onco_annotation.therapy_name
            }
            
            if true_label and not pred_label:
                error_analysis['false_negatives'].append(error_info)
            elif not true_label and pred_label:
                error_analysis['false_positives'].append(error_info)
            else:
                error_analysis['correct_predictions'].append(error_info)
        
        if save_results:
            self._save_error_analysis(classifier.model_name, error_analysis)
        
        return error_analysis
    
    def evaluate_by_therapy_type(self, 
                                classifier: BaseClassifier,
                                test_dataset: ClinicalTrialDataset) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate model performance by therapy type.
        """
        therapy_results = {}
        
        # Group sentences by therapy type
        therapy_groups = {}
        for sentence in test_dataset.sentences:
            if sentence.onco_annotation.therapy_type:
                therapy_type = sentence.onco_annotation.therapy_type.value
                if therapy_type not in therapy_groups:
                    therapy_groups[therapy_type] = []
                therapy_groups[therapy_type].append(sentence)
        
        # Evaluate each therapy type separately
        for therapy_type, sentences in therapy_groups.items():
            if len(sentences) < 5:  # Skip therapy types with too few examples
                continue
            
            # Create subset dataset
            subset_dataset = ClinicalTrialDataset(
                dataset_name=f"subset_{therapy_type}",
                dataset_version="1.0",
                sentences=sentences
            )
            
            # Evaluate
            metrics = self.evaluate_model(classifier, subset_dataset, save_results=False)
            therapy_results[therapy_type] = metrics
        
        return therapy_results
    
    def _save_evaluation_results(self, 
                               model_name: str,
                               metrics: EvaluationMetrics,
                               y_true: List[bool],
                               y_pred: List[bool],
                               y_proba: List[float]) -> None:
        """Save evaluation results to files."""
        
        # Save metrics to JSON
        metrics_file = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            import json
            json.dump(metrics.dict(), f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(metrics.confusion_matrix, model_name)
        
        # Plot ROC curve if probabilities are available
        if len(set(y_proba)) > 1:  # Check if probabilities vary
            self._plot_roc_curve(y_true, y_proba, model_name)
            self._plot_precision_recall_curve(y_true, y_proba, model_name)
    
    def _save_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Save model comparison results."""
        comparison_file = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
    
    def _save_error_analysis(self, model_name: str, error_analysis: Dict[str, Any]) -> None:
        """Save error analysis results."""
        
        # Save to CSV files
        for error_type, errors in error_analysis.items():
            if errors:
                df = pd.DataFrame(errors)
                error_file = self.output_dir / f"{model_name}_{error_type}.csv"
                df.to_csv(error_file, index=False)
    
    def _plot_confusion_matrix(self, cm: List[List[int]], model_name: str) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Prior Therapy', 'Prior Therapy'],
                   yticklabels=['No Prior Therapy', 'Prior Therapy'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_file = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, y_true: List[bool], y_proba: List[float], model_name: str) -> None:
        """Plot ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
            plot_file = self.output_dir / f"{model_name}_roc_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not plot ROC curve for {model_name}: {e}")
    
    def _plot_precision_recall_curve(self, y_true: List[bool], y_proba: List[float], model_name: str) -> None:
        """Plot Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            
            plot_file = self.output_dir / f"{model_name}_pr_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not plot PR curve for {model_name}: {e}")
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Plot model comparison."""
        plt.figure(figsize=(12, 8))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, comparison_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 1.5, comparison_df['Model'], rotation=45)
        plt.legend()
        plt.ylim([0, 1])
        
        plot_file = self.output_dir / "model_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()


def generate_evaluation_report(evaluator: ModelEvaluator, 
                             classifier: BaseClassifier,
                             test_dataset: ClinicalTrialDataset) -> str:
    """Generate a comprehensive evaluation report."""
    
    # Run evaluation
    metrics = evaluator.evaluate_model(classifier, test_dataset)
    error_analysis = evaluator.analyze_errors(classifier, test_dataset)
    therapy_metrics = evaluator.evaluate_by_therapy_type(classifier, test_dataset)
    
    # Generate report
    report = f"""
# Clinical Trial Prior Therapy Classification Evaluation Report

## Model: {classifier.model_name}

## Overall Performance
- **Accuracy**: {metrics.accuracy:.3f}
- **Precision**: {metrics.precision:.3f}
- **Recall**: {metrics.recall:.3f}
- **F1-Score**: {metrics.f1_score:.3f}

## Confusion Matrix
- True Negatives: {metrics.confusion_matrix[0][0]}
- False Positives: {metrics.confusion_matrix[0][1]}
- False Negatives: {metrics.confusion_matrix[1][0]}
- True Positives: {metrics.confusion_matrix[1][1]}

## Error Analysis
- **False Positives**: {len(error_analysis['false_positives'])} sentences
- **False Negatives**: {len(error_analysis['false_negatives'])} sentences
- **Correct Predictions**: {len(error_analysis['correct_predictions'])} sentences

## Performance by Therapy Type
"""
    
    for therapy_type, therapy_metrics in therapy_metrics.items():
        report += f"""
### {therapy_type.replace('_', ' ').title()}
- Accuracy: {therapy_metrics.accuracy:.3f}
- Precision: {therapy_metrics.precision:.3f}
- Recall: {therapy_metrics.recall:.3f}
- F1-Score: {therapy_metrics.f1_score:.3f}
"""
    
    # Save report
    report_file = evaluator.output_dir / f"{classifier.model_name}_evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    return report