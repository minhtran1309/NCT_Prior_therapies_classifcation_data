"""
Main pipeline for clinical trial prior therapy classification.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.utils.config_utils import load_config, setup_logging, create_directories
from src.data_processing.data_loader import DataLoader, TextPreprocessor, DataSplitter
from src.classification.prior_therapy_classifier import (
    TfidfClassifier, RuleBasedClassifier, EnsembleClassifier, evaluate_classifier
)
from src.evaluation.model_evaluator import ModelEvaluator, generate_evaluation_report
from src.models.clinical_trial_data import ClinicalTrialDataset


class PriorTherapyClassificationPipeline:
    """Main pipeline for prior therapy classification."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        create_directories(self.config)
        
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor(
            lowercase=self.config['data_processing']['preprocessing']['lowercase'],
            remove_special_chars=self.config['data_processing']['preprocessing']['remove_special_chars']
        )
        self.evaluator = ModelEvaluator(self.config['evaluation']['output_dir'])
        
        self.logger.info("Pipeline initialized")
    
    def load_and_preprocess_data(self, data_path: str) -> ClinicalTrialDataset:
        """Load and preprocess the dataset."""
        self.logger.info(f"Loading data from {data_path}")
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            dataset = self.data_loader.load_from_csv(data_path)
        elif data_path.endswith('.json'):
            dataset = self.data_loader.load_from_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        self.logger.info(f"Loaded {len(dataset.sentences)} sentences from {len(set(s.nct_id for s in dataset.sentences))} trials")
        
        # Preprocess data
        if self.config['data_processing']['preprocessing']['lowercase'] or \
           self.config['data_processing']['preprocessing']['remove_special_chars']:
            self.logger.info("Preprocessing text data")
            dataset = self.preprocessor.preprocess_dataset(dataset)
        
        return dataset
    
    def split_data(self, dataset: ClinicalTrialDataset) -> Tuple[ClinicalTrialDataset, ClinicalTrialDataset, ClinicalTrialDataset]:
        """Split dataset into train, validation, and test sets."""
        split_config = self.config['data_processing']['data_split']
        
        self.logger.info("Splitting data into train/val/test sets")
        train_dataset, val_dataset, test_dataset = DataSplitter.split_by_nct_id(
            dataset,
            train_ratio=split_config['train_ratio'],
            val_ratio=split_config['val_ratio'],
            test_ratio=split_config['test_ratio'],
            random_state=split_config['random_state']
        )
        
        self.logger.info(f"Split sizes - Train: {len(train_dataset.sentences)}, "
                        f"Val: {len(val_dataset.sentences)}, Test: {len(test_dataset.sentences)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_classifier(self, model_name: str) -> object:
        """Create a classifier based on configuration."""
        model_config = self.config['models'][model_name]
        model_type = model_config['type']
        
        if model_type == 'TfidfClassifier':
            return TfidfClassifier(
                model_name=model_name,
                classifier_type=model_config['classifier_type'],
                max_features=model_config['max_features'],
                ngram_range=tuple(model_config['ngram_range'])
            )
        elif model_type == 'RuleBasedClassifier':
            return RuleBasedClassifier(model_name=model_name)
        elif model_type == 'EnsembleClassifier':
            base_classifiers = [
                self.create_classifier(base_model) 
                for base_model in model_config['base_models']
            ]
            return EnsembleClassifier(model_name=model_name, classifiers=base_classifiers)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_and_evaluate_model(self, 
                                model_name: str,
                                train_dataset: ClinicalTrialDataset,
                                test_dataset: ClinicalTrialDataset) -> Dict[str, Any]:
        """Train and evaluate a single model."""
        self.logger.info(f"Training and evaluating model: {model_name}")
        
        # Create and train classifier
        classifier = self.create_classifier(model_name)
        classifier.fit(train_dataset)
        
        # Evaluate on test set
        metrics = self.evaluator.evaluate_model(classifier, test_dataset)
        
        # Save model
        models_dir = Path(self.config['paths']['models_dir'])
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{model_name}.joblib"
        classifier.save_model(str(model_path))
        
        self.logger.info(f"Model {model_name} - {metrics}")
        
        return {
            'model_name': model_name,
            'classifier': classifier,
            'metrics': metrics
        }
    
    def run_full_pipeline(self, data_path: str, model_names: List[str] = None) -> Dict[str, Any]:
        """Run the complete classification pipeline."""
        self.logger.info("Starting full classification pipeline")
        
        # Load and preprocess data
        dataset = self.load_and_preprocess_data(data_path)
        
        # Split data
        train_dataset, val_dataset, test_dataset = self.split_data(dataset)
        
        # Train and evaluate models
        if model_names is None:
            model_names = list(self.config['models'].keys())
        
        results = {}
        model_comparisons = []
        
        for model_name in model_names:
            try:
                result = self.train_and_evaluate_model(model_name, train_dataset, test_dataset)
                results[model_name] = result
                
                model_comparisons.append((
                    result['classifier'],
                    test_dataset,
                    model_name
                ))
                
            except Exception as e:
                self.logger.error(f"Error training model {model_name}: {e}")
                continue
        
        # Compare models
        if len(model_comparisons) > 1:
            self.logger.info("Comparing model performance")
            comparison_df = self.evaluator.compare_models(model_comparisons)
            self.logger.info(f"Model comparison:\n{comparison_df}")
        
        # Generate comprehensive evaluation report for best model
        if results:
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k]['metrics'].f1_score)
            best_result = results[best_model_name]
            
            self.logger.info(f"Generating detailed report for best model: {best_model_name}")
            report = generate_evaluation_report(
                self.evaluator,
                best_result['classifier'],
                test_dataset
            )
            
            self.logger.info("Pipeline completed successfully")
            
            # Save summary results
            self._save_pipeline_summary(results, best_model_name)
        
        return results
    
    def predict_new_data(self, model_path: str, sentences: List[str]) -> List[Dict[str, Any]]:
        """Use a trained model to predict on new sentences."""
        from src.classification.prior_therapy_classifier import BaseClassifier
        
        # Load model
        classifier = BaseClassifier.load_model(model_path)
        
        # Make predictions
        predictions = classifier.predict(sentences)
        
        # Convert to dictionary format
        results = []
        for sentence, prediction in zip(sentences, predictions):
            results.append({
                'sentence': sentence,
                'has_prior_therapy': prediction.predicted_has_prior_therapy,
                'confidence': prediction.confidence_score,
                'therapy_type': prediction.predicted_therapy_type,
                'therapy_name': prediction.predicted_therapy_name
            })
        
        return results
    
    def _save_pipeline_summary(self, results: Dict[str, Any], best_model_name: str) -> None:
        """Save a summary of pipeline results."""
        import json
        
        summary = {
            'best_model': best_model_name,
            'model_performance': {}
        }
        
        for model_name, result in results.items():
            summary['model_performance'][model_name] = {
                'accuracy': result['metrics'].accuracy,
                'precision': result['metrics'].precision,
                'recall': result['metrics'].recall,
                'f1_score': result['metrics'].f1_score
            }
        
        summary_path = Path(self.config['paths']['results_dir']) / 'pipeline_summary.json'
        summary_path.parent.mkdir(exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Clinical Trial Prior Therapy Classification Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data file (CSV or JSON)')
    parser.add_argument('--models', nargs='+', 
                       help='Model names to train (default: all models in config)')
    parser.add_argument('--predict', type=str,
                       help='Path to trained model for making predictions')
    parser.add_argument('--sentences', nargs='+',
                       help='Sentences to classify (when using --predict)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PriorTherapyClassificationPipeline(args.config)
    
    if args.predict and args.sentences:
        # Prediction mode
        results = pipeline.predict_new_data(args.predict, args.sentences)
        print("\nPrediction Results:")
        for result in results:
            print(f"Sentence: {result['sentence']}")
            print(f"Prior Therapy: {result['has_prior_therapy']} (confidence: {result['confidence']:.3f})")
            if result['therapy_type']:
                print(f"Therapy Type: {result['therapy_type']}")
            if result['therapy_name']:
                print(f"Therapy Name: {result['therapy_name']}")
            print("-" * 50)
    else:
        # Training mode
        results = pipeline.run_full_pipeline(args.data, args.models)
        
        print("\nPipeline Results:")
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()