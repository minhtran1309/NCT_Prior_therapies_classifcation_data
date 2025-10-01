#!/usr/bin/env python3
"""
Simple runner script for the clinical trial prior therapy classification pipeline.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.classification.prior_therapy_classifier import RuleBasedClassifier, TfidfClassifier
from src.data_processing.data_loader import DataLoader
from src.evaluation.model_evaluator import ModelEvaluator


def quick_demo():
    """Run a quick demonstration of the pipeline."""
    print("="*60)
    print("CLINICAL TRIAL PRIOR THERAPY CLASSIFICATION")
    print("="*60)
    
    # Load sample data
    print("Loading sample data...")
    loader = DataLoader()
    dataset = loader.load_from_csv('data/sample_clinical_trial_data.csv')
    print(f"Loaded {len(dataset.sentences)} sentences from {len(set(s.nct_id for s in dataset.sentences))} trials")
    
    # Quick analysis
    prior_therapy_count = len(dataset.get_prior_therapy_sentences())
    print(f"Prior therapy mentions: {prior_therapy_count}/{len(dataset.sentences)}")
    
    # Test rule-based classifier
    print("\nTesting Rule-Based Classifier...")
    classifier = RuleBasedClassifier("quick_demo")
    classifier.fit(None)  # Rule-based doesn't need training
    
    # Test on a few sample sentences
    test_sentences = [
        "Prior chemotherapy is required for enrollment.",
        "Patients must be 18 years or older.",
        "No previous immunotherapy allowed.",
        "ECOG performance status 0-1."
    ]
    
    predictions = classifier.predict(test_sentences)
    
    print("\nSample Predictions:")
    print("-" * 40)
    for sentence, pred in zip(test_sentences, predictions):
        status = "YES" if pred.predicted_has_prior_therapy else "NO"
        print(f"'{sentence}' -> Prior Therapy: {status} (Confidence: {pred.confidence_score:.2f})")
    
    print("\n" + "="*60)
    print("Quick demo completed! For full pipeline, use:")
    print("python src/pipeline.py --data data/sample_clinical_trial_data.csv")
    print("="*60)


if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")