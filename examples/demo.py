#!/usr/bin/env python3
"""
Example usage of the Clinical Trial Prior Therapy Classification Pipeline.

This script demonstrates how to:
1. Load and preprocess clinical trial data
2. Train multiple classification models
3. Evaluate model performance
4. Make predictions on new sentences
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import PriorTherapyClassificationPipeline
from src.data_processing.data_loader import DataLoader
from src.models.clinical_trial_data import (
    ClinicalTrialSentence, OncoAnnotation, EligibilityCriteriaSection, TherapyType
)


def create_sample_data():
    """Create additional sample data for demonstration."""
    print("Creating sample clinical trial data...")
    
    sample_sentences = [
        {
            'nct_id': 'NCT00567890',
            'sentence_id': 'sent_021',
            'sentence_text': 'Prior treatment with anti-EGFR therapy is required.',
            'section': EligibilityCriteriaSection.PRIOR_THERAPY,
            'has_prior_therapy': True,
            'therapy_type': TherapyType.TARGETED_THERAPY,
            'therapy_name': 'anti-EGFR therapy',
            'is_inclusion_criteria': True,
            'is_exclusion_criteria': False,
            'confidence_score': 0.95
        },
        {
            'nct_id': 'NCT00567890',
            'sentence_id': 'sent_022',
            'sentence_text': 'Patients must have progressive disease following standard therapy.',
            'section': EligibilityCriteriaSection.INCLUSION,
            'has_prior_therapy': True,
            'therapy_type': None,
            'therapy_name': 'standard therapy',
            'is_inclusion_criteria': True,
            'is_exclusion_criteria': False,
            'confidence_score': 0.80
        },
        {
            'nct_id': 'NCT00567890',
            'sentence_id': 'sent_023',
            'sentence_text': 'Adequate renal function as defined by creatinine <= 1.5 mg/dL.',
            'section': EligibilityCriteriaSection.INCLUSION,
            'has_prior_therapy': False,
            'therapy_type': None,
            'therapy_name': None,
            'is_inclusion_criteria': True,
            'is_exclusion_criteria': False,
            'confidence_score': 1.0
        }
    ]
    
    # Create additional sample data file
    import pandas as pd
    
    # Load existing data
    existing_df = pd.read_csv(project_root / 'data' / 'sample_clinical_trial_data.csv')
    
    # Add new sample data
    new_data = []
    for sample in sample_sentences:
        new_data.append({
            'nct_id': sample['nct_id'],
            'sentence_id': sample['sentence_id'],
            'sentence_text': sample['sentence_text'],
            'section': sample['section'].value,
            'has_prior_therapy': sample['has_prior_therapy'],
            'therapy_type': sample['therapy_type'].value if sample['therapy_type'] else None,
            'therapy_name': sample['therapy_name'],
            'is_inclusion_criteria': sample['is_inclusion_criteria'],
            'is_exclusion_criteria': sample['is_exclusion_criteria'],
            'confidence_score': sample['confidence_score'],
            'notes': None,
            'original_document_position': None
        })
    
    # Combine data
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Save extended dataset
    extended_data_path = project_root / 'data' / 'extended_sample_data.csv'
    combined_df.to_csv(extended_data_path, index=False)
    
    print(f"Extended sample data saved to: {extended_data_path}")
    return str(extended_data_path)


def demonstrate_basic_usage():
    """Demonstrate basic pipeline usage."""
    print("\n" + "="*60)
    print("CLINICAL TRIAL PRIOR THERAPY CLASSIFICATION DEMO")
    print("="*60)
    
    # Create sample data
    data_path = create_sample_data()
    
    # Initialize pipeline
    config_path = project_root / 'config' / 'config.yaml'
    pipeline = PriorTherapyClassificationPipeline(str(config_path))
    
    print(f"\nLoading data from: {data_path}")
    
    # Run the complete pipeline
    print("\nRunning classification pipeline...")
    results = pipeline.run_full_pipeline(data_path)
    
    # Display results
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics.accuracy:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall:    {metrics.recall:.3f}")
        print(f"  F1-Score:  {metrics.f1_score:.3f}")
    
    return results


def demonstrate_prediction():
    """Demonstrate prediction on new sentences."""
    print("\n" + "="*50)
    print("PREDICTION DEMONSTRATION")
    print("="*50)
    
    # Sample sentences for prediction
    test_sentences = [
        "Patients must have received at least one prior systemic therapy for metastatic disease.",
        "No prior immunotherapy with checkpoint inhibitors is allowed.",
        "ECOG performance status must be 0 or 1.",
        "Previous treatment with bevacizumab is permitted.",
        "Patients should be treatment-naive for advanced disease.",
        "Progression on platinum-based chemotherapy is required.",
        "Adequate liver function tests are mandatory."
    ]
    
    print("Test sentences:")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i}. {sentence}")
    
    # Use rule-based classifier for quick demonstration
    from src.classification.prior_therapy_classifier import RuleBasedClassifier
    
    classifier = RuleBasedClassifier("demo_rule_based")
    classifier.fit(None)  # Rule-based doesn't need training
    
    predictions = classifier.predict(test_sentences)
    
    print("\nPrediction Results:")
    print("-" * 50)
    
    for sentence, prediction in zip(test_sentences, predictions):
        print(f"Sentence: {sentence}")
        print(f"Prior Therapy: {'YES' if prediction.predicted_has_prior_therapy else 'NO'}")
        print(f"Confidence: {prediction.confidence_score:.3f}")
        if prediction.predicted_therapy_type:
            print(f"Therapy Type: {prediction.predicted_therapy_type}")
        if prediction.predicted_therapy_name:
            print(f"Therapy Name: {prediction.predicted_therapy_name}")
        print("-" * 50)


def demonstrate_data_analysis():
    """Demonstrate data analysis capabilities."""
    print("\n" + "="*50)
    print("DATA ANALYSIS DEMONSTRATION")
    print("="*50)
    
    # Load sample data
    data_path = project_root / 'data' / 'sample_clinical_trial_data.csv'
    loader = DataLoader()
    dataset = loader.load_from_csv(data_path)
    
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Total sentences: {len(dataset.sentences)}")
    print(f"Unique NCT IDs: {len(set(s.nct_id for s in dataset.sentences))}")
    
    # Analyze by section
    print("\nSentences by section:")
    sections = {}
    for sentence in dataset.sentences:
        section = sentence.section if isinstance(sentence.section, str) else sentence.section.value
        sections[section] = sections.get(section, 0) + 1
    
    for section, count in sections.items():
        print(f"  {section}: {count}")
    
    # Analyze prior therapy mentions
    prior_therapy_sentences = dataset.get_prior_therapy_sentences()
    print(f"\nSentences with prior therapy: {len(prior_therapy_sentences)}")
    
    # Analyze therapy types
    therapy_types = {}
    for sentence in prior_therapy_sentences:
        if sentence.onco_annotation.therapy_type:
            therapy_type = sentence.onco_annotation.therapy_type if isinstance(sentence.onco_annotation.therapy_type, str) else sentence.onco_annotation.therapy_type.value
            therapy_types[therapy_type] = therapy_types.get(therapy_type, 0) + 1
    
    print("\nTherapy types mentioned:")
    for therapy_type, count in therapy_types.items():
        print(f"  {therapy_type}: {count}")
    
    # Show examples by NCT ID
    print("\nExample sentences by NCT ID:")
    for nct_id in list(set(s.nct_id for s in dataset.sentences))[:2]:
        nct_sentences = dataset.get_sentences_by_nct(nct_id)
        print(f"\n{nct_id} ({len(nct_sentences)} sentences):")
        for sentence in nct_sentences[:3]:  # Show first 3 sentences
            therapy_status = "YES" if sentence.onco_annotation.has_prior_therapy else "NO"
            print(f"  - {sentence.sentence_text} [Prior Therapy: {therapy_status}]")


def main():
    """Main function to run all demonstrations."""
    try:
        # Demonstrate data analysis
        demonstrate_data_analysis()
        
        # Demonstrate prediction
        demonstrate_prediction()
        
        # Demonstrate basic pipeline usage
        demonstrate_basic_usage()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Provide your own clinical trial data in CSV format")
        print("2. Adjust configuration in config/config.yaml")
        print("3. Run the pipeline with: python src/pipeline.py --data your_data.csv")
        print("4. Check results in the evaluation_results/ directory")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check that all dependencies are installed and files are accessible.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())