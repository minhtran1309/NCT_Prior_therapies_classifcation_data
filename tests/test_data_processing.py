"""
Unit tests for data processing functionality.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.data_processing.data_loader import DataLoader, TextPreprocessor, DataSplitter
from src.models.clinical_trial_data import (
    ClinicalTrialSentence, OncoAnnotation, ClinicalTrialDataset,
    EligibilityCriteriaSection, TherapyType
)


def create_test_csv():
    """Create a temporary CSV file for testing."""
    test_data = {
        'nct_id': ['NCT001', 'NCT001', 'NCT002'],
        'sentence_id': ['s1', 's2', 's3'],
        'sentence_text': [
            'Prior chemotherapy is required.',
            'Age must be >= 18 years.',
            'No previous immunotherapy allowed.'
        ],
        'section': ['prior_therapy', 'inclusion', 'prior_therapy'],
        'has_prior_therapy': [True, False, True],
        'therapy_type': ['chemotherapy', None, 'immunotherapy'],
        'therapy_name': ['chemotherapy', None, 'immunotherapy'],
        'is_inclusion_criteria': [True, True, False],
        'is_exclusion_criteria': [False, False, True],
        'confidence_score': [0.9, 1.0, 0.85],
        'notes': [None, None, None],
        'original_document_position': [1, 2, 3]
    }
    
    df = pd.DataFrame(test_data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name


def test_data_loader_csv():
    """Test loading data from CSV."""
    csv_file = create_test_csv()
    
    try:
        loader = DataLoader()
        dataset = loader.load_from_csv(csv_file)
        
        assert isinstance(dataset, ClinicalTrialDataset)
        assert len(dataset.sentences) == 3
        assert dataset.sentences[0].nct_id == 'NCT001'
        assert dataset.sentences[0].onco_annotation.has_prior_therapy is True
        
    finally:
        os.unlink(csv_file)


def test_text_preprocessor():
    """Test text preprocessing functionality."""
    preprocessor = TextPreprocessor(lowercase=True, remove_special_chars=False)
    
    # Test basic preprocessing
    text = "PRIOR Chemotherapy is Required!!!"
    processed = preprocessor.preprocess_text(text)
    assert processed == "prior chemotherapy is required!!!"
    
    # Test therapy mention extraction
    mentions = preprocessor.extract_therapy_mentions("Patient received prior cisplatin treatment")
    assert len(mentions) > 0
    assert any('chemotherapy' in mention[0] for mention in mentions)
    
    # Test prior therapy context detection
    assert preprocessor.is_prior_therapy_context("Patient previously received treatment")
    assert not preprocessor.is_prior_therapy_context("Patient will receive treatment")


def test_data_splitter():
    """Test data splitting functionality."""
    # Create test dataset
    sentences = []
    for i in range(10):
        nct_id = f"NCT00{i % 3}"  # 3 different NCT IDs
        annotation = OncoAnnotation(
            has_prior_therapy=(i % 2 == 0),
            is_inclusion_criteria=True,
            is_exclusion_criteria=False
        )
        sentence = ClinicalTrialSentence(
            nct_id=nct_id,
            sentence_id=f"sent_{i}",
            sentence_text=f"Test sentence {i}",
            section=EligibilityCriteriaSection.INCLUSION,
            onco_annotation=annotation
        )
        sentences.append(sentence)
    
    dataset = ClinicalTrialDataset(
        dataset_name="test_dataset",
        dataset_version="1.0",
        sentences=sentences
    )
    
    # Split data
    train_dataset, val_dataset, test_dataset = DataSplitter.split_by_nct_id(
        dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    
    # Check that all datasets are created
    assert len(train_dataset.sentences) > 0
    assert len(val_dataset.sentences) > 0
    assert len(test_dataset.sentences) > 0
    
    # Check that total sentences match
    total_sentences = len(train_dataset.sentences) + len(val_dataset.sentences) + len(test_dataset.sentences)
    assert total_sentences == len(dataset.sentences)
    
    # Check that no NCT ID appears in multiple splits
    train_ncts = set(s.nct_id for s in train_dataset.sentences)
    val_ncts = set(s.nct_id for s in val_dataset.sentences)
    test_ncts = set(s.nct_id for s in test_dataset.sentences)
    
    assert len(train_ncts.intersection(val_ncts)) == 0
    assert len(train_ncts.intersection(test_ncts)) == 0
    assert len(val_ncts.intersection(test_ncts)) == 0


if __name__ == "__main__":
    pytest.main([__file__])