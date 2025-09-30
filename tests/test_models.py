"""
Unit tests for clinical trial data models.
"""

import pytest
from src.models.clinical_trial_data import (
    ClinicalTrialSentence, OncoAnnotation, ClinicalTrialDataset,
    EligibilityCriteriaSection, TherapyType, ClassificationResult, EvaluationMetrics
)


def test_onco_annotation_creation():
    """Test OncoAnnotation model creation."""
    annotation = OncoAnnotation(
        has_prior_therapy=True,
        therapy_type=TherapyType.CHEMOTHERAPY,
        therapy_name="cisplatin",
        is_inclusion_criteria=False,
        is_exclusion_criteria=True,
        confidence_score=0.95,
        notes="Clear chemotherapy mention"
    )
    
    assert annotation.has_prior_therapy is True
    assert annotation.therapy_type == TherapyType.CHEMOTHERAPY
    assert annotation.therapy_name == "cisplatin"
    assert annotation.confidence_score == 0.95


def test_clinical_trial_sentence_creation():
    """Test ClinicalTrialSentence model creation."""
    annotation = OncoAnnotation(
        has_prior_therapy=True,
        is_inclusion_criteria=False,
        is_exclusion_criteria=True
    )
    
    sentence = ClinicalTrialSentence(
        nct_id="NCT00123456",
        sentence_id="sent_001",
        sentence_text="Prior chemotherapy is required.",
        section=EligibilityCriteriaSection.PRIOR_THERAPY,
        onco_annotation=annotation
    )
    
    assert sentence.nct_id == "NCT00123456"
    assert sentence.sentence_text == "Prior chemotherapy is required."
    assert sentence.section == EligibilityCriteriaSection.PRIOR_THERAPY


def test_clinical_trial_dataset_methods():
    """Test ClinicalTrialDataset filtering methods."""
    # Create test sentences
    annotation1 = OncoAnnotation(has_prior_therapy=True, is_inclusion_criteria=True, is_exclusion_criteria=False)
    annotation2 = OncoAnnotation(has_prior_therapy=False, is_inclusion_criteria=True, is_exclusion_criteria=False)
    
    sentence1 = ClinicalTrialSentence(
        nct_id="NCT001", sentence_id="s1", sentence_text="Prior therapy required.",
        section=EligibilityCriteriaSection.PRIOR_THERAPY, onco_annotation=annotation1
    )
    
    sentence2 = ClinicalTrialSentence(
        nct_id="NCT002", sentence_id="s2", sentence_text="Age >= 18 years.",
        section=EligibilityCriteriaSection.INCLUSION, onco_annotation=annotation2
    )
    
    dataset = ClinicalTrialDataset(
        dataset_name="test_dataset",
        dataset_version="1.0",
        sentences=[sentence1, sentence2]
    )
    
    # Test filtering methods
    nct001_sentences = dataset.get_sentences_by_nct("NCT001")
    assert len(nct001_sentences) == 1
    assert nct001_sentences[0].nct_id == "NCT001"
    
    prior_therapy_sentences = dataset.get_prior_therapy_sentences()
    assert len(prior_therapy_sentences) == 1
    assert prior_therapy_sentences[0].onco_annotation.has_prior_therapy is True
    
    prior_therapy_section = dataset.get_sentences_by_section(EligibilityCriteriaSection.PRIOR_THERAPY)
    assert len(prior_therapy_section) == 1


def test_classification_result():
    """Test ClassificationResult model."""
    result = ClassificationResult(
        sentence_id="sent_001",
        predicted_has_prior_therapy=True,
        confidence_score=0.95,
        predicted_therapy_type=TherapyType.IMMUNOTHERAPY,
        model_name="test_model"
    )
    
    assert result.predicted_has_prior_therapy is True
    assert result.confidence_score == 0.95
    assert result.predicted_therapy_type == TherapyType.IMMUNOTHERAPY


def test_evaluation_metrics():
    """Test EvaluationMetrics model."""
    metrics = EvaluationMetrics(
        accuracy=0.95,
        precision=0.90,
        recall=0.85,
        f1_score=0.875,
        confusion_matrix=[[10, 2], [3, 15]]
    )
    
    assert metrics.accuracy == 0.95
    assert metrics.f1_score == 0.875
    assert len(metrics.confusion_matrix) == 2


if __name__ == "__main__":
    pytest.main([__file__])