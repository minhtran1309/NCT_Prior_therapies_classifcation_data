"""
Data models for clinical trial prior therapy classification.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EligibilityCriteriaSection(str, Enum):
    """Enumeration of eligibility criteria sections."""
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    PRIOR_THERAPY = "prior_therapy"
    CONCURRENT_THERAPY = "concurrent_therapy"
    OTHER = "other"


class TherapyType(str, Enum):
    """Enumeration of therapy types."""
    CHEMOTHERAPY = "chemotherapy"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    RADIATION_THERAPY = "radiation_therapy"
    HORMONE_THERAPY = "hormone_therapy"
    SURGERY = "surgery"
    COMBINATION_THERAPY = "combination_therapy"
    OTHER = "other"


class OncoAnnotation(BaseModel):
    """Oncologist annotation for a sentence."""
    has_prior_therapy: bool = Field(..., description="Whether the sentence mentions prior therapy")
    therapy_type: Optional[TherapyType] = Field(None, description="Type of therapy mentioned")
    therapy_name: Optional[str] = Field(None, description="Specific therapy name mentioned")
    is_inclusion_criteria: bool = Field(..., description="Whether this is an inclusion criteria")
    is_exclusion_criteria: bool = Field(..., description="Whether this is an exclusion criteria")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Annotation confidence")
    notes: Optional[str] = Field(None, description="Additional notes from annotator")


class ClinicalTrialSentence(BaseModel):
    """Represents a single sentence from a clinical trial description."""
    nct_id: str = Field(..., description="Clinical trial NCT identifier")
    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    sentence_text: str = Field(..., description="The actual sentence text")
    section: EligibilityCriteriaSection = Field(..., description="Section this sentence belongs to")
    onco_annotation: OncoAnnotation = Field(..., description="Oncologist annotation")
    original_document_position: Optional[int] = Field(None, description="Position in original document")
    
    class Config:
        use_enum_values = True


class ClinicalTrialDataset(BaseModel):
    """Collection of clinical trial sentences with metadata."""
    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_version: str = Field(..., description="Version of the dataset")
    description: Optional[str] = Field(None, description="Description of the dataset")
    sentences: List[ClinicalTrialSentence] = Field(..., description="List of annotated sentences")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    def get_sentences_by_nct(self, nct_id: str) -> List[ClinicalTrialSentence]:
        """Get all sentences for a specific NCT ID."""
        return [s for s in self.sentences if s.nct_id == nct_id]
    
    def get_prior_therapy_sentences(self) -> List[ClinicalTrialSentence]:
        """Get sentences that mention prior therapy."""
        return [s for s in self.sentences if s.onco_annotation.has_prior_therapy]
    
    def get_sentences_by_section(self, section: EligibilityCriteriaSection) -> List[ClinicalTrialSentence]:
        """Get sentences from a specific section."""
        return [s for s in self.sentences if s.section == section]


class ClassificationResult(BaseModel):
    """Result of prior therapy classification."""
    sentence_id: str = Field(..., description="Identifier of the classified sentence")
    predicted_has_prior_therapy: bool = Field(..., description="Predicted prior therapy presence")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    predicted_therapy_type: Optional[TherapyType] = Field(None, description="Predicted therapy type")
    predicted_therapy_name: Optional[str] = Field(None, description="Predicted therapy name")
    model_name: str = Field(..., description="Name of the model used for classification")
    
    class Config:
        use_enum_values = True


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for classification performance."""
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    classification_report: Optional[Dict[str, Any]] = Field(None, description="Detailed classification report")
    
    def __str__(self) -> str:
        return (f"Accuracy: {self.accuracy:.3f}, "
                f"Precision: {self.precision:.3f}, "
                f"Recall: {self.recall:.3f}, "
                f"F1: {self.f1_score:.3f}")