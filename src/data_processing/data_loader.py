"""
Data processing utilities for clinical trial prior therapy classification.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import csv

from ..models.clinical_trial_data import (
    ClinicalTrialSentence, ClinicalTrialDataset, OncoAnnotation,
    EligibilityCriteriaSection, TherapyType
)


class DataLoader:
    """Loads clinical trial data from various formats."""
    
    @staticmethod
    def load_from_csv(file_path: Union[str, Path]) -> ClinicalTrialDataset:
        """
        Load clinical trial data from CSV file.
        
        Expected CSV columns:
        - nct_id: NCT identifier
        - sentence_id: Unique sentence identifier
        - sentence_text: The sentence text
        - section: Eligibility criteria section
        - has_prior_therapy: Boolean for prior therapy presence
        - therapy_type: Type of therapy (optional)
        - therapy_name: Specific therapy name (optional)
        - is_inclusion_criteria: Boolean for inclusion criteria
        - is_exclusion_criteria: Boolean for exclusion criteria
        - confidence_score: Annotation confidence (optional)
        - notes: Additional notes (optional)
        """
        df = pd.read_csv(file_path)
        
        sentences = []
        for _, row in df.iterrows():
            # Handle therapy_type conversion
            therapy_type = None
            if pd.notna(row.get('therapy_type')):
                therapy_type_str = str(row.get('therapy_type')).lower()
                try:
                    therapy_type = TherapyType(therapy_type_str)
                except ValueError:
                    # If not a valid enum value, keep as None
                    therapy_type = None
            
            # Handle section conversion
            section_str = str(row['section']).lower()
            try:
                section = EligibilityCriteriaSection(section_str)
            except ValueError:
                # Default to 'other' if not a valid section
                section = EligibilityCriteriaSection.OTHER
            
            onco_annotation = OncoAnnotation(
                has_prior_therapy=bool(row['has_prior_therapy']),
                therapy_type=therapy_type,
                therapy_name=row.get('therapy_name') if pd.notna(row.get('therapy_name')) else None,
                is_inclusion_criteria=bool(row['is_inclusion_criteria']),
                is_exclusion_criteria=bool(row['is_exclusion_criteria']),
                confidence_score=row.get('confidence_score') if pd.notna(row.get('confidence_score')) else None,
                notes=row.get('notes') if pd.notna(row.get('notes')) else None
            )
            
            sentence = ClinicalTrialSentence(
                nct_id=str(row['nct_id']),
                sentence_id=str(row['sentence_id']),
                sentence_text=str(row['sentence_text']),
                section=section,
                onco_annotation=onco_annotation,
                original_document_position=row.get('original_document_position') if pd.notna(row.get('original_document_position')) else None
            )
            sentences.append(sentence)
        
        dataset_name = Path(file_path).stem
        return ClinicalTrialDataset(
            dataset_name=dataset_name,
            dataset_version="1.0",
            description=f"Dataset loaded from {file_path}",
            sentences=sentences
        )
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> ClinicalTrialDataset:
        """Load clinical trial data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ClinicalTrialDataset(**data)
    
    @staticmethod
    def save_to_csv(dataset: ClinicalTrialDataset, file_path: Union[str, Path]) -> None:
        """Save clinical trial dataset to CSV file."""
        data = []
        for sentence in dataset.sentences:
            row = {
                'nct_id': sentence.nct_id,
                'sentence_id': sentence.sentence_id,
                'sentence_text': sentence.sentence_text,
                'section': sentence.section.value,
                'has_prior_therapy': sentence.onco_annotation.has_prior_therapy,
                'therapy_type': sentence.onco_annotation.therapy_type.value if sentence.onco_annotation.therapy_type else None,
                'therapy_name': sentence.onco_annotation.therapy_name,
                'is_inclusion_criteria': sentence.onco_annotation.is_inclusion_criteria,
                'is_exclusion_criteria': sentence.onco_annotation.is_exclusion_criteria,
                'confidence_score': sentence.onco_annotation.confidence_score,
                'notes': sentence.onco_annotation.notes,
                'original_document_position': sentence.original_document_position
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    @staticmethod
    def save_to_json(dataset: ClinicalTrialDataset, file_path: Union[str, Path]) -> None:
        """Save clinical trial dataset to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.dict(), f, indent=2, ensure_ascii=False)


class TextPreprocessor:
    """Preprocesses clinical trial text data."""
    
    def __init__(self, lowercase: bool = True, remove_special_chars: bool = False):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        
        # Common therapy-related keywords for identification
        self.therapy_keywords = {
            'chemotherapy': ['chemotherapy', 'chemo', 'cytotoxic', 'platinum', 'carboplatin', 'cisplatin'],
            'immunotherapy': ['immunotherapy', 'checkpoint inhibitor', 'pd-1', 'pd-l1', 'ctla-4', 'pembrolizumab'],
            'targeted_therapy': ['targeted therapy', 'tyrosine kinase inhibitor', 'tki', 'egfr', 'her2'],
            'radiation_therapy': ['radiation', 'radiotherapy', 'rt', 'irradiation'],
            'hormone_therapy': ['hormone therapy', 'endocrine therapy', 'tamoxifen', 'aromatase inhibitor'],
            'surgery': ['surgery', 'surgical', 'resection', 'excision', 'biopsy']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text string."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special_chars:
            # Keep alphanumeric, spaces, and common punctuation
            text = re.sub(r'[^\w\s\-\.\,\;\:]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_therapy_mentions(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract therapy mentions from text.
        Returns list of tuples (therapy_type, matched_text).
        """
        text_lower = text.lower()
        mentions = []
        
        for therapy_type, keywords in self.therapy_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    mentions.append((therapy_type, keyword))
        
        return mentions
    
    def is_prior_therapy_context(self, text: str) -> bool:
        """Check if text contains prior therapy context indicators."""
        prior_indicators = [
            'prior', 'previous', 'previously', 'past', 'history of',
            'received', 'treated with', 'exposure to', 'administration of'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in prior_indicators)
    
    def preprocess_dataset(self, dataset: ClinicalTrialDataset) -> ClinicalTrialDataset:
        """Preprocess all sentences in a dataset."""
        processed_sentences = []
        
        for sentence in dataset.sentences:
            processed_text = self.preprocess_text(sentence.sentence_text)
            
            # Create new sentence with processed text
            processed_sentence = ClinicalTrialSentence(
                nct_id=sentence.nct_id,
                sentence_id=sentence.sentence_id,
                sentence_text=processed_text,
                section=sentence.section,
                onco_annotation=sentence.onco_annotation,
                original_document_position=sentence.original_document_position
            )
            processed_sentences.append(processed_sentence)
        
        return ClinicalTrialDataset(
            dataset_name=f"{dataset.dataset_name}_preprocessed",
            dataset_version=dataset.dataset_version,
            description=f"Preprocessed version of {dataset.description}",
            sentences=processed_sentences,
            metadata=dataset.metadata
        )


class DataSplitter:
    """Splits clinical trial data for training, validation, and testing."""
    
    @staticmethod
    def split_by_nct_id(dataset: ClinicalTrialDataset, 
                       train_ratio: float = 0.7, 
                       val_ratio: float = 0.15, 
                       test_ratio: float = 0.15,
                       random_state: int = 42) -> Tuple[ClinicalTrialDataset, ClinicalTrialDataset, ClinicalTrialDataset]:
        """
        Split dataset by NCT ID to ensure no trial appears in multiple splits.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Get unique NCT IDs
        nct_ids = list(set(sentence.nct_id for sentence in dataset.sentences))
        np.random.seed(random_state)
        np.random.shuffle(nct_ids)
        
        # Calculate split indices
        n_ncts = len(nct_ids)
        train_end = int(n_ncts * train_ratio)
        val_end = train_end + int(n_ncts * val_ratio)
        
        train_ncts = set(nct_ids[:train_end])
        val_ncts = set(nct_ids[train_end:val_end])
        test_ncts = set(nct_ids[val_end:])
        
        # Split sentences
        train_sentences = [s for s in dataset.sentences if s.nct_id in train_ncts]
        val_sentences = [s for s in dataset.sentences if s.nct_id in val_ncts]
        test_sentences = [s for s in dataset.sentences if s.nct_id in test_ncts]
        
        # Create datasets
        train_dataset = ClinicalTrialDataset(
            dataset_name=f"{dataset.dataset_name}_train",
            dataset_version=dataset.dataset_version,
            description=f"Training split of {dataset.description}",
            sentences=train_sentences
        )
        
        val_dataset = ClinicalTrialDataset(
            dataset_name=f"{dataset.dataset_name}_val",
            dataset_version=dataset.dataset_version,
            description=f"Validation split of {dataset.description}",
            sentences=val_sentences
        )
        
        test_dataset = ClinicalTrialDataset(
            dataset_name=f"{dataset.dataset_name}_test",
            dataset_version=dataset.dataset_version,
            description=f"Test split of {dataset.description}",
            sentences=test_sentences
        )
        
        return train_dataset, val_dataset, test_dataset