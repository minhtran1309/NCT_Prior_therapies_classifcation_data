"""
Prior therapy classification models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
from pathlib import Path

from ..models.clinical_trial_data import (
    ClinicalTrialSentence, ClinicalTrialDataset, ClassificationResult,
    EvaluationMetrics, TherapyType
)


class BaseClassifier(ABC):
    """Base class for prior therapy classifiers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
    
    @abstractmethod
    def fit(self, dataset: ClinicalTrialDataset) -> None:
        """Train the classifier on the dataset."""
        pass
    
    @abstractmethod
    def predict(self, sentences: List[str]) -> List[ClassificationResult]:
        """Predict prior therapy for a list of sentences."""
        pass
    
    def predict_dataset(self, dataset: ClinicalTrialDataset) -> List[ClassificationResult]:
        """Predict prior therapy for all sentences in a dataset."""
        sentences = [s.sentence_text for s in dataset.sentences]
        predictions = self.predict(sentences)
        
        # Update sentence IDs
        for i, pred in enumerate(predictions):
            pred.sentence_id = dataset.sentences[i].sentence_id
        
        return predictions
    
    def save_model(self, file_path: str) -> None:
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self, file_path)
    
    @classmethod
    def load_model(cls, file_path: str) -> 'BaseClassifier':
        """Load a trained model from file."""
        model = joblib.load(file_path)
        if not isinstance(model, cls):
            raise ValueError(f"Loaded model is not of type {cls.__name__}")
        return model


class TfidfClassifier(BaseClassifier):
    """TF-IDF based classifier for prior therapy detection."""
    
    def __init__(self, 
                 model_name: str = "tfidf_classifier",
                 classifier_type: str = "random_forest",
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2)):
        super().__init__(model_name)
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize classifier
        if classifier_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == "logistic_regression":
            classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif classifier_type == "naive_bayes":
            classifier = MultinomialNB()
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True
            )),
            ('classifier', classifier)
        ])
    
    def fit(self, dataset: ClinicalTrialDataset) -> None:
        """Train the classifier on the dataset."""
        X = [sentence.sentence_text for sentence in dataset.sentences]
        y = [sentence.onco_annotation.has_prior_therapy for sentence in dataset.sentences]
        
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, sentences: List[str]) -> List[ClassificationResult]:
        """Predict prior therapy for a list of sentences."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.pipeline.predict(sentences)
        probabilities = self.pipeline.predict_proba(sentences)
        
        results = []
        for i, (sentence, pred, prob) in enumerate(zip(sentences, predictions, probabilities)):
            result = ClassificationResult(
                sentence_id=f"sent_{i}",  # Will be updated by caller
                predicted_has_prior_therapy=bool(pred),
                confidence_score=float(max(prob)),
                model_name=self.model_name
            )
            results.append(result)
        
        return results


class RuleBasedClassifier(BaseClassifier):
    """Rule-based classifier using keyword matching."""
    
    def __init__(self, model_name: str = "rule_based_classifier"):
        super().__init__(model_name)
        
        # Prior therapy keywords
        self.prior_keywords = [
            'prior', 'previous', 'previously', 'past', 'history of',
            'received', 'treated with', 'exposure to', 'administration of',
            'failed', 'refractory to', 'resistant to', 'progression on'
        ]
        
        # Therapy keywords
        self.therapy_keywords = [
            'chemotherapy', 'chemo', 'cytotoxic', 'immunotherapy', 'targeted therapy',
            'radiation', 'radiotherapy', 'surgery', 'hormone therapy', 'treatment',
            'therapy', 'drug', 'medication', 'agent', 'regimen', 'protocol'
        ]
        
        # Specific drug names and therapy types
        self.specific_therapies = {
            TherapyType.CHEMOTHERAPY: [
                'cisplatin', 'carboplatin', 'paclitaxel', 'docetaxel', 'doxorubicin',
                'cyclophosphamide', 'fluorouracil', '5-fu', 'gemcitabine'
            ],
            TherapyType.IMMUNOTHERAPY: [
                'pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab',
                'ipilimumab', 'checkpoint inhibitor', 'pd-1', 'pd-l1', 'ctla-4'
            ],
            TherapyType.TARGETED_THERAPY: [
                'erlotinib', 'gefitinib', 'imatinib', 'trastuzumab', 'bevacizumab',
                'cetuximab', 'tyrosine kinase inhibitor', 'tki', 'egfr', 'her2'
            ],
            TherapyType.HORMONE_THERAPY: [
                'tamoxifen', 'anastrozole', 'letrozole', 'exemestane',
                'aromatase inhibitor', 'antiestrogen'
            ]
        }
    
    def fit(self, dataset: ClinicalTrialDataset) -> None:
        """Rule-based classifier doesn't require training."""
        self.is_trained = True
    
    def predict(self, sentences: List[str]) -> List[ClassificationResult]:
        """Predict prior therapy using rule-based approach."""
        results = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check for prior therapy indicators
            has_prior_indicator = any(keyword in sentence_lower for keyword in self.prior_keywords)
            has_therapy_keyword = any(keyword in sentence_lower for keyword in self.therapy_keywords)
            
            # Check for specific therapies
            detected_therapy_type = None
            detected_therapy_name = None
            
            for therapy_type, therapy_names in self.specific_therapies.items():
                for therapy_name in therapy_names:
                    if therapy_name in sentence_lower:
                        detected_therapy_type = therapy_type
                        detected_therapy_name = therapy_name
                        break
                if detected_therapy_type:
                    break
            
            # Make prediction
            has_prior_therapy = has_prior_indicator and (has_therapy_keyword or detected_therapy_type is not None)
            
            # Calculate confidence based on number of matching indicators
            confidence = 0.5
            if has_prior_indicator:
                confidence += 0.3
            if has_therapy_keyword:
                confidence += 0.15
            if detected_therapy_type:
                confidence += 0.05
            
            confidence = min(confidence, 1.0)
            
            result = ClassificationResult(
                sentence_id=f"sent_{i}",
                predicted_has_prior_therapy=has_prior_therapy,
                confidence_score=confidence,
                predicted_therapy_type=detected_therapy_type,
                predicted_therapy_name=detected_therapy_name,
                model_name=self.model_name
            )
            results.append(result)
        
        return results


class EnsembleClassifier(BaseClassifier):
    """Ensemble classifier combining multiple models."""
    
    def __init__(self, 
                 model_name: str = "ensemble_classifier",
                 classifiers: Optional[List[BaseClassifier]] = None):
        super().__init__(model_name)
        
        if classifiers is None:
            # Default ensemble
            self.classifiers = [
                TfidfClassifier("tfidf_rf", "random_forest"),
                TfidfClassifier("tfidf_lr", "logistic_regression"),
                RuleBasedClassifier("rule_based")
            ]
        else:
            self.classifiers = classifiers
    
    def fit(self, dataset: ClinicalTrialDataset) -> None:
        """Train all classifiers in the ensemble."""
        for classifier in self.classifiers:
            classifier.fit(dataset)
        self.is_trained = True
    
    def predict(self, sentences: List[str]) -> List[ClassificationResult]:
        """Predict using ensemble voting."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions from all classifiers
        all_predictions = []
        for classifier in self.classifiers:
            predictions = classifier.predict(sentences)
            all_predictions.append(predictions)
        
        # Combine predictions using majority voting
        ensemble_results = []
        for i in range(len(sentences)):
            votes = [pred[i].predicted_has_prior_therapy for pred in all_predictions]
            confidences = [pred[i].confidence_score for pred in all_predictions]
            
            # Majority vote
            has_prior_therapy = sum(votes) > len(votes) / 2
            
            # Average confidence, weighted by prediction agreement
            avg_confidence = np.mean(confidences)
            
            result = ClassificationResult(
                sentence_id=f"sent_{i}",
                predicted_has_prior_therapy=has_prior_therapy,
                confidence_score=avg_confidence,
                model_name=self.model_name
            )
            ensemble_results.append(result)
        
        return ensemble_results


def evaluate_classifier(classifier: BaseClassifier, 
                       test_dataset: ClinicalTrialDataset) -> EvaluationMetrics:
    """Evaluate a classifier on a test dataset."""
    predictions = classifier.predict_dataset(test_dataset)
    
    # Extract true labels and predictions
    y_true = [sentence.onco_annotation.has_prior_therapy for sentence in test_dataset.sentences]
    y_pred = [pred.predicted_has_prior_therapy for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=cm.tolist()
    )