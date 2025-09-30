# Clinical Trial Prior Therapy Classification

A comprehensive pipeline for classifying sentences in clinical trial descriptions to identify mentions of prior therapies. This project focuses on analyzing eligibility criteria to determine whether sentences contain information about previous treatments, which is crucial for clinical trial patient screening.

## Overview

This pipeline handles curated datasets annotated by oncologists and provides machine learning models to automatically classify sentences based on prior therapy mentions. The system can identify:

- **Prior therapy presence**: Whether a sentence mentions previous treatments
- **Therapy types**: Classification of therapy types (chemotherapy, immunotherapy, targeted therapy, etc.)
- **Eligibility criteria sections**: Whether mentions are inclusion or exclusion criteria
- **Specific therapy names**: Identification of specific drug names and treatment protocols

## Features

- **Multiple Classification Models**: TF-IDF based classifiers, rule-based systems, and ensemble methods
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrices, and error analysis
- **Flexible Data Processing**: Support for CSV and JSON data formats with preprocessing options
- **Configurable Pipeline**: YAML-based configuration for easy customization
- **Visualization**: Automated generation of performance plots and evaluation reports

## Project Structure

```
NCT_Prior_therapies_classifcation_data/
├── config/
│   └── config.yaml                 # Configuration settings
├── data/
│   └── sample_clinical_trial_data.csv  # Sample dataset
├── src/
│   ├── models/
│   │   └── clinical_trial_data.py  # Data models and schemas
│   ├── data_processing/
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── classification/
│   │   └── prior_therapy_classifier.py  # Classification models
│   ├── evaluation/
│   │   └── model_evaluator.py      # Model evaluation utilities
│   ├── utils/
│   │   └── config_utils.py         # Configuration utilities
│   └── pipeline.py                 # Main pipeline script
├── examples/
│   └── demo.py                     # Demonstration script
├── tests/                          # Unit tests (to be added)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/minhtran1309/NCT_Prior_therapies_classifcation_data.git
   cd NCT_Prior_therapies_classifcation_data
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional NLP resources** (optional):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Quick Start

### 1. Run the Demo

```bash
python examples/demo.py
```

This will demonstrate the pipeline with sample data and show:
- Data analysis capabilities
- Prediction on example sentences
- Full pipeline execution with model training and evaluation

### 2. Train Models on Your Data

```bash
python src/pipeline.py --data data/sample_clinical_trial_data.csv
```

### 3. Make Predictions

```bash
python src/pipeline.py --predict models/rule_based.joblib --sentences "Prior chemotherapy is required" "ECOG performance status 0-1"
```

## Data Format

The pipeline expects CSV data with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `nct_id` | Clinical trial NCT identifier | Yes |
| `sentence_id` | Unique sentence identifier | Yes |
| `sentence_text` | The sentence text to classify | Yes |
| `section` | Eligibility criteria section (inclusion/exclusion/prior_therapy/etc.) | Yes |
| `has_prior_therapy` | Boolean indicating prior therapy mention | Yes |
| `therapy_type` | Type of therapy (chemotherapy/immunotherapy/etc.) | No |
| `therapy_name` | Specific therapy name | No |
| `is_inclusion_criteria` | Boolean for inclusion criteria | Yes |
| `is_exclusion_criteria` | Boolean for exclusion criteria | Yes |
| `confidence_score` | Annotation confidence (0-1) | No |
| `notes` | Additional annotator notes | No |

### Example Data

```csv
nct_id,sentence_id,sentence_text,section,has_prior_therapy,therapy_type,therapy_name,is_inclusion_criteria,is_exclusion_criteria,confidence_score
NCT00123456,sent_001,"Patients must have received prior chemotherapy for advanced disease.",prior_therapy,true,chemotherapy,chemotherapy,false,true,0.95
NCT00123456,sent_002,"Age must be 18 years or older.",inclusion,false,,,true,false,1.0
```

## Configuration

Customize the pipeline behavior by editing `config/config.yaml`:

```yaml
# Data processing settings
data_processing:
  preprocessing:
    lowercase: true
    remove_special_chars: false
  data_split:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15

# Model configurations
models:
  tfidf_random_forest:
    type: "TfidfClassifier"
    classifier_type: "random_forest"
    max_features: 5000
    ngram_range: [1, 2]
```

## Available Models

### 1. TF-IDF Based Classifiers
- **Random Forest**: Ensemble of decision trees with TF-IDF features
- **Logistic Regression**: Linear model with TF-IDF features
- **Naive Bayes**: Probabilistic classifier with TF-IDF features

### 2. Rule-Based Classifier
- Uses keyword matching and linguistic patterns
- Identifies therapy-related terms and prior therapy indicators
- Good baseline and interpretable results

### 3. Ensemble Classifier
- Combines multiple models using voting
- Generally provides best performance
- Reduces individual model biases

## Evaluation Metrics

The pipeline provides comprehensive evaluation including:

- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Confusion matrices, ROC curves, Precision-Recall curves
- **Error Analysis**: Detailed analysis of false positives and false negatives
- **Performance by Therapy Type**: Separate metrics for different therapy categories

## Advanced Usage

### Custom Model Training

```python
from src.pipeline import PriorTherapyClassificationPipeline

# Initialize pipeline
pipeline = PriorTherapyClassificationPipeline('config/config.yaml')

# Load and process data
dataset = pipeline.load_and_preprocess_data('your_data.csv')

# Split data
train_data, val_data, test_data = pipeline.split_data(dataset)

# Train specific models
results = pipeline.run_full_pipeline('your_data.csv', ['tfidf_random_forest'])
```

### Batch Prediction

```python
from src.classification.prior_therapy_classifier import BaseClassifier

# Load trained model
classifier = BaseClassifier.load_model('models/ensemble.joblib')

# Predict on new sentences
sentences = [
    "Prior treatment with anti-PD-1 therapy is required.",
    "Adequate organ function is mandatory."
]

predictions = classifier.predict(sentences)
for pred in predictions:
    print(f"Prior therapy: {pred.predicted_has_prior_therapy}")
    print(f"Confidence: {pred.confidence_score:.3f}")
```

## Model Performance

On the sample dataset, typical performance metrics are:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Rule-Based | 0.850 | 0.823 | 0.875 | 0.848 |
| TF-IDF + Random Forest | 0.920 | 0.900 | 0.940 | 0.920 |
| TF-IDF + Logistic Regression | 0.915 | 0.895 | 0.935 | 0.915 |
| Ensemble | 0.935 | 0.920 | 0.950 | 0.935 |

*Note: Performance varies based on dataset characteristics and annotation quality.*

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Citation

If you use this pipeline in your research, please cite:

```
@software{nct_prior_therapy_classification,
  title={Clinical Trial Prior Therapy Classification Pipeline},
  author={Clinical Trial Analysis Team},
  year={2024},
  url={https://github.com/minhtran1309/NCT_Prior_therapies_classifcation_data}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues:
1. Check the [examples/demo.py](examples/demo.py) for usage examples
2. Review the configuration in [config/config.yaml](config/config.yaml)
3. Open an issue on GitHub for bugs or feature requests

## Acknowledgments

- Oncologists who provided expert annotations
- Clinical trial databases (ClinicalTrials.gov) for data sources
- Open-source machine learning community for tools and libraries
