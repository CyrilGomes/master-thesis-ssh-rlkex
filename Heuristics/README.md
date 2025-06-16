# Heuristics Component

This component implements various heuristic approaches for SSH key detection and classification in memory graphs.

## Structure

```
Heuristics/
├── nb_keys_classifier.ipynb    # Training notebook for key number classifier
├── nb_keys_classifier.py      # Key number classifier implementation
├── root_heuristic_eval.ipynb # Evaluation notebook for root heuristic
├── root_heuristic.py        # Root node prediction implementation
└── subtree_heuristic.py    # Subtree-based heuristic implementation
```

## Components

### Key Number Classifier
- Implementation: `nb_keys_classifier.py`
- Training: `nb_keys_classifier.ipynb`
- Purpose: Predicts the number of SSH keys in a memory graph
- Features used: [Add key features used for classification]

### Root Heuristic
- Implementation: `root_heuristic.py`
- Evaluation: `root_heuristic_eval.ipynb`
- Purpose: Predicts potential root nodes for key extraction
- Model: Random Forest (saved in `../models/root_heuristic_model.joblib`)

### Subtree Heuristic
- Implementation: `subtree_heuristic.py`
- Purpose: Analyzes subtree patterns for key detection

## Usage

### Training the Key Number Classifier

```bash
# Using the notebook
jupyter notebook nb_keys_classifier.ipynb

# Or using the Python script
python nb_keys_classifier.py --train --data-path /path/to/training/data
```

### Using the Root Heuristic

```python
from root_heuristic import RootPredictor

predictor = RootPredictor()
root_nodes = predictor.predict(graph)
```

## Model Performance

[Add performance metrics and comparisons from your thesis]
