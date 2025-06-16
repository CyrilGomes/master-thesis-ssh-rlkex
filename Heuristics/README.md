# Heuristics Component

This component implements various machine learning-based heuristics for SSH key detection and classification in memory graphs.

## Overview

The heuristics component provides two main predictors:
1. Key Count Classifier: Predicts the number of SSH keys in a graph
2. Root Node Predictor: Identifies optimal starting nodes for the DRL agent

## Structure

```
Heuristics/
├── nb_keys_classifier.ipynb    # Key count classifier training
├── nb_keys_classifier.py      # Key count classifier implementation
├── root_heuristic_eval.ipynb # Root predictor evaluation
├── root_heuristic.py        # Root node predictor
└── subtree_heuristic.py    # Subtree analysis utilities
```

## Components

### Key Count Classifier
- **Purpose**: Predicts number of SSH keys (2, 4, or 6) in a memory graph
- **Model**: GATv2Conv-based Graph Neural Network
- **Features**: Graph structure and node attributes
- **Performance**: 100% accuracy on test dataset

### Root Node Predictor
- **Purpose**: Identifies optimal starting nodes for DRL exploration
- **Model**: Random Forest classifier
- **Features**: Node-level and graph-level attributes
- **Performance**: 100% accuracy on test dataset

## Usage

### Key Count Classification

```bash
# Training
python nb_keys_classifier.py --train \
    --data-path /path/to/training/graphs \
    --model-output models/key_classifier.pt

# Prediction
python nb_keys_classifier.py --predict \
    --model-path models/key_classifier.pt \
    --input /path/to/graph.graphml
```

### Root Node Prediction

```bash
# Training
python root_heuristic.py --train \
    --data-path /path/to/training/graphs \
    --model-output models/root_predictor.joblib

# Prediction
python root_heuristic.py --predict \
    --model-path models/root_predictor.joblib \
    --input /path/to/graph.graphml
```
