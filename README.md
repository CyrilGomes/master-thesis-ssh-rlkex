# Master Thesis: SSH Key Retrieval using Graph-based Reinforcement Learning

## Overview
This repository contains the implementation of my master thesis research at Universität Passau, focusing on extracting SSH session keys from heap memory dumps using Deep Reinforcement Learning and graph-based approaches.

## Author & Supervision
- **Author**: Cyril GOMES
- **Institution**: Universität Passau
- **Supervisors**: 
  - Prof. Dr. Michael Granitzer
  - Prof. Dr. Harald Kosch

## Abstract
This project introduces a novel approach for extracting SSH session keys directly from heap memory dumps of an OpenSSH process. The core of this method is the use of Deep Reinforcement Learning (DRL) to navigate a graph-based representation of the heap memory. This approach is designed to be highly effective even with limited training data and to generalize well to new, unseen data (such as new OpenSSH versions) with minimal or no retraining. The goal is to create a powerful digital forensics tool that can non-intrusively decrypt SSH communications to analyze malicious activity or monitor honeypots.

## Research Contributions
1. **Pointer Graph Generation Pipeline**: High-performance Rust implementation for converting heap dumps into pointer graphs
2. **Root Node Predictor**: Classifier for identifying optimal starting nodes (100% accuracy)
3. **Key Count Classifier**: GNN model for predicting SSH key counts (100% accuracy)
4. **Novel DRL Agent**: Goal-oriented Deep Q-Learning agent with 95% accuracy in key retrieval

## Repository Structure
```
.
├── Graph_Gen/               # Memory dump to graph converter (Rust)
├── Graph_Insights/         # Graph analysis and visualization tools
├── Heuristics/            # ML-based heuristic models
├── RL/                    # Reinforcement Learning implementation
│   └── GNNDRL/           # Graph Neural Network + DRL
├── models/               # Pre-trained model checkpoints
└── docs/                # Thesis document and documentation
```

## Key Results
1. **Root Predictor Performance**: 
   - 100% accuracy on test dataset
   - Successful classification of valid root nodes

2. **Key Count Prediction**: 
   - 100% accuracy on test dataset
   - Reliable prediction of 2, 4, or 6 keys

3. **DRL Agent Performance**:
   - 95% overall accuracy on validation dataset
   - High success rate across OpenSSH versions
   - Effective with limited training data (7 files per sub-folder)

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/CyrilGomes/master-thesis-ssh-rlkex.git
cd master-thesis-ssh-rlkex

# Create and activate conda environment
conda env create -f environment.yml
conda activate ssh-rlkex

# Install Rust for graph generation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Usage Guide

### 1. Graph Generation
```bash
cd Graph_Gen/graph_gen_rust
cargo run --release -- /path/to/heap.dump /path/to/output/graph.graphml
```

### 2. Key Count Prediction
```bash
cd Heuristics
python nb_keys_classifier.py --data-path /path/to/graph.graphml
```

### 3. Root Node Prediction
```bash
cd Heuristics
python root_heuristic.py --input /path/to/graph.graphml
```

### 4. Key Retrieval
```bash
cd RL/GNNDRL/variable_action_space_goal
python graph_obs_variable_action_space_GDQL.py --eval \
    --model-path models/model.pt \
    --input-graph /path/to/graph.graphml
```

## Technology Stack
- **Languages**: Python, Rust (graph generation)
- **Deep Learning**: PyTorch, PyTorch Geometric
- **Machine Learning**: Scikit-learn
- **Graph Processing**: NetworkX

## Citation
If you use this code or find our work helpful, please cite:

```bibtex
@mastersthesis{gomes2025rlkex,
  author      = {Gomes, Cyril},
  title       = {SSH Key Retrieval using Graph-based Reinforcement Learning},
  school      = {Universität Passau},
  year        = {2025},
  address     = {Passau, Germany},
  month       = {1},
  type        = {Master's Thesis},
  supervisor  = {Granitzer, Michael and Kosch, Harald}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
[1] Fellicious et al., "Memory Dumps Analysis for Forensic Investigation", 2022.

## Documentation
- [Full Thesis PDF](docs/master_thesis_cyril_gomes.pdf)
- [Technical Documentation](docs/)
