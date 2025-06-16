# SSH-RLKEX: Reinforcement Learning for SSH Key Retrieval using Graph Structures

This repository contains the implementation of my master thesis research on utilizing Reinforcement Learning for SSH key retrieval in memory dumps using graph-based approaches.

## Abstract

[Brief abstract of your thesis - describing the problem, approach, and main findings]

## Project Structure

```
.
├── Graph_Gen/               # Rust project for converting heap dumps to graphs
├── Graph_Insights/         # Analysis tools for understanding graph properties
├── Heuristics/            # Implementation of various heuristic approaches
├── RL/                    # Reinforcement Learning implementations
│   └── GNNDRL/           # Graph Neural Network + Deep RL implementation
└── models/               # Trained model checkpoints
```

## Main Components

1. **Graph Generation (Graph_Gen/)**
   - Rust implementation for efficient heap dump to graph conversion
   - See `Graph_Gen/README.md` for usage instructions

2. **Graph Analysis (Graph_Insights/)**
   - Tools for analyzing graph properties
   - Visualization utilities
   - Path analysis tools

3. **Heuristic Approaches (Heuristics/)**
   - `nb_keys_classifier.py`: Implementation of key number classifier
   - `root_heuristic.py`: Root node prediction implementation
   - Jupyter notebooks for training and analysis

4. **Reinforcement Learning (RL/GNNDRL/)**
   - Main implementation used in the thesis: `variable_action_space_goal/`
   - Various experimental approaches in other directories
   - GNN-based DRL implementations

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/SSH-RLKEX.git
cd SSH-RLKEX

# Create and activate conda environment
conda env create -f environment.yml
conda activate ssh-rlkex

# Install Rust for graph generation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Usage

[Add specific instructions on how to use your code, including examples]

## Reproduction

The exact code version used in the thesis can be found in commit [cbe1cd8](https://github.com/CyrilGomes/RLKEX/commit/cbe1cd8d46d8f5c301fd0e241eeec8a2538802f9).

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
[Add your thesis citation here]
```

## License

[Add your chosen license]

## Links

- [Thesis PDF](link-to-your-thesis-pdf)
- [ArXiv Paper](link-if-applicable)
- [Project Page](link-if-applicable)

