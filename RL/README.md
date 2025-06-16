# Reinforcement Learning Component

This component implements the Deep Reinforcement Learning (DRL) approach for SSH key retrieval in memory graphs.

## Overview

The DRL component uses a novel goal-oriented approach combining Graph Neural Networks with Deep Q-Learning to navigate memory graphs and locate SSH keys.

## Structure

```
RL/
├── actor_critic_branch_selection.py
├── test_graph_embedding.py
├── train_test_spliter.py
└── GNNDRL/                           # Main implementation
    ├── fixed_action_space/          # Fixed action space variants
    ├── fixed_action_space_goal/     # Goal-oriented fixed space
    ├── variable_action_space_goal/  # Main thesis implementation
    └── rl_base/                     # Base components
        ├── gnn.py
        ├── rl_environment.py
        └── ...
```

## Main Implementation (variable_action_space_goal)

### Architecture
- **State Space**: Graph structure + current node + visited nodes
- **Action Space**: Variable, based on current node's neighbors
- **Goals**: Specific key type to find (Key A, B, C, or D)
- **Model**: GATv2Conv layers + goal conditioning

### Features
- Goal-oriented training
- Prioritized Experience Replay (PER)
- Double Deep Q-Network (DDQN)
- Variable action space handling
- BFS-based state representation

## Usage

### Training

```bash
cd GNNDRL/variable_action_space_goal
python graph_obs_variable_action_space_GDQL.py --train \
    --training-data /path/to/training/graphs \
    --test-data /path/to/test/graphs \
    --model-output models/model.pt
```

### Evaluation

```bash
python graph_obs_variable_action_space_GDQL.py --eval \
    --model-path models/model.pt \
    --input-graph /path/to/graph.graphml
```

## Performance

- 95% accuracy on validation dataset
- Effective with limited training data (7 files per subset)
- Generalizes well across OpenSSH versions

## Model Architecture

### GNN Component
- Input: Graph structure + node features
- Multiple GATv2Conv layers
- Edge feature integration
- Goal conditioning

### Q-Network
- State encoding through GNN
- Action value prediction
- Independent Q-value per neighbor
- Goal integration through concatenation
