# Reinforcement Learning Component

This component contains the implementation of various Reinforcement Learning approaches for SSH key retrieval, with the main focus on Graph Neural Network-based Deep RL methods.

## Structure

```
RL/
├── actor_critic_branch_selection.py
├── test_graph_embedding.py
├── train_test_spliter.py
└── GNNDRL/                           # Main GNN + DRL implementations
    ├── fixed_action_space/          # Fixed action space approach
    ├── fixed_action_space_goal/     # Goal-oriented fixed action space
    ├── variable_action_space_goal/  # Main implementation used in thesis
    └── rl_base/                     # Base RL components
        ├── gnn.py
        ├── rl_environment.py
        └── ...
```

## Main Components

### Variable Action Space with Goal (Main Thesis Implementation)
Located in `GNNDRL/variable_action_space_goal/`
- Dynamic action space based on graph structure
- Goal-oriented reward system
- GNN-based state representation

### Base Components (RL_base)
- `gnn.py`: Graph Neural Network architectures
- `rl_environment.py`: Base RL environment implementation
- `rl_environment_full_env.py`: Full graph environment
- `rl_environment_key_detect_single_state.py`: Single state key detection

### Alternative Approaches
- Fixed action space implementations
- Actor-Critic implementations
- DQN variants

## Training

```bash
# Train the main model (variable action space with goal)
cd GNNDRL/variable_action_space_goal
python train.py --config configs/default.yml

# Evaluate
python evaluate.py --model-path /path/to/model --test-graphs /path/to/test/data
```

## Model Architecture

[Add description of your GNN architecture and RL approach]

## Results

[Add key results from your thesis]

## Pre-trained Models

Pre-trained models are available in the `models/` directory:
- `model.pt`: Main model used in thesis
- `complex_model_0_79.pt`: Alternative model with 0.79 performance

## Citations

If you use this code, please cite our work:
[Add citation]
