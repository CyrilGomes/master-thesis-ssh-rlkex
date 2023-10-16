# -------------------------
# IMPORTS AND SETUP
# -------------------------

import os
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import keyboard
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool
from torch_geometric.utils import from_networkx
from torch.distributions import Categorical
from rl_base.rl_environment import GraphTraversalEnv
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# GRAPH PROCESSING
# -------------------------

def preprocess_graph(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    
    # Removing string attributes from nodes and edges
    for _, attributes in graph.nodes(data=True):
        for key in list(attributes):
            if isinstance(attributes[key], str):
                del attributes[key]

    for u, v, attributes in graph.edges(data=True):
        for key in list(attributes):
            if isinstance(attributes[key], str):
                del attributes[key]

    nx.set_node_attributes(graph, 0, 'visited')
    graph = nx.subgraph(graph, nx.bfs_tree(graph, 0))
    return graph

def load_graphs_from_directory(directory_path):
    graph_files = [f for f in os.listdir(directory_path) if f.endswith('.graphml')]
    graphs = [nx.read_graphml(os.path.join(directory_path, f)) for f in graph_files]
    return [preprocess_graph(g) for g in graphs]

def graph_to_data(graph):
    x = torch.tensor([[
        attributes['struct_size'],
        attributes['valid_pointer_count'],
        attributes['invalid_pointer_count'],
        attributes['first_pointer_offset'],
        attributes['last_pointer_offset'],
        attributes['first_valid_pointer_offset'],
        attributes['last_valid_pointer_offset'],
        attributes['visited']
    ] for _, attributes in graph.nodes(data=True)], dtype=torch.float)
    
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([graph[u][v]['offset'] for u, v in graph.edges], dtype=torch.float).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# -------------------------
# MODEL DEFINITION
# -------------------------
class PolicyNetwork(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size):
        super(PolicyNetwork, self).__init__()
        self.sage_conv1 = SAGEConv(node_feature_size, 64)
        self.sage_conv2 = SAGEConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.dropout = torch.nn.Dropout(0.5)

        self.saved_log_probs = []
        self.rewards = []

    def normalize_features(self, data):
        feature_mean = data.x.mean(dim=0)
        feature_std = data.x.std(dim=0)
        feature_std = feature_std + 1e-10
        data.x = (data.x - feature_mean) / feature_std
        return data

    def forward(self, data, data_prime):
        data_prime.to(device)
        print(data_prime.x)
        x = F.relu(self.sage_conv1(data_prime.x, data_prime.edge_index))
        x = F.relu(self.sage_conv2(x, data_prime.edge_index))
        x_global = global_mean_pool(x, torch.zeros(data_prime.num_nodes, dtype=torch.long).to(device))
        
        action_probs = F.relu(self.fc1(x_global))
        action_probs = self.fc2(action_probs)
        action_probs = F.softmax(action_probs, dim=-1)
        
        return action_probs


# -------------------------
# TRAINING & EVALUATION
# -------------------------

def weights_init_he(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque(maxlen=100)
    for r in model.rewards[::-1]:
        R = r + 0.99 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
    for log_prob, R in zip(model.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]

def select_action(state, is_supervised=False):
    state = state.to(device)
    probs = model(data, state)
    print(probs)
    m = Categorical(probs)
    action = m.sample().item()
    if is_supervised:
        best_action = env._get_best_action()
        action = best_action
    model.saved_log_probs.append(m.log_prob(torch.tensor(action, dtype=torch.long).to(device)))
    return action

# -------------------------
# MAIN EXECUTION
# -------------------------

# Load and preprocess graph
FILE = "Generated_Graphs/64/18038-1643986141-heap.graphml"
graph = nx.read_graphml(FILE)
graph = preprocess_graph(graph)
target_node = next((node for node, attributes in graph.nodes(data=True) if attributes["cat"] == 1), None)

data = graph_to_data(graph)
env = GraphTraversalEnv(graph, target_node)
model = PolicyNetwork(env.state_size, 1).to(device)
#model.apply(weights_init_he)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

episode = 0
success_window_size = 100
windowed_success = 0
num_episodes = 10000
stats = {"nb_success": 0}
total_successes = []
while True:
    if keyboard.is_pressed('q'):
        break
    episode += 1
    observation = env.reset()
    episode_reward = 0
    episode_stats = {"nb_of_moves": 0}
    is_supervised_prob = 0.5 * (1 - (episode / num_episodes))
    is_supervised = random.random() < is_supervised_prob
    while True:
        data_prime = observation
        action = select_action(data_prime, is_supervised)
        new_observation, reward, done, info = env.step(action, False)
        model.rewards.append(reward)
        episode_reward += reward
        episode_stats["nb_of_moves"] += 1
        if done:
            if info["found_target"]:
                stats["nb_success"] += 1
                windowed_success += 1
            break
        observation = new_observation
    finish_episode()
    if episode % success_window_size == 0:
        total_successes.append(windowed_success / success_window_size)
        windowed_success = 0
    if episode % 500 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1} \t is_supervised_prob = {is_supervised_prob}")

# Visualization
window_size = 10
success_array = np.array(total_successes)
success = np.convolve(success_array, np.ones((window_size,))/window_size, mode='valid')
plt.plot(success)
plt.plot([0, len(success)], [is_supervised_prob, is_supervised_prob], 'k-', lw=2)
plt.show()

# Save Model
save_model = input("Do you want to save the model? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name: ")
    torch.save(model.state_dict(), f"models/{model_name}.pt")
