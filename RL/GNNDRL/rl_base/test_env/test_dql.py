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
from torch_geometric.nn import SAGEConv, global_mean_pool

from test_environment import GraphTraversalEnv
from collections import deque
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import optim
from collections import namedtuple, deque
#import range tqdm
from tqdm import tqdm
from tqdm import trange
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader, Batch
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# GRAPH PROCESSING
# -------------------------




def connect_components(graph):
    undi_graph = graph.to_undirected()

    # Connect components
    components = list(nx.connected_components(undi_graph))
    for i in range(1, len(components)):

        graph.add_edge(random.choice(list(components[0])), random.choice(list(components[i])), offset=0)
        graph.add_edge(random.choice(list(components[i])), random.choice(list(components[0])), offset=0)

    return graph

def add_global_root_node(graph):
    root_node = "root"
    graph.add_node(root_node, label=root_node, cat=0, struct_size=0, pointer_count=0, valid_pointer_count=0, invalid_pointer_count=0, first_pointer_offset=0, last_pointer_offset=0, first_valid_pointer_offset=0, last_valid_pointer_offset=0, visited=1)
    [graph.add_edge(root_node, node, offset=0) for node in graph.nodes() if len(list(graph.predecessors(node))) == 0 and node != root_node]
    return graph


def remove_all_isolated_nodes(graph):
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph
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
    #graph = connect_components(graph)
    #graph = nx.subgraph(graph, nx.bfs_tree(graph, 0))
    graph = remove_all_isolated_nodes(graph)
    #graph = add_global_root_node(graph)

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
# HYPERPARAMETERS
# -------------------------
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # batch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network



# -------------------------
# MODEL DEFINITION
# -------------------------
class QNetwork(torch.nn.Module):
    def __init__(self, node_feature_size, action_size, seed, dropout_rate=0.2):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = torch.nn.Linear(node_feature_size, 128)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = torch.nn.Linear(128, 128)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        # Dueling DQN
        self.value_stream = torch.nn.Linear(128, 128)
        self.value = torch.nn.Linear(128, 1)
        
        self.advantage_stream = torch.nn.Linear(128, 128)
        self.advantage = torch.nn.Linear(128, action_size)

    def forward(self, state, action_mask):
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        value = F.relu(self.value_stream(x))
        value = self.value(value)

        advantage = F.relu(self.advantage_stream(x))
        advantage = self.advantage(advantage)

        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Apply the action mask: set the value of masked actions to a large negative number
        masked_qvals = qvals + action_mask

        return masked_qvals



# -------------------------
# AGENT DEFINITION
# -------------------------
# DQNAgent for action selection
class DQNAgent:
    def __init__(self, model_path, node_feature_size, action_size, seed, dropout_rate=0.2):
        self.q_network = QNetwork(node_feature_size, action_size, seed, dropout_rate)
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval()  # Set the network to evaluation mode

    def select_action(self, state, action_mask):
        """
        Selects the action with the highest Q-value given a state and action mask.
        :param state: The current state representation
        :param action_mask: A mask that indicates valid actions (e.g., -1e10 for invalid actions)
        :return: Index of the action with the highest Q-value
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action_mask = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():  # No need to track gradients
            q_values = self.q_network(state, action_mask)
            action = q_values.argmax(dim=1).item()  # Get the index of the max Q-value
        return action


# -------------------------
# MAIN EXECUTION
# -------------------------

# Load and preprocess graph

FOLDER = "Generated_Graphs/64/"
ACTION_SPACE = 50
STATE_SPACE = 11

agent = DQNAgent("models/test.pt", STATE_SPACE, ACTION_SPACE, 0)



def check_parameters(env):
    #check if state space and action space are correct
    if env.action_space.n != ACTION_SPACE:
        raise ValueError("Action space is not correct")
    if env.observation_space.spaces['x'].shape[0] != STATE_SPACE:
        raise ValueError("State space is not correct")
    
    

def execute_for_graph(file):
    graph= nx.read_graphml(file)
    graph_post = preprocess_graph(graph)



    env = GraphTraversalEnv(graph_post)

    check_parameters(env)
    observation = env.reset()   
    while True:
        first_node = observation.x[0]
        action_mask = env._get_action_mask()    
        action = agent.select_action(first_node,action_mask)
        new_observation, done, info = env.step(action)  
        if done:    
            found_keys = info['found_keys']
            print(f"Done ! The found keys are {found_keys}")
            #print the data of the found keys from the graph
            for key in found_keys:
                print(f"Data for key {key}: {graph.nodes[str(key)]}")
            
            break
        observation = new_observation

    

# Testing

#take N random graphs from the folder and execute
N = 10
file_names = os.listdir(FOLDER)
#keep only the first 100 graphs
filtered_file_names = file_names[0:N]
for file_name in filtered_file_names:
    execute_for_graph(os.path.join(FOLDER, file_name))

