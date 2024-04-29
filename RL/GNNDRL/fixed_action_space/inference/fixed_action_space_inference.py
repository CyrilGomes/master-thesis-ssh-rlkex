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
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_scatter import scatter_max

from fixed_action_space_env import GraphTraversalEnv
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
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from root_heuristic_rf import GraphPredictor

from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling, global_add_pool, global_max_pool
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch.utils.tensorboard import SummaryWriter

import heapq  # For priority queue
import time
from fixed_action_space_agent_inference import Agent
from utils import preprocess_graph, convert_types, add_global_root_node, connect_components, remove_all_isolated_nodes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


# -------------------------
# HYPERPARAMETERS
# -------------------------



def check_parameters(env):

    if env.observation_space.spaces['x'].shape[0] != STATE_SPACE:
        raise ValueError("State space is not correct")
    
    
def test_for_graph(graph, target_nodes, agent, root_detector):
    """Basically the same as the training function, but without training"""

    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes,ACTION_SPACE , root_detector)
    check_parameters(env)

    
    observation = env.reset()
    done = False
    total_reward = 0
    total_key_found = 0

    while not done:
        action_mask = env._get_action_mask()
        action = agent.act(observation,action_mask)
        new_observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:

            total_key_found = info["nb_keys_found"]
            break
        
        observation = new_observation
    
    return total_reward, total_key_found, len(target_nodes)

FOLDER = "Generated_Graphs/output/"
STATE_SPACE = 13
EDGE_ATTR_SIZE = 1
ACTION_SPACE = 50



def main():

    model_path = "/home/cyril/ssh-rlkex/models/wow.pt"
    agent = Agent(STATE_SPACE,EDGE_ATTR_SIZE, ACTION_SPACE,  seed=0, device=device, local_network_path=model_path)

    #get all files in the folder recursively
    all_files = []
    for root, dirs, files in os.walk(FOLDER):
        for file in files:
            if file.endswith(".graphml"):
                all_files.append(os.path.join(root, file))
    
    root_detection_model_path="/home/cyril/ssh-rlkex/models/root_heuristic_model.joblib"
    root_detector = GraphPredictor(root_detection_model_path)

    nb_random_files = 1000
    random_files = random.sample(all_files, nb_random_files)

    graphs = []
    target_nodes_per_graph = []
    for file in tqdm(random_files):
        graph = nx.read_graphml(file)
        graph = preprocess_graph(graph)

        #get all target_nodes, check if nodes has 'cat' = 1
        target_nodes = [node for node, attributes in graph.nodes(data=True) if attributes['cat'] == 1]
        graphs.append(graph)
        target_nodes_per_graph.append(target_nodes)


    print(f"Testing on {nb_random_files} random files")
    nb_success = 0
    nb_key_found_ratio = []
    for i, graph in tqdm(enumerate(graphs)):
        total_reward, total_key_found, nb_keys = test_for_graph(graph,target_nodes_per_graph[i], agent, root_detector)
        if total_key_found == nb_keys:
            nb_success += 1
        nb_key_found_ratio.append(total_key_found / nb_keys)

    print(f"Success rate : {nb_success / nb_random_files}")
    print(f"Average key found ratio : {np.mean(nb_key_found_ratio)}")






if __name__ == "__main__":
    #load the agent
    main()



