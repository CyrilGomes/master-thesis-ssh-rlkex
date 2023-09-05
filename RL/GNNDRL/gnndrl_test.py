import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import gym
import networkx as nx
import numpy as np
import os
from rl_base.rl_environment import GraphTraversalEnv
import torch.nn.functional as F
def preprocess_graph(graph):
    # Relabel the nodes to use integers
    graph = nx.convert_node_labels_to_integers(graph)
    
    # Filter out string attributes from nodes
    for node, attributes in list(graph.nodes(data=True)):
        for key, value in list(attributes.items()):
            if isinstance(value, str):
                del graph.nodes[node][key]

    # Filter out string attributes from edges
    for u, v, attributes in list(graph.edges(data=True)):
        for key, value in list(attributes.items()):
            if isinstance(value, str):
                del graph[u][v][key]

    return graph



def load_graphs_from_directory(directory_path):
    graph_files = [f for f in os.listdir(directory_path) if f.endswith('.gml')]
    graphs = [nx.read_gml(os.path.join(directory_path, graph_file)) for graph_file in graph_files]
    return [preprocess_graph(g) for g in graphs]



class GActor(torch.nn.Module):
    def __init__(self,state_size, hidden_channels):
        super(GActor, self).__init__()
        self.conv1 = GCNConv(state_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x
    
class GCritic(torch.nn.Module):
    def __init__(self,state_size, hidden_channels):
        super(GCritic, self).__init__()
        self.conv1 = GCNConv(state_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


import torch.optim as optim



FILE = "Generated_Graphs/64/18038-1643986141-heap.graphml"
graph = nx.read_graphml(FILE)
graph = preprocess_graph(graph)
target_node = None
#loop over the graph, if a node has cat = 1, then save it in target_node
for node, attributes in graph.nodes(data=True):
    if attributes["cat"] == 1:
        target_node = node
        break

env = GraphTraversalEnv(graph=graph, target_node=target_node)

# Actor-Critic architecture

class ActorCritic(torch.nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x, edge_index):
        action_scores = self.actor(x, edge_index).squeeze(-1)
        all_state_values = self.critic(x, edge_index).squeeze(-1)
        state_value = all_state_values[0]  # assuming the current node is the first in the list

        #create a mask for softmax, that mask the first action
        mask = torch.ones(action_scores.shape, dtype=torch.bool)
        mask[0] = 0
        action_scores_masked = action_scores.masked_fill(~mask, float('-inf'))
        action_probs = F.softmax(action_scores_masked, dim=-1)
        return action_probs, state_value
    


state_size = env.state_size
gactor = GActor(state_size, 64)
gcritic = GCritic(state_size, 64)
model = ActorCritic(gactor, gcritic)

optimizer = optim.Adam(model.parameters())

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES = 1000

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
entropy_term = 0.001

for episode in range(EPISODES):
    done = False
    state = env.reset()
    episode_reward = 0
    
    while not done:
        action_probs, state_value = model(state['x'], state['edge_index'])

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        next_state, reward, done, _ = env.step(action.item())
        #print(f"Episode : {episode} \t Current Node : {env.current_node} \t Action : {action.item()-1} / {env.action_space.n}")
        
        _, next_value = model(next_state['x'], next_state['edge_index'])
        
        # Compute TD error
        td_error = reward + GAMMA * next_value * (1-int(done)) - state_value
        actor_loss = -action_dist.log_prob(action) * td_error.detach()  # detach to avoid double backprop
        critic_loss = td_error**2
        
        entropy = entropy_term * action_dist.entropy().mean()
        
        loss = actor_loss + critic_loss - entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_reward += reward
        state = next_state
        
    print(f"Episode {episode + 1}: Reward = {episode_reward}")
