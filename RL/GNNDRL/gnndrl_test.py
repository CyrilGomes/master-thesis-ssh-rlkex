import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import gym
import networkx as nx
import numpy as np
import os
from rl_base.rl_environment_negative_reward import GraphTraversalEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import keyboard

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


    #add the "visited" property, starting at 0 to each node
    for node, attributes in graph.nodes(data=True):
        graph.nodes[node]["visited"] = 0
    



    #use BFS to get the tree
     
    graph = nx.subgraph(graph, nx.bfs_tree(graph, 0))
    print(len(graph.nodes))

    return graph



def load_graphs_from_directory(directory_path):
    graph_files = [f for f in os.listdir(directory_path) if f.endswith('.graphml')]
    graphs = [nx.read_graphml(os.path.join(directory_path, graph_file)) for graph_file in graph_files]
    return [preprocess_graph(g) for g in graphs]


"""
class GActor(torch.nn.Module):
    def __init__(self,state_size, hidden_channels,nb_hidden_layers=3):
        super(GActor, self).__init__()
        self.conv1 = GCNConv(state_size, hidden_channels)
        self.conv2 = [GCNConv(hidden_channels, hidden_channels) for i in range(nb_hidden_layers)]
        self.conv3 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        for conv in self.conv2:
            x = conv(x, edge_index).relu()

        x = self.conv3(x, edge_index)
        return x
    
class GCritic(torch.nn.Module):
    def __init__(self,state_size, hidden_channels, nb_hidden_layers=3):
        super(GCritic, self).__init__()
        self.conv1 = GCNConv(state_size, hidden_channels)
        self.conv2 = [GCNConv(hidden_channels, hidden_channels) for i in range(nb_hidden_layers)]
        self.conv3 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        for conv in self.conv2:
            x = conv(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

"""
class GActor(torch.nn.Module):
    def __init__(self,state_size, hidden_channels,nb_hidden_layers=3):
        super(GActor, self).__init__()
        self.global_conv1 = GCNConv(state_size, hidden_channels)
        self.global_conv2 = [GCNConv(hidden_channels, hidden_channels) for i in range(nb_hidden_layers)]
        self.global_conv3 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.global_conv1(x, edge_index).relu()
        for conv in self.global_conv2:
            x = conv(x, edge_index).relu()

        x = self.global_conv3(x, edge_index)
        return x
    
class GCritic(torch.nn.Module):
    def __init__(self,state_size):
        super(GCritic, self).__init__()
        self.g_conv_1 = GraphConv(state_size, 64)
        self.g_conv_2 = GraphConv(state_size, 64)
        #self.g_prime_conv = GCNConv(8, 128)

        self.action_head_1 = torch.nn.Linear(12, 11)

        self.action_head_5 = torch.nn.Linear(11, 1)

    def forward(self, data):
        x_prime, edge_index_prime = data.x, data.edge_index

        x1 = F.relu(self.g_conv_1(x_prime, edge_index_prime))
        x1 = gap(x_prime, torch.zeros(len(x_prime), dtype=torch.long))


        x2 = F.relu(self.g_conv_2(x_prime, edge_index_prime))
        x2 = gap(x_prime, torch.zeros(len(x_prime), dtype=torch.long))

        x_prime = x1 + x2
        action1 = F.relu(self.action_head_1(x_prime))
        

        action5 = self.action_head_5(action1)
        return action5
         
    

import torch.optim as optim



FILE = "Generated_Graphs/64/18038-1643986141-heap.graphml"
graph = nx.read_graphml(FILE)
graph = preprocess_graph(graph)
target_node = None
#loop over the graph, if a node has cat = 1, then save it in target_node
for node, attributes in graph.nodes(data=True):
    if attributes["cat"] == 1:
        target_node = node
        print("Found : " + str(node))
        break

env = GraphTraversalEnv(graph=graph, target_node=target_node)

# Actor-Critic architecture

class ActorCritic(torch.nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, data):
        action_scores = self.actor(data).squeeze(-1)
        state_value = self.critic(data).squeeze(-1)

        #create a mask for softmax, that mask the first action
        mask = torch.ones(action_scores.shape, dtype=torch.bool)
        mask[0] = 0
        action_scores_masked = action_scores.masked_fill(~mask, float('-inf'))
        action_probs = F.softmax(action_scores_masked, dim=-1)
        return action_probs, state_value
    


state_size = env.state_size
NB_HIDDEN_LAYERS = 1
gactor = GActor(state_size, 128, NB_HIDDEN_LAYERS)
gcritic = GCritic(state_size)
model = ActorCritic(gactor, gcritic)


# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 2**-15
EPISODES = 100000

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
entropy_term = 0.001


rewards = []


stats = {
             "nb_success" : 0,
             }


losses = []
td_errors = []
entropies = []
actor_losses = []
critic_losses = []
episode = 0
while True:
    episode += 1
    done = False
    state = env.reset()
    episode_reward = 0
    
    episode_stats = {"nb_of_moves" : 0,

             }
    
    printing = False # True if episode > 3000 else False
    is_supervised_prob = 0.5 * (1 - (episode / (EPISODES/20)))
    is_supervised = random.random() < is_supervised_prob
    
    while not done:
        action_probs, state_value = model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item() - 1

        if is_supervised:
            best_action = env._get_best_action()

            action = best_action

        next_state, reward, done, stat = env.step(action, printing = printing)
        if done and stat["found_target"] == True:
            #print(f"Found target with {episode_stats['nb_of_moves'] + 1} moves")
            stats["nb_success"] = stats["nb_success"] + 1 
            

        #print(f"Episode : {episode} \t Current Node : {env.current_node} \t Action : {action.item()-1} / {env.action_space.n}")
        
        _, next_value = model(next_state)
        
        # Compute TD error
        td_error = reward + GAMMA * next_value * (1-int(done)) - state_value
        actor_loss = -action_dist.log_prob(torch.tensor(action)) * td_error.detach()  # detach to avoid double backprop
        critic_loss = td_error**2
        
        entropy = entropy_term * action_dist.entropy().mean()
        
        loss = actor_loss + critic_loss - entropy
        
        losses.append(loss.item())
        td_errors.append(td_error.item())
        entropies.append(entropy.item())
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_reward += reward
        state = next_state

        episode_stats["nb_of_moves"] = episode_stats["nb_of_moves"] + 1
        #env.render()
        
    rewards.append(episode_reward)

    if keyboard.is_pressed('q'):
        break
    #print every 500 episodes :
    if episode % 500 == 0:
        print(f"Episode {episode + 1}: Supervised Prob : {is_supervised_prob} \t Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1}")





#plot losses
plt.plot(losses)
plt.plot(td_errors)
plt.plot(entropies)
plt.plot(actor_losses)
plt.plot(critic_losses)
plt.legend(["loss", "td_error", "entropy", "actor_loss", "critic_loss"])
plt.show()



#do a rolling mean of the rewards
rewards = np.array(rewards)
rewards = np.convolve(rewards, np.ones((100,))/100, mode='valid')

plt.plot(rewards)
plt.show()