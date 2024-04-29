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
from torch_geometric.data import Batch
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
    def __init__(self):
        super(GActor, self).__init__()
        self.g_conv_1 = GraphConv(12, 64)
        self.pool_1 = TopKPooling(64, ratio=0.8)
        self.g_conv_2 = GraphConv(64, 64)
        self.pool_2 = TopKPooling(64, ratio=0.8)

        self.action_head_1 = torch.nn.Linear(128, 64)

        self.action_head_5 = torch.nn.Linear(64, 10)
        
        self.dropout = torch.nn.Dropout(0.5)  # Add dropout after each linear layer

    def forward(self, data_prime):
        #For the agent's state (G')
        x_prime, edge_index_prime, batch = data_prime.x, data_prime.edge_index, data_prime.batch
        x_prime = F.relu(self.g_conv_1(x_prime, edge_index_prime))
        x_prime, edge_index_prime, _, batch, _, _ = self.pool_1(x_prime, edge_index_prime, None, batch)
        x1 = torch.cat([gmp(x_prime, batch), gap(x_prime, batch)], dim=1)
#
        x_prime = F.relu(self.g_conv_2(x_prime, edge_index_prime))
        x_prime, edge_index_prime, _, batch, _, _ = self.pool_2(x_prime, edge_index_prime, None, batch)
        x2 = torch.cat([gmp(x_prime, batch), gap(x_prime, batch)], dim=1)
#
        x_prime = x1 + x2

        # Action probabilities with dropout
        action1 = F.relu(self.action_head_1(x_prime))
        action5 = self.action_head_5(action1)
        action_probs = F.softmax(action5, dim=0).unsqueeze(0)
        
        return action_probs

class GCritic(torch.nn.Module):
    def __init__(self):
        super(GCritic, self).__init__()
        self.g_conv_1 = GraphConv(12, 64)
        self.pool_1 = TopKPooling(64, ratio=0.8)
        self.g_conv_2 = GraphConv(64, 64)
        self.pool_2 = TopKPooling(64, ratio=0.8)

        self.action_head_1 = torch.nn.Linear(128, 64)

        self.action_head_5 = torch.nn.Linear(64, 1)
        
        self.dropout = torch.nn.Dropout(0.5)  # Add dropout after each linear layer

    def forward(self, data_prime):
        #For the agent's state (G')
        x_prime, edge_index_prime, batch = data_prime.x, data_prime.edge_index, data_prime.batch
        x_prime = F.relu(self.g_conv_1(x_prime, edge_index_prime))
        x_prime, edge_index_prime, _, batch, _, _ = self.pool_1(x_prime, edge_index_prime, None, batch)
        x1 = torch.cat([gmp(x_prime, batch), gap(x_prime, batch)], dim=1)
#
        x_prime = F.relu(self.g_conv_2(x_prime, edge_index_prime))
        x_prime, edge_index_prime, _, batch, _, _ = self.pool_2(x_prime, edge_index_prime, None, batch)
        x2 = torch.cat([gmp(x_prime, batch), gap(x_prime, batch)], dim=1)
#
        x_prime = x1 + x2

        # Action probabilities with dropout
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
        action_probs = self.actor(data).squeeze(-1)
        state_value = self.critic(data).squeeze(-1)

        #create a mask for softmax, that mask the first action
        return action_probs, state_value
    


state_size = env.state_size
NB_HIDDEN_LAYERS = 1
gactor = GActor()
gcritic = GCritic()
model = ActorCritic(gactor, gcritic)


# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 2**-1
EPISODES = 100000

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
entropy_term = 0.001


rewards = []


stats = {
             "nb_success" : 0,
             }



episode = 0

PPO_EPOCHS = 1
def ppo_train(actor_critic, optimizer, state, action, reward, old_prob, clip_epsilon=0.2):
    for _ in range(PPO_EPOCHS):  # PPO_EPOCHS is a hyperparameter
        prob, value = actor_critic(state)

        ratio = (prob / old_prob)
        advantage = reward - value.detach()
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        critic_loss = F.mse_loss(reward, value)
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


while True:
    episode += 1
    done = False
    state = env.reset()
    episode_reward = 0
    
    episode_stats = {"nb_of_moves" : 0,

             }
    
    printing = False # True if episode > 3000 else False
    is_supervised_prob =  0.5 * (1 - (episode / (EPISODES/20)))
    is_supervised = random.random() < is_supervised_prob
    states = []
    actions = []
    episode_rewards = []
    episode_action_probs = []
    while not done:
        
        action_probs, state_value = model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        if is_supervised:
            best_action = env._get_best_action()

            action = best_action #best action index



        next_state, reward, done, stat = env.step(action, printing = printing)
        if done and stat["found_target"] == True:
            #print(f"Found target with {episode_stats['nb_of_moves'] + 1} moves")
            stats["nb_success"] = stats["nb_success"] + 1 
            
        
        #print(f"Episode : {episode} \t Current Node : {env.current_node} \t Action : {action.item()-1} / {env.action_space.n}")
        #convert reward to tensor
        reward = torch.tensor([reward], dtype=torch.float)
        if is_supervised:
            #chave a best_vector variable that is an array of size 10 (action space), the element at the index of the action is 1, the others are 0

            best_vector = torch.tensor([best_action], dtype=torch.long)  # This creates a tensor of shape (1)
            # Print shapes for debugging

            action_probs = action_probs.squeeze(1)
            supervised_loss = F.cross_entropy(action_probs, best_vector)
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()
        else :
            ppo_train(model, optimizer, state, action, reward, action_probs)


        #states.append(state)
        #actions.append(action)
        #episode_rewards.append(reward)
        #episode_action_probs.append(action_probs)
        episode_reward += reward
        state = next_state
        
        episode_stats["nb_of_moves"] = episode_stats["nb_of_moves"] + 1
        #env.render()
    
    #convert states and actions to tensors
    #states = Batch.from_data_list(states)
    #actions = torch.tensor(actions, dtype=torch.long)
    #episode_rewards = torch.tensor(episode_rewards, dtype=torch.float)
    #add 0 to the end of probs of the action_probs tensors to have the same size
     
    #episode_action_probs = [torch.cat((prob, torch.zeros(1))) for prob in episode_action_probs]
    
    #episode_action_probs = torch.stack(episode_action_probs)


    rewards.append(episode_reward)

    if keyboard.is_pressed('q'):
        break
    #print every 500 episodes :
    if episode % 500 == 0:
        print(f"Episode {episode + 1}: Supervised Prob : {is_supervised_prob} \t Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1}")





#do a rolling mean of the rewards
rewards = np.array(rewards)
rewards = np.convolve(rewards, np.ones((100,))/100, mode='valid')

plt.plot(rewards)
plt.show()