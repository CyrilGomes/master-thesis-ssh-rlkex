import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.distributions import Categorical
import networkx as nx
import gym
from rl_base.rl_environment import GraphTraversalEnv
import os
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from collections import deque
import numpy as np
import keyboard


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


from torch_geometric.nn import GATConv

from torch_geometric.nn import NNConv



class PolicyNetwork(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size):
        super(PolicyNetwork, self).__init__()

        self.g_conv = GCNConv(5, 128)
        self.g_prime_conv = GCNConv(8, 128)

        self.action_head_1 = torch.nn.Linear(256, 128)
        self.action_head_2 = torch.nn.Linear(128, 64)
        self.action_head_3 = torch.nn.Linear(64, 32)
        self.action_head_4 = torch.nn.Linear(32, 16)
        self.action_head_5 = torch.nn.Linear(16, 10)
        
        self.dropout = torch.nn.Dropout(0.5)  # Add dropout after each linear layer

        self.saved_log_probs = []
        self.rewards = []


    def normalize_features(self, data):
        """
        Normalize the node features of a PyTorch Geometric Data object.
        
        Args:
        - data (torch_geometric.data.Data): The Data object containing node features to normalize.

        Returns:
        - torch_geometric.data.Data: The Data object with normalized node features.
        """
        # Calculate mean and standard deviation for each feature
        feature_mean = data.x.mean(dim=0)
        feature_std = data.x.std(dim=0)
        
        # Avoid division by zero by adding a small value
        feature_std = feature_std + 1e-10
        
        # Normalize features
        data.x = (data.x - feature_mean) / feature_std

        return data

    def forward(self, data, data_prime):


        # For the main graph
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.g_conv(x, edge_index))
        x_global = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long).to(device))

        # For the agent's state (G')
        x_prime, edge_index_prime, edge_attr_prime = data_prime.x, data_prime.edge_index, data_prime.edge_attr
        x_prime = F.relu(self.g_prime_conv(x_prime, edge_index_prime))
        x_prime_global = global_mean_pool(x_prime, torch.zeros(data_prime.num_nodes, dtype=torch.long).to(device))

        # Combine the embeddings
        combined_x = torch.cat([x_global, x_prime_global], dim=1).to(device)

        # Action probabilities with dropout
        action1 = F.relu(self.action_head_1(combined_x))
        
        action2 = F.relu(self.action_head_2(action1))

        action3 = F.relu(self.action_head_3(action2))

        action4 = F.relu(self.action_head_4(action3))

        action5 = self.action_head_5(action4)
        action_probs = F.softmax(action5, dim=-1)
         
    
        return action_probs

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


def graph_to_data(graph):
    # Node features
    x = torch.tensor([[
            data['struct_size'],
            data['pointer_count'],
            data['valid_pointer_count'],
            data['invalid_pointer_count'],
            data['visited']
        ] for node, data in graph.nodes(data=True)], dtype=torch.float)
    
    # Edge indices
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Edge features (assuming a single feature per edge for demonstration)
    edge_attr = torch.tensor([graph[u][v]['offset'] for u, v in graph.edges], dtype=torch.float).unsqueeze(1)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data






data = graph_to_data(graph)

# Create the environment
env = GraphTraversalEnv(graph, target_node)

# Initialize model and optimizer
model = PolicyNetwork(env.state_size - 1, 1).to(device)
data = model.normalize_features(data).to(device)


def weights_init_he(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

model.apply(weights_init_he)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
eps = np.finfo(np.float32).eps.item()
rewards = []
num_episodes = 50000
stats = {
             "nb_success" : 0,
             }
             
total_successes = []
def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in model.rewards[::-1]:
        R = r + 0.99 * R
        returns.appendleft(R)

    #print(model.rewards)
    returns = torch.tensor(returns)

    returns = (returns - returns.mean()) / (returns.std() if returns.std() > eps else eps)
    for log_prob, R in zip(model.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]


def select_action(state, is_supervised = False, printing = False):
    state = model.normalize_features(state).to(device)


    probs = model(data, state)
    m = Categorical(probs)
    action = m.sample().item()
    if printing:
        print(probs)
    if is_supervised:
        best_action = env._get_best_action()

        action = best_action
    model.saved_log_probs.append(m.log_prob(torch.tensor(action, dtype=torch.long).to(device)))
    return action

episode = 0

success_window_size = 100
windowed_success = 0
while True:
    if keyboard.is_pressed('q'):
        break
    episode = episode + 1
    observation = env.reset()
    done = False
    episode_reward = 0
    episode_stats = {"nb_of_moves" : 0,
             }
    printing = False

    #make supervised probability dependent on the episode number, maximum 0.5, min 0
    is_supervised_prob = 0.4
    is_supervised = random.random() < is_supervised_prob
    while not done:

        # Convert your observation to a suitable PyG data format (if it isn't already)
        # For this example, I'm assuming observation is your G' in PyG Data format.
        data_prime = observation
        #data_prime = normalize_features(data_prime)
        action = select_action(data_prime, is_supervised, printing)


        # Take a step in the environment
        new_observation, reward, done, info = env.step(action, False)
        if done and info["found_target"] == True:
            #print(f"Found target with {episode_stats['nb_of_moves'] + 1} moves")
            stats["nb_success"] = stats["nb_success"] + 1 
            windowed_success = windowed_success + 1

            
        observation = new_observation
        model.rewards.append(reward)
        episode_reward += reward
        episode_stats["nb_of_moves"] = episode_stats["nb_of_moves"] + 1
        finish_episode()



    rewards.append(episode_reward)
    if episode % success_window_size == 0:
        total_successes.append(windowed_success/ success_window_size)
        windowed_success = 0

    if episode % 500 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1} \t is_supervised_prob = {is_supervised_prob}")


#plot rewards
import matplotlib.pyplot as plt
import numpy as np
#plot sliding window average
window_size = 10
success_array = np.array(total_successes)
success = np.convolve(success_array, np.ones((window_size,))/window_size, mode='valid')
plt.plot(success)
#add line for the is_supervised_prob
plt.plot([0, len(success)], [is_supervised_prob, is_supervised_prob], 'k-', lw=2)
plt.show()




#ask if the agent want to save the model
save_model = input("Do you want to save the model ? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name : ")
    torch.save(model.state_dict(), "models/" + model_name + ".pt")


