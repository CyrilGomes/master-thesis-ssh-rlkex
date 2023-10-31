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

from rl_base.rl_environment import GraphTraversalEnv
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

class GNN(torch.nn.Module):
    def __init__(self, dimension=7):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dimension, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Use global_mean_pool to compute the mean for each graph in the batch
        x = torch.sigmoid(global_mean_pool(x, batch))
        return x



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
    def __init__(self, node_feature_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)


        self.fc1 = torch.nn.Linear(node_feature_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)

        # Dueling DQN
        self.value_stream = torch.nn.Linear(128, 128)
        self.value = torch.nn.Linear(128, 1)
        
        self.advantage_stream = torch.nn.Linear(128, 128)
        self.advantage = torch.nn.Linear(128, action_size)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        value = F.relu(self.value_stream(x))
        value = self.value(value)
        
        advantage = F.relu(self.advantage_stream(x))
        advantage = self.advantage(advantage)
        
        # Combine value and advantage streams
        qvals = value + (advantage - advantage.mean())
        return qvals



# -------------------------
# AGENT DEFINITION
# -------------------------
class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        
        state = state.float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # DDQN
        indices = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, indices)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample(self):
        experiences = random.sample(self.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)




# -------------------------
# MAIN EXECUTION
# -------------------------

# Load and preprocess graph

FOLDER = "Generated_Graphs/64/"
agent = Agent(10, 50, seed=0)


def execute_for_graph(file, training = True, level = 0):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)
    target_node = next((node for node, attributes in graph.nodes(data=True) if attributes["cat"] == 1), None)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_node, level)
    windowed_success = 0
    num_episodes = 3000
    stats = {"nb_success": 0}
    range_episode = trange(num_episodes, desc="Episode", leave=True)
    max_reward = -500
    for episode in range_episode:

        observation = env.reset()
        episode_reward = 0
        episode_stats = {"nb_of_moves": 0}
        while True:
            first_node = observation.x[0]
            eps = 0.2 if training else 0.0
            action = agent.act(first_node, eps)
            new_observation, reward, done, info = env.step(action, False)
            episode_reward += reward
            if training:
                agent.step(first_node, action, reward, new_observation.x[0], done)

            episode_stats["nb_of_moves"] += 1
            if done:
                if info["found_target"]:
                    stats["nb_success"] += 1
                    windowed_success += 1
                break
            observation = new_observation

        if episode_reward > max_reward:
            max_reward = episode_reward

        avg_reward = np.mean(episode_rewards[-100:])
        range_episode.set_description(f"Max Episode Reward : {max_reward:.2f} Avg Reward : {avg_reward:.2f} Success Rate : {(stats['nb_success'] / (episode + 1)):.2f}")
        range_episode.refresh() # to show immediately the update
        episode_rewards.append(episode_reward)
    
    return episode_rewards, stats["nb_success"] / num_episodes

        #if episode % 500 == 0:
        #    print(f"Episode {episode + 1}: Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1}")




def visualize(rewards):
    # Visualization
    window_size = 30
    success_array = np.array(rewards)
    success = np.convolve(success_array, np.ones((window_size,))/window_size, mode='valid')
    plt.plot(success)
    plt.show()


#take random files from folder and execute
nb_random_files = 6

max_levels = 4
for level in range( max_levels):
    random_files = random.sample(os.listdir(FOLDER), nb_random_files)
    i = 0
    for file in random_files:
        if file.endswith(".graphml"):
            i+=1
            print(f"[{i} / {nb_random_files}] : Executing Training for {file}")
            execute_for_graph(FOLDER + file, True, level)
    
    #take random file from folder and execute
    random_file = random.choice(os.listdir(FOLDER))
    print(f"Executing Testing for {random_file}")
    rewards, succes_rate = execute_for_graph(FOLDER + random_file, False, level)
    print(f"Success rate : {succes_rate}")

#take random file from folder and execute
random_file = random.choice(os.listdir(FOLDER))
print(f"Executing Testing for {random_file}")
rewards, succes_rate = execute_for_graph(FOLDER + random_file, False, 0)
print(f"Success rate : {succes_rate}")
visualize(rewards)

        

# Save Model
save_model = input("Do you want to save the model? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name: ")
    #check if folder exists
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(agent.qnetwork_local.state_dict(), f"models/{model_name}.pt")
