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

from rl_env_graph_obs_variable_action_space import GraphTraversalEnv
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

from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch.utils.tensorboard import SummaryWriter

import heapq  # For priority queue
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

# -------------------------
# GRAPH PROCESSING
# -------------------------

class MyGraphData(Data):

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'action':
            return None
        if key == 'reward':
            return None
        if key == 'done':
            return None
        if key == 'exp_index':
            return None

        
        return super().__cat_dim__(key, value, *args, **kwargs)


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

def convert_types(G):
    # Convert the string attributes to their corresponding types
    for node, data in G.nodes(data=True):
        # The label remains a string, so no conversion is needed for 'label'
        # Convert struct_size, valid_pointer_count, invalid_pointer_count,
        # first_pointer_offset, last_pointer_offset, first_valid_pointer_offset,
        # last_valid_pointer_offset, and address to int
        data['struct_size'] = int(data['struct_size'])
        data['valid_pointer_count'] = int(data['valid_pointer_count'])
        data['invalid_pointer_count'] = int(data['invalid_pointer_count'])
        data['first_pointer_offset'] = int(data['first_pointer_offset'])
        data['last_pointer_offset'] = int(data['last_pointer_offset'])
        data['first_valid_pointer_offset'] = int(data['first_valid_pointer_offset'])
        data['last_valid_pointer_offset'] = int(data['last_valid_pointer_offset'])
        data['address'] = int(data['address'])

        # Convert cat to an integer and ensure it's within the range of a byte (0-255)
        data['cat'] = int(data['cat'])
        if not (0 <= data['cat'] <= 255):
            raise ValueError(f"Value of 'cat' out of range for u8: {data['cat']}")

    #Same for edges attributes (offset)
    for u, v, data in G.edges(data=True):
        data['offset'] = int(data['offset'])


    return G

def remove_all_isolated_nodes(graph):
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph
def preprocess_graph(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    
    # Removing string attributes from nodes and edges
    graph = remove_all_isolated_nodes(graph)
    graph = convert_types(graph)
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

    #graph = add_global_root_node(graph)
    #Check if all edges possess an offset attribute

    return graph

def load_graphs_from_directory(directory_path):
    graph_files = [f for f in os.listdir(directory_path) if f.endswith('.graphml')]
    graphs = [nx.read_graphml(os.path.join(directory_path, f)) for f in graph_files]
    return [preprocess_graph(g) for g in graphs]


# -------------------------
# HYPERPARAMETERS
# -------------------------
BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 256         # batch size
GAMMA = 0.99            # discount factor
TAU = 5e-1              # soft update of target parameters
LR = 0.001               # learning rate
UPDATE_EVERY = 30        # how often to update the network



# -------------------------
# MODEL DEFINITION
# -------------------------

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, dense_diff_pool, DenseSAGEConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
class GraphQNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, seed):
        super(GraphQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm0 = GraphNorm(num_node_features)

        # Define GAT layers
        self.conv1 = GATv2Conv(num_node_features, 8, edge_dim=num_edge_features)
        self.norm1 = GraphNorm(8)
        self.conv2 = GATv2Conv(8, 6, edge_dim=num_edge_features)
        self.norm2 = GraphNorm(6)
        self.conv3 = GATv2Conv(6, 4, edge_dim=num_edge_features)
        self.norm3 = GraphNorm(4)


        # DiffPool layer
        self.diffpool = DenseSAGEConv(4, 6)

        # Define dropout
        self.dropout = torch.nn.Dropout(p=0.5)

        # Dueling DQN layers
        self.value_stream = torch.nn.Linear(4, 8)
        self.value = torch.nn.Linear(8, 1)
        self.advantage_stream = torch.nn.Linear(8+6+4 ,8)
        self.advantage = torch.nn.Linear(8, 1)

    def forward(self, x, edge_index, edge_attr, batch, action_mask=None):    
        #ensure everything is on device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        if batch is not None:
            batch = batch.to(device)


        # Process with GAT layers
        x1 = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        
        x2 = F.relu(self.norm2(self.conv2(x1, edge_index, edge_attr)))

        x3 = F.relu(self.norm3(self.conv3(x2, edge_index, edge_attr)))

        x_conc = torch.cat([x1, x2, x3], dim=-1)
        # Compute node-level advantage
        advantage = F.relu(self.advantage_stream(x_conc))
        advantage = self.advantage(advantage).squeeze(-1)  # Remove last dimension

        # Prepare for DiffPool
        x, mask = to_dense_batch(x3, batch)
        adj = to_dense_adj(edge_index, batch)

        # Apply DiffPool
        s = self.diffpool(x, adj)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask=mask)

        # Global pooling for Dueling DQN
        pooled_val = torch.mean(x, dim=1)

        #combined_features = torch.cat([pooled_val, x_conc], dim=-1)
        value = F.relu(self.value_stream(pooled_val))
        value = self.value(value).squeeze(-1)  # Remove last dimension
            
        # Calculate the mean advantage for each graph in the batch
        mean_advantage = global_mean_pool(advantage, batch)

        # Expand mean advantage to match the number of nodes
        expanded_mean_advantage = mean_advantage[batch]
        value = value[batch]
        qvals = value + (advantage - expanded_mean_advantage)

        # Apply action mask if provided
        if action_mask is not None:
            action_mask = action_mask.to(qvals.device)
            # Set Q-values of valid actions (where action_mask is 0) as is, and others to -inf
            qvals = torch.where(action_mask == 0, qvals, torch.tensor(float('-1e8')).to(qvals.device))


        #check if qvals contains nan
        if torch.isnan(qvals).any():
            raise ValueError("Qvals contains nan")

        return qvals






# -------------------------
# AGENT DEFINITION
# -------------------------
class Agent:
    def __init__(self, state_size, edge_attr_size, seed, use_prioritized_replay=True):
        self.writer = SummaryWriter('runs/DQL_GRAPH_VARIABLE_ACTION_SPACE')  # Choose an appropriate experiment name
        self.use_prioritized_replay = use_prioritized_replay  # Flag to use prioritized replay
        self.state_size = state_size
        self.seed = random.seed(seed)
        self.edge_attr_size = edge_attr_size

        # Q-Network
        self.qnetwork_local = GraphQNetwork(state_size, self.edge_attr_size, seed).to(device)
        self.qnetwork_target = GraphQNetwork(state_size, self.edge_attr_size,seed).to(device)


        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory_counter = 0  # Counter to track the number of experiences added
        # Replay memory
        self.memory = []
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "action_mask", "next_action_mask", "priority"])
        self.t_step = 0

        self.losses = []
        self.steps = 0

        self.max_priority = 1.0  # Initial max priority for new experiences



    def add_experience(self, state, action, reward, next_state, done, action_mask, next_action_mask, priority):        
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, action_mask, next_action_mask, priority)
        heapq.heappush(self.memory, (-priority, self.memory_counter, e))  # Use priority and counter
        self.memory_counter += 1  # Increment counter

        if len(self.memory) > BUFFER_SIZE:
            heapq.heappop(self.memory)



    def log_environment_change(self, env_name):
        self.writer.add_text('Environment Change', f'Changed to {env_name}', self.steps)


    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.steps)



    def step(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None):
        self.steps += 1
        #ensure everything is on device
        state = state
        next_state = next_state
        

        # Save experience in replay memory
        self.add_experience(state, action, reward, next_state, done, action_mask, next_action_mask, self.max_priority*0.8)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences, indices = self.sample()
                self.learn(experiences, indices, GAMMA)

    def act(self, state, eps=0, action_mask=None):
        state = state
        action_mask = torch.from_numpy(action_mask)

        if random.random() > eps:
            self.qnetwork_local.eval()
            x = state.x
            edge_index = state.edge_index
            edge_attr = state.edge_attr
            batch = state.batch

            with torch.no_grad():  # Wrap in no_grad
                action_values = self.qnetwork_local(x, edge_index, edge_attr, batch, action_mask)
            return_values = action_values.cpu()
            self.qnetwork_local.train()

            selected_action = torch.argmax(return_values).item()
            torch.cuda.empty_cache()

            return selected_action
        else:
            return random.choice((action_mask.cpu() == 0).nonzero(as_tuple=True)[0]).item()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,  # Optional: save replay memory
            'steps': self.steps  # Optional: save training steps
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint.get('memory', self.memory)  # Load memory if saved
        self.steps = checkpoint.get('steps', self.steps)  # Load steps if saved

    def learn(self, experiences, indices, gamma):
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = experiences

        # Create a list of Data objects, each representing a graph experience
        data_list = []
        for i in range(len(states)):
            state = states[i]  # Assuming states are Data objects
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            exp_index = indices[i]

            data = MyGraphData(x_s=state.x, edge_index_s=state.edge_index, edge_attr_s=state.edge_attr,
                            action=action, reward=reward, x_t=next_state.x,
                            edge_index_t=next_state.edge_index, edge_attr_t=next_state.edge_attr,
                            done=done, exp_index=exp_index)
            data_list.append(data)

        # Create a DataLoader for batching
        batch_size = 32
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['x_s', 'x_t'], pin_memory=True)

        for batch in data_loader:
            batch.to(device)  # Send the batch to the GPU if available
            b_exp_indices = batch.exp_index
            # Extract batched action, reward, done, and action_mask
            b_action = batch.action.to(device)
            b_reward = batch.reward.to(device)
            b_done = batch.done.to(device)
            b_action_mask = None
            b_next_action_mask = None

            b_x_s = batch.x_s.to(device)
            b_edge_index_s = batch.edge_index_s.to(device)
            b_edge_attr_s = batch.edge_attr_s.to(device)
            b_x_t = batch.x_t.to(device)
            b_edge_index_t = batch.edge_index_t.to(device)
            b_edge_attr_t = batch.edge_attr_t.to(device)
            x_s_batch = batch.x_s_batch.to(device)
            x_t_batch = batch.x_t_batch.to(device)

            # DDQN Update for the individual graph
            self.qnetwork_target.eval()
            # Calculate Q values for next states
            with torch.no_grad():
                Q_targets_next = self.qnetwork_target(b_x_t, b_edge_index_t, 
                                                    b_edge_attr_t, x_t_batch, 
                                                    action_mask=b_next_action_mask).detach().squeeze()



            Q_targets_next_max = scatter_max(Q_targets_next, x_t_batch, dim=0)[0]
            Q_targets = b_reward + (gamma * Q_targets_next_max * (1 - b_done))

            self.qnetwork_local.train()
            Q_expected_result = self.qnetwork_local(b_x_s, b_edge_index_s, b_edge_attr_s, batch.batch, action_mask=b_action_mask).squeeze()


            zero_tensor = torch.tensor([0], device=x_s_batch.device)

            # Ensure b_action is 1D with shape [num_graphs]
            b_action = b_action.squeeze()

            # Count the number of nodes in each graph
            nodes_per_graph = x_s_batch.bincount()

            # Calculate the start index of each graph in the concatenated batch
            cumulative_sizes = torch.cat([torch.tensor([0], device=x_s_batch.device), nodes_per_graph.cumsum(0)[:-1]])

            # Offset b_action based on the starting index of each graph
            adjusted_b_action = b_action + cumulative_sizes

            # Ensure adjusted_b_action is 1D
            adjusted_b_action = adjusted_b_action.squeeze()



            # Now gather the Q values
            Q_expected = Q_expected_result.gather(0, adjusted_b_action)


            
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            # Soft update target network
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

            # Update priorities, losses for logging, etc.
            td_errors = abs(Q_expected - Q_targets).detach().cpu()
            for i, td_error in enumerate(td_errors):

                memory_idx = b_exp_indices[i]  # Get the original index in the memory
                self.update_priority(memory_idx, td_error.item())

            self.losses.append(loss.item())

    def update_priority(self, idx, new_priority):
        if self.use_prioritized_replay:
            # Update the priority of the experience at idx in memory
            _, counter, experience = self.memory[idx]
            self.memory[idx] = (-new_priority, counter, experience)
            # Optionally update the max priority
            self.max_priority = max(self.max_priority, new_priority)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample(self):
        """Sample a batch of experiences from memory."""
        if self.use_prioritized_replay:
            # Prioritized sampling logic
            experiences = [exp for _, _, exp in self.memory]
            priorities = np.array([exp.priority for exp in experiences])
            sum_priorities = priorities.sum()
            if sum_priorities <= 1e-5:
                probabilities = np.full(len(experiences), 1.0 / len(experiences))
            else:
                probabilities = priorities / sum_priorities

            sampled_indices = np.random.choice(len(experiences), size=BATCH_SIZE, p=probabilities)
        else:
            # Uniform sampling logic
            sampled_indices = np.random.choice(len(self.memory), size=min(BATCH_SIZE, len(self.memory)), replace=False)

        sampled_experiences = [self.memory[i][2] for i in sampled_indices]  # Adjust indexing if necessary


        states = [e.state for e in sampled_experiences if e is not None]
        actions = torch.tensor([e.action for e in sampled_experiences if e is not None], dtype=torch.long).unsqueeze(-1)
        rewards = torch.tensor([e.reward for e in sampled_experiences if e is not None])
        
        # You should handle next_states in the same manner as states, given that it also contains graph data
        next_states = [e.next_state for e in sampled_experiences if e is not None]
        
        dones = torch.tensor([torch.tensor(e.done, dtype=torch.uint8) for e in sampled_experiences if e is not None]).float()
        # Convert the list of numpy arrays to a single numpy array before converting to a tensor
        action_masks = [e.action_mask for e in sampled_experiences if e is not None]
        next_action_masks = [e.next_action_mask for e in sampled_experiences if e is not None]

        return (states, actions, rewards, next_states, dones, action_masks, next_action_masks), sampled_indices





# -------------------------
# MAIN EXECUTION
# -------------------------

# Load and preprocess graph

FOLDER = "Generated_Graphs/output/"
STATE_SPACE = 15
EDGE_ATTR_SIZE = 1

agent = Agent(STATE_SPACE,EDGE_ATTR_SIZE, seed=0)


INIT_EPS = 0.98
EPS_DECAY = 0.99991


def check_parameters(env):

    if env.observation_space.spaces['x'].shape[0] != STATE_SPACE:
        raise ValueError("State space is not correct")
    
    

def execute_for_graph(file, training = True):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = [node for node, attributes in graph.nodes(data=True) if attributes['cat'] == 1]
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True)

    check_parameters(env)
    windowed_success = 0

    num_episode_multiplier = len(target_nodes)
    num_episodes = 200 * num_episode_multiplier if training else 2
    stats = {"nb_success": 0}
    range_episode = trange(num_episodes, desc="Episode", leave=True)
    max_reward = -np.inf
    max_key_found = 0

    #agent.log_environment_change(file)
    #find the factor to which I have to multiply curr_eps such that at the end of the training it is 0.05
    
    curr_eps = 0.99

    keys_per_episode = []
    
    for episode in range_episode:
        observation = env.reset()
        episode_reward = 0
        episode_stats = {"nb_of_moves": 0,
                         "nb_key_found": 0,
                         'nb_possible_keys' : 0}
        global EPS 
        if training:
            EPS = EPS * EPS_DECAY
        
        #a function of episode over num_epsiode, such that at the end it is 0.05, linear
        curr_eps =    (0.99) * (1 - episode / num_episodes) if training else 0
        curr_episode_rewards = []
        done = False

        while not done:
            action_mask = env._get_action_mask()
            action = agent.act(observation, curr_eps, action_mask)
            new_observation, reward, done, info = env.step(action)
            next_action_mask = env._get_action_mask()
            curr_episode_rewards.append(reward)
            if training:

                agent.step(observation, action, reward, new_observation, done, action_mask, next_action_mask)

            episode_stats["nb_of_moves"] += 1
            
            if done:
                episode_stats["nb_key_found"] = info["nb_keys_found"]
                max_key_found = max(max_key_found, info["nb_keys_found"])
                if info["found_target"]:
                    stats["nb_success"] += 1
                    #print("Success !")
                    windowed_success += 1
                break
            
            observation = new_observation
        keys_per_episode.append(episode_stats["nb_key_found"])
        episode_reward = np.sum(curr_episode_rewards)
        """
        if episode == num_episodes - 1:
            plt.plot(curr_episode_rewards)
            plt.show()
        """
        """
        if episode_stats["nb_key_found"] > max_key_found:
            max_key_found = episode_stats["nb_key_found"]
            max_posssible_key = episode_stats["nb_possible_keys"]
        """
        if episode_reward > max_reward:
            max_reward = episode_reward


        """
        if episode % 500 == 499:
            #plot the losses of the agent
            moving_average = np.convolve(agent.losses, np.ones((100,))/100, mode='valid')
            plt.plot(moving_average)
            plt.show()
        """
        # Update the plot after each episode

        """
        ax.clear()
        ax.plot(agent.gradient_norms)
        ax.set_title("Gradient Norms During Training")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Average Gradient Norm")
        plt.pause(0.001)  # Pause briefly to update the plot
        """
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0

        key_found_ratio = episode_stats["nb_key_found"] / len(target_nodes)

        metrics = {
            'Average Reward': episode_reward,
            'Loss': agent.losses[-1] if len(agent.losses) > 0 else 0.0,
            'Epsilon': curr_eps,
            'MaxKeyFound' : max_key_found,
            'KeyFoundRatio' : key_found_ratio
        }
        agent.log_metrics(metrics)


        avg_key_found = np.mean(keys_per_episode[-10:]) if len(keys_per_episode) > 0 else 0.0
        range_episode.set_description(f"MER : {max_reward:.2f} MaxKeysFound : {max_key_found} AvgKEyFound : {avg_key_found:.2f} Avg Reward : {avg_reward:.2f} SR : {(stats['nb_success'] / (episode + 1)):.2f} eps : {curr_eps:.2f}")
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
nb_random_files = 20

nb_try = 10

EPS = INIT_EPS

#get all files in the folder recursively
all_files = []
for root, dirs, files in os.walk(FOLDER):
    for file in files:
        if file.endswith(".graphml"):
            all_files.append(os.path.join(root, file))


for curr_try in range( nb_try):
    random_files = random.sample(all_files, nb_random_files)
    i = 0
    print(f"Try {curr_try + 1} / {nb_try}")
    for file in random_files:
        if file.endswith(".graphml"):
            i+=1
            print(f"[{i} / {nb_random_files}] : Executing Training for {file}")
            execute_for_graph(file, True)
    random_test_file = random.choice(all_files)
    print(f"Training done ")
    print(f"Executing Testing for {random_test_file}")
    rewards, succes_rate = execute_for_graph(random_test_file, False)
    print(f"Success rate : {succes_rate}")

    

#take random file from folder and execute
random_file = random.choice(all_files)
print(f"Executing Testing for {random_file}")
rewards, succes_rate = execute_for_graph(random_file, False)
print(f"Success rate : {succes_rate}")
#visualize(rewards)


# Save Model
save_model = input("Do you want to save the model? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name: ")
    #check if folder exists
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(agent.qnetwork_local.state_dict(), f"models/{model_name}.pt")
