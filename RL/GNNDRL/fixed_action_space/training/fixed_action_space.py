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

from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling, global_add_pool, global_max_pool
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch.utils.tensorboard import SummaryWriter

import heapq  # For priority queue
import time
from fixed_action_space_agent import Agent
from utils import preprocess_graph, convert_types, add_global_root_node, connect_components, remove_all_isolated_nodes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


# -------------------------
# HYPERPARAMETERS
# -------------------------
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # batch size
GAMMA = 0.99            # discount factor
TAU = 0.005              # soft update of target parameters
LR = 0.1               # learning rate
UPDATE_EVERY = 5        # how often to update the network

FOLDER = "Generated_Graphs/output/"
STATE_SPACE = 14
EDGE_ATTR_SIZE = 1
ACTION_SPACE = 50
agent = Agent(STATE_SPACE,EDGE_ATTR_SIZE, ACTION_SPACE,  seed=0, device=device, lr=LR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, update_every=UPDATE_EVERY)




def check_parameters(env):

    if env.observation_space.spaces['x'].shape[0] != STATE_SPACE:
        raise ValueError("State space is not correct")
    
    
def test_for_graph(file):
    """Basically the same as the training function, but without training"""
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = [node for node, attributes in graph.nodes(data=True) if attributes['cat'] == 1]
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes,ACTION_SPACE ,obs_is_full_graph=True)
    check_parameters(env)

    
    observation = env.reset()
    done = False
    total_reward = 0
    total_key_found = 0

    while not done:
        action_mask = env._get_action_mask()
        action = agent.act(observation, 0, action_mask)
        new_observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            #print all the info
            for key, value in info.items():
                print(f"{key} : {value}")

            total_key_found = info["nb_keys_found"]
            break
        
        observation = new_observation
    
    return total_reward, total_key_found, len(target_nodes)

  
def supervised_training(file):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = [node for node, attributes in graph.nodes(data=True) if attributes['cat'] == 1]
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes,ACTION_SPACE, obs_is_full_graph=True)

    check_parameters(env)
    windowed_success = 0

    num_episode_multiplier = len(target_nodes)
    num_episodes = 2000
    stats = {"nb_success": 0}
    range_episode = trange(num_episodes, desc="Episode", leave=True)
    max_reward = -np.inf
    max_key_found = 0

    #agent.log_environment_change(file)
    #find the factor to which I have to multiply curr_eps such that at the end of the training it is 0.05
    
    keys_per_episode = []

    nb_moves_per_episode = []
    
    good_action_prob = 0.7

    for episode in range_episode:
        observation = env.reset()
        episode_reward = 0
        episode_stats = {"nb_of_moves": 0,
                         "nb_key_found": 0,
                         'nb_possible_keys' : 0}


        curr_episode_rewards = []
        done = False

        while not done:
            action_mask = env._get_action_mask()
            weight_array = None# env._get_probability_distribution(action_mask)

            random_action_result =random.random() < good_action_prob
            if random_action_result:
                action = env.get_good_action()  
            else:
                action = None
            
            if action is None:
                action = agent.act(observation, 1, action_mask, weight_array)

            new_observation, reward, done, info = env.step(action)
            next_action_mask = env._get_action_mask()
            curr_episode_rewards.append(reward)

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
        nb_moves_per_episode.append(episode_stats["nb_of_moves"])

        if episode_reward > max_reward:
            max_reward = episode_reward


        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0

        key_found_ratio = episode_stats["nb_key_found"] / len(target_nodes)

        metrics = {
            'Average Reward': episode_reward,
            'Loss': agent.losses[-1] if len(agent.losses) > 0 else 0.0,
            'MaxKeyFound' : max_key_found,
            'KeyFoundRatio' : key_found_ratio,
            'NbMoves' : episode_stats["nb_of_moves"],
            'TD_Error' : agent.td_errors[-1] if len(agent.td_errors) > 0 else 0.0,
            'IS_Weights' : agent.is_weights[-1] if len(agent.is_weights) > 0 else 0.0,

        }
        agent.log_metrics(metrics)


        avg_key_found = np.mean(keys_per_episode[-10:]) if len(keys_per_episode) > 0 else 0.0
        avg_nb_moves = np.mean(nb_moves_per_episode[-10:]) if len(nb_moves_per_episode) > 0 else 0.0
        range_episode.set_description(f"MER : {max_reward:.2f} MaxKeyFnd : {max_key_found:.2f} NbMvs : {avg_nb_moves:.2f} AvgKeyFound : {avg_key_found:.2f} Avg Reward : {avg_reward:.2f} SR : {(stats['nb_success'] / (episode + 1)):.2f}")
        range_episode.refresh() # to show immediately the update
        episode_rewards.append(episode_reward)

        
    return episode_rewards, stats["nb_success"] / num_episodes



INIT_EPS = 0.98
EPS_DECAY = 0.9999
MIN_EPS = 0.01

def execute_for_graph(file, random_walk_training = True):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = [node for node, attributes in graph.nodes(data=True) if attributes['cat'] == 1]
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes,ACTION_SPACE, obs_is_full_graph=True)

    check_parameters(env)
    windowed_success = 0

    num_episode_multiplier = len(target_nodes)
    num_episodes = 2000
    stats = {"nb_success": 0}
    range_episode = trange(num_episodes, desc="Episode", leave=True)
    max_reward = -np.inf
    max_key_found = 0

    #agent.log_environment_change(file)
    #find the factor to which I have to multiply curr_eps such that at the end of the training it is 0.05
    
    keys_per_episode = []

    nb_moves_per_episode = []
    
    for episode in range_episode:
        observation = env.reset()
        episode_reward = 0
        episode_stats = {"nb_of_moves": 0,
                         "nb_key_found": 0,
                         'nb_possible_keys' : 0}
        if not agent.is_ready or random_walk_training:
            curr_eps = 1
        else:
            global EPS 
  
            EPS = EPS * EPS_DECAY if EPS > MIN_EPS else MIN_EPS
            
            #a function of episode over num_epsiode, such that at the end it is 0.05, linear
            curr_eps = EPS

        curr_episode_rewards = []
        done = False

        while not done:
            action_mask = env._get_action_mask()
            weight_array = None# env._get_probability_distribution(action_mask)

            action = agent.act(observation, curr_eps, action_mask, weight_array)
            new_observation, reward, done, info = env.step(action)
            next_action_mask = env._get_action_mask()
            curr_episode_rewards.append(reward)

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
        nb_moves_per_episode.append(episode_stats["nb_of_moves"])

        if episode_reward > max_reward:
            max_reward = episode_reward


        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0

        key_found_ratio = episode_stats["nb_key_found"] / len(target_nodes)

        metrics = {
            'Average Reward': episode_reward,
            'Loss': agent.losses[-1] if len(agent.losses) > 0 else 0.0,
            'Epsilon': curr_eps,
            'MaxKeyFound' : max_key_found,
            'KeyFoundRatio' : key_found_ratio,
            'NbMoves' : episode_stats["nb_of_moves"],

        }
        agent.log_metrics(metrics)


        avg_key_found = np.mean(keys_per_episode[-10:]) if len(keys_per_episode) > 0 else 0.0
        avg_nb_moves = np.mean(nb_moves_per_episode[-10:]) if len(nb_moves_per_episode) > 0 else 0.0
        range_episode.set_description(f"MER : {max_reward:.2f} MaxKeyFnd : {max_key_found:.2f} NbMvs : {avg_nb_moves:.2f} AvgKeyFound : {avg_key_found:.2f} Avg Reward : {avg_reward:.2f} SR : {(stats['nb_success'] / (episode + 1)):.2f} eps : {curr_eps:.2f}")
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




nb_random_walk_files = 0.3
nb_supervised_training_files = 0.5
nb_testing = 0.2
EPS = INIT_EPS

#get all files in the folder recursively
all_files = []
for root, dirs, files in os.walk(FOLDER):
    for file in files:
        if file.endswith(".graphml"):
            all_files.append(os.path.join(root, file))


total_number_of_training_files = len(all_files)*0.009

#shuffle allfiles
random.shuffle(all_files)
all_files = all_files[:int(total_number_of_training_files)]
print(f"Total number of training files : {len(all_files)}")
#split into supervised_training, normal_training and testing
all_files_size = len(all_files)
supervised_training_files = all_files[:int(all_files_size * nb_supervised_training_files)]
random_walk_files = all_files[int(all_files_size * nb_supervised_training_files):int(all_files_size * (nb_supervised_training_files + nb_random_walk_files))]
testing_files = all_files[int(all_files_size * (nb_supervised_training_files + nb_random_walk_files)):]
print(f"Supervised Training : {len(supervised_training_files)} files")
print(f"Random Walk Training : {len(random_walk_files)} files")
print(f"Testing : {len(testing_files)} files")




for i, file in enumerate(supervised_training_files):
    if file.endswith(".graphml"):
        print(f"[{i} / {nb_supervised_training_files*all_files_size}] : Executing Supervised Training for {file}")
        supervised_training(file)

print(f"Supervised Training done ")

for i, file in enumerate(random_walk_files):
    if file.endswith(".graphml"):
        print(f"[{i} / {nb_random_walk_files*all_files_size}] : Executing Random Walk Training for {file}")
        execute_for_graph(file, False)
        print(f"Executing Testing for {file}")
        reward, nb_found_keys, nb_keys = test_for_graph(file)
        print(f"Found {nb_found_keys} / {nb_keys} keys with a total reward of {reward}")

print(f"Random Walk Training done ")

for i, file in enumerate(testing_files):
    if file.endswith(".graphml"):
        print(f"[{i} / {nb_testing*all_files_size}] : Executing Testing for {file}")
        reward, nb_found_keys, nb_keys = test_for_graph(file)
        print(f"Found {nb_found_keys} / {nb_keys} keys with a total reward of {reward}")

print(f"Testing done ")



# Save Model
save_model = input("Do you want to save the model? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name: ")
    #check if folder exists
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(agent.qnetwork_local.state_dict(), f"models/{model_name}.pt")
