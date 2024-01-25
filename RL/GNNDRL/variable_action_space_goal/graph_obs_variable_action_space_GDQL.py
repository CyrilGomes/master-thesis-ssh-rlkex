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

from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling, global_add_pool, global_max_pool
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch.utils.tensorboard import SummaryWriter

import heapq  # For priority queue
import time
from agent_variable_action_space import Agent
from utils import preprocess_graph, convert_types, add_global_root_node, connect_components, remove_all_isolated_nodes
from networkx.drawing.nx_pydot import graphviz_layout

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


# -------------------------
# HYPERPARAMETERS
# -------------------------
BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 600         # batch size
GAMMA = 0.99            # discount factor
TAU = 0.05              # soft update of target parameters
LR = 0.01               # learning rate
UPDATE_EVERY = 50        # how often to update the network

FOLDER = "Generated_Graphs/output/"
STATE_SPACE = 7
EDGE_ATTR_SIZE = 1
agent = Agent(STATE_SPACE,EDGE_ATTR_SIZE, seed=0, device=device, lr=LR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, update_every=UPDATE_EVERY)




def check_parameters(env):

    if env.observation_space.spaces['x'].shape[0] != STATE_SPACE:
        raise ValueError("State space is not correct")
    
    




def show_graph(graph, goal, current_node, neighbours_qvalues):
    #for all target nodes, if the value of target_nodes[node] is 0 then label is 'A', if 1 then label is 'B', etc.. up to 5 (F)
    labels = {}
    for node, attributes in graph.nodes(data=True):
        if attributes['cat'] >= 0:
            labels[str(node)] = chr(ord('A') + attributes['cat'])
            #if is goal concatenate with "G"
            if node == goal:
                labels[str(node)] = labels[str(node)] + " G"

        elif node == current_node:
            labels[str(node)] = "X"
        else:
            labels[str(node)] = ""
        #if node is a neighbou concatenate the qvalue
        if node in neighbours_qvalues:
            labels[str(node)] = f"{labels[str(node)]} : {neighbours_qvalues[node].item():.2f}"
    #set colors of target nodes to red
            
    #set colors of neighbours as a heatmap of the qvalues, closer to 1 is red, closer to 0 is blue
    colors = []
    for node, attributes in graph.nodes(data=True):
        if attributes['cat'] >= 0:
            colors.append((1, 0, 0))
        elif node == current_node:
            colors.append((0, 1, 0))
        elif node in neighbours_qvalues:
            
            min_qvalue = min(neighbours_qvalues.values()).item()
            max_qvalue = max(neighbours_qvalues.values()).item()


            qvalue = neighbours_qvalues[node] 

            #convert qvalue to regular float
            qvalue = qvalue.item()
            

            #normalize between 0 and 1 in case
            qvalue = 0 if max_qvalue == min_qvalue else (qvalue - min_qvalue) / (max_qvalue - min_qvalue)
            
            colors.append((qvalue, 0, 1 - qvalue))
        else:
            colors.append((0, 0, 0))
    
    G_temp = nx.DiGraph()
    G_temp.add_nodes_from(str(n) for n in graph.nodes())    
    G_temp.add_edges_from((str(u), str(v)) for u, v in graph.edges())
    #draw the graph with labels and colors
    pos = graphviz_layout(G_temp, prog='dot')
    nx.draw(G_temp, labels=labels, node_color=colors, pos = pos)
    plt.show()

def test_for_graph(file):
    """Basically the same as the training function, but without training"""
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = define_targets(graph=graph)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True)
    check_parameters(env)

    
    total_reward = 0
    total_key_found = 0

    for target in target_nodes:
        done = False

        goal = target_nodes[target]
        observation = env.reset()
        goal_one_hot = env.get_goal_one_hot(goal)
        env.set_target_goal(goal)
        display_graph = env.graph
        while not done:
            
            action_mask = env._get_action_mask()
            visited_subgraph = env.get_visited_subgraph()
            action = agent.act(observation, action_mask, goal_one_hot, visited_subgraph)
            qvalues = agent.get_qvalues(observation, action_mask, goal_one_hot, visited_subgraph)
            node_qvalues_map = {}
            for i, qvalue in enumerate(qvalues):
                if action_mask[i] == 1:
                    node_qvalues_map[env.inverse_node_mapping[i]] = qvalue
            if SHOW_GAPH_TEST:
                show_graph(display_graph, target, env.current_node, node_qvalues_map)
            
            new_observation, reward, done, info, new_goal = env.step(action)
            total_reward += reward
            if done:
                if info["found_target"]:
                    total_key_found += 1
            
            observation = new_observation
    
    return total_reward, total_key_found, len(target_nodes)

  




INIT_EPS = 0.99
EPS_DECAY = 0.999998
MIN_EPS = 0.05


def define_targets(graph):
    target_nodes_map = {}
    for node, attributes in graph.nodes(data=True):
        if attributes['cat'] >= 0:
            target_nodes_map[node] = attributes['cat'] 
    return target_nodes_map



def supervised_training(file):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = define_targets(graph=graph)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True)

    check_parameters(env)
    windowed_success = 0

    num_episode_multiplier = len(target_nodes)
    num_episodes = 300
    range_episode = trange(num_episodes, desc="Episode", leave=True)
    max_key_found = 0

    

    
    for episode in range_episode:
        observation = env.reset()

 
        done = False
        #get a random index goal from the target map
        goal = random.choice(list(target_nodes.keys()))
        goal = target_nodes[goal]
        goal_one_hot = env.get_goal_one_hot(goal)
        env.set_target_goal(goal)
        experience = namedtuple("Experience", field_names=["observation", "action", "reward", "new_observation", "done", "next_action_mask", "visited_subgraph", "next_visited_subgraph"])
        experiences = []
        while not done:

            action_mask = env._get_action_mask()
            visited_subgraph = env.get_visited_subgraph()

            best_actions = env._get_best_action()
            if len(best_actions) == 0:
                print("No best action")
            action = random.choice(best_actions)
            new_observation, reward, done, info, new_goal = env.step(action)

            next_action_mask = env._get_action_mask()
            next_visited_subgraph = env.get_visited_subgraph()

            experiences.append(experience(observation, action, reward, new_observation, done, next_action_mask, visited_subgraph, next_visited_subgraph))

            
            if done:
                break
            
            observation = new_observation
        for exp in experiences:
            agent.step(exp.observation, exp.action, exp.reward, exp.new_observation, exp.done, exp.next_action_mask, goal_one_hot, exp.visited_subgraph, exp.next_visited_subgraph)




        
    return episode_rewards

        #if episode % 500 == 0:
        #    print(f"Episode {episode + 1}: Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1}")




def execute_for_graph(file, training = True):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = define_targets(graph=graph)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True)

    check_parameters(env)
    windowed_success = 0

    num_episode_multiplier = len(target_nodes)
    num_episodes = 1000
    stats = {"nb_success": 0}
    range_episode = trange(num_episodes, desc="Episode", leave=True)
    max_reward = -np.inf
    max_key_found = 0

    #agent.log_environment_change(file)
    #find the factor to which I have to multiply curr_eps such that at the end of the training it is 0.05
    
    keys_per_episode = []

    nb_moves_per_episode = []
    nb_keys_found = np.full(6,0)

    for episode in range_episode:
        observation = env.reset()
        episode_reward = 0
        episode_stats = {"nb_of_moves": 0}
        global EPS 
        if training:
            EPS = EPS * EPS_DECAY if EPS > MIN_EPS else MIN_EPS
        
        #a function of episode over num_epsiode, such that at the end it is 0.05, linear
        curr_eps = EPS
        curr_episode_rewards = []
        done = False

        goal = random.choice(list(target_nodes.keys()))
        goal = target_nodes[goal]
        goal_one_hot = env.get_goal_one_hot(goal)
        env.set_target_goal(goal)
        experience = namedtuple("Experience", field_names=["observation", "action", "reward", "new_observation", "done", "next_action_mask", "visited_subgraph", "next_visited_subgraph"])
        experiences = []
        while not done:

            action_mask = env._get_action_mask()
            visited_subgraph = env.get_visited_subgraph()
            action = agent.act(observation, action_mask, goal_one_hot,visited_subgraph, curr_eps)
            new_observation, reward, done, info, new_goal = env.step(action)
            if new_goal is not None:
                goal_one_hot = env.get_goal_one_hot(new_goal)
            next_action_mask = env._get_action_mask()
            next_visited_subgraph = env.get_visited_subgraph()
            curr_episode_rewards.append(reward)

            metrics = {
                'Reward': reward,
                'Epsilon': curr_eps,

            }
            agent.log_metrics(metrics)
            experiences.append(experience(observation, action, reward, new_observation, done, next_action_mask, visited_subgraph, next_visited_subgraph))


            episode_stats["nb_of_moves"] += 1

            
            if done:
                if info["found_target"]:
                    if new_goal is None:
                        nb_keys_found[goal] = nb_keys_found[goal] + 1
                        stats["nb_success"] += 1
                    #print("Success !")
                    windowed_success += 1
                break
            
            observation = new_observation
        
        for exp in experiences:
            agent.step(exp.observation, exp.action, exp.reward, exp.new_observation, exp.done, exp.next_action_mask, goal_one_hot, exp.visited_subgraph, exp.next_visited_subgraph)
        episode_reward = np.sum(curr_episode_rewards)
        nb_moves_per_episode.append(episode_stats["nb_of_moves"])

        if episode_reward > max_reward:
            max_reward = episode_reward

        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0

        #keep only nb_keys_found for the targets that are in the graph
        total_key_found = np.sum(nb_keys_found)
        nb_keys_found_ratio = nb_keys_found if total_key_found == 0 else np.divide(nb_keys_found, total_key_found)

        metrics = {

            'NbMoves' : episode_stats["nb_of_moves"],

        }
        agent.log_metrics(metrics)
        

        avg_nb_moves = np.mean(nb_moves_per_episode[-10:]) if len(nb_moves_per_episode) > 0 else 0.0

        description = f"MER : {max_reward:.2f} NbMvs : {avg_nb_moves:.2f} Avg Reward : {avg_reward:.2f} SR : {(stats['nb_success'] / (episode + 1)):.2f} eps : {curr_eps:.2f}"
        #add to the decription the number of keys found
        for i, nb_keys in enumerate(nb_keys_found_ratio):
            description += f" NbKeys{i} : {nb_keys:.2f}"

        range_episode.set_description(description)
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



SHOW_GAPH_TEST = False
EPS = INIT_EPS

#get all files in the folder recursively
all_files = []
for root, dirs, files in os.walk(FOLDER):
    for file in files:
        if file.endswith(".graphml"):
            all_files.append(os.path.join(root, file))




#shuffle the files
random.shuffle(all_files)

nb_file_overall = 180
all_files = all_files[:nb_file_overall]

nb_files = len(all_files)
nb_supervised_files = int(nb_files * 0.05)
nb_training_files = int(nb_files *0.75)
nb_testing_files = int(nb_files * 0.2)


supervised_training_files = all_files[:nb_supervised_files]
training_files = all_files[nb_supervised_files:nb_supervised_files + nb_training_files]
testing_files = all_files[nb_supervised_files + nb_training_files:]

test_every = 10

print(f"Executing supervised training ...")
for i, file in enumerate(supervised_training_files):
    if file.endswith(".graphml"):
        print(f"[{i} / {nb_supervised_files}] : Executing Training for {file}")
        supervised_training(file)
        if i % test_every == 0:
            print(f"Executing Testing for {file}")
            reward, nb_found_keys, nb_keys = test_for_graph(file)
            print(f"Found {nb_found_keys} / {nb_keys} keys with a mean reward of {reward}")


print(f"Executing Training ...")

changed_lr = False
for i, file in enumerate(training_files):
    if i > nb_training_files * 0.8 and not changed_lr:
        SHOW_GAPH_TEST = True
        changed_lr = True

    if file.endswith(".graphml"):
        print(f"[{i} / {nb_training_files}] : Executing Training for {file}")
        execute_for_graph(file, True)
        if i % test_every == 0:
            print(f"Executing Testing for {file}")
            reward, nb_found_keys, nb_keys = test_for_graph(file)
            print(f"Found {nb_found_keys} / {nb_keys} keys with a mean reward of {reward}")

print(f"Training done ")
SHOW_GAPH_TEST = False

print(f"Executing Testing ...")
test_rewards = []
test_success_rate = []
for i, file in enumerate(testing_files):
    if file.endswith(".graphml"):
        print(f"[{i} / {nb_testing_files}] : Executing Testing for {file}")
        reward, nb_found_keys, nb_keys = test_for_graph(file)
        print(f"Found {nb_found_keys} / {nb_keys} keys with a mean reward of {reward}")
        test_rewards.append(reward)
        test_success_rate.append(nb_found_keys / nb_keys)

print(f"Testing done with a mean reward of {np.mean(test_rewards)} and a success rate of {np.mean(test_success_rate)}")


# Save Model
save_model = input("Do you want to save the model? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name: ")
    #check if folder exists
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(agent.qnetwork_local.state_dict(), f"models/{model_name}.pt")
