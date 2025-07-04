# -------------------------
# IMPORTS AND SETUP
# -------------------------

import os
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_scatter import scatter_max
from root_heuristic_rf import GraphPredictor

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
import concurrent.futures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)


# -------------------------
# HYPERPARAMETERS
# -------------------------
BUFFER_SIZE = int(15e5)  # replay buffer size
BATCH_SIZE = 500         # batch size
GAMMA = 0.99            # discount factor
TAU = 100              # soft update of target parameters
LR = 0.000001               # learning rate
UPDATE_EVERY = 10        # how often to update the network

FOLDER = "Generated_Graphs/"
STATE_SPACE = 7
EDGE_ATTR_SIZE = 1
agent = Agent(STATE_SPACE,EDGE_ATTR_SIZE, seed=0, device=device, lr=LR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, update_every=UPDATE_EVERY)
agent.load_checkpoint("/root/ssh-rlkex/models/81_model.pt")
root_detection_model_path="/root/ssh-rlkex/models/root_heuristic_model.joblib"
root_detector = GraphPredictor(root_detection_model_path)

def check_parameters(env):

    if env.observation_space.spaces['x'].shape[0] != STATE_SPACE:
        raise ValueError("State space is not correct")
    

def test_for_graph(graph):
    """Basically the same as the training function, but without training"""

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = define_targets(graph=graph)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True, root_detector=root_detector)
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
            current_node_id = env.get_current_node()
            action = agent.act(observation, action_mask, goal_one_hot, visited_subgraph, current_node_id)
            qvalues = agent.get_qvalues(observation, action_mask, goal_one_hot, visited_subgraph, current_node_id)
            node_qvalues_map = {}
            for i, qvalue in enumerate(qvalues):
                if action_mask[i] == 1:
                    node_qvalues_map[env.inverse_node_mapping[i]] = qvalue
            #if SHOW_GAPH_TEST:
                #show_graph(display_graph, target, env.current_node, node_qvalues_map)
            
            new_observation, reward, done, info, new_goal = env.step(action)
            total_reward += reward
            if done:
                if info["found_target"]:
                    total_key_found += 1
            
            observation = new_observation
    
    return total_reward, total_key_found, len(target_nodes)

  


INIT_EPS = 1
EPS_DECAY = 0.999993
MIN_EPS = 0.03


def define_targets(graph):
    target_nodes_map = {}
    for node, attributes in graph.nodes(data=True):
        cat_value = attributes['cat']
        if cat_value >= 0 and cat_value <= 3: #Only take the encryption and initialization keys, ignore integrity keys
            target_nodes_map[node] = attributes['cat'] 
    return target_nodes_map



def supervised_training(graph):

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = define_targets(graph=graph)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True, root_detector=root_detector)
    nb_of_nodes = env.get_number_of_nodes()
    check_parameters(env)
    windowed_success = 0

    num_episode_multiplier = 1
    if nb_of_nodes > 100:
        num_episode_multiplier = 2
    num_episodes = 50 * num_episode_multiplier
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
        experience = namedtuple("Experience", field_names=["observation", "action", "reward", "new_observation", "done", "next_action_mask", "visited_subgraph", "next_visited_subgraph","current_node_id", "next_current_node_id"])
        experiences = []
        while not done:

            action_mask = env._get_action_mask()
            visited_subgraph = env.get_visited_subgraph()
            current_node_id = env.get_current_node()
            best_actions = env._get_best_action()
            if len(best_actions) == 0:
                print("No best action")
            action = random.choice(best_actions)
            new_observation, reward, done, info, new_goal = env.step(action)

            next_action_mask = env._get_action_mask()
            next_visited_subgraph = env.get_visited_subgraph()
            next_current_node_id = env.get_current_node()

            experiences.append(experience(observation, action, reward, new_observation, done, next_action_mask, visited_subgraph, next_visited_subgraph, current_node_id, next_current_node_id))

            
            if done:
                break
            
            observation = new_observation
        for exp in experiences:
            agent.step(exp.observation, exp.action, exp.reward, exp.new_observation, exp.done, exp.next_action_mask, goal_one_hot, exp.visited_subgraph, exp.next_visited_subgraph, exp.current_node_id, exp.next_current_node_id)




        
    return episode_rewards

        #if episode % 500 == 0:
        #    print(f"Episode {episode + 1}: Reward = {episode_reward} \t Nb_Moves = {episode_stats['nb_of_moves']} \t Nb_Success = {stats['nb_success']} / {episode + 1}")




def pick_least_visited_goal(target_nodes, nb_keys_found):
    # Extract the goal indices from target_nodes values
    goal_indices = list(target_nodes.values())
    
    # Determine the number of keys found for each goal
    relevant_nb_keys_found = np.array([nb_keys_found[i] for i in goal_indices])
    
    # Find the minimum number of keys found among the relevant goals
    min_keys_found = np.min(relevant_nb_keys_found)
    
    # Identify all goals that are tied for the minimum number of keys found
    least_visited_goals_indices = np.where(relevant_nb_keys_found == min_keys_found)[0]
    
    # If there's more than one, select one at random
    selected_index = random.choice(least_visited_goals_indices)
    
    # Find the corresponding node ID for the selected goal index
    selected_node_id = list(target_nodes.keys())[selected_index]

    return selected_node_id
def sample_goal(target_nodes, nb_keys_found):
    return pick_least_visited_goal(target_nodes, nb_keys_found)
    # Extract the goal indices from target_nodes values
    goal_indices = list(target_nodes.values())
    
    # Determine the number of keys found for each goal
    relevant_nb_keys_found = np.array([nb_keys_found[i] for i in goal_indices])
    
    # Identify visited and unvisited goals
    unvisited_goals_indices = np.where(relevant_nb_keys_found == 0)[0]
    visited_goals_indices = np.where(relevant_nb_keys_found > 0)[0]
    
    # Initialize probabilities array
    probabilities = np.zeros(len(goal_indices))
    
    # Assign probabilities for unvisited goals
    if len(unvisited_goals_indices) > 0:
        probabilities[unvisited_goals_indices] = 1 / len(unvisited_goals_indices)
    
    # Adjust probabilities for visited goals
    if len(visited_goals_indices) > 0:
        # For visited goals, calculate inverse of keys found for normalization
        visited_inverse = 1 / relevant_nb_keys_found[visited_goals_indices]
        total_visited_inverse = np.sum(visited_inverse)
        
        # Normalize and allocate remaining probability to visited goals
        probabilities[visited_goals_indices] = (visited_inverse / total_visited_inverse) * (1 - np.sum(probabilities[unvisited_goals_indices]))
    
    # Convert target_nodes keys (node IDs) to a list for sampling
    node_ids = list(target_nodes.keys())
    
    # Sample a node ID based on the calculated probabilities
    chosen_node_id = random.choices(node_ids, weights=probabilities, k=1)[0]

    return chosen_node_id
def execute_for_graph(graph, training = True):

    #get all target_nodes, check if nodes has 'cat' = 1
    target_nodes = define_targets(graph=graph)
    episode_rewards = []
    #data = graph_to_data(graph)
    env = GraphTraversalEnv(graph, target_nodes, obs_is_full_graph=True, root_detector=root_detector)

    check_parameters(env)
    windowed_success = 0

    nb_of_nodes = env.get_number_of_nodes()



    num_episode_multiplier = 1
    if nb_of_nodes > 200:
        num_episode_multiplier = 2
    elif nb_of_nodes > 400:
        num_episode_multiplier = 3
    num_episodes = 350 * num_episode_multiplier
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
        #if training:
            #EPS = EPS * EPS_DECAY if EPS > MIN_EPS else MIN_EPS
        
        #a function of episode over num_epsiode, such that at the end it is 0.05, linear
        curr_eps = (1 - (episode / num_episodes)) * INIT_EPS
        curr_episode_rewards = []
        done = False

        goal = sample_goal(target_nodes, nb_keys_found)
        goal = target_nodes[goal]
        goal_one_hot = env.get_goal_one_hot(goal)
        env.set_target_goal(goal)
        experience = namedtuple("Experience", field_names=["observation", "action", "reward", "new_observation", "done", "next_action_mask", "visited_subgraph", "next_visited_subgraph", "current_node_id", "next_current_node_id"])
        experiences = []
        while not done:

            action_mask = env._get_action_mask()
            visited_subgraph = env.get_visited_subgraph()
            current_node_id = env.get_current_node()
            action = agent.act(observation, action_mask, goal_one_hot,visited_subgraph, current_node_id, curr_eps)
            new_observation, reward, done, info, new_goal = env.step(action)
            if new_goal is not None:
                goal_one_hot = env.get_goal_one_hot(new_goal)
            next_action_mask = env._get_action_mask()
            next_visited_subgraph = env.get_visited_subgraph()
            next_current_node_id = env.get_current_node()
            curr_episode_rewards.append(reward)

            metrics = {
                'Reward': reward,
                'Epsilon': curr_eps,

            }
            agent.log_metrics(metrics)
            experiences.append(experience(observation, action, reward, new_observation, done, next_action_mask, visited_subgraph, next_visited_subgraph, current_node_id, next_current_node_id))


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
            agent.step(exp.observation, exp.action, exp.reward, exp.new_observation, exp.done, exp.next_action_mask, goal_one_hot, exp.visited_subgraph, exp.next_visited_subgraph, exp.current_node_id, exp.next_current_node_id)
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


def read_graph(file):
    graph = nx.read_graphml(file)
    graph = preprocess_graph(graph)
    return graph


SHOW_GAPH_TEST = False
EPS = INIT_EPS

TEST_FOLDER = "Test_Graphs/"

N = 20  # Set the number of files you want from each subfolder

# Get all files in the folder recursively
all_files = []
for root, dirs, files in os.walk(FOLDER):
    count = 0  # Initialize counter for each subfolder
    for file in files:
        if file.endswith(".graphml"):
            all_files.append(os.path.join(root, file))
            count += 1  # Increment counter
        if count >= N:  # Check if counter has reached N
            break  # If so, break the loop and move on to the next subfolder

#shuffle the files
random.shuffle(all_files)

nb_file_overall = int(len(all_files))
all_files = all_files[:nb_file_overall]

all_graphs = []
print(f"Reading {len(all_files)} Training graphs ...")
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Map the read_graph function to all files
    results = list(tqdm(executor.map(read_graph, all_files), total=len(all_files)))
    all_graphs.extend(results)

nb_files = len(all_graphs)
nb_supervised_files = 20
nb_training_files = nb_file_overall


#take nb_supervised_files random graphs


test_every = 100
supervised_every = 50


print(f"Executing Training ...")
start_time = time.time()

training_graphs = all_graphs#random.sample(all_graphs, nb_training_files)
changed_lr = False
for i, graph in enumerate(training_graphs):
    file = all_files[i]

    print(f"[{i} / {nb_training_files}] : Executing Training for {file}")
    execute_for_graph(graph, True)
    if i % 10 == 0:
        agent.save_checkpoint(i)
    if i % test_every == 0:
        print(f"Executing Testing for {file}")
        reward, nb_found_keys, nb_keys = test_for_graph(graph)
        print(f"Found {nb_found_keys} / {nb_keys} keys with a mean reward of {reward}")
    if i % supervised_every == 0:
        print(f"Executing supervised training ...")
        supervised_training_graphs = random.sample(all_graphs, nb_supervised_files)
        nb_supervised_files = len(supervised_training_graphs)
        for j, graph in enumerate(supervised_training_graphs):
            file = all_files[i]
            print(f"[{j} / {nb_supervised_files}] : Executing STraining for {file}")
            supervised_training(graph)


stop_training_time = time.time()

SHOW_GAPH_TEST = False

#get all test files in the test folder recursively
test_files = []
for root, dirs, files in os.walk(TEST_FOLDER):
    for file in files:
        if file.endswith(".graphml"):
            test_files.append(os.path.join(root, file))

nb_testing_files = len(test_files)

testing_graphs = []
print(f"Reading {nb_testing_files} Testing graphs ...")
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Map the read_graph function to all files
    results = list(tqdm(executor.map(read_graph, test_files), total=len(test_files)))
    testing_graphs.extend(results)

print(f"Executing Testing ...")
test_rewards = []
test_success_rate = []
for i, graph in enumerate(testing_graphs):
    file = test_files[i]

    print(f"[{i} / {nb_testing_files}] : Executing Testing for {file}")
    reward, nb_found_keys, nb_keys = test_for_graph(graph)
    print(f"Found {nb_found_keys} / {nb_keys} keys with a mean reward of {reward}")
    test_rewards.append(reward)
    test_success_rate.append(nb_found_keys / nb_keys)

print(f"Testing done with a mean reward of {np.mean(test_rewards)} and a success rate of {np.mean(test_success_rate)}")
print(f"Training done in {stop_training_time - start_time} seconds")


# Save Model
save_model = input("Do you want to save the model? (y/n)")
if save_model == "y":
    model_name = input("Enter the model name: ")
    #check if folder exists
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(agent.qnetwork_local.state_dict(), f"models/{model_name}.pt")
