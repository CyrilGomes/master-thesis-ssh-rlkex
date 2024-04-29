"""
A code that tries to find the optimal branch from the root node using DQN
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networkx as nx
import os
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_space=2, action_space=4):
        super(ActorCritic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc = nn.Sequential(
            nn.Linear(state_space, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.actor = nn.Linear(16, action_space)
        self.critic = nn.Linear(16, 1)

    def forward(self, state):
        x = self.fc(state)
        return F.softmax(self.actor(x)), self.critic(x)

class ActorCriticAgent:
    def __init__(self, state_space=2, action_space=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.model = ActorCritic(state_space, action_space).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = device

    def act(self, state):
        probs, _ = self.model(state.to(self.device))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action.cpu().item()
    

    def select_action(self, state, invalid_actions_mask=None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs, state_value = self.model(state)
        
        if invalid_actions_mask is not None:
            invalid_actions_mask = torch.tensor(invalid_actions_mask, dtype=torch.bool)  # Convert to a Boolean tensor
            probs[invalid_actions_mask] = 0.  # Set the invalid actions' probability to 0
        
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item(), state_value


    def compute_returns(self, rewards, next_value, dones):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def learn(self, states, actions, log_probs, returns, values):
        returns = torch.tensor(returns).to(self.device)
        values = torch.stack(values)
        advantage = returns - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)



def get_distance_between_nodes(G, node1, node2):
    #check if a path exists between the two nodes
    
    if nx.has_path(G, node1, node2):
        return nx.shortest_path_length(G, node1, node2)
    else:
        print('No path exists between the two nodes')
        return 99999999


def is_destination_in_successors(G, node):
    #check recursively if the destination node is in the successors of the current node
    if G.nodes[node]['cat'] == 1:
        return True
    else:
        for successor in G.successors(node):
            if is_destination_in_successors(G, successor):
                print("Yay!")
                return True
        return False




def main():

    #Create a folder to store weights, the folder name is the current time
    folder_name = str(int(time.time()))
    os.mkdir(folder_name)


    agent = ActorCriticAgent(state_space=12)
    #agent.load("dqn_10508-1650982250-heap.graphml_weights.pth")

    batch_size = 32
    n_episodes = 500

    base_folder = "Generated Graphs"
    sub_folders = ["16", "24", "32"]
    max_depth = 0
    tresh_mul = 1.2
    tresh = 0.01

    for sub_folder in sub_folders:
        for filename in os.listdir(os.path.join(base_folder, sub_folder)):
            max_depth+=1
            if filename.endswith('.graphml'):



                print(f"Training on {filename}")
                agent.epsilon = 1.0
                G = nx.read_graphml(os.path.join(base_folder, sub_folder, filename))
                

                

                #check if there are loops
                if not nx.is_directed_acyclic_graph(G):
                    print("There are loops in the graph")

                #print number of loops
                print(f"Number of loops : {len(list(nx.simple_cycles(G)))}")


                #convert G to a non directed graph
                G_undirected = G.to_undirected()
                episode_reward_list = []
                dest_node = None
                for node in G.nodes:
                    if G.nodes[node]['cat'] == 1:
                        dest_node = node
                        break
                print('Best path : ' + str(nx.shortest_path_length(G_undirected, "root", dest_node)))
                

                success_nodes = []
                for node in G.neighbors("root"):
                    if(is_destination_in_successors(G, node)):
                        success_nodes.append(node)
                number_of_success_nodes = len(success_nodes)



                print(f"Keeping successors with probability {tresh}")
                
                node_to_remove = []
                for node in G.neighbors("root"):
                    if np.random.rand() > tresh and node not in success_nodes:
                        node_to_remove.append(node)

                for node in node_to_remove:
                    G.remove_node(node)
                number_of_successors = len(list(G.neighbors("root")))

                print(f"Number of successors : {number_of_successors} with {number_of_success_nodes} success nodes, depth is {max_depth}, probability of success : {number_of_success_nodes/number_of_successors}")
                

                for e in range(n_episodes):

                    state = np.array([0, len(list(G.neighbors("root"))), 0,0,0, 0, 0,0,1,0,0,0])  # num of hops, num of successors
                    #state = torch.tensor(np.array([state]), dtype=torch.float32)
                    done = False
                    current_node = "root"
                    visited = ["root"]
                    visit_count_map = {}
                    visit_count_map["root"] = 1

                    num_hops = 0
                    num_successors = len(list(G.neighbors("root")))
                    max_steps_per_episode = 500  # Set this to a value that makes sense for your task
                    steps_taken = 0
                    total_episode_reward = 0
                    is_max_depth_reached = False
                    curr_depth = 0
                    while not done and steps_taken < max_steps_per_episode:
                        is_max_depth_reached=curr_depth > max_depth

                        steps_taken += 1
                        can_dive = not (state[0][-4] and not is_max_depth_reached)
                        invalid_actions_mask = np.array([can_dive, not state[0][-3], not state[0][-2] , not state[0][-1]])
                        action, _ = agent.select_action(state, invalid_actions_mask=invalid_actions_mask)                        #print action and the number of hops
                        # Handle action and get reward
                        if action == 0:  # dive into the first successor
                            successors = list(G.successors(current_node))
                            if successors:  # if there are successors
                                current_node = successors[0]
                                visited.append(current_node)
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                done = G.nodes[current_node]['cat'] == 1
                                num_visits = visit_count_map.get(current_node, 1)
                                #reward = 100 if G.nodes[current_node]['cat'] == 1 else -1 * num_visits
                                reward = 100 if is_destination_in_successors(G, current_node) else -1 * num_visits

                                if G.nodes[current_node]['cat'] == 1 :
                                    reward = 1000

                                curr_depth += 1

                            else:
                                print("Shoundt be here (dive)")
                            
                        elif action == 1:  # go to the next node from the successors
                            successors = list(G.successors(visited[-2])) if len(visited) >= 2 else []
                            if current_node in successors and successors.index(current_node) < len(successors) - 1:
                                current_node = successors[successors.index(current_node) + 1]
                                visited[-1] = current_node
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                done = G.nodes[current_node]['cat'] == 1
                                num_visits = visit_count_map.get(current_node, 1)

                                #reward = 100 if G.nodes[current_node]['cat'] == 1 else -1 * num_visits
                                reward = 100 if is_destination_in_successors(G, current_node) else -1 * num_visits

                                if G.nodes[current_node]['cat'] == 1 :
                                    reward = 1000                               

                            else:
                                print("Shoundt be here (next)")
                            
                        elif action == 2:  # go to the previous node from the successors
                            successors = list(G.successors(visited[-2])) if len(visited) >= 2 else []
                            if current_node in successors and successors.index(current_node) > 0:
                                current_node = successors[successors.index(current_node) - 1]
                                visited[-1] = current_node
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                done = G.nodes[current_node]['cat'] == 1
                                num_visits = visit_count_map.get(current_node, 1)
                                #reward = 1000 if G.nodes[current_node]['cat'] == 1 else -1 * num_visits

                                #reward = 100 if is_destination_in_successors(G, current_node) else -1 * num_visits

                                #if G.nodes[current_node]['cat'] == 1 :
                                #    reward = 1000

                                
                            else:
                                print("Shoundt be here (prev)")
                            
                        elif action == 3:  # backtrack
                            if len(visited) >= 2:
                                visited.pop()  # remove current node
                                current_node = visited[-1]  # backtrack to the previous node
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                num_visits = visit_count_map.get(current_node, 1)
                                
                                reward = -2*num_visits  # negative reward for backtracking
                                curr_depth -= 1
                            else:
                                print("Shoundt be here (backtrack)")
                        
                        visit_count_map[current_node] = visit_count_map.get(current_node, 0) + 1


                        """
                        if steps_taken >= max_steps_per_episode:
                            reward = -100
                        """
                            
                        #state is of form
                        """
                        <key id="d5" for="node" attr.name="invalid_pointer_count" attr.type="long" />
                        <key id="d4" for="node" attr.name="valid_pointer_count" attr.type="long" />
                        <key id="d3" for="node" attr.name="pointer_count" attr.type="long" />
                        <key id="d2" for="node" attr.name="struct_size" attr.type="long" />
                        <key id="d1" for="node" attr.name="cat" attr.type="long" />
                        <key id="d0" for="node" attr.name="label" attr.type="string" />
                        """
                        successors = list(G.successors(visited[-2])) if len(visited) >= 2 else []
                        visit_count = visit_count_map[current_node]
                        has_prev_node = 1 if current_node in successors and successors.index(current_node) > 0 else 0
                        has_next_node = 1 if current_node in successors and successors.index(current_node) < len(successors) - 1 else 0

                        can_backtrack = 1 if len(visited) >= 2 else 0
                        can_dive = 1 if len(list(G.successors(current_node))) > 0 else 0

                        #get the sum of the times the succesors nodes have been visited
                        sum_visit_successors = 0
                        for successor in list(G.successors(current_node)):
                            sum_visit_successors += visit_count_map.get(successor, 0)
                             
                        next_state = np.array([curr_depth,
                                                num_successors,
                                                sum_visit_successors,
                                                G.nodes[current_node]['struct_size'],
                                                G.nodes[current_node]['pointer_count'],
                                                  G.nodes[current_node]['valid_pointer_count'],
                                                    G.nodes[current_node]['invalid_pointer_count'],
                                                      visit_count,
                                                      can_dive,
                                                      has_next_node,
                                                      has_prev_node,
                                                      can_backtrack])
                        next_state = torch.tensor([next_state], dtype=torch.float32)
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        total_episode_reward += reward
                        if done:
                            print(f" FOUND !!!!! episode: {e}/{n_episodes}, score: {total_episode_reward}, e: {agent.epsilon:.2}")
                            break

                    episode_reward_list.append(total_episode_reward)
                    if len(agent.memory) > batch_size:
                        states, actions, rewards, next_states, dones = agent.memory.sample(batch_size)
                        agent.update_model(states, actions, rewards, next_states, dones)
                    
                    print(f"episode: {e}/{n_episodes}, score: {total_episode_reward}, e: {agent.epsilon:.2}, different nodes visited : {len(visit_count_map)}")

                    #print(f"Distance before reaching goal : {curr_dist}, mask : {invalid_actions_mask}")
                tresh*=tresh_mul
                print(episode_reward_list)
                #plot_progress(episode_reward_list)
                agent_save_filename = f"dqn_{filename}_weights.pth"
                agent.save(os.path.join(folder_name, agent_save_filename))
        

def plot_progress(reward_list):
    #make rolling mean
    rolling_mean = []
    for i in range(len(reward_list)):
        if i < 100:
            rolling_mean.append(np.mean(reward_list[:i]))
        else:
            rolling_mean.append(np.mean(reward_list[i-100:i]))

    plt.plot(rolling_mean)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.show()

if __name__ == "__main__":
    main()
