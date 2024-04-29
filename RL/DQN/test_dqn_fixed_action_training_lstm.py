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
# Define the Q-network
class DQN(nn.Module):
    #Define a DQN that take into account the states, and history of states with lstm
    def __init__(self, state_space=2, action_space=4):
        super(DQN, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.lstm = nn.LSTM(self.state_space, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.action_space)
    
    def forward(self, x):
        #x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = F.relu(self.fc1(x[:,-1,:]))
        x = self.fc2(x)
        return x



# Define the DQN agent
class DQNAgent:
    def __init__(self, state_space=2, action_space=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_space, action_space).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.device = device

    def remember(self, state, action, reward, state_history, done):
        self.memory.append((state, action, reward, state_history, done))

    def act(self, state, invalid_actions_mask=None):
        
        if np.random.rand() <= self.epsilon:
            #choose a random action based on the invalid_actions_mask
            if invalid_actions_mask is not None:
                invalid_actions_mask = torch.tensor(invalid_actions_mask, dtype=torch.bool)
                valid_actions = torch.where(invalid_actions_mask == False)[0]
                return np.random.choice(valid_actions)
                   

            else : 
                return np.random.choice(self.action_space)
        
        
        model_out = self.model(state.to(self.device))
        if invalid_actions_mask is not None:
            invalid_actions_mask = torch.tensor(invalid_actions_mask, dtype=torch.bool)  # Convert to a Boolean tensor
            model_out[0][invalid_actions_mask] = -np.inf  # Set the invalid actions to -inf


        actout = np.argmax(model_out.cpu().detach().numpy())

        #print(f"actout : {actout}")
        return np.argmax(model_out.cpu().detach())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            target = self.model(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(self.model(next_state).cpu().detach().numpy())
            target = target.view(1, -1)
            current = self.model(state)
            loss = self.loss_fn(current, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

def main():
    agent = DQNAgent(state_space=17)
    #agent.load("dqn_10508-1650982250-heap.graphml_weights.pth")

    batch_size = 32
    n_episodes = 1300

    base_folder = "Generated Graphs"
    sub_folders = ["64"]

    for sub_folder in sub_folders:
        for filename in os.listdir(os.path.join(base_folder, sub_folder)):
            if filename.endswith('.graphml'):
                print(filename)


                print(f"Training on {filename}")
                agent.epsilon = 1.0
                G = nx.read_graphml(os.path.join(base_folder, sub_folder, filename))
                #convert G to a non directed graph
                G_undirected = G.to_undirected()
                episode_reward_list = []
                dest_node = None
                root_node = None
                for node in G.nodes:
                    if G.nodes[node]['label'] == 'root':
                        root_node = node
                        
                    if G.nodes[node]['cat'] == 1:
                        dest_node = node
                    
                    if dest_node is not None and root_node is not None:
                        break
                print(f"Destination node : {dest_node}")
                print('Best path : ' + str(nx.shortest_path_length(G_undirected, root_node, dest_node)))
                reward = 0

                state_history = []
                for e in range(n_episodes):

                    reward = 0
                    state = np.array([0, len(list(G.neighbors(root_node))), 0,0, 0,0,0,0,0,0, 0, 0,0,1,0,0,0])  # num of hops, num of successors
                    state = torch.tensor(np.array([state]), dtype=torch.float32)
                    done = False
                    current_node = root_node
                    visited = [root_node]
                    visit_count_map = {}
                    visit_count_map[root_node] = 1

                    num_hops = 0
                    num_successors = len(list(G.neighbors(root_node)))
                    max_steps_per_episode = 500  # Set this to a value that makes sense for your task
                    steps_taken = 0
                    total_episode_reward = 0
                    curr_dist = 0
                    while not done and steps_taken < max_steps_per_episode:
                        steps_taken += 1
                        invalid_actions_mask = np.array([not state[0][-4], not state[0][-3], not state[0][-2] , not state[0][-1]])
                        

                        action = agent.act(state, invalid_actions_mask=invalid_actions_mask)

                        #print action and the number of hops
                        # Handle action and get reward
                        if action == 0:  # dive into the first successor
                            successors = list(G.successors(current_node))
                            if successors:  # if there are successors
                                current_node = successors[0]
                                visited.append(current_node)
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                done = G.nodes[current_node]['cat'] == 1
                            
                                #reward = 100 if G.nodes[current_node]['cat'] == 1 else -1
                            
                            else:
                                reward = -9999  # negative reward if no successors
                                print("Shoundt be here (dive)")
                            
                        elif action == 1:  # go to the next node from the successors
                            successors = list(G.successors(visited[-2])) if len(visited) >= 2 else []
                            if current_node in successors and successors.index(current_node) < len(successors) - 1:
                                current_node = successors[successors.index(current_node) + 1]
                                visited[-1] = current_node
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                done = G.nodes[current_node]['cat'] == 1
                            
                                #reward = 100 if G.nodes[current_node]['cat'] == 1 else -1
                            else:
                                reward = -9999  # negative reward if no next node
                                print("Shoundt be here (next)")
                            
                        elif action == 2:  # go to the previous node from the successors
                            successors = list(G.successors(visited[-2])) if len(visited) >= 2 else []
                            if current_node in successors and successors.index(current_node) > 0:
                                current_node = successors[successors.index(current_node) - 1]
                                visited[-1] = current_node
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                done = G.nodes[current_node]['cat'] == 1
                            
                                #reward = 100 if G.nodes[current_node]['cat'] == 1 else -1
                            else:
                                reward = -9999  # negative reward if no previous node
                                print("Shoundt be here (prev)")
                            
                        elif action == 3:  # backtrack
                            if len(visited) >= 2:
                                visited.pop()  # remove current node
                                current_node = visited[-1]  # backtrack to the previous node
                                num_hops = len(visited) - 1
                                num_successors = len(list(G.neighbors(current_node)))
                                
                                #reward = -2  # negative reward for backtracking
                            else:
                                reward = -9999  # negative reward if cannot backtrack
                                print("Shoundt be here (backtrack)")
                        
                        visit_count_map[current_node] = visit_count_map.get(current_node, 0) + 1


                        #if the current node is closer to the node with 'cat' label, give a positive reward
                        #if the current node is farther from the node with 'cat' label, give a negative reward
                        
                        new_dist = get_distance_between_nodes(G_undirected, current_node, dest_node)
                        #print(f'new dist: {new_dist}, curr dist: {curr_dist}')
                        action_label = ['dive', 'next', 'prev', 'backtrack']
                        masken_action_labels = np.ma.array(action_label, mask=invalid_actions_mask)
                        #print(f"masked action {masken_action_labels} choosed action label { action_label[action]}")
                        chosen_action = action_label[action]
                        #print(f'current node: {current_node}, dest node: {dest_node}, action : {chosen_action}, reward : {reward}' )
                        if new_dist == 0 or done:
                            print('Wow')
                            reward = 100
                        if new_dist < curr_dist:
                            reward = 2
                        else :
                            reward = -1*new_dist*visit_count_map[current_node]
                            #print(f'\tAction : {chosen_action}, reward : {reward}')


                        curr_dist = new_dist


                        if steps_taken >= max_steps_per_episode:
                            reward -= 0

                            
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
                        parent_node = list(G.predecessors(current_node))[0] if len(list(G.predecessors(current_node))) > 0 else ''
                        next_state = np.array([num_hops,
                                                num_successors,
                                                G.nodes[current_node]['offset'],
                                                G.nodes[current_node]['struct_size'],
                                                G.nodes[current_node]['pointer_count'],
                                                G.nodes[current_node]['valid_pointer_count'],
                                                    G.nodes[current_node]['invalid_pointer_count'],
                                                    G.nodes[parent_node]['offset'] if parent_node != '' else 0,
                                                    G.nodes[parent_node]['struct_size'] if parent_node != '' else 0,
                                                    G.nodes[parent_node]['pointer_count'] if parent_node != '' else 0,
                                                    G.nodes[parent_node]['valid_pointer_count'] if parent_node != '' else 0,
                                                    G.nodes[parent_node]['invalid_pointer_count'] if parent_node != ''  else 0,
                                                      visit_count,
                                                      can_dive,
                                                      has_next_node,
                                                      has_prev_node,
                                                      can_backtrack])
                        

                        state_history.append(next_state)
                        next_state = torch.tensor([next_state], dtype=torch.float32)
                        
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        total_episode_reward += reward
                        if done:
                            print(f" FOUND !!!!! episode: {e}/{n_episodes}, score: {total_episode_reward}, e: {agent.epsilon:.2}")
                            break

                    episode_reward_list.append(total_episode_reward)
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                    
                    print(f"episode: {e}/{n_episodes}, score: {total_episode_reward}, e: {agent.epsilon:.2}, different nodes visited : {len(visit_count_map)}")

                    #print(f"Distance before reaching goal : {curr_dist}, mask : {invalid_actions_mask}")
                print(episode_reward_list)
                plot_progress(episode_reward_list)
                agent.save(f"dqn_{filename}_weights.pth")


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
