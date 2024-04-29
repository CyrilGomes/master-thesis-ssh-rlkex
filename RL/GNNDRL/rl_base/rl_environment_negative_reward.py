import gym
from gym import spaces
import networkx as nx
import numpy as np
import torch
import random 
from torch_geometric.data import Data
import torch_geometric.transforms as T

class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_node):
        super(GraphTraversalEnv, self).__init__()

        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

        self.graph = graph
        self.target_node = target_node
        self.episode_index = 0
        self.current_node = self._sample_start_node()
        
        self.visited_stack = []


        self.state_size = 10
        # Update action space for the current node
        self._update_action_space()
        
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.state_size,)),  # Assuming node features are 1-D
            'edge_index': spaces.Tuple((spaces.Discrete(len(self.graph.nodes())), spaces.Discrete(len(self.graph.nodes()))))
        })



    def _get_dist_to_target(self):
        return nx.shortest_path_length(self.graph, self.current_node, self.target_node)



    def _sample_start_node(self):
        
        random_node = None
        while random_node == None or random_node == self.target_node:

            random_node = np.random.choice(list(self.graph.nodes()))
            is_target_reachable = nx.has_path(self.graph, random_node, self.target_node)
            if len(list(self.graph.neighbors(random_node))) == 0 or not is_target_reachable:
                random_node = None
        return random_node
        #get all nodes that has neighbours
        #nodes = [node for node in self.graph.nodes() if len(list(self.graph.neighbors(node))) > 0]
        best_path = nx.shortest_path(self.graph, 0, self.target_node)
        #similarly to simulated annealing, based on the current episode index, we less chance to choose a node from the best path
        
        exploration_prob = np.exp(-self.episode_index/1500)
        if random.random() < exploration_prob:
            return random.choice(list(best_path[0:-1]))


        return 0
    def _update_action_space(self):
        # Number of neighbors
        #TODO : Add + 1 for Backtrack
        

        num_actions = len(list(self.graph.neighbors(self.current_node)))
        self.action_space = spaces.Discrete(num_actions)
        



    def _get_best_action(self):
            neighbors = list(self.graph.neighbors(self.current_node))

            best_path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            best_neighbor = best_path[1]

            #get the index of the best neighbor in the neighbors list
            best_neighbor_index = neighbors.index(best_neighbor)
            return best_neighbor_index



    def step(self, action, printing = False):

        neighbors = list(self.graph.neighbors(self.current_node))
        is_target_reachable = self.target_node in neighbors or nx.has_path(self.graph, self.current_node, self.target_node)

                

        if len(neighbors) > 50:
            print("NEIHBORS : ", neighbors)

        if action >= len(neighbors) or len(neighbors) == 0 or not is_target_reachable:
            return self._get_obs(), -6, True, {'found_target': False}
        
        #Printing stats for debugging
        if printing:
            if not self.target_node in neighbors:
                best_path = nx.shortest_path(self.graph, self.current_node, self.target_node)
                best_neighbor = best_path[1]

                print(f"Current node : {self.current_node} \t Distance to target {len(best_path)} \t Neihbours : {neighbors} \t Target : {self.target_node} \t Action : {neighbors[action]} \t best action : {best_neighbor}")
            else : 
                print(f"Current node : {self.current_node} \t Neihbours : {neighbors} \t Target : {self.target_node} \t Action : {neighbors[action]} \t best action : {self.target_node}")

        self.current_node = neighbors[action]
        has_found_target = self.current_node == self.target_node

        self.increment_visited(self.current_node)
        self.visited_stack.append(self.current_node)
        
        if self.current_node == self.target_node:
            reward = 100 + 1/len(self.visited_stack)
            return self._get_obs(), reward, True, {'found_target': True}


        #check if has neighbors
        if len(list(self.graph.neighbors(self.current_node))) == 0:
            reward = -5
            return self._get_obs(), reward, True, {'found_target': False}


        #check if target is reachable
        reachable = nx.has_path(self.graph, self.current_node, self.target_node)

        r_dist = 0
        if reachable:
            #add a small reward if it get closer to the target
            dist_to_target = self._get_dist_to_target()
            r_dist += 1/dist_to_target
            self._update_action_space()

        return self._get_obs(), r_dist, False, {'found_target': False}


        

    def increment_visited(self, node):
        self.graph.nodes[node].update({'visited': self.graph.nodes[node]['visited'] + 1})

    def reset_visited(self):
        for node in self.graph.nodes():
            self.graph.nodes[node].update({'visited': 0})
    def reset(self):
        self.episode_index += 1

        self.current_node = self._sample_start_node()
        self.visited_stack = []
        self._update_action_space()
        self.reset_visited()
        return self._get_obs()

        
    def _get_obs(self):
        # Current node and its successors
        nodes = [self.current_node] + list(self.graph.neighbors(self.current_node))

        # Extract node attributes. Assuming all nodes have the same attributes. remove the 'cat' feature
        
        subgraph = self.graph.subgraph(nodes)

        #Sinusoidal positional encoding
        pos_enc = torch.zeros((len(nodes), 2))
        pos_enc[:, 0] = torch.sin(torch.arange(0, len(nodes), 1) * (2 * np.pi / len(nodes)))
        pos_enc[:, 1] = torch.cos(torch.arange(0, len(nodes), 1) * (2 * np.pi / len(nodes)))
         

        

        x = torch.tensor([[
            data['struct_size'],
            data['valid_pointer_count'],
            data['invalid_pointer_count'],
            data['visited'],
            data['first_pointer_offset'],
            data['last_pointer_offset'],
            data['first_valid_pointer_offset'],
            data['last_valid_pointer_offset'],
            1 if self.graph.out_degree(node) > 0 else 0,
            #get edge atribute offset from current node to target node
            subgraph[self.current_node][node]['offset'] if node in subgraph[self.current_node] else 0
            
        ] for node, data in subgraph.nodes(data=True)], dtype=torch.float)

        #x = torch.cat((x, pos_enc), dim=1)
        
        # Build edge index for the subgraph
        edge_list = [(self.current_node, neighbor) for neighbor in self.graph.neighbors(self.current_node)]
        edge_index = [[], []]
        edge_attributes = []

        for edge in edge_list:
            src = nodes.index(edge[0])
            dst = nodes.index(edge[1])
            edge_index[0].append(src)
            edge_index[1].append(dst)

        
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        #check if the shape of x is equal to self.state_size
        if x.shape[1] != self.state_size:
            raise ValueError("The shape of x is not equal to self.state_size, x.shape[1] = " + str(x.shape[1]) + " self.state_size = " + str(self.state_size) )
         

        data = Data(x = x, edge_index= edge_index)
        data = T.ToUndirected()(data)


        return data


    def render(self, mode='human'):

        #use matplotlib to plot the graph
        import matplotlib.pyplot as plt
        #nx.draw(G, with_labels=True, labels=labels)
        #spring layout
        labels = {}
        for node in self.graph.nodes:
            labels[node] = str(node) + ' ' + str(self.graph.nodes[node]['cat'])

        #higlight the nodes of the shortest path from start to target by color red
        path = nx.shortest_path(self.graph, 0, self.target_node)
        color_map = []
        for node in self.graph:
            curr_color = 'lightblue'
            if node == self.current_node:
                curr_color = 'green'
            if node in path:
                curr_color = 'red'
            color_map.append(curr_color)
         
        nx.draw_spring(self.graph, with_labels=True, labels = labels, node_color = color_map)
        plt.show()


        
    def close(self):
        pass
