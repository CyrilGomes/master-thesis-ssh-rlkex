import gym
from gym import spaces
import networkx as nx
import numpy as np
import torch
import random 
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
import concurrent.futures
from numba import jit

@jit(nopython=True)
def compute_reward(previous_dist, current_dist, has_found_target, visit_count, 
                TARGET_FOUND_REWARD, STEP_PENALTY, REVISIT_PENALTY, 
                PROXIMITY_BONUS, NEW_NODE_BONUS, NO_PATH_PENALTY, 
                ADDITIONAL_TARGET_MULTIPLIER, visited_keys_count):
    if has_found_target:
        target_reward = TARGET_FOUND_REWARD * (ADDITIONAL_TARGET_MULTIPLIER ** visited_keys_count)
        return target_reward
    if current_dist is None:
        return NO_PATH_PENALTY
    distance_reward = PROXIMITY_BONUS if current_dist < previous_dist else 0
    revisit_penalty = REVISIT_PENALTY * visit_count if visit_count > 0 else 0
    new_node_bonus = NEW_NODE_BONUS if visit_count == 0 else 0
    total_reward = distance_reward + STEP_PENALTY + revisit_penalty + new_node_bonus
    return total_reward

class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_nodes, subgraph_detection_model_path = "models/model.pt", max_episode_steps=30):
        super(GraphTraversalEnv, self).__init__()

        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

        self.graph = graph
        self.target_nodes = target_nodes
        self.episode_index = 0
        self.max_episode_steps = max_episode_steps
        self.shortest_path_cache = {}

        # Define reward and penalty constants
        self.TARGET_FOUND_REWARD = 50
        self.STEP_PENALTY = -1
        self.REVISIT_PENALTY = -1
        self.PROXIMITY_BONUS = 6
        self.NEW_NODE_BONUS = 10
        self.NO_PATH_PENALTY = -10
        self.ADDITIONAL_TARGET_MULTIPLIER = 1.5


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.subgraph_detection_model_path = subgraph_detection_model_path
        self.load_model()
        print("Model loaded !")
        self.visited_stack = []

        self.promising_nodes = self._get_most_promising_subgraph()
        self.graph = self.promising_nodes

        """
        #print the minimum, average and maximum number of children
        print(f"Min children : {min([self.promising_nodes.out_degree(node) for node in self.promising_nodes.nodes()])}")
        print(f"Average children : {np.mean([self.promising_nodes.out_degree(node) for node in self.promising_nodes.nodes()])}")
        print(f"Max children : {max([self.promising_nodes.out_degree(node) for node in self.promising_nodes.nodes()])}")
         
        """

        self.sorted_promising_nodes = self.sort_promising_nodes()
        self.current_node_iterator = 0
        #self.centrality = nx.betweenness_centrality(self.graph)

        self.current_node = self._sample_start_node()
        self.current_subtree_root = self.current_node


        self.state_size = 11
        # Update action space for the current node
        self._update_action_space()
        
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.state_size,)),  # Assuming node features are 1-D
            'edge_index': spaces.Tuple((spaces.Discrete(len(self.graph.nodes())), spaces.Discrete(len(self.graph.nodes()))))
        })

        self.visited_keys = []

        self.nb_targets = len(self.target_nodes)

        #self.curr_dist = self._get_dist_to_target()

        self.nb_actions_taken = 0



    def get_closest_target(self):
        closest_target = None
        closest_target_dist = float('inf')
        minimum_dist = 0
        nb_reachable_targets = 0
        for target in [t for t in self.target_nodes if t not in self.visited_keys]:
            if (self.current_node, target) not in self.shortest_path_cache:
                try:
                    path = nx.shortest_path(self.promising_nodes, self.current_node, target)
                    self.shortest_path_cache[(self.current_node, target)] = path
                    minimum_dist += len(path)
                    nb_reachable_targets += 1
                except nx.NetworkXNoPath:
                    continue
            path = self.shortest_path_cache[(self.current_node, target)]
            if len(path) < closest_target_dist:
                closest_target_dist = len(path)
                closest_target = target

        return closest_target


    def update_target(self):
        #update the target node to the closest target node
        self.target_node = self.get_closest_target()


    def load_model(self):
        if not hasattr(self, 'subgraph_detection_model'):
            self.subgraph_detection_model = torch.load(self.subgraph_detection_model_path)


    def sort_promising_nodes(self):
        promising_nodes = np.array([node for node in self.promising_nodes.nodes() if self.promising_nodes.in_degree(node) == 0])
        children_count = np.array([self.promising_nodes.out_degree(node) for node in promising_nodes])
        sorted_indices = np.argsort(children_count)[::-1]
        return promising_nodes[sorted_indices].tolist()

    

    def graph_to_data(self, graph):
        # Get a mapping from old node indices to new ones
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        # Use the node mapping to convert node indices
        edge_index = torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in graph.edges()], dtype=torch.long).t().contiguous()

        x = torch.tensor([[
            attributes['struct_size'],
            attributes['valid_pointer_count'],
            attributes['invalid_pointer_count'],
            attributes['first_pointer_offset'],
            attributes['last_pointer_offset'],
            attributes['first_valid_pointer_offset'],
            attributes['last_valid_pointer_offset'],
        ] for _, attributes in graph.nodes(data=True)], dtype=torch.float)

        edge_attr = torch.tensor([graph[u][v]['offset'] for u, v in graph.edges], dtype=torch.float).unsqueeze(1)
        # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise
        y = torch.tensor([1 if any(attributes['cat'] == 1 for _, attributes in graph.nodes(data=True)) else 0], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(self.device)
    
    def create_subgraph_data(self):
        subgraphs = [self.graph.subgraph(c) for c in nx.connected_components(self.graph.to_undirected())]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            subgraphs_data = list(executor.map(self.graph_to_data, subgraphs))

        return subgraphs, subgraphs_data



    def _get_dist_to_target(self):
        # If no path exists, return None.
        try:
            path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            return len(path) - 1
        except nx.NetworkXNoPath:
            return None


    def _get_most_promising_subgraph(self):
        #iterate over all subgraphs to get scores
        subgraphs, subgraphs_data = self.create_subgraph_data()
        model = self.subgraph_detection_model.eval()
        best_score = -1
        best_graph = None
        with torch.no_grad():
            for subgraph, data in zip(subgraphs, subgraphs_data):
                score = model(data)
                if score > best_score:
                    best_score = score
                    best_graph = subgraph

        return best_graph


    def get_random_promising_node(self):

        """
        # Compute betweenness centrality for all nodes in the graph
        centrality = self.centrality

        #return the a node based on probability distribution based on the centrality score
        nodes = list(centrality.keys())
        centrality_scores = list(centrality.values())
        centrality_scores = [score/sum(centrality_scores) for score in centrality_scores]
        return np.random.choice(nodes, p=centrality_scores)
        """
        


        #get the node with no parents from the subgraph with the highest score
        #weight based on the number of children

        #get all nodes that has no parents
        nodes = [node for node in self.promising_nodes.nodes() if len(list(self.promising_nodes.predecessors(node))) == 0]

        #get the number of children for each node
        children_count = [len(list(self.promising_nodes.successors(node))) for node in nodes]
        #normalize the children count
        children_count = [count/sum(children_count) for count in children_count]
        #sample a node based on the children count
        return np.random.choice(nodes, p=children_count)
    
    def get_start_from_best_bath(self):
        #get all nodes that has neighbours
        #nodes = [node for node in self.graph.nodes() if len(list(self.graph.neighbors(node))) > 0]
        best_path = nx.shortest_path(self.graph, 0, self.target_node)
        #similarly to simulated annealing, based on the current episode index, we less chance to choose a node from the best path
        
        exploration_prob = np.exp(-self.episode_index/1500)
        if random.random() < exploration_prob:
            return random.choice(list(best_path[0:-1]))


        return 0
    
        
    def get_neighbour_from_root(self):
        #return the episode_index-th neighbour of node 0 modulo the number of neighbours
        node_neighbors = list(self.graph.neighbors("root"))
        #print(f"Sampling from {len(node_neighbors)} neighbours the {self.episode_index % len(node_neighbors)}th neighbour")
        return node_neighbors[self.episode_index % len(node_neighbors)]
    
    def get_first_neighbour_from_root(self):
        #return the episode_index-th neighbour of node 0 modulo the number of neighbours
        node_neighbors = list(self.graph.neighbors("root"))
        #print(f"Sampling from {len(node_neighbors)} neighbours the {self.episode_index % len(node_neighbors)}th neighbour")
        return node_neighbors[0]


    def skip_to_next_root(self):
        self.current_node_iterator = (self.current_node_iterator + 1) % len(self.sorted_promising_nodes)


    def _get_next_subgraph_root_node(self):
        node_to_return = self.sorted_promising_nodes[self.current_node_iterator]
        return node_to_return

    def _sample_start_node(self):

        return self._get_next_subgraph_root_node()

    def _update_action_space(self):
        # Number of neighbors
        #TODO : Add + 1 for Backtrack
        
        num_actions = 50#len(list(self.graph.neighbors(self.current_node)))

        self.action_space = spaces.Discrete(num_actions)
        

    def _get_best_action(self):
            neighbors = list(self.graph.neighbors(self.current_node))

            best_path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            best_neighbor = best_path[1]

            #get the index of the best neighbor in the neighbors list
            best_neighbor_index = neighbors.index(best_neighbor)
            return best_neighbor_index


    def get_next_root_neighbour(self):
        #ge the next root neighbour to iterate
        node_neighbors = list(self.graph.neighbors("root"))
        return node_neighbors[(node_neighbors.index(self.current_node) + 1) % len(node_neighbors)]


    def is_target_in_subtree(self):
        return nx.has_path(self.graph, self.current_node, self.target_node)



    def restart_agent_from_root(self):
        self.current_node = self._sample_start_node()
        self.visited_stack.append(self.current_node)
        self.increment_visited(self.current_node)
        self._update_action_space()


    def get_valid_actions(self):
        #valid actions are the neighbours of the current node + the skip action (the last action = 48)
        neighbors = list(self.graph.neighbors(self.current_node))
        valid_actions = [i for i in range(len(neighbors))] + [49]


        return valid_actions


    def get_action_mask(self):
        # Example implementation, customize based on your environment's logic
        valid_actions = self.get_valid_actions()  # You need to implement this
        action_mask = -1e6 * np.ones(self.action_space.n)
        action_mask[valid_actions] = 0
        return action_mask


    def step(self, action, printing=False):

        if self.nb_actions_taken > self.max_episode_steps:
            return self._get_obs(), -1, True, {'found_target': True,
                                               'max_episode_steps_reached': True,
                          
                             'nb_keys_found': len(self.visited_keys)}

        if action == self.action_space.n - 1: #skip to the next subtree

            #this is the stop action, meaning the agent think found all the keys
            if len(self.visited_keys) == self.nb_targets:
                return self._get_obs(), 100, True, {'found_target': False,
                                               'nb_keys_found': len(self.visited_keys)}
            else :
                return self._get_obs(), -1, True, {'found_target': False, 'nb_keys_found': len(self.visited_keys)}


            self.skip_to_next_root()
            self.restart_agent_from_root()


            return self._get_obs(), 0, False, {'found_target': False,
                                               'nb_keys_found': len(self.visited_keys)}

        self.nb_actions_taken += 1
        self.update_target()
        if self.target_node is None:
            #return to base node
            self.restart_agent_from_root()
            return self._get_obs(), -1, False, {'found_target': False,
                                                'nb_keys_found': len(self.visited_keys)}
        
        neighbors = list(self.graph.neighbors(self.current_node))
        
        # Note: The function nx.has_path can raise an exception if no path exists. 
        # We need to handle that exception and set is_target_reachable accordingly.
        try:
            is_target_reachable = self.target_node in neighbors or nx.has_path(self.graph, self.current_node, self.target_node)
        except nx.NetworkXNoPath:
            is_target_reachable = False


    

        if not is_target_reachable:
            self.restart_agent_from_root()

            return self._get_obs(), -1, False, {'found_target': False,
                                                'nb_keys_found': len(self.visited_keys)}

        # Calculate previous distance before executing action
        previous_dist = self._get_dist_to_target()

        # Execute the action
        self.current_node = neighbors[action]
        has_found_target = self.current_node == self.target_node

        # After moving to the new node:
        current_dist = self._get_dist_to_target() if not has_found_target else 0
        is_revisited = self.current_node in self.visited_stack

        self.increment_visited(self.current_node)

        # Update visited nodes
        if not is_revisited:
            self.visited_stack.append(self.current_node)

        # Compute the reward
        visit_count = self.graph.nodes[self.current_node]['visited']
        reward = compute_reward(previous_dist, current_dist, has_found_target, visit_count, 
                                     self.TARGET_FOUND_REWARD, self.STEP_PENALTY, self.REVISIT_PENALTY, 
                                     self.PROXIMITY_BONUS, self.NEW_NODE_BONUS, self.NO_PATH_PENALTY, 
                                     self.ADDITIONAL_TARGET_MULTIPLIER, len(self.visited_keys)) 
        

        if has_found_target:
            self.visited_keys.append(self.current_node)
            is_done = False #len(self.visited_keys) == self.nb_targets

            if not is_done:
                self.restart_agent_from_root()


            return self._get_obs(), reward, is_done, {'found_target': True,
                                                      'nb_keys_found': len(self.visited_keys)}

        # Recalculate the neighbors after updating the current_node
        neighbors = list(self.graph.neighbors(self.current_node))

        # If the current node has no neighbors, it's a terminal state (unless the target was found, which we've already checked).
        if not len(neighbors):
            self.restart_agent_from_root()

            return self._get_obs(), reward, False, {'found_target': False,
                                                    'nb_keys_found': len(self.visited_keys)}

        # Otherwise, update action space (if such a method is required) and continue.
        self._update_action_space()
        
        return self._get_obs(), reward, False, {'found_target': False,
                                                'nb_keys_found': len(self.visited_keys)}

    def increment_visited(self, node):
        self.graph.nodes[node].update({'visited': self.graph.nodes[node]['visited'] + 1})

    def reset_visited(self):
        for node in self.graph.nodes():
            self.graph.nodes[node].update({'visited': 0})
        
        
    def reset(self):
        self.episode_index += 1
        self.current_node_iterator = 0
        self.current_node = self._sample_start_node()
        self.current_subtree_root = self.current_node
        self.visited_stack = []
        self._update_action_space()
        self.reset_visited()
        self.visited_keys = []
        self.nb_actions_taken = 0
        #self.shortest_path_cache = {}
        return self._get_obs()


    def _get_obs(self):
        # Current node and its successors
        nodes = [self.current_node] + list(self.graph.neighbors(self.current_node))

        # Extract node attributes in a vectorized way using list comprehensions
        attributes = np.array([[data['struct_size'],
                                data['valid_pointer_count'],
                                data['invalid_pointer_count'],
                                data['visited'],
                                data['first_pointer_offset'],
                                data['last_pointer_offset'],
                                data['first_valid_pointer_offset'],
                                data['last_valid_pointer_offset'],
                                1 if self.graph.out_degree(node) > 0 else 0,
                                self.graph[self.current_node][node]['offset'] if node in self.graph[self.current_node] else 0,
                                len(self.visited_keys)
                                ] for node, data in self.graph.subgraph(nodes).nodes(data=True)], dtype=np.float32)

        # Convert attributes to a PyTorch tensor
        x = torch.from_numpy(attributes)

        # Build edge index for the subgraph
        edge_index = [[], []]
        for neighbor in self.graph.neighbors(self.current_node):
            edge_index[0].append(nodes.index(self.current_node))
            edge_index[1].append(nodes.index(neighbor))

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Check if the shape of x matches self.state_size
        if x.shape[1] != self.state_size:
            raise ValueError(f"The shape of x ({x.shape[1]}) does not match self.state_size ({self.state_size})")

        # Create PyTorch Geometric data
        data = Data(x=x, edge_index=edge_index)

        # Apply transform to convert data to undirected, if necessary
        data = T.ToUndirected()(data)

        return data




    def _print_step_debug(self, neighbors, action, printing):
        if not printing:
            return

        if self.target_node not in neighbors:
            best_path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            best_neighbor = best_path[1]
            print(f"Current node: {self.current_node} \t Distance to target {len(best_path)} \t Neighbors: {neighbors} \t Target: {self.target_node} \t Action: {neighbors[action]} \t Best action: {best_neighbor}")
        else:
            print(f"Current node: {self.current_node} \t Neighbors: {neighbors} \t Target: {self.target_node} \t Action: {neighbors[action]} \t Best action: {self.target_node}")



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

    def step_ugly(self, action, printing = False):
        neighbors = list(self.graph.neighbors(self.current_node))

        """
        #if action == len(neighbors), then skip to the next subtree
        if action == len(neighbors):
            self.current_node = self.get_next_root_neighbour()
            self.current_subtree_root = self.current_node
            self.increment_visited(self.current_node)
            self.visited_stack.append(self.current_node)
            self._update_action_space()
            reward = -1
            if self.is_target_in_subtree():
                reward = 3

            print(f"Skipping to the next subtree, reward = {reward}")

            return self._get_obs(), reward, False, {'found_target': False}
        
        """
        is_target_reachable = self.target_node in neighbors or nx.has_path(self.graph, self.current_node, self.target_node)

                

        #if len(neighbors) > 50:
        #    print("NEIHBORS : ", neighbors)

        if action >= len(neighbors) or len(neighbors) == 0 or not is_target_reachable:
            return self._get_obs(), 0, True, {'found_target': False}
        

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
        
        r_global = 0
        #check if the current node is the target node
        if has_found_target:
            r_global = 3
        else :
            reachable = nx.has_path(self.graph, self.current_node, self.target_node)
            if reachable:
                dist = self._get_dist_to_target()
                
                r_global = 1/dist
            else :
                r_global = 0


        r_efficiency = 1/len(self.visited_stack)

        r_newly_visited = 0
        if self.current_node not in self.visited_stack:
            r_newly_visited = 1
        else :
            r_newly_visited = 0

        w_global = 10
        w_efficiency = 1
        w_newly_visited = 2

        #check if it has neighbours, if no then stop
        has_neighbors = len(list(self.graph.neighbors(self.current_node))) > 0
        if has_neighbors:
            self._update_action_space()
        else:
            if has_found_target:
                return self._get_obs(), w_global*r_global + w_efficiency*r_efficiency + w_newly_visited*r_newly_visited, True, {'found_target': True}
        
        #self.current_node = self.current_subtree_root
        self._update_action_space()
        #print(is_done)
        return self._get_obs(), w_global*r_global + w_efficiency*r_efficiency + w_newly_visited*r_newly_visited, False, {'found_target': False}
