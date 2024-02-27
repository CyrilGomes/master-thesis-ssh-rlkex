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
from torch_geometric.utils import to_undirected
from root_heuristic_rf import GraphPredictor

def compute_reward(has_found_target, 
                   TARGET_FOUND_REWARD, STEP_PENALTY, 
                    NO_PATH_PENALTY, visited_keys_count, has_path):
    

    if has_found_target:
        target_reward =  TARGET_FOUND_REWARD
        return target_reward
    if has_path == False:
        return NO_PATH_PENALTY

    total_reward = STEP_PENALTY
    #print(f"Distance reward: {distance_reward}, Step penalty: {STEP_PENALTY}, Revisit penalty: {revisit_penalty}, New node bonus: {new_node_bonus}")
    return total_reward
class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_nodes, root_detector, max_episode_steps=20, obs_is_full_graph=False):
        """
        Initializes the Graph Traversal Environment.

        Args:
            graph (nx.Graph): A NetworkX graph.
            target_nodes (list): A list of target nodes within the graph.
            subgraph_detection_model_path (str): Path to the trained subgraph detection model.
            max_episode_steps (int): Maximum steps allowed per episode.
        """
        super(GraphTraversalEnv, self).__init__()
        self.path_cache = {}

        self._validate_graph(graph)
        self.main_graph = graph


        self.target_nodes_map = target_nodes
        #create an array of target nodes sorted by their value in the map
        self.target_nodes = [node for node in self.target_nodes_map.keys()]
        self.target_nodes = sorted(self.target_nodes, key=self.target_nodes_map.get)

        self.max_episode_steps = max_episode_steps
        self.shortest_path_cache = {}
        self.visited_stack = []
        self.current_node_iterator = 0
        self.obs_is_full_graph = obs_is_full_graph

        self._init_rewards_and_penalties()
        #load model
        self.root_detector = root_detector
        print("Model loaded!")
        #get proba for all root nodes, returns a map of root node -> proba
        self.root_proba = self.root_detector.predict_probabilities(self.main_graph)


        #sort every roots based on proba, ommit the ones with less than 0.5 proba
         
        self.sorted_roots = sorted(self.root_proba, key=self.root_proba.get, reverse=True)

        #remvove the roots with proba < 0.5
        self.sorted_roots = [root for root in self.sorted_roots if self.root_proba[root] > 0.5]
        #from the current_node_iterator, get the corresponding root
        self.best_root, ref_graph = self._get_best_root()
        #update the graph to be the subgraph of the root (using BFS)
        self.reference_graph = ref_graph

        #create a copy of the reference graph
        self.graph = self.reference_graph.copy()

        self.state_size = 7 if self.obs_is_full_graph else 11

        self.observation_space = self._define_observation_space()
        self.visited_keys = {}
        self.nb_targets = len(self.target_nodes)
        self.nb_actions_taken = 0
        self.node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.inverse_node_mapping = {v: k for k, v in self.node_mapping.items()}
        self.target_node = None
        self.reset()
        self.assess_target_complexity()


    def assess_target_complexity(self):
        print("-------------------- ASSESSING TARGET COMPLEXITY ----------------------")
        cycles = list(nx.simple_cycles(self.graph))

        has_cycle = len(cycles) > 0
        print(f"Has cycles: {has_cycle}")
        if has_cycle:
            print("Cycles found:", cycles)
        print(f"Number of targets: {self.nb_targets}")
        print(f"Number of nodes in the graph: {len(self.graph.nodes)}")
        print(f"Number of edges in the graph: {len(self.graph.edges)}")
        #get the number of nodes between the current node and the target nodes
        sum_path = 0
        for target in self.target_nodes:
            sum_path += self._get_path_length(self.current_node, target)
        print(f"Path length from current node to target nodes: {sum_path}")
        mean_number_of_neighbors = np.mean([self.graph.out_degree(node) for node in self.graph.nodes])
        print(f"Mean number of neighbors: {mean_number_of_neighbors}")
        depth = nx.dag_longest_path_length(self.graph)
        print(f"Depth of the graph: {depth}")
        nb_neighbours_root = self.graph.out_degree(self.best_root)
        print(f"Number of neighbors of the root: {nb_neighbours_root}")
        print("------------------------------------------------------------------------")


    def get_number_of_nodes(self):
        return len(self.graph.nodes)

    def _get_best_root(self):

        #get root with highest proba
        best_root = self.sorted_roots[self.current_node_iterator]

        best_subgraph = nx.bfs_tree(self.main_graph, best_root)


        #check if there is a path from root to all target nodes
        has_path = {}
        for target in self.target_nodes:
            try:
                path = nx.shortest_path(best_subgraph, best_root, target)
                has_path[target] = True
            except nx.NetworkXNoPath:
                has_path[target] = False

        #if there is no path from root to all target nodes, throw an error
        if not all(has_path.values()):
            raise ValueError("There is no path from root to all target nodes")

        for node in best_subgraph.nodes():
            # Copy node attributes
            best_subgraph.nodes[node].update(self.main_graph.nodes[node])
        for u, v in best_subgraph.edges():
            # In a multigraph, there might be multiple edges between u and v.
            # Here, we take the attributes of the first edge.
            if self.main_graph.has_edge(u, v):
                key = next(iter(self.main_graph[u][v]))
                best_subgraph.edges[u, v].update(self.main_graph.edges[u, v, key])

        return best_root, best_subgraph



    def _validate_graph(self, graph):
        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

    def _init_rewards_and_penalties(self):
        self.TARGET_FOUND_REWARD = 10
        self.PROXIMITY_MULTIPLIER = 4
        self.INCORRECT_TARGET_REWARD = -10
        self.NO_PATH_PENALTY = -10
        self.NO_TARGET_FOUND_PENALTY = -10


    def _get_closest_target(self):
        """
        Finds the closest target node to the current node.

        Returns:
            Node: The closest target node.
        """
        closest_target, closest_target_dist = None, float('inf')
        for target in [t for t in self.target_nodes if t not in self.visited_keys]:
            path_length = self._get_path_length(self.current_node, target)
            if path_length is not None and path_length < closest_target_dist:
                closest_target_dist, closest_target = path_length, target
        return closest_target

    def _get_path_length(self, source, target):
        """
        Gets the length of the shortest path between two nodes.

        Args:
            source: The source node.
            target: The target node.

        Returns:
            int or None: The length of the shortest path or None if no path exists.
        """
        if (source, target) not in self.shortest_path_cache:
            try:
                path = nx.shortest_path(self.graph, source, target)
                self.shortest_path_cache[(source, target)] = len(path) - 1
            except nx.NetworkXNoPath:
                return None
        return self.shortest_path_cache[(source, target)]
    


    def _get_dist_to_target(self):
        """
        Calculates the distance from the current node to the target node.

        Returns:
            int or None: Distance to the target node or None if no path exists.
        """
        # If no path exists, return None.
        try:
            # Check if the path is in the cache
            if (self.current_node, self.target_node) in self.path_cache:
                return self.path_cache[(self.current_node, self.target_node)]
            
            path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            dist = len(path) - 1

            # Store the path length in the cache
            self.path_cache[(self.current_node, self.target_node)] = dist

            return dist
        except nx.NetworkXNoPath:
            return None


    def _sample_start_node(self):
        """
        Samples a start node for the agent.

        Returns:
            Node: A node from sorted promising roots to start the episode.
        """

        #only keep the roots with proba > 0.5
        return self.best_root




    def _update_action_space(self):
        """
        Updates the action space based on the current node.
        """
        num_actions = self.graph.out_degree(self.current_node)
        #if num action is 0, it means we are at a leaf node, we should restart the agent from the root
        if num_actions > 0:
            self.action_space = spaces.Discrete(num_actions)

        
    def _define_observation_space(self):
        """
        Defines the observation space for the environment.

        Returns:
            gym.spaces: Observation space object.
        """
        return spaces.Dict({
            'x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.state_size,)),
            'edge_index': spaces.Tuple((spaces.Discrete(len(self.graph.nodes())), spaces.Discrete(len(self.graph.nodes()))))
        })
    

    def _restart_agent_from_root(self):
        """
        Resets the agent's position to the start node.
        """
        self.current_node = self._sample_start_node()
        self.visited_stack.append(self.current_node)
        self._increment_visited(self.current_node)
        self._update_action_space()

    def _get_valid_actions(self):
        """
        Determines the valid actions that can be taken from the current node.

        Returns:
            list: A list of valid action indices in PyTorch Geometric format.
        """
        neighbors = list(self.graph.successors(self.current_node))
        #For sanity check that the node mapping is correct
        
        
        valid_actions = [self.node_mapping[nx_id] for nx_id in neighbors if nx_id in self.node_mapping]
        #convert the valid actions to the node mapping
        reversed_valid_actions = [self.inverse_node_mapping[valid_action] for valid_action in valid_actions]

        #check if neighbors are the same as the reversed valid actions
        if not set(neighbors) == set(reversed_valid_actions):
            raise ValueError(f"Neighbors: {neighbors}, reversed valid actions: {reversed_valid_actions}")
        


        return valid_actions



    def _get_best_action(self):
        """
        Gets the best action to take from the current node.

        Returns:
            int[]: The best actions to take.
        """


        neighbors = list(self.graph.successors(self.current_node))
        has_path_neighbors = []
        for neighbor in neighbors:
            #TODO : FIX the no target node
            has_path = nx.has_path(self.graph, neighbor, self.target_node)
            if has_path:
                has_path_neighbors.append(neighbor)

        neighbors_data_idx = [self.node_mapping[neighbor] for neighbor in has_path_neighbors]
        return neighbors_data_idx


    def _get_action_mask(self, supervised_prob = 0):
        """
        Creates a mask for valid actions in the action space.

        Returns:
            np.array: An array representing the action mask.
        """
        valid_actions = self._get_valid_actions()


        if np.random.rand() < supervised_prob:
            #get the best action
            best_actions = self._get_best_action()
            #chech if best action is in valid actions
            if not set(best_actions).issubset(set(valid_actions)):
                raise ValueError(f"Best actions: {best_actions}, valid actions: {valid_actions}")
            valid_actions = best_actions


        action_mask = torch.full((len(self.node_mapping),), 0)
        action_mask[valid_actions] = 1


        return action_mask
    
    def get_goal_one_hot(self, goal):
        one_hot_targets = torch.zeros(6, dtype=torch.float)

        one_hot_targets[goal] = 1
        return one_hot_targets


    def set_target_goal(self, goal):
        #goal is the value from 0 to 6, get the key from the map that has this value
        goal_node = [key for key, value in self.target_nodes_map.items() if value == goal][0]
        self.target_node = goal_node


    def step(self, action):
        


        #check if has neighbors
        if self.graph.out_degree(self.current_node) == 0:
            raise ValueError(f"Current node {self.current_node} has no neighbors")


        #convert the id from Data to the id from the graph
        action = self.inverse_node_mapping[action]

        prev_dist = self._get_dist_to_target()

        self._perform_action(action)

        new_dist = self._get_dist_to_target()
        reward = 0

        has_found_target = self.current_node == self.target_node

        if not has_found_target:
            if new_dist is None and prev_dist is not None:
                reward -= self.NO_PATH_PENALTY
            elif new_dist is not None:
                    reward += self.PROXIMITY_MULTIPLIER

        self.nb_actions_taken += 1

        is_incorect_leaf = False
        new_goal = None
        if has_found_target:
            
            obs = self._get_obs()
            reward = self.TARGET_FOUND_REWARD
            return obs, reward, True, self._episode_info(found_target=True), new_goal
        elif self.graph.out_degree(self.current_node) == 0:

            reward = self.INCORRECT_TARGET_REWARD if self.current_node in self.target_nodes else self.NO_TARGET_FOUND_PENALTY 

            #prob_of_new_target = 0.5
            #if self.current_node in self.target_nodes:
                #if np.random.rand() < prob_of_new_target:
                #    new_goal = self.target_nodes_map[self.current_node]
                #    reward = 1

            is_incorect_leaf = True

        obs = self._get_obs()

        return obs, reward, is_incorect_leaf, self._episode_info(), new_goal
        

    def get_current_node(self):
        #return the mapped index of the current node
        return self.node_mapping[self.current_node]

    def _perform_action(self, action):
        """
        Performs the given action (moving to a neighboring node) and updates the environment state.

        Args:
            action (int): The action to be performed.
        """
        neighbors = list(self.graph.neighbors(self.current_node))

        if action not in neighbors:
            print(f"Neighbors: {neighbors}")
            raise ValueError(f"Action {action} is not a neighbor of node {self.current_node}")
        else :
            self.current_node = action
        
        # Update the visited stack
        self.visited_stack.append(self.current_node)

        # Increment the visit count for the current node
        self._increment_visited(self.current_node)

        # Update the action space
        self._update_action_space()



    def _episode_info(self, found_target=False):
        """
        Constructs additional info about the current episode.

        Args:
            found_target (bool): Flag indicating whether the target was found.

        Returns:
            dict: Additional info about the episode.
        """
        nb_keys_found = len(self.visited_keys)
        return {
            'found_target': found_target,
            'curent_node': self.current_node,
        }

    def _increment_visited(self, node):
        """
        Increments the visit count for a given node.

        Args:
            node: The node whose visit count is to be incremented.
        """
        self.graph.nodes[node]['visited'] = self.graph.nodes[node].get('visited', 0) + 1

    def _reset_visited(self):
        """
        Resets the 'visited' status of all nodes in the graph.
        """
        for node in self.graph.nodes():
            self.graph.nodes[node]['visited'] = 0
        

    def calculate_number_visited_nodes(self):
        #itearte over all nodes in the graph and count the number of visited nodes
        count = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['visited'] > 0:
                count += 1
        return count
    


    def reset(self):
        """
        Resets the environment for a new episode.

        Returns:
            Data: Initial observation data after resetting.
        """
        #self.graph = self.reference_graph.copy()
        self.data = None
        self.current_target_iterator = 0
        self.current_node_iterator = 0
        self.current_node = self._sample_start_node()
        self.current_subtree_root = self.current_node
        self.visited_stack = [self.current_node]
        self._update_action_space()
        self.visited_keys = set()
        self.nb_actions_taken = 0
        self.observation_space = self._define_observation_space()
        self.target_node = self.target_nodes[self.current_target_iterator]

        return self._get_obs()
    


    def compute_centralities(self, graph):
        """
        Computes the centrality measures for all nodes in the graph.

        Returns:
            dict: A mapping from node to centrality measure.
        """
        return {
            node: {
                'degree_centrality': nx.degree_centrality(graph)[node]
            } for node in graph.nodes()
        }



    def get_visited_subgraph(self):
        """
        Gets the subgraph of visited nodes.
        returns a list of the mapped indices of the visited nodes
        

        """

        mapped_visited_nodes = [self.node_mapping[node] for node in self.visited_stack]
        mapped_visited_nodes = torch.Tensor(mapped_visited_nodes).long()
        return mapped_visited_nodes
        

    def _get_obs(self):

        if self.data is not None:
            return self.data


        # Use the node mapping to convert node indices
        edge_index = torch.tensor([(self.node_mapping[u], self.node_mapping[v]) for u, v in self.graph.edges()], dtype=torch.long).t().contiguous()

        #set the node "is_current" to 1 if it is the current node, 0 otherwise
        #set the number of keys found

        #give each node the index from the visited stack

        

        #fill other nodes with -1

        x = torch.tensor([[
            attributes['struct_size'],
            attributes['valid_pointer_count'],
            attributes['invalid_pointer_count'],
            attributes['first_pointer_offset'],
            attributes['last_pointer_offset'],
            attributes['first_valid_pointer_offset'],
            attributes['last_valid_pointer_offset'],
        ] for node, attributes in self.graph.nodes(data=True)], dtype=torch.float)

        #get a one hot encoding of the found target nodes, if the target node with value 0 in the map is in visited_keys, then it will be 100000
        #if the target node with value 1 AND 0 in the map is in visited_keys, then it will be 110000 etc...
        """
        one_hot_targets = torch.zeros(6, dtype=torch.float)
        for i, target in enumerate(self.target_nodes):
            if target in self.visited_keys:
                one_hot_targets[i] = 1
        """
        #add the one hot encoding to the node features
        #match the shape of the one hot encoding with the shape of the node features
        #one_hot_targets = one_hot_targets.repeat(x.shape[0], 1)
        #x = torch.cat((x, one_hot_targets), dim=1)


        edge_attr = torch.tensor([data['offset'] for u, v, data in self.graph.edges(data=True)], dtype=torch.float).unsqueeze(1)        # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise
        # Normalize edge attributes
        #edge_attr_np = edge_attr.numpy()
        #edge_attr_np = (edge_attr_np - np.mean(edge_attr_np, axis=0)) / np.std(edge_attr_np, axis=0)
        #edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
        
        # Convert back to tensor
        #x = torch.tensor(x_np, dtype=torch.float)
        # Check if the shape of x matches self.state_size

        if x.shape[1] != self.state_size:
            raise ValueError(f"The shape of x ({x.shape[1]}) does not match self.state_size ({self.state_size})")
        
        #transform = T.Compose([T.ToUndirected()])

        #reverse edges
        #edge_index = edge_index.flip(dims=[0])

        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #data = transform(data)
        return self.data

    def close(self):
        pass

    