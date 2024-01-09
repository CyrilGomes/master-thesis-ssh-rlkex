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
from torch_geometric.utils import to_undirected
from root_heuristic_rf import GraphPredictor
from torch_geometric.transforms import Compose, ToUndirected, AddSelfLoops, NormalizeFeatures
from torch_geometric.utils import add_self_loops

class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_nodes,num_actions, root_predictor = None):
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
        self.action_space = spaces.Discrete(num_actions)
        self._validate_graph(graph)
        self.main_graph = graph


        self.target_nodes = target_nodes
        self.shortest_path_cache = {}
        self.visited_stack = []
        self.current_node_iterator = 0

        #load model
        self.root_detector = root_predictor
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

        self.state_size = 13 

        self.observation_space = self._define_observation_space()
        self.visited_keys = {}
        self.nb_targets = len(self.target_nodes)
        self.nb_actions_taken = 0
        self.reset()




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
                raise ValueError(f"There is no path from root {best_root} to target {target}")

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


    def _sample_start_node(self):
        """
        Samples a start node for the agent.

        Returns:
            Node: A node from sorted promising roots to start the episode.
        """

        #only keep the roots with proba > 0.5
        return self.best_root


        
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

    def _get_valid_actions(self):
        """
        Determines the valid actions that can be taken from the current node.

        Returns:
            list: A list of valid action indices in PyTorch Geometric format.
        """
        neighbors = list(self.graph.successors(self.current_node))[:self.action_space.n]
        #For sanity check that the node mapping is correct
        
        #valid actions are the index of the neighbors
        #for example, if we have a space of 50 actions, but if the current node has only 3 neighbors, then the valid actions are [0,1,2]
        valid_actions = [i for i, _ in enumerate(neighbors)]

        return valid_actions







    def _get_action_mask(self):
        """
        Creates a mask for valid actions in the action space.

        Returns:
            np.array: An array representing the action mask.
        """
        valid_actions = self._get_valid_actions()

        action_mask = np.full(self.action_space.n, 0)
        action_mask[valid_actions] = 1

        return action_mask
    




    def step(self, action):
        self._perform_action(action)

        self.nb_actions_taken += 1

        #check if the current_node has neighbors
        has_neighbors = self.graph.out_degree(self.current_node) > 0
        if not has_neighbors:
            if self.current_node in self.visited_keys:
                return self._get_obs(), 0, True, self._episode_info()
            self.visited_keys.add(self.current_node)
            if len(self.visited_keys) == self.nb_targets:
                return self._get_obs(), 0, True, self._episode_info()
            self._restart_agent_from_root()


        obs = self._get_obs()
        return obs, 0, False, self._episode_info()



    def _perform_action(self, action):
        """
        Performs the given action (moving to a neighboring node) and updates the environment state.

        Args:
            action (int): The action to be performed.
        """
        neighbors = list(self.graph.neighbors(self.current_node))

        if action >= len(neighbors):
            raise ValueError(f"Action {action} is out of range for neighbors {neighbors}")
        
        self.current_node = neighbors[action]
        # Update the visited stack
        self.visited_stack.append(self.current_node)

        # Increment the visit count for the current node
        self._increment_visited(self.current_node)




    def _episode_info(self):
        """
        Constructs additional info about the current episode.

        Args:
            found_target (bool): Flag indicating whether the target was found.

        Returns:
            dict: Additional info about the episode.
        """
        #count the number of target nodes found in the visited keys
        nb_keys_found = 0
        for key in self.visited_keys:
            if key in self.target_nodes:
                nb_keys_found += 1
        return {
            'nb_keys_found': nb_keys_found,
            'nb_actions_taken': self.nb_actions_taken,
            'nb_nodes_visited': self.calculate_number_visited_nodes(),
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
        self.graph = self.reference_graph.copy()

        self.current_node_iterator = 0
        self.current_node = self._sample_start_node()
        self.current_subtree_root = self.current_node
        self.visited_stack = []
        self._reset_visited()
        self.visited_keys = set()
        self.nb_actions_taken = 0
        self.observation_space = self._define_observation_space()
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




    def _get_obs(self):
        """
        convvert self.graph to data, only keep
        data['struct_size'],
        data['valid_pointer_count'],
        data['invalid_pointer_count'],
        data['visited'],
        data['first_pointer_offset'],
        data['last_pointer_offset'],
        data['first_valid_pointer_offset'],
        data['last_valid_pointer_offset'],
        """


        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}

        data_space_current_node_idx = node_mapping[self.current_node]
        # Use the node mapping to convert node indices

        edge_index = torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in self.graph.edges()], dtype=torch.long).t().contiguous()

        #set the node "is_current" to 1 if it is the current node, 0 otherwise
        #set the number of keys found

        #give each node the index from the visited stack

        found_keys = len(self.visited_keys) / self.nb_targets
        num_nodes_in_graph = len(self.graph.nodes)

        map_neighbour_to_index = {neighbour: i for i, neighbour in enumerate(self.graph.successors(self.current_node))}
        #fill other nodes with -1
        #for each node check if it is a visited key, for example if the node is the first visited key, then give it a value of 1, if it is the second visited key, then give it a value of 2, etc...

        #create a mapping between node id and the index in the visited keys
        map_visited_key_to_index = {key: i for i, key in enumerate(self.visited_keys)}
        
        nb_nodes_visited = len(self.visited_keys)

        x = torch.tensor([[
            attributes['struct_size'],
            attributes['valid_pointer_count'],
            attributes['invalid_pointer_count'],
            attributes['first_pointer_offset'],
            attributes['last_pointer_offset'],
            attributes['first_valid_pointer_offset'],
            attributes['last_valid_pointer_offset'],
            attributes['visited'] / nb_nodes_visited if nb_nodes_visited > 0 else 0,
            self.nb_targets,
            found_keys,
            self.graph.out_degree(node),
            node == self.best_root,
            num_nodes_in_graph        
        ] for node, attributes in self.graph.nodes(data=True)], dtype=torch.float)

        
        edge_attr = torch.tensor([data['offset'] for u, v, data in self.graph.edges(data=True)], dtype=torch.float).unsqueeze(1)        # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise
        
        """

        # Normalize edge attributes
        edge_attr_np = edge_attr.numpy()
        edge_attr_np = (edge_attr_np - np.mean(edge_attr_np, axis=0)) / np.std(edge_attr_np, axis=0)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
        """
        """
        # Standardize features (subtract mean, divide by standard deviation), ignore the last two features
        x_np = x.numpy()
        eps = 1e-8
        x_np[:, :-2] = (x_np[:, :-2] - np.mean(x_np[:, :-2] , axis=0)) / (np.std(x_np[:, :-2] , axis=0) + eps)
        """

        # Convert back to tensor
        #x = torch.tensor(x_np, dtype=torch.float)
        # Check if the shape of x matches self.state_size



        transform = T.Compose([
            NormalizeFeatures(),    # Normalize node features
        ])


        #reverse the direction of the edges
        edge_index = edge_index[[1,0],:]
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0, num_nodes=x.shape[0])


        

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, current_node_id = data_space_current_node_idx)
        data = transform(data)

        if data.x.shape[1] != self.state_size:
            raise ValueError(f"The shape of x ({x.shape[1]}) does not match self.state_size ({self.state_size})")
        
        return data

    def close(self):
        pass

    