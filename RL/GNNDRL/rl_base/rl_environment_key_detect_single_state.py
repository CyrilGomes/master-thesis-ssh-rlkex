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

from rl_base.root_heuristic_rf import GraphPredictor
from .gnn import GNN
@jit(nopython=True)
def compute_reward(previous_dist, current_dist, has_found_target, visit_count, 
                   TARGET_FOUND_REWARD, STEP_PENALTY, REVISIT_PENALTY, 
                   PROXIMITY_BONUS, NEW_NODE_BONUS, NO_PATH_PENALTY, 
                   ADDITIONAL_TARGET_MULTIPLIER, visited_keys_count, 
                   max_episode_steps, nb_actions_taken, true_keys_count):
    
    if nb_actions_taken > max_episode_steps:
        # Logic for final reward when max episode steps are reached
        return -1 * nb_actions_taken 
    if has_found_target:
        target_reward =  TARGET_FOUND_REWARD * (ADDITIONAL_TARGET_MULTIPLIER ** true_keys_count) 
        return target_reward
    if current_dist is None:
        return NO_PATH_PENALTY
    distance_reward = PROXIMITY_BONUS * (previous_dist - current_dist) if current_dist < previous_dist else 0
    revisit_penalty = REVISIT_PENALTY * visit_count if visit_count > 1 else 0
    new_node_bonus = NEW_NODE_BONUS if visit_count == 0 else 0
    
    total_reward = distance_reward + STEP_PENALTY + revisit_penalty + new_node_bonus
    #print(f"Distance reward: {distance_reward}, Step penalty: {STEP_PENALTY}, Revisit penalty: {revisit_penalty}, New node bonus: {new_node_bonus}")
    return total_reward

class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_nodes, root_detection_model_path="models/root_heuristic_model.joblib", max_episode_steps=20, obs_is_full_graph=False):
        """
        Initializes the Graph Traversal Environment.

        Args:
            graph (nx.Graph): A NetworkX graph.
            target_nodes (list): A list of target nodes within the graph.
            subgraph_detection_model_path (str): Path to the trained subgraph detection model.
            max_episode_steps (int): Maximum steps allowed per episode.
        """
        super(GraphTraversalEnv, self).__init__()

        self._validate_graph(graph)
        self.main_graph = graph
        self.target_nodes = target_nodes
        self.max_episode_steps = max_episode_steps
        self.shortest_path_cache = {}
        self.visited_stack = []
        self.current_node_iterator = 0
        self.obs_is_full_graph = obs_is_full_graph

        self._init_rewards_and_penalties()
        #load model
        self.root_detection_model_path = root_detection_model_path
        self.root_detector = GraphPredictor(self.root_detection_model_path)
        print("Model loaded!")
        #get proba for all root nodes, returns a map of root node -> proba
        self.root_proba = self.root_detector.predict_probabilities(self.main_graph)
        #sort every roots based on proba, ommit the ones with 0 proba
        self.sorted_roots = sorted(self.root_proba, key=self.root_proba.get, reverse=True)
        #from the current_node_iterator, get the corresponding root
        self.current_root = self.sorted_roots[self.current_node_iterator]
        #update the graph to be the subgraph of the root (using BFS)
        self.reference_graph = self.main_graph.subgraph(nx.bfs_tree(self.main_graph, self.current_root))
        #create a copy of the reference graph
        self.graph = self.reference_graph.copy()

        self.state_size = 9 if self.obs_is_full_graph else 11
        self._update_action_space()

        self.observation_space = self._define_observation_space()
        self.visited_keys = {}
        self.nb_targets = len(self.target_nodes)
        self.nb_actions_taken = 0

    def _validate_graph(self, graph):
        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

    def _init_rewards_and_penalties(self):
        self.TARGET_FOUND_REWARD = 100
        self.STEP_PENALTY = -2
        self.REVISIT_PENALTY = -0.5
        self.PROXIMITY_BONUS = 20
        self.NEW_NODE_BONUS = 3
        self.NO_PATH_PENALTY = -500
        self.ADDITIONAL_TARGET_MULTIPLIER = 1.5


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
    
    def update_target(self):
        """
        Updates the target node to the closest target node.
        """
        self.target_node = self._get_closest_target()


    def _graph_to_data(self, graph):
        """
        Converts a graph to data usable by the model.

        Args:
            graph (nx.Graph): A NetworkX graph.

        Returns:
            Data: PyTorch Geometric data object.
        """
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

        edge_attr = torch.tensor([data['offset'] for u, v, data in graph.edges(data=True)], dtype=torch.float).unsqueeze(1)        # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise
        y = torch.tensor([1 if any(attributes['cat'] == 1 for _, attributes in graph.nodes(data=True)) else 0], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(self.device)


    def _get_dist_to_target(self):
        """
        Calculates the distance from the current node to the target node.

        Returns:
            int or None: Distance to the target node or None if no path exists.
        """
        # If no path exists, return None.
        try:
            path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            return len(path) - 1
        except nx.NetworkXNoPath:
            return None



    def _sample_start_node(self):
        """
        Samples a start node for the agent.

        Returns:
            Node: A node from sorted promising roots to start the episode.
        """
        return self.sorted_roots[self.current_node_iterator]

    def _update_action_space(self):
        """
        Updates the action space based on the current node.
        """
        num_actions = 50  # placeholder, adjust as needed
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

    def _restart_agent_from_prev(self):
        """
        Resets the agent's position to the start node.
        """
        if len (self.visited_stack) < 2:
            self._restart_agent_from_root()
            return
        prev = self.current_node
        self.current_node = self.visited_stack[-2]
        self.visited_stack.append(self.current_node)
        self._increment_visited(self.current_node)
        self._update_action_space()

    def _get_valid_actions(self):
        """
        Determines the valid actions that can be taken from the current node.

        Returns:
            list: A list of valid action indices.
        """

        neighbors = list(self.graph.neighbors(self.current_node))[:self.action_space.n]
        valid_actions = [i for i, _ in enumerate(neighbors)]
        valid_actions.append(self.action_space.n - 1)  # Adding the stop action
        #remove the stop action if exist
        """
        if self.action_space.n - 1 in valid_actions:
            valid_actions.remove(self.action_space.n - 1)
        """
        #if there are acion over the action space, remove them

        return valid_actions


    def _get_action_mask(self):
        """
        Creates a mask for valid actions in the action space.

        Returns:
            np.array: An array representing the action mask.
        """
        valid_actions = self._get_valid_actions()
        action_mask = np.full(self.action_space.n, -np.inf)
        action_mask[valid_actions] = 0

        #action_mask[valid_actions] = 0
        return action_mask
    
    def _get_true_keys_count(self):
        """
        Get the actual number of keys in self.visited_keys
        Args:
            None
        Returns:
            int: The actual number of keys in self.visited_keys
        """
        count = 0
        #check if there are duplicates in self.visited_keys
        if len(self.visited_keys) != len(set(self.visited_keys)):
            raise ValueError("There are duplicates in self.visited_keys")

        for key in self.visited_keys:
            if key in self.target_nodes:
                count += 1
        return count


    def step(self, action):
        """
        Executes a step in the environment given an action.

        Args:
            action (int): The action to be taken.

        Returns:
            tuple: Observation, reward, done flag, and additional info.
        """


        # Check if the action is to stop searching for more keys
        if action == self.action_space.n - 1:
            
            nb_true_keys = self._get_true_keys_count()
            # Special action for stopping the search
            reward = 0

            if self.nb_targets == nb_true_keys:
                reward = 200
                done = True
                found_target = True
                reward -=  self.nb_actions_taken

            else:
                reward = -3 * (len(self.visited_keys)- nb_true_keys) - 10*(self.nb_targets - nb_true_keys)
                done = True
                found_target = False
            return self._get_obs(), reward, done, {'found_target': found_target, 
                                                   'nb_keys_found': self._get_true_keys_count(),
                                                   'nb_possible_keys' : len(self.visited_keys)}

        # Update the target node based on the current node
        self.update_target()

        self.nb_actions_taken += 1

        if self.target_node is not None:
            # Calculate distance to the target node before performing the action
            previous_dist = self._get_dist_to_target()

        # Regular action: moving to another node
        self._perform_action(action)

        # Get the number of times the current node has been visited
        visit_count = self.graph.nodes[self.current_node]['visited']

        reward = -200
        current_dist = None
        if self.target_node is not None :

            # Calculate distance to the target node after performing the action
            current_dist = self._get_dist_to_target() if self.current_node != self.target_node else 0



            # Check if the current node is the target node
            has_found_target = self.current_node == self.target_node


            # Compute the reward based on the state before and after the action
            reward = compute_reward(previous_dist, current_dist, has_found_target, visit_count,
                                    self.TARGET_FOUND_REWARD, self.STEP_PENALTY, self.REVISIT_PENALTY,
                                    self.PROXIMITY_BONUS, self.NEW_NODE_BONUS, self.NO_PATH_PENALTY,
                                    self.ADDITIONAL_TARGET_MULTIPLIER, len(self.visited_keys),
                                    self.max_episode_steps, self.nb_actions_taken, self._get_true_keys_count())
            if has_found_target:
                self.nb_actions_taken = int(self.nb_actions_taken  *0.70)

        # Check if the episode is over
        done = self.nb_actions_taken > self.max_episode_steps or current_dist is None


        #check if leaf node and if so, restart from root
        if self.graph.out_degree(self.current_node) == 0:
            #remove the edge from the previous node to the current node
            #print the number of edges before removing
            if len (self.visited_stack) >= 2:
                self.graph.remove_edge(self.visited_stack[-2], self.current_node)


        

            self.visited_keys.add(self.current_node)
            self._restart_agent_from_root()


        return self._get_obs(), reward, done, self._episode_info()

    
    def _is_episode_over(self):
        """
        Checks if the episode has reached its end conditions.

        Returns:
            bool: True if episode is over, False otherwise.
        """
        return self.nb_actions_taken > self.max_episode_steps

    def _finalize_episode(self):
        """
        Finalizes the episode, providing the final observation and other details.

        Returns:
            tuple: Final observation, reward, done flag, and additional info.
        """
        reward = self._final_reward()
        return self._get_obs(), reward, True, self._episode_info(found_target=len(self.visited_keys) == self.nb_targets)
    

    def _perform_action(self, action):
        """
        Performs the given action (moving to a neighboring node) and updates the environment state.

        Args:
            action (int): The action to be performed.
        """
        neighbors = list(self.graph.neighbors(self.current_node))

        if action < len(neighbors):
            # Move to the selected neighboring node
            self.current_node = neighbors[action]
        else:
            raise ValueError(f"Invalid action: {action}")
        
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
        return {
            'found_target': found_target,
            'max_episode_steps_reached': self._is_episode_over(),
            'nb_keys_found': self._get_true_keys_count(),
            'nb_possible_keys' : len(self.visited_keys)
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
        self.update_target()
        self.current_subtree_root = self.current_node
        self.visited_stack = []
        self._update_action_space()
        self._reset_visited()
        self.visited_keys = {self.current_node}
        self.nb_actions_taken = 0
        self.observation_space = self._define_observation_space()


        return self._get_obs()
    


    def _get_graph_obs(self):
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
        # Get a mapping from old node indices to new ones
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}

        # Use the node mapping to convert node indices
        edge_index = torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in self.graph.edges()], dtype=torch.long).t().contiguous()

        #set the node "is_current" to 1 if it is the current node, 0 otherwise
        #set the number of keys found
        x = torch.tensor([[
            attributes['struct_size'],
            attributes['valid_pointer_count'],
            attributes['invalid_pointer_count'],
            attributes['first_pointer_offset'],
            attributes['last_pointer_offset'],
            attributes['first_valid_pointer_offset'],
            attributes['last_valid_pointer_offset'],
            1 if node == self.current_node else 0,
            len(self.visited_keys),
        ] for node, attributes in self.graph.nodes(data=True)], dtype=torch.float)

        edge_attr = torch.tensor([data['offset'] for u, v, data in self.graph.edges(data=True)], dtype=torch.float).unsqueeze(1)        # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise

        # Check if the shape of x matches self.state_size
        if x.shape[1] != self.state_size:
            raise ValueError(f"The shape of x ({x.shape[1]}) does not match self.state_size ({self.state_size})")

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _get_current_node_obs(self):
        # Current node and its successors
        nodes = [self.current_node] + list(self.graph.neighbors(self.current_node))

        #print the offset for each current_node -> neighbor edge
        
        for neighbor in self.graph.neighbors(self.current_node):
            #print the data of the edge
            edge_data = self.graph.get_edge_data(self.current_node, neighbor)

            edge_data = self.graph.get_edge_data(self.current_node, neighbor)
            if edge_data is not None:
                # Iterate over all items in the edge data dictionary
                for edge_id, attrs in edge_data.items():
                    if 'offset' in attrs:
                        offset = attrs['offset']


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
                                max((edge_attr['offset'] for edge_id, edge_attr in self.graph[self.current_node].get(node, {}).items() if 'offset' in edge_attr), default=0),                                self._get_true_keys_count()
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

    def _get_obs(self):
        """
        Gets the current observation.

        Returns:
            Data: Current observation data.
        """
        if self.obs_is_full_graph:
            return self._get_graph_obs()
        else:
            return self._get_current_node_obs()
        

    def _print_step_debug(self, neighbors, action, printing=False):
        """
        Prints debugging information for each step, if enabled.

        Args:
            neighbors (list): List of neighbors of the current node.
            action (int): The action taken.
            printing (bool): Flag to enable/disable printing.
        """
        if not printing:
            return

        if self.target_node not in neighbors:
            best_path = nx.shortest_path(self.graph, self.current_node, self.target_node)
            best_neighbor = best_path[1]
            print(f"Current node: {self.current_node}, Distance to target: {len(best_path)}, Neighbors: {neighbors}, Target: {self.target_node}, Action: {action}, Best action: {best_neighbor}")
        else:
            print(f"Current node: {self.current_node}, Neighbors: {neighbors}, Target: {self.target_node}, Action: {action}, Best action: {self.target_node}")

    def _build_edge_index(self, nodes):
        """
        Builds an edge index tensor for the given nodes.

        Args:
            nodes (list): List of nodes.

        Returns:
            torch.Tensor: Edge index tensor.
        """
        edge_index = []
        for neighbor in self.graph.neighbors(self.current_node):
            edge_index.append([nodes.index(self.current_node), nodes.index(neighbor)])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

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

    