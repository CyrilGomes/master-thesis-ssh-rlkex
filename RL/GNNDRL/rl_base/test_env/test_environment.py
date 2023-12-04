import gym
from gym import spaces
import networkx as nx
import numpy as np
import torch
import random 
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv
import concurrent.futures
from numba import jit
from gnn import GNN



class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, subgraph_detection_model_path="models/model.pt", max_episode_steps=30):
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
        self.graph = graph
        self.max_episode_steps = max_episode_steps

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.subgraph_detection_model_path = subgraph_detection_model_path
        self.subgraph_detection_model = self._load_model()
        print("Model loaded!")

        self.visited_stack = []
        self.promising_nodes = self._get_most_promising_subgraph()
        self.graph = self.promising_nodes
        self.sorted_promising_nodes = self._sort_promising_nodes()
        self.current_node_iterator = 0
        self.current_node = self._sample_start_node()
        self.current_subtree_root = self.current_node

        self.state_size = 11
        self._update_action_space()

        self.observation_space = self._define_observation_space()
        self.visited_keys = []
        self.nb_actions_taken = 0

    def _validate_graph(self, graph):
        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

    def _load_model(self):
        """
        Loads the subgraph detection model.
        """
        model = GNN()
        model.load_state_dict(torch.load(self.subgraph_detection_model_path))
        model.eval()  # Set the network to evaluation mode
        model = model.to(self.device)
        return model
    
    def _sort_promising_nodes(self):
        """
        Sorts promising nodes based on their out-degree in descending order.

        Returns:
            list: Sorted list of promising nodes.
        """
        nodes = [node for node in self.promising_nodes.nodes() if self.promising_nodes.in_degree(node) == 0]
        children_count = [self.promising_nodes.out_degree(node) for node in nodes]
        sorted_indices = np.argsort(children_count)[::-1]
        return [nodes[i] for i in sorted_indices]

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

        edge_attr = torch.tensor([graph[u][v]['offset'] for u, v in graph.edges], dtype=torch.float).unsqueeze(1)
        # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise
        y = torch.tensor([1 if any(attributes['cat'] == 1 for _, attributes in graph.nodes(data=True)) else 0], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(self.device)
    
    def _create_subgraph_data(self):
        """
        Creates data for each subgraph in the graph.

        Returns:
            tuple: A tuple containing subgraphs and their corresponding data.
        """
        subgraphs = [self.graph.subgraph(c) for c in nx.connected_components(self.graph.to_undirected())]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            subgraphs_data = list(executor.map(self._graph_to_data, subgraphs))
        return subgraphs, subgraphs_data




    def _get_most_promising_subgraph(self):
        """
        Finds the most promising subgraph in the graph.
        """
        #iterate over all subgraphs to get scores
        subgraphs, subgraphs_data = self._create_subgraph_data()
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
    


    def _sample_start_node(self):
        """
        Samples a start node for the agent.

        Returns:
            Node: A node from sorted promising nodes to start the episode.
        """
        return self.sorted_promising_nodes[self.current_node_iterator]

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


    def _get_valid_actions(self):
        """
        Determines the valid actions that can be taken from the current node.

        Returns:
            list: A list of valid action indices.
        """
        neighbors = list(self.graph.neighbors(self.current_node))
        valid_actions = [i for i, _ in enumerate(neighbors)]
        valid_actions.append(self.action_space.n - 1)  # Adding the skip action
        return valid_actions


    def _get_action_mask(self):
        """
        Creates a mask for valid actions in the action space.

        Returns:
            np.array: An array representing the action mask.
        """
        valid_actions = self._get_valid_actions()
        action_mask = np.full(self.action_space.n, -1e6)
        action_mask[valid_actions] = 0
        return action_mask
    


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
            
            return self._get_obs(), True, self._episode_info()


        self.nb_actions_taken += 1


        # Regular action: moving to another node
        self._perform_action(action)

        # Check if the episode is over
        done = self.nb_actions_taken > self.max_episode_steps

        #check if leaf node and if so, restart from root
        if self.graph.out_degree(self.current_node) == 0:
            self.visited_keys.append(self.current_node)
            self._restart_agent_from_root()

        
        return self._get_obs(), done, self._episode_info()

    
    def _is_episode_over(self):
        """
        Checks if the episode has reached its end conditions.

        Returns:
            bool: True if episode is over, False otherwise.
        """
        return self.nb_actions_taken > self.max_episode_steps

    
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



    def _episode_info(self):
        """
        Constructs additional info about the current episode.

        Args:
            None
        Returns:
            dict: Additional info about the episode.
        """
        return {
            'max_episode_steps_reached': self._is_episode_over(),
            'nb_keys_found': len(self.visited_keys),
            'found_keys': self.visited_keys
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
        
    def reset(self):
        """
        Resets the environment for a new episode.

        Returns:
            Data: Initial observation data after resetting.
        """
        self.current_node_iterator = 0
        self.current_node = self._sample_start_node()
        self.current_subtree_root = self.current_node
        self.visited_stack = []
        self._update_action_space()
        self._reset_visited()
        self.visited_keys = []
        self.nb_actions_taken = 0
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

    