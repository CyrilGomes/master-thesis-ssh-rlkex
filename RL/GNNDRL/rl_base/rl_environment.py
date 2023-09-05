import gym
from gym import spaces
import networkx as nx
import numpy as np
import torch

class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_node):
        super(GraphTraversalEnv, self).__init__()

        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

        self.graph = graph
        self.target_node = target_node
        self.current_node = self._sample_start_node()
        
        self.visited_stack = []

        self.state_size = len(self.graph.nodes[self.current_node])
        # Update action space for the current node
        self._update_action_space()
        
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.state_size,)),  # Assuming node features are 1-D
            'edge_index': spaces.Tuple((spaces.Discrete(len(self.graph.nodes())), spaces.Discrete(len(self.graph.nodes()))))
        })



    def _sample_start_node(self):

        #get all nodes that has neighbours
        nodes = [node for node in self.graph.nodes() if len(list(self.graph.neighbors(node))) > 0]
        return np.random.choice(nodes)

    def _update_action_space(self):
        # Number of neighbors + 1 for the "Backtrack" action
        num_actions = len(list(self.graph.neighbors(self.current_node))) + 1
        self.action_space = spaces.Discrete(num_actions)
        

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))
        
        # Check if the action is "Backtrack"
        if action == len(neighbors):
            if len(self.visited_stack) == 0:
                return self._get_obs(), -10, True, {}
            self.current_node = self.visited_stack.pop()
            self._update_action_space()
            return self._get_obs(), -1, False, {}
            # if self.current_node == self.target_node:
            #     return self._get_obs(), 100, True, {}
            # else:
            #     return self._get_obs(), -100, True, {}
        
        # If action is valid, move to the corresponding neighbor
        elif action < len(neighbors):
            self.current_node = neighbors[action]

            #check if it has neighbours, if no then stop
            if len(list(self.graph.neighbors(self.current_node))) == 0:
                return self._get_obs(), +100, True, {}


            # After moving to a new node, update the action space
            self.visited_stack.append(self.current_node)
            self._update_action_space()
            return self._get_obs(), -1, False, {}  
        
        # Invalid action
        else:
            return self._get_obs(), -10, False, {}  
        
    def reset(self):
        self.current_node = self._sample_start_node()
        self.visited_stack = []
        self._update_action_space()
        return self._get_obs()

        
    def _get_obs(self):
        # Current node and its successors
        nodes = [self.current_node] + list(self.graph.neighbors(self.current_node))

        # Extract node attributes. Assuming all nodes have the same attributes.
        node_attributes = [self.graph.nodes[node] for node in nodes]

        # Convert node attributes to tensor
        x = torch.tensor([list(attr.values()) for attr in node_attributes], dtype=torch.float)

        # Build edge index for the subgraph
        edge_list = [(self.current_node, neighbor) for neighbor in self.graph.neighbors(self.current_node)]
        edge_index = [[], []]

        for edge in edge_list:
            src = nodes.index(edge[0])
            dst = nodes.index(edge[1])
            edge_index[0].append(src)
            edge_index[1].append(dst)

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return {'x': x, 'edge_index': edge_index}

    def render(self, mode='human'):
        print(f"Current Node: {self.current_node}")
        
    def close(self):
        pass
