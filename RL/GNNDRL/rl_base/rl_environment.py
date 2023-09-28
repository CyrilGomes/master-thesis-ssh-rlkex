import gym
from gym import spaces
import networkx as nx
import numpy as np
import torch
import random 
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


        self.state_size = len(self.graph.nodes[self.current_node])
        # Update action space for the current node
        self._update_action_space()
        
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.state_size,)),  # Assuming node features are 1-D
            'edge_index': spaces.Tuple((spaces.Discrete(len(self.graph.nodes())), spaces.Discrete(len(self.graph.nodes()))))
        })

        self.prev_dist = 8
        self.curr_dist = self._get_dist_to_target()



        

    def _get_dist_to_target(self):
        G_undi = self.graph.to_undirected()
        is_target_reachable = nx.has_path(self.graph, self.current_node, self.target_node)
        return nx.shortest_path_length(G_undi, self.current_node, self.target_node), is_target_reachable



    def _sample_start_node(self):

        #get all nodes that has neighbours
        #nodes = [node for node in self.graph.nodes() if len(list(self.graph.neighbors(node))) > 0]
        best_path = nx.shortest_path(self.graph, 0, self.target_node)
        #similarly to simulated annealing, based on the current episode index, we less chance to choose a node from the best path
        
        exploration_prob = np.exp(-self.episode_index/3800)
        if random.random() < exploration_prob:
            return random.choice(list(best_path[1:-2]))

        return 0
    def _update_action_space(self):
        # Number of neighbors
        #TODO : Add + 1 for Backtrack
        
        num_actions = len(list(self.graph.neighbors(self.current_node)))
        self.action_space = spaces.Discrete(num_actions)
        


    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))

        # Check if the action is "Backtrack" (never entered)
        if action == len(neighbors):
            print("Not supposed for now")
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
            self.increment_visited(self.current_node)


            #check if the current node is the target node
            if self.current_node == self.target_node:
                #print("Found the target node !!")
                return self._get_obs(), +100, True, {'found_target': True}



            #check if it has neighbours, if no then stop
            if len(list(self.graph.neighbors(self.current_node))) == 0:
                return self._get_obs(), 0.3, True, {'found_target': False}


            #check if the target is reachable
            self.prev_dist = self.curr_dist
            self.curr_dist, is_reachable = self._get_dist_to_target()
            if not is_reachable:
                return self._get_obs(), -10, True,{'found_target': False}

            #check if the distance has reduced

            dist_theta = self.curr_dist - self.prev_dist
            alpha = 2
            beta = 0.3
            self.visited_stack.append(self.current_node)

            self._update_action_space()
            nb_visited = self.graph.nodes[self.current_node]['visited']
            reward = (dist_theta*alpha) - beta*nb_visited
            return self._get_obs(), reward, False, {}



        
        # Invalid action
        else:
            return self._get_obs(), -10, False, {}  
        

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
        self.prev_dist = 8
        self.curr_dist, _ = self._get_dist_to_target()
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
