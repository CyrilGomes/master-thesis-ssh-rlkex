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



class GraphTraversalEnv(gym.Env):
    def __init__(self, graph, target_nodes, level = 0, subgraph_detection_model_path = "models/model.pt", max_episode_steps=50):
        super(GraphTraversalEnv, self).__init__()

        if not isinstance(graph, nx.Graph):
            raise ValueError("Graph should be a NetworkX graph.")

        self.level = level
        self.graph = graph
        self.target_nodes = target_nodes
        self.episode_index = 0
        self.max_episode_steps = max_episode_steps

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.subgraph_detection_model_path = subgraph_detection_model_path
        self.load_model()
        print("Model loaded !")
        self.visited_stack = []

        self.promising_nodes = self._get_most_promising_subgraph()


        self.sorted_promising_nodes = self.sort_promosing_nodes()
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
        # Get the closest reachable target node from the current node
        closest_target = None
        closest_target_dist = float('inf')
        for target in [t for t in self.target_nodes if t not in self.visited_keys]:
            try:
                path = nx.shortest_path(self.promising_nodes, self.current_node, target)
                if len(path) < closest_target_dist:
                    closest_target_dist = len(path)
                    closest_target = target
            except nx.NetworkXNoPath:

                continue


        return closest_target

    def update_target(self):
        #update the target node to the closest target node
        self.target_node = self.get_closest_target()


    def load_model(self):
        self.subgraph_detection_model = torch.load(self.subgraph_detection_model_path)

    def sort_promosing_nodes(self):
        #return a sorted list of promising nodes based on the number of children
        promising_nodes = [node for node in self.promising_nodes.nodes() if len(list(self.promising_nodes.predecessors(node))) == 0]
        children_count = [len(list(self.promising_nodes.successors(node))) for node in promising_nodes]
        sorted_nodes = [node for _, node in sorted(zip(children_count, promising_nodes), reverse=True)]
        return sorted_nodes
    

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

        #get all subgraphs
        subgraphs = [self.graph.subgraph(c) for c in nx.connected_components(self.graph.to_undirected())]
        subgraphs_data = [self.graph_to_data(g) for g in subgraphs]
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
                print(score)
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
    
    def get_start_from_level(self):
        #get a random parent node of node 0, if level is 0 then select from direct parents, if level is 1 then select from grand parents and so on
        node_parents = list(self.graph.predecessors(0))
        if self.level == 0:
            print(len(node_parents))
            random_parent = random.choice(node_parents)
            print(random_parent)
            return random.choice(node_parents)
        else :
            for i in range(self.level):
                node_parents = [node for node in node_parents if len(list(self.graph.predecessors(node))) > 0]
                node_parents = [random.choice(list(self.graph.predecessors(node))) for node in node_parents]
            return random.choice(node_parents)
        
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
        
        num_actions = len(list(self.graph.neighbors(self.current_node)))
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


    def compute_reward(self, previous_dist, current_dist, has_found_target, visit_count):
        TARGET_FOUND_REWARD = 50
        STEP_PENALTY = -1
        REVISIT_PENALTY = -5
        PROXIMITY_BONUS = 6
        NEW_NODE_BONUS = 10
        NO_PATH_PENALTY = -10
        ADDITIONAL_TARGET_MULTIPLIER = 1.5  # Multiplier for each additional target found

        # Reward for finding a target
        if has_found_target:
            target_reward = TARGET_FOUND_REWARD * (ADDITIONAL_TARGET_MULTIPLIER ** len(self.visited_keys))
            return target_reward

        # Penalty for no path to the target
        if current_dist is None:
            return NO_PATH_PENALTY

        # Reward for moving closer to the target
        distance_reward = PROXIMITY_BONUS if current_dist < previous_dist else 0

        # Penalty for revisiting nodes (scaled by visit count)
        revisit_penalty = REVISIT_PENALTY * visit_count if visit_count > 0 else 0

        # Bonus for visiting a new node
        new_node_bonus = NEW_NODE_BONUS if visit_count == 0 else 0

        # Aggregate and return the total reward
        total_reward = distance_reward + STEP_PENALTY + revisit_penalty + new_node_bonus
        return total_reward


    def restart_agent_from_root(self):
        self.current_node = self._sample_start_node()
        self.visited_stack.append(self.current_node)
        self.increment_visited(self.current_node)
        self._update_action_space()

    def step(self, action, printing=False):

        if self.nb_actions_taken > self.max_episode_steps:
            return self._get_obs(), -1, True, {'found_target': False,
                                               'max_episode_steps_reached': True,
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

        if action == self.action_space.n : #skip to the next subtree
            self.skip_to_next_root()
            self.restart_agent_from_root()


            return self._get_obs(), 0, False, {'found_target': False,
                                               'nb_keys_found': len(self.visited_keys)}
    

        if action >= len(neighbors) or not is_target_reachable:
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
        reward = self.compute_reward(previous_dist, current_dist, has_found_target, visit_count)        
        

        if has_found_target:
            self.visited_keys.append(self.current_node)
            is_done = len(self.visited_keys) == self.nb_targets

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
        return self._get_obs()

    def _get_obs(self):
        # Current node and its successors
        nodes = [self.current_node] + list(self.graph.neighbors(self.current_node))

        # Extract node attributes. Assuming all nodes have the same attributes. remove the 'cat' feature
        
        subgraph = self.graph.subgraph(nodes)

        #Sinusoidal positional encoding
        #pos_enc = torch.zeros((len(nodes), 2))
        #pos_enc[:, 0] = torch.sin(torch.arange(0, len(nodes), 1) * (2 * np.pi / len(nodes)))
        #pos_enc[:, 1] = torch.cos(torch.arange(0, len(nodes), 1) * (2 * np.pi / len(nodes)))
         

        

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
            subgraph[self.current_node][node]['offset'] if node in subgraph[self.current_node] else 0,
            len(self.visited_keys)
            
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
