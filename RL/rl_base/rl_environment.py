from enum import Enum
import random
import networkx as nx
class Actions(Enum):
    BACKTRACK = 0
    DIVE = 1
    STAY = 2


class State:
    def __init__(self, in_degree, out_degree, depth, visit_count, cat, struct_size, invalid_pointer_count, valid_pointer_count, pointer_count):
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.depth = depth
        self.visit_count = visit_count
        self.cat = cat
        self.struct_size = struct_size
        self.invalid_pointer_count = invalid_pointer_count
        self.valid_pointer_count = valid_pointer_count
        self.pointer_count = pointer_count

    def update(curr_node):
        #get the current node (networkx node)
        #update the state
        pass
class MyEnvironment:
    def __init__(self, graph_file):
        self.graph = nx.read_graphml(graph_file)
        self.state = State(0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.reward = None
        self.done = None
        self.info = None
        self.current_node = "root"

    def reset(self):
        # Reset the environment to its initial state
        self.state = None
        self.reward = None
        self.done = False
        self.info = {}
        self.current_node = "root"

        return self.state

    def step(self, action):
        # Update the environment based on the given action
        if action == Actions.BACKTRACK:
            parent_nodes = list(self.graph.predecessors(self.current_node))
            if len(parent_nodes) > 0:
                self.current_node = parent_nodes[0]
        elif action == Actions.DIVE:
            child_nodes = list(self.graph.successors(self.current_node))
            if len(child_nodes) > 0:
                child_index = int(action[1] * len(child_nodes))
                self.current_node = child_nodes[child_index]
        elif action == Actions.STAY:
            pass

        self.state.update(self.current_node)
        self.reward = None
        self.done = False
        self.info = {}

        return self.state, self.reward, self.done, self.info
