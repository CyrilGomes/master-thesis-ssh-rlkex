import os
import joblib
import networkx as nx
import numpy as np
from tqdm import tqdm

class GraphPredictor:
    def __init__(self, model_path):

        self.model = joblib.load(model_path)

    @staticmethod
    def _extract_features(G, root_node):
        """ Extract features for a single node in the graph. """
        root_attributes = G.nodes[root_node]
        feature = [
            len(G),  # number of nodes
            G.degree(root_node),  # degree of root node
            nx.density(G),  # graph density
            root_attributes['struct_size'],
            root_attributes['valid_pointer_count'],
            root_attributes['invalid_pointer_count'],
            root_attributes['first_pointer_offset'],
            root_attributes['last_pointer_offset'],
            root_attributes['first_valid_pointer_offset'],
            root_attributes['last_valid_pointer_offset'],
        ]
        return np.array(feature, dtype=np.float32)

    @staticmethod
    def _get_root_nodes(G):
        """ Get the root nodes of a graph. """
        sccs = list(nx.strongly_connected_components(G))
        return GraphPredictor._get_root_nodes_from_scc(sccs, G)

    @staticmethod
    def _get_root_nodes_from_scc(sccs, G):
        """ Get root nodes from strongly connected components. """
        root_nodes = []
        for scc in sccs:
            for node in scc:
                if len([n for n in G.predecessors(node) if n not in scc]) == 0:
                    root_nodes.extend(list(scc))
                    break
        return root_nodes

    def predict_probabilities(self, G):
        """ Predict probabilities for a graph, returning a dict with node IDs and probabilities. """
        probabilities = {}
        root_nodes = self._get_root_nodes(G)
        for node in root_nodes:
            features = self._extract_features(G, node)
            #only keep nodes with probability > 0.5
            probability = self.model.predict_proba([features])[0][1]
            
            probabilities[node] = probability

        return probabilities
    
