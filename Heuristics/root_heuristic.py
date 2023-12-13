# -------------------------
# IMPORTS AND SETUP
# -------------------------

import os
import random
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import joblib
from collections import deque
import numpy as np
import random

#import range tqdm
from tqdm import tqdm
from tqdm import trange

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# -------------------------
# GRAPH PROCESSING
# -------------------------

def get_root_nodes_from_scc(sccs, G):
    """If we consider each scc as a node, then we want to get the sccs that have no incoming edges."""
    root_nodes = []
    for scc in sccs:
        for node in scc:
            if len([n for n in G.predecessors(node) if n not in scc]) == 0:
                #append the first node of the scc that has no incoming edges
                #transform the scc to list
                root_nodes.extend(list(scc))
                break
    return root_nodes


def get_root_nodes(G):
    """Get the root nodes of a graph."""
    sccs = list(nx.strongly_connected_components(G))
    return get_root_nodes_from_scc(sccs, G)

def remove_all_isolated_nodes(graph):
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph

def convert_types(G):
    # Convert the string attributes to their corresponding types
    for node, data in G.nodes(data=True):
        # The label remains a string, so no conversion is needed for 'label'
        # Convert struct_size, valid_pointer_count, invalid_pointer_count,
        # first_pointer_offset, last_pointer_offset, first_valid_pointer_offset,
        # last_valid_pointer_offset, and address to int
        data['struct_size'] = int(data['struct_size'])
        data['valid_pointer_count'] = int(data['valid_pointer_count'])
        data['invalid_pointer_count'] = int(data['invalid_pointer_count'])
        data['first_pointer_offset'] = int(data['first_pointer_offset'])
        data['last_pointer_offset'] = int(data['last_pointer_offset'])
        data['first_valid_pointer_offset'] = int(data['first_valid_pointer_offset'])
        data['last_valid_pointer_offset'] = int(data['last_valid_pointer_offset'])
        data['address'] = int(data['address'])

        # Convert cat to an integer and ensure it's within the range of a byte (0-255)
        data['cat'] = int(data['cat'])
        if not (0 <= data['cat'] <= 255):
            raise ValueError(f"Value of 'cat' out of range for u8: {data['cat']}")
    return G


def generate_single_graph_data(G):
    """ Generate data for a single graph with multiple root nodes. """

    #get the target nodes by getting all nodes that has feature 'cat' == 1
    target_nodes = [node for node, attributes in G.nodes(data=True) if attributes['cat'] == 1]


    #get root nodes by gettinng all nodes that has no predecessors
    root_nodes = get_root_nodes(G)

    data = []

    count_with_path = 0
    count_without_path = 0

    for node in root_nodes:
        if node not in target_nodes:
            #Check if the root node has a path to all target nodes
            has_path = 1
            for target_node in target_nodes:
                if not nx.has_path(G, node, target_node):
                    has_path = 0
                    break
            if has_path == 1:
                count_with_path += 1
            else:
                count_without_path += 1
            data.append((G, node, has_path))
    #check if at least 1 root node has a path to all target nodes
    if count_with_path == 0:
        raise ValueError("No root node has a path to all target nodes")
    
    #print(f"Number of root nodes with path: {count_with_path} over {len(root_nodes)}")
    return data



def extract_features(graph_data):
    """ Extract features from the graph data. """
    features = []
    for G, root_node, path_exists in graph_data:
        root_attributes = G.nodes[root_node] # Get attributes of the root node
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
        features.append(feature)
    return np.array(features, dtype=np.float32)


def prepare_data(Graph):
    """ Prepare training data and labels """
    graph_data = generate_single_graph_data(Graph)
    features = extract_features(graph_data)
    labels = [int(data[-1]) for data in graph_data]  # Extracting path_exists as labels
    return features, labels




def load_graphs(root_folder, max_per_subfolder=10, shuffle=False):
    all_graphs = []

    for subdir, dirs, files in os.walk(root_folder):
        print(f"Processing {subdir}...")
        graph_count = 0
        for file in files:
            if file.endswith('.graphml') and (max_per_subfolder == -1 or graph_count < max_per_subfolder ):
                file_path = os.path.join(subdir, file)
                try:
                    graph = nx.read_graphml(file_path)
                    all_graphs.append(graph)
                    graph_count += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if shuffle:
        random.shuffle(all_graphs)

    return all_graphs



# Configuration

FOLDER = 'Generated_Graphs/output/'


features = []
labels = []
print("Loading graphs...")
graphs = load_graphs(FOLDER,6, shuffle=True)

total_nb_nodes = 0
for G in tqdm(graphs):
    G = remove_all_isolated_nodes(G)
    G = convert_types(G)
    # Prepare data
    total_nb_nodes += G.number_of_nodes()
    curr_features, curr_labels = prepare_data(G)
    features.extend(curr_features)
    labels.extend(curr_labels)
    
print("Done loading graphs!")
print(f"Total number of graphs: {len(graphs)}")

# Split data into training and testing
print("Splitting data into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
print("Done splitting data!")
# Train a model
print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Done training model!")


"""
# Evaluate the model on the same graph
y_pred = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
for i, prob in enumerate(y_pred):
    if prob > 0.5:
        print(f"Graph : {X_test[i][0]} Probability of reaching target nodes: {prob:.2f} has path: {y_test[i]}")
"""
# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")

#Save the model
joblib.dump(model, 'models/root_heuristic_model.joblib')
