# -------------------------
# IMPORTS AND SETUP
# -------------------------

import os
import random
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import random

#import range tqdm
from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import Data



class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)

        # Global Mean Pooling
        x = global_mean_pool(x, batch)

        # Fully Connected Layer for Classification
        x = F.relu(self.fc(x))

        return x


# -------------------------
# GRAPH PROCESSING
# -------------------------


def graph_to_data(graph):
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

    edge_attr = torch.tensor([data['offset'] for u, v, data in graph.edges(data=True)], dtype=torch.float).unsqueeze(1)
    # if there are 2 keys then y = 0, if there are 4 keys then y = 1, if there are 6 keys then y = 2
    key_count = len([node for node in graph.nodes() if graph.nodes[node]['cat'] >= 0])

    #Create a tensor [1,0,0] if there are 2 keys, [0,1,0] if there are 4 keys, [0,0,1] if there are 6 keys
    if key_count == 2:
        y = torch.tensor([1,0,0], dtype=torch.float)
    elif key_count == 4:
        y =  torch.tensor([0,1,0], dtype=torch.float)
    elif key_count == 6:
        y =  torch.tensor([0,0,1], dtype=torch.float)
    else:
        raise ValueError(f"Invalid number of keys: {key_count}")
    
    

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

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


    # Convert edges to their corresponding types
    for u, v, data in G.edges(data=True):
        # Convert offset to int
        data['offset'] = int(data['offset'])
    return G






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


def train(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=8, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training Loop
    for epoch in range(200):
        model.train()
        total_loss = 0
        for data in loader:
            print(data)
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss / len(loader)}')


def main():
    folder = 'Generated_Graphs/output/'
    graphs = load_graphs(folder)
    print(f"Loaded {len(graphs)} graphs")
    graphs = [convert_types(graph) for graph in graphs]
    graphs = [remove_all_isolated_nodes(graph) for graph in graphs]
    print(f"Removed isolated nodes from graphs")
    print(f"Loaded {len(graphs)} graphs")
    dataset = [graph_to_data(graph) for graph in graphs]
    print(f"Converted graphs to data")
    print(f"Loaded {len(dataset)} graphs")
    train(dataset)
    print(f"Trained model")


    

if __name__ == '__main__':
    main()


