# -------------------------
# IMPORTS AND SETUP
# -------------------------

import os
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import keyboard

from collections import deque
import numpy as np
import random
import torch
from torch import optim
from collections import namedtuple, deque
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch_geometric import compile

#import range tqdm
from tqdm import tqdm
from tqdm import trange
from torch_geometric.data import Data, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# GRAPH PROCESSING
# -------------------------




def connect_components(graph):
    undi_graph = graph.to_undirected()

    # Connect components
    components = list(nx.connected_components(undi_graph))
    for i in range(1, len(components)):

        graph.add_edge(random.choice(list(components[0])), random.choice(list(components[i])), offset=0)
        graph.add_edge(random.choice(list(components[i])), random.choice(list(components[0])), offset=0)

    return graph

def add_global_root_node(graph):
    root_node = "root"
    graph.add_node(root_node, label=root_node, cat=0, struct_size=0, pointer_count=0, valid_pointer_count=0, invalid_pointer_count=0, first_pointer_offset=0, last_pointer_offset=0, first_valid_pointer_offset=0, last_valid_pointer_offset=0, visited=1)
    [graph.add_edge(root_node, node, offset=0) for node in graph.nodes() if len(list(graph.predecessors(node))) == 0 and node != root_node]
    return graph


def remove_all_isolated_nodes(graph):
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph
def preprocess_graph(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    
    # Removing string attributes from nodes and edges
    for _, attributes in graph.nodes(data=True):
        for key in list(attributes):
            if isinstance(attributes[key], str):
                del attributes[key]

    for u, v, attributes in graph.edges(data=True):
        for key in list(attributes):
            if isinstance(attributes[key], str):
                del attributes[key]

    #graph = connect_components(graph)
    #graph = nx.subgraph(graph, nx.bfs_tree(graph, 0))
    graph = remove_all_isolated_nodes(graph)

    print(graph.nodes(data=True))
    #graph = add_global_root_node(graph)

    return graph

def load_graphs_from_directory(directory_path):
    graph_files = [f for f in os.listdir(directory_path) if f.endswith('.graphml')]
    graphs = [nx.read_graphml(os.path.join(directory_path, f)) for f in graph_files]
    return [preprocess_graph(g) for g in graphs]

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

    edge_attr = torch.tensor([graph[u][v]['offset'] for u, v in graph.edges], dtype=torch.float).unsqueeze(1)
    # y is 1 if there's at least one node with cat=1 in the graph, 0 otherwise
    y = torch.tensor([1 if any(attributes['cat'] == 1 for _, attributes in graph.nodes(data=True)) else 0], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(device)




def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        #data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

def get_best_graph(model, graphs):
    model.eval()
    best_score = -1
    best_graph = None
    with torch.no_grad():
        for graph in graphs:
            #graph = graph.to(device)
            score = model(graph)
            if score > best_score:
                best_score = score.item()
                best_graph = graph
    return best_graph, best_score




def create_subgraph_data(file_path):
    graph = nx.read_graphml(file_path)
    graph = preprocess_graph(graph)

    #get all subgraphs
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph.to_undirected())]
    subgraphs_data = [graph_to_data(g) for g in subgraphs]
    return subgraphs_data


def export_model(model, path, loader):
    os.makedirs(path, exist_ok=True)
    model.eval()

    # Using torch_geometric.compile instead of torch.jit.trace
    #compiled_model = compile(model, dynamic=True, fullgraph=True)

    torch.save(model.state_dict(), os.path.join(path, 'model_state_dict.pt'))
    
class GNN(torch.nn.Module):
    def __init__(self, dimension=7):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dimension, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Use global_mean_pool to compute the mean for each graph in the batch
        x = torch.sigmoid(global_mean_pool(x, batch))
        return x
    

def main():
    folder = 'Generated_Graphs/output/'
    file_names = os.listdir(folder)
    #keep only the first 100 graphs
    filtered_file_names = file_names[0:20]
    subgraphs_data = []
    for file_name in filtered_file_names:
        subgraphs_data += create_subgraph_data(os.path.join(folder, file_name))

    loader = DataLoader(subgraphs_data, batch_size=126, shuffle=True, collate_fn=Batch.from_data_list)

    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
     
    for epoch in range(300):
        loss = train(model, loader, optimizer)
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss))


    #get random graph and test
    random_file = random.choice(file_names)
    random_graph_data = create_subgraph_data(os.path.join(folder, random_file))
    
    best_graph, best_score = get_best_graph(model, random_graph_data)
    print(f"{best_graph} has the best score of {best_score} with y = {best_graph.y}")

    export_model(model, 'models', loader)


if __name__ == '__main__':
    main()