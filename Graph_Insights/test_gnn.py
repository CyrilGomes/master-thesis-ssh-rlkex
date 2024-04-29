import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
import random
import os
from torch_geometric.utils import from_networkx

"""
Testing a regulat GNN to classify nodes,

Seems to overfit, since the training data contain little positive examples.
Tried Augmentation, but didn't help much.

Reinforcement Learning might help with the imbalanced data.
"""

def preprocess_graph(graph):
    # Relabel the nodes to use integers
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)

    # Remove string properties from nodes
    for _, attributes in graph.nodes(data=True):
        for key, value in list(attributes.items()):
            if isinstance(value, str):
                del attributes[key]

    # Remove string properties from edges
    for _, _, attributes in graph.edges(data=True):
        for key, value in list(attributes.items()):
            if isinstance(value, str):
                del attributes[key]

    return graph

def read_graphs_from_directory(directory, percentage=100, augment=True):
    graph_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.graphml')]

    # Read a subset of the graph files based on the specified percentage
    subset_size = int(len(graph_files) * (percentage / 100))
    graph_files_subset = random.sample(graph_files, subset_size)

    graphs = []
    for graph_file in graph_files_subset:
        graph = nx.read_graphml(graph_file)
        graph = preprocess_graph(graph)

        if not graph.nodes():  # skip empty graphs
            continue

        graphs.append(graph)

        if augment and any(graph.nodes[node]['cat'] == 1 for node in graph.nodes()):
            # Augment the graph by randomly selecting a positive labeled node and adding it with perturbation
            positive_nodes = [node for node in graph.nodes() if graph.nodes[node]['cat'] == 1]
            selected_node = random.choice(positive_nodes)
            selected_node_neighbors = list(graph.neighbors(selected_node))
            new_node = max(graph.nodes()) + 1
            graph.add_node(new_node, **graph.nodes[selected_node])
            graph.add_edge(selected_node, new_node)
            for neighbor in selected_node_neighbors:
                if random.random() < 0.5:
                    graph.add_edge(new_node, neighbor)
                else:
                    graph.remove_edge(selected_node, neighbor)

            graphs.append(graph)

    return graphs

def graph_to_data(graph):
    # Extract node attributes for 'cat'
    for node, attributes in graph.nodes(data=True):
        graph.nodes[node]['cat'] = int(attributes.get('cat', 0))

    data = from_networkx(graph)
    # Convert 'cat' attribute to a tensor
    data.y = torch.tensor([graph.nodes[node]['cat'] for node in graph.nodes()], dtype=torch.long)
    return data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)  # 2 for binary classification

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def random_graph():
    num_nodes = random.randint(10, 50)
    p_edge = 0.5
    G = nx.fast_gnp_random_graph(num_nodes, p_edge)
    nx.set_node_attributes(G, 0, 'cat')
    
    # Randomly mark a node as 'cat'=1
    random_node = random.choice(list(G.nodes()))
    G.nodes[random_node]['cat'] = 1
    
    return G

def graph_to_data(graph):
    # Node features (1-dimensional in this case)
    x = torch.tensor([[graph.nodes[node]['cat']] for node in graph.nodes()], dtype=torch.float)
    
    # Edge indices
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Node labels (targets for training)
    y = torch.tensor([graph.nodes[node]['cat'] for node in graph.nodes()], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Read graphs
graphs_path = "Generated_Graphs/64"
graphs = read_graphs_from_directory(graphs_path, percentage=0.3)
datasets = [graph_to_data(g) for g in graphs]
random.shuffle(datasets)  # Shuffle the dataset

split = int(0.8 * len(datasets))
train_dataset = datasets[:split]
test_dataset = datasets[split:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
    return correct / total

for epoch in range(500):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Inference on a new graph
new_graph = nx.read_graphml('Generated_Graphs/64/18041-1643986141-heap.graphml')
new_graph = preprocess_graph(new_graph)
new_data = graph_to_data(new_graph).to(device)
model.eval()
with torch.no_grad():
    out = model(new_data.x, new_data.edge_index)
    pred = out.argmax(dim=1)
    detected_node_indices = (pred == 1).nonzero(as_tuple=True)[0]
if detected_node_indices.numel() == 0:
    print("No nodes detected.")
else:
    detected_node_index = detected_node_indices.item()
    print("Detected node index:", detected_node_index)