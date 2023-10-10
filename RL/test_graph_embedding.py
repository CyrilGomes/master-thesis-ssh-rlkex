import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx

# Read graph from graphml file using networkx
G = nx.read_graphml('Generated_Graphs/64/18038-1643986141-heap.graphml')
data = from_networkx(G)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)

# Dummy node features, replace with actual node features if available
data.x = torch.ones(data.num_nodes, 1).to(device)

model.eval()
with torch.no_grad():
    embeddings = model(data)

print(embeddings)  # Output node embeddings
