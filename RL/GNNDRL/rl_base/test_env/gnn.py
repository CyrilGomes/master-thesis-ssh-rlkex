#class used for rl_environment for graph heuristic
import torch

from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv


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
