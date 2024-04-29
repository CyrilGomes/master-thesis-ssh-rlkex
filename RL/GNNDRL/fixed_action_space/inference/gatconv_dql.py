import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, TopKPooling

class GraphQNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_actions, seed):
        super(GraphQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.embedding_size = 3
        self.heads = 2
        # Define GAT layers
        self.conv1 = GATv2Conv(num_node_features, self.embedding_size, edge_dim=num_edge_features, heads=self.heads, add_self_loops=True, dropout=0.5)
        self.norm1 = GraphNorm(self.embedding_size*self.heads)
        self.topkpool1 = TopKPooling(self.embedding_size*self.heads, ratio=0.8)
        self.conv2 = GATv2Conv(self.embedding_size*self.heads, self.embedding_size, edge_dim=num_edge_features,heads=self.heads, add_self_loops=True, dropout=0.5)
        self.norm2 = GraphNorm(self.embedding_size*self.heads)
        self.topkpool2 = TopKPooling(self.embedding_size*self.heads, ratio=0.8)
        self.conv3 = GATv2Conv(self.embedding_size*self.heads, self.embedding_size, edge_dim=num_edge_features, heads=self.heads, add_self_loops=True, dropout=0.5)
        self.norm3 = GraphNorm(self.embedding_size*self.heads)
        self.topkpool3 = TopKPooling(self.embedding_size*self.heads, ratio=0.8)


        # Dueling DQN layers
        self.value_stream = torch.nn.Linear(self.embedding_size*self.heads*3 + num_node_features, 16)
        self.value_2 = torch.nn.Linear(16, 8)
        self.value = torch.nn.Linear(8, 1)

        self.advantage_stream = torch.nn.Linear(self.embedding_size*self.heads*3 + num_node_features, 64)
        self.advantage_2 = torch.nn.Linear(64, 64)
        self.advantage = torch.nn.Linear(64, num_actions)

    def forward(self, x, edge_index, edge_attr, batch , current_node_ids, action_mask=None):  

        
        #ensure everything is on device
        x = x
        x0 = x
        edge_index = edge_index
        edge_attr = edge_attr
        action_mask = action_mask.to(x.get_device())
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long)

        batch = batch.to(x.get_device())

        # Process with GAT layers
        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=0.5, training=self.training)
        x1 = x
        #x, edge_index, edge_attr, batch, _, _ = self.topkpool1(x, edge_index, edge_attr, batch=batch)
        x = F.relu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=0.5, training=self.training)
        x2 = x
        #x, edge_index, edge_attr, batch, _, _ = self.topkpool2(x, edge_index, edge_attr, batch=batch)
        x = F.relu(self.norm3(self.conv3(x, edge_index, edge_attr)))
        x = F.dropout(x, p=0.5, training=self.training)
        x3 = x
        #x, edge_index, edge_attr, batch, _, _ = self.topkpool3(x, edge_index, edge_attr, batch=batch)


        cumulative_nodes = torch.cumsum(batch.bincount(), 0)
        global_node_indices = current_node_ids + torch.cat((torch.tensor([0], device=batch.device), cumulative_nodes[:-1]))
        x0 = x0[global_node_indices]
        x1 = x1[global_node_indices]
        x2 = x2[global_node_indices]
        x3 = x3[global_node_indices]

        x = torch.cat((x0, x1, x2, x3), dim=1)
        # Compute node-level advantage
        advantage = F.relu(self.advantage_stream(x))
        advantage = F.dropout(advantage, p=0.5, training=self.training)
        advantage = F.relu(self.advantage_2(advantage))
        advantage = F.dropout(advantage, p=0.5, training=self.training)
        advantage = self.advantage(advantage)
        #advantage should be of shape [num_graphs, num_actions]

        value = F.relu(self.value_stream(x))
        value = F.dropout(value, p=0.5, training=self.training)
        value = F.relu(self.value_2(value))
        value = F.dropout(value, p=0.5, training=self.training)
        value = self.value(value)
        #value should be of shape [num_graphs, 1]

        mean_advantage = torch.mean(advantage, dim=1, keepdim=True)
        qvals = value + (advantage - mean_advantage)


        qvals = qvals.masked_fill(action_mask == 0, -1e8)




        #check if qvals contains nan
        if torch.isnan(qvals).any():
            raise ValueError("Qvals contains nan")

        return qvals

