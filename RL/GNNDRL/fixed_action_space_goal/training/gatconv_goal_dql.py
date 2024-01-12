import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, TopKPooling, PANConv, PANPooling, global_mean_pool

class GATConcDQL(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_actions, goal_size, seed):
        super(GATConcDQL, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.embedding_size = 5
        self.heads = 1


        self.curr_node_embedder = torch.nn.Linear(num_node_features, self.embedding_size)

        # Define GATConv layers
        self.conv1 = GATv2Conv(num_node_features, self.embedding_size, heads=self.heads, edge_dim=num_edge_features, add_self_loops=True, dropout=0.5)
        self.norm1 = GraphNorm(self.embedding_size * self.heads)
        self.conv2 = GATv2Conv(self.embedding_size *self.heads , self.embedding_size, edge_dim=num_edge_features, heads=self.heads, add_self_loops=True, dropout=0.5)
        self.norm2 = GraphNorm(self.embedding_size * self.heads)

        self.pooling = TopKPooling(self.embedding_size * self.heads, 0.8)

        # Dueling DQN layers
        self.value_stream = torch.nn.Linear(self.embedding_size  + (self.embedding_size * self.heads)*2+ goal_size, 20)
        self.value = torch.nn.Linear(20, 1)

        self.advantage_stream = torch.nn.Linear(self.embedding_size  + (self.embedding_size * self.heads)*2 + goal_size, num_actions//2)
        self.advantage = torch.nn.Linear(num_actions//2, num_actions)

    def forward(self, x, edge_index, edge_attr, batch , current_node_ids, action_mask=None, one_hot_goal=None):  

        
        #ensure everything is on device
        x = x
        x0 = x
        edge_index = edge_index
        edge_attr = edge_attr
        action_mask = action_mask.to(x.get_device())

        if batch is None:
            one_hot_goal = one_hot_goal.unsqueeze(0)

            batch = torch.zeros(x.shape[0], dtype=torch.long)

        batch = batch.to(x.get_device())

        

        # Process with GAT layers

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.norm1(x))
        x1 = x

        x = self.conv2(x, edge_index)
        x = F.relu(self.norm2(x))


        x, edge_index, edge_attr, batch, _, _ = self.pooling(x, edge_index, edge_attr, batch=batch)

        global_pool = global_mean_pool(x, batch)

       


        cumulative_nodes = torch.cumsum(batch.bincount(), 0)
        global_node_indices = current_node_ids + torch.cat((torch.tensor([0], device=batch.device), cumulative_nodes[:-1]))
        x0 = x0[global_node_indices]
        x0 = F.relu(self.curr_node_embedder(x0))


        x1 = x1[global_node_indices]

        x = torch.cat((x0, x1, global_pool, one_hot_goal), dim=1)

        

        # Compute node-level advantage
        advantage = F.relu(self.advantage_stream(x))
        advantage = F.dropout(advantage, p=0.5, training=self.training)

        advantage = self.advantage(advantage)
        #advantage should be of shape [num_graphs, num_actions]

        value = F.relu(self.value_stream(x))
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

