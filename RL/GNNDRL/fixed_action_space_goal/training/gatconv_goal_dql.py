import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, TopKPooling, PANConv, PANPooling, global_mean_pool
import torch_scatter
class GATConcDQL(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_actions, goal_size, seed):
        super(GATConcDQL, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.embedding_size = 5
        self.heads = 1

        self.norm = GraphNorm(num_node_features)
        # Define GATConv layers
        self.conv1 = GATv2Conv(num_node_features, self.embedding_size, heads=self.heads, edge_dim=num_edge_features, add_self_loops=True)
        #self.norm1 = GraphNorm(self.embedding_size * self.heads)
        self.conv2 = GATv2Conv(13 , self.embedding_size, edge_dim=num_edge_features, heads=self.heads, add_self_loops=True)
        #self.norm2 = GraphNorm(self.embedding_size * self.heads)


        # Dueling DQN layers
        self.value_stream = torch.nn.Linear(42, 10)
        self.value = torch.nn.Linear(10, 1)

        self.advantage_stream = torch.nn.Linear(42,10)
        self.advantage = torch.nn.Linear(10, num_actions)

    def forward(self, x, edge_index, edge_attr, batch , current_node_ids, action_mask=None, one_hot_goal=None):  

        
        #ensure everything is on device

        edge_index = edge_index
        edge_attr = edge_attr
        action_mask = action_mask.to(x.get_device())

        if batch is None:
            one_hot_goal = one_hot_goal.unsqueeze(0)

            batch = torch.zeros(x.shape[0], dtype=torch.long)

        batch = batch.to(x.get_device())

        x = self.norm(x, batch)

        # Process with GAT layers
        identity = x
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = torch.cat((x, identity), dim=1)

        identity = x
        #apply conv 2 with return attention weights
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = torch.cat((x, identity), dim=1)
        

 
        global_pool = global_mean_pool(x, batch)

        cumulative_nodes = torch.cumsum(batch.bincount(), 0)
        global_node_indices = current_node_ids + torch.cat((torch.tensor([0], device=batch.device), cumulative_nodes[:-1]))
        x = x[global_node_indices]

        state = torch.cat((x, global_pool), dim=1)

        x = torch.cat((state, one_hot_goal), dim=1)
        
        # Compute node-level advantage
        advantage = F.relu(self.advantage_stream(x))

        advantage = self.advantage(advantage)
        #advantage should be of shape [num_graphs, num_actions]

        value = F.relu(self.value_stream(x))
        value = self.value(value)
        #value should be of shape [num_graphs, 1]

        mean_advantage = torch.mean(advantage, dim=1, keepdim=True)
        qvals = value + (advantage - mean_advantage)


        qvals = qvals.masked_fill(action_mask == 0, -1e8)




        #check if qvals contains nan
        if torch.isnan(qvals).any():
            raise ValueError("Qvals contains nan")

        return qvals

