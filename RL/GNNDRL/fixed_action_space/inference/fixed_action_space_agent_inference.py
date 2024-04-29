



# -------------------------
# MODEL DEFINITION
# -------------------------

from collections import namedtuple
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, dense_diff_pool, DenseSAGEConv, TopKPooling
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter_max, scatter_mean
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from panconv_dql import PANConcDQL
from utils import MyGraphData





# -------------------------
# AGENT DEFINITION
# -------------------------
class Agent:
    def __init__(self, state_size, edge_attr_size, action_space, seed, device, local_network_path=None):
        self.writer = SummaryWriter('runs/DQL_GRAPH_FIXED_ACTION_SPACE')  # Choose an appropriate experiment name
        self.state_size = state_size
        self.seed = random.seed(seed)
        self.edge_attr_size = edge_attr_size
        self.device = device
        self.action_space = action_space


        # Q-Network
        self.qnetwork_local = GraphQNetwork(state_size, self.edge_attr_size, action_space, seed).to(device)
        #load local network if path is provided
        if local_network_path is not None:
            self.qnetwork_local.load_state_dict(torch.load(local_network_path))




    def act(self, state, action_mask=None):
        state = state
        action_mask = torch.from_numpy(action_mask)

        self.qnetwork_local.eval()
        x = state.x.to(self.device)
        curr_node = state.current_node_id
        edge_index = state.edge_index.to(self.device)
        edge_attr = state.edge_attr.to(self.device)

        with torch.no_grad():  # Wrap in no_grad
            action_values = self.qnetwork_local(x, edge_index, edge_attr, None, curr_node, action_mask)
        return_values = action_values.cpu()
        selected_action = torch.argmax(return_values).item()
        torch.cuda.empty_cache()

        return selected_action

    