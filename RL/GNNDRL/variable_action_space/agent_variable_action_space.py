



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

from torch.utils.tensorboard import SummaryWriter
import random

from utils import MyGraphData
from per import SumTree, Memory


class GraphQNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, seed):
        super(GraphQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm0 = GraphNorm(num_node_features)

        # Define GAT layers
        self.conv1 = GATv2Conv(num_node_features, 5, edge_dim=num_edge_features, add_self_loops=False)
        self.norm1 = GraphNorm(5)
        self.topkpool1 = TopKPooling(5, ratio=0.8)
        self.conv2 = GATv2Conv(5, 3, edge_dim=num_edge_features, add_self_loops=False)
        self.norm2 = GraphNorm(3)
        self.topkpool2 = TopKPooling(3, ratio=0.8)
        self.conv3 = GATv2Conv(3, 3, edge_dim=num_edge_features, add_self_loops=False)
        self.norm3 = GraphNorm(3)


        # DiffPool layer


        # Define dropout
        self.dropout = torch.nn.Dropout(p=0.5)

        # Dueling DQN layers
        self.value_stream = torch.nn.Linear(3, 3)
        self.value = torch.nn.Linear(3, 1)
        self.advantage_stream = torch.nn.Linear(14 ,6)
        self.advantage = torch.nn.Linear(6, 1)

    def forward(self, x, edge_index, edge_attr, batch , action_mask=None):  

        
        #ensure everything is on device
        x = x
        edge_index = edge_index
        edge_attr = edge_attr
        action_mask = action_mask.to(x.get_device())
        if batch is None:
            
            batch = torch.zeros(x.shape[0], dtype=torch.long)

        batch = batch.to(x.get_device())


        # Process with GAT layers
        x1 = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x1 = self.dropout(x1)

        x1_pool, x1_edge_index_pool, x1_edge_attr_pool, x1_batch_pool, _, _ = self.topkpool1(x1, edge_index, edge_attr, batch=batch)

        x2 = F.relu(self.norm2(self.conv2(x1_pool, x1_edge_index_pool, x1_edge_attr_pool)))
        x2 = self.dropout(x2)
        x2_pool, x2_edge_index_pool, x2_edge_attr_pool, x2_batch_pool, _, _ = self.topkpool2(x2, x1_edge_index_pool, x1_edge_attr_pool, batch=x1_batch_pool)
        x3 = F.relu(self.norm3(self.conv3(x2_pool, x2_edge_index_pool, x2_edge_attr_pool)))



        pooled_mean_3 = global_mean_pool(x3, x2_batch_pool)


        pooled_add_3 = global_add_pool(x3, x2_batch_pool)


        pooled_max_3 = global_max_pool(x3, x2_batch_pool)





        #expanded_mean_1 = pooled_mean_1[batch]
        #expanded_mean_2 = pooled_mean_2[batch]
        expanded_mean_3 = pooled_mean_3[batch]

        #expanded_add_1 = pooled_add_1[batch]
        #expanded_add_2 = pooled_add_2[batch]
        expanded_add_3 = pooled_add_3[batch]

        #expanded_max_1 = pooled_max_1[batch]
        #expanded_max_2 = pooled_max_2[batch]
        expanded_max_3 = pooled_max_3[batch]

        expanded_pooled_val_1 = torch.cat([expanded_mean_3, expanded_add_3, expanded_max_3], dim=-1).squeeze(-1)
        #expanded_pooled_val_2 = torch.cat([expanded_mean_2, expanded_add_2, expanded_max_2], dim=-1).squeeze(-1)
        #expanded_pooled_val_3 = torch.cat([expanded_mean_3, expanded_add_3, expanded_max_3], dim=-1).squeeze(-1)



        pooled_val = torch.cat([expanded_pooled_val_1], dim=-1).squeeze(-1)

        final_conc = torch.cat([x1, pooled_val], dim=1)

        
        # Compute node-level advantage
        advantage = F.relu(self.advantage_stream(final_conc))
        advantage = self.dropout(advantage)
        advantage = self.advantage(advantage).squeeze(-1)  # Remove last dimension


        value_input = torch.cat([pooled_max_3, pooled_mean_3, pooled_add_3], dim=0)
        #combined_features = torch.cat([pooled_val, x_conc], dim=-1)
        value = F.relu(self.value_stream(value_input))
        value = self.dropout(value)
        value = self.value(value).squeeze(-1)  # Remove last dimension
            

        total_batches = batch.unique().shape[0]

        valid_actions = (action_mask == 1).nonzero(as_tuple=True)[0]

        advantage_batch = batch[valid_actions]

        #get the valid actions advantage
        masked_advantage = advantage[valid_actions]

        mean_advantage = torch.zeros(total_batches, device=advantage.get_device())


        mean_advantage = scatter_mean(masked_advantage, advantage_batch, dim=0, dim_size=total_batches)

        # Expand mean advantage to match the number of nodes
        expanded_mean_advantage = mean_advantage[batch]
        value = value[batch]
        
        qvals = value + (advantage - expanded_mean_advantage)

        num_graphs = batch.unique().shape[0]
        for i in range(num_graphs):
            graph_mask = action_mask[batch == i]
            #keep only the qvals of the masked actions
            filtered_actions = (advantage - expanded_mean_advantage)[batch == i][graph_mask == 1]


        # Apply action mask if provided
        if action_mask is not None:
            action_mask = action_mask.to(qvals.device)
            # Set Q-values of valid actions (where action_mask is 0) as is, and others to -inf
            qvals = torch.where(action_mask == 1, qvals, torch.tensor(float('-1e9')).to(qvals.device))
        

        #check if qvals contains nan
        if torch.isnan(qvals).any():
            raise ValueError("Qvals contains nan")

        return qvals




# -------------------------
# AGENT DEFINITION
# -------------------------
class Agent:
    def __init__(self, state_size, edge_attr_size, seed, device, lr, buffer_size, batch_size, gamma, tau, update_every):
        self.writer = SummaryWriter('runs/DQL_GRAPH_VARIABLE_ACTION_SPACE')  # Choose an appropriate experiment name
        self.state_size = state_size
        self.seed = random.seed(seed)
        self.edge_attr_size = edge_attr_size
        self.device = device
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every


        # Q-Network
        self.qnetwork_local = GraphQNetwork(state_size, self.edge_attr_size, seed).to(device)
        self.qnetwork_target = GraphQNetwork(state_size, self.edge_attr_size,seed).to(device)

        #init the weights of the target network to be the same as the local network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())



        self.optimizer = Adam(self.qnetwork_local.parameters(), lr=lr)

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "action_mask", "next_action_mask"])
        
        self.buffer = Memory(capacity=buffer_size)
        
        self.t_step = 0

        self.losses = []
        self.steps = 0



    def add_experience(self, state, action, reward, next_state, done, action_mask, next_action_mask):        
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done, action_mask, next_action_mask)
        #check if there is None in experience
        for key, value in experience._asdict().items():
            if value is None:
                raise ValueError(f"Value of {key} is None")
        self.buffer.store(experience)



    def log_environment_change(self, env_name):
        self.writer.add_text('Environment Change', f'Changed to {env_name}', self.steps)


    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.steps)



    def step(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None):
        self.steps += 1
        #ensure everything is on device
        state = state
        next_state = next_state
        
        # Save experience in replay memory
        self.add_experience(state, action, reward, next_state, done, action_mask, next_action_mask)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.buffer) >= self.buffer_size:
                indices, experiences, is_weights = self.buffer.sample(self.batch_size)
                self.learn(experiences, indices, is_weights,  self.gamma)

    def act(self, state, eps=0, action_mask=None):
        state = state
        action_mask = torch.from_numpy(action_mask)

        if random.random() > eps:
            self.qnetwork_local.eval()
            x = state.x.to(self.device)
            edge_index = state.edge_index.to(self.device)
            edge_attr = state.edge_attr.to(self.device)

            with torch.no_grad():  # Wrap in no_grad
                action_values = self.qnetwork_local(x, edge_index, edge_attr, None, action_mask)
            return_values = action_values.cpu()
            self.qnetwork_local.train()

            selected_action = torch.argmax(return_values).item()
            torch.cuda.empty_cache()

            return selected_action
        else:
            choices = (action_mask.cpu() == 1).nonzero(as_tuple=True)[0]

            return random.choice(choices).item()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,  # Optional: save replay memory
            'steps': self.steps  # Optional: save training steps
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint.get('memory', self.memory)  # Load memory if saved
        self.steps = checkpoint.get('steps', self.steps)  # Load steps if saved
    def get_max_q_indices(self, q_values, batch):
        max_q_indices = []
        max_q_values = []
        for graph_id in batch.unique():
            # Mask for nodes in the current graph
            mask = (batch == graph_id)

            # Q-values for the current graph
            q_values_graph = q_values[mask]

            # Find the max Q-value and its index for this graph
            max_q_value, max_q_index = q_values_graph.max(dim=0)
            max_q_values.append(max_q_value)
            max_q_indices.append(max_q_index + mask.nonzero(as_tuple=True)[0].min())

        return torch.tensor(max_q_indices), torch.tensor(max_q_values)
    def learn(self, experiences, indices, is_weights, gamma):
        # Create a list of Data objects, each representing a graph experience


        data_list = []
        for i, e in enumerate(experiences):
            state = e.state
            action = torch.tensor(e.action, dtype=torch.long)
            reward = e.reward
            next_state = e.next_state
            done = torch.tensor(e.done, dtype=torch.uint8)
            exp_index = indices[i]
            mask = torch.tensor(e.action_mask, dtype=torch.int8)
            next_mask = torch.tensor(e.next_action_mask, dtype=torch.int8)
            is_weight = torch.tensor(is_weights[i][0])

            data = MyGraphData(x_s=state.x, edge_index_s=state.edge_index, edge_attr_s=state.edge_attr,
                            action=action, reward=reward, x_t=next_state.x,
                            edge_index_t=next_state.edge_index, edge_attr_t=next_state.edge_attr,
                            done=done, exp_idx=exp_index, mask = mask, next_mask = next_mask, is_weight=is_weight)
            
            


            data_list.append(data)


        # Create a DataLoader for batching
        batch_size = self.batch_size
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['x_s', 'x_t'], pin_memory=True)
        device = self.device
        for batch in data_loader:

            batch.to(device)  # Send the batch to the GPU if available
            b_exp_indices = batch.exp_idx
            # Extract batched action, reward, done, and action_mask
            b_action = batch.action.to(device)
            b_reward = batch.reward.to(device)
            b_done = batch.done.to(device)
            b_action_mask = batch.mask
            b_next_action_mask = batch.next_mask


            b_x_s = batch.x_s.to(device)
            b_edge_index_s = batch.edge_index_s.to(device)
            b_edge_attr_s = batch.edge_attr_s.to(device)
            b_x_t = batch.x_t.to(device)
            b_edge_index_t = batch.edge_index_t.to(device)
            b_edge_attr_t = batch.edge_attr_t.to(device)
            x_s_batch = batch.x_s_batch.to(device)
            x_t_batch = batch.x_t_batch.to(device)
            b_is_weight = batch.is_weight.to(device)

            # DDQN Update for the individual graph
            self.qnetwork_target.eval()
            self.qnetwork_local.train()
            # Calculate Q values for next states
            with torch.no_grad():

                #Double DQN
                q_local_next = self.qnetwork_local(b_x_t, b_edge_index_t, b_edge_attr_t, x_t_batch, action_mask=b_next_action_mask)
                #get the indices actions with the highest q values, for each graph, using x_t_batch, since the dimension of q_local_next is [num_actions]
                q_local_max, q_indices = scatter_max(q_local_next, x_t_batch, dim=0)

                Q_targets_next = self.qnetwork_target(b_x_t, b_edge_index_t, 
                                                    b_edge_attr_t, x_t_batch, 
                                                    action_mask=b_next_action_mask)
                
                
            q_indices = q_indices.to(Q_targets_next.get_device())
            Q_targets_next = Q_targets_next.gather(0, q_indices).detach().squeeze()

            Q_targets = b_reward + (gamma * Q_targets_next * (1 - b_done))

            self.qnetwork_local.train()
            Q_expected_result = self.qnetwork_local(b_x_s, b_edge_index_s, b_edge_attr_s, batch.batch, action_mask=b_action_mask).squeeze()



            # Ensure b_action is 1D with shape [num_graphs]
            b_action = b_action.squeeze()

            # Count the number of nodes in each graph
            nodes_per_graph = x_s_batch.bincount()

            # Calculate the start index of each graph in the concatenated batch
            cumulative_sizes = torch.cat([torch.tensor([0], device=x_s_batch.device), nodes_per_graph.cumsum(0)[:-1]])

            # Offset b_action based on the starting index of each graph
            adjusted_b_action = b_action + cumulative_sizes

            # Ensure adjusted_b_action is 1D
            adjusted_b_action = adjusted_b_action.squeeze()



            # Now gather the Q values
            Q_expected = Q_expected_result.gather(0, adjusted_b_action)

            
            # Compute loss


            td_error = torch.abs(Q_expected - Q_targets)
            loss = torch.mean(td_error.pow(2))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            # Soft update target network


            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

            # Update priorities, losses for logging, etc.
            td_error = td_error.detach().cpu().numpy()
            self.buffer.batch_update(b_exp_indices, td_error)

            self.losses.append(loss.item())



    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

