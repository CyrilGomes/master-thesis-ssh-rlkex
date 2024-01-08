



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

from utils import MyGraphData
from per import SumTree, Memory


class GraphQNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_actions, seed):
        super(GraphQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.embedding_size = 3
        # Define GAT layers
        self.conv1 = GATv2Conv(num_node_features, self.embedding_size, edge_dim=num_edge_features, heads=4, add_self_loops=True, dropout=0.5)
        self.norm1 = GraphNorm(self.embedding_size*4)
        self.topkpool1 = TopKPooling(self.embedding_size*4, ratio=0.8)
        self.conv2 = GATv2Conv(self.embedding_size*4, self.embedding_size, edge_dim=num_edge_features,heads=4, add_self_loops=True, dropout=0.5)
        self.norm2 = GraphNorm(self.embedding_size*4)
        self.topkpool2 = TopKPooling(self.embedding_size*4, ratio=0.8)
        self.conv3 = GATv2Conv(self.embedding_size*4, self.embedding_size, edge_dim=num_edge_features, heads=4, add_self_loops=True, dropout=0.5)
        self.norm3 = GraphNorm(self.embedding_size*4)
        self.topkpool3 = TopKPooling(self.embedding_size*4, ratio=0.8)


        # Dueling DQN layers
        self.value_stream = torch.nn.Linear(self.embedding_size*4*3 + num_node_features, 16)
        self.value_2 = torch.nn.Linear(16, 8)
        self.value = torch.nn.Linear(8, 1)

        self.advantage_stream = torch.nn.Linear(self.embedding_size*4*3 + num_node_features, 64)
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




# -------------------------
# AGENT DEFINITION
# -------------------------
class Agent:
    def __init__(self, state_size, edge_attr_size, action_space, seed, device, lr, buffer_size, batch_size, gamma, tau, update_every):
        self.writer = SummaryWriter('runs/DQL_GRAPH_FIXED_ACTION_SPACE')  # Choose an appropriate experiment name
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
        self.action_space = action_space


        # Q-Network
        self.qnetwork_local = GraphQNetwork(state_size, self.edge_attr_size, action_space, seed).to(device)
        self.qnetwork_target = GraphQNetwork(state_size,self.edge_attr_size,action_space, seed).to(device)

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

    def act(self, state, eps=0, action_mask=None, weight_array=None):
        state = state
        action_mask = torch.from_numpy(action_mask)

        if random.random() > eps:
            self.qnetwork_local.eval()
            x = state.x.to(self.device)
            curr_node = state.current_node_id
            edge_index = state.edge_index.to(self.device)
            edge_attr = state.edge_attr.to(self.device)

            with torch.no_grad():  # Wrap in no_grad
                action_values = self.qnetwork_local(x, edge_index, edge_attr, None, curr_node, action_mask)
            return_values = action_values.cpu()
            self.qnetwork_local.train()

            selected_action = torch.argmax(return_values).item()
            torch.cuda.empty_cache()

            return selected_action
        else:
            

            if weight_array is not None:
                #weight_array is of shape [num_actions]
                #some has probability 0 if they are not valid actions

                #select the action based on the probability distribution
                #given by the weight_array
                np.random.seed()
                selected_action = np.random.choice(np.arange(len(weight_array)), p=weight_array)
            else:
                choices = (action_mask.cpu() == 1).nonzero(as_tuple=True)[0]

                selected_action = np.random.choice(choices)
            return selected_action
    
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

    def learn(self, experiences, indices, is_weights, gamma):
        # Create a list of Data objects, each representing a graph experience
        # DDQN Update for the individual graph
        self.qnetwork_local.train()
        self.qnetwork_target.train()

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
            current_node_id = torch.tensor(e.state.current_node_id, dtype=torch.long)
            next_current_node_id = torch.tensor(e.next_state.current_node_id, dtype=torch.long)

            data = MyGraphData(x_s=state.x, edge_index_s=state.edge_index, edge_attr_s=state.edge_attr,
                            action=action, reward=reward, x_t=next_state.x,
                            edge_index_t=next_state.edge_index, edge_attr_t=next_state.edge_attr,
                            done=done, exp_idx=exp_index, mask = mask, next_mask = next_mask, is_weight=is_weight, 
                            cnid=current_node_id, next_cnid=next_current_node_id)
            
            


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
            b_current_node_id = batch.cnid.to(device)
            b_next_current_node_id = batch.next_cnid.to(device)

            # Calculate Q values for next states
            with torch.no_grad():

                #Double DQN
                q_local_next = self.qnetwork_local(b_x_t, b_edge_index_t, b_edge_attr_t, x_t_batch, b_next_current_node_id, action_mask=b_next_action_mask)
                #q_local_next is of shape [num_graphs, num_actions]
                indices = torch.argmax(q_local_next, dim=1)

                Q_targets_next = self.qnetwork_target(b_x_t, b_edge_index_t, 
                                                    b_edge_attr_t, x_t_batch, b_next_current_node_id,
                                                    action_mask=b_next_action_mask)
                
                
            Q_targets_next = Q_targets_next.gather(1, indices.unsqueeze(1)).squeeze(1)

            Q_targets = b_reward + (gamma * Q_targets_next * (1 - b_done))

            Q_expected_result = self.qnetwork_local(b_x_s, b_edge_index_s, b_edge_attr_s, x_s_batch, b_current_node_id, action_mask=b_action_mask)


            #b_action is of shape [batch_size]

            # Now gather the Q values
            Q_expected = Q_expected_result.gather(1, b_action.unsqueeze(1)).squeeze(1)

            
            # Compute loss


            td_error = torch.abs(Q_expected - Q_targets)
            loss = torch.mean(b_is_weight * td_error.pow(2))

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

