



# -------------------------
# MODEL DEFINITION
# -------------------------

from collections import namedtuple
import datetime
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
import os
from torch.utils.tensorboard import SummaryWriter
import random

from utils import MyGraphData
from per import SumTree, Memory


class GraphQNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, seed):
        super(GraphQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)


        # Goal-specific branches as a list of modules
        self.goal_branches = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(76, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 26)
            ) for _ in range(6)
        ])


        # Define GAT layers
        self.norm0 = GraphNorm(num_node_features)
        self.norm_edge = GraphNorm(num_edge_features)

        self.conv1 = GATv2Conv(16 , 5, edge_dim=num_edge_features, heads=2) #  16 + 10 = 26
        self.conv2 = GATv2Conv(26, 10, edge_dim=num_edge_features, heads=2) # 26 + 20 = 44
        self.conv3 = GATv2Conv(46, 30, edge_dim=num_edge_features) # 44 + 30 = 74

        self.conv_whole_graph = GATv2Conv(57 , 22, edge_dim=num_edge_features, heads=2)

        # Layer to integrate goal-specific output with general features
        self.merge_layer = torch.nn.Linear(102, 32)  # Adjust sizes accordingly


        self.graph_embedder = torch.nn.Linear(32, 16)
        self.graph_embedder_1 = torch.nn.Linear(16, 10)
        self.graph_embedder_2 = torch.nn.Linear(10, 6)

        self.subgraph_embedder = torch.nn.Linear(32, 16)
        self.subgraph_embedder_1 = torch.nn.Linear(16, 10)
        self.subgraph_embedder_2 = torch.nn.Linear(10, 6)

        #28 + 12 =  

        self.state_goal_branches = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(76, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 12),
                torch.nn.ReLU(),
                torch.nn.Linear(12, 10)
            ) for _ in range(6)
        ])


        self.qvalue_goal_branches = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(44, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1)
            ) for _ in range(6)
        ])

        

    def forward(self, x, edge_index, edge_attr, batch , action_mask, goal, visited_subgraph, subgraph_node_indices_batch, current_node):  

 

        if action_mask is not None:
            action_mask = action_mask.to(x.get_device())

        if batch is None:
            
            batch = torch.zeros(x.shape[0], dtype=torch.long)
        

        batch = batch.to(x.get_device())



        # Process with GAT layers
        x = self.norm0(x, batch)

        edge_attr = self.norm_edge(edge_attr, batch[edge_index[0]])
        current_node_feature = x[current_node]


        #embed the mean of the attributes of the outgoing edges of each node
        edge_attr_in_mean = scatter_mean(edge_attr, edge_index[0], dim=0, dim_size=x.shape[0])
        edge_attr_out_mean = scatter_mean(edge_attr, edge_index[1], dim=0, dim_size=x.shape[0])

        x = torch.cat([x, edge_attr_in_mean, edge_attr_out_mean, current_node_feature[batch]], dim=-1)

        # Initial input features
        identity = x

        # First GAT layer with skip connection
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.cat([x, identity], dim=1)

        identity = x
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.cat([x, identity], dim=1)

        identity = x
        x = self.conv3(x, edge_index, edge_attr)
        x_base = torch.cat([x, identity], dim=1)

        goal_indices = goal.argmax(dim=1)  # Assuming one-hot encoded goals
        #expand goal indices for each node of the correpsonding batch
        goal_indices = goal_indices[batch]

        x_goal_specific = torch.zeros_like(x_base[:,:26])
        # Iterate over each goal-specific branch
        for i, branch in enumerate(self.goal_branches):
            mask = goal_indices == i
            if mask.any():  # Check if the current goal index is present in the batch
                x_goal_specific[mask] = branch(x_base[mask])

        # Combine goal-specific output with the base features
        merged = torch.cat([x_base, x_goal_specific], dim=1)
        x = self.merge_layer(merged)

        #identity = x
        #x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        #x = torch.cat([x, identity], dim=1)

        #identity = x
        #x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        #x = torch.cat([x, identity], dim=1)



        #graph_embedding_mean = global_mean_pool(x, batch)
        graph_embedding_add = global_max_pool(x_goal_specific, batch)
        graph_embedding = graph_embedding_add
        graph_embedding = torch.cat([graph_embedding, goal], dim=-1)

        graph_embedding = F.relu(self.graph_embedder(graph_embedding))
        graph_embedding = F.relu(self.graph_embedder_1(graph_embedding))
        graph_embedding = self.graph_embedder_2(graph_embedding)
        
        # Extract subgraph embeddings using the adjusted indices
        subgraph_embeddings = x_goal_specific[visited_subgraph]


        # Compute the global mean for each subgraph
        #subgraph_embedding_mean = global_mean_pool(subgraph_embeddings, subgraph_node_indices_batch)
        subgraph_embedding_add = global_max_pool(subgraph_embeddings, subgraph_node_indices_batch)
        subgraph_embedding = subgraph_embedding_add
        subgraph_embedding = torch.cat([subgraph_embedding, goal], dim=-1)

        subgraph_embedding = F.relu(self.subgraph_embedder(subgraph_embedding))
        subgraph_embedding = F.relu(self.subgraph_embedder_1(subgraph_embedding))
        subgraph_embedding = self.subgraph_embedder_2(subgraph_embedding)

        #concat the graph level embedding with each node of the corresponding graph with batch

        #we have one current node for each graph (per batch)
        conc_state = torch.cat([x, graph_embedding[batch], subgraph_embedding[batch], current_node_feature[batch], goal[batch]], dim=-1)
        #concatenate the current node with the conc_state, select the appropriate current node for each graph
        

        conc_state = self.conv_whole_graph(conc_state, edge_index, edge_attr)

        #state_embedding = torch.zeros_like(conc_state[:,:10])  # Adjust size accordingly
        """
        # Iterate over each goal-specific branch
        for i, branch in enumerate(self.state_goal_branches):
            mask = goal_indices == i
            if mask.any():
                state_embedding[mask] = branch(conc_state[mask])
        """
        # Compute Q-values for each goal
        qvals = torch.zeros(x.shape[0]).to(x.get_device())
        for i, branch in enumerate(self.qvalue_goal_branches):
            mask = goal_indices == i
            if mask.any():
                qvals[mask] = branch(conc_state[mask]).squeeze()


        # Apply action mask if provided
        if action_mask is not None:
            action_mask = action_mask.to(qvals.device)

            #action_mask = action_mask.unsqueeze(1)
            # Set Q-values of valid actions (where action_mask is 1) as is, and others to -inf should be on dimension 0
            qvals = torch.where(action_mask == 1, qvals, torch.tensor(-1e+8).to(qvals.device))
        
        #check if qvals contains nan
        if torch.isnan(qvals).any():
            raise ValueError("Qvals contains nan")

        return qvals



# -------------------------
# AGENT DEFINITION
# -------------------------
class Agent:
    def __init__(self, state_size, edge_attr_size, seed, device, lr, buffer_size, batch_size, gamma, tau, update_every):
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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
        self.lr = lr


        # Q-Network
        self.qnetwork_local = GraphQNetwork(state_size, self.edge_attr_size, seed).to(device)
        self.qnetwork_target = GraphQNetwork(state_size, self.edge_attr_size,seed).to(device)

        class_name = self.qnetwork_local.__class__.__name__
        self.file_name = f"VACTION_SPACE_GOAL_{class_name}_{date}"
        name = f"runs/{self.file_name}"
        self.writer = SummaryWriter(name)  # Choose an appropriate experiment name
        #init the weights of the target network to be the same as the local network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())



        self.optimizer = Adam(self.qnetwork_local.parameters(), lr=lr)

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "next_action_mask", "goal", "visited_subgraph", "next_visited_subgraph", "current_node_id", "next_current_node_id"])
        
        self.buffer = Memory(capacity=buffer_size)
        
        self.t_step = 0
        self.tau_step = 0

        self.losses = []
        self.steps = 0



    def add_experience(self, state, action, reward, next_state, done, next_action_mask, goal, visited_subgraph, next_visited_subgraph, current_node_id, next_current_node_id):        
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done, next_action_mask, goal, visited_subgraph, next_visited_subgraph, current_node_id, next_current_node_id)
        #check if there is None in experience
        for key, value in experience._asdict().items():
            if value is None:
                raise ValueError(f"Value of {key} is None")
        self.buffer.store(experience)



    def log_environment_change(self, env_name):
        self.writer.add_text('Environment Change', f'Changed to {env_name}', self.steps)


    def log_loss(self, loss):
        self.writer.add_scalar('Loss', loss, len(self.losses))

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.steps)



    def step(self, state, action, reward, next_state, done, next_action_mask, goal, visited_subgraph, next_visited_subgraph, current_node_id, next_current_node_id):
        self.steps += 1
        self.tau_step += 1
        #ensure everything is on device
        state = state
        next_state = next_state
        
        # Save experience in replay memory
        self.add_experience(state, action, reward, next_state, done, next_action_mask, goal, visited_subgraph, next_visited_subgraph, current_node_id, next_current_node_id)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                indices, experiences, is_weights = self.buffer.sample(self.batch_size)
                self.learn(experiences, indices, is_weights,  self.gamma)
        self.tau_step = (self.tau_step + 1) % self.tau
        if self.tau_step == 0:
            #do a hard update of the target network
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())


    def get_qvalues(self, state, action_mask, goal, visited_subgraph, current_node):
        state = state
        action_mask = action_mask.to(self.device)
        goal = goal.to(self.device)
        goal = goal.unsqueeze(0)

        self.qnetwork_local.eval()
        x = state.x.to(self.device)
        edge_index = state.edge_index.to(self.device)
        edge_attr = state.edge_attr.to(self.device)
        self.qnetwork_local.eval()
        current_node = torch.tensor([current_node], dtype=torch.long).to(self.device)


        with torch.no_grad():
            action_values = self.qnetwork_local(x, edge_index, edge_attr, None, action_mask, goal, visited_subgraph, None, current_node)

        return_values = action_values.cpu()
        self.qnetwork_local.train()
        return return_values

    def act(self, state, action_mask, goal, visited_subgraph, current_node, eps=0, ):
        state = state
        action_mask = action_mask.to(self.device)
        goal = goal.to(self.device)
        goal = goal.unsqueeze(0)

        if random.random() > eps:
            current_node = torch.tensor([current_node], dtype=torch.long).to(self.device)

            self.qnetwork_local.eval()
            x = state.x.to(self.device)
            edge_index = state.edge_index.to(self.device)
            edge_attr = state.edge_attr.to(self.device)

            with torch.no_grad():  # Wrap in no_grad
                action_values = self.qnetwork_local(x, edge_index, edge_attr, None, action_mask, goal, visited_subgraph, None, current_node)
            return_values = action_values.cpu()
            self.qnetwork_local.train()

            selected_action = torch.argmax(return_values).item()
            torch.cuda.empty_cache()

            return selected_action
        else:
            choices = (action_mask.cpu() == 1).nonzero(as_tuple=True)[0]

            return random.choice(choices).item()
    
    def save_checkpoint(self, nb_training):
        if self.losses == []:
            return
        # Create directory if it does not exist
        directory = f"models/rl/{self.file_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # File name for saving the checkpoint
        name = f"{directory}/{nb_training}_{self.losses[-1]:.2f}.pt"

        # Save the model
        torch.save(self.qnetwork_local.state_dict(), name)

    def load_checkpoint(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = Adam(self.qnetwork_local.parameters(), lr=self.lr)

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
            next_mask = e.next_action_mask
            is_weight = torch.tensor(is_weights[i][0])
            goal = e.goal
            visited_subgraph = e.visited_subgraph
            next_visited_subgraph = e.next_visited_subgraph

            current_node_id = torch.tensor(e.current_node_id, dtype=torch.long)
            next_current_node_id = torch.tensor(e.next_current_node_id, dtype=torch.long)

            data = MyGraphData(x_s=state.x, edge_index_s=state.edge_index, edge_attr_s=state.edge_attr,
                            action=action, next_current_nd = next_current_node_id, current_nd = current_node_id, reward=reward, x_t=next_state.x,
                            edge_index_t=next_state.edge_index, edge_attr_t=next_state.edge_attr,
                            done=done, exp_idx=exp_index, next_mask = next_mask, is_weight=is_weight,
                            goal = goal, visited_subgraph = visited_subgraph, next_visited_subgraph = next_visited_subgraph
                            )
            
            


            data_list.append(data)


        # Create a DataLoader for batching
        batch_size = self.batch_size
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['x_s', 'x_t', 'visited_subgraph', 'next_visited_subgraph'], pin_memory=True)
        device = self.device
        for batch in data_loader:

            batch.to(device)  # Send the batch to the GPU if available
            b_exp_indices = batch.exp_idx
            # Extract batched action, reward, done, and action_mask
            b_action = batch.action.to(device)
            b_reward = batch.reward.to(device)
            b_done = batch.done.to(device)
            b_next_action_mask = batch.next_mask.to(device)


            b_x_s = batch.x_s.to(device)
            b_edge_index_s = batch.edge_index_s.to(device)
            b_edge_attr_s = batch.edge_attr_s.to(device)
            b_x_t = batch.x_t.to(device)
            b_edge_index_t = batch.edge_index_t.to(device)
            b_edge_attr_t = batch.edge_attr_t.to(device)

            x_s_batch = batch.x_s_batch.to(device)
            x_t_batch = batch.x_t_batch.to(device)
            b_is_weight = batch.is_weight.to(device)
            b_goal = batch.goal.to(device)
            b_visited_subgraph = batch.visited_subgraph.to(device)
            b_visited_subgraph_batch = batch.visited_subgraph_batch.to(device)

            b_next_visited_subgraph = batch.next_visited_subgraph.to(device)
            b_next_visited_subgraph_batch = batch.next_visited_subgraph_batch.to(device)

            b_current_node = batch.current_nd.to(device)
            b_next_current_node = batch.next_current_nd.to(device)


            # DDQN Update for the individual graph
            self.qnetwork_target.eval()
            self.qnetwork_local.train()
            # Calculate Q values for next states
            with torch.no_grad():
                
                #Double DQN
                q_local_next = self.qnetwork_local(b_x_t, 
                                                   b_edge_index_t, 
                                                   b_edge_attr_t, 
                                                   x_t_batch, 
                                                   action_mask=b_next_action_mask, goal=b_goal, 
                                                   visited_subgraph=b_next_visited_subgraph, 
                                                   subgraph_node_indices_batch=b_next_visited_subgraph_batch,
                                                   current_node = b_next_current_node,
                                                   )
                #get the indices actions with the highest q values, for each graph, using x_t_batch, since the dimension of q_local_next is [num_actions]
                q_local_max, q_indices = scatter_max(q_local_next, x_t_batch, dim=0)

                Q_targets_next = self.qnetwork_target(b_x_t, b_edge_index_t, 
                                                    b_edge_attr_t, x_t_batch, 
                                                    action_mask=b_next_action_mask,
                                                    goal=b_goal,
                                                    visited_subgraph=b_next_visited_subgraph,
                                                    subgraph_node_indices_batch=b_next_visited_subgraph_batch,
                                                    current_node = b_next_current_node
                                                    )
                
                
            q_indices = q_indices.to(Q_targets_next.get_device())
            Q_targets_next = Q_targets_next.gather(0, q_indices).detach().squeeze()

            Q_targets = b_reward + (gamma * Q_targets_next * (1 - b_done))
            Q_expected_result = self.qnetwork_local(b_x_s, b_edge_index_s, b_edge_attr_s, x_s_batch, None, goal = b_goal, visited_subgraph=b_visited_subgraph, subgraph_node_indices_batch=b_visited_subgraph_batch, current_node = b_current_node).squeeze()


            # Now gather the Q values
            Q_expected = Q_expected_result.gather(0, b_action)

            
            # Compute loss

            td_error = torch.abs(Q_expected - Q_targets)
            loss = torch.mean(td_error.pow(2))
            self.log_loss(loss.item())
            

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            # Soft update target network


            #self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

            # Update priorities, losses for logging, etc.
            td_error = td_error.detach().cpu().numpy()
            self.buffer.batch_update(b_exp_indices, td_error)

            self.losses.append(loss.item())



    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

