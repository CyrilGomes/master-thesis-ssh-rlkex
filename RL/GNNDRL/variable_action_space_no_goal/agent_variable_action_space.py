



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
        self.conv1 = GATv2Conv(num_node_features, num_node_features, edge_dim=num_edge_features, dropout=0.6, heads=1)
        self.norm1 = GraphNorm(num_node_features)
        self.conv2 = GATv2Conv(num_node_features, num_node_features, edge_dim=num_edge_features, dropout=0.6, heads=1)
        self.norm2 = GraphNorm(num_node_features)
        self.conv3 = GATv2Conv(num_node_features, num_node_features, edge_dim=num_edge_features,dropout=0.6, heads=1)
        self.norm3 = GraphNorm(num_node_features)


        #graph embedder
        self.graph_embedder = torch.nn.Linear(num_node_features, 5)

        #subgraph embedder
        self.subgraph_embedder = torch.nn.Linear(num_node_features, 5)

        # Define dropout
        self.dropout = torch.nn.Dropout(p=0.5)

        # Dueling DQN layers
        self.qvalue_stream = torch.nn.Linear(21, 10)
        self.qvalue = torch.nn.Linear(10, 1)





    def forward(self, x, edge_index, edge_attr , action_node_idx, goal, visited_subgraph_nodes):  

        
        #ensure everything is on device
        x = F.dropout(x, p=0.6, training=self.training)
        edge_index = edge_index
        edge_attr = edge_attr


        #Graph embedding

        identity = x

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = x + identity

        identity = x
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = x + identity

        identity = x
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = x + identity



        x = self.graph_embedder(x)
        x = F.elu(x)
        graph_embedding = global_mean_pool(x, batch=None)
        
        graph_embedding = graph_embedding.squeeze()

        #graph_embedding is not dependent on the action

        #get the node embeddings of the visited subgraph
        #the visited_subgraph_nodes tensor contains the ids of the nodes in the visited subgraph, that correspond to the main graph
        #we need to get the embeddings of these nodes from the main graph x

        visited_nodes_embeddings = x[visited_subgraph_nodes]
        visited_nodes_embeddings = global_mean_pool(visited_nodes_embeddings, batch=None)
        #visited_nodes_embeddings = self.subgraph_embedder(visited_nodes_embeddings)
        visited_nodes_embeddings = visited_nodes_embeddings.squeeze()

        action_node_embeddings = x[action_node_idx]

        #Now we concatenate the state embedding with the action node embedding
        x = torch.cat((graph_embedding, action_node_embeddings, visited_nodes_embeddings, goal), dim=0)
        #here the shape of x is [num_graphs, embedding_size + embedding_size]
        x = F.elu(self.qvalue_stream(x))
        qvals = self.qvalue(x)

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

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "next_action_space", "goal", "visited_subgraph"])
        
        self.buffer = Memory(capacity=buffer_size)
        
        self.t_step = 0

        self.losses = []
        self.steps = 0



    def add_experience(self, state, action, reward, next_state, done, next_action_space, goal, visited_subgraph):        
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done, next_action_space, goal, visited_subgraph)
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
            if key == 'Reward' and value > 0:
                print(f"Reward: {value}")
                
            self.writer.add_scalar(key, value, self.steps)



    def step(self, state, action, reward, next_state, done, next_action_space, goal, visited_subgraph):
        self.steps += 1
        #ensure everything is on device
        state = state
        next_state = next_state
        
        # Save experience in replay memory
        self.add_experience(state, action, reward, next_state, done, next_action_space, goal, visited_subgraph)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                indices, experiences, is_weights = self.buffer.sample(self.batch_size)
                self.learn(experiences, indices, is_weights,  self.gamma)

    def act(self, state, action_space, goal, visited_subgraph, eps=0):
        state = state
        goal = goal.to(self.device)
        visited_subgraph = visited_subgraph.to(self.device)

        if random.random() > eps:
            self.qnetwork_local.eval()
            x = state.x.to(self.device)
            edge_index = state.edge_index.to(self.device)
            edge_attr = state.edge_attr.to(self.device)

            action_qvalue_map = {}
            with torch.no_grad():  # Wrap in no_grad
                for action_node_idx in action_space:
                    torch.Tensor(action_node_idx).to(self.device)
                    next_qvalue = self.qnetwork_local(x, edge_index, edge_attr, action_node_idx, goal, visited_subgraph)
                    action_qvalue_map[action_node_idx] = next_qvalue.item()

            #get action with max qvalue
            max_action = max(action_qvalue_map, key=action_qvalue_map.get)
            max_qvalue = action_qvalue_map[max_action]

            selected_action = max_action
            torch.cuda.empty_cache()

            self.qnetwork_local.train()
            return selected_action
        else :
            
            return random.choice(action_space)
    
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

        
        for i, e in enumerate(experiences):
            state = e.state.to(self.device)
            action = torch.tensor(e.action, dtype=torch.long).to(self.device)
            reward = e.reward
            next_state = e.next_state.to(self.device)
            done = torch.tensor(e.done, dtype=torch.uint8).to(self.device)
            next_action_space = torch.tensor(e.next_action_space, dtype=torch.long).to(self.device)
            is_weight = torch.tensor(is_weights[i][0])
            exp_index = indices[i]
            goal = e.goal.to(self.device)
            visited_subgraph = e.visited_subgraph.to(self.device)


            # DDQN Update for the individual graph
            self.qnetwork_target.eval()
            self.qnetwork_local.train()
            if done:
                Q_targets = reward
            else :
                # Calculate Q values for each action in next state
                action_qvalue_map = {}
                with torch.no_grad():
                    for action_node_idx in next_action_space:
                        torch.Tensor(action_node_idx).to(self.device)
                        next_visited_subgraph = torch.cat((visited_subgraph, action_node_idx.unsqueeze(0)), dim=0)
                        next_qvalue = self.qnetwork_target(next_state.x, next_state.edge_index, next_state.edge_attr, action_node_idx, goal, next_visited_subgraph)
                        action_qvalue_map[action_node_idx.item()] = next_qvalue.item()

                #compute the target qvalue
                #get the max action with the max qvalue, along with the qvalue
                max_action = max(action_qvalue_map, key=action_qvalue_map.get)
                max_qvalue = action_qvalue_map[max_action]

                #target qvalue is the reward + gamma * max_qvalue with the done flag
                Q_targets = reward + gamma * max_qvalue * (1 - done)
                
            #Compute the expected qvalue
            #get the qvalue of the action taken
            Q_expected = self.qnetwork_local(state.x, state.edge_index, state.edge_attr, action, goal, visited_subgraph)

            # Compute loss


            td_error = torch.abs(Q_expected - Q_targets)
            loss = torch.mean(td_error.pow(2))
            self.log_loss(loss.item())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Soft update target network

            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

            # Update priorities, losses for logging, etc.
            td_error = td_error.detach().cpu().numpy()
            self.buffer.single_update(exp_index, td_error)

            self.losses.append(loss.item())



    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

