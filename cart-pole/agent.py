import numpy as np
import torch.nn as nn
import random
import kerasncp as kncp
from kerasncp.torch import LTCCell
import torch
import torch.utils.data as data

class NCPNetwork(nn.Module):
    def __init__(
        self,
        ncp_cell
    ):
        super(NCPNetwork, self).__init__()
        self.ncp_cell = ncp_cell

    def forward(self, x):

        device = x.device
        batch_size = x.size(0)
        hidden_state = torch.zeros(
            (batch_size, self.ncp_cell.state_size), device=device
        )

        output, _ = self.ncp_cell(x, hidden_state)
        return output

class DQNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(DQNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        return self.layers(x)

class Agent:
    def __init__(self, in_features, n_actions, device, config_obj):

        self.in_features = in_features
        self.n_actions = n_actions

        wiring = kncp.wirings.NCP(
            inter_neurons=config_obj['inter_neurons'],  # Number of inter neurons
            command_neurons=config_obj['command_neurons'],  # Number of command neurons
            motor_neurons=n_actions,  # Number of motor neurons
            sensory_fanout=config_obj['sensory_fanout'],  # How many outgoing synapses has each sensory neuron
            inter_fanout=config_obj['inter_fanout'],  # How many outgoing synapses has each inter neuron
            recurrent_command_synapses=config_obj['recurrent_command_synapses'],  # Now many recurrent synapses are in the
            # command neuron layer
            motor_fanin=config_obj['motor_fanin'],  # How many incoming synapses has each motor neuron
        )
        ncp_cell = LTCCell(wiring, in_features)
        ncp_cell.to(device)

        self.model = DQNetwork(in_features, n_actions)
        #self.model = NCPNetwork(ncp_cell)
        self.model.to(device)
        self.target_model = DQNetwork(in_features, n_actions)
        #self.target_model = NCPNetwork(ncp_cell)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_obj['lr'])

        self.update_counter = 0
        self.update_every = 5

        self.discount_factor = config_obj['discount_factor']
        self.epsilon = 1
        self.epsilon_decay = config_obj['epsilon_decay']
        self.min_epsilon = 0.01

        self.device = device

    def get_action(self, state, eval=False):
        if eval or random.random() > self.epsilon:
            with torch.no_grad():
                return torch.argmax(self.model(torch.tensor([state]).float().to(self.device))).item()
        else:
            return random.randrange(0, self.n_actions);

    def train(self, transitions):

        state_batch = torch.tensor([t.state for t in transitions]).float().to(self.device)
        new_state_batch = torch.tensor([t.new_state for t in transitions]).float().to(self.device)
        reward_batch = torch.tensor([t.reward for t in transitions]).to(self.device)
        done_batch = torch.tensor([t.done for t in transitions]).to(self.device)
        action_batch = [t.action for t in transitions]

        q_vals = self.model(state_batch)[np.arange(len(transitions)), action_batch]
        q_next = self.target_model(new_state_batch)
        q_next[done_batch] = 0.0
        
        q_targets = reward_batch + self.discount_factor * torch.max(q_next, dim=1)[0]

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_vals, q_targets)
        loss.backward()
        self.optimizer.step()

        #self.model.ncp_cell.apply_weight_constraints()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        self.update_counter += 1
        if self.update_counter >= self.update_every:
            self.update_counter = 0
            self.target_model.load_state_dict(self.model.state_dict())
