import numpy as np
import torch.nn as nn
import random
import kerasncp as kncp
import time
from kerasncp.torch import LTCCell
from collections import deque
import torch
import torch.utils.data as data

class NCPNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        n_actions,
        config_obj,
        device
    ):
        super(NCPNetwork, self).__init__()

        wiring = kncp.wirings.NCP(
            inter_neurons=config_obj['inter_neurons'],  # Number of inter neurons
            command_neurons=config_obj['command_neurons'],  # Number of command neurons
            motor_neurons=n_actions,  # Number of motor neurons
            sensory_fanout=config_obj['sensory_fanout'],  # How many outgoing synapses has each sensory neuron
            inter_fanout=config_obj['inter_fanout'],  # How many outgoing synapses has each inter neuron
            recurrent_command_synapses=config_obj['recurrent_command_synapses'],  # How many recurrent synapses are in the
            # command neuron layer
            motor_fanin=config_obj['motor_fanin'],  # How many incoming synapses has each motor neuron
            seed=int(time.time()) if config_obj['random_seed'] else 22222 # seed for wiring configuration (22222 is paper repo default)
        )
        ncp_cell = LTCCell(wiring, in_features)
        ncp_cell.to(device)

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
    def __init__(self, in_features, n_actions):
        super(DQNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, n_actions)
        )

    def forward(self, x):
        return self.layers(x)

class Agent:
    def __init__(self, in_features, n_actions, device, config_obj):

        self.n_actions = n_actions

        self.model = NCPNetwork(in_features, n_actions, config_obj, device)
        #self.model = DQNetwork(in_features, n_actions)
        self.model.to(device)

        self.target_model = NCPNetwork(in_features, n_actions, config_obj, device)
        #self.target_model = DQNetwork(in_features, n_actions)
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

        self.min_mem = config_obj['min_mem']
        self.replay_mem = deque(maxlen=config_obj['max_mem'])
        self.batch_size = config_obj['batch_size']

        self.device = device

    def get_action(self, state, train=True):
        if not train or random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor([state]).to(self.device)
                actions = self.model(state.float())
                return torch.argmax(actions).item()        
        else:
            return random.randrange(0, self.n_actions)

    
    def update_mem(self, transition):
        self.replay_mem.append(transition)

    def train(self):

        if len(self.replay_mem) < self.min_mem:
            return

        self.optimizer.zero_grad()

        batch_size = min(self.batch_size, len(self.replay_mem))
        batch_index = np.random.choice(len(self.replay_mem), batch_size, replace=False)
        batch = [self.replay_mem[i] for i in batch_index]

        state_batch = torch.tensor([t.state for t in batch]).float().to(self.device)
        new_state_batch = torch.tensor([t.new_state for t in batch]).float().to(self.device)
        reward_batch = torch.tensor([t.reward for t in batch]).to(self.device)
        done_batch = torch.tensor([t.done for t in batch]).to(self.device)
        action_batch = [t.action for t in batch]

        q_vals = self.model(state_batch)[np.arange(batch_size), action_batch]
        q_next = self.target_model(new_state_batch)
        q_next[done_batch] = 0.0
        
        q_targets = reward_batch + self.discount_factor * torch.max(q_next, dim=1)[0]

        loss = self.loss_fn(q_vals.float(), q_targets.float())
        loss.backward()
        self.optimizer.step()

        self.model.ncp_cell.apply_weight_constraints()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        self.update_counter += 1
        if self.update_counter >= self.update_every:
            self.update_counter = 0
            self.target_model.load_state_dict(self.model.state_dict())