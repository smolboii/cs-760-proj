import gym
import torch
from transition import Transition
from agent import Agent
from collections import deque
import numpy as np
import torch.nn as nn
import random
import kerasncp as kncp
from kerasncp.torch import LTCCell
from agent import NCPNetwork
import torch
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, '\n')

wiring = kncp.wirings.NCP(
    inter_neurons=20,  # Number of inter neurons
    command_neurons=10,  # Number of command neurons
    motor_neurons=1,  # Number of motor neurons
    sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
    inter_fanout=5,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=0,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=4,  # How many incoming synapses has each motor neuron
)
# wiring = kncp.wirings.FullyConnected(
#     64, n_actions
# )
ncp_cell = LTCCell(wiring, 2)
ncp_cell.to(device)

model = NCPNetwork(ncp_cell)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

epochs = 10
epoch_size = 100
for epoch in range(1, epochs+1):

    xs = []
    ys = []
    for _ in range(epoch_size):
        x = [random.random() > 0.5, random.random() > 0.5]
        y = x[0] and x[1]
        xs.append(x)
        ys.append(y)

    cum_loss = 0
    for i, x in enumerate(xs):
        x = torch.tensor([x]).to(device)
        y = model(x)

        optimizer.zero_grad()
        loss = loss_fn(y, torch.tensor([[ys[i]]]).float().to(device))
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()

        model.ncp_cell.apply_weight_constraints()

    print(f'epoch #{epoch} avg loss: {cum_loss / epoch_size}')