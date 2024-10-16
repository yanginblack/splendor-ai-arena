import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSplendorNNet(nn.Module):
    def __init__(self, game, args):
        super(SimSplendorNNet, self).__init__()
        self.channels, self.rows, self.cols = game.getStateSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.fc1 = nn.Linear(31, 128)
        self.fc2 = nn.Linear(128, 128)

        # define output layers
        self.q_values = nn.Linear(128, self.action_size)

    # x: batch_size x channels x rows x cols
    def forward(self, x):
        # flatten the input
        x = x.view(-1, 31)

        # Pass through hidden layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Policy and value heads
        q_values = F.softmax(self.q_values(x), dim=1)

        return q_values
