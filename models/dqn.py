# DQN model used by ai player in 4 in a row game

import torch.nn as nn
import torch
import numpy as np

class Dqn(nn.Module):
    def __init__(self, input_shape=(6,7),
                        n_actions=7,
                        in_channels=1,
                        n_channels=12):
        super(Dqn, self).__init__()

        # Define some layers
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                      out_channels=n_channels,
                      kernel_size=(3, 3))
        self.linear1 = nn.Linear((input_shape[0] - 2) * (input_shape[1] - 2) * n_channels,
                        n_actions)

        
    def simpleNet(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = torch.flatten(x, start_dim=0, end_dim=-1)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        return x

    
    def forward(self, x):
        return self.simpleNet(x)
