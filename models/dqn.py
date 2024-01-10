# DQN model used by ai player in 4 in a row game

import torch.nn as nn

class Dqn(nn.Module):
    def __init__(self,input_shape=(6,7),
                        n_actions=7,
                        in_channels=1,
                        n_channels=3):
        super(Dqn, self).__init__()

        self.simpleNet = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, input_shape),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_shape[0] - 2) * (input_shape[1] - 2) * n_channels,
                        n_actions),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.simpleNet(x)
