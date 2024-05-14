import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#CNN as the base learner
class BaseLearner(nn.Module):
    def __init__(self):
        super(BaseLearner, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Meta Learner (MAML)

