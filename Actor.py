import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc1=256, fc2=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.mu = nn.Linear(fc2, action_size)

        self.max_action = max_action
    
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        computed_value = torch.tanh(self.mu(x))  # "tanh"(sigmoid which outputs values between -1 and 1); chosen by trail and error
        return self.max_action * computed_value
