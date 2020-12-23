import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc1=256, fc2=256):  # Layers sizes
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1)  # Layer 1
        self.fc2 = nn.Linear(fc1, fc2)  # Layer 2
        self.mu = nn.Linear(fc2, action_size)  # Out layer

        self.max_action = max_action
    
    def forward(self, state):
        x = self.fc1(state)  # Layer 1
        x = F.relu(x)
        x = self.fc2(x)  # Layer 2
        x = F.relu(x)  # Layer 3

        computed_value = torch.tanh(self.mu(x))  # "tanh"(sigmoid which outputs values between -1 and 1); chosen by trail and error
        return self.max_action * computed_value
