import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=256):  # fc1, fc2 = layers sizes
        super(Critic, self).__init__()

        # Primary network
        self.l1 = nn.Linear(state_size + action_size, fc1)  # Performs a linear transformation
        # (chosen based on trail and error)
        self.l2 = nn.Linear(fc1, fc2)
        self.l3 = nn.Linear(fc2, 1)

        # Twin network
        self.l4 = nn.Linear(state_size + action_size, fc1)
        self.l5 = nn.Linear(fc1, fc2)
        self.l6 = nn.Linear(fc2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Resize the [state, action] tensor, by the 1st dimension

        # Primary network
        q1 = F.relu(self.l1(x))  # ReLU overcomes the vanishing gradient issue; Uses strictly positive values
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Twin network
        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
