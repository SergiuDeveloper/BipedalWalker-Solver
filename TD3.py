"""
TD3 = Twin Delayed DDPG

Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of
the two Q-values to form the targets in the Bellman error loss functions.

Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function.

Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit
Q-function errors by smoothing out Q along changes in action.
"""

"""
(as per keras.io)
DDPG = Deep Deterministic Policy Gradient

Actor - It proposes an action given a state.
Critic - It predicts if the action is good (positive value) or bad (negative value) given a state and an action.

Trick One: Uses two target networks. Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better", as opposed to saying "I'm going to re-learn how to
play this entire game after every move".

Trick Two: Uses experience replay(instead of learning only from recent experience, it learns from sampling all of its 
past experiences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from Actor import Actor
from Critic import Critic
from utils import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(
            self,
            state_size,
            action_size,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        super(Agent, self).__init__()

        self.actor = Actor(state_size, action_size, max_action).to(device)
        self.actor.apply(self.init_weights)
        self.actor_target = copy.deepcopy(self.actor)  # Actor time-delayed network
        # (network that explores ahead of the learned network, in order to improve stability)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)  # More "efficient" than SGD

        self.critic = Critic(state_size, action_size).to(device)
        self.critic.apply(self.init_weights)
        self.critic_target = copy.deepcopy(self.critic)  # Critic time-delayed network
        # (network that explores ahead of the learned network, in order to improve stability)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)  # More "efficient" than SGD

        self.max_action = max_action  # Action numeric limits
        self.discount = discount  # (= Gamma) In order not to seek instant gratification(gamma=1 -> don't take past
        # actions into consideration at all; gamma=0 -> consider all of the past actions)
        self.tau = tau  # Update coefficient for the weights(used to update the weights slowly)
        self.policy_noise = policy_noise  # Policy noise value
        self.noise_clip = noise_clip  # Noise numeric limits
        self.policy_freq = policy_freq  # Calculate the loss and update the weights after X iterations
        self.total_it = 0  # Total number of iterations

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)  # Values in normal distribution
            layer.bias.data.fill_(0.01)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()  # Returns an action suggested by the actor
        # (takes noise into consideration(self.policy_noise)

    def train(self, replay_buffer: ReplayBuffer):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample()

        with torch.no_grad():  # Don't compute gradient automatically during the training initialization phase
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)  # Clamp the noise

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)  # Select the next actions and apply noise for exploring

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)  # Target critic network Q func & twin
            target_Q = torch.min(target_Q1, target_Q2)  # Pick the best one
            target_Q = reward + (1 - done) * self.discount * target_Q  # Recompute after applying discount

        current_Q1, current_Q2 = self.critic(state, action)  # Compute the critic Q func & twin

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()  # Reset gradients
        critic_loss.backward()  # Compute gradients
        self.critic_optimizer.step()  # Updates the network

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic(state, self.actor(state))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # Update critic target network weights ever %policy_freq% iterations

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # Update actor target network weights ever %policy_freq% iterations

    def save(self, filename):
        torch.save(self.critic.state_dict(), 'models/checkpoint/' + filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), 'models/checkpoint/' + filename + '_critic_optimizer')

        torch.save(self.actor.state_dict(), 'models/checkpoint/' + filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), 'models/checkpoint/' + filename + '_actor_optimizer')

    def load(self, filename):
        self.critic.load_state_dict(torch.load('models/checkpoint/' + filename + '_critic', map_location='cpu'))
        self.critic_optimizer.load_state_dict(
            torch.load('models/checkpoint/' + filename + '_critic_optimizer', map_location='cpu'))  # optional
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('models/checkpoint/' + filename + '_actor', map_location='cpu'))
        self.actor_optimizer.load_state_dict(
            torch.load('models/checkpoint/' + filename + '_actor_optimizer', map_location='cpu'))  # optional
        self.actor_target = copy.deepcopy(self.actor)
