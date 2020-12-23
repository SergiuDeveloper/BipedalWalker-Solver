from TD3 import Agent

import gym

import numpy as np

env_id = 'BipedalWalker-v3'
env = gym.make(env_id)

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

policy = Agent(state_space, action_space, max_action, policy_noise=0)  # Set noise to 0(no exploring needed)
policy.load('20')

env.reset()

done = False
state = np.array([0. for _ in range(24)])
while not done:
    action = policy.select_action(state)

    state, reward, done, _ = env.step(action)
    env.render()

env.close()
