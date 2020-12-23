from TD3 import Agent

import gym

import numpy as np
import matplotlib.pyplot as plt

import utils

env_id = 'BipedalWalker-v3'
env = gym.make(env_id)

exploration_noise = 0.1  # Noise used in order to explore new actions even after learning certain patterns

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = Agent(state_space, action_space, max_action)

try:
    policy.load('final')
except:
    pass

replay_buffer = utils.ReplayBuffer()  # In order to take into consideration the previous experience of the agent

max_episodes = 2000
max_timesteps = 2000

ep_reward = []

for episode in range(1, max_episodes + 1):
    avg_reward = 0
    state = env.reset()
    for t in range(1, max_timesteps + 1):
        action = policy.select_action(state) + np.random.normal(0, max_action * exploration_noise, size=action_space)
        # Select an action sample from the action space, based on noise
        action = action.clip(env.action_space.low, env.action_space.high)  # Limits the action value

        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        avg_reward += reward

        if len(replay_buffer) > 100:
            policy.train(replay_buffer)

        if done or t >= max_timesteps:
            print(f'Episode {episode} reward: {avg_reward} | Rolling average: {np.mean(ep_reward)}')
            print(f'Current time step: {t}')

            ep_reward.append(avg_reward)
            break

    if np.mean(ep_reward[-10:]) >= 300:
        policy.save('final')
        break

    if episode % 100 == 0 and episode > 0:
        policy.save(str('%02d' % (episode // 100)))

env.close()

# Plot the training score per each episode
fig = plt.figure()
plt.plot(np.arange(1, len(ep_reward) + 1), ep_reward)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
