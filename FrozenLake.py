import gym
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
from torch import nn

"""
class FrozenLakeAgent:

    def __init__(self, epsilon, step_size, gamma):
        self.q = np.zeros((16, 4))  # 12 states, 4 actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.gamma = gamma

    def learn(self):
        env = gym.make("FrozenLake-v0")
        total_episodes = 500000

        games_won_every_100_episodes = []
        games_won = 0

        for i_episode in range(total_episodes):
            s = env.reset()
            done = False  # Episode status
            while not done:

                if random.uniform(0, 1) < self.epsilon*(1 - (i_episode/total_episodes)):
                    action = env.action_space.sample()

                else:
                    action = np.argmax(self.q[s])

                s_prime, reward, done, info = env.step(action)

                if reward == 1:
                    games_won += 1

                if reward == 0 and done == True:
                    reward = -1

                reward -= 0.05
                self.q[s, action] += self.step_size * (reward + self.gamma*np.max(self.q[s_prime]) - self.q[s, action])
                s = s_prime

            if (i_episode + 1) % 100 == 0:
                games_won_every_100_episodes.append(games_won)
                games_won = 0

        return games_won_every_100_episodes
"""


class Network(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentDeepQ:

    def __init__(self, epsilon, gamma, input_dims, n_actions):

        self.deepQNetwork = Network(input_dims, n_actions).to(device)  # 16 states, 4 actions
        self.epsilon = epsilon
        self.gamma = gamma

    def learn(self):
        env = gym.make("FrozenLake-v0")
        total_episodes = 1000

        games_won_every_100_episodes = []
        games_won = 0

        optimizer = torch.optim.SGD(self.deepQNetwork.parameters(), lr=1e-3)

        for i_episode in range(total_episodes):

            position = env.reset()
            s = torch.zeros((16,), device=device)
            s[position] = 1

            done = False  # Episode status
            while not done:

                q = self.deepQNetwork(s)

                if random.uniform(0, 1) < self.epsilon * (1 - (i_episode / total_episodes)):
                    action = env.action_space.sample()
                    q_for_selected_action = q[action]

                else:
                    q_for_selected_action = torch.max(q)
                    action = torch.argmax(q).item()

                position, reward, done, info = env.step(action)

                s_prime = torch.zeros((16,), device=device)
                s_prime[position] = 1

                q_prime = self.deepQNetwork(s_prime)
                q_prime_max = torch.max(q_prime).item()

                s = s_prime

                if reward == 1:
                    games_won += 1

                if reward == 0 and done == True:
                    reward = -1

                reward -= 0.05

                loss = (q_for_selected_action - reward - (self.gamma * q_prime_max)) ** 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (i_episode + 1) % 100 == 0:
                print("Episode: ", i_episode)
                games_won_every_100_episodes.append(games_won)
                games_won = 0

        return games_won_every_100_episodes


def main():
    agent = AgentDeepQ(epsilon=0.1, gamma=0.9, input_dims=16, n_actions=4)
    games_won_every_100_episodes = agent.learn()
    plt.plot(games_won_every_100_episodes)
    plt.show()


if __name__ == '__main__':
    main()
