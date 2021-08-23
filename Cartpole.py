import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2),
            nn.Softmax()
        )

    def forward(self, X):
        probs = self.network(X)
        return probs


def reinforce():
    policy = Policy()
    policy = policy.double().to(device)

    env = gym.make("CartPole-v0")

    total_episodes = 1000
    learning_rate = 5e-3
    gamma = 1

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    timesteps_survived = np.zeros(total_episodes)  # in each episode

    for i_episode in range(total_episodes):
        observation = env.reset()
        done = False

        log_prob_sum = 0
        rewards = []
        while not done:
            observation = torch.tensor(observation, dtype=torch.double, device=device)
            timesteps_survived[i_episode] += 1
            probs = policy(observation)

            m = torch.distributions.categorical.Categorical(probs)
            action = m.sample().item()

            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            log_prob_sum += torch.log(probs[action])

        discounts = [gamma ** i for i in range(1, len(rewards) + 1)]

        R = sum([a * b for a, b in zip(discounts, rewards)])

        loss = -1 * log_prob_sum * R
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(timesteps_survived)


reinforce()