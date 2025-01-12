import itertools
import random
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from replay_memory import ReplayMemory
import flappy_bird_gymnasium
import os

from dqn import DQN


class Agent:

    def __init__(self):
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        self.epsilon_init = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.network_sync_rate = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = 0.005
        self.discount_factor_g = 0.99
        self.best_reward_stop = 1000

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        self.MODEL_FILE = os.path.join(os.path.dirname(__file__), "flappy_model.pth")
        self.GRAPH_FILE = os.path.join(os.path.dirname(__file__), 'flappy_model.png')


    def run(self, training=False, render=True):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar = False)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        epsilon_history = []
        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions).to(self.device)

        if training:
            print("Device: ", self.device)
            start_time = datetime.now()
            last_graph_update_time = start_time

            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init

            step_count = 0
            target_dqn = DQN(num_states, num_actions).to(self.device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
            self.optimizer = torch.optim.RMSprop(policy_dqn.parameters(), lr=self.learning_rate)

            best_reward = -np.inf
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in tqdm(itertools.count(), desc="Episodes", unit="episode"):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

            done = False
            episode_reward = 0.0

            while not done:
                if training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()

                next_state, reward, done, _, info = env.step(action)
                episode_reward += reward

                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

                if training:
                    memory.append((state, action, reward, next_state, done))

                    step_count += 1

                state = next_state

            if training:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"Episode: {episode}, Best reward: {best_reward}, Best score: {info.get('score')} Epsilon: {epsilon}")

                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                epsilon_history.append(epsilon)

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=5):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.minibatch_size:
                    mini_batch = memory.sample(self.minibatch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                if episode_reward > self.best_reward_stop:
                    print(f"Best reward reached: {episode_reward}")
                    break
            else:
                if info.get('score') > 100:
                    print(f"Score: {info.get('score')} Reward: {episode_reward}")
                # break

            rewards_per_episode.append(episode_reward)

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        q_values = policy_dqn(states)
        q_values = q_values.gather(1, actions.unsqueeze(dim=1)).squeeze()

        with torch.no_grad():
            target_q_values = target_dqn(next_states)
            target_q_values = rewards + self.discount_factor_g * target_q_values.max(dim=1).values * ~dones

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def play(self):
        self.run(training=False, render=True)

    def train(self):
        self.run(training=True, render=False)

    def test(self):
        self.run(training=False, render=False)


if __name__ == '__main__':
    game = Agent()

    # game.train()
    # game.play()
    # game.test()