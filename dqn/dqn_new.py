#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)

        return (torch.stack(states),
                torch.tensor(actions),
                torch.tensor(rewards),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.bool))
        # END STUDENT SOLUTION


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(transition)
        # END STUDENT SOLUTION



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device
        self.c = 0
        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay_buffer = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)
        self.q_net = q_net_init().to(device)
        self.target = q_net_init().to(device)
        self.target.load_state_dict(self.q_net.state_dict())
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        if stochastic and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                return self.q_net(state).argmax().item()
        # END STUDENT SOLUTION


    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        with torch.no_grad():
            next_q_values = self.target(next_states).max(1)[0]
            y = torch.where(dones,
                          rewards,
                          rewards + self.gamma * next_q_values)
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q_values, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.c += 1
        if self.c % self.target_update == 0:
            self.target.load_state_dict(self.q_net.state_dict())



    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        if init_buffer:
            state, _ = env.reset()
            state = torch.FloatTensor(state).to(self.device)
            for _ in range(self.burn_in):
                action = random.randrange(self.action_size)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = torch.FloatTensor(next_state).to(self.device)
                self.replay_buffer.append((state, action, reward, next_state, done or truncated))

                if done or truncated:
                    state, _ = env.reset()
                    state = torch.FloatTensor(state).to(self.device)
                else:
                    state = next_state
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).to(self.device)
            episode_reward = 0

            for t in range(max_steps):
                action = self.get_action(state, train)

                next_state, reward, done, truncated, _ = env.step(action)
                next_state = torch.FloatTensor(next_state).to(self.device)
                episode_reward += reward

                if train:
                    self.replay_buffer.append((state, action, reward, next_state, done or truncated))
                    self.train()

                if done or truncated:
                    break

                state = next_state
            total_rewards.append(episode_reward)
        # END STUDENT SOLUTION
        return total_rewards



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    all_rewards = []
    graph_every = 100

    for agent in agents:
        rewards = agent.run(env, max_steps, num_episodes, train=True, init_buffer=True)
        all_rewards.append(rewards)

    all_rewards = np.array(all_rewards)
    average_total_rewards = []
    min_total_rewards = []
    max_total_rewards = []

    for i in range(0, num_episodes, graph_every):
        end_idx = min(i + graph_every, num_episodes)
        rewards_slice = all_rewards[:, i:end_idx]
        avg_rewards = np.mean(np.mean(rewards_slice, axis=1))
        min_rewards = np.min(np.mean(rewards_slice, axis=1))
        max_rewards = np.max(np.mean(rewards_slice, axis=1))

        average_total_rewards.append(avg_rewards)
        min_total_rewards.append(min_rewards)
        max_total_rewards.append(max_rewards)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agents for multiple runs
    agents = [
        DeepQNetwork(
            state_size=state_size,
            action_size=action_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ) for _ in range(args.num_runs)
    ]
    graph_agents(f"DQN_{args.env_name}", agents, env, args.max_steps, args.num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
