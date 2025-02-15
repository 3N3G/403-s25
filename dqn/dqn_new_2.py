#! python3

import argparse
import random
import math
from collections import namedtuple, deque


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory = deque([], maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        return random.sample(self.memory, self.batch_size)
        # END STUDENT SOLUTION


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(Transition(*transition))
        # END STUDENT SOLUTION

    def __len__(self):
      return len(self.memory)
    

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

        self.replay_buffer_batch_size = replay_buffer_batch_size

        self.EPS_START = 0.9
        self.EPS_END = self.epsilon
        self.EPS_DECAY = 1000
        self.steps_done = 0
        self.TAU = 0.005

        hidden_layer_size = 128

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.memory = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        self.q_net = q_net_init()
        self.target = q_net_init()
        self.target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr_q_net, amsgrad=True)
        self.q_net = self.q_net.to(self.device)
        self.target = self.target.to(self.device)
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if not stochastic or sample > eps_threshold:
          with torch.no_grad():
            return self.q_net(state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[random.choice([0, 1])]], device=self.device, dtype=torch.long)
        # END STUDENT SOLUTION


    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        transitions = self.memory.sample_batch()
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.q_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.replay_buffer_batch_size, device=self.device)
        
        with torch.no_grad():
          next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()
        # END STUDENT SOLUTION

    def test(self, env, max_steps=200, num_episodes=20):
        total_rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            total_reward = 0
            for t in count():
                action = self.get_action(state, False)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                state = next_state
                if done:
                    total_rewards.append(total_reward)
                    break
        return sum(total_rewards) / len(total_rewards)


    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        if init_buffer:
          while(len(self.memory) <= self.burn_in):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
              action = torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
              observation, reward, terminated, truncated, _ = env.step(action.item())
              reward = torch.tensor([reward], device=self.device)
              done = terminated or truncated
              next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
              self.memory.append((state, action, next_state, reward))
              state = next_state
              if done:
                break
        if train:
          for i_episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            total_reward = 0
            for t in count():
              action = self.get_action(state, True)
              self.steps_done += 1
              observation, reward, terminated, truncated, _ = env.step(action.item())
              reward = torch.tensor([reward], device=self.device)
              done = terminated or truncated
              total_reward += reward
              next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
              self.memory.append((state, action, next_state, reward))
              state = next_state
              self.train()

              if True or (t+1) % self.target_update == 0:
                q_net_state_dict = self.q_net.state_dict()
                target_state_dict = self.target.state_dict()
                for key in q_net_state_dict:
                  target_state_dict[key] = q_net_state_dict[key]*self.TAU + target_state_dict[key]*(1-self.TAU)
                self.target.load_state_dict(target_state_dict)
            
              if done:
                break
              
            if (i_episode+1)%100 == 0:
              total_rewards.append(self.test(env))
        
        print('Complete')
        # END STUDENT SOLUTION
        return(total_rewards)



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    all_rewards = []
    graph_every = 100
    for agent in agents:
      total_rewards = agent.run(env, max_steps, num_episodes, True, True)
      all_rewards.append(total_rewards)
    
    all_rewards = np.array(all_rewards)
    print("all_rewards ", all_rewards)
    average_total_rewards = []
    min_total_rewards = []
    max_total_rewards = []

    for i in range(0, int(num_episodes/graph_every)):
        rewards_slice = all_rewards[:, i]
        avg_rewards = np.mean(rewards_slice)
        min_rewards = np.min(rewards_slice)
        max_rewards = np.max(rewards_slice)

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
    parser.add_argument('--lr_q_net', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--replay_buffer_batch_size', type=int, default=128, help='Replay Buffer Batch Size')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name,max_episode_steps=args.max_steps)
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)
    device = torch.device(
          "cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu"
    )
    print("device: ",device)
    agents = [DeepQNetwork(state_size=n_observations, action_size=n_actions, device=device, lr_q_net=args.lr_q_net, replay_buffer_batch_size=args.replay_buffer_batch_size) for _ in range(0, args.num_runs)]
    graph_agents("2.2", agents, env, args.max_steps, args.num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()



# get action
# run