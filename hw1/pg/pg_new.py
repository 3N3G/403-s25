#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class PolicyGradient(nn.Module):
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, mode='REINFORCE', n=128, gamma=0.99, device='cpu'):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.Softmax(dim=-1)
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, 1)
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.to(device)
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.actor(state), self.critic(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        action_probs = self.actor(state)
        if stochastic:
            action = torch.distributions.Categorical(action_probs).sample()
        else:
            action = torch.argmax(action_probs)
        return action.item()
        # END STUDENT SOLUTION


    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        T = len(rewards_tensor)
        returns = torch.zeros_like(rewards_tensor)

        if self.mode == 'A2C':

            for t in range(T):
                end_t = min(t + self.n, T)

                V_end = values[t + self.n] if t + self.n < T else 0

                G_t = 0
                for k in range(t, end_t):
                    G_t += (self.gamma ** (k - t)) * rewards_tensor[k]

                G_t += (self.gamma ** self.n) * V_end

                returns[t] = G_t
        else:
            G_t = 0
            for t in reversed(range(T)):
                G_t = rewards_tensor[t] + self.gamma * G_t
                returns[t] = G_t

        return returns
        # END STUDENT SOLUTION


    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        action_probs = self.actor(states)
        selected_probs = action_probs[range(len(actions)), actions]
        values = self.critic(states).squeeze() if self.mode != 'REINFORCE' else None

        if self.mode == 'REINFORCE':

            returns = self.calculate_n_step_bootstrap(rewards, None)
            actor_loss = -torch.mean(returns * torch.log(selected_probs))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            return actor_loss.item()

        elif self.mode == 'REINFORCE_WITH_BASELINE':

            returns = self.calculate_n_step_bootstrap(rewards, None)

        else:

            returns = self.calculate_n_step_bootstrap(rewards, values)
        advantages = returns - values.detach()

        actor_loss = -torch.mean(advantages * torch.log(selected_probs))
        critic_loss = torch.mean((returns - values) ** 2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
        # END STUDENT SOLUTION


    def run(self, env, max_steps, num_episodes, train):

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        total_rewards = []

        for episode in range(num_episodes):
            states, actions, rewards = [], [], []
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = self.get_action(state, stochastic=train)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if train:
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                state = next_state

                if terminated or truncated:
                    break

            total_rewards.append(episode_reward)

            if train:
                self.train(states, actions, rewards)

        return total_rewards
        # END STUDENT SOLUTION



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    graph_every = 100
    num_test_episodes = 20
    num_trials = len(agents)

    D = torch.zeros(num_trials, num_episodes // graph_every)

    for trial in range(num_trials):
        agent = agents[trial]

        for episode in range(0, num_episodes, graph_every):

            agent.run(env, max_steps, graph_every, train=True)

            test_rewards = agent.run(env, max_steps, num_test_episodes, train=False)
            D[trial, episode // graph_every] = sum(test_rewards) / num_test_episodes

    average_total_rewards = torch.mean(D, dim=0).numpy()
    min_total_rewards = torch.min(D, dim=0)[0].numpy()
    max_total_rewards = torch.max(D, dim=0)[0].numpy()
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
    mode_choices = ['REINFORCE', 'REINFORCE_WITH_BASELINE', 'A2C']

    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--mode', type=str, default='REINFORCE', choices=mode_choices, help='Mode to run the agent in')
    parser.add_argument('--n', type=int, default=64, help='The n to use for n step A2C')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=3500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = [
        PolicyGradient(
            state_size=state_size,
            action_size=action_size,
            mode=args.mode, # default 'REINFORCE'
            n=args.n,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        for _ in range(args.num_runs)
    ]

    graph_name = f'{args.env_name}_{args.mode}_n{args.n}'
    graph_agents(graph_name, agents, env, args.max_steps, args.num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
