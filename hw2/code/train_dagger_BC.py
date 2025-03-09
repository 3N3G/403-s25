import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from modules import PolicyNet, ValueNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
from PIL import Image
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

class TrainDaggerBC:

    def __init__(self, env, model, expert_model, optimizer, states, actions, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        model.set_device(self.device)
        self.expert_model = self.expert_model.to(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, len(trajectory_mask)))
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
        """

        states, old_actions, timesteps, rewards = [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        new_states = []
        new_actions = []
        new_timesteps = []
        rewards_all = []
        for _ in range(num_trajectories_per_batch_collection):
            states, _, timesteps, rewards = self.generate_trajectory(self.env, self.model)
            rewards_all.append(sum(rewards))
            expert_actions = []
            for state in states:
                expert_actions.append(self.call_expert_policy(state))
            new_states.extend(states)
            new_actions.extend(expert_actions)
            new_timesteps.extend(timesteps)
        
        if self.states is None:
            self.states = np.array(new_states)
            self.actions = np.array(new_actions)
            self.timesteps = np.array(new_timesteps)
        else:
            self.states = np.concatenate([self.states, np.array(new_states)], axis=0)
            self.actions = np.concatenate([self.actions, np.array(new_actions)], axis=0)
            self.timesteps = np.concatenate([self.timesteps,  np.array(new_timesteps)], axis=0)
        # END STUDENT SOLUTION

        return rewards_all

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        rewards = []
        # BEGIN STUDENT SOLUTION
        for _ in range(num_trajectories_per_batch_collection):
            _, _, _, total_reward = self.generate_trajectory(self.env, self.model)
            rewards.append(sum(total_reward))
        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps, 
        num_BC_training_steps=20000,
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=500, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_BC_training_steps: the number of iterations to train the model using BC.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        # BEGIN STUDENT SOLUTION

        losses = []
        rewards_list = []

        if self.mode == 'BC':
            for step in range(num_BC_training_steps):
                loss = self.training_step(batch_size)
                losses.append(loss)
            
                if (step+1) % print_every == 0:
                    print(f"Step {step+1}/{num_BC_training_steps} - Loss: {loss:.4f}")
                
                if (step+1) % 1000 == 0:
                    rewards = self.generate_trajectories(num_trajectories_per_batch_collection)
                    avg_reward = np.mean(rewards)
                    median_reward = np.median(rewards)
                    max_reward = np.max(rewards)
                    rewards_list.append((step+1, avg_reward, median_reward, max_reward))
                    print(f"At step {step+1}: Avg={avg_reward:.2f} Median={median_reward:.2f} Max={max_reward:.2f}")
                
                if (step+1) % save_every == 0 and step > 0:
                    torch.save(self.model.state_dict(), f"bc_model_step_{step+1}.pt")
        elif self.mode == 'DAgger':
            for batch in range(num_batch_collection_steps):
                rewards = self.update_training_data(num_trajectories_per_batch_collection)
                for step in range(num_training_steps_per_batch_collection):
                    loss = self.training_step(batch_size)
                    losses.append(loss)
                    if (step+1) % print_every == 0:
                        print(f"Batch {batch+1}, Step {step+1}/{num_training_steps_per_batch_collection} - Loss: {loss:.4f}")
                avg_reward = np.mean(rewards)
                median_reward = np.median(rewards)
                max_reward = np.max(rewards)
                rewards_list.append((batch+1, avg_reward, median_reward, max_reward))
                print(f"At batch {batch+1}: Avg={avg_reward:.2f} Median={median_reward:.2f} Max={max_reward:.2f}")
                
                if (batch+1) % 5 == 0 and batch > 0:
                    torch.save(self.model.state_dict(), f"dagger_model_step_{batch+1}.pt")

        # END STUDENT SOLUTION

        return losses, rewards_list

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
            
        
        return states, actions, timesteps

def save_losses_plot(losses, save_path="training_loss.png"):
    # Save the loss plot
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Behavior Cloning Training Loss Curve")
    plt.legend()
    plt.savefig(save_path)  # Save as a PNG file
    plt.close()

def save_trainig_reward_plot(rewards_list, save_path="training_rewards_plot.png"):
    steps = [entry[0] for entry in rewards_list]
    avg_rewards = [entry[1] for entry in rewards_list]
    median_rewards = [entry[2] for entry in rewards_list]
    max_rewards = [entry[3] for entry in rewards_list]
    plt.figure()
    plt.plot(steps, avg_rewards, label="Average Reward", marker="o")
    plt.plot(steps, median_rewards, label="Median Reward", marker="s")
    plt.plot(steps, max_rewards, label="Max Reward", marker="^")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.title("Training Reward Progression")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    
def generate_failure_gif(model, env, save_path="gifs_imitation.gif"):
    obs, _ = env.reset()
    frames = []
    total_reward = 0
    done = False
    timestep = 0
    while not done:
        frame = env.render()
        frames.append(Image.fromarray(frame))
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        timestep_tensor = torch.tensor([timestep], dtype=torch.long, device=model.device)
        action = model(obs_tensor, timestep_tensor).detach().cpu().numpy().squeeze()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        timestep += 1
    
    if total_reward < 0 and len(frames) > 1:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"Failure run GIF saved at: {save_path}")
    else:
        print("No failure run detected. GIF not saved")


def generate_success_gif(model, env, save_path="gifs_imitation.gif"):
    obs, _ = env.reset()
    frames = []
    total_reward = 0
    done = False
    timestep = 0
    while not done:
        frame = env.render()
        frames.append(Image.fromarray(frame))
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        timestep_tensor = torch.tensor([timestep], dtype=torch.long, device=model.device)
        action = model(obs_tensor, timestep_tensor).detach().cpu().numpy().squeeze()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        timestep += 1
    
    if total_reward > 260 and len(frames) > 1:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"Success run GIF saved at: {save_path}")
    else:
        print(f"total_reward: {total_reward}, frames: {len(frames)}")
        print("No success run detected. GIF not saved")

def run_training():
    """
    Simple Run Training Function
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print("device: ", device)

    env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
    mode="DAgger"

    model_name = "super_expert_PPO_model"
    expert_model = PolicyNet(24, 4)
    model_weights = torch.load(f"data/models/{model_name}.pt", map_location=device)
    expert_model.load_state_dict(model_weights["PolicyNet"])

    states_path = "data/states_BC.pkl"
    actions_path = "data/actions_BC.pkl"

    with open(states_path, "rb") as f:
        states = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    # BEGIN STUDENT SOLUTION
    
    model = SimpleNet(
        state_dim=24,
        action_dim=4,
        hidden_layer_dimension=128,
        max_episode_length=1600,
        device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr= 0.0001, weight_decay= 0.0001)
    if mode == "BC":
        trainer = TrainDaggerBC(env=env, model=model, expert_model=expert_model, optimizer=optimizer, states=states, actions=actions, device=device, mode=mode)
        losses, rewards_list = trainer.train(num_batch_collection_steps=0)
    elif mode == "DAgger":
        trainer = TrainDaggerBC(env=env, model=model, expert_model=expert_model, optimizer=optimizer, states=states, actions=actions, device=device, mode=mode)
        losses, rewards_list = trainer.train(num_batch_collection_steps=20, batch_size=128)

    save_losses_plot(losses)
    save_trainig_reward_plot(rewards_list)

    if mode == "BC":
        generate_failure_gif(model, env, save_path="gifs_imitation.gif")
    elif mode == "DAgger":
        generate_success_gif(model, env, save_path="gifs_DAgger.gif")
    # END STUDENT SOLUTION

if __name__ == "__main__":
    run_training()