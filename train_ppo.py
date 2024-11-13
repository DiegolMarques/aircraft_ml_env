import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from typing import Tuple, List, Dict
from train_bc import load_policy
from collections import deque
import time
from aircraft_env import AircraftEnv

class PPOPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Main encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable std
        self.action_log_std = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.state_encoder(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.state_encoder(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        
        if deterministic:
            return action_mean, value
        
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        return action, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.state_encoder(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        
        action_log_probs = dist.log_prob(action).squeeze()
        dist_entropy = dist.entropy().squeeze()
        
        return action_log_probs, value, dist_entropy

class PPOTrainer:
    def __init__(
        self,
        env,
        bc_policy_path: str = None,
        hidden_dim: int = 512,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        batch_size: int = 64,
        n_epochs: int = 10,
        max_grad_norm: float = 0.5
    ):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        
        # Initialize policy
        self.policy = PPOPolicy(self.state_dim, hidden_dim)
        
        if bc_policy_path is not None:
            self._initialize_from_bc(bc_policy_path)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
    
    def _initialize_from_bc(self, bc_path: str):
        print(f"Loading BC policy from {bc_path}")
        bc_policy = load_policy(bc_path, self.state_dim)
        
        with torch.no_grad():
            # Copy encoder weights
            self.policy.state_encoder[0].weight.copy_(bc_policy.state_encoder[0].weight)
            self.policy.state_encoder[0].bias.copy_(bc_policy.state_encoder[0].bias)
            
            # Copy normalization parameters
            for param_name in ['pos_mean', 'pos_std', 'yaw_mean', 'yaw_std', 
                             'sensor_mean', 'sensor_std', 'boundary_mean', 
                             'boundary_std', 'action_mean', 'action_std']:
                self.policy.register_buffer(param_name, getattr(bc_policy, param_name))
        
        print("Initialized from BC policy")
    
    def collect_rollout(self, n_steps: int):
        states, actions, rewards, values = [], [], [], []
        log_probs, dones = [], []
        
        state, _ = self.env.reset()
        
        for step in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action and value
            with torch.no_grad():
                action, value = self.policy.act(state_tensor)
                action_np = action.squeeze().numpy()
                if isinstance(action_np, np.ndarray):
                    action_np = action_np.item()
            
            # Take environment step
            next_state, reward, done, _, _ = self.env.step(np.array([action_np]))
            
            # Store samples
            states.append(state_tensor)
            actions.append(action)
            rewards.append(torch.FloatTensor([reward]))
            values.append(value)
            
            # Get log prob
            with torch.no_grad():
                action_std = torch.exp(self.policy.action_log_std)
                dist = Normal(action, action_std)
                log_prob = dist.log_prob(action).squeeze()
            
            log_probs.append(log_prob)
            dones.append(torch.FloatTensor([done]))
            
            state = next_state
            if done:
                state, _ = self.env.reset()
        
        # Convert to tensors
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        dones = torch.cat(dones)
        
        return states, actions, rewards, values, log_probs, dones
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   dones: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values
        return advantages, returns

    def train_epoch(self, states: torch.Tensor, actions: torch.Tensor, 
                   old_log_probs: torch.Tensor, advantages: torch.Tensor, 
                   returns: torch.Tensor) -> Dict[str, float]:
        indices = np.random.permutation(len(states))
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0
        
        for start_idx in range(0, len(states), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            state_batch = states[batch_indices]
            action_batch = actions[batch_indices]
            old_log_prob_batch = old_log_probs[batch_indices]
            advantage_batch = advantages[batch_indices]
            return_batch = returns[batch_indices]
            
            # Get current policy predictions
            new_log_probs, values, entropy = self.policy.evaluate(state_batch, action_batch)
            
            # Ensure correct shapes
            values = values.squeeze()
            advantage_batch = advantage_batch.squeeze()
            return_batch = return_batch.squeeze()
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantage_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * (return_batch - values).pow(2).mean()
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_batches += 1
        
        return {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches
        }
    
    def train(self, total_timesteps: int):
        timesteps_per_batch = 2048
        total_updates = total_timesteps // timesteps_per_batch
        
        best_reward = float('-inf')
        episode_rewards = deque(maxlen=100)
        total_timesteps_done = 0
        start_time = time.time()
        
        print(f"Total updates to perform: {total_updates}")
        print(f"Timesteps per update: {timesteps_per_batch}")
        print(f"Total timesteps: {total_timesteps}\n")
        
        for update in range(total_updates):
            # Collect rollout
            states, actions, rewards, values, log_probs, dones = self.collect_rollout(timesteps_per_batch)
            total_timesteps_done += timesteps_per_batch
            
            # Calculate advantages
            advantages, returns = self.compute_gae(rewards, values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Train for multiple epochs
            for _ in range(self.n_epochs):
                metrics = self.train_epoch(states, actions, log_probs, advantages, returns)
            
            # Log metrics
            episode_reward = rewards.sum().item()
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards)
            
            # Save if best
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(self.policy.state_dict(), 'ppo_policy_best.pth')
                print(f"\n✨ New best model saved! Mean reward: {best_reward:.2f}")
            
            # Progress update
            elapsed_time = time.time() - start_time
            updates_per_sec = (update + 1) / elapsed_time
            remaining_updates = total_updates - (update + 1)
            estimated_time = remaining_updates / updates_per_sec if updates_per_sec > 0 else 0
            
            print(f"\rUpdate {update + 1}/{total_updates} | "
                  f"Timesteps: {total_timesteps_done}/{total_timesteps} | "
                  f"Mean reward: {mean_reward:.2f} | "
                  f"Policy loss: {metrics['policy_loss']:.4f} | "
                  f"Elapsed: {elapsed_time:.1f}s | "
                  f"ETA: {estimated_time:.1f}s", end="")
            
            # Detailed update every 10 iterations
            if (update + 1) % 10 == 0:
                print(f"\n\nDetailed metrics at update {update + 1}:")
                print(f"Value Loss: {metrics['value_loss']:.4f}")
                print(f"Entropy: {metrics['entropy']:.4f}")
                print(f"Recent rewards: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
                print("")

if __name__ == "__main__":
    # Create environment without rendering for faster training
    env = AircraftEnv(render_mode=None, mode="ppo")
    
    trainer = PPOTrainer(
        env=env,
        bc_policy_path="bc_policy_best",
        hidden_dim=512,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        c1=1.0,
        c2=0.01,
        batch_size=64,
        n_epochs=10,
        max_grad_norm=0.5
    )
    
    print("\nStarting PPO training...")
    print("Training without visualization for speed")
    print("Press Ctrl+C to stop training\n")
    
    try:
        trainer.train(total_timesteps=1_000_000)
        print("\nTraining completed!")
        print("Best model saved as 'ppo_policy_best.pth'")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        env.close()