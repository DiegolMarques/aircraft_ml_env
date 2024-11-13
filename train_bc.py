import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Tuple, List
import matplotlib.pyplot as plt
import glob

class ExpertDataset(Dataset):
    def __init__(self, demonstrations: List[str]):
        self.observations = []
        self.actions = []
        total_timesteps = 0
        successful_trajectories = 0
        
        print("\nLoading demonstration files...")
        for demo_file in demonstrations:
            print(f"Loading {demo_file}")
            with open(demo_file, 'rb') as f:
                demo_data = pickle.load(f)
                self.observations.append(demo_data['observations'])
                self.actions.append(demo_data['actions'])
                successful_trajectories += 1
                total_timesteps += len(demo_data['observations'])
        
        if not self.observations:
            raise ValueError("No demonstrations found!")
        
        print(f"\nDataset Statistics:")
        print(f"Loaded trajectories: {successful_trajectories}/{len(demonstrations)}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Average trajectory length: {total_timesteps/successful_trajectories:.1f} steps")
        
        # Concatenate data
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        
        # Print data statistics
        print("\nData Statistics:")
        print(f"Action range: [{np.min(self.actions):.3f}, {np.max(self.actions):.3f}]")
        print(f"Position range x: [{np.min(self.observations[:, 0]):.1f}, {np.max(self.observations[:, 0]):.1f}]")
        print(f"Position range y: [{np.min(self.observations[:, 1]):.1f}, {np.max(self.observations[:, 1]):.1f}]")
        print(f"Yaw range: [{np.min(self.observations[:, 2]):.3f}, {np.max(self.observations[:, 2]):.3f}]")
        
        # Normalize different components separately
        # Position normalization (-1000 to 1000 for x,y)
        self.pos_mean = torch.FloatTensor([0.0, 0.0])
        self.pos_std = torch.FloatTensor([1000.0, 1000.0])
        
        # Yaw normalization (-pi to pi)
        self.yaw_mean = torch.FloatTensor([0.0])
        self.yaw_std = torch.FloatTensor([np.pi])
        
        # Sensor readings normalization (0 to 500)
        sensor_data = self.observations[:, 3:8]  # 5 sensor readings
        self.sensor_mean = torch.FloatTensor(np.mean(sensor_data, axis=0))
        self.sensor_std = torch.FloatTensor(np.clip(np.std(sensor_data, axis=0), 1e-2, None))
        
        # Boundary distances normalization (0 to 2000)
        boundary_data = self.observations[:, 8:12]  # 4 boundary distances
        self.boundary_mean = torch.FloatTensor(np.mean(boundary_data, axis=0))
        self.boundary_std = torch.FloatTensor(np.clip(np.std(boundary_data, axis=0), 1e-2, None))
        
        # Action normalization
        self.action_mean = torch.FloatTensor(np.mean(self.actions, axis=0))
        self.action_std = torch.FloatTensor(np.clip(np.std(self.actions, axis=0), 1e-2, None))
        
        # Normalize observations
        normalized_obs = np.zeros_like(self.observations)
        normalized_obs[:, :2] = (self.observations[:, :2] - self.pos_mean.numpy()) / self.pos_std.numpy()
        normalized_obs[:, 2] = (self.observations[:, 2] - self.yaw_mean.numpy()) / self.yaw_std.numpy()
        normalized_obs[:, 3:8] = (self.observations[:, 3:8] - self.sensor_mean.numpy()) / self.sensor_std.numpy()
        normalized_obs[:, 8:12] = (self.observations[:, 8:12] - self.boundary_mean.numpy()) / self.boundary_std.numpy()
        
        # Convert to tensors
        self.observations = torch.FloatTensor(normalized_obs)
        self.actions = torch.FloatTensor(self.actions)
        
        print(f"\nFinal dataset size: {len(self.observations)} samples")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

class BCPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Main feature encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Specialized encoders for different state components
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 8),
            nn.ReLU()
        )
        
        self.sensor_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 8),
            nn.ReLU()
        )
        
        self.boundary_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 8),
            nn.ReLU()
        )
        
        # Calculate total features dimension
        total_features = hidden_dim + (hidden_dim // 8) * 3  # main features + specialized features
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, x):
        # Split state into components
        pos = x[:, :2]
        sensors = x[:, 3:8]
        boundaries = x[:, 8:12]
        
        # Encode each component
        state_features = self.state_encoder(x)
        pos_features = self.pos_encoder(pos)
        sensor_features = self.sensor_encoder(sensors)
        boundary_features = self.boundary_encoder(boundaries)
        
        # Concatenate features
        combined = torch.cat([
            state_features, 
            pos_features,
            sensor_features, 
            boundary_features
        ], dim=1)
        
        return self.policy_net(combined)
    
    def predict(self, state):
        self.eval()
        with torch.no_grad():
            # Normalize state components
            normalized_state = np.zeros_like(state)
            normalized_state[:2] = (state[:2] - self.pos_mean.numpy()) / self.pos_std.numpy()
            normalized_state[2] = (state[2] - self.yaw_mean.numpy()) / self.yaw_std.numpy()
            normalized_state[3:8] = (state[3:8] - self.sensor_mean.numpy()) / self.sensor_std.numpy()
            normalized_state[8:12] = (state[8:12] - self.boundary_mean.numpy()) / self.boundary_std.numpy()
            
            state_tensor = torch.FloatTensor(normalized_state)
            action = self.forward(state_tensor.unsqueeze(0))
            return action.squeeze(0).numpy()

def save_policy(policy: BCPolicy, filename: str):
    """Save trained policy with normalization parameters and model config"""
    save_dict = {
        'network_state_dict': policy.state_dict(),
        'hidden_dim': policy.state_encoder[0].out_features,  # Save hidden dimension
        'pos_mean': policy.pos_mean,
        'pos_std': policy.pos_std,
        'yaw_mean': policy.yaw_mean,
        'yaw_std': policy.yaw_std,
        'sensor_mean': policy.sensor_mean,
        'sensor_std': policy.sensor_std,
        'boundary_mean': policy.boundary_mean,
        'boundary_std': policy.boundary_std,
        'action_mean': policy.action_mean,
        'action_std': policy.action_std
    }
    torch.save(save_dict, f"{filename}.pth")

def load_policy(filename: str, state_dim: int) -> BCPolicy:
    """Load trained policy with normalization parameters"""
    checkpoint = torch.load(f"{filename}.pth")
    
    # Create policy with same hidden dimension as saved model
    hidden_dim = checkpoint.get('hidden_dim', 512)  # Default to 512 if not found
    policy = BCPolicy(state_dim, hidden_dim)
    
    # Load state dict
    policy_state_dict = {}
    for key, value in checkpoint['network_state_dict'].items():
        if not key.startswith(('pos_mean', 'pos_std', 'yaw_mean', 'yaw_std', 
                             'sensor_mean', 'sensor_std', 'boundary_mean', 
                             'boundary_std', 'action_mean', 'action_std')):
            policy_state_dict[key] = value
    
    policy.load_state_dict(policy_state_dict, strict=True)
    
    # Load normalization parameters
    for param_name in ['pos_mean', 'pos_std', 'yaw_mean', 'yaw_std', 
                      'sensor_mean', 'sensor_std', 'boundary_mean', 
                      'boundary_std', 'action_mean', 'action_std']:
        policy.register_buffer(param_name, checkpoint[param_name])
    
    policy.eval()
    return policy

def train_bc(demonstrations: List[str], 
             epochs: int = 1000,
             batch_size: int = 256,
             hidden_dim: int = 512,  # Make sure this matches
             learning_rate: float = 1e-4,
             weight_decay: float = 1e-4):
    
    # Create dataset
    dataset = ExpertDataset(demonstrations)
    
    # Split into train/val sets (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Create policy with consistent hidden dimensions
    state_dim = dataset.observations[0].shape[0]
    policy = BCPolicy(state_dim, hidden_dim)
    print(f"Created policy with hidden_dim={hidden_dim}")
    
    # Copy normalization parameters from dataset
    for param_name in ['pos_mean', 'pos_std', 'yaw_mean', 'yaw_std', 
                      'sensor_mean', 'sensor_std', 'boundary_mean', 
                      'boundary_std', 'action_mean', 'action_std']:
        policy.register_buffer(param_name, getattr(dataset, param_name))
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=True
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=False
    )
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 0
    max_patience = 50
    
    print("\nStarting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    try:
        for epoch in range(epochs):
            # Training phase
            policy.train()
            train_loss = 0.0
            for obs, actions in train_loader:
                optimizer.zero_grad()
                pred_actions = policy(obs)
                loss = criterion(pred_actions, actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            policy.eval()
            val_loss = 0.0
            val_errors = []
            
            with torch.no_grad():
                for obs, actions in val_loader:
                    pred_actions = policy(obs)
                    val_loss += criterion(pred_actions, actions).item()
                    errors = torch.abs(pred_actions - actions)
                    val_errors.extend(errors.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_policy(policy, 'bc_policy_best')
                patience = 0
            else:
                patience += 1
            
            # Logging
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                
                error_stats = np.percentile(val_errors, [25, 50, 75])
                print(f"Validation Error Quartiles: {error_stats[0]:.4f}, {error_stats[1]:.4f}, {error_stats[2]:.4f}")
            
            # Early stopping
            if patience >= max_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Final visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(val_errors, bins=50, density=True, alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Density')
        plt.title('Final Error Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print("Training curves saved to 'training_curves.png'")
        
        return load_policy('bc_policy_best', state_dim)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if best_val_loss != float('inf'):
            print("Loading best model...")
            return load_policy('bc_policy_best', state_dim)
        raise
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    try:
        # Find all demonstration files
        demo_files = glob.glob("expert_demo_*.pkl")
        
        if not demo_files:
            print("No demonstration files found!")
            print("Please record some demonstrations first.")
            print("Press 'R' to start/stop recording in manual mode")
            exit(1)
        
        print(f"Found {len(demo_files)} demonstration files")
        
        # Train BC policy
        policy = train_bc(
            demonstrations=demo_files,
            epochs=1000,
            batch_size=256,
            hidden_dim=512,
            learning_rate=1e-4,
            weight_decay=1e-4
        )
        
        print("\nTraining completed successfully!")
        print("- Best model saved as 'bc_policy_best.pth'")
        print("- Training curves saved as 'training_curves.png'")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise