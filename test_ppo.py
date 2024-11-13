import torch
import numpy as np
from aircraft_env import AircraftEnv
from ppo import PPOPolicy
import json
import time

def test_ppo_policy(
    model_path: str = "ppo_policy_best.pth",
    episodes: int = 20,
    render_delay: float = 0.01
):
    """Test PPO policy"""
    # Create environment
    env = AircraftEnv(render_mode="human", mode="ppo")
    
    # Create and load policy
    policy = PPOPolicy(
        state_dim=env.observation_space.shape[0],
        hidden_dim=512,
        action_std=0.1  # Lower for deterministic behavior
    )
    
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    # Testing metrics
    results = {
        'successes': 0,
        'collisions': 0,
        'timeouts': 0,
        'total_reward': 0,
        'total_steps': 0,
        'episode_rewards': [],
        'episode_steps': [],
        'episode_outcomes': []
    }
    
    try:
        for episode in range(episodes):
            print(f"\nStarting Episode {episode + 1}/{episodes}")
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                if render_delay > 0:
                    time.sleep(render_delay)
                
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, _ = policy.act(state_tensor, deterministic=True)
                
                # Take step in environment
                state, reward, done, _, info = env.step(action.numpy())
                episode_reward += reward
                steps += 1
            
            # Update statistics
            results['total_steps'] += steps
            results['total_reward'] += episode_reward
            results['episode_rewards'].append(float(episode_reward))
            results['episode_steps'].append(steps)
            
            # Determine episode outcome
            if info['success_count'] > 0:
                outcome = 'success'
                results['successes'] += 1
            elif info['collision_count'] > 0:
                outcome = 'collision'
                results['collisions'] += 1
            else:
                outcome = 'timeout'
                results['timeouts'] += 1
            
            results['episode_outcomes'].append(outcome)
            
            print(f"Episode {episode + 1}: {outcome.capitalize()}")
            print(f"Steps: {steps}")
            print(f"Reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    finally:
        # Calculate final statistics
        if episodes > 0:
            results['success_rate'] = (results['successes']/episodes) * 100
            results['avg_reward'] = results['total_reward'] / episodes
            results['avg_steps'] = results['total_steps'] / episodes
            
            print(f"\nTest Complete!")
            print(f"Success rate: {results['success_rate']:.1f}%")
            print(f"Average reward: {results['avg_reward']:.2f}")
            print(f"Average steps: {results['avg_steps']:.1f}")
            print(f"Collisions: {results['collisions']}")
            print(f"Timeouts: {results['timeouts']}")
            
            # Save results
            with open('ppo_test_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            print("\nDetailed results saved to: ppo_test_results.json")
        
        env.close()

if __name__ == "__main__":
    test_ppo_policy(
        model_path="ppo_policy_best.pth",
        episodes=20,
        render_delay=0.01
    )