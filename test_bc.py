from train_bc import load_policy
from aircraft_env import AircraftEnv
import numpy as np
import torch
import pygame
import time
import json
from datetime import datetime
import os

def test_bc_policy(model_path: str, episodes: int = 20, render_delay: float = 0.01):
    """Test BC policy"""
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"bc_testing_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = AircraftEnv(render_mode="human", mode="bc")
    
    try:
        # Load trained policy
        policy = load_policy(
            model_path,
            state_dim=len(env.observation_space.low)
        )
        print("Successfully loaded policy")
    except Exception as e:
        print(f"Error loading policy: {e}")
        env.close()
        return
    
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
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                if render_delay > 0:
                    time.sleep(render_delay)
                
                # Get action from policy
                with torch.no_grad():
                    action = policy.predict(obs)
                
                # Take step in environment
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                steps += 1
            
            # Update statistics
            results['total_steps'] += steps
            results['total_reward'] += episode_reward
            results['episode_rewards'].append(episode_reward)
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
            
            # Episode summary
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
            results_file = f"{results_dir}/test_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nDetailed results saved to: {results_file}")
        
        env.close()

if __name__ == "__main__":
    try:
        test_bc_policy(
            model_path="bc_policy_best",
            episodes=20,
            render_delay=0.01
        )
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"Error during testing: {e}")
        pygame.quit()