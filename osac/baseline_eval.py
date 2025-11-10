import numpy as np
import math
from osac_env import OSAC_V2X_Env # Imports your environment
# No need for Stable-Baselines3 (PPO) here, as this is the non-RL comparison

## 1. Helper Function: Calculate Ideal Angle (LoS)
def calculate_los_angle(tx_pos, rx_pos):
    """Calculates the Line-of-Sight (LoS) angle from tx_pos to rx_pos."""
    dist_vec = rx_pos - tx_pos
    return math.atan2(dist_vec[1], dist_vec[0])

## 2. Baseline Agent Logic
def baseline_predict(env):
    """
    Predicts the 'greedy' best discrete action by choosing the action
    that results in the minimum total misalignment with the LoS angles.
    """
    
    # Get current LoS angles
    los_v2v = calculate_los_angle(env.car1_pos, env.car2_pos)
    los_v2i = calculate_los_angle(env.car1_pos, env.C.RSU_POS)
    
    current_angles = env.car1_beam_angles # [V2V_angle, V2I_angle]
    
    best_action = 4 # Default to 'No change'
    min_misalignment = float('inf')

    # Iterate through all 5 discrete actions
    for action in env.action_to_angle_change:
        angle_change = env.action_to_angle_change[action]
        
        # Calculate resulting angles if this action is taken
        new_v2v = current_angles[0] + angle_change[0]
        new_v2i = current_angles[1] + angle_change[1]
        
        # Calculate misalignment with LoS for V2V and V2I
        misalignment_v2v = abs(new_v2v - los_v2v)
        misalignment_v2i = abs(new_v2i - los_v2i)
        
        # Normalize angle error to [-pi, pi] for both links
        misalignment_v2v = min(misalignment_v2v, 2 * np.pi - misalignment_v2v)
        misalignment_v2i = min(misalignment_v2i, 2 * np.pi - misalignment_v2i)
        
        # Total squared misalignment (the baseline agent tries to minimize this)
        total_misalignment = misalignment_v2v**2 + misalignment_v2i**2
        
        if total_misalignment < min_misalignment:
            min_misalignment = total_misalignment
            best_action = action
            
    return best_action

## 3. Evaluation Loop
def evaluate_baseline(env, num_episodes=10):
    all_rewards = []
    
    print("\n--- Running Baseline Demonstration (Greedy Tracker) ---")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Baseline runs for the same number of steps as your training episodes
        while not done and step_count < env.C.MAX_EPISODE_STEPS:
            # Baseline agent predicts the action
            action = baseline_predict(env)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:
                 print(f"Ep {episode+1}, Step {step_count}: V2V SNR: {info['snr_v2v']:.2f} dB, V2I SNR: {info['snr_v2i']:.2f} dB, Total Reward: {total_reward:.2f}")

        all_rewards.append(total_reward)
        print(f"Episode {episode+1} Finished. Total Reward: {total_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    print(f"\n--- Baseline Average Reward over {num_episodes} episodes: {avg_reward:.2f} ---")
    return avg_reward


if __name__ == "__main__":
    # Initialize the environment for evaluation (render_mode=None for faster testing)
    baseline_env = OSAC_V2X_Env(render_mode=None) 
    evaluate_baseline(baseline_env, num_episodes=10) 
    baseline_env.close()