import numpy as np
import math
from osac_env02 import OSAC_V2X_Env # Imports your Phase 5 environment

## 1. Helper Function: Calculate Ideal Angle (LoS)
def calculate_los_angle(tx_pos, rx_pos):
    """Calculates the Line-of-Sight (LoS) angle from tx_pos to rx_pos."""
    dist_vec = rx_pos - tx_pos
    return math.atan2(dist_vec[1], dist_vec[0])

## 2. Baseline Agent Logic (Continuous)
def baseline_predict(env):
    """
    Predicts the 'greedy' continuous action.
    It calculates the exact difference between the current beam angle and the
    ideal LoS angle, and tries to turn that amount (limited by MAX_SLEW_RATE).
    """
    
    # 1. Get Ideal LoS Angles
    los_v2v = calculate_los_angle(env.car1_pos, env.car2_pos)
    los_v2i = calculate_los_angle(env.car1_pos, env.C.RSU_POS)
    
    current_angles = env.car1_beam_angles # [V2V_angle, V2I_angle]
    
    # 2. Calculate Required Change (Delta)
    # We want: current + delta = ideal  =>  delta = ideal - current
    delta_v2v = los_v2v - current_angles[0]
    delta_v2i = los_v2i - current_angles[1]
    
    # Normalize deltas to [-pi, pi] so we turn the short way
    delta_v2v = math.atan2(math.sin(delta_v2v), math.cos(delta_v2v))
    delta_v2i = math.atan2(math.sin(delta_v2i), math.cos(delta_v2i))
    
    # 3. Scale to Action Space [-1, 1]
    # The environment multiplies action * MAX_SLEW_RATE.
    # So we need to divide our desired delta by MAX_SLEW_RATE.
    
    action_v2v = delta_v2v / env.C.MAX_SLEW_RATE
    action_v2i = delta_v2i / env.C.MAX_SLEW_RATE
    
    # 4. Clip to Valid Range [-1, 1]
    action_v2v = np.clip(action_v2v, -1.0, 1.0)
    action_v2i = np.clip(action_v2i, -1.0, 1.0)
    
    return np.array([action_v2v, action_v2i], dtype=np.float32)

## 3. Evaluation Loop
def evaluate_baseline(env, num_episodes=4000):
    all_rewards = []
    
    print("\n--- Running Baseline Demonstration (Continuous Greedy Tracker) ---")
    
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
            
            # Optional: Visual rendering
            # env.render() 
            
            if step_count % 50 == 0:
                 # Calculate misalignment for display
                 v2v_err = abs(info.get('snr_v2v', 0)) # Just using SNR as proxy for connection quality
                 # Note: Real error calc is inside env, this is just progress log
        
        all_rewards.append(total_reward)
        print(f"Episode {episode+1} Finished. Total Reward: {total_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    print(f"\n--- Baseline Average Reward over {num_episodes} episodes: {avg_reward:.2f} ---")
    return avg_reward


if __name__ == "__main__":
    # Initialize the environment
    baseline_env = OSAC_V2X_Env(render_mode="human") # Set to "human" if you want to see it
    evaluate_baseline(baseline_env, num_episodes=10) 
    baseline_env.close()