# demo.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from osac_env02 import OSAC_V2X_Env # Make sure this path is correct
import numpy as np

# --- 1. Configuration ---
# ⚠️ IMPORTANT: Update this path to your trained model file
MODEL_PATH = "osac_beam_tracker_ppo.zip" 
DEMO_EPISODES = 5 # Number of episodes to run for demonstration
MAX_STEPS_PER_EPISODE = 500 # Should match your C.MAX_EPISODE_STEPS

def run_demonstration():
    """
    Loads the trained PPO model and runs a visualization loop.
    """
    print(f"--- Loading trained model from: {MODEL_PATH} ---")
    
    # 1. Initialize the Environment with the "human" render mode
    # We use a single, non-vectorized environment for visualization
    try:
        env = OSAC_V2X_Env(render_mode="human")
    except Exception as e:
        print(f"Error initializing environment for rendering: {e}")
        print("Ensure 'render_mode=\"human\"' is supported and Pygame is installed.")
        return

    # 2. Load the trained model
    try:
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Did you run training.py first?")
        env.close()
        return

    print("Model loaded successfully. Starting demonstration...")
    
    # 3. Demonstration Loop
    for episode in range(DEMO_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        current_step = 0

        while not done and not truncated and current_step < MAX_STEPS_PER_EPISODE:
            # The model predicts the best action based on the current observation
            action_array, _states = model.predict(obs, deterministic=True) 
            action_int = int(action_array.item())
            # Apply the action and step the environment
            obs, reward, done, truncated, info = env.step(action_int)
            total_reward += reward
            current_step += 1
            
            # Render the environment frame
            env.render()

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {current_step}")

    # 4. Clean up
    env.close()

if __name__ == "__main__":
    # Ensure your trained model is saved as "osac_ppo_model.zip" 
    # and is in the same directory, or update MODEL_PATH.
    run_demonstration()