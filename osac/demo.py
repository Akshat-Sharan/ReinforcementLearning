import osac_env  # Make sure this imports your Phase 3 script (e.g., osac_env.py)
import gymnasium as gym
from stable_baselines3 import PPO, DDPG, SAC, A2C
from sb3_contrib import TRPO
import numpy as np
import time

# --- CONFIGURATION ---
# Select the algorithm/model you want to demo
# Options: "PPO", "DDPG", "SAC", "A2C", "TRPO"
ALGO_NAME = "SAC"

import os

# Path to the saved model file (ensure the name matches your saved file)
# Use absolute path relative to this script to ensure it works from any CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "models", f"osac_beam_tracker_{ALGO_NAME.lower()}")

# Number of episodes to watch
NUM_DEMO_EPISODES = 25

def run_demonstration():
    # 1. Create the Environment (Phase 3)
    env = osac_env.OSAC_V2X_Env()

    # 2. Load the Trained Model
    print(f"--- Loading trained model from: {MODEL_PATH}.zip ---")
    
    try:
        if ALGO_NAME == "PPO":
            model = PPO.load(MODEL_PATH, env=env)
        elif ALGO_NAME == "DDPG":
            model = DDPG.load(MODEL_PATH, env=env)
        elif ALGO_NAME == "SAC":
            model = SAC.load(MODEL_PATH, env=env)
        elif ALGO_NAME == "A2C":
            model = A2C.load(MODEL_PATH, env=env)
        elif ALGO_NAME == "TRPO":
            model = TRPO.load(MODEL_PATH, env=env)
        else:
            raise ValueError(f"Unknown Algorithm: {ALGO_NAME}")
            
        print("Model loaded successfully. Starting demonstration...")
        
    except FileNotFoundError:
        print(f"Error: Could not find model file '{MODEL_PATH}.zip'.")
        print("Please check the file name or run training first.")
        return

    # 3. Run Episodes
    for episode in range(NUM_DEMO_EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            # Predict the action
            # deterministic=True is safer for demos (uses the mean action)
            action, _states = model.predict(obs, deterministic=True)
            
            # --- CRITICAL FIX FOR CONTINUOUS ACTIONS ---
            # We do NOT convert to int. We pass the array directly.
            # action is typically shape (2,) e.g., [-0.54, 0.12]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update loop variables
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Render the environment (Pygame window)
            env.render()
            
            # Optional: Slow down the demo slightly if it's too fast
            # time.sleep(0.01) 

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {steps}")

    print("\nDemonstration Complete.")
    env.close()

if __name__ == "__main__":
    run_demonstration()