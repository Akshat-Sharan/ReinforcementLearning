import osac_env01
import time

# --- 4.2 Training the PPO Agent ---
from stable_baselines3 import PPO

# Recreate the environment instance
env = osac_env01.OSAC_V2X_Env()

# Initialize the PPO agent with a Multi-Layer Perceptron (MlpPolicy)
# policy: "MlpPolicy" (a standard feedforward neural network)
# env: The custom environment
# verbose=1: Prints training information
PPO_params = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2, 
    "ent_coef": 0.01,         # Keeps a slight exploration pressure
    "vf_coef": 0.5,
    
    # *** KEY CHANGE 1: Higher Learning Rate ***
    # The default 0.0003 is too slow for strong penalties.
    "learning_rate": 0.0007,
    
    # *** KEY CHANGE 2: More Training Epochs ***
    # Process the collected data 20 times to force the agent to learn the right policy faster.
    "n_epochs": 20,
    
    # Use a large batch size if possible (1024 or 2048)
    "n_steps": 2048 
}
model = PPO("MlpPolicy", env, verbose=1, **PPO_params,tensorboard_log="./osac_rl_log/", device = "cpu")

print("\n--- Starting RL Training ---")
start_time = time.time()
# The agent will learn over 20,000 time steps.
# The policy will learn to adjust the beam angles (actions) to maximize the SNR (reward).
model.learn(total_timesteps=5000000) 

# Save the trained model
model.save("osac_beam_tracker_ppo")
print("--- Training Complete. Model Saved. ---")

end_time = time.time()
print(f"--- Training Complete in {end_time - start_time:.2f} seconds. Model Saved as osac_ppo_model.zip ---")

# Close the environment used for training
env.close()