import osac_env
import time

# --- 4.2 Training the PPO Agent ---
from stable_baselines3 import PPO

# Recreate the environment instance
env = osac_env04.OSAC_V2X_Env()

# Initialize the PPO agent with a Multi-Layer Perceptron (MlpPolicy)
# policy: "MlpPolicy" (a standard feedforward neural network)
# env: The custom environment
# verbose=1: Prints training information
PPO_params = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2, 
    "ent_coef": 0.01,         # Good: Encourages exploration in the stochastic env
    "vf_coef": 0.5,
    
    # *** CHANGE 1: Standard Learning Rate for Precision ***
    # Slower learning = Better final alignment accuracy (Higher SNR)
    "learning_rate": 0.0003,
    
    # *** CHANGE 2: Standard Epochs for Stability ***
    # Prevents overfitting to random noise/turbulence
    "n_epochs": 10,
    
    # *** CHANGE 3: Larger Batch Size ***
    # Collecting more data before updating smoothens out the random noise
    "n_steps": 4096,  # Increased from 2048
    "batch_size": 128 # increased from 64 (optional, but good for GPU)
}

model = PPO("MlpPolicy", env, verbose=1, **PPO_params,tensorboard_log="./osac_rl_log_PPO_02/", device = "cuda")

print("\n--- Starting RL Training ---")
start_time = time.time()
# The agent will learn over 20,000 time steps.
# The policy will learn to adjust the beam angles (actions) to maximize the SNR (reward).
model.learn(total_timesteps=5000000) 

# Save the trained model
model.save("osac_beam_tracker_ppo02")
print("--- Training Complete. Model Saved. ---")

end_time = time.time()
print(f"--- Training Complete in {end_time - start_time:.2f} seconds. Model Saved as osac_beam_tracker_ppo02.zip ---")

# Close the environment used for training
env.close()