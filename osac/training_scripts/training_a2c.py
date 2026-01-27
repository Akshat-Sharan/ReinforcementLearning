import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import osac_env
import gymnasium as gym
from stable_baselines3 import A2C
import time

# 1. Setup Logging Directory
logdir = "../osac_rl_log_A2C"

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env.OSAC_V2X_Env()

# 3. Model
print("--- Initializing A2C Model ---")
# A2C is faster per step but might require more samples to converge
model = A2C(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    device="cuda", # Change to "cuda" if you want GPU
    ent_coef=0.01,    # Entropy coefficient for exploration
    learning_rate=7e-4
)

# 4. Training
TOTAL_TIMESTEPS = 5000000 

print(f"--- Starting A2C Training for {TOTAL_TIMESTEPS} timesteps ---")
start_time = time.time()

# Single learn call for the entire duration
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="A2C_Phase3")

# 5. Save Final Model
save_name = "../osac_beam_tracker_a2c"
model.save(save_name)

end_time = time.time()
print(f"--- Training Complete in {end_time - start_time:.2f} seconds. ---")
print(f"--- Final Model Saved as {save_name}.zip ---")

env.close()