import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import osac_env
import gymnasium as gym
from stable_baselines3 import SAC
import time

# 1. Setup Logging Directory
logdir = "../osac_rl_log_SAC"

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env.OSAC_V2X_Env()

# 3. Model
print("--- Initializing SAC Model ---")
# SAC is off-policy and entropy-regularized.
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    
    # *** GPU ACCELERATION ***
    device="cuda", 
    
    # *** MAX PERFORMANCE SETTINGS ***
    learning_rate=3e-4,     
    buffer_size=1000000,    # 1 Million steps (Best for single-process training)
    batch_size=256,         
    ent_coef='auto',        
    gamma=0.99,
    tau=0.005,
    learning_starts=10000   # 10k Warmup to fill the buffer a bit before starting
)

# 4. Training
TOTAL_TIMESTEPS = 5000000 

print(f"--- Starting SAC Training for {TOTAL_TIMESTEPS} timesteps ---")
start_time = time.time()

# Single learn call for the entire duration
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="SAC_Phase3")

# 5. Save Final Model
save_name = "../osac_beam_tracker_sac"
model.save(save_name)

end_time = time.time()
print(f"--- Training Complete in {end_time - start_time:.2f} seconds. ---")
print(f"--- Final Model Saved as {save_name}.zip ---")

env.close()