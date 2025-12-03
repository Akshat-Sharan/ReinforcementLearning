import osac_env
import gymnasium as gym
from sb3_contrib import TRPO  # Import from contrib
import os

# 1. Setup Directories
models_dir = "models/TRPO_Phase3"
logdir = "osac_rl_log"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env.OSAC_V2X_Env()

# 3. Model
# TRPO is very stable for continuous control
model = TRPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    device="cuda",
    learning_rate=1e-3
)

# 4. Training
TIMESTEPS = 20000
TOTAL_TIMESTEPS = 5000000 
iters = 0

print("--- Starting TRPO Training ---")
while iters * TIMESTEPS < TOTAL_TIMESTEPS:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TRPO_Phase3")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    print(f"Iteration {iters}: Saved TRPO model.")

env.close()