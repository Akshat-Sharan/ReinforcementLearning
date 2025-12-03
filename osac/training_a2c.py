import osac_env02
import gymnasium as gym
from stable_baselines3 import A2C
import os

# 1. Setup Directories
models_dir = "osac"
logdir = "osac_rl_log_A2C"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env02.OSAC_V2X_Env()

# 3. Model
# A2C is faster per step but might require more samples to converge
model = A2C(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    device="cpu",
    ent_coef=0.0,    # Entropy coefficient for exploration
    learning_rate=7e-4
)

# 4. Training
TIMESTEPS = 20000
TOTAL_TIMESTEPS = 5000000 
iters = 0

print("--- Starting A2C Training ---")
while iters * TIMESTEPS < TOTAL_TIMESTEPS:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C_Phase3")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    print(f"Iteration {iters}: Saved A2C model.")

env.close()