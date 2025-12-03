import osac_env
import gymnasium as gym
from stable_baselines3 import SAC
import os

# 1. Setup Directories
models_dir = "models/SAC_Phase3"
logdir = "osac_rl_log"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env.OSAC_V2X_Env()

# 3. Model
# SAC is off-policy and entropy-regularized.
# - ent_coef='auto': Automatically adjusts exploration temperature.
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    device="cuda",
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    ent_coef='auto',  # Crucial for stable exploration
    gamma=0.99,
    tau=0.005
)

# 4. Training
# SAC is usually sample-efficient, so it might converge faster than PPO.
TIMESTEPS = 20000
TOTAL_TIMESTEPS = 5000000 
iters = 0

print(f"--- Starting SAC Training on {model.device} ---")

while iters * TIMESTEPS < TOTAL_TIMESTEPS:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC_Phase3")
    
    save_path = f"{models_dir}/{TIMESTEPS*iters}"
    model.save(save_path)
    print(f"Iteration {iters}: Saved SAC model to {save_path}")

print("--- Training Complete ---")
env.close()