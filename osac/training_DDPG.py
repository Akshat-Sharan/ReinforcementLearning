import osac_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import os

# 1. Create Directories for Saving Models and Logs
models_dir = "models/DDPG_Phase3"
logdir = "osac_rl_log"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Initialize the Environment (Phase 3)
env = osac_env.OSAC_V2X_Env()

# Verify that the action space is continuous (Box)
# DDPG will throw an error if this is Discrete
print(f"Action Space: {env.action_space}")
assert isinstance(env.action_space, gym.spaces.Box), "DDPG requires a Continuous (Box) Action Space!"

# 3. Define Action Noise for Exploration
# Since DDPG is deterministic, we add Gaussian noise to the actions during training.
# sigma=0.2 means noise is 20% of the action range standard deviation.
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# 4. Initialize the DDPG Model
# - buffer_size: Size of the Replay Buffer (DDPG needs memory of past events)
# - learning_starts: How many steps to collect before updating weights (warm-up)
model = DDPG(
    "MlpPolicy", 
    env, 
    action_noise=action_noise, 
    verbose=1, 
    tensorboard_log=logdir,
    device="cuda",          # Uses your RTX 4070
    learning_rate=1e-3,     # Standard LR for DDPG
    buffer_size=200000,     # Store last 200k steps
    learning_starts=1000,   # Warmup steps
    batch_size=128,
    gamma=0.99,             # Discount factor
    tau=0.005               # Soft update coefficient
)

# 5. Training Loop
# We train in chunks to save the model periodically
TIMESTEPS = 20000
TOTAL_TIMESTEPS = 5000000 # Your goal of 5 million steps
iters = 0

print(f"--- Starting DDPG Training on {model.device} ---")

while iters * TIMESTEPS < TOTAL_TIMESTEPS:
    iters += 1
    
    # tb_log_name organizes the logs in TensorBoard under "DDPG_Phase3"
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG_Phase3")
    
    # Save the model
    save_path = f"{models_dir}/{TIMESTEPS*iters}"
    model.save(save_path)
    print(f"Iteration {iters}: Saved model to {save_path}")

print("--- Training Complete ---")
env.close()