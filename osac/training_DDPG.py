import osac_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import time
import os

# 1. Setup Logging Directory
logdir = "osac_rl_log_DDPG"

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env04.OSAC_V2X_Env()

# Verify Action Space
print(f"Action Space: {env.action_space}")
assert isinstance(env.action_space, gym.spaces.Box), "DDPG requires a Continuous (Box) Action Space!"

# 3. Define Action Noise
# DDPG is deterministic, so we add noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# 4. Model
print("--- Initializing DDPG Model ---")
# Ensure action_noise is defined!
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy", 
    env, 
    action_noise=action_noise, 
    verbose=1, 
    tensorboard_log=logdir,
    device="cuda",          
    
    # *** OPTIMIZATION 1: Lower Learning Rate ***
    # Standard 1e-3 is too aggressive for the new 0.3 Slew Rate. 
    # 3e-4 prevents the beam from oscillating wildly.
    learning_rate=3e-4,     
    
    buffer_size=200000,     
    
    # *** OPTIMIZATION 2: Extended Warmup ***
    # 1,000 steps is too short. 10,000 ensures the buffer has examples 
    # of braking, crossing, and accelerating before training starts.
    learning_starts=10000,   
    
    batch_size=128,         
    gamma=0.99, 
    tau=0.005
)
# 5. Training
TOTAL_TIMESTEPS = 5000000 

print(f"--- Starting DDPG Training for {TOTAL_TIMESTEPS} timesteps ---")
start_time = time.time()

# Single learn call for the entire duration
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="DDPG_Phase3")

# 6. Save Final Model
save_name = "osac_beam_tracker_ddpg"
model.save(save_name)

end_time = time.time()
print(f"--- Training Complete in {end_time - start_time:.2f} seconds. ---")
print(f"--- Final Model Saved as {save_name}.zip ---")

env.close()