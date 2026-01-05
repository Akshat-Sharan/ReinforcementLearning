import osac_env04
import gymnasium as gym
from sb3_contrib import TRPO  # Import from contrib
import time
import os

# 1. Setup Logging Directory
logdir = "osac_rl_log_TRPO"

if not os.path.exists(logdir):
    os.makedirs(logdir)

# 2. Environment
env = osac_env04.OSAC_V2X_Env()

# 3. Model
print("--- Initializing TRPO Model ---")
# TRPO is very stable for continuous control
model = TRPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    device="cuda", 
    
    # *** OPTIMIZATION 1: Stability Batch ***
    # Collect more data per update to smooth out the noise/braking events.
    # Matches your successful PPO configuration.
    n_steps=4096,
    
    # *** OPTIMIZATION 2: Standard LR ***
    # TRPO's trust region protects it, so 1e-3 is safe and faster than 3e-4.
    learning_rate=1e-3,
    
    # Optional: Gamma (Discount Factor)
    # 0.99 is standard for continuous control.
    gamma=0.99
)

# 4. Training
TOTAL_TIMESTEPS = 5000000 

print(f"--- Starting TRPO Training for {TOTAL_TIMESTEPS} timesteps ---")
start_time = time.time()

# Single learn call for the entire duration
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="TRPO_Phase3")

# 5. Save Final Model
save_name = "osac_beam_tracker_trpo"
model.save(save_name)

end_time = time.time()
print(f"--- Training Complete in {end_time - start_time:.2f} seconds. ---")
print(f"--- Final Model Saved as {save_name}.zip ---")

env.close()