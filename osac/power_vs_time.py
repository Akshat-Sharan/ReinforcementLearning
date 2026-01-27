import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import PPO, DDPG, SAC
from sb3_contrib import TRPO
import osac_env  # Ensure this matches your environment file

# --- CONFIGURATION ---
NUM_EPISODES = 4000   # The "100 times" aggregation you asked for
MAX_STEPS = 400      # 20 seconds of simulation time
NORMALIZE = True     # Map [-100, 100] dB to [0, 1] scale

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "DDPG": os.path.join(script_dir, "models", "osac_beam_tracker_ddpg"),
    "PPO":  os.path.join(script_dir, "models", "osac_beam_tracker_ppo02"),
    "TRPO": os.path.join(script_dir, "models", "osac_beam_tracker_trpo"),
    "SAC":  os.path.join(script_dir, "models", "osac_beam_tracker_sac")
}

COLORS = {
    "DDPG": "#E74C3C",  # Red
    "PPO":  "#3498DB",  # Blue
    "TRPO": "#9B59B6",  # Purple
    "SAC":  "#2ECC71"   # Green
}

def normalize_snr(snr_array):
    """
    Normalizes SNR from physical range [-100, 100] dB to [0, 1].
    -100 dB -> 0.0 (Failure)
    +100 dB -> 1.0 (Perfect)
    """
    min_db = -100
    max_db = 100
    # Linear scaling
    norm = (snr_array - min_db) / (max_db - min_db)
    return np.clip(norm, 0, 1)

def get_time_aligned_stats(algo_name, model_path):
    env = osac_env.OSAC_V2X_Env()
    print(f"--- Processing {algo_name} ({NUM_EPISODES} episodes) ---")
    
    try:
        if algo_name == "PPO": model = PPO.load(model_path, env=env)
        elif algo_name == "DDPG": model = DDPG.load(model_path, env=env)
        elif algo_name == "SAC": model = SAC.load(model_path, env=env)
        elif algo_name == "TRPO": model = TRPO.load(model_path, env=env)
    except:
        print(f"Error loading {algo_name}")
        return None, None

    # Matrix to store history: [Episode, TimeStep]
    history_matrix = np.zeros((NUM_EPISODES, MAX_STEPS))

    for i in range(NUM_EPISODES):
        # Use different seeds to get different traffic patterns
        obs, _ = env.reset(seed=i) 
        
        for t in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            
            # Store SNR
            snr = info['snr_v2v']
            history_matrix[i, t] = snr
            
            # If episode ends early (crash/goal), fill rest with -100 dB (Outage)
            if term or trunc:
                history_matrix[i, t+1:] = -100.0
                break
                
    env.close()
    
    # Calculate Mean and Std Dev across the 100 episodes
    mean_trace = np.mean(history_matrix, axis=0)
    std_trace = np.std(history_matrix, axis=0)
    
    if NORMALIZE:
        # Normalize the Mean
        mean_norm = normalize_snr(mean_trace)
        # Normalize the Std Dev (scale down by range size)
        std_norm = std_trace / 200.0 
        return mean_norm, std_norm
    else:
        return mean_trace, std_trace

# --- PLOTTING ---
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create X-axis (Time in seconds)
time_axis = np.linspace(0, MAX_STEPS * 0.05, MAX_STEPS)

for algo, path in MODELS.items():
    mean, std = get_time_aligned_stats(algo, path)
    
    if mean is not None:
        # Plot Mean Line (Solid)
        plt.plot(time_axis, mean, label=algo, color=COLORS[algo], linewidth=2.5)
        
        # Plot Shaded Region (Confidence Interval)
        plt.fill_between(time_axis, mean - std, mean + std, color=COLORS[algo], alpha=0.15)

# --- FORMATTING ---
plt.title(f"Temporal Stability Analysis: Received Power Profile (Avg of {NUM_EPISODES} runs)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Time (seconds)", fontsize=14)

if NORMALIZE:
    plt.ylabel("Normalized Link Quality (0.0 - 1.0)", fontsize=14)
    plt.ylim(0, 1.05)
    # Add Threshold line at approx 0.55 (which is 10dB)
    plt.axhline(y=0.55, color='black', linestyle='--', label='Min QoS Threshold')
else:
    plt.ylabel("Received SNR (dB)", fontsize=14)
    plt.ylim(-110, 110)

plt.legend(loc='lower right', frameon=True, fontsize=12, shadow=True)
plt.xlim(0, 20)

plt.tight_layout()
filename = os.path.join(os.path.dirname(__file__), "results", "rigorous_power_vs_time.png")
os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=300)
print(f"\nSuccess! Plot saved as '{filename}'")
plt.show()