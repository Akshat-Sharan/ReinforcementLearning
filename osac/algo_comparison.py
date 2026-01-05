import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---

# 1. Map your Algorithm Names to their Log Directories
# Check your folder to ensure these names match exactly!
LOG_DIRS = {
    "SAC":  "osac_rl_log_SAC",
    "PPO":  "osac_rl_log_PPO_02",  # Updated to match your latest run
    # If you didn't save logs for DDPG/TRPO, comment them out or point to correct folder
    "TRPO": "osac_rl_log_TRPO",    
    "DDPG": "osac_rl_log_DDPG"
}

# 2. Colors for the paper
COLORS = {
    "DDPG": "#E74C3C",  # Red
    "PPO":  "#3498DB",  # Blue
    "TRPO": "#9B59B6",  # Purple
    "SAC":  "#2ECC71"   # Green
}

# 3. Smoothing Factor (The "40 episodes" average you mentioned)
# Higher number = Smoother line, Less noise
SMOOTHING_WINDOW = 50 

def extract_tensorboard_data(log_dir):
    """Recursively finds tfevents files and extracts reward data."""
    if not os.path.exists(log_dir):
        print(f"Warning: Folder '{log_dir}' not found. Skipping.")
        return None

    # Find the tfevents file inside the folders
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"Warning: No event files found in '{log_dir}'.")
        return None

    # We assume the most recent file is the correct one, or combine them
    # For simplicity, let's take the largest file (most data)
    best_file = max(event_files, key=os.path.getsize)
    print(f"Reading logs from: {best_file}")

    ea = EventAccumulator(best_file)
    ea.Reload()

    # Stable Baselines3 logs rewards under 'rollout/ep_rew_mean'
    if 'rollout/ep_rew_mean' not in ea.Tags()['scalars']:
        print("Warning: No reward data found in logs.")
        return None

    events = ea.Scalars('rollout/ep_rew_mean')
    
    steps = [e.step for e in events]
    rewards = [e.value for e in events]
    
    return pd.DataFrame({"steps": steps, "rewards": rewards})

# --- MAIN PLOTTING LOOP ---
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid") # Academic style background

print("Extracting Training Data...")

for algo, log_folder in LOG_DIRS.items():
    df = extract_tensorboard_data(log_folder)
    
    if df is not None:
        # 1. Normalize Rewards to Millions (Optional, purely for looks)
        # df['rewards'] = df['rewards'] / 1e6 
        
        # 2. Apply Rolling Average (Smoothing)
        # This creates the "Trend Line"
        df['smooth_mean'] = df['rewards'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        
        # 3. Apply Rolling Standard Deviation
        # This creates the "Shaded Region" width
        df['smooth_std'] = df['rewards'].rolling(window=SMOOTHING_WINDOW, min_periods=1).std()
        
        # 4. Plotting
        # Solid Line
        plt.plot(df['steps'], df['smooth_mean'], label=algo, color=COLORS.get(algo, 'black'), linewidth=2)
        
        # Shaded Region (Mean +/- Std Dev)
        plt.fill_between(df['steps'], 
                         df['smooth_mean'] - df['smooth_std'], 
                         df['smooth_mean'] + df['smooth_std'], 
                         color=COLORS.get(algo, 'black'), alpha=0.15)

# --- FORMATTING ---
plt.title(f"Training Progress (Learning Curve)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Total Timesteps", fontsize=14)
plt.ylabel("Average Episode Reward", fontsize=14)

# Format X-axis to show "Millions" (e.g., 1M, 2M)
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.0f}M'.format(x/1e6) for x in current_values])

plt.legend(loc='lower right', frameon=True, fontsize=12, shadow=True)
plt.tight_layout()

# Save
plt.savefig("final_learning_curve.png", dpi=300)
print("\nSuccess! Plot saved as 'final_learning_curve.png'")
plt.show()