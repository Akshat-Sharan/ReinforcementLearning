import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. DATA ENTRY (Your Exact 25-Episode Results) ---

ddpg_rewards = [
    -1655.59, 10116.84, 538.30, 11366.55, -930.51,
    -5387.25, -4829.44, -6011.16, -3837.79, -3185.30,
    -4822.50, 433.89, -4300.17, -83.51, 451.96,
    3442.36, -4128.35, -6003.44, 6470.16, 4155.96,
    18052.86, -5149.67, 1121.33, 26040.37, 949.29
]

ppo_rewards = [
    -4486.48, 130064.92, 792868.85, 17014.72, 352826.22,
    472530.21, 524713.97, 3601.01, 356498.73, 593103.62,
    466235.30, 234952.83, 290730.94, 73189.27, 408929.96,
    442146.73, 843.39, 1232.60, 339165.89, 167088.66,
    448043.68, 657228.64, 466332.63, 300212.57, 115263.39
]

trpo_rewards = [
    990617.27, 1324707.30, 1191726.83, 1090028.78, 954296.77,
    968078.05, 1401524.35, 954363.16, 124707.50, 554051.03,
    966799.55, 629884.94, 455115.94, 759910.93, 750820.89,
    981844.77, 1041098.55, 990630.23, 936453.00, 1102692.10,
    79644.95, 33957.61, 964608.30, 953968.50, 700120.98
]

sac_rewards = [
    2627914.69, 2482397.17, 2191727.19, 432000.89, 2669522.98,
    1961590.26, 1710156.37, 3640770.88, 3258416.02, 3309787.12,
    2820261.16, 2770446.76, 3131890.58, 2530592.05, 651922.72,
    171381.98, 1584128.37, 1967799.71, 185097.76, 2554885.32,
    158444.32, 2087372.60, 2543929.84, 3224979.73, 477653.61
]

# Combine into dictionary
data = {
    'DDPG': ddpg_rewards,
    'PPO': ppo_rewards,
    'TRPO': trpo_rewards,
    'SAC': sac_rewards
}

# --- 2. CALCULATE STATISTICS ---
algorithms = list(data.keys())
means = [np.mean(data[alg]) for alg in algorithms]
std_devs = [np.std(data[alg]) for alg in algorithms]

print(f"{'Algorithm':<10} | {'Mean Reward':<15} | {'Std Dev':<15}")
print("-" * 45)
for alg, mean, std in zip(algorithms, means, std_devs):
    print(f"{alg:<10} | {mean:,.2f}".ljust(28) + f" | {std:,.2f}")

# --- 3. PLOTTING ---
sns.set_style("ticks") # Professional academic style
plt.figure(figsize=(12, 8))

# Distinct Colors (DDPG=Red, PPO=Blue, TRPO=Purple, SAC=Green)
colors = ['#E74C3C', '#3498DB', '#9B59B6', '#2ECC71']

# 1. Bar Chart (Means)
bars = plt.bar(algorithms, means, yerr=std_devs, capsize=8, 
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.5, label='Mean Reward')

# 2. Scatter/Swarm Plot (Individual Episodes)
for i, alg in enumerate(algorithms):
    y = data[alg]
    # Add jitter to x-axis so points don't overlap
    x = np.random.normal(i, 0.06, size=len(y))
    plt.scatter(x, y, color='black', alpha=0.5, s=20, zorder=3, marker='o')

# 3. Formatting
plt.title('Algorithm Performance: Optical Beam Tracking (Phase 6 Crossroad)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Total Cumulative Reward', fontsize=14)
plt.xlabel('Reinforcement Learning Algorithm', fontsize=14)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)

# Add grid only on Y axis
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add Mean Labels on top of bars
for bar in bars:
    height = bar.get_height()
    # Handle DDPG's low/negative bar placement
    y_pos = height + (max(means)*0.02) if height > 0 else height - (max(means)*0.05)
    plt.text(bar.get_x() + bar.get_width()/2.0, y_pos,
             f'{height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.tight_layout()

# Save
out_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)
filename = os.path.join(out_dir, 'final_phase6_comparison.png')
plt.savefig(filename, dpi=300)
print(f"\nPlot saved successfully as '{filename}'")
plt.show()