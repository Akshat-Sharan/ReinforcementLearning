import osac_env

# --- 3.1 Environment Testing ---

# Create an instance of the environment
env = osac_env.OSAC_V2X_Env()

# 1. Reset the environment
observation, info = env.reset()
print(f"Initial Observation (State): {observation}")
print(f"Initial Info: {info}")

# 2. Run a few random steps to verify the loop
for step in range(5):
    # Action 0: V2V Beam +d(theta)
    action = 0 
    
    # Run the step
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Print results
    print(f"\n--- Step {step+1} (Action: {action}) ---")
    print(f"Reward: {reward:.4f}")
    print(f"New V2V SNR (dB): {info['snr_v2v']:.2f}")
    print(f"New V2I SNR (dB): {info['snr_v2i']:.2f}")
    
    if terminated:
        print("Episode terminated.")
        break

env.close()