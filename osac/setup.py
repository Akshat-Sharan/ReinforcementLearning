import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# --- 1.1 CONSTANTS AND PARAMETERS ---
class OSACEnvConstants:
    """Stores all physical and environmental constants."""
    MAP_SIZE = 100.0  # meters (200m x 200m)
    TIME_STEP = 0.1   # seconds (simulation granularity)
    MAX_EPISODE_STEPS = 500

    # Car/Mobility Parameters
    CAR_SPEED = 5.0  # m/s (approx 36 km/h)
    MAX_TURN_ANGLE = np.pi / 36 # Max 5 degree turn per step

    # Optical/OSAC Parameters
    TX_POWER_DBM = 10.0  # 10 dBm (10 mW)
    NOISE_POWER_DBM = -100.0 # Thermal noise power
    BEAM_WIDTH_DEG = 5.0 # Narrow beam width for optical communication
    BEAM_WIDTH_RAD = math.radians(BEAM_WIDTH_DEG)
    WAVELENGTH = 1550e-9 # Optical wavelength (not strictly needed for simplified model, but good practice)
    
    # RSU Position (Fixed)
    RSU_POS = np.array([50.0, 90.0]) # Center top of the map