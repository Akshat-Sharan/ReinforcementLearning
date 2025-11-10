import setup
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class OSACEnvConstants:
    # ... other constants
    PIXELS_PER_METER = 8 # Scaling factor for visualization
    WINDOW_SIZE = int(setup.OSACEnvConstants.MAP_SIZE * PIXELS_PER_METER) 
    
    # Colors
    RSU_COLOR = (255, 0, 0)       # Red
    CAR_1_COLOR = (0, 0, 255)     # Blue (Agent)
    CAR_2_COLOR = (0, 255, 0)     # Green (Target)
    BEAM_COLOR = (255, 165, 0)    # Orange (OSAC Beam)
    LOS_COLOR = (128, 128, 128)
    # NEW ROAD COLOR
    ROAD_COLOR = (150, 150, 150) # Grey

class OSAC_V2X_Env(setup.gym.Env):
    """
    A custom Gymnasium environment for V2V/V2I Optical Sensing and Communication (OSAC)
    beam tracking using a simplified 2D model.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.C = setup.OSACEnvConstants()
        self.current_step = 0
        self.window = None
        self.clock = None
        
        # Define the State Space (Observation Space)
        # 12-dimensional continuous state vector:
        # [Car1_x, Car1_y, Car1_vx, Car1_vy, Car2_x, Car2_y, Car2_vx, Car2_vy, 
        #  Current_Beam_Angle_V2V, Current_Beam_Angle_V2I, Last_SNR_V2V, Last_SNR_V2I]
        low_obs = np.array([
            0.0, 0.0, -self.C.CAR_SPEED, -self.C.CAR_SPEED,  # Car 1 state (Position/Velocity)
            0.0, 0.0, -self.C.CAR_SPEED, -self.C.CAR_SPEED,  # Car 2 state (Position/Velocity)
            -np.pi, -np.pi, 0.0, 0.0                          # Angles (rad) & SNR (dB)
        ], dtype=np.float32)
        high_obs = np.array([
            self.C.MAP_SIZE, self.C.MAP_SIZE, self.C.CAR_SPEED, self.C.CAR_SPEED,
            self.C.MAP_SIZE, self.C.MAP_SIZE, self.C.CAR_SPEED, self.C.CAR_SPEED,
            np.pi, np.pi, 50.0, 50.0                          # Angles (rad) & SNR (dB) (Max realistic SNR)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Define the Action Space (Discrete: Beam Adjustment)
        self.action_space = spaces.Discrete(5)
        self.action_to_angle_change = {
            0: np.array([self.C.MAX_TURN_ANGLE, 0.0]),     # Adjust V2V Pos
            1: np.array([-self.C.MAX_TURN_ANGLE, 0.0]),    # Adjust V2V Neg
            2: np.array([0.0, self.C.MAX_TURN_ANGLE]),     # Adjust V2I Pos
            3: np.array([0.0, -self.C.MAX_TURN_ANGLE]),    # Adjust V2I Neg
            4: np.array([0.0, 0.0])                        # Hold
        }
        
        # Internal state variables
        self.current_step = 0
        self.car1_pos = None  # [x, y]
        self.car2_pos = None
        self.car1_vel = None  # [vx, vy]
        self.car2_vel = None
        self.car1_beam_angles = None # [V2V_angle, V2I_angle] in radians
        self.last_snr_v2v = 0.0
        self.last_snr_v2i = 0.0

    # --- 2.1 OSAC/Optical Channel Model (Simplified) ---
    def _calculate_osac_snr(self, tx_pos, tx_angle, rx_pos):
        """Calculates the received SNR (dB) including pointing loss (beam tracking)."""
        
        # 1. Distance Calculation
        dist_vec = rx_pos - tx_pos
        distance = np.linalg.norm(dist_vec)
        
        # Safety check: if distance is zero or car is out of range, signal is zero.
        if distance < 1.0 or distance > self.C.MAP_SIZE * 2:
            return -100.0 # Effectively no signal

        # 2. Ideal Angle (Line-of-Sight or LoS)
        ideal_angle = math.atan2(dist_vec[1], dist_vec[0])

        # 3. Beam Misalignment (Pointing Error) - The core OSAC/Tracking factor
        angle_error = abs(tx_angle - ideal_angle)
        # Normalize angle error to [-pi, pi]
        angle_error = min(angle_error, 2 * np.pi - angle_error)

        # Simplified Pointing Loss Model (Gaussian beam approximation)
        pointing_loss = math.exp(-2 * (angle_error / self.C.BEAM_WIDTH_RAD)**2)
        
        # 4. Path Loss (Simplified Free Space)
        path_loss_dB = 10 * np.log10(distance**2 / 1.0) # Scaled Path Loss

        # 5. Received Power (dBm)
        tx_power_dBm = self.C.TX_POWER_DBM
        EPSILON = 1e-9
        safe_pointing_loss = np.maximum(pointing_loss, EPSILON)
        rx_power_dBm = tx_power_dBm - path_loss_dB + 10 * np.log10(safe_pointing_loss)
        
        # 6. SNR (dB)
        noise_power_dBm = self.C.NOISE_POWER_DBM
        snr_dB = rx_power_dBm - noise_power_dBm

        # Clip SNR to a reasonable max for stability and ensure non-negative in log scale
        return max(0.0, snr_dB)
        
    # --- 2.2 Helper Functions ---
    def _get_obs(self):
        """Packs the internal state into the observation space vector."""
        obs = np.concatenate([
            self.car1_pos, self.car1_vel,
            self.car2_pos, self.car2_vel,
            self.car1_beam_angles,
            np.array([self.last_snr_v2v, self.last_snr_v2i])
        ]).astype(np.float32)
        return obs

    def _get_info(self):
        """Provides debugging and monitoring information."""
        return {
            "snr_v2v": self.last_snr_v2v,
            "snr_v2i": self.last_snr_v2i,
            "distance_v2v": np.linalg.norm(self.car2_pos - self.car1_pos),
            "distance_v2i": np.linalg.norm(self.C.RSU_POS - self.car1_pos)
        }

    # --- 2.3 Reset Function (UPDATED: New Road Scenario) ---
    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new starting configuration on a straight road.
        The road runs diagonally from bottom-left to top-right.
        Cars start far apart and move toward each other along this line.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Scenario Constants
        ROAD_MIN = 5.0
        ROAD_MAX = 95.0
        CAR_SPEED = self.C.CAR_SPEED  # Use your existing constant velocity

        # 1. Position Initialization (Constrained to the y=x diagonal)
        # Car 1 (Blue Agent): Starts bottom-left
        start_pos_1 = self.np_random.uniform(low=ROAD_MIN, high=ROAD_MIN + 20.0) 
        self.car1_pos = np.array([start_pos_1, start_pos_1], dtype=np.float32)
        
        # Car 2 (Green Target): Starts top-right
        start_pos_2 = self.np_random.uniform(low=ROAD_MAX - 20.0, high=ROAD_MAX)
        self.car2_pos = np.array([start_pos_2, start_pos_2], dtype=np.float32)

        # 2. Fixed Velocity for Straight Path (Aligned with 45 degrees)
        # Calculate x and y components for the speed along the diagonal
        diag_vel_component = CAR_SPEED / np.sqrt(2) 
        
        # Car 1 (Moving Up-Right)
        self.car1_vel = np.array([diag_vel_component, diag_vel_component], dtype=np.float32)
        
        # Car 2 (Moving Down-Left - Opposite Direction)
        self.car2_vel = np.array([-diag_vel_component, -diag_vel_component], dtype=np.float32)
        
        # Ensure RSU position is set for the V2I link (e.g., on the left side)
        # Assuming self.C.RSU_POS is already set to something like (10.0, 50.0)
        
        # 3. Initial Beam Angles
        self.car1_beam_angles = self.np_random.uniform(low=-np.pi, high=np.pi, size=2)
        
        # Initialize last SNR values
        self.last_snr_v2v = 0.0
        self.last_snr_v2i = 0.0

        # 4. Return initial observation (state) and info dictionary
        return self._get_obs(), self._get_info()

    # --- 2.4 Step Function (UPDATED: New Mobility and Reward) ---
    def step(self, action):
        """
        Runs one time-step of the environment's dynamics.
        """
        self.current_step += 1
        
        # 1. Apply Action: Update Car 1 Beam Angles
        angle_change = self.action_to_angle_change[action]
        self.car1_beam_angles += angle_change
        # Ensure angles wrap around [-pi, pi]
        self.car1_beam_angles = np.arctan2(np.sin(self.car1_beam_angles), np.cos(self.car1_beam_angles))

        # 2. Update World: Simple Mobility (Integration of velocity)
        # Cars move along their predefined straight path velocity
        self.car1_pos += self.car1_vel * self.C.TIME_STEP
        self.car2_pos += self.car2_vel * self.C.TIME_STEP
        
        # REMOVED: Boundary reflection/bounce logic to ensure straight path movement

        # 3. Perform OSAC/Communication (Uplink Echo Feedback)
        new_snr_v2v = self._calculate_osac_snr(
            self.car1_pos, self.car1_beam_angles[0], self.car2_pos
        )
        new_snr_v2i = self._calculate_osac_snr(
            self.car1_pos, self.car1_beam_angles[1], self.C.RSU_POS
        )

        # 4. Define Reward (R) - FINAL OPTIMIZED REWARD SCHEME

        # Get Misalignment Errors (Keep calculation as is)
        dist_vec_v2v = self.car2_pos - self.car1_pos
        ideal_angle_v2v = math.atan2(dist_vec_v2v[1], dist_vec_v2v[0])

        dist_vec_v2i = self.C.RSU_POS - self.car1_pos
        ideal_angle_v2i = math.atan2(dist_vec_v2i[1], dist_vec_v2i[0])

        error_v2v = abs(self.car1_beam_angles[0] - ideal_angle_v2v)
        error_v2i = abs(self.car1_beam_angles[1] - ideal_angle_v2i)
        
        # Normalize error to [-pi, pi]
        error_v2v = min(error_v2v, 2 * np.pi - error_v2v)
        error_v2i = min(error_v2i, 2 * np.pi - error_v2i)

        # A. Alignment Incentive: Use the inverse of the error as a strong, immediate incentive
        # When error is 0, incentive is high (e.g., 100). When error is pi, incentive is low.
        # This replaces the weak R_Alignment and creates a continuous positive pull.
        
        # We cap the positive influence to prevent explosion when error is near zero (1e-3 is safe)
        R_Incentive = 100.0 / np.maximum((error_v2v + error_v2i), 1e-3)

        # B. Softened Misalignment Penalty
        # Reduced from -100 to -10. This makes misalignment costly, but not deadly.
        R_Penalty_Misalignment = -10.0 * (error_v2v**2 + error_v2i**2)
        
        # C. SUCCESS BONUS (Remains the same massive target)
        SUCCESS_THRESHOLD_SNR = 10.0 
        R_Bonus = 0
        if new_snr_v2v >= SUCCESS_THRESHOLD_SNR and new_snr_v2i >= SUCCESS_THRESHOLD_SNR:
            R_Bonus = 5000 

        # D. Beam Change Penalty (Small cost for movement)
        angle_change = self.action_to_angle_change[action]
        R_Penalty_Change = -0.5 * np.sum(angle_change ** 2)
        
        # Combine rewards
        reward = R_Incentive + R_Bonus + R_Penalty_Misalignment + R_Penalty_Change
        
        # Update last SNR for the next state observation
        self.last_snr_v2v = new_snr_v2v
        self.last_snr_v2i = new_snr_v2i
        
        # 5. Check Termination Conditions
        # Terminate when the episode length is reached OR the cars have passed each other
        distance_v2v = np.linalg.norm(self.car2_pos - self.car1_pos)
        terminated = self.current_step >= self.C.MAX_EPISODE_STEPS or distance_v2v > 150.0 # terminate if they pass by a lot

        truncated = False 

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    # Optional: Render method for visualization (e.g., using Pygame or Matplotlib)
    def render(self):
        """Gymnasium standard render method."""
        # This is correct: it calls _render_frame()
        if self.metadata["render_modes"] == ["human"]:
             return self._render_frame()

    def close(self):
        """Gymnasium standard close method."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _convert_to_pixels(self, coords):
        """Converts meters [x, y] to pygame pixels [x_p, y_p] with inverted Y-axis."""
        x_m, y_m = coords
        x_p = int(x_m * OSACEnvConstants.PIXELS_PER_METER)
        
        # Invert Y-axis for standard Pygame drawing (y=0 is top)
        y_p = OSACEnvConstants.WINDOW_SIZE - int(y_m * OSACEnvConstants.PIXELS_PER_METER) 
        return np.array([x_p, y_p])

    def _render_frame(self):
        """Renders one frame of the simulation using Pygame (UPDATED: Added Road)."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (OSACEnvConstants.WINDOW_SIZE, OSACEnvConstants.WINDOW_SIZE)
            )
            pygame.display.set_caption("OSAC Beam Tracking RL")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((OSACEnvConstants.WINDOW_SIZE, OSACEnvConstants.WINDOW_SIZE))
        
        # 0. Draw the Background (Green/Environment)
        canvas.fill((0, 100, 0))
        
        # 0.5. Draw the Road (A wide diagonal grey line)
        # Road runs from normalized (5, 5) to (95, 95)
        start_point = self._convert_to_pixels(np.array([5.0, 5.0]))  
        end_point = self._convert_to_pixels(np.array([95.0, 95.0]))
        
        ROAD_WIDTH = 20 # Wide road
        
        # Draw the main road line
        pygame.draw.line(canvas, OSACEnvConstants.ROAD_COLOR, start_point, end_point, ROAD_WIDTH)

        # --- Draw all objects ---
        agent_pos_p = self._convert_to_pixels(self.car1_pos)
        car2_pos_p = self._convert_to_pixels(self.car2_pos)
        rsu_pos_p = self._convert_to_pixels(self.C.RSU_POS)
        
        CAR_RADIUS = 5

        # Draw the LoS lines (Grey)
        self._draw_los(canvas, agent_pos_p, self.car2_pos, OSACEnvConstants.LOS_COLOR)
        self.C.RSU_POS[0] = 10.0 # RSU is fixed at a point, but make sure it is on the left
        self.C.RSU_POS[1] = 50.0 # Fixed Y point
        self._draw_los(canvas, agent_pos_p, self.C.RSU_POS, OSACEnvConstants.LOS_COLOR)

        # 1. Draw RSU (Red Square)
        pygame.draw.rect(canvas, OSACEnvConstants.RSU_COLOR, 
                         (rsu_pos_p[0] - CAR_RADIUS, rsu_pos_p[1] - CAR_RADIUS, 
                          CAR_RADIUS * 2, CAR_RADIUS * 2))

        # 2. Draw Car 1 (Agent - Blue Circle)
        pygame.draw.circle(canvas, OSACEnvConstants.CAR_1_COLOR, agent_pos_p, CAR_RADIUS)

        # 3. Draw Car 2 (Target - Green Circle)
        pygame.draw.circle(canvas, OSACEnvConstants.CAR_2_COLOR, car2_pos_p, CAR_RADIUS)

        # 4. Draw Beams (V2V and V2I - Orange)
        beam_length = 50 * OSACEnvConstants.PIXELS_PER_METER # 50 meters in pixels

        # V2V Beam (Index 0)
        angle_v2v = self.car1_beam_angles[0]
        end_point_v2v_p = agent_pos_p + beam_length * np.array([np.cos(angle_v2v), -np.sin(angle_v2v)])
        pygame.draw.line(canvas, OSACEnvConstants.BEAM_COLOR, agent_pos_p, end_point_v2v_p.astype(int), 2)
        
        # V2I Beam (Index 1)
        angle_v2i = self.car1_beam_angles[1]
        end_point_v2i_p = agent_pos_p + beam_length * np.array([np.cos(angle_v2i), -np.sin(angle_v2i)])
        pygame.draw.line(canvas, OSACEnvConstants.BEAM_COLOR, agent_pos_p, end_point_v2i_p.astype(int), 2)
        
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])


    def _draw_los(self, canvas, start_pos_p, end_pos_m, los_color):
        """Calculates and draws the Line-of-Sight (LoS) line from Car 1 to a target."""
        
        dist_vec = end_pos_m - self.car1_pos
        distance = np.linalg.norm(dist_vec)
        
        if distance > 1.0:
            ideal_angle = math.atan2(dist_vec[1], dist_vec[0])
            
            beam_length = 50 * OSACEnvConstants.PIXELS_PER_METER 
            
            end_point_p = start_pos_p + beam_length * np.array([np.cos(ideal_angle), -np.sin(ideal_angle)])
            
            # Draw the LoS line (thin solid grey line)
            pygame.draw.line(canvas, los_color, start_pos_p, end_point_p.astype(int), 1)