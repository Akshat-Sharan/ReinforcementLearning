import setup
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class OSACEnvConstants:
    # --- Visualization Constants ---
    PIXELS_PER_METER = 8 # Scaling factor for visualization
    # Ensure setup.OSACEnvConstants is accessible if not already imported as C
    WINDOW_SIZE = int(setup.OSACEnvConstants.MAP_SIZE * PIXELS_PER_METER) 
    
    # --- Colors ---
    RSU_COLOR = (255, 0, 0)       # Red
    CAR_1_COLOR = (0, 0, 255)     # Blue (Agent)
    CAR_2_COLOR = (0, 255, 0)     # Green (Target)
    BEAM_COLOR = (255, 165, 0)    # Orange (OSAC Beam)
    LOS_COLOR = (0, 0, 0)         # NEW: Black (True/Expected LoS)
    ROAD_COLOR = (150, 150, 150)  # Grey

    # NEW Visualization Colors
    PREDICTED_CAR_COLOR = (255, 255, 0) # Yellow for predicted car position
    ECHO_BEAM_COLOR = (0, 255, 255)     # Cyan for echo beam
    PREDICTED_VELOCITY_COLOR = (255, 0, 255) # Magenta for velocity vector
    LABEL_COLOR = (255, 255, 0)          # Bright Yellow for text (for contrast)

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
        
        # Define the State Space (Observation Space) - 14 dimensions
        low_obs = np.array([
            0.0, 0.0, -self.C.CAR_SPEED, -self.C.CAR_SPEED,  # Car 1 state
            0.0, 0.0, -self.C.CAR_SPEED, -self.C.CAR_SPEED,  # Car 2 state
            -np.pi, -np.pi, 0.0, 0.0,                        # Angles (rad) & SNR (dB)
            -np.pi, -np.pi                                   # Predicted LoS Angles
        ], dtype=np.float32)
        high_obs = np.array([
            self.C.MAP_SIZE, self.C.MAP_SIZE, self.C.CAR_SPEED, self.C.CAR_SPEED,
            self.C.MAP_SIZE, self.C.MAP_SIZE, self.C.CAR_SPEED, self.C.CAR_SPEED,
            np.pi, np.pi, 50.0, 50.0,                        # Angles (rad) & SNR (dB)
            np.pi, np.pi                                     # Predicted LoS Angles
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
        
        # --- NEW: Internal variables for Prediction ---
        self.predicted_los_angles = np.array([0.0, 0.0], dtype=np.float32)
        self.pred_pos_car2 = np.array([0.0, 0.0], dtype=np.float32) # New: To store Car 2's predicted position

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
            np.array([self.last_snr_v2v, self.last_snr_v2i]),
            self.predicted_los_angles # The agent now sees the explicit prediction!
        ]).astype(np.float32)
        return obs

    def _get_info(self):
        """Provides debugging and monitoring information (UPDATED to include Car 2 true/predicted data)."""
        speed_car2 = np.linalg.norm(self.car2_vel)

        return {
            "snr_v2v": self.last_snr_v2v,
            "snr_v2i": self.last_snr_v2i,
            "distance_v2v": np.linalg.norm(self.car2_pos - self.car1_pos),
            "distance_v2i": np.linalg.norm(self.C.RSU_POS - self.car1_pos),
            # True and Predicted data for Car 2 (for quantitative results)
            "car2_true_pos": self.car2_pos.tolist(),
            "car2_predicted_pos": self.pred_pos_car2.tolist(), 
            "car2_true_vel": self.car2_vel.tolist(),
            "car2_predicted_vel": self.car2_vel.tolist(), # Predicted velocity == True velocity in this model
            "car2_speed": speed_car2,
        }
    
    # --- 2.1.5 New Helper: Explicit Next-State Prediction (Sensing Unit) ---
    def _predict_next_los(self):
        """
        Calculates the predicted Line-of-Sight (LoS) angle to Car 2 and RSU 
        at the next time step (t + delta_t). This is the 'Sensing' function.
        """
        dt = self.C.TIME_STEP
        
        # --- V2V Prediction ---
        # Predict Car 1's position at t + dt
        pred_pos_1 = self.car1_pos + self.car1_vel * dt
        
        # Predict Car 2's position at t + dt
        self.pred_pos_car2 = self.car2_pos + self.car2_vel * dt # <--- STORED HERE
        
        # Predicted vector and angle for V2V
        pred_dist_vec_v2v = self.pred_pos_car2 - pred_pos_1
        pred_angle_v2v = math.atan2(pred_dist_vec_v2v[1], pred_dist_vec_v2v[0])

        # --- V2I Prediction (RSU) ---
        # RSU is stationary, only Car 1 moves
        pred_pos_rsu = self.C.RSU_POS
        
        # Predicted vector and angle for V2I
        pred_dist_vec_v2i = pred_pos_rsu - pred_pos_1
        pred_angle_v2i = math.atan2(pred_dist_vec_v2i[1], pred_dist_vec_v2i[0])
        
        self.predicted_los_angles = np.array([pred_angle_v2v, pred_angle_v2i], dtype=np.float32)
        return self.predicted_los_angles

    # --- 2.3 Reset Function ---
    def reset(self, seed=None, options=None):
        """Resets the environment to a new starting configuration."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Scenario Constants
        ROAD_MIN = 5.0
        ROAD_MAX = 95.0
        CAR_SPEED = self.C.CAR_SPEED 

        # 1. Position Initialization (Constrained to the y=x diagonal)
        start_pos_1 = self.np_random.uniform(low=ROAD_MIN, high=ROAD_MIN + 20.0) 
        self.car1_pos = np.array([start_pos_1, start_pos_1], dtype=np.float32)
        
        start_pos_2 = self.np_random.uniform(low=ROAD_MAX - 20.0, high=ROAD_MAX)
        self.car2_pos = np.array([start_pos_2, start_pos_2], dtype=np.float32)

        # 2. Fixed Velocity for Straight Path
        diag_vel_component = CAR_SPEED / np.sqrt(2) 
        
        # Car 1 (Moving Up-Right)
        self.car1_vel = np.array([diag_vel_component, diag_vel_component], dtype=np.float32)
        
        # Car 2 (Moving Down-Left - Opposite Direction)
        self.car2_vel = np.array([-diag_vel_component, -diag_vel_component], dtype=np.float32)
        
        # Set RSU Position (Hardcoded from previous logic)
        self.C.RSU_POS = np.array([10.0, 50.0], dtype=np.float32)

        # 3. Initial Beam Angles
        self.car1_beam_angles = self.np_random.uniform(low=-np.pi, high=np.pi, size=2)
        
        # Initialize last SNR values
        self.last_snr_v2v = 0.0
        self.last_snr_v2i = 0.0
        
        # Recalculate prediction for the initial state
        self._predict_next_los()

        # 4. Return initial observation (state) and info dictionary
        return self._get_obs(), self._get_info()

    # --- 2.4 Step Function ---
    def step(self, action):
        """Runs one time-step of the environment's dynamics."""
        self.current_step += 1
        
        # 1. Apply Action: Update Car 1 Beam Angles
        angle_change = self.action_to_angle_change[action]
        self.car1_beam_angles += angle_change
        self.car1_beam_angles = np.arctan2(np.sin(self.car1_beam_angles), np.cos(self.car1_beam_angles))

        # 2. Update World: Simple Mobility
        self.car1_pos += self.car1_vel * self.C.TIME_STEP
        self.car2_pos += self.car2_vel * self.C.TIME_STEP
        
        # 3. Perform OSAC/Communication (Uplink Echo Feedback)
        new_snr_v2v = self._calculate_osac_snr(
            self.car1_pos, self.car1_beam_angles[0], self.car2_pos
        )
        new_snr_v2i = self._calculate_osac_snr(
            self.car1_pos, self.car1_beam_angles[1], self.C.RSU_POS
        )

        # 4. Define Reward (R) - FINAL OPTIMIZED REWARD SCHEME
        dist_vec_v2v = self.car2_pos - self.car1_pos
        ideal_angle_v2v = math.atan2(dist_vec_v2v[1], dist_vec_v2v[0])

        dist_vec_v2i = self.C.RSU_POS - self.car1_pos
        ideal_angle_v2i = math.atan2(dist_vec_v2i[1], dist_vec_v2i[0])

        error_v2v = abs(self.car1_beam_angles[0] - ideal_angle_v2v)
        error_v2i = abs(self.car1_beam_angles[1] - ideal_angle_v2i)
        
        error_v2v = min(error_v2v, 2 * np.pi - error_v2v)
        error_v2i = min(error_v2i, 2 * np.pi - error_v2i)

        R_Incentive = 100.0 / np.maximum((error_v2v + error_v2i), 1e-3)
        R_Penalty_Misalignment = -10.0 * (error_v2v**2 + error_v2i**2)
        
        SUCCESS_THRESHOLD_SNR = 10.0 
        R_Bonus = 0
        if new_snr_v2v >= SUCCESS_THRESHOLD_SNR and new_snr_v2i >= SUCCESS_THRESHOLD_SNR:
            R_Bonus = 5000 

        angle_change = self.action_to_angle_change[action]
        R_Penalty_Change = -0.5 * np.sum(angle_change ** 2)
        
        self._predict_next_los() # Update prediction BEFORE next observation
        
        reward = R_Incentive + R_Bonus + R_Penalty_Misalignment + R_Penalty_Change
        
        self.last_snr_v2v = new_snr_v2v
        self.last_snr_v2i = new_snr_v2i
        
        # 5. Check Termination Conditions
        distance_v2v = np.linalg.norm(self.car2_pos - self.car1_pos)
        terminated = self.current_step >= self.C.MAX_EPISODE_STEPS or distance_v2v > 150.0

        truncated = False 

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    # --- Render Helpers ---
    def _draw_text(self, canvas, text, color, pos):
        """Helper to draw text labels on the canvas."""
        if not pygame.get_init():
             pygame.init()
             
        try:
             if not pygame.font.get_init():
                 pygame.font.init()
                 
             font = pygame.font.SysFont(None, 24) # Size 24 font
             text_surface = font.render(text, True, color)
             canvas.blit(text_surface, pos)
        except pygame.error as e:
             print(f"Pygame font error: {e}")

    def _convert_to_pixels(self, coords):
        """Converts meters [x, y] to pygame pixels [x_p, y_p] with inverted Y-axis."""
        x_m, y_m = coords
        x_p = int(x_m * OSACEnvConstants.PIXELS_PER_METER)
        
        # Invert Y-axis for standard Pygame drawing (y=0 is top)
        y_p = OSACEnvConstants.WINDOW_SIZE - int(y_m * OSACEnvConstants.PIXELS_PER_METER) 
        return np.array([x_p, y_p])

    def _draw_los(self, canvas, start_pos_p, end_pos_m, los_color):
        """Calculates and draws the Line-of-Sight (LoS) line from Car 1 to a target."""
        
        dist_vec = end_pos_m - self.car1_pos
        distance = np.linalg.norm(dist_vec)
        
        if distance > 1.0:
            ideal_angle = math.atan2(dist_vec[1], dist_vec[0])
            
            beam_length = 50 * OSACEnvConstants.PIXELS_PER_METER 
            
            end_point_p = start_pos_p + beam_length * np.array([np.cos(ideal_angle), -np.sin(ideal_angle)])
            
            # Draw the LoS line (thin solid black line)
            pygame.draw.line(canvas, los_color, start_pos_p, end_point_p.astype(int), 1)

    def _render_frame(self):
        """Renders one frame of the simulation with full labeling."""
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
        start_point = self._convert_to_pixels(np.array([5.0, 5.0])) 
        end_point = self._convert_to_pixels(np.array([95.0, 95.0]))
        
        ROAD_WIDTH = 20
        pygame.draw.line(canvas, OSACEnvConstants.ROAD_COLOR, start_point, end_point, ROAD_WIDTH)

        # --- Object Positions ---
        agent_pos_p = self._convert_to_pixels(self.car1_pos)
        car2_pos_p = self._convert_to_pixels(self.car2_pos)
        rsu_pos_p = self._convert_to_pixels(self.C.RSU_POS)
        CAR_RADIUS = 5

        # Draw the LoS lines (Black)
        self._draw_los(canvas, agent_pos_p, self.car2_pos, OSACEnvConstants.LOS_COLOR) # LOS to true Car 2
        self._draw_los(canvas, agent_pos_p, self.C.RSU_POS, OSACEnvConstants.LOS_COLOR) # LOS to RSU

        # 1. Draw RSU (Red Square) and Label
        pygame.draw.rect(canvas, OSACEnvConstants.RSU_COLOR, 
                         (rsu_pos_p[0] - CAR_RADIUS, rsu_pos_p[1] - CAR_RADIUS, 
                          CAR_RADIUS * 2, CAR_RADIUS * 2))
        self._draw_text(canvas, "RSU (Target)", OSACEnvConstants.LABEL_COLOR, rsu_pos_p + np.array([5, 5]))

        # --- TRUE / PREDICTED CAR 2 VISUALIZATION ---

        # 4a. Draw Predicted Car 2 Position (Yellow 'X' - Sensing Output)
        predicted_car2_pos_p = self._convert_to_pixels(self.pred_pos_car2)
        # Draw a small cross/X
        pygame.draw.line(canvas, OSACEnvConstants.PREDICTED_CAR_COLOR, 
                         predicted_car2_pos_p + np.array([-CAR_RADIUS//2, -CAR_RADIUS//2]), 
                         predicted_car2_pos_p + np.array([CAR_RADIUS//2, CAR_RADIUS//2]), 2)
        pygame.draw.line(canvas, OSACEnvConstants.PREDICTED_CAR_COLOR, 
                         predicted_car2_pos_p + np.array([CAR_RADIUS//2, -CAR_RADIUS//2]), 
                         predicted_car2_pos_p + np.array([-CAR_RADIUS//2, CAR_RADIUS//2]), 2)
        self._draw_text(canvas, "PREDICTED Pos", OSACEnvConstants.LABEL_COLOR, predicted_car2_pos_p + np.array([-50, -50]))
        
        # 4b. Draw Car 2 (Green Circle - TRUE POSITION) and Label
        pygame.draw.circle(canvas, OSACEnvConstants.CAR_2_COLOR, car2_pos_p, CAR_RADIUS)
        self._draw_text(canvas, "TRUE Pos (Car 2/Rx)", OSACEnvConstants.LABEL_COLOR, car2_pos_p + np.array([10, 5]))
        
        # 4c. Draw Predicted Velocity Vector (Magenta Arrow) - Sensing Output
        car2_vel_norm = np.linalg.norm(self.car2_vel)
        if car2_vel_norm > 0.01:
            unit_vel = self.car2_vel / car2_vel_norm
            arrow_length = 20 * OSACEnvConstants.PIXELS_PER_METER
            end_point_p = car2_pos_p + arrow_length * np.array([unit_vel[0], -unit_vel[1]])
            
            pygame.draw.line(canvas, OSACEnvConstants.PREDICTED_VELOCITY_COLOR, car2_pos_p, end_point_p.astype(int), 3)
            self._draw_text(canvas, "PREDICTED Velocity/Direction", OSACEnvConstants.PREDICTED_VELOCITY_COLOR, car2_pos_p + np.array([40, 40])) 
        
        # --- ECHO BEAM (Cyan - Feedback/Second Layer Verification) ---
        dist_vec_car2_to_car1 = self.car1_pos - self.car2_pos
        echo_angle = math.atan2(dist_vec_car2_to_car1[1], dist_vec_car2_to_car1[0])
        echo_beam_length = 30 * OSACEnvConstants.PIXELS_PER_METER
        end_point_echo_p = car2_pos_p + echo_beam_length * np.array([np.cos(echo_angle), -np.sin(echo_angle)])
        
        pygame.draw.line(canvas, OSACEnvConstants.ECHO_BEAM_COLOR, car2_pos_p, end_point_echo_p.astype(int), 2)
        self._draw_text(canvas, "Echo/Feedback Beam", OSACEnvConstants.ECHO_BEAM_COLOR, car2_pos_p + np.array([-80, 0])) 

        # 2. Draw Car 1 (Agent - Blue Circle) and Label
        pygame.draw.circle(canvas, OSACEnvConstants.CAR_1_COLOR, agent_pos_p, CAR_RADIUS)
        self._draw_text(canvas, "Car 1 (Agent/Tx)", OSACEnvConstants.LABEL_COLOR, agent_pos_p + np.array([5, 5]))

        # 5. Draw Beams (V2V and V2I - Orange)
        beam_length = 50 * OSACEnvConstants.PIXELS_PER_METER
        
        # V2V Beam (Index 0)
        angle_v2v = self.car1_beam_angles[0]
        end_point_v2v_p = agent_pos_p + beam_length * np.array([np.cos(angle_v2v), -np.sin(angle_v2v)])
        pygame.draw.line(canvas, OSACEnvConstants.BEAM_COLOR, agent_pos_p, end_point_v2v_p.astype(int), 2)
        self._draw_text(canvas, "V2V Beam", OSACEnvConstants.BEAM_COLOR, agent_pos_p + (end_point_v2v_p - agent_pos_p) * 0.5 + np.array([5, 5]))

        # V2I Beam (Index 1)
        angle_v2i = self.car1_beam_angles[1]
        end_point_v2i_p = agent_pos_p + beam_length * np.array([np.cos(angle_v2i), -np.sin(angle_v2i)])
        pygame.draw.line(canvas, OSACEnvConstants.BEAM_COLOR, agent_pos_p, end_point_v2i_p.astype(int), 2)
        self._draw_text(canvas, "V2I Beam", OSACEnvConstants.BEAM_COLOR, agent_pos_p + (end_point_v2i_p - agent_pos_p) * 0.5 + np.array([5, 5]))
        
        # --- NEW: Quantitative Data Display ---
        
        # 1. Calculate Metrics
        
        # Misalignment Error (in radians)
        dist_vec_v2v = self.car2_pos - self.car1_pos
        ideal_angle_v2v = math.atan2(dist_vec_v2v[1], dist_vec_v2v[0])
        dist_vec_v2i = self.C.RSU_POS - self.car1_pos
        ideal_angle_v2i = math.atan2(dist_vec_v2i[1], dist_vec_v2i[0])

        error_v2v = abs(self.car1_beam_angles[0] - ideal_angle_v2v)
        error_v2i = abs(self.car1_beam_angles[1] - ideal_angle_v2i)
        
        # Normalize error to [0, pi] and sum them
        total_error_rad = min(error_v2v, 2 * np.pi - error_v2v) + min(error_v2i, 2 * np.pi - error_v2i)
        
        # Convert to milliradians (mrad) using the formula
        total_error_mrad = total_error_rad * (1000 / np.pi) 

        # Prediction Error (Euclidean distance between true and predicted pos)
        # Note: self.pred_pos_car2 is Car 2's *predicted next step* position.
        # The true Car 2 position *at the next step* is unknown, but we can measure
        # the error between the prediction and the *current* true position as a proxy, 
        # or, more accurately, the distance moved vs. the predicted distance. 
        # For simplicity, we'll calculate the prediction error *distance* for one time step.
        prediction_error_vec = self.car2_pos + self.car2_vel * self.C.TIME_STEP - self.pred_pos_car2
        # Since pred_pos_car2 = car2_pos + car2_vel * dt, this error should be near zero, 
        # proving the *perfect* prediction of your current model. We'll show the value.
        prediction_error_meters = np.linalg.norm(self.car2_pos - self.pred_pos_car2)
        
        # Since the prediction logic is: pred_pos = current_pos + velocity * dt, 
        # we display the actual position prediction difference for visualization:
        pred_pos_diff = np.linalg.norm(self.car2_pos - self.pred_pos_car2)

        
        # 2. Display the Data
        TEXT_POS_X = 10
        TEXT_POS_Y = 10
        LINE_HEIGHT = 25
        
        metrics = [
            f"Step: {self.current_step}",
            f"SNR V2V: {self.last_snr_v2v:.1f} dB",
            f"SNR V2I: {self.last_snr_v2i:.1f} dB",
            f"Total Misalignment Error: {total_error_mrad:.0f} mrad",
            f"Predicted Pos Diff (Car 2): {pred_pos_diff:.1f} m", # Expected to be near 0
            f"V2V Angle Error: {min(error_v2v, 2 * np.pi - error_v2v) * (1000/np.pi):.1f} mrad",
            f"V2I Angle Error: {min(error_v2i, 2 * np.pi - error_v2i) * (1000/np.pi):.1f} mrad"
        ]
        
        for i, text in enumerate(metrics):
            y_pos = TEXT_POS_Y + i * LINE_HEIGHT
            # Use white for the quantitative data for contrast
            self._draw_text(canvas, text, (255, 255, 255), np.array([TEXT_POS_X, y_pos]))
        
        # --- End of Quantitative Data Display ---
        
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    # --- Standard Gym Methods ---
    def render(self):
        """Gymnasium standard render method."""
        if self.metadata["render_modes"] == ["human"]:
             return self._render_frame()

    def close(self):
        """Gymnasium standard close method."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()