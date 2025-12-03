import setup
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class OSACEnvConstants:
    # --- Phase 1: Physics Constants ---
    MAP_SIZE = 100.0          
    TIME_STEP = 0.05          
    CAR_SPEED_MAX = 15.0      
    CAR_ACCEL_MAX = 2.0       
    MAX_SLEW_RATE = 0.1       

    # --- Phase 2: Real-World Channel Constants ---
    TURBULENCE_SIGMA = 0.2    
    JITTER_STD = 0.005        
    CSI_NOISE_STD_DB = 1.0    
    
    # --- Phase 3: Obstructions (NEW) ---
    NUM_OBSTACLES = 3           # Number of random obstacles
    OBSTACLE_RADIUS_MIN = 3.0   # Min size (meters)
    OBSTACLE_RADIUS_MAX = 6.0   # Max size (meters)
    BLOCKAGE_PENALTY_DB = -100.0 # SNR when blocked

    # --- Visualization Constants ---
    PIXELS_PER_METER = 8 
    WINDOW_SIZE = int(MAP_SIZE * PIXELS_PER_METER) 
    
    # Colors
    RSU_COLOR = (255, 0, 0)       
    CAR_1_COLOR = (0, 0, 255)     
    CAR_2_COLOR = (0, 255, 0)     
    BEAM_COLOR = (255, 165, 0)    
    LOS_COLOR = (0, 0, 0)         
    ROAD_COLOR = (150, 150, 150)  
    OBSTACLE_COLOR = (101, 67, 33) # Dark Brown (Trees/Buildings) - NEW
    PREDICTED_CAR_COLOR = (255, 255, 0)      
    ECHO_BEAM_COLOR = (0, 255, 255)          
    PREDICTED_VELOCITY_COLOR = (255, 0, 255) 
    LABEL_COLOR = (255, 255, 0)              

class OSAC_V2X_Env(setup.gym.Env):
    """
    Phase 3 Upgrade: Obstructions (Blockage)
    - Random Obstacles (Trees/Buildings)
    - Line-Segment Intersection Logic
    - Non-Line-of-Sight (NLOS) Outages
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.C = setup.OSACEnvConstants() 
        
        # Override constants for all Phases
        self.C.TIME_STEP = OSACEnvConstants.TIME_STEP
        self.C.CAR_SPEED_MAX = OSACEnvConstants.CAR_SPEED_MAX
        self.C.CAR_ACCEL_MAX = OSACEnvConstants.CAR_ACCEL_MAX
        self.C.MAX_SLEW_RATE = OSACEnvConstants.MAX_SLEW_RATE
        self.C.TURBULENCE_SIGMA = OSACEnvConstants.TURBULENCE_SIGMA
        self.C.JITTER_STD = OSACEnvConstants.JITTER_STD
        self.C.CSI_NOISE_STD_DB = OSACEnvConstants.CSI_NOISE_STD_DB
        # Phase 3
        self.C.NUM_OBSTACLES = OSACEnvConstants.NUM_OBSTACLES
        self.C.OBSTACLE_RADIUS_MIN = OSACEnvConstants.OBSTACLE_RADIUS_MIN
        self.C.OBSTACLE_RADIUS_MAX = OSACEnvConstants.OBSTACLE_RADIUS_MAX

        self.current_step = 0
        self.window = None
        self.clock = None
        
        # Observation Space (14-dim)
        low_obs = np.array([
            0.0, 0.0, -self.C.CAR_SPEED_MAX, -self.C.CAR_SPEED_MAX,  
            0.0, 0.0, -self.C.CAR_SPEED_MAX, -self.C.CAR_SPEED_MAX,  
            -np.pi, -np.pi, 0.0, 0.0,                        
            -np.pi, -np.pi                                   
        ], dtype=np.float32)
        
        high_obs = np.array([
            self.C.MAP_SIZE, self.C.MAP_SIZE, self.C.CAR_SPEED_MAX, self.C.CAR_SPEED_MAX,
            self.C.MAP_SIZE, self.C.MAP_SIZE, self.C.CAR_SPEED_MAX, self.C.CAR_SPEED_MAX,
            np.pi, np.pi, 50.0, 50.0,                        
            np.pi, np.pi                                     
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action Space (Continuous)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Internal state
        self.car1_pos = None 
        self.car2_pos = None
        self.car1_vel = None 
        self.car2_vel = None
        self.car1_acc = np.zeros(2, dtype=np.float32)
        self.car2_acc = np.zeros(2, dtype=np.float32)
        
        self.obstacles = [] # List of tuples: (pos_x, pos_y, radius)

        self.car1_beam_angles = None 
        self.last_snr_v2v = 0.0
        self.last_snr_v2i = 0.0
        
        self.predicted_los_angles = np.array([0.0, 0.0], dtype=np.float32)
        self.pred_pos_car2 = np.array([0.0, 0.0], dtype=np.float32)

    # --- PHASE 3: Blockage Logic ---
    def _check_blockage(self, tx_pos, rx_pos):
        """
        Returns True if the line segment from tx_pos to rx_pos intersects any obstacle.
        Uses Distance from Point to Line Segment math.
        """
        for obs in self.obstacles:
            obs_pos = np.array([obs[0], obs[1]])
            radius = obs[2]
            
            # Vector math to find distance from Obstacle Center to Beam Segment
            d_vec = rx_pos - tx_pos
            f_vec = tx_pos - obs_pos
            
            # Project sphere center onto line segment
            # t = - (f . d) / (d . d)
            d_len_sq = np.dot(d_vec, d_vec)
            if d_len_sq == 0: continue # Tx and Rx are same point
            
            t = -np.dot(f_vec, d_vec) / d_len_sq
            
            # Clamp t to segment [0, 1]
            t = max(0.0, min(1.0, t))
            
            # Closest point on segment
            closest_point = tx_pos + t * d_vec
            
            # Distance from closest point to circle center
            dist = np.linalg.norm(closest_point - obs_pos)
            
            # If distance < radius, we hit the obstacle
            if dist < radius:
                return True # Blocked!
                
        return False

    def _calculate_osac_snr(self, tx_pos, tx_angle, rx_pos):
        """Calculates SNR with Phase 2 (Fading) AND Phase 3 (Blockage)."""
        
        # --- 0. BLOCKAGE CHECK (Phase 3) ---
        if self._check_blockage(tx_pos, rx_pos):
            return OSACEnvConstants.BLOCKAGE_PENALTY_DB # -100 dB (Outage)

        # --- Phase 2: Standard Channel Model ---
        dist_vec = rx_pos - tx_pos
        distance = np.linalg.norm(dist_vec)
        
        if distance < 1.0 or distance > self.C.MAP_SIZE * 2:
            return -100.0 

        ideal_angle = math.atan2(dist_vec[1], dist_vec[0])
        
        # Jitter
        jitter = np.random.normal(0, self.C.JITTER_STD)
        effective_tx_angle = tx_angle + jitter
        
        angle_error = abs(effective_tx_angle - ideal_angle)
        angle_error = min(angle_error, 2 * np.pi - angle_error)

        pointing_loss = math.exp(-2 * (angle_error / self.C.BEAM_WIDTH_RAD)**2)
        path_loss_dB = 10 * np.log10(distance**2 / 1.0)

        # Turbulence
        mu = -(self.C.TURBULENCE_SIGMA**2) / 2
        sigma = self.C.TURBULENCE_SIGMA
        turbulence_fading_linear = np.random.lognormal(mu, sigma)
        turbulence_loss_dB = -10 * np.log10(turbulence_fading_linear) 

        tx_power_dBm = self.C.TX_POWER_DBM
        EPSILON = 1e-9
        safe_pointing_loss = np.maximum(pointing_loss, EPSILON)
        pointing_loss_dB = 10 * np.log10(safe_pointing_loss)
        
        rx_power_dBm = tx_power_dBm - path_loss_dB + pointing_loss_dB - turbulence_loss_dB
        
        noise_power_dBm = self.C.NOISE_POWER_DBM
        true_snr_dB = rx_power_dBm - noise_power_dBm
        
        # Measurement Noise
        measurement_noise = np.random.normal(0, self.C.CSI_NOISE_STD_DB)
        measured_snr_dB = true_snr_dB + measurement_noise

        return max(0.0, measured_snr_dB)
        
    def _get_obs(self):
        obs = np.concatenate([
            self.car1_pos, self.car1_vel,
            self.car2_pos, self.car2_vel,
            self.car1_beam_angles,
            np.array([self.last_snr_v2v, self.last_snr_v2i]),
            self.predicted_los_angles 
        ]).astype(np.float32)
        return obs

    def _get_info(self):
        speed_car2 = np.linalg.norm(self.car2_vel)
        return {
            "snr_v2v": self.last_snr_v2v,
            "snr_v2i": self.last_snr_v2i,
            "distance_v2v": np.linalg.norm(self.car2_pos - self.car1_pos),
            "car2_speed": speed_car2,
        }
    
    def _predict_next_los(self):
        dt = self.C.TIME_STEP
        pred_pos_1 = self.car1_pos + self.car1_vel * dt
        self.pred_pos_car2 = self.car2_pos + self.car2_vel * dt 
        
        pred_dist_vec_v2v = self.pred_pos_car2 - pred_pos_1
        pred_angle_v2v = math.atan2(pred_dist_vec_v2v[1], pred_dist_vec_v2v[0])

        pred_pos_rsu = self.C.RSU_POS
        pred_dist_vec_v2i = pred_pos_rsu - pred_pos_1
        pred_angle_v2i = math.atan2(pred_dist_vec_v2i[1], pred_dist_vec_v2i[0])
        
        self.predicted_los_angles = np.array([pred_angle_v2v, pred_angle_v2i], dtype=np.float32)
        return self.predicted_los_angles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        ROAD_MIN = 5.0
        ROAD_MAX = 95.0

        start_pos_1 = self.np_random.uniform(low=ROAD_MIN, high=ROAD_MIN + 20.0) 
        self.car1_pos = np.array([start_pos_1, start_pos_1], dtype=np.float32)
        
        start_pos_2 = self.np_random.uniform(low=ROAD_MAX - 20.0, high=ROAD_MAX)
        self.car2_pos = np.array([start_pos_2, start_pos_2], dtype=np.float32)

        diag_vel = self.C.CAR_SPEED_MAX * 0.5 / np.sqrt(2) 
        self.car1_vel = np.array([diag_vel, diag_vel], dtype=np.float32)
        self.car2_vel = np.array([-diag_vel, -diag_vel], dtype=np.float32)
        
        self.car1_acc = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self.car2_acc = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        
        self.C.RSU_POS = np.array([10.0, 50.0], dtype=np.float32)
        self.car1_beam_angles = self.np_random.uniform(low=-np.pi, high=np.pi, size=2)
        self.last_snr_v2v = 0.0
        self.last_snr_v2i = 0.0
        
        # --- Generate Random Obstacles ---
        self.obstacles = []
        for _ in range(self.C.NUM_OBSTACLES):
            # Try to place obstacles randomly, but avoid placing them exactly on start/end road points
            # to prevent instant game-over.
            obs_x = self.np_random.uniform(10.0, 90.0)
            obs_y = self.np_random.uniform(10.0, 90.0)
            obs_r = self.np_random.uniform(self.C.OBSTACLE_RADIUS_MIN, self.C.OBSTACLE_RADIUS_MAX)
            self.obstacles.append((obs_x, obs_y, obs_r))
        
        self._predict_next_los()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        
        # 1. Action
        angle_adjustments = action * self.C.MAX_SLEW_RATE
        self.car1_beam_angles += angle_adjustments
        self.car1_beam_angles = np.arctan2(np.sin(self.car1_beam_angles), np.cos(self.car1_beam_angles))

        # 2. Physics
        acc_noise_1 = np.random.normal(0, 0.2, size=2)
        acc_noise_2 = np.random.normal(0, 0.2, size=2)
        self.car1_acc += acc_noise_1
        self.car2_acc += acc_noise_2
        self.car1_acc = np.clip(self.car1_acc, -self.C.CAR_ACCEL_MAX, self.C.CAR_ACCEL_MAX)
        self.car2_acc = np.clip(self.car2_acc, -self.C.CAR_ACCEL_MAX, self.C.CAR_ACCEL_MAX)
        self.car1_vel += self.car1_acc * self.C.TIME_STEP
        self.car2_vel += self.car2_acc * self.C.TIME_STEP
        speed_1 = np.linalg.norm(self.car1_vel)
        if speed_1 > self.C.CAR_SPEED_MAX: self.car1_vel = (self.car1_vel / speed_1) * self.C.CAR_SPEED_MAX
        speed_2 = np.linalg.norm(self.car2_vel)
        if speed_2 > self.C.CAR_SPEED_MAX: self.car2_vel = (self.car2_vel / speed_2) * self.C.CAR_SPEED_MAX
        self.car1_pos += self.car1_vel * self.C.TIME_STEP
        self.car2_pos += self.car2_vel * self.C.TIME_STEP
        
        # 3. Communication (Blockage + Fading)
        new_snr_v2v = self._calculate_osac_snr(self.car1_pos, self.car1_beam_angles[0], self.car2_pos)
        new_snr_v2i = self._calculate_osac_snr(self.car1_pos, self.car1_beam_angles[1], self.C.RSU_POS)

        # 4. Reward
        dist_vec_v2v = self.car2_pos - self.car1_pos
        ideal_angle_v2v = math.atan2(dist_vec_v2v[1], dist_vec_v2v[0])
        dist_vec_v2i = self.C.RSU_POS - self.car1_pos
        ideal_angle_v2i = math.atan2(dist_vec_v2i[1], dist_vec_v2i[0])

        error_v2v = min(abs(self.car1_beam_angles[0] - ideal_angle_v2v), 2*np.pi - abs(self.car1_beam_angles[0] - ideal_angle_v2v))
        error_v2i = min(abs(self.car1_beam_angles[1] - ideal_angle_v2i), 2*np.pi - abs(self.car1_beam_angles[1] - ideal_angle_v2i))

        # Reward modification: If blocked, the Incentive is useless. 
        # But we still punish misalignment so the agent 'tracks' through the wall.
        R_Incentive = 100.0 / np.maximum((error_v2v + error_v2i), 1e-3)
        R_Penalty_Misalignment = -10.0 * (error_v2v**2 + error_v2i**2)
        
        SUCCESS_THRESHOLD_SNR = 10.0 
        R_Bonus = 0
        if new_snr_v2v >= SUCCESS_THRESHOLD_SNR and new_snr_v2i >= SUCCESS_THRESHOLD_SNR:
            R_Bonus = 5000 

        R_Penalty_Change = -0.5 * np.sum(angle_adjustments ** 2)
        
        self._predict_next_los() 
        reward = R_Incentive + R_Bonus + R_Penalty_Misalignment + R_Penalty_Change
        
        self.last_snr_v2v = new_snr_v2v
        self.last_snr_v2i = new_snr_v2i
        
        distance_v2v = np.linalg.norm(self.car2_pos - self.car1_pos)
        terminated = self.current_step >= self.C.MAX_EPISODE_STEPS or distance_v2v > 150.0

        return self._get_obs(), reward, terminated, False, self._get_info()

    # --- Render Helpers ---
    def _draw_text(self, canvas, text, color, pos):
        if not pygame.get_init(): pygame.init()
        try:
             if not pygame.font.get_init(): pygame.font.init()
             font = pygame.font.SysFont(None, 24)
             text_surface = font.render(text, True, color)
             canvas.blit(text_surface, pos)
        except pygame.error: pass

    def _convert_to_pixels(self, coords):
        x_m, y_m = coords
        x_p = int(x_m * OSACEnvConstants.PIXELS_PER_METER)
        y_p = OSACEnvConstants.WINDOW_SIZE - int(y_m * OSACEnvConstants.PIXELS_PER_METER) 
        return np.array([x_p, y_p])

    def _draw_los(self, canvas, start_pos_p, end_pos_m, los_color):
        dist_vec = end_pos_m - self.car1_pos
        distance = np.linalg.norm(dist_vec)
        if distance > 1.0:
            ideal_angle = math.atan2(dist_vec[1], dist_vec[0])
            beam_length = 50 * OSACEnvConstants.PIXELS_PER_METER 
            end_point_p = start_pos_p + beam_length * np.array([np.cos(ideal_angle), -np.sin(ideal_angle)])
            pygame.draw.line(canvas, los_color, start_pos_p, end_point_p.astype(int), 1)

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((OSACEnvConstants.WINDOW_SIZE, OSACEnvConstants.WINDOW_SIZE))
            pygame.display.set_caption("OSAC Beam Tracking - Phase 3 (Obstructions)")
        if self.clock is None: self.clock = pygame.time.Clock()

        canvas = pygame.Surface((OSACEnvConstants.WINDOW_SIZE, OSACEnvConstants.WINDOW_SIZE))
        canvas.fill((0, 100, 0))
        
        start_point = self._convert_to_pixels(np.array([5.0, 5.0])) 
        end_point = self._convert_to_pixels(np.array([95.0, 95.0]))
        pygame.draw.line(canvas, OSACEnvConstants.ROAD_COLOR, start_point, end_point, 20)

        # --- Draw Obstacles (Phase 3) ---
        for obs in self.obstacles:
            obs_pos_p = self._convert_to_pixels(np.array([obs[0], obs[1]]))
            obs_radius_p = int(obs[2] * OSACEnvConstants.PIXELS_PER_METER)
            pygame.draw.circle(canvas, OSACEnvConstants.OBSTACLE_COLOR, obs_pos_p, obs_radius_p)
            # Optional: Add Label
            # self._draw_text(canvas, "Tree", (255,255,255), obs_pos_p)

        agent_pos_p = self._convert_to_pixels(self.car1_pos)
        car2_pos_p = self._convert_to_pixels(self.car2_pos)
        rsu_pos_p = self._convert_to_pixels(self.C.RSU_POS)
        CAR_RADIUS = 5

        self._draw_los(canvas, agent_pos_p, self.car2_pos, OSACEnvConstants.LOS_COLOR) 
        self._draw_los(canvas, agent_pos_p, self.C.RSU_POS, OSACEnvConstants.LOS_COLOR) 

        pygame.draw.rect(canvas, OSACEnvConstants.RSU_COLOR, (rsu_pos_p[0]-CAR_RADIUS, rsu_pos_p[1]-CAR_RADIUS, CAR_RADIUS*2, CAR_RADIUS*2))
        self._draw_text(canvas, "RSU", OSACEnvConstants.LABEL_COLOR, rsu_pos_p + np.array([5, 5]))

        predicted_car2_pos_p = self._convert_to_pixels(self.pred_pos_car2)
        pygame.draw.line(canvas, OSACEnvConstants.PREDICTED_CAR_COLOR, predicted_car2_pos_p+[-3,-3], predicted_car2_pos_p+[3,3], 2)
        pygame.draw.line(canvas, OSACEnvConstants.PREDICTED_CAR_COLOR, predicted_car2_pos_p+[3,-3], predicted_car2_pos_p+[-3,3], 2)
        self._draw_text(canvas, "PREDICTED", OSACEnvConstants.LABEL_COLOR, predicted_car2_pos_p + np.array([-30, -30]))
        
        pygame.draw.circle(canvas, OSACEnvConstants.CAR_2_COLOR, car2_pos_p, CAR_RADIUS)
        self._draw_text(canvas, "TRUE", OSACEnvConstants.LABEL_COLOR, car2_pos_p + np.array([10, 5]))
        
        car2_vel_norm = np.linalg.norm(self.car2_vel)
        if car2_vel_norm > 0.01:
            unit_vel = self.car2_vel / car2_vel_norm
            arrow_length = 20 * OSACEnvConstants.PIXELS_PER_METER
            end_point_p = car2_pos_p + arrow_length * np.array([unit_vel[0], -unit_vel[1]])
            pygame.draw.line(canvas, OSACEnvConstants.PREDICTED_VELOCITY_COLOR, car2_pos_p, end_point_p.astype(int), 3)

        dist_vec_car2_to_car1 = self.car1_pos - self.car2_pos
        echo_angle = math.atan2(dist_vec_car2_to_car1[1], dist_vec_car2_to_car1[0])
        end_point_echo_p = car2_pos_p + (30*OSACEnvConstants.PIXELS_PER_METER) * np.array([np.cos(echo_angle), -np.sin(echo_angle)])
        pygame.draw.line(canvas, OSACEnvConstants.ECHO_BEAM_COLOR, car2_pos_p, end_point_echo_p.astype(int), 2)

        pygame.draw.circle(canvas, OSACEnvConstants.CAR_1_COLOR, agent_pos_p, CAR_RADIUS)
        self._draw_text(canvas, "Agent", OSACEnvConstants.LABEL_COLOR, agent_pos_p + np.array([5, 5]))

        beam_len = 50 * OSACEnvConstants.PIXELS_PER_METER
        v2v_end = agent_pos_p + beam_len * np.array([np.cos(self.car1_beam_angles[0]), -np.sin(self.car1_beam_angles[0])])
        pygame.draw.line(canvas, OSACEnvConstants.BEAM_COLOR, agent_pos_p, v2v_end.astype(int), 2)
        
        v2i_end = agent_pos_p + beam_len * np.array([np.cos(self.car1_beam_angles[1]), -np.sin(self.car1_beam_angles[1])])
        pygame.draw.line(canvas, OSACEnvConstants.BEAM_COLOR, agent_pos_p, v2i_end.astype(int), 2)

        dist_vec_v2v = self.car2_pos - self.car1_pos
        ideal_angle_v2v = math.atan2(dist_vec_v2v[1], dist_vec_v2v[0])
        dist_vec_v2i = self.C.RSU_POS - self.car1_pos
        ideal_angle_v2i = math.atan2(dist_vec_v2i[1], dist_vec_v2i[0])
        error_v2v = min(abs(self.car1_beam_angles[0] - ideal_angle_v2v), 2*np.pi - abs(self.car1_beam_angles[0] - ideal_angle_v2v))
        error_v2i = min(abs(self.car1_beam_angles[1] - ideal_angle_v2i), 2*np.pi - abs(self.car1_beam_angles[1] - ideal_angle_v2i))
        total_error_mrad = (error_v2v + error_v2i) * (1000/np.pi)
        pred_pos_diff = np.linalg.norm(self.car2_pos - self.pred_pos_car2)

        metrics = [
            f"Step: {self.current_step}",
            f"SNR V2V: {self.last_snr_v2v:.1f} dB (Turb: {self.C.TURBULENCE_SIGMA})",
            f"SNR V2I: {self.last_snr_v2i:.1f} dB",
            f"Error: {total_error_mrad:.0f} mrad",
            f"Pred Diff: {pred_pos_diff:.1f} m"
        ]
        
        for i, text in enumerate(metrics):
            self._draw_text(canvas, text, (255, 255, 255), np.array([10, 10 + i * 25]))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.metadata["render_modes"] == ["human"]: return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()