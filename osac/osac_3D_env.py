import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

class OSAC3DEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode='human'):
        super(OSAC3DEnv, self).__init__()
        
        # 1. Setup PyBullet
        if render_mode == 'human':
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 2. Action Space: [Steer, Gas, Turret_Pan, Turret_Tilt]
        # Range: [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 3. Observation Space (Placeholder for now)
        # We will upgrade this to include 6-DoF data later
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.carId = None
        self.targetId = None
        self.planeId = None
        
        # Turret Joint Indices (will be determined dynamically)
        self.joint_indices = {}

    def _create_smart_car(self):
        """
        Procedurally builds a Car with a 2-Axis Laser Turret on the roof.
        """
        car_start_pos = [0, 0, 0.5]
        car_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Load standard Racecar
        car = p.loadURDF("racecar/racecar.urdf", car_start_pos, car_start_orientation)
        
        # --- Add the Turret (The O-ISAC Component) ---
        # 1. Turret Base (Pan/Yaw)
        turret_base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02])
        turret_base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02], rgbaColor=[0.2, 0.2, 0.2, 1])
        
        # 2. Laser Emitter (Tilt/Pitch)
        emitter_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.2)
        emitter_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, height=0.2, rgbaColor=[1, 0, 0, 1], visualFrameOrientation=p.getQuaternionFromEuler([1.57, 0, 0]))

        # Create the Turret Multibody attached to the car
        # We attach it to the chassis link (index 1 usually, but we verify)
        chassis_link_index = -1 # Base
        
        # Create Turret Base (Joint 0: Pan)
        self.turret_base_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=turret_base_shape,
            baseVisualShapeIndex=turret_base_visual,
            basePosition=[0, 0, 0.35] # On top of car
        )
        
        # Constraint: Attach Turret Base to Car Chassis
        cid = p.createConstraint(car, chassis_link_index, self.turret_base_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.15], [0, 0, 0])
        
        # Note: For a proper controllable turret in PyBullet without a single URDF, 
        # it is often cleaner to treat the turret as a separate robot attached by constraints, 
        # or edit the URDF. 
        # FOR PHASE 1 SIMPLICITY: We will control the ray orientation mathematically relative to the car frame
        # to ensure stability, rather than fighting the constraint solver immediately.
        
        return car

    def reset(self, seed=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        
        # Load Car
        self.carId = p.loadURDF("racecar/racecar.urdf", [0,0,0])
        
        # Create a Target (Floating Cube representing RSU)
        self.targetId = p.loadURDF("cube.urdf", [3, 3, 1], globalScaling=0.5)
        
        # Reset State
        self.turret_pan = 0.0
        self.turret_tilt = 0.0
        
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        # Action Breakdown
        steer = action[0] * 0.5
        gas = action[1] * 20
        d_pan = action[2] * 0.05 # Incremental change
        d_tilt = action[3] * 0.05
        
        # 1. Apply Vehicle Controls
        # Racecar indices: 4 and 6 are rear wheels, 0 and 2 are steering
        p.setJointMotorControl2(self.carId, 4, p.VELOCITY_CONTROL, targetVelocity=gas, force=10)
        p.setJointMotorControl2(self.carId, 6, p.VELOCITY_CONTROL, targetVelocity=gas, force=10)
        p.setJointMotorControl2(self.carId, 0, p.POSITION_CONTROL, targetPosition=-steer)
        p.setJointMotorControl2(self.carId, 2, p.POSITION_CONTROL, targetPosition=-steer)
        
        # 2. Update Turret State (Virtual Gimbal)
        self.turret_pan += d_pan
        self.turret_tilt += d_tilt
        
        # 3. Ray Casting Logic (The Optical Link)
        car_pos, car_orn = p.getBasePositionAndOrientation(self.carId)
        car_mat = p.getMatrixFromQuaternion(car_orn)
        
        # Convert matrix to directional vectors
        # X is forward, Y is left, Z is up (in this URDF)
        vec_fwd = np.array([car_mat[0], car_mat[3], car_mat[6]])
        vec_up = np.array([car_mat[2], car_mat[5], car_mat[8]])
        vec_right = np.array([car_mat[1], car_mat[4], car_mat[7]])
        
        # Calculate Ray Direction based on Pan/Tilt
        # We rotate the forward vector around Up (Pan) and Right (Tilt)
        # Simplified rotation for Phase 1
        ray_dir = vec_fwd * math.cos(self.turret_pan) * math.cos(self.turret_tilt) + \
                  vec_right * math.sin(self.turret_pan) * math.cos(self.turret_tilt) + \
                  vec_up * math.sin(self.turret_tilt)
        
        ray_start = np.array(car_pos) + np.array([0, 0, 0.3]) # Roof height
        ray_len = 100
        ray_end = ray_start + ray_dir * ray_len
        
        # Fire Ray
        results = p.rayTest(ray_start, ray_end)
        hitObjectUid = results[0][0]
        
        # 4. Visualization
        hit = (hitObjectUid == self.targetId)
        beam_color = [0, 1, 0] if hit else [1, 0, 0] # Green if hit, Red if miss
        p.addUserDebugLine(ray_start, results[0][3] if hitObjectUid != -1 else ray_end, beam_color, lifeTime=0.1, lineWidth=3)
        
        # Simple Logic Step
        p.stepSimulation()
        time.sleep(1./240.)
        
        reward = 1.0 if hit else -0.1
        done = False
        
        return np.zeros(10, dtype=np.float32), reward, done, False, {}

# --- MANUAL TEST LOOP ---
if __name__ == "__main__":
    env = OSAC3DEnv(render_mode='human')
    obs, _ = env.reset()
    
    print("Use Slider Controls to move car and turret!")
    
    # Add Debug Sliders for easy testing
    sl_steer = p.addUserDebugParameter("Steering", -1, 1, 0)
    sl_gas = p.addUserDebugParameter("Gas", -1, 1, 0)
    sl_pan = p.addUserDebugParameter("Turret Pan", -1, 1, 0)
    sl_tilt = p.addUserDebugParameter("Turret Tilt", -1, 1, 0)
    
    while True:
        steer = p.readUserDebugParameter(sl_steer)
        gas = p.readUserDebugParameter(sl_gas)
        pan = p.readUserDebugParameter(sl_pan)
        tilt = p.readUserDebugParameter(sl_tilt)
        
        # We pass raw slider values as "action"
        # Note: In the step function, I made pan/tilt incremental. 
        # For this debug test, we want absolute control, so we might see drift.
        # Ideally, the agent outputs velocity (d_pan), here we input velocity.
        
        env.step([steer, gas, pan, tilt])