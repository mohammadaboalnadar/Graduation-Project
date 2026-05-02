import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco, mujoco.viewer

class UnitreeA1Env(gym.Env):
	"""
	Gymnasium wrapper around a Unitree A1 MuJoCo model.
	Task: walk forward as fast as possible without falling.
	"""

	def __init__(self, xml_path: str, max_episode_steps: int = 1000, render_mode=None):
		super().__init__()

		# ── Load your existing MuJoCo model ──────────────────────────
		self.model = mujoco.MjModel.from_xml_path(xml_path)
		self.data  = mujoco.MjData(self.model)

		foot_geom_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
		self._foot_geom_ids = set(
			mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
			for name in foot_geom_names
		)
		self._robot_geom_ids = set(
			i for i in range(self.model.ngeom)
			if self.model.geom_bodyid[i] > 0
		)

		self.max_episode_steps = max_episode_steps
		self._step_count = 0

		# ── Action space: 12 joint torques, normalised to [-1, 1] ─────
		# SB3 will scale these; you denormalise inside step()
		n_actuators = self.model.nu  # should be 12 for A1
		self.action_space = spaces.Box(
			low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32
		)

		# ── Observation space ─────────────────────────────────────────
		# 12 joint positions + 12 joint velocities = 24
		# 4 quaternion (base orientation) + 3 angular velocity = 7
		# 3 linear velocity of base = 3
		# 4 foot contact booleans = 4
		# Total = 38
		obs_dim = 38
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
		)

		# torque limits (check your MJCF — adjust as needed)
		self.torque_limit = 33.5  # Nm, A1 spec

		# ── Rendering setup ───────────────────────────────────────────
		self.render_mode = render_mode
		self._viewer: mujoco.viewer.MjViewer | mujoco.Renderer | None = None

	# ──────────────────────────────────────────────────────────────────
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# Reset sim to keyframe 0 (or default pose) + small random noise
		mujoco.mj_resetData(self.model, self.data)

		# Optional: add tiny noise to avoid identical rollouts
		self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, self.model.nq)
		self.data.qvel[:] += self.np_random.uniform(-0.01, 0.01, self.model.nv)

		mujoco.mj_forward(self.model, self.data)  # recompute everything

		self._step_count = 0
		return self._get_obs(), {}

	# ──────────────────────────────────────────────────────────────────
	def step(self, action: np.ndarray):
		# Denormalise action from [-1,1] → actual torques
		torques = action * self.torque_limit
		self.data.ctrl[:] = torques

		# Advance physics (default timestep * n_substeps)
		mujoco.mj_step(self.model, self.data)
		self._step_count += 1

		obs    = self._get_obs()
		reward, components = self._compute_reward(action)

		# Episode ends if robot falls or time limit reached
		terminated = self._is_fallen()
		truncated  = self._step_count >= self.max_episode_steps

		return obs, reward, terminated, truncated, {"reward_components": components}

	# ──────────────────────────────────────────────────────────────────
	def _get_obs(self) -> np.ndarray:
		# Joint positions and velocities (12 each)
		# qpos[0:3] = base xyz, qpos[3:7] = base quaternion, qpos[7:] = joints
		joint_pos = self.data.qpos[7:].copy()   # 12 values
		joint_vel = self.data.qvel[6:].copy()    # 12 values

		# Base orientation (quaternion) and angular velocity
		base_quat    = self.data.qpos[3:7].copy()   # 4 values
		base_ang_vel = self.data.qvel[3:6].copy()   # 3 values

		# Base linear velocity (in world frame)
		base_lin_vel = self.data.qvel[0:3].copy()   # 3 values

		# Foot contacts: check if each foot geom has contact force > threshold
		foot_contacts = self._get_foot_contacts()    # 4 values

		return np.concatenate([
			joint_pos, joint_vel,
			base_quat, base_ang_vel, base_lin_vel,
			foot_contacts
		]).astype(np.float32)

	def _get_foot_contacts(self) -> np.ndarray:
		contacts = np.zeros(4, dtype=np.float32)
		foot_geom_list = list(self._foot_geom_ids)
		for i, geom_id in enumerate(foot_geom_list):
			for contact in self.data.contact:
				if geom_id in (contact.geom1, contact.geom2):
					contacts[i] = 1.0
					break
		return contacts

	# ──────────────────────────────────────────────────────────────────
	def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
		forward_vel  = self.data.qvel[0]
		lateral_vel  = abs(self.data.qvel[1])
		vertical_vel = abs(self.data.qvel[2])  # penalise bouncing
		ang_vel      = self.data.qvel[3:6]
		base_quat    = self.data.qpos[3:7]     # w, x, y, z

		# Reward forward velocity but cap it — no reward for flipping fast
		# target_vel  = 0.8
		# vel_reward  = np.exp(-2.0 * abs(forward_vel - target_vel))  # gaussian peak at target
		vel_reward = forward_vel  # linear reward for moving forward, no cap

		# Penalise moving sideways or bouncing
		lateral_penalty  = -0.5 * lateral_vel
		vertical_penalty = -0.5 * vertical_vel

		# Reward staying upright — w component of quaternion is 1 when flat
		uprightness       = float(base_quat[0] ** 2)
		orientation_reward = 2.0 * uprightness

		# Penalise spinning — this directly punishes flipping
		ang_vel_penalty = -0.2 * float(np.sum(np.square(ang_vel)))

		# Penalise large torques
		energy_penalty = 0 #-0.01 * float(np.sum(np.square(action)))

		components = {
			"forward":     float(vel_reward),
			"lateral":     float(lateral_penalty),
			"orientation": float(orientation_reward),
			"ang_vel":     float(ang_vel_penalty),
			"energy":      float(energy_penalty),
		}
		return float(sum(components.values())), components

	def _is_fallen(self) -> bool:
		base_height = self.data.qpos[2]
		
		# Get base orientation — large tilt also counts as fallen
		base_quat = self.data.qpos[3:7]  # w, x, y, z
		# w component close to 1 = upright, close to 0 = tipped over
		uprightness = base_quat[0] ** 2
		
		return base_height < 0.15 or uprightness < 0.5

	# ──────────────────────────────────────────────────────────────────
	def render(self):
		if self.render_mode == "human":
			if self._viewer is None:
				self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
				self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
				self._viewer.cam.trackbodyid = 0  # track the base
			self._viewer.sync()  # pushes current self.data state to the window

		elif self.render_mode == "rgb_array":
			if self._viewer is None:
				self._viewer = mujoco.Renderer(self.model, height=480, width=640)
				self._cam = mujoco.MjvCamera()
				mujoco.mjv_defaultFreeCamera(self.model, self._cam)
				self._cam.distance = 1.5   # how far the camera sits from the robot
				self._cam.elevation = -20  # angle in degrees, negative = looking down
				self._cam.azimuth   = 130   # side view, 0 = front, 90 = side

			# Follow the robot's base position every frame
			base_pos = self.data.qpos[0:3]
			self._cam.lookat[0] = base_pos[0]  # x
			self._cam.lookat[1] = base_pos[1]  # y
			self._cam.lookat[2] = base_pos[2]  # z

			self._viewer.update_scene(self.data, camera=self._cam)
			return self._viewer.render()
	
	def close(self):
		if self._viewer is not None:
			self._viewer.close()
			self._viewer = None