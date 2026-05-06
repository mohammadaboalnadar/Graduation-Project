import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco, mujoco.viewer

class UnitreeA1Env(gym.Env):

	def __init__(self, xml_path: str, max_episode_steps: int = 1000, n_substeps=10, render_mode=None):
		super().__init__()

		self.model = mujoco.MjModel.from_xml_path(xml_path)
		self.data  = mujoco.MjData(self.model)

		self.n_substeps = n_substeps
		self.dt = self.model.opt.timestep * self.n_substeps

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

		n_actuators = self.model.nu
		self.action_space = spaces.Box(
			low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32
		)
		self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
		self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()
		self.default_dof_pos = np.array([
			-0.1, 0.8, -1.5,   # Front Right
			0.1, 0.8, -1.5,   # Front Left
			-0.1, 1.0, -1.5,   # Rear Right
			0.1, 1.0, -1.5    # Rear Left
		], dtype=np.float32)
		self.action_scale = 0.25 # 0.25 deviation from default pose at max action

		# ── Observation space ─────────────────────────────────────────
		# 12 joint pos + 12 joint vel							= 24
		# base quaternion (4) + ang vel (3) + lin vel (3)		= 10
		# foot contacts											=  4
		# target velocity vector [vx, vy]						=  2
		# target base quaternion								=  4
		# target height											=  1
		# ─────────────────────────────────────────────────────────────
		# Total													= 45
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32
		)

		# ── Command state (randomised each episode) ───────────────────
		self.target_vel     = np.zeros(2, dtype=np.float32)  # [vx, vy] m/s
		self.target_yaw     = 0.0                            # radians
		self.target_pitch   = 0.0    
		self.target_roll    = 0.0
		self.target_quat	= np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
		self.target_height  = 0.35

		# ── Rendering setup ───────────────────────────────────────────
		self.render_mode = render_mode
		self._viewer: mujoco.viewer.MjViewer | mujoco.Renderer | None = None
		self._cam      = None
		self._scene    = None

	# ──────────────────────────────────────────────────────────────────
	def set_commands(self, vx: float, vy: float,
					yaw: float, pitch: float = 0.0, roll: float = 0.0):
		"""
		Override commands at runtime without resetting the episode.
		Call this from your external controller to change targets on the fly.
		"""
		self.target_vel  = np.array([vx, vy], dtype=np.float32)
		self.target_yaw  = float(yaw)
		self.target_pitch = float(pitch)
		self.target_roll = float(roll)
		self.target_quat = self._quat_from_yaw_pitch_roll(yaw, pitch, roll)

	# ──────────────────────────────────────────────────────────────────
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
		self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, self.model.nq)
		self.data.qvel[:] += self.np_random.uniform(-0.01, 0.01, self.model.nv)
		# Randomise starting yaw so the robot learns from any orientation
		# start_yaw = self.np_random.uniform(-np.pi, np.pi)
		# cy, sy = np.cos(start_yaw / 2), np.sin(start_yaw / 2)
		# self.data.qpos[3:7] = [cy, 0.0, 0.0, sy]  # w, x, y, z — pure yaw quaternion
		mujoco.mj_forward(self.model, self.data)

		self._step_count = 0

		# Randomise commands each episode so the robot generalises
		# speed         = self.np_random.uniform(0, 1.2)
		# direction     = self.np_random.uniform(-np.pi, np.pi)
		# self.target_vel   = np.array([
		# 	speed * np.cos(direction),
		# 	speed * np.sin(direction),
		# ], dtype=np.float32)
		# self.target_yaw   = self.np_random.uniform(-np.pi, np.pi)
		# self.target_pitch = self.np_random.uniform(-0.3, 0.3)  # ~±17 degrees
		# self.target_roll  = self.np_random.uniform(-0.2, 0.2)  # ~±11 degrees
		# self.target_quat  = self._quat_from_yaw_pitch_roll(self.target_yaw, self.target_pitch, self.target_roll)
		self.target_height = self.np_random.uniform(0.15, 0.35)

		return self._get_obs(), {}

	# ──────────────────────────────────────────────────────────────────
	def step(self, action: np.ndarray):
		# Clip action to ensure it strictly stays in [-1, 1] before scaling
		action = np.clip(action, -1.0, 1.0)

		# Map [-1, 1] to [ctrl_min, ctrl_max]
		mapped_action = self.default_dof_pos + (action * self.action_scale)
		
		# Apply absolute target angles to the position actuators
		self.data.ctrl[:] = mapped_action

		for _ in range(self.n_substeps):
			mujoco.mj_step(self.model, self.data)
		self._step_count += 1

		obs                        = self._get_obs()
		reward, components         = self._compute_reward(action)
		terminated                 = self._is_fallen()
		truncated                  = self._step_count >= self.max_episode_steps

		return obs, reward, terminated, truncated, {"reward_components": components}

	# ──────────────────────────────────────────────────────────────────
	def _get_euler(self):
		"""Extract yaw and pitch from base quaternion."""
		qw, qx, qy, qz = self.data.qpos[3:7]
		yaw   = np.arctan2(2*(qw*qz + qx*qy),
						1 - 2*(qy**2 + qz**2))
		pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
		roll  = np.arctan2(2*(qw*qx + qy*qz),
						1 - 2*(qx**2 + qy**2))
		return float(yaw), float(pitch), float(roll)

	def _angle_error(self, current: float, target: float):
		"""Signed angle error wrapped to [-pi, pi], returned as (sin, cos)."""
		err = (current - target + np.pi) % (2 * np.pi) - np.pi
		return np.array([np.sin(err), np.cos(err)], dtype=np.float32), float(err)
	
	def _quat_from_yaw_pitch_roll(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
		"""Build quaternion from yaw, pitch, roll (ZYX rotation order)."""
		cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
		cp, sp = np.cos(pitch/2), np.sin(pitch/2)
		cr, sr = np.cos(roll/2),  np.sin(roll/2)

		return np.array([
			cr*cp*cy + sr*sp*sy,  # w
			sr*cp*cy - cr*sp*sy,  # x
			cr*sp*cy + sr*cp*sy,  # y
			cr*cp*sy - sr*sp*cy,  # z
		], dtype=np.float32)

	# ──────────────────────────────────────────────────────────────────
	def _get_obs(self) -> np.ndarray:
		joint_pos     = self.data.qpos[7:].copy()
		joint_vel     = self.data.qvel[6:].copy()
		base_quat     = self.data.qpos[3:7].copy()
		base_ang_vel  = self.data.qvel[3:6].copy()
		base_lin_vel  = self.data.qvel[0:3].copy()
		foot_contacts = self._get_foot_contacts()

		return np.concatenate([
			joint_pos, joint_vel,
			base_quat, base_ang_vel, base_lin_vel,
			foot_contacts,
			self.target_vel,   # [vx, vy]
			self.target_quat,  # [w, x, y, z]
			np.array([self.target_height], dtype=np.float32)
		]).astype(np.float32)

	# ──────────────────────────────────────────────────────────────────
	def _get_foot_contacts(self) -> np.ndarray:
		contacts      = np.zeros(4, dtype=np.float32)
		foot_id_list  = list(self._foot_geom_ids)
		for i, geom_id in enumerate(foot_id_list):
			for contact in self.data.contact:
				if geom_id in (contact.geom1, contact.geom2):
					contacts[i] = 1.0
					break
		return contacts

	# ──────────────────────────────────────────────────────────────────
	def _compute_reward(self, action: np.ndarray):
		actual_vel   = self.data.qvel[0:2]          # [vx, vy]
		actual_speed = np.linalg.norm(actual_vel)
		target_speed = np.linalg.norm(self.target_vel)

		# ── Velocity reward ───────────────────────────────────────────
		# Direction: cosine similarity between actual and target velocity
		if actual_speed > 1e-4 and target_speed > 1e-4:
			cos_sim = float(np.dot(actual_vel, self.target_vel) /
							(actual_speed * target_speed))
		else:
			cos_sim = 1.0

		# Magnitude: gaussian peak when speed matches target
		speed_reward  = float(np.exp(-20.0 * (actual_speed - target_speed) ** 2))

		# ── Heading rewards ───────────────────────────────────────────
		quat_similarity = float(np.dot(self.data.qpos[3:7], self.target_quat) ** 2)

		# ── Stability penalties ───────────────────────────────────────
		vertical_vel    = abs(float(self.data.qvel[2]))
		ang_vel         = self.data.qvel[3:6]
		# ang_vel_penalty = -0.05 * float(np.sum(np.square(ang_vel)))
		# vertical_penalty = -0.1 * vertical_vel

		torques = self.data.qfrc_actuator
		energy_penalty   = -2e-4 * float(np.sum(np.square(torques)))

		# ── Fall penalty ──────────────────────────────────────────────
		fall_penalty = -1000.0 if self._is_fallen() else 0.0

		# ── Height reward ─────────────────────────────────────────────
		height_reward = float(np.exp(-1000.0 * (self.data.qpos[2] - self.target_height) ** 2))

		components = {
			"vel_direction": cos_sim,
			"vel_magnitude": speed_reward,
			"heading":        quat_similarity,
			"height":         height_reward,
			# "ang_vel":       ang_vel_penalty,
			# "vertical":      vertical_penalty,
			"energy":        energy_penalty,
			"fall":          fall_penalty,
			"goal_product":  cos_sim * speed_reward * quat_similarity * height_reward * 4,
		}

		rewards = {
			# "ang_vel": components["ang_vel"],
			# "vertical": components["vertical"],
			"energy": components["energy"],
			"fall": components["fall"],
			"goal_product": components["goal_product"],
			"alive": 4.0
		}

		return float(sum(rewards.values())), components

	# ──────────────────────────────────────────────────────────────────
	def _is_fallen(self) -> bool:
		base_height  = self.data.qpos[2]
		base_quat    = self.data.qpos[3:7]
		uprightness  = base_quat[0] ** 2
		return bool(base_height < 0.12 or uprightness < 0.5)

	# ──────────────────────────────────────────────────────────────────
	def render(self):
		if self.render_mode == "human":
			if self._viewer is None:
				self._viewer = mujoco.viewer.launch_passive(
					self.model, self.data,
					show_left_ui=True, show_right_ui=True
				)
				self._viewer.cam.type       = mujoco.mjtCamera.mjCAMERA_TRACKING
				self._viewer.cam.trackbodyid = 0
				self._scene = self._viewer.user_scn
			
			# Reset custom geom count then re-add each frame
			self._scene.ngeom = 0
			self._add_debug_visuals()
			self._viewer.sync()

		elif self.render_mode == "rgb_array":
			if self._viewer is None:
				self._viewer = mujoco.Renderer(self.model, height=480, width=640)
				self._cam      = mujoco.MjvCamera()
				mujoco.mjv_defaultFreeCamera(self.model, self._cam)
				self._cam.distance = 1.5   # how far the camera sits from the robot
				self._cam.elevation = -20  # angle in degrees, negative = looking down
				self._cam.azimuth   = 130   # side view, 0 = front, 90 = side

			# Follow the robot's base position every frame
			base_pos = self.data.qpos[0:3]
			self._cam.lookat[:] = base_pos

			self._viewer.update_scene(self.data, camera=self._cam)

			# Access the renderer's internal scene after update_scene populates it
			self._scene = self._viewer.scene
			self._add_debug_visuals()

			return self._viewer.render()
	
	# ──────────────────────────────────────────────────────────────────
	def _add_debug_visuals(self):
		"""Add debug arrows for target velocity and target heading to the scene."""
		base_pos = self.data.qpos[0:3].copy()

		# ── Target velocity arrow (green) ────────────────────────────────
		vel_magnitude = np.linalg.norm(self.target_vel)
		if vel_magnitude > 1e-4:
			vel_dir_3d = np.array([
				self.target_vel[0] / vel_magnitude,
				self.target_vel[1] / vel_magnitude,
				0.0
			])
			self._add_arrow(
				start  = base_pos + np.array([0, 0, 0.1]),  # above the robot
				dir    = vel_dir_3d,
				length = float(vel_magnitude),              # scale length to speed
				radius = 0.01,
				rgba   = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)  # green
			)

		# ── Target heading arrow (blue) ───────────────────────────────────
		heading_dir = np.array([
			np.cos(self.target_yaw),
			np.sin(self.target_yaw),
			np.tan(self.target_pitch)   # z component encodes pitch
		])
		heading_dir /= np.linalg.norm(heading_dir)
		self._add_arrow(
			start  = base_pos + np.array([0, 0, 0.1]),
			dir    = heading_dir,
			length = 0.5,
			radius = 0.01,
			rgba   = np.array([0.0, 0.4, 1.0, 0.8], dtype=np.float32)  # blue
		)

	def _add_arrow(self, start: np.ndarray, dir: np.ndarray, length: float, radius: float, rgba: np.ndarray):
		scene = self._scene
		if scene.ngeom >= scene.maxgeom:
			return

		g = scene.geoms[scene.ngeom]
		mujoco.mjv_initGeom(
			g,
			mujoco.mjtGeom.mjGEOM_ARROW,
			np.zeros(3),
			np.zeros(3),
			np.eye(3).flatten(),
			rgba.astype(np.float32)
		)

		from_ = start.astype(np.float64)
		to_   = (start + dir * length).astype(np.float64)

		mujoco.mjv_connector(
			g,
			mujoco.mjtGeom.mjGEOM_ARROW,
			radius,
			from_,
			to_
		)

		scene.ngeom += 1
	
	# ──────────────────────────────────────────────────────────────────

	def close(self):
		if self._viewer is not None:
			self._viewer.close()
			self._viewer = None
		if self._viewer is not None:
			self._viewer.close()
			self._viewer = None
		self._scene = None