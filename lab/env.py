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
		# self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
		# self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()
		self.default_dof_pos = np.array([
			-0.1, 0.8, -1.5,   # Front Right
			0.1, 0.8, -1.5,   # Front Left
			-0.1, 1.0, -1.5,   # Rear Right
			0.1, 1.0, -1.5    # Rear Left
		], dtype=np.float32)
		self.action_scale = 0.8 # deviation from default pose at max action
		self.last_action = np.zeros(n_actuators, dtype=np.float32)

		# ── Observation space ─────────────────────────────────────────
		# Base Linear Velocities								=  3
		# Base Rotational Velocities							=  3
		# Orientation Angles (roll, pitch)						=  2
		# 12 joint pos + 12 joint vel							= 24
		# last action 											= 12
		# reference velocity vector [vx, vy, wz]				=  3
		# reference height										=  1
		# ─────────────────────────────────────────────────────────────
		# Total													= 48
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
		)

		# ── Command state (randomised each episode) ───────────────────
		self.ref_vel     = np.zeros(3, dtype=np.float32)  # [vx, vy, wz] [m/s, m/s, rad/s]
		self.ref_height  = 0.28

		# ── Rendering setup ───────────────────────────────────────────
		self.render_mode = render_mode
		self._viewer: mujoco.viewer.MjViewer | mujoco.Renderer | None = None
		self._cam      = None
		self._scene    = None

	# ──────────────────────────────────────────────────────────────────
	def set_commands(self, vx: float, vy: float, wz: float, height: float):
		"""
		Override commands at runtime without resetting the episode.
		Call this from your external controller to change targets on the fly.
		"""
		self.ref_vel  = np.array([vx, vy, wz], dtype=np.float32)
		self.ref_height = height

	# ──────────────────────────────────────────────────────────────────
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
		# Randomise starting yaw so the robot learns from any orientation
		self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, self.model.nq)
		self.data.qvel[:] += self.np_random.uniform(-0.01, 0.01, self.model.nv)
		mujoco.mj_forward(self.model, self.data)

		self._step_count = 0

		# Randomise commands each episode so the robot generalises
		# self.ref_vel = np.array([3, 0, 0], dtype=np.float32)
		if self.np_random.choice([True, False]):
			self.ref_vel   = np.array([
				self.np_random.uniform(-2.5, 2.5),
				self.np_random.uniform(-1, 1),
				0
			], dtype=np.float32)
		else:
			self.ref_vel = np.zeros(3, dtype=np.float32)
		
		# if self.np_random.choice([True, False]):
		# 	self.ref_vel[2] = self.np_random.uniform(-1.0, 1.0)  # random yaw rate
		# self.ref_vel = np.array([self.np_random.uniform(0, 3.0),0,0], dtype=np.float32)

		# self.ref_height = self.np_random.uniform(0.2, 0.36)
		self.last_action = np.zeros(self.model.nu, dtype=np.float32)

		return self._get_obs(), {}

	# ──────────────────────────────────────────────────────────────────
	def step(self, action: np.ndarray):
		# Clip action to ensure it strictly stays in [-1, 1] before scaling
		action = np.clip(action, -1.0, 1.0)

		# Map [-1, 1] to [ctrl_min, ctrl_max]
		mapped_action = self.default_dof_pos + (action * self.action_scale)
		# mapped_action = self.ctrl_min + (0.5 * (action + 1.0) * (self.ctrl_max - self.ctrl_min))
		
		# Apply absolute target angles to the position actuators
		self.data.ctrl[:] = mapped_action

		for _ in range(self.n_substeps):
			mujoco.mj_step(self.model, self.data)
		self._step_count += 1

		obs                        = self._get_obs()
		reward, components         = self._compute_reward(action)
		terminated                 = self._is_fallen()
		truncated                  = self._step_count >= self.max_episode_steps

		self.last_action = action.copy()

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
		joint_pos     		= self.data.qpos[7:].copy()
		joint_vel     		= self.data.qvel[6:].copy()
		raw, pitch, roll    = self._get_euler()
		base_ang_vel  		= self.data.qvel[3:6].copy()
		base_lin_vel  		= self.data.qvel[0:3].copy()

		return np.concatenate([
			base_lin_vel / 3.0, # normalize to ~[-1, 1] range based on expected max speeds
			base_ang_vel / 2.0, # normalize to ~[-1, 1] range based on expected max speeds
			np.array([roll, pitch], dtype=np.float32) / np.pi,  # normalize angles to [-1, 1]
			(joint_pos - self.default_dof_pos) / self.action_scale,  # express joint positions as deviation from default pose, normalized to [-1, 1]
			joint_vel / 21.0, # Unitree A1 max joint speed is around 21 rad/s, so this normalizes to ~[-1, 1]
			self.last_action, # already in [-1, 1]
			
			self.ref_vel / 3.0, # [vx, vy, wz] # normalize to ~[-1, 1] range based on expected max speeds
			np.array([self.ref_height - 0.28 / 0.08], dtype=np.float32), # normalize height to ~[-1, 1] range around nominal height of 0.28m with expected variation of ±0.08m
		]).astype(np.float32)

	# ──────────────────────────────────────────────────────────────────

	def gaus(self, x: float, *alpha: float) -> float:
		return np.sum([np.exp(-a*x**2) for a in alpha]) / len(alpha)

	def _compute_reward(self, action: np.ndarray):
		# Get actual velocity in world frame and transform to robot frame
		world_vel   = self.data.qvel[0:2]           # [vx, vy] in world frame
		yaw, pitch, roll = self._get_euler()
		cos_yaw = np.cos(-yaw)
		sin_yaw = np.sin(-yaw)
		actual_vel = np.array([
			cos_yaw * world_vel[0] - sin_yaw * world_vel[1],
			sin_yaw * world_vel[0] + cos_yaw * world_vel[1]
		], dtype=np.float32)  # [vx, vy] in robot frame
		
		ref_vel    = self.ref_vel[0:2]              # [vx, vy] (already robot frame)
		actual_speed = np.linalg.norm(actual_vel)
		ref_speed = np.linalg.norm(ref_vel)

		# ── Velocity reward ───────────────────────────────────────────
		# Direction: cosine similarity between actual and target velocity
		if ref_speed > 1e-4:
			if actual_speed > 1e-4:
				cos_sim = float(np.dot(actual_vel, ref_vel) / (actual_speed * ref_speed))
			else:
				cos_sim = 0.0
		else:
			cos_sim = 1.0

		cos_sim = cos_sim / 2.0 + 0.5  # rescale from [-1, 1] to [0, 1]

		# Magnitude: gaussian peak when speed matches target
		speed_reward = self.gaus(actual_speed - ref_speed, 10.0, 0.1)

		# Angular velocity:
		ref_angular_vel = self.ref_vel[2]  # target yaw rate
		actual_angular_vel = self.data.qvel[5]  # actual yaw rate

		angular_vel_reward = self.gaus(actual_angular_vel - ref_angular_vel, 100.0, 10.0, 1.0)

		# ── Height reward ─────────────────────────────────────────────
		height_reward = self.gaus(self.data.qpos[2] - self.ref_height, 2000.0, 200.0)

		# ── Stability penalties ───────────────────────────────────────
		pose_similarity = -float(np.sum(np.square(self.data.qpos[7:] - self.default_dof_pos)))
		action_rate_penalty = -float(np.sum(np.square(action - self.last_action))) * 0.1
		vertical_vel    = -float(self.data.qvel[2])**2
		pitch_error, roll_error = -pitch**2, -roll**2

		# torques = self.data.qfrc_actuator
		# energy_penalty   = -1e-5 * float(np.sum(np.square(torques)))

		# ── Symmetry penalty (encourage diagonal pairs to do similar things) ─────────────────────
		# Extract the 12 physical joint positions
		q = self.data.qpos[7:19].copy()
		q_FR, q_FL, q_RR, q_RL = q[0:3], q[3:6], q[6:9], q[9:12]

		# Invert right hips for mirroring
		q_FR[0] *= -1
		q_RR[0] *= -1

		# Calculate symmetry for every pair of legs
		diagonal_symmetry_penalty = -(
			float(np.sum(np.square(q_FR - q_RL))) +
			float(np.sum(np.square(q_FL - q_RR)))
		)
		# horizontal_symmetry_penalty = -0.2 * (
		# 	float(np.sum(np.square(q_FR - q_FL))) +
		# 	float(np.sum(np.square(q_RR - q_RL)))
		# )

		# symmetry_penalty = max(diagonal_symmetry_penalty, horizontal_symmetry_penalty)
		symmetry_penalty = diagonal_symmetry_penalty

		# ── Combine everything ─────────────────────────────────────────────

		penalty_multiplier = np.exp(
			(0.3 * pose_similarity) +
			(0.5 * vertical_vel) +
			(0.5 * pitch_error) +
			(0.5 * roll_error) +
			(0.5 * symmetry_penalty) +
			(0.05 * action_rate_penalty)
		)

		total_reward = (
			1.0 * cos_sim * speed_reward +
			0.2 * angular_vel_reward +
			0.1 * height_reward
		)
		
		# ── Fall penalty ──────────────────────────────────────────────
		fall_penalty = -1 if self._is_fallen() else 0.0

		return total_reward * penalty_multiplier + fall_penalty, {
			"velocity_direction": cos_sim,
			"velocity_magnitude": speed_reward,
			"angular_velocity": angular_vel_reward,
			"height": height_reward,
			"pose_similarity": pose_similarity,
			"action_rate": action_rate_penalty,
			"vertical_velocity": vertical_vel,
			"a_pitch_error": pitch_error,
			"a_roll_error": roll_error,
			# "energy_penalty": energy_penalty,
			"symmetry_penalty": symmetry_penalty,
			"fall_penalty": fall_penalty,
			"_total_reward": total_reward,
			"_penalty_multiplier": penalty_multiplier
		}

	# ──────────────────────────────────────────────────────────────────
	def _is_fallen(self) -> bool:
		yaw, pitch, roll = self._get_euler()
		if abs(pitch) > np.radians(25) or abs(roll) > np.radians(25):
			return True
		z = self.data.qpos[2]
		if z < 0.18 or z > 0.5:
			return True
		
		return False

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
		
		yaw, pitch, roll = self._get_euler()
		cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
		world_ref_vel = np.array([
			self.ref_vel[0] * cos_yaw - self.ref_vel[1] * sin_yaw,
			self.ref_vel[0] * sin_yaw + self.ref_vel[1] * cos_yaw
		], dtype=np.float32)  # [vx, vy] in robot frame
		self._add_arrow(
			start  = base_pos + np.array([0, 0, 0.1]),  # above the robot
			dir    = np.array([
				world_ref_vel[0],
				world_ref_vel[1],
				0.0
			]),
			radius = 0.01,
			rgba   = np.array([0.0, 1.0, 0.0, 0.4], dtype=np.float32)  # green
		)

		# ── Actual velocity arrow (red) ─────────────────────────────────────
		world_vel   = self.data.qvel[0:2].copy()
		self._add_arrow(
			start  = base_pos + np.array([0, 0, 0.1]),  # above the robot
			dir    = np.array([
				world_vel[0],
				world_vel[1],
				0.0
			]),
			radius = 0.01,
			rgba   = np.array([1.0, 0.0, 0.0, 0.4], dtype=np.float32)  # red
		)

		# ── Target ang_vel arrow (blue) ───────────────────────────────────
		body_right_vector = np.array([-sin_yaw, cos_yaw, 0], dtype=np.float32)  # points to the robot's right side
		heading_dir = body_right_vector / np.linalg.norm(body_right_vector)
		self._add_arrow(
			start  = base_pos + np.array([0, 0, 0.15]),
			dir    = heading_dir * self.ref_vel[2],  # point left or right based on sign of target yaw rate
			radius = 0.01,
			rgba   = np.array([0.0, 0.0, 1.0, 0.4], dtype=np.float32)  # blue
		)

		# ── Actual ang_vel arrow (red) ─────────────────────────────────────────────
		actual_angular_vel = self.data.qvel[5]  # actual yaw rate
		self._add_arrow(
			start  = base_pos + np.array([0, 0, 0.15]),
			dir    = heading_dir * actual_angular_vel,  # point left or right based on sign of actual yaw rate
			radius = 0.01,
			rgba   = np.array([1.0, 0.0, 0.0, 0.4], dtype=np.float32)  # red
		)

		# ── Target height marker (cyan) ─────────────────────────────────────────────
		self._add_arrow(
			start  = np.array([base_pos[0] - 0.02, base_pos[1] - 0.32, 0], dtype=np.float32),
			dir    = np.array([0, 0, self.ref_height * 2], dtype=np.float32),
			radius = 0.01,
			rgba   = np.array([0.0, 0.0, 1.0, 0.4], dtype=np.float32)  # blue
		)

		# ── Actual height marker (red) ─────────────────────────────────────────────
		self._add_arrow(
			start  = np.array([base_pos[0], base_pos[1] - 0.3, 0], dtype=np.float32),
			dir    = np.array([0, 0, base_pos[2] * 2], dtype=np.float32),
			radius = 0.01,
			rgba   = np.array([1.0, 0.0, 0.0, 0.4], dtype=np.float32)  # red
		)

	def _add_arrow(self, start: np.ndarray, dir: np.ndarray, length: float = 1, radius: float = 0.01, rgba: np.ndarray = np.array([1.0, 0.0, 0.0, 0.4], dtype=np.float32)):
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