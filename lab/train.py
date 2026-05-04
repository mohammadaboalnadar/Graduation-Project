import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from env import UnitreeA1Env
from pathlib import Path
import multiprocessing
import subprocess
import shutil
import threading
import time

xmlPath = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
modelsPath = r".\Models"
if not Path(modelsPath).exists():
	Path(modelsPath).mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path(xmlPath)
data = mujoco.MjData(model)

dt = float(model.opt.timestep)

#[OPTIONS]:

VERSION = "7.1"
TOTAL_TIMESTEPS = 500_000_000
CHECKPOINT_FREQ = 10_000_000  # Save a checkpoint every N timesteps
MAX_EPISODE_STEPS = 30/dt  # 30 seconds per episode
N_ENVS = 8

#[]#####################

def make_env(xml):
	def _init():
		return Monitor(UnitreeA1Env(xml, max_episode_steps=MAX_EPISODE_STEPS))
	return _init

class LogCallback(BaseCallback):
	def _on_step(self):
		# Log episode stats periodically
		# if len(self.model.ep_info_buffer) > 0 and self.n_calls % 10_000 == 0:
		# 	mean_reward = np.mean([e["r"] for e in self.model.ep_info_buffer])
		# 	mean_len    = np.mean([e["l"] for e in self.model.ep_info_buffer])
			# print(f"steps={self.num_timesteps:>8} | mean_ep_reward={mean_reward:>8.2f} | mean_ep_len={mean_len:>6.0f}")

		# Log reward components to TensorBoard every step
		infos = self.locals.get("infos", [])
		for info in infos:
			if "reward_components" in info:
				for k, v in info["reward_components"].items():
					self.logger.record(f"rewards/{k}", v)

		return True

class CheckpointCallback(BaseCallback):
	def __init__(self, save_freq, save_path, vec_normalize):
		super().__init__()
		self.save_freq     = save_freq
		self.save_path     = Path(save_path)
		self.vec_normalize = vec_normalize
		self.save_path.mkdir(parents=True, exist_ok=True)

	def _on_step(self):
		if self.model.num_timesteps % self.save_freq == 0:
			steps        = self.model.num_timesteps
			model_path   = self.save_path / f"{steps}_steps"
			vecnorm_path = self.save_path / f"{steps}_steps_vecnorm.pkl"
			self.model.save(str(model_path))
			self.vec_normalize.save(str(vecnorm_path))
			print(f"Checkpoint saved at {steps} steps → {model_path.name}")
		return True

def make_schedule(start_val: float, end_val: float, total_steps: int, completed_steps: int = 0):
    """
    Returns a schedule function that decays from start_val to end_val over
    total_steps, accounting for already-completed steps on resume.
    Progress is based on absolute timesteps so resuming mid-run works correctly.
    """
    def schedule(progress_remaining: float) -> float:
        # SB3 passes progress_remaining = 1.0 at start, 0.0 at end
        # Convert to absolute current step
        current_step = (1.0 - progress_remaining) * total_steps

        # Offset by completed steps so resumed runs continue the curve
        adjusted_step = current_step + completed_steps
        fraction = min(adjusted_step / total_steps, 1.0)

        return start_val + fraction * (end_val - start_val)

    return schedule

if __name__ == "__main__":
	# See how many cores you have
	print(f"Available cores: {multiprocessing.cpu_count()}")

	# Rule of thumb: n_envs = number of physical cores
	# Leave 1-2 cores free for the OS and main training thread
	print(f"Using {N_ENVS} envs")

	# # check if model version already exists
	# if Path(f"{modelsPath}/a1_walk_v{VERSION}.zip").exists():
	# 	response = input(f"Model [a1_walk_v{VERSION}.zip] already exists. Continue training? (y/n): ")
	# 	if response.lower() != "y":
	# 		print("Aborting training.")
	# 		exit()
	
	n_envs = N_ENVS or multiprocessing.cpu_count() - 2
	print(f"Using {n_envs} envs for training")

	# # Launch tensorboard if available (logdir relative to repo root)
	# tb_cmd = ["tensorboard", "--logdir", "./lab/tb_logs/"]
	# if shutil.which("tensorboard") is not None:
	# 	try:
	# 		subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	# 		print("TensorBoard started: tensorboard --logdir ./lab/tb_logs/")
	# 		print("Live view: http://localhost:6006/")
	# 	except Exception:
	# 		print("Failed to start TensorBoard")
	# else:
	# 	print("TensorBoard not found in PATH")

	env = SubprocVecEnv([make_env(xmlPath) for _ in range(N_ENVS)])
	env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

	# Load existing model if available, otherwise create new one
	modelExists = Path(f"{modelsPath}/a1_walk_v{VERSION}.zip").exists()
	if modelExists:
		print(f"Loading existing model a1_walk_v{VERSION}.zip")
		VecNormalize.load(f"{modelsPath}/a1_walk_v{VERSION}_vecnormalize.pkl", env)
		model = PPO.load(f"{modelsPath}/a1_walk_v{VERSION}.zip", env=env)

		model.learning_rate = make_schedule(3e-4, 1e-6, TOTAL_TIMESTEPS, model.num_timesteps)
		model.clip_range    = make_schedule(0.2,  0.02, TOTAL_TIMESTEPS, model.num_timesteps)
		model.target_kl     = 0.01
	else:
		print("Creating new model")
		model = PPO(
			"MlpPolicy",
			env,
			# ── Rollout ────────────────────────────────────────────────────
			n_steps=4096,           # larger buffer = more stable gradient estimates
			# ── Optimization ──────────────────────────────────────────────
			batch_size=512,
			n_epochs=10,             # reduced from 10 — less reuse per rollout
			# ── Schedules — the fix for every previous collapse ───────────
			learning_rate=make_schedule(5e-4, 1e-6, TOTAL_TIMESTEPS, 0),
			clip_range=make_schedule(0.2,  0.02, TOTAL_TIMESTEPS, 0),
			# ── Stability guards ──────────────────────────────────────────
			# target_kl=0.02,         # hard stop if policy drifts too far per update
			# ── Discount and GAE ──────────────────────────────────────────
			gamma=0.99,
			gae_lambda=0.95,
			# ── Entropy ───────────────────────────────────────────────────
			ent_coef=0,#0.005,         # small but nonzero — keeps exploration alive
			# ── Value function ────────────────────────────────────────────
			vf_coef=0.5,
			# max_grad_norm=0.5,      # gradient clipping — extra protection against explosions
			# ── Logging ───────────────────────────────────────────────────
			verbose=0,
			tensorboard_log="./lab/tb_logs/",
			policy_kwargs=dict(
				net_arch=[256, 256],
				# log_std_init=-0.5,  # initialise std to ~0.37 instead of default 1.0
									# smaller initial actions = less chaos in early training
			)
		)

	try:
		model.learn(
			total_timesteps=TOTAL_TIMESTEPS - model.num_timesteps,
			callback=[LogCallback(), CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=f"{modelsPath}/checkpoints/v{VERSION}", vec_normalize=env)],
			tb_log_name=f"a1_walk_v{VERSION}",
			progress_bar=True,
			reset_num_timesteps = not modelExists
		)
	except KeyboardInterrupt:
		print("Training interrupted by user. Saving model...")
	except Exception as e:
		print(f"An error occurred: {e}")
	finally:
		model.save(f"{modelsPath}/a1_walk_v{VERSION}.zip")
		env.save(f"{modelsPath}/a1_walk_v{VERSION}_vecnormalize.pkl")
		env.close()
		print(f"Model saved to {Path(f'{modelsPath}/a1_walk_v{VERSION}.zip').absolute()}")