import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
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

VERSION = "2.2"
TOTAL_TIMESTEPS = 10_000_000
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

if __name__ == "__main__":
	# See how many cores you have
	print(f"Available cores: {multiprocessing.cpu_count()}")

	# Rule of thumb: n_envs = number of physical cores
	# Leave 1-2 cores free for the OS and main training thread
	print(f"Using {N_ENVS} envs")

	# check if model version already exists
	if Path(f"{modelsPath}/a1_walk_v{VERSION}.zip").exists():
		response = input(f"Model [a1_walk_v{VERSION}.zip] already exists. Continue training? (y/n): ")
		if response.lower() != "y":
			print("Aborting training.")
			exit()
	
	n_envs = N_ENVS or multiprocessing.cpu_count() - 2
	print(f"Using {n_envs} envs for training")

	# Launch tensorboard if available (logdir relative to repo root)
	tb_cmd = ["tensorboard", "--logdir", "./lab/tb_logs/"]
	if shutil.which("tensorboard") is not None:
		try:
			subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			print("TensorBoard started: tensorboard --logdir ./lab/tb_logs/")
			print("Live view: http://localhost:6006/")
		except Exception:
			print("Failed to start TensorBoard")
	else:
		print("TensorBoard not found in PATH")

	env = SubprocVecEnv([make_env(xmlPath) for _ in range(N_ENVS)])

	# Load existing model if available, otherwise create new one
	modelExists = Path(f"{modelsPath}/a1_walk_v{VERSION}.zip").exists()
	if modelExists:
		print(f"Loading existing model a1_walk_v{VERSION}.zip")
		model = PPO.load(f"{modelsPath}/a1_walk_v{VERSION}.zip", env=env)
	else:
		print("Creating new model")
		model = PPO(
			"MlpPolicy",
			env,
			n_steps=2048,
			batch_size=64 * n_envs,
			n_epochs=10,
			gamma=0.99,
			gae_lambda=0.95,
			clip_range=0.2,
			ent_coef=0.01,
			learning_rate=3e-4,
			verbose=0,
			tensorboard_log="./lab/tb_logs/",
			policy_kwargs=dict(
				net_arch=[256, 256]  # bigger network than default [64, 64]
			)
		)

	try:
		model.learn(
			total_timesteps=TOTAL_TIMESTEPS,
			callback=LogCallback(),
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
		env.close()
		print(f"Model saved to {Path(f'{modelsPath}/a1_walk_v{VERSION}.zip').absolute()}")