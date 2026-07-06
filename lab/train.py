import os

import wandb; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import torch

# ---------------------------------------------------------------------
# SB3 / PYTORCH 2.4+ STREAM FIX
# Intercepts save/load to force memory buffering for file-like objects
# ---------------------------------------------------------------------
_original_save = torch.save
_original_load = torch.load

def _safe_save(obj, f, *args, **kwargs):
    if hasattr(f, "write"):
        buffer = io.BytesIO()
        _original_save(obj, buffer, *args, **kwargs)
        f.write(buffer.getvalue())
    else:
        _original_save(obj, f, *args, **kwargs)

def _safe_load(f, *args, **kwargs):
    if hasattr(f, "read"):
        buffer = io.BytesIO(f.read())
        return _original_load(buffer, *args, **kwargs)
    return _original_load(f, *args, **kwargs)

torch.save = _safe_save
torch.load = _safe_load
# ---------------------------------------------------------------------

import mujoco
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from env import UnitreeA1Env
from pathlib import Path
import multiprocessing

xmlPath = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
modelsPath = r".\Models"
if not Path(modelsPath).exists():
	Path(modelsPath).mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path(xmlPath)
data = mujoco.MjData(model)

dt = float(model.opt.timestep)

#[OPTIONS]:

VERSION = "19.3"
TOTAL_TIMESTEPS = 1_500_000_000
CHECKPOINT_FREQ = 10_000_000  # Save a checkpoint every N timesteps
MAX_EPISODE_STEPS = 10*50 # N seconds at 50Hz
N_ENVS = 8

SCHEDULES_UPDATE_FREQ = 10_000
SCHEDULES = {
	"pose": {"start": 900_000_000, "end": 1_000_000_000},
	"hip_pose": {"start": 750_000_000, "end": 900_000_000},
	"vertical_velocity": {"start": 50_000_000, "end": 200_000_000},
	"orientation": {"start": 200_000_000, "end": 400_000_000},
	"symmetry": {"start": 300_000_000, "end": 500_000_000},
	"action_accel": {"start": 500_000_000, "end": 750_000_000}
}

LR_START = 3e-4
LR_END = 1e-5
LR_END_FRACTION = 0.25
HYPERPARAMS = {
	# "use_sde":True,          # better exploration in continuous action spaces
	# "sde_sample_freq":16,   # how many steps to wait before resampling noise
	# ── Rollout ───────────────────────────────────────────────────
	"n_steps":2**16,           # larger buffer = more stable gradient estimates
	# ── Optimization ──────────────────────────────────────────────
	"batch_size":2**12,
	"n_epochs":20,
	"learning_rate":2e-4,#LinearSchedule(LR_START, LR_END, LR_END_FRACTION),
	# "clip_range":#LinearSchedule(0.2,  0.02, TOTAL_TIMESTEPS, 0),
	"clip_range_vf":0.2,
	# ── Stability guards ──────────────────────────────────────────
	"target_kl":0.02,         # hard stop if policy drifts too far per update
	"max_grad_norm":0.5,      # gradient clipping — extra protection against explosions
	# ── Discount and GAE ──────────────────────────────────────────
	"gamma":0.99,
	"gae_lambda":0.95,
	# ── Entropy ───────────────────────────────────────────────────
	"ent_coef":0.01,         # small but nonzero — keeps exploration alive
	# ── Value function ────────────────────────────────────────────
	"vf_coef":0.5,
	# ── Policy kwargs ─────────────────────────────────────────────
	"policy_kwargs":{
		"net_arch":[256, 256],
		# "activation_fn":torch.nn.ELU,
		# "use_expln":True,
		# "squash_output":True,
		# "full_std":True,

		# "log_std_init":-2.0
	}
}

#[]#####################

def make_env(xml):
	def _init():
		return Monitor(UnitreeA1Env(xml, max_episode_steps=MAX_EPISODE_STEPS))
	return _init

class LogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_rewards = {}
        self.step_counts = {}

    def _on_step(self):
        infos = self.locals.get("infos", [])
        
        if infos and "reward_components" in infos[0]:
            # Accumulate the components across all parallel environments
            for info in infos:
                if "reward_components" in info:
                    for key, val in info["reward_components"].items():
                        if key not in self.rollout_rewards:
                            self.rollout_rewards[key] = 0.0
                            self.step_counts[key] = 0
                        self.rollout_rewards[key] += val
                        self.step_counts[key] += 1

        return True

    def _on_rollout_end(self):
        # Calculate the true mean of the entire batch at rollout end
        if self.step_counts:
            for key in self.rollout_rewards.keys():
                if self.step_counts[key] > 0:
                    avg_val = self.rollout_rewards[key] / self.step_counts[key]
                    self.logger.record(f"rewards/{key}", avg_val)
            
            # Reset the accumulators for the next rollout
            self.rollout_rewards = {}
            self.step_counts = {}

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
			print(f"Checkpoint saved at {steps} steps → {model_path.absolute()}")
		return True
	
class ModularCurriculumCallback(BaseCallback):
    def __init__(self, schedules: dict, update_freq: int = 10_000, verbose=0):
        super().__init__(verbose)
        self.schedules = schedules
        self.update_freq = update_freq
        self.current_fades = {k: 0.0 for k in schedules.keys()}

    def _on_step(self) -> bool:
        # Throttle IPC overhead: Only broadcast every N steps
        if self.num_timesteps % self.update_freq != 0:
            return True

        step = self.num_timesteps

        for key, bounds in self.schedules.items():
            start, end = bounds["start"], bounds["end"]
            
            # Calculate linear interpolation
            if step <= start:
                fade = 0.0
            elif step >= end:
                fade = 1.0
            else:
                fade = (step - start) / (end - start)
                
            self.current_fades[key] = fade
            
            # Log to TensorBoard so you can see the curves
            self.logger.record(f"curriculum/{key}_fade", fade)

        # Broadcast the updated dictionary to all isolated environments
        self.training_env.env_method("set_penalty_fades", self.current_fades)
        
        return True

if __name__ == "__main__":
	# See how many cores you have
	print(f"Available cores: {multiprocessing.cpu_count()}")

	# Rule of thumb: n_envs = number of physical cores
	# Leave 1-2 cores free for the OS and main training thread
	print(f"Using {N_ENVS} envs")

	run = wandb.init(
		project="unitree_a1_rl",
		name=f"a1_walk_v{VERSION}",
		config={
			# Architecture
			"version": VERSION,
			"n_envs": N_ENVS,
			"max_episode_steps": MAX_EPISODE_STEPS,

			# PPO Hyperparameters
			"lr_start": LR_START,
			"lr_end": LR_END,
			"lr_end_fraction": LR_END_FRACTION,
			**{k: str(v) for k, v in HYPERPARAMS.items()}
		},
		sync_tensorboard=True
	)
	wandb.define_metric("*", step_metric="global_step")

	env = SubprocVecEnv([make_env(xmlPath) for _ in range(N_ENVS)])
	env = VecNormalize(env, norm_obs=False, gamma=HYPERPARAMS.get("gamma", 0.99), norm_reward=True, clip_reward=10.0)

	# Load existing model if available, otherwise create new one
	modelExists = Path(f"{modelsPath}/a1_walk_v{VERSION}.zip").exists()
	if modelExists:
		print(f"Loading existing model a1_walk_v{VERSION}.zip")
		env = VecNormalize.load(f"{modelsPath}/a1_walk_v{VERSION}_vecnormalize.pkl", env)
		model = PPO.load(f"{modelsPath}/a1_walk_v{VERSION}.zip", env=env)#, custom_objects={
		# 	"learning_rate": get_linear_fn(3e-4, 1e-5, 1),
		# 	"ent_coef": 0.01
		# })
	else:
		print("Creating new model")
		model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./lab/tb_logs/", **HYPERPARAMS)
	try:
		model.learn(
			total_timesteps=TOTAL_TIMESTEPS - model.num_timesteps,
			callback=[
				LogCallback(),
				ModularCurriculumCallback(schedules=SCHEDULES, update_freq=SCHEDULES_UPDATE_FREQ),
				CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=f"{modelsPath}/checkpoints/v{VERSION}", vec_normalize=env)
			],
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
		print(f"Model saved to {Path(f'{modelsPath}/a1_walk_v{VERSION}.zip').absolute()}")
		env.save(f"{modelsPath}/a1_walk_v{VERSION}_vecnormalize.pkl")
		env.close()
		print(f"VecNormalize saved to {Path(f'{modelsPath}/a1_walk_v{VERSION}_vecnormalize.pkl').absolute()}")
		run.finish()
		print("WandB run closed.")