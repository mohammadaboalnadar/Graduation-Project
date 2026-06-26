import os
# python lab\optimize_ppo.py --study-name "a1_walk_ppo_optuna_long" --tensorboard-log "D:\Files\Scripts\py\Graduation Project\lab\tb_logs\optuna3" --n-trials 300 --warmup-trials 0 --refine-trials 0 --final-timesteps 20000000
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import io
import json
import multiprocessing
from pathlib import Path

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, sync_envs_normalization

from env import UnitreeA1Env

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


ROOT_DIR = Path(__file__).resolve().parents[1]
XML_PATH = ROOT_DIR / "external" / "mujoco_menagerie" / "unitree_a1" / "scene.xml"
MODELS_DIR = ROOT_DIR / "Models"
OPTUNA_DIR = MODELS_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

N_ENVS = 8
DEFAULT_N_TRIALS = 50
DEFAULT_N_EVALUATIONS = 20
DEFAULT_N_EVAL_EPISODES = 10
DEFAULT_WARMUP_TRIALS = 10
DEFAULT_WARMUP_TIMESTEPS = 500_000
DEFAULT_REFINE_TRIALS = 25
DEFAULT_REFINE_TIMESTEPS = 3_000_000
DEFAULT_FINAL_TIMESTEPS = 15_000_000
DEFAULT_MAX_EPISODE_STEPS = 20 * 50
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TENSORBOARD_LOG = ROOT_DIR / "lab" / "tb_logs" / "optuna"
DEFAULT_TENSORBOARD_LOG.mkdir(parents=True, exist_ok=True)


def build_storage_url(db_path: Path) -> str:
	# Ensure parent directory exists (support custom storage paths)
	db_path.parent.mkdir(parents=True, exist_ok=True)
	# Use plain absolute posix path (no percent-encoding) so sqlite can open the file on Windows
	return f"sqlite:///{db_path.resolve().as_posix()}"


def make_env(xml_path: Path, max_episode_steps: int):
	def _init():
		return Monitor(UnitreeA1Env(str(xml_path), max_episode_steps=max_episode_steps))

	return _init


def sample_ppo_params(trial: optuna.Trial) -> dict:
	return {
		"use_sde": True,
		"sde_sample_freq": trial.suggest_categorical("sde_sample_freq_pow", [-1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
		"n_steps": 2**trial.suggest_int("n_steps_pow", 8, 20),
		"batch_size": 2**trial.suggest_int("batch_size_pow", 4, 14),
		"n_epochs": trial.suggest_int("n_epochs", 3, 20),
		"gamma": trial.suggest_float("gamma", 0.95, 0.9999),
		"gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
		"learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True),
		"ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
	}


def get_trial_timesteps(trial_number: int, warmup_trials: int, warmup_timesteps: int, refine_trials: int, refine_timesteps: int, final_timesteps: int) -> tuple[int, str]:
	if trial_number < warmup_trials:
		return warmup_timesteps, "warmup"
	if trial_number < warmup_trials + refine_trials:
		return refine_timesteps, "refine"
	return final_timesteps, "final"


class TrialEvalCallback(EvalCallback):
	def __init__(self, eval_env, trial: optuna.Trial, **kwargs):
		super().__init__(eval_env=eval_env, **kwargs)
		self.trial = trial
		self.eval_idx = 0
		self.is_pruned = False

	def _on_step(self) -> bool:
		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			train_env = self.model.get_env()
			if train_env is not None:
				sync_envs_normalization(train_env, self.eval_env)
			super()._on_step()
			self.eval_idx += 1
			self.trial.report(self.last_mean_reward, self.eval_idx)
			if self.trial.should_prune():
				self.is_pruned = True
				return False
		return True


def build_model(trial: optuna.Trial, max_episode_steps: int, tensorboard_log: str, timesteps_per_trial: int):
	hyperparams = {
		"policy": "MlpPolicy",
		"batch_size": DEFAULT_BATCH_SIZE,
		"target_kl": 0.02,
		"ent_coef": 0.0,
		"vf_coef": 0.5,
		"max_grad_norm": 0.5,
		"verbose": 0,
		"tensorboard_log": tensorboard_log,
		"policy_kwargs": dict(
			net_arch=[256, 256],
			squash_output=True,
			full_std=True,
		),
	}
	hyperparams.update(sample_ppo_params(trial))

	train_env = SubprocVecEnv(
		[make_env(XML_PATH, max_episode_steps) for _ in range(N_ENVS)],
		start_method="spawn",
	)
	train_env = VecNormalize(
		train_env,
		norm_obs=False,
		norm_reward=True,
		clip_reward=10.0,
		gamma=hyperparams["gamma"],
	)

	eval_env = SubprocVecEnv(
		[make_env(XML_PATH, max_episode_steps)],
		start_method="spawn",
	)
	eval_env = VecNormalize(
		eval_env,
		norm_obs=False,
		norm_reward=False,
		training=False,
		clip_reward=10.0,
		gamma=hyperparams["gamma"],
	)

	model = PPO(env=train_env, **hyperparams)
	eval_freq = max(timesteps_per_trial // DEFAULT_N_EVALUATIONS // N_ENVS, 1)
	eval_callback = TrialEvalCallback(
		eval_env,
		trial,
		n_eval_episodes=DEFAULT_N_EVAL_EPISODES,
		eval_freq=eval_freq,
		deterministic=True,
		verbose=0,
	)

	return model, train_env, eval_env, eval_callback


def objective(
	trial: optuna.Trial,
	max_episode_steps: int,
	tensorboard_log: str,
	warmup_trials: int,
	warmup_timesteps: int,
	refine_trials: int,
	refine_timesteps: int,
	final_timesteps: int,
) -> float:
	timesteps_per_trial, stage_name = get_trial_timesteps(
		trial.number,
		warmup_trials,
		warmup_timesteps,
		refine_trials,
		refine_timesteps,
		final_timesteps,
	)
	trial.set_user_attr("trial_stage", stage_name)
	trial.set_user_attr("timesteps_per_trial", timesteps_per_trial)

	model, train_env, eval_env, eval_callback = build_model(trial, max_episode_steps, tensorboard_log, timesteps_per_trial)
	nan_encountered = False
	try:
		model.learn(timesteps_per_trial, callback=eval_callback)
	except (AssertionError, ValueError, FloatingPointError) as exc:
		print(exc)
		nan_encountered = True
	finally:
		train_env.close()
		eval_env.close()

	if nan_encountered:
		return float("nan")

	if eval_callback.is_pruned:
		raise optuna.exceptions.TrialPruned()

	return float(eval_callback.last_mean_reward)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Unitree A1 PPO training.")
	parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Maximum number of Optuna trials to run.")
	parser.add_argument(
		"--warmup-trials",
		type=int,
		default=DEFAULT_WARMUP_TRIALS,
		help="How many initial trials use the short warmup budget.",
	)
	parser.add_argument(
		"--warmup-timesteps",
		type=int,
		default=DEFAULT_WARMUP_TIMESTEPS,
		help="Training budget for the early warmup trials.",
	)
	parser.add_argument(
		"--refine-trials",
		type=int,
		default=DEFAULT_REFINE_TRIALS,
		help="How many middle trials use the intermediate refine budget.",
	)
	parser.add_argument(
		"--refine-timesteps",
		type=int,
		default=DEFAULT_REFINE_TIMESTEPS,
		help="Training budget for the middle refine trials.",
	)
	parser.add_argument(
		"--final-timesteps",
		type=int,
		default=DEFAULT_FINAL_TIMESTEPS,
		help="Training budget for later-stage trials.",
	)
	parser.add_argument(
		"--max-episode-steps",
		type=int,
		default=DEFAULT_MAX_EPISODE_STEPS,
		help="Fixed episode length for both train and eval environments.",
	)
	parser.add_argument(
		"--study-name",
		type=str,
		default="a1_walk_ppo_optuna",
		help="Name of the Optuna study to create or resume.",
	)
	parser.add_argument(
		"--storage",
		type=str,
		default=build_storage_url(OPTUNA_DIR / "a1_walk_ppo_optuna.db"),
		help="Optuna storage URL. Defaults to a local SQLite database for resumable runs.",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=None,
		help="Optional wall-clock timeout in seconds for study.optimize.",
	)
	parser.add_argument(
		"--tensorboard-log",
		type=str,
		default=str(DEFAULT_TENSORBOARD_LOG),
		help="TensorBoard log directory for the trial training runs.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print(f"Available cores: {multiprocessing.cpu_count()}")
	print(f"Using {N_ENVS} envs for each trial")
	print(
		"Trial budgets: "
		f"warmup={args.warmup_timesteps} x {args.warmup_trials}, "
		f"refine={args.refine_timesteps} x {args.refine_trials}, "
		f"final={args.final_timesteps}"
	)
	print(f"Episode cap: {args.max_episode_steps}")
	print(f"Storage: {args.storage}")

	torch.set_num_threads(1)

	sampler = TPESampler(n_startup_trials=min(10, args.n_trials), multivariate=True)
	pruner = MedianPruner(n_startup_trials=0, n_warmup_steps=max(DEFAULT_N_EVALUATIONS // 3, 1))
	study = optuna.create_study(
		direction="maximize",
		study_name=args.study_name,
		storage=args.storage,
		load_if_exists=True,
		sampler=sampler,
		pruner=pruner,
	)

	try:
		study.optimize(
			lambda trial: objective(
				trial,
				args.max_episode_steps,
				args.tensorboard_log,
				args.warmup_trials,
				args.warmup_timesteps,
				args.refine_trials,
				args.refine_timesteps,
				args.final_timesteps,
			),
			n_trials=args.n_trials,
			timeout=args.timeout,
		)
	except KeyboardInterrupt:
		print("Optimization interrupted by user.")

	completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
	print(f"Number of finished trials: {len(study.trials)}")
	if not completed_trials:
		print("No completed trials yet, so there is no best trial to report.")
		return

	trial = study.best_trial
	print("Best trial:")
	print(f"  Value: {trial.value}")
	print("  Params:")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")

	best_params_path = OPTUNA_DIR / "best_params.json"
	best_params_path.write_text(
		json.dumps(
			{
				"value": trial.value,
				"params": trial.params,
				"user_attrs": trial.user_attrs,
			},
			indent=2,
		),
		encoding="utf-8",
	)
	print(f"Saved best trial details to {best_params_path}")


if __name__ == "__main__":
	main()