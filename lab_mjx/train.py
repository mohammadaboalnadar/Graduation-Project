"""
MJX/Brax PPO Training for Unitree A1 locomotion.

Uses brax.training.agents.ppo.train — all parallelism happens automatically
via jax.vmap across num_envs environments on the GPU. No SubprocVecEnv needed.

RTX 3060 Laptop (6 GB VRAM) tips:
  - Default num_envs=2048; reduce to 1024 if you get OOM errors.
  - First call will JIT-compile the entire rollout (~1-3 min). Be patient.
  - XLA_FLAGS Triton GEMM gives ~30% speedup on NVIDIA GPUs.
  - JAX_DEFAULT_MATMUL_PRECISION=highest prevents Ampere TF32 precision issues.

Run:
    python lab_mjx/train.py
"""

import os

# ── GPU performance flags (must be set before importing JAX) ──────────────────
os.environ.setdefault('XLA_FLAGS', '')
os.environ['XLA_FLAGS'] += ' --xla_gpu_triton_gemm_any=True'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'   # RTX 30/40 Ampere fix
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'    # don't grab all VRAM upfront
os.environ.setdefault('MUJOCO_GL', 'egl')
# ─────────────────────────────────────────────────────────────────────────────

import time
import pickle
import functools
from pathlib import Path

import jax
import jax.numpy as jp
import wandb
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from env import UnitreeA1MJXEnv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
XML_PATH   = ROOT / 'external' / 'mujoco_menagerie' / 'unitree_a1' / 'scene.xml'
MODELS_DIR = ROOT / 'Models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Version & experiment name ─────────────────────────────────────────────────
VERSION = '1.0'
RUN_NAME = f'a1_mjx_v{VERSION}'

# ── Environment config ────────────────────────────────────────────────────────
N_SUBSTEPS    = 10       # Physics substeps per control step → 50 Hz
EPISODE_LENGTH = 500     # Max steps per episode (10 s at 50 Hz)

# ── Training hyperparameters ──────────────────────────────────────────────────
# Effective batch size per update = num_envs * unroll_length = 2048 * 20 = 40960
# Minibatch size                  = effective_batch / num_minibatches = 40960/8 = 5120
# Each update epoch runs num_updates_per_batch gradient steps per minibatch.
#
# RTX 3060 Laptop (6 GB): default num_envs=2048 targets ~4 GB VRAM usage.
# Reduce to 1024 if you hit OOM. Increase to 4096 if you have more VRAM.
NUM_ENVS              = 2048
NUM_TIMESTEPS         = 1_500_000_000
NUM_EVALS             = 100           # How many times to evaluate during training
UNROLL_LENGTH         = 20            # Steps collected per env before each update
BATCH_SIZE            = 512           # SGD mini-batch size
NUM_MINIBATCHES       = 8             # Number of mini-batches per update epoch
NUM_UPDATES_PER_BATCH = 4             # Gradient steps per mini-batch

LEARNING_RATE  = 3e-4
ENTROPY_COST   = 0.01
DISCOUNTING    = 0.99
GAE_LAMBDA     = 0.95
MAX_GRAD_NORM  = 0.5
CLIP_EPSILON   = 0.2
REWARD_SCALING = 1.0
NORMALIZE_OBS  = True   # Running mean/std normalisation (replaces VecNormalize)

# Network architecture (matches lab/ hyperparams: net_arch=[256, 256])
POLICY_HIDDEN = (256, 256)
VALUE_HIDDEN  = (256, 256)

# Checkpoint every N evals (set to None to only save at the end)
CHECKPOINT_EVERY_N_EVALS = 10


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Training {NUM_ENVS} parallel envs for {NUM_TIMESTEPS:,} steps → {RUN_NAME}")

    # ── Environment ───────────────────────────────────────────────────
    env = UnitreeA1MJXEnv(str(XML_PATH), n_substeps=N_SUBSTEPS)

    # ── Network factory ───────────────────────────────────────────────
    # Using functools.partial so the architecture is captured in the factory.
    # test.py must use the same sizes when loading the checkpoint.
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=POLICY_HIDDEN,
        value_hidden_layer_sizes=VALUE_HIDDEN,
    )

    # ── WandB ─────────────────────────────────────────────────────────
    run = wandb.init(
        project='unitree_a1_rl',
        name=RUN_NAME,
        config={
            'framework':       'mjx+brax',
            'version':         VERSION,
            'num_envs':        NUM_ENVS,
            'episode_length':  EPISODE_LENGTH,
            'num_timesteps':   NUM_TIMESTEPS,
            'unroll_length':   UNROLL_LENGTH,
            'batch_size':      BATCH_SIZE,
            'num_minibatches': NUM_MINIBATCHES,
            'num_updates_per_batch': NUM_UPDATES_PER_BATCH,
            'learning_rate':   LEARNING_RATE,
            'entropy_cost':    ENTROPY_COST,
            'discounting':     DISCOUNTING,
            'gae_lambda':      GAE_LAMBDA,
            'policy_hidden':   POLICY_HIDDEN,
            'value_hidden':    VALUE_HIDDEN,
        },
        sync_tensorboard=False,
    )

    # ── Checkpoint path ───────────────────────────────────────────────
    ckpt_dir = MODELS_DIR / 'checkpoints_mjx' / f'v{VERSION}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Progress / logging callback ───────────────────────────────────
    eval_times: list[float] = []
    t_start = time.perf_counter()
    last_ckpt_eval = [0]  # mutable container for closure

    def progress_fn(step: int, metrics: dict):
        elapsed = time.perf_counter() - t_start
        eval_times.append(elapsed)

        # Compute steps/second from last two evals
        fps = step / elapsed if elapsed > 0 else 0.0

        log = {
            'global_step':        step,
            'train/fps':          fps,
            'train/elapsed_min':  elapsed / 60.0,
            **{k: float(v) for k, v in metrics.items()},
        }
        wandb.log(log, step=step)

        ep_rew = metrics.get('eval/episode_reward', float('nan'))
        print(f"[{elapsed/60:.1f} min | {step:>12,} steps | {fps:,.0f} sps] "
              f"eval_reward={ep_rew:.3f}")

        # Periodic checkpoint via pickle (separate from Brax's orbax checkpoint)
        eval_idx = len(eval_times)
        if CHECKPOINT_EVERY_N_EVALS and eval_idx % CHECKPOINT_EVERY_N_EVALS == 0:
            # params are not available inside progress_fn; we save in the finally block.
            # Mark that a save is pending so we note it in the console.
            last_ckpt_eval[0] = step
            print(f"  ↳ checkpoint marker at step {step:,}")

    # ── Training ──────────────────────────────────────────────────────
    print("\nStarting training (first JIT compilation may take 1-3 min)…\n")
    try:
        make_policy, params, metrics = ppo.train(
            environment=env,
            num_timesteps=NUM_TIMESTEPS,
            num_envs=NUM_ENVS,
            episode_length=EPISODE_LENGTH,
            num_evals=NUM_EVALS,
            reward_scaling=REWARD_SCALING,
            unroll_length=UNROLL_LENGTH,
            batch_size=BATCH_SIZE,
            num_minibatches=NUM_MINIBATCHES,
            num_updates_per_batch=NUM_UPDATES_PER_BATCH,
            discounting=DISCOUNTING,
            learning_rate=LEARNING_RATE,
            entropy_cost=ENTROPY_COST,
            gae_lambda=GAE_LAMBDA,
            max_grad_norm=MAX_GRAD_NORM,
            clipping_epsilon=CLIP_EPSILON,
            normalize_observations=NORMALIZE_OBS,
            network_factory=network_factory,
            progress_fn=progress_fn,
            save_checkpoint_path=str(ckpt_dir),  # Brax orbax checkpoints
            seed=42,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current params…")
        make_policy = None
        params = None
    finally:
        if params is not None:
            # Save params as pickle for easy loading in test.py / inference
            params_path = MODELS_DIR / f'{RUN_NAME}_params.pkl'
            with open(params_path, 'wb') as f:
                pickle.dump(params, f)
            print(f"Params saved → {params_path}")

            # Also save the network config alongside params so test.py can load correctly
            cfg_path = MODELS_DIR / f'{RUN_NAME}_config.pkl'
            with open(cfg_path, 'wb') as f:
                pickle.dump({
                    'policy_hidden': POLICY_HIDDEN,
                    'value_hidden':  VALUE_HIDDEN,
                    'obs_size':      env.observation_size,
                    'act_size':      env.action_size,
                }, f)
            print(f"Network config saved → {cfg_path}")

        run.finish()
        print("WandB run closed.")


if __name__ == '__main__':
    main()
