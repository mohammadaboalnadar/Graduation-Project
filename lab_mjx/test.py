"""
Inference / visualisation script for the MJX-trained Unitree A1 policy.

Loads the trained JAX policy params (saved by train.py via pickle),
then runs inference using the standard CPU mujoco.viewer for rendering.

This is the "sim-to-real bridge" step:
  train on GPU (MJX thousands of parallel envs)
    ↓ export params
  deploy on CPU (standard MuJoCo viewer / hardware)

The observation function here must exactly match lab_mjx/env.py._get_obs.
We compute observations directly from the CPU mujoco.MjData to avoid any
dependency on the Brax environment during inference.

Usage:
    python lab_mjx/test.py
    python lab_mjx/test.py --version 1.0 --speed 1.0 --deterministic
"""

import os
import sys
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jp

from brax.training.agents.ppo import networks as ppo_networks

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
XML_PATH  = ROOT / 'external' / 'mujoco_menagerie' / 'unitree_a1' / 'scene.xml'
MODELS_DIR = ROOT / 'Models'

# ── Constants matching env.py ─────────────────────────────────────────────────
DEFAULT_DOF_POS = np.array([
    -0.1, 0.8, -1.5,  # Front Right
     0.1, 0.8, -1.5,  # Front Left
    -0.1, 1.0, -1.5,  # Rear Right
     0.1, 1.0, -1.5,  # Rear Left
], dtype=np.float32)
ACTION_SCALE = 0.8
SKEW_ALPHA   = 0.1   # EMA decay for running pitch/roll
FALL_ANGLE   = 25.0 * np.pi / 180.0
MIN_HEIGHT   = 0.18
MAX_HEIGHT   = 0.50


# ── Observation (CPU NumPy version of env.py._get_obs) ────────────────────────

def get_euler(qpos: np.ndarray):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    return float(yaw), float(pitch), float(roll)

def get_obs(data: mujoco.MjData,
            last_action: np.ndarray,
            running_pitch: float,
            running_roll: float,
            ref_vel: np.ndarray,
            ref_height: float) -> np.ndarray:
    """Identical to UnitreeA1MJXEnv._get_obs but in NumPy for CPU inference."""
    qpos = data.qpos
    qvel = data.qvel
    _, pitch, roll = get_euler(qpos)

    return np.concatenate([
        qvel[0:3] / 3.0,
        qvel[3:6] / 2.0,
        np.array([roll, pitch], dtype=np.float32) / np.pi,
        (qpos[7:] - DEFAULT_DOF_POS) / ACTION_SCALE,
        qvel[6:] / 21.0,
        last_action,
        np.array([running_pitch, running_roll], dtype=np.float32),
        ref_vel,
        np.array([ref_height], dtype=np.float32),
    ]).astype(np.float32)

def is_fallen(data: mujoco.MjData) -> bool:
    _, pitch, roll = get_euler(data.qpos)
    z = data.qpos[2]
    return (abs(pitch) > FALL_ANGLE or abs(roll) > FALL_ANGLE
            or z < MIN_HEIGHT or z > MAX_HEIGHT)


# ── Load policy ───────────────────────────────────────────────────────────────

def load_policy(version: str, deterministic: bool):
    """Load params from pickle and return a JIT-compiled policy function."""
    params_path = MODELS_DIR / f'a1_mjx_v{version}_params.pkl'
    cfg_path    = MODELS_DIR / f'a1_mjx_v{version}_config.pkl'

    if not params_path.exists():
        raise FileNotFoundError(
            f"Params not found: {params_path}\n"
            f"Run train.py first, or adjust --version."
        )

    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    # Load network config (saved by train.py alongside params)
    if cfg_path.exists():
        with open(cfg_path, 'rb') as f:
            cfg = pickle.load(f)
        policy_hidden = tuple(cfg['policy_hidden'])
        value_hidden  = tuple(cfg['value_hidden'])
        obs_size      = cfg['obs_size']
        act_size      = cfg['act_size']
    else:
        # Fall back to defaults matching train.py
        policy_hidden = (256, 256)
        value_hidden  = (256, 256)
        obs_size      = 50
        act_size      = 12

    print(f"Loaded params from {params_path}")
    print(f"  obs={obs_size}  act={act_size}  "
          f"policy={policy_hidden}  value={value_hidden}")

    network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    policy_fn   = jax.jit(make_policy(params, deterministic=deterministic))
    return policy_fn


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise MJX-trained Unitree A1 policy.')
    parser.add_argument('--version',       default='1.0',  help='Model version tag')
    parser.add_argument('--speed',         default=1.0, type=float, help='Playback speed multiplier')
    parser.add_argument('--deterministic', action='store_true',     help='Use deterministic policy')
    parser.add_argument('--ref-vx',        default=1.0, type=float, help='Reference forward velocity')
    parser.add_argument('--ref-vy',        default=0.0, type=float, help='Reference lateral velocity')
    parser.add_argument('--ref-wz',        default=0.0, type=float, help='Reference yaw rate')
    parser.add_argument('--ref-height',    default=0.0, type=float, help='Reference height (±1 normalised)')
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    policy_fn = load_policy(args.version, args.deterministic)

    # ── CPU MuJoCo setup ──────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)

    # Control timestep matching the training env
    dt_ctrl = float(model.opt.timestep) * 10  # n_substeps=10

    # Commands
    ref_vel    = np.array([args.ref_vx, args.ref_vy, args.ref_wz], dtype=np.float32)
    ref_height = float(args.ref_height)

    # State buffers
    last_action   = np.zeros(model.nu, dtype=np.float32)
    running_pitch = 0.0
    running_roll  = 0.0
    rng           = jax.random.PRNGKey(0)
    step_count    = 0

    # Reset to keyframe 0
    def reset_env():
        nonlocal last_action, running_pitch, running_roll, step_count
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        last_action   = np.zeros(model.nu, dtype=np.float32)
        running_pitch = 0.0
        running_roll  = 0.0
        step_count    = 0

    reset_env()
    t_start = time.perf_counter()

    print(f"\nOpening MuJoCo viewer  (speed={args.speed}×, "
          f"ref_vel={ref_vel}, ref_height={ref_height})\n"
          "Close the viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data,
                                       show_left_ui=True,
                                       show_right_ui=False) as viewer:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0

        while viewer.is_running():
            t0 = time.perf_counter()

            # ── Compute observation ───────────────────────────────────
            obs = get_obs(data, last_action, running_pitch, running_roll,
                          ref_vel, ref_height)

            # ── Policy inference (JAX → numpy) ────────────────────────
            rng, key = jax.random.split(rng)
            action_jax, _ = policy_fn(jp.array(obs), key)
            action = np.array(action_jax, dtype=np.float32)

            # ── Update running EMA ────────────────────────────────────
            _, pitch, roll = get_euler(data.qpos)
            running_pitch = (1 - SKEW_ALPHA) * running_pitch + SKEW_ALPHA * pitch
            running_roll  = (1 - SKEW_ALPHA) * running_roll  + SKEW_ALPHA * roll

            # ── Apply action → physics step ───────────────────────────
            last_action = action.copy()
            ctrl = DEFAULT_DOF_POS + np.clip(action, -1, 1) * ACTION_SCALE
            data.ctrl[:] = ctrl
            for _ in range(10):  # n_substeps
                mujoco.mj_step(model, data)

            step_count += 1
            viewer.sync()

            # ── Reset if fallen ───────────────────────────────────────
            if is_fallen(data):
                print(f"  Fallen at step {step_count} — resetting.")
                reset_env()

            # ── Periodic stats ────────────────────────────────────────
            if step_count % 250 == 0:
                elapsed  = time.perf_counter() - t_start
                expected = step_count * dt_ctrl
                ratio    = elapsed / expected
                spd      = float(np.linalg.norm(data.qvel[0:2]))
                print(f"step={step_count:5d} | t={elapsed:.1f}s | "
                      f"realtime_ratio={ratio:.2f} | "
                      f"z={data.qpos[2]:.3f} | spd={spd:.2f} m/s")

            # ── Real-time pacing ──────────────────────────────────────
            target = dt_ctrl / args.speed
            while time.perf_counter() - t0 < target:
                time.sleep(0)


if __name__ == '__main__':
    main()
