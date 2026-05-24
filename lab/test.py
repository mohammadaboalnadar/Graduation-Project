import os; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import io
import torch
_original_load = torch.load
def _safe_load(f, *args, **kwargs):
    if hasattr(f, "read"):
        buffer = io.BytesIO(f.read())
        return _original_load(buffer, *args, **kwargs)
    return _original_load(f, *args, **kwargs)
torch.load = _safe_load

from sbx import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env import UnitreeA1Env
import numpy as np
import time

VERSION = "16.1"
SPEED_MULTIPLIER = 0.5  # Adjust this to speed up or slow down the simulation

# Load the trained model
model = PPO.load(r".\Models\a1_walk_v" + VERSION)
# model = PPO.load(r"D:\Files\Scripts\py\Graduation Project\Models\checkpoints\v11.2s1.0\3000000_steps.zip")

# Create a render env
xmlPath = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
raw_env = UnitreeA1Env(xmlPath, render_mode="human", max_episode_steps=200)
venv = DummyVecEnv([lambda: raw_env])
env = VecNormalize.load(rf".\Models\a1_walk_v{VERSION}_vecnormalize.pkl", venv)
# env = VecNormalize.load(r"D:\Files\Scripts\py\Graduation Project\Models\checkpoints\v11.2s1.0\3000000_steps_vecnorm.pkl", venv)

env.training = False
env.norm_reward = False

dt = float(raw_env.dt)
obs = env.reset()
raw_env.render()

start_time = time.perf_counter()
step_count = 1
while raw_env._viewer is not None and raw_env._viewer.is_running():  # Keep running until the window is closed
	t0 = time.perf_counter()
	action, _ = model.predict(obs, deterministic=True)  # deterministic=True = no random sampling
	obs, reward, done, info = env.step(action)
	raw_env.render()

	step_count += 1
	if step_count % 500 == 0:
		elapsed      = time.perf_counter() - start_time
		expected     = step_count * dt
		print(f"steps={step_count} | elapsed={elapsed:.2f}s | expected={expected:.2f}s | ratio={elapsed/expected:.3f}")
	
	if step_count % 10 == 0:
		print(f"Target Height: {raw_env.ref_height:.3f} | Actual Height: {raw_env.data.qpos[2]:.3f}")

	while time.perf_counter() - t0 < dt / SPEED_MULTIPLIER:
		time.sleep(0)  # Yield to other processes to prevent CPU hogging