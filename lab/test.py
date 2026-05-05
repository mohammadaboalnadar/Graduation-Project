from stable_baselines3 import PPO
from env import UnitreeA1Env
import numpy as np
import time

VERSION = "8.0"
SPEED_MULTIPLIER = 0.5  # Adjust this to speed up or slow down the simulation

# Load the trained model
xmlPath = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
# model = PPO.load(r".\Models\a1_walk_v" + VERSION)
model = PPO.load(r"D:\Files\Scripts\py\Graduation Project\Models\checkpoints\v8.2\6000000_steps.zip")

# Create a render env
env = UnitreeA1Env(xmlPath, render_mode="human", max_episode_steps=200)
dt = float(env.dt)
obs, _ = env.reset()
env.render()

start_time = time.perf_counter()
step_count = 1
while env._viewer.is_running():  # Keep running until the window is closed
	t0 = time.perf_counter()
	action, _ = model.predict(obs, deterministic=True)  # deterministic=True = no random sampling
	obs, reward, terminated, truncated, _ = env.step(action)
	env.render()

	step_count += 1
	if step_count % 500 == 0:
		elapsed      = time.perf_counter() - start_time
		expected     = step_count * dt
		print(f"steps={step_count} | elapsed={elapsed:.2f}s | expected={expected:.2f}s | ratio={elapsed/expected:.3f}")
	
	if step_count % 10 == 0:
		print(f"Target Height: {env.target_height:.3f} | Actual Height: {env.data.qpos[2]:.3f}")

	while time.perf_counter() - t0 < dt / SPEED_MULTIPLIER:
		time.sleep(0)  # Yield to other processes to prevent CPU hogging

	if terminated or truncated:
		obs, _ = env.reset()