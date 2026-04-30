from stable_baselines3 import PPO
from env import UnitreeA1Env
import numpy as np
import time

VERSION = "2.2"
SPEED_MULTIPLIER = 0.5  # Adjust this to speed up or slow down the simulation

# Load the trained model
xmlPath = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
model = PPO.load(r".\Models\a1_walk_v" + VERSION)

# Create a render env
env = UnitreeA1Env(xmlPath, render_mode="human", max_episode_steps=10000)
dt = float(env.model.opt.timestep)
obs, _ = env.reset()
env.render()

while env._viewer.is_running():  # Keep running until the window is closed
	action, _ = model.predict(obs, deterministic=True)  # deterministic=True = no random sampling
	obs, reward, terminated, truncated, _ = env.step(action)
	env.render()
	time.sleep(dt / SPEED_MULTIPLIER)
	if terminated or truncated:
		obs, _ = env.reset()