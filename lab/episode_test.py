
import mujoco
from env import UnitreeA1Env

import numpy as np
xmlPath = r".\external\mujoco_menagerie\unitree_a1\scene.xml"

# Run this once to sanity check your reward function
env = UnitreeA1Env(xmlPath, render_mode="human")
obs, _ = env.reset()

# Simulate perfect standing — zero action for full episode
total_reward = 0
components_sum = {}

for step in range(env.max_episode_steps):
	obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
	total_reward += reward
	for k, v in info["reward_components"].items():
		components_sum[k] = components_sum.get(k, 0) + v
	if terminated:
		print(f"Robot fell at step {step}!")
		break
	if truncated:
		print(f"Episode completed successfully")
		break

print(f"Total reward: {total_reward:.2f}")
print("Components:")
for k, v in components_sum.items():
    print(f"  {k}: {v:.2f}")