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

import imageio
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from env import UnitreeA1Env
from pathlib import Path

modelsPath = r".\Models"
videosPath = r".\Videos"
def record_video(modelVersion, xml_path, output_path, duration_seconds=10):
	model = PPO.load(f"{modelsPath}/a1_walk_v{modelVersion}")
	raw_env = UnitreeA1Env(xml_path, render_mode="rgb_array", max_episode_steps=200)
	venv = DummyVecEnv([lambda: raw_env])
	env = VecNormalize.load(rf"{modelsPath}/a1_walk_v{modelVersion}_vecnormalize.pkl", venv)

	physics_dt   = raw_env.model.opt.timestep
	substeps     = round(raw_env.dt / physics_dt)
	step_dt      = raw_env.dt  # real seconds per env.step()
	
	# Model runs at 50Hz (env.dt = 0.02s = 10 physics substeps of 0.002s each)
	model_hz     = 1.0 / step_dt
	sim_fps      = model_hz

	print(f"Physics dt:    {physics_dt*1000:.2f}ms")
	print(f"Substeps:      {substeps}")
	print(f"Real dt/step:  {step_dt*1000:.2f}ms  ({sim_fps:.1f} Hz)")


	obs = env.reset()
	step_count = 0

	frames = []
	for i in range(int(sim_fps * duration_seconds / 2)):
		action, _ = model.predict(obs, deterministic=True)  # deterministic=True = no random sampling
		obs, reward, done, info = env.step(action)
		frames.append(env.render())

	env.close()

	imageio.mimsave(output_path, frames, fps=sim_fps/2)
	print(f"Saved {duration_seconds}s @ {sim_fps/2:.1f}fps to {output_path}")

if __name__ == "__main__":
	xml_path = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
	modelVersion = "16.3"
	exportTitle = "Baseline Forward Sprint"

	outputPath = f"{videosPath}/v{modelVersion}-{exportTitle or 'unnamed'}.mp4"
	if Path(outputPath).exists():
		response = input(f"Video [v{modelVersion}-{exportTitle or 'unnamed'}.mp4] already exists. Overwrite? (y/n): ")
		if response.lower() != "y":
			print("Aborting video export.")
			exit()
	record_video(modelVersion, xml_path, outputPath, duration_seconds=10)
	# record_video(r"D:\Files\Scripts\py\Graduation Project\Models\checkpoints\v9.0\7000000_steps.zip", UnitreeA1Env, xml_path, outputPath, duration_seconds=10, fps=60, speed_multiplier=speedMultiplier)