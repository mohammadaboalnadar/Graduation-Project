import imageio
import numpy as np
from stable_baselines3 import PPO

def record_video(model_path, env, xml_path, output_path, duration_seconds=10, fps=60):
	model = PPO.load(model_path)
	env = env(xml_path, render_mode="rgb_array")

	# Get the actual simulation timestep from the model
	sim_fps = 1.0 / env.model.opt.timestep
	print(f"Simulation runs at {sim_fps:.0f} Hz")

	# Only render every Nth step to match desired output fps
	render_every = int(sim_fps / fps)
	print(f"Capturing 1 frame every {render_every} steps to produce {fps}fps video")

	total_frames = duration_seconds * fps
	frames = []

	obs, _ = env.reset()
	step_count = 0

	while len(frames) < total_frames:
		action, _ = model.predict(obs, deterministic=True)
		obs, _, terminated, truncated, _ = env.step(action)
		step_count += 1

		if step_count % render_every == 0:
			frames.append(env.render())

		if terminated or truncated:
			obs, _ = env.reset()

	env.close()

	imageio.mimsave(output_path, frames, fps=fps)
	print(f"Saved {duration_seconds}s @ {fps}fps to {output_path}")

if __name__ == "__main__":
	xml_path = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
	modelVersion = "1"
	exportTitle = ""

	from env import UnitreeA1Env
	from pathlib import Path
	modelsPath = r".\Models"
	videosPath = r".\Videos"
	outputPath = f"{videosPath}/v{modelVersion}-{exportTitle or 'unnamed'}.mp4"
	if Path(outputPath).exists():
		response = input(f"Video [v{modelVersion}-{exportTitle or 'unnamed'}.mp4] already exists. Overwrite? (y/n): ")
		if response.lower() != "y":
			print("Aborting video export.")
			exit()
	record_video(f"{modelsPath}/a1_walk_v{modelVersion}.zip", UnitreeA1Env, xml_path, outputPath, duration_seconds=10, fps=60)