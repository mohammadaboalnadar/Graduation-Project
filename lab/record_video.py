import imageio
import numpy as np
from stable_baselines3 import PPO

def record_video(model_path, env, xml_path, output_path, duration_seconds=10, fps=60, speed_multiplier=1.0):
	model = PPO.load(model_path)
	env = env(xml_path, render_mode="rgb_array")

	physics_dt   = env.model.opt.timestep
	n_substeps   = env.model.opt.nsubsteps if hasattr(env.model.opt, 'nsubsteps') else 1
	step_dt      = physics_dt * n_substeps  # real seconds per env.step()

	sim_fps      = speed_multiplier / step_dt
	render_every = max(1, round(sim_fps / fps))

	print(f"Physics dt:    {physics_dt*1000:.2f}ms")
	print(f"Substeps:      {n_substeps}")
	print(f"Real dt/step:  {step_dt*1000:.2f}ms  ({sim_fps:.1f} Hz)")
	print(f"Render every:  {render_every} steps → {sim_fps/render_every:.1f}fps recorded")

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
	modelVersion = "9.0"
	exportTitle = "Checkpoint_7M"
	speedMultiplier = 0.25

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
	# record_video(f"{modelsPath}/a1_walk_v{modelVersion}", UnitreeA1Env, xml_path, outputPath, duration_seconds=10, fps=60, speed_multiplier=speedMultiplier)
	record_video(r"D:\Files\Scripts\py\Graduation Project\Models\checkpoints\v9.0\7000000_steps.zip", UnitreeA1Env, xml_path, outputPath, duration_seconds=10, fps=60, speed_multiplier=speedMultiplier)