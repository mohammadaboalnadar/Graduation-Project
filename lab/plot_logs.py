import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def plot_training_logs(log_dir: str, output_path: str = "training_progress.png"):
	# Load the tensorboard event file
	acc = EventAccumulator(log_dir)
	acc.Reload()

	print("Available tags:", acc.Tags()["scalars"])

	# Define which metrics to plot and their display names
	metrics = {
		"rollout/ep_rew_mean":   "Mean Episode Reward",
		"rollout/ep_len_mean":   "Mean Episode Length",
		"train/policy_loss":     "Policy Loss",
		"train/value_loss":      "Value Loss",
		"train/entropy_loss":    "Entropy Loss",
		"train/approx_kl":       "Approx KL Divergence",
		"train/clip_fraction":   "Clip Fraction",
		"train/explained_variance": "Explained Variance",
	}

	# Filter to only metrics that exist in the log
	available = {k: v for k, v in metrics.items() if k in acc.Tags()["scalars"]}

	n = len(available)
	cols = 2
	rows = (n + 1) // cols

	# number of steps is the same for all metrics, so we can just take it from the first one
	nSteps = acc.Scalars(next(iter(available.keys())))[-1].step
	nStepsStr = f"{nSteps/1e6:.1f}M" if nSteps >= 1e6 else f"{nSteps/1e3:.1f}K" if nSteps >= 1e3 else str(nSteps)
	print(nSteps, nStepsStr)

	fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
	fig.suptitle(f"PPO Training Progress — Unitree A1 - {nStepsStr} steps", fontsize=14, fontweight="bold", y=1.01)
	axes = axes.flatten()

	for i, (tag, title) in enumerate(available.items()):
		events = acc.Scalars(tag)
		steps  = [e.step  for e in events]
		values = [e.value for e in events]

		ax = axes[i]
		ax.plot(steps, values, linewidth=1.2, color="#5B6AD0")

		# Smooth trendline (rolling average)
		if len(values) > 20:
			window = max(1, len(values) // 20)
			smoothed = np.convolve(values, np.ones(window)/window, mode="valid")
			smooth_steps = steps[window-1:]
			ax.plot(smooth_steps, smoothed, linewidth=2, color="#E8593C", label="smoothed")
			ax.legend(fontsize=9)

		ax.set_title(title, fontsize=11)
		ax.set_xlabel("Timesteps", fontsize=9)
		ax.grid(True, alpha=0.3)
		ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

	# Hide any unused subplots
	for j in range(i + 1, len(axes)):
		axes[j].set_visible(False)

	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"Saved to {output_path}")



if __name__ == "__main__":
	version = "2.2"
	iteration = 2

	log_dir = rf".\lab\tb_logs\a1_walk_v{version}_{iteration}"
	title = f"v{version} - Iteration {iteration}"

	figuresPath = r".\Figures"
	if not os.path.exists(figuresPath):
		os.makedirs(figuresPath)
	if os.path.exists(f"{figuresPath}\\{title}.png"):
		response = input(f"Figure [{title}.png] already exists. Overwrite? (y/n): ")
		if response.lower() != "y":
			print("Aborting figure export.")
			exit()
	plot_training_logs(log_dir, output_path=f"{figuresPath}\\{title}.png")